"""Shared MCL4-CED MoE Backbone with Per-Layer Embeddings and Recurrent Feedback.

Combines:
1. Lower Causal Encoder Layers (`n_encoder_layers`):
   Process the shared conditioning state/prefix once using MCL4 sparse graph
   attention, Per-Layer Embeddings (PLE), and dropless top-k SwiGLU MoE.
2. Upper Causal Encoder-Decoder (CED) Layers (`n_decoder_layers`):
   Process candidate options, operator queries, or `T$ -> A$` tokens using:
   - Local MCL4 attention (restarting RoPE at position 0 per candidate so candidate
     evaluation is 100% bitwise permutation-equivariant), and
   - Global Cross-Attention into the lower encoder's cached K/V projections
     (computed once per state row and broadcast across candidates).
3. RecurrentFeedback (`anchor + sigmoid(gate) * LayerNorm(state)`):
   Enables multi-pass latent refinement across Cognitive Operators without
   leaking future tokens or breaking candidate permutation invariance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .markov_local4 import MarkovLocal4Router
from .ple import PerLayerEmbedding, RMSNorm


@dataclass
class MCL4CEDConfig:
    vocab_size: int = 260
    pad_id: int = 256
    d_model: int = 256
    n_heads: int = 4
    n_encoder_layers: int = 2
    n_decoder_layers: int = 2
    ple_dim: int = 64
    expert_hidden_dim: int = 512
    num_experts: int = 4
    top_k: int = 2
    max_seq_len: int = 512
    dropout: float = 0.0
    aux_loss_coef: float = 0.01
    router_z_loss_coef: float = 0.001
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10_000.0
    attention_type: str = "markov_local4"
    markov_router_dim: int = 16
    markov_graph_degree: int = 4
    markov_num_walkers: int = 4
    markov_steps: int = 3
    markov_attend_tokens: int = 24
    markov_local_tokens: int = 8
    markov_beam_size: int = 8
    markov_restart: float = 0.1
    markov_query_chunk_size: int = 512

    def __post_init__(self) -> None:
        for name in (
            "vocab_size",
            "d_model",
            "n_heads",
            "n_encoder_layers",
            "n_decoder_layers",
            "expert_hidden_dim",
            "num_experts",
            "top_k",
            "max_seq_len",
        ):
            val = getattr(self, name)
            if isinstance(val, bool) or not isinstance(val, int) or val <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if (self.d_model // self.n_heads) % 2 != 0:
            raise ValueError("RoPE requires an even head dimension")
        if self.top_k > self.num_experts:
            raise ValueError("top_k cannot exceed num_experts")
        if self.attention_type not in ("markov_local4", "standard"):
            raise ValueError("attention_type must be 'markov_local4' or 'standard'")


@dataclass
class EncodedState:
    """Cached lower-encoder state and MoE routing statistics."""

    hidden: Tensor
    valid: Tensor
    aux_loss: Tensor
    router_z_loss: Tensor
    expert_load: Tensor


@dataclass
class BackboneOutput:
    """Output of a full or candidate-decoded pass through MCL4CEDBackbone."""

    hidden: Tensor
    valid: Tensor
    aux_loss: Tensor
    router_z_loss: Tensor
    expert_load: Tensor
    anchor: Tensor | None = None


@dataclass
class CausalLMOutput:
    logits: Tensor
    loss: Tensor | None
    ce_loss: Tensor | None
    aux_loss: Tensor
    router_z_loss: Tensor
    expert_load: Tensor


class RecurrentFeedback(nn.Module):
    """Token-local recurrent state refinement adapter (`anchor + sigmoid(gate) * LN(state)`)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, anchor: Tensor, state: Tensor, valid: Tensor | None = None) -> Tensor:
        if anchor.shape != state.shape or anchor.device != state.device:
            raise ValueError("anchor and state must match in shape and device")
        normed = self.norm(state.to(self.norm.weight.dtype)).to(anchor.dtype)
        updated = anchor + self.gate.to(anchor.dtype).sigmoid() * normed
        if valid is not None:
            updated = updated.masked_fill(~valid.unsqueeze(-1), 0)
        return updated


class SwiGLUExpert(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.up_proj = nn.Linear(d_model, 2 * hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate, value = self.up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * value)


class SparseMoE(nn.Module):
    """Dropless top-k SwiGLU MoE with FP32 routing and load-balance penalties.

    Validates token masks so right-padding tokens never skew expert load statistics
    or corrupt router z-loss.
    """

    def __init__(self, config: MCL4CEDConfig) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.router = nn.Linear(config.d_model, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            SwiGLUExpert(config.d_model, config.expert_hidden_dim)
            for _ in range(config.num_experts)
        )

    def forward(
        self, x: Tensor, valid: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        flat = x.reshape(-1, x.shape[-1])
        flat_valid = (
            torch.ones(flat.shape[0], dtype=torch.bool, device=x.device)
            if valid is None
            else valid.reshape(-1)
        )
        active_indices = flat_valid.nonzero(as_tuple=False).squeeze(-1)
        if active_indices.numel() == 0:
            zero = x.new_zeros((), dtype=torch.float32)
            loads = x.new_zeros(self.num_experts, dtype=torch.float32)
            return torch.zeros_like(x), zero, zero, loads

        active_x = flat.index_select(0, active_indices)
        with torch.autocast(device_type=x.device.type, enabled=False):
            router_logits = F.linear(active_x.float(), self.router.weight.float())
            probabilities = F.softmax(router_logits, dim=-1)
            weights, selected = probabilities.topk(self.top_k, dim=-1)
            weights = weights / weights.sum(dim=-1, keepdim=True)
            one_hot_sel = F.one_hot(selected, num_classes=self.num_experts)
            counts_int = one_hot_sel.sum(dim=(0, 1))
            counts = counts_int.float() / max(1, selected.numel())
            aux_loss = self.num_experts * (counts * probabilities.mean(dim=0)).sum()
            router_z_loss = torch.logsumexp(router_logits, dim=-1).square().mean()

        flat_selected = selected.reshape(-1)
        perm = flat_selected.argsort(stable=True)
        sorted_tokens = torch.div(perm, self.top_k, rounding_mode="floor")
        sorted_weights = weights.reshape(-1).index_select(0, perm).unsqueeze(-1)
        gathered_x = active_x.index_select(0, sorted_tokens)
        expert_chunks = gathered_x.split(counts_int.tolist(), dim=0)

        expert_outs = [
            expert(chunk)
            for expert, chunk in zip(self.experts, expert_chunks)
            if chunk.shape[0] > 0
        ]
        active_out = torch.zeros_like(active_x)
        if expert_outs:
            all_expert_out = torch.cat(expert_outs, dim=0)
            weighted = (all_expert_out * sorted_weights.to(all_expert_out.dtype)).to(
                active_out.dtype
            )
            active_out.index_add_(0, sorted_tokens, weighted)

        output = torch.zeros_like(flat)
        output.index_copy_(0, active_indices, active_out)
        return output.view_as(x), aux_loss, router_z_loss, counts.detach()


class LocalAttentionBlock(nn.Module):
    """Causal or bidirectional self-attention supporting both MCL4 and standard SDPA."""

    def __init__(self, config: MCL4CEDConfig) -> None:
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        inv_freq = 1.0 / (
            config.rope_theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.markov_router: MarkovLocal4Router | None = None
        if config.attention_type == "markov_local4":
            self.markov_router = MarkovLocal4Router(
                config.d_model,
                config.n_heads,
                router_dim=config.markov_router_dim,
                graph_degree=config.markov_graph_degree,
                num_walkers=config.markov_num_walkers,
                steps=config.markov_steps,
                attend_tokens=config.markov_attend_tokens,
                local_tokens=config.markov_local_tokens,
                beam_size=config.markov_beam_size,
                restart=config.markov_restart,
                query_chunk_size=config.markov_query_chunk_size,
            )

    def _apply_rope(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

    def forward(self, x_norm: Tensor, valid: Tensor, *, causal: bool = True) -> Tensor:
        b, n, width = x_norm.shape
        qkv = self.qkv(x_norm).view(b, n, 3, self.n_heads, self.head_dim)
        q, k, v = (part.transpose(1, 2) for part in qkv.unbind(dim=2))
        q, k = self.q_norm(q), self.k_norm(k)
        positions = torch.arange(n, device=x_norm.device, dtype=torch.float32)
        angles = positions[:, None] * self.inv_freq.float()[None, :]
        cos, sin = angles.cos().to(q.dtype), angles.sin().to(q.dtype)
        q, k = self._apply_rope(q, cos, sin), self._apply_rope(k, cos, sin)

        if self.markov_router is not None:
            attended = self.markov_router(
                q,
                k,
                v,
                x_norm,
                valid,
                causal=causal,
                dropout_p=self.config.dropout,
            )
        else:
            allowed = valid[:, None, None, :].expand(b, 1, n, n)
            if causal:
                causal_mask = torch.ones(n, n, dtype=torch.bool, device=x_norm.device).tril()
                allowed = allowed & causal_mask[None, None]
            fallback = torch.eye(n, dtype=torch.bool, device=x_norm.device)[None, None]
            allowed = allowed | (~allowed.any(-1, keepdim=True) & fallback)
            attended = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=allowed,
                dropout_p=self.config.dropout if self.training else 0.0,
            )
        out = self.out_proj(attended.transpose(1, 2).contiguous().view(b, n, width))
        return out.masked_fill(~valid.unsqueeze(-1), 0)


class CEDEncoderLayer(nn.Module):
    """Lower Causal Encoder Layer: PLE + MCL4 Self-Attention + SparseMoE."""

    def __init__(self, config: MCL4CEDConfig) -> None:
        super().__init__()
        self.ple = (
            PerLayerEmbedding(
                config.d_model,
                config.ple_dim,
                vocab_size=config.vocab_size,
                pad_id=config.pad_id,
                rms_norm_eps=config.rms_norm_eps,
            )
            if config.ple_dim > 0
            else None
        )
        self.attn_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.attention = LocalAttentionBlock(config)
        self.moe_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.moe = SparseMoE(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: Tensor,
        original_inputs: Tensor,
        token_ids: Tensor | None,
        valid: Tensor,
        *,
        causal: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if self.ple is not None:
            x = self.ple(x, original_inputs, token_ids=token_ids, valid=valid)
        x = x + self.dropout(self.attention(self.attn_norm(x), valid, causal=causal))
        x = x.masked_fill(~valid.unsqueeze(-1), 0)
        moe_out, aux_loss, z_loss, load = self.moe(self.moe_norm(x), valid=valid)
        x = (x + self.dropout(moe_out)).masked_fill(~valid.unsqueeze(-1), 0)
        return x, aux_loss, z_loss, load


class CEDDecoderLayer(nn.Module):
    """Upper Causal Encoder-Decoder Layer: Local MCL4 + Global State Cross-Attention + PLE + MoE.

    Global K/V projections are computed once per unique state row (`[B_state, T_state, d_model]`)
    and indexed via `owner_rows` when scoring multiple candidates per state.
    """

    def __init__(self, config: MCL4CEDConfig) -> None:
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.ple = (
            PerLayerEmbedding(
                config.d_model,
                config.ple_dim,
                vocab_size=config.vocab_size,
                pad_id=config.pad_id,
                rms_norm_eps=config.rms_norm_eps,
            )
            if config.ple_dim > 0
            else None
        )
        self.attn_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.local_attention = LocalAttentionBlock(config)

        # Global cross-attention into the lower encoder's cached representation with QK-Norm + RoPE.
        self.global_q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.global_kv_proj = nn.Linear(config.d_model, 2 * config.d_model, bias=False)
        self.global_q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.global_k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.global_context_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.global_out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        inv_freq = 1.0 / (
            config.rope_theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.moe_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.moe = SparseMoE(config)
        self.dropout = nn.Dropout(config.dropout)

    @staticmethod
    def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

    def project_global_kv(self, encoded_state: Tensor) -> tuple[Tensor, Tensor]:
        """Project state `[B_state, T_state, d_model]` into `(K_global, V_global)` once with RoPE."""
        b_state, t_state, _ = encoded_state.shape
        kv = self.global_kv_proj(encoded_state).view(
            b_state, t_state, 2, self.n_heads, self.head_dim
        )
        k_global, v_global = (part.transpose(1, 2).contiguous() for part in kv.unbind(dim=2))
        k_normed = self.global_k_norm(k_global)
        positions = torch.arange(t_state, device=encoded_state.device, dtype=torch.float32)
        angles = positions[:, None] * self.inv_freq.float()[None, :]
        cos, sin = angles.cos().to(k_normed.dtype), angles.sin().to(k_normed.dtype)
        return self._apply_rope(k_normed, cos, sin), v_global

    def _cross_attend(
        self,
        q_norm: Tensor,
        k_global: Tensor,
        v_global: Tensor,
        state_valid: Tensor,
        cand_valid: Tensor,
        owner_rows: Tensor | None = None,
        causal_cross_mask: Tensor | None = None,
        causal_cross: bool = False,
    ) -> Tensor:
        r, t_cand, width = q_norm.shape
        q = self.global_q_proj(q_norm).view(r, t_cand, self.n_heads, self.head_dim).transpose(1, 2)
        q = self.global_q_norm(q)
        positions = torch.arange(t_cand, device=q_norm.device, dtype=torch.float32)
        angles = positions[:, None] * self.inv_freq.float()[None, :]
        cos, sin = angles.cos().to(q.dtype), angles.sin().to(q.dtype)
        q = self._apply_rope(q, cos, sin)
        if owner_rows is not None:
            k = k_global.index_select(0, owner_rows)
            v = v_global.index_select(0, owner_rows)
            s_valid = state_valid.index_select(0, owner_rows)
        else:
            k, v, s_valid = k_global, v_global, state_valid

        t_state = k.shape[2]
        if causal_cross and causal_cross_mask is None and t_cand == t_state and bool(s_valid.all()):
            context = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.config.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            allowed = s_valid[:, None, None, :].expand(r, 1, t_cand, t_state)
            if causal_cross and causal_cross_mask is None:
                causal_cross_mask = torch.ones(
                    1, 1, t_cand, t_state, dtype=torch.bool, device=q_norm.device
                ).tril()
            if causal_cross_mask is not None:
                allowed = allowed & causal_cross_mask
            # Prevent all-False rows if a state row were ever completely masked.
            fallback = torch.zeros_like(allowed)
            fallback[..., 0] = True
            allowed = allowed | (~allowed.any(-1, keepdim=True) & fallback)

            context = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=allowed,
                dropout_p=self.config.dropout if self.training else 0.0,
            )
        merged = self.global_context_norm(
            context.transpose(1, 2).contiguous().view(r, t_cand, width)
        )
        out = self.global_out_proj(merged)
        return out.masked_fill(~cand_valid.unsqueeze(-1), 0)

    def forward(
        self,
        x: Tensor,
        original_inputs: Tensor,
        token_ids: Tensor | None,
        cand_valid: Tensor,
        k_global: Tensor,
        v_global: Tensor,
        state_valid: Tensor,
        *,
        owner_rows: Tensor | None = None,
        causal_local: bool = True,
        causal_cross_mask: Tensor | None = None,
        causal_cross: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if self.ple is not None:
            x = self.ple(x, original_inputs, token_ids=token_ids, valid=cand_valid)
        normed = self.attn_norm(x)
        local_out = self.local_attention(normed, cand_valid, causal=causal_local)
        global_out = self._cross_attend(
            normed,
            k_global,
            v_global,
            state_valid,
            cand_valid,
            owner_rows=owner_rows,
            causal_cross_mask=causal_cross_mask,
            causal_cross=causal_cross,
        )
        x = (x + self.dropout(local_out) + self.dropout(global_out)).masked_fill(
            ~cand_valid.unsqueeze(-1), 0
        )
        moe_out, aux_loss, z_loss, load = self.moe(self.moe_norm(x), valid=cand_valid)
        x = (x + self.dropout(moe_out)).masked_fill(~cand_valid.unsqueeze(-1), 0)
        return x, aux_loss, z_loss, load


class MCL4CEDBackbone(nn.Module):
    """Unified MCL4 + CED + PLE + SparseMoE + RecurrentFeedback Backbone."""

    def __init__(self, config: MCL4CEDConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_id
        )
        self.encoder_layers = nn.ModuleList(
            CEDEncoderLayer(config) for _ in range(config.n_encoder_layers)
        )
        self.encoder_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.decoder_layers = nn.ModuleList(
            CEDDecoderLayer(config) for _ in range(config.n_decoder_layers)
        )
        self.final_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.feedback = RecurrentFeedback(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._initialize)
        total_layers = config.n_encoder_layers + config.n_decoder_layers
        residual_scale = 1.0 / math.sqrt(2.0 * total_layers)
        for layer in self.encoder_layers:
            nn.init.normal_(
                layer.attention.out_proj.weight,
                std=residual_scale / math.sqrt(config.d_model),
            )
            for expert in layer.moe.experts:
                nn.init.normal_(
                    expert.down_proj.weight,
                    std=residual_scale / math.sqrt(config.expert_hidden_dim),
                )
        for layer in self.decoder_layers:
            nn.init.normal_(
                layer.local_attention.out_proj.weight,
                std=residual_scale / math.sqrt(config.d_model),
            )
            nn.init.normal_(
                layer.global_out_proj.weight,
                std=residual_scale / math.sqrt(config.d_model),
            )
            for expert in layer.moe.experts:
                nn.init.normal_(
                    expert.down_proj.weight,
                    std=residual_scale / math.sqrt(config.expert_hidden_dim),
                )
        if 0 <= config.pad_id < config.vocab_size:
            with torch.no_grad():
                self.token_embedding.weight[config.pad_id].zero_()
        with torch.no_grad():
            self.lm_head.weight.copy_(self.token_embedding.weight)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=1.0 / math.sqrt(module.in_features))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=1.0 / math.sqrt(module.embedding_dim))

    def encode_state(
        self,
        state_ids: Tensor,
        state_valid: Tensor | None = None,
        *,
        causal: bool = True,
    ) -> EncodedState:
        """Encode shared state/context once through the lower CED encoder stack."""
        if state_ids.ndim != 2 or min(state_ids.shape) < 1:
            raise ValueError("state_ids must have nonempty shape [batch, seq_len]")
        if state_valid is None:
            state_valid = state_ids != self.config.pad_id
        if not state_valid.any(dim=-1).all():
            raise ValueError("Every state row must have at least one valid token")
        x = self.token_embedding(state_ids.long()).masked_fill(~state_valid.unsqueeze(-1), 0)
        original_inputs = x
        aux_losses, z_losses, loads = [], [], []
        for layer in self.encoder_layers:
            x, aux, z, load = layer(x, original_inputs, state_ids, state_valid, causal=causal)
            aux_losses.append(aux)
            z_losses.append(z)
            loads.append(load)
        hidden = self.encoder_norm(x).masked_fill(~state_valid.unsqueeze(-1), 0)
        return EncodedState(
            hidden=hidden,
            valid=state_valid,
            aux_loss=torch.stack(aux_losses).mean(),
            router_z_loss=torch.stack(z_losses).mean(),
            expert_load=torch.stack(loads),
        )

    def decode_candidates(
        self,
        encoded_state: EncodedState,
        candidate_ids: Tensor,
        candidate_valid: Tensor | None = None,
        *,
        owner_rows: Tensor | None = None,
        recurrence_steps: int = 1,
        causal_local: bool = True,
        causal_cross_mask: Tensor | None = None,
        causal_cross: bool = False,
        candidate_encoded: EncodedState | None = None,
    ) -> BackboneOutput:
        """Run upper CED decoder layers over candidates cross-attending to `encoded_state`."""
        if recurrence_steps < 1:
            raise ValueError("recurrence_steps must be at least 1")
        if candidate_ids.ndim != 2 or min(candidate_ids.shape) < 1:
            raise ValueError("candidate_ids must have nonempty shape [rows, seq_len]")
        if candidate_valid is None:
            candidate_valid = candidate_ids != self.config.pad_id
        if not candidate_valid.any(dim=-1).all():
            raise ValueError("Every decoded candidate row must have at least one valid token")

        original_inputs = self.token_embedding(candidate_ids.long()).masked_fill(
            ~candidate_valid.unsqueeze(-1), 0
        )
        if candidate_encoded is None:
            candidate_encoded = self.encode_state(
                candidate_ids, candidate_valid, causal=causal_local
            )
        anchor = candidate_encoded.hidden
        # Precompute K_global, V_global per decoder layer once for the unique state rows.
        global_kv_cache = [
            layer.project_global_kv(encoded_state.hidden) for layer in self.decoder_layers
        ]

        state = anchor
        aux_losses = [candidate_encoded.aux_loss]
        z_losses = [candidate_encoded.router_z_loss]
        loads = [candidate_encoded.expert_load]
        for step in range(recurrence_steps):
            x = anchor if step == 0 else self.feedback(anchor, state, valid=candidate_valid)
            for layer, (k_global, v_global) in zip(self.decoder_layers, global_kv_cache):
                x, aux, z, load = layer(
                    x,
                    original_inputs,
                    candidate_ids,
                    candidate_valid,
                    k_global,
                    v_global,
                    encoded_state.valid,
                    owner_rows=owner_rows,
                    causal_local=causal_local,
                    causal_cross_mask=causal_cross_mask,
                    causal_cross=causal_cross,
                )
                aux_losses.append(aux)
                z_losses.append(z)
                loads.append(load.unsqueeze(0))
            state = self.final_norm(x).masked_fill(~candidate_valid.unsqueeze(-1), 0)

        return BackboneOutput(
            hidden=state,
            valid=candidate_valid,
            aux_loss=torch.stack(aux_losses).mean(),
            router_z_loss=torch.stack(z_losses).mean(),
            expert_load=torch.cat(loads, dim=0),
            anchor=anchor,
        )

    def encode(self, input_ids: Tensor, *, recurrence_steps: int = 1) -> BackboneOutput:
        """Causal full-sequence pass through lower encoder and upper CED decoder."""
        valid = input_ids != self.config.pad_id
        if not valid.any(dim=-1).all():
            # Fallback if an entire block is pad_id: treat index 0 as valid for numeric safety.
            valid = valid.clone()
            valid[:, 0] = True
        encoded = self.encode_state(input_ids, valid, causal=True)
        decoded = self.decode_candidates(
            encoded,
            input_ids,
            valid,
            recurrence_steps=recurrence_steps,
            causal_local=True,
            causal_cross=True,
            candidate_encoded=encoded,
        )
        return BackboneOutput(
            hidden=decoded.hidden,
            valid=valid,
            aux_loss=decoded.aux_loss,
            router_z_loss=decoded.router_z_loss,
            expert_load=decoded.expert_load,
            anchor=decoded.anchor,
        )

    def forward(
        self,
        input_ids: Tensor,
        targets: Tensor | None = None,
        *,
        recurrence_steps: int = 1,
    ) -> CausalLMOutput:
        """Autoregressive `U$ -> T$ -> A$` language modeling pass."""
        if targets is not None and targets.shape != input_ids.shape:
            raise ValueError("targets must match input_ids shape [batch, seq_len]")
        encoded = self.encode(input_ids, recurrence_steps=recurrence_steps)
        logits = self.lm_head(encoded.hidden)
        ce_loss, loss = None, None
        if targets is not None:
            ce_sum = F.cross_entropy(
                logits.float().reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            ce_loss = ce_sum / targets.ne(-100).sum().clamp_min(1)
            loss = (
                ce_loss
                + self.config.aux_loss_coef * encoded.aux_loss
                + self.config.router_z_loss_coef * encoded.router_z_loss
            )
        return CausalLMOutput(
            logits=logits,
            loss=loss,
            ce_loss=ce_loss,
            aux_loss=encoded.aux_loss,
            router_z_loss=encoded.router_z_loss,
            expert_load=encoded.expert_load,
        )
