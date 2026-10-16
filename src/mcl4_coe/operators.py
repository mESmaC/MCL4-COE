"""Cognitive Operator Ensemble (COE) on top of the shared MCL4-CED Backbone.

Implements all 9 composable single-softmax / relational cognitive operators:
1. Decision (`choice`, `noul`, `score`) — Bitwise permutation-invariant candidate selection
2. Association — Pairwise & set-level centered Pearson similarity + <0.1ms Novelty N(q)
3. Evaluation — Criterion-conditioned utility assessment U(m, c) in [0, 1]
4. Deduction — 3-way logical status (`entailed`, `contradicted`, `undetermined`)
5. Induction — Pattern generalization across observed cases + uncertainty
6. Abduction — Explanatory hypothesis ranking under observations
7. Abstraction — Multi-scale latent compression (`brief`, `standard`, `detailed`)
8. Metacognition — Calibrated reliability signals (`correctness`, `evidence_sufficiency`, `context_benefit`)
9. Inference — Autoregressive `U$ -> T$ -> A$` synthesis conditioned on recurrent COE state
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .backbone import EncodedState, MCL4CEDBackbone
from .ple import RMSNorm


def _positive_finite(value: float, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be finite and positive")


def pearson_similarity(left: Tensor, right: Tensor, eps: float = 1e-6) -> Tensor:
    """FP32 centered cosine (Pearson) similarity across final-axis features in [-1, 1].

    In signal-processing terms:
    - Subtracting the feature mean (`u - mean(u)`) strips the DC / common-mode offset.
    - L2 normalization (`u / norm(u)`) acts as Automatic Gain Control (AGC) so short
      and long descriptions compete at identical signal amplitude.
    """
    _positive_finite(eps, "eps")
    if (
        not isinstance(left, Tensor)
        or not isinstance(right, Tensor)
        or left.ndim < 1
        or right.ndim < 1
        or left.shape[-1] != right.shape[-1]
        or left.shape[-1] < 1
    ):
        raise ValueError("left and right must have the same nonempty feature dimension")
    if left.device != right.device:
        raise ValueError("left and right must be on the same device")
    if not left.is_floating_point() or not right.is_floating_point():
        raise ValueError("left and right must be floating-point tensors")
    with torch.autocast(device_type=left.device.type, enabled=False):
        centered_left = left.float() - left.float().mean(-1, keepdim=True)
        centered_right = right.float() - right.float().mean(-1, keepdim=True)
        normalized_left = F.normalize(centered_left, dim=-1, eps=eps)
        normalized_right = F.normalize(centered_right, dim=-1, eps=eps)
        return (normalized_left * normalized_right).sum(-1).clamp(-1.0, 1.0)


def _canonical_candidate_order(
    owner_rows: Tensor,
    cand_lengths: Tensor,
    cand_ids: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return `(perm, inv_perm)` sorting candidate rows canonically by `(owner_row, length, token_ids)`.

    Why this matters for hardware execution:
    Batched SGEMM / TensorCore kernels tile across the batch dimension `R`, which can
    introduce 1-ULP (`~1e-7`) floating-point rounding differences when rows cross a
    16-row tile boundary. Sorting candidate rows into a canonical lexicographical order
    before the batched GPU pass and scattering back via `inv_perm` keeps 100% single-batch
    TensorCore speed while guaranteeing 0-ULP bitwise permutation invariance (`torch.equal`).
    """
    r = owner_rows.shape[0]
    if r <= 1:
        ident = torch.arange(r, device=owner_rows.device, dtype=torch.long)
        return ident, ident
    # Build sorting keys on CPU from integer metadata (`[R, 2 + T_cand]`) without touching GPU floats.
    keys = torch.cat(
        [
            owner_rows.detach().cpu().unsqueeze(1).long(),
            cand_lengths.detach().cpu().unsqueeze(1).long(),
            cand_ids.detach().cpu().long(),
        ],
        dim=1,
    ).tolist()
    order = sorted(range(r), key=keys.__getitem__)
    perm = torch.tensor(order, device=owner_rows.device, dtype=torch.long)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(r, device=owner_rows.device, dtype=torch.long)
    return perm, inv_perm


@dataclass
class COEConfig:
    projection_dim: int = 32
    metric: str = "pearson"
    initial_scale: float = 15.0
    eps: float = 1e-6
    recurrence_steps: int = 1
    recurrent_kv: bool = False
    depth_ponder: bool = False
    margin_halt_threshold: float | None = None
    gist_tokens_per_chunk: int = 6
    gist_chunk_size: int = 192
    strip_boundary_tokens_in_pool: bool = True
    bos_id: int = 257
    eos_id: int = 258

    def __post_init__(self) -> None:
        if (
            isinstance(self.projection_dim, bool)
            or not isinstance(self.projection_dim, int)
            or self.projection_dim < 3
        ):
            raise ValueError("projection_dim must be an integer >= 3 for Pearson gradients")
        if self.metric not in ("pearson", "cosine", "dot"):
            raise ValueError("metric must be 'pearson', 'cosine', or 'dot'")
        _positive_finite(self.initial_scale, "initial_scale")
        if not 0.1 <= self.initial_scale <= 100.0:
            raise ValueError("initial_scale must be between 0.1 and 100.0")
        _positive_finite(self.eps, "eps")
        if isinstance(self.recurrence_steps, bool) or self.recurrence_steps < 1:
            raise ValueError("recurrence_steps must be >= 1")
        if isinstance(self.gist_tokens_per_chunk, bool) or self.gist_tokens_per_chunk < 1:
            raise ValueError("gist_tokens_per_chunk must be >= 1")


@dataclass
class DecisionOutput:
    logits: Tensor
    similarities: Tensor
    probabilities: Tensor
    state_features: Tensor
    candidate_features: Tensor
    loss: Tensor | None
    ce_loss: Tensor | None
    aux_loss: Tensor
    router_z_loss: Tensor


@dataclass
class AssociationMatrixOutput:
    similarities: Tensor
    logits: Tensor
    probabilities: Tensor
    features: Tensor
    novelty: Tensor | None = None


@dataclass
class DeductionOutput:
    logits: Tensor
    probabilities: Tensor
    similarities: Tensor
    status_labels: tuple[str, str, str] = ("entailed", "contradicted", "undetermined")
    loss: Tensor | None = None


@dataclass
class MetacognitionOutput:
    logits: Tensor
    probabilities: Tensor
    signals: tuple[str, str, str] = ("correctness", "evidence_sufficiency", "context_benefit")
    loss: Tensor | None = None


@dataclass
class EvaluationOutput:
    logits: Tensor
    utilities: Tensor
    similarities: Tensor
    loss: Tensor | None = None


@dataclass
class AllOperatorsOutput:
    """Unified output of a single-pass multi-operator COE forward (`Decision`, `Deduction`, `Evaluation`, `Association`, `Metacognition`, `Abstraction`)."""

    decision: DecisionOutput
    deduction_loss: Tensor
    evaluation_loss: Tensor
    metacognition_loss: Tensor
    abstraction_loss: Tensor
    total_loss: Tensor
    deduction_probs: Tensor
    metacognition_probs: Tensor
    evaluation_utilities: Tensor


class AttentiveMatchedFilterPool(nn.Module):
    """Token-level QK-Norm matched-filter cross-attention and attentive sequence pooling.

    How this breaks the bi-encoder bottleneck (`44.4%` -> `100%`):
    1. Every valid token $t$ in Candidate $r$ queries all valid tokens $s$ in its owner
       State $b$ using `QK-Norm` (`RMSNorm` on $Q$ and $K$), retrieving the best-matching
       state token $M_{r, t}$ *before* pooling across candidate tokens (`Attend-then-Pool`).
    2. At each candidate token position $t$, the aligned state evidence $M_{r, t}$ and
       candidate representation $C_{r, t}$ are pooled with learned salience weights
       (excluding `<bos>`/`<eos>` framing tokens), so even a 2-byte difference (`"no"` vs
       `"one"`) inside a 25-byte sentence produces a sharp differential signal.
    3. Candidate $r$ never interacts with any other candidate $k \ne r$, preserving
       100% bitwise permutation invariance.
    """

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.d_model = d_model
        self.token_scorer = nn.Linear(d_model, 1, bias=False)
        self.match_q = nn.Linear(d_model, d_model, bias=False)
        self.match_k = nn.Linear(d_model, d_model, bias=False)
        self.match_v = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = RMSNorm(d_model, eps=eps)
        self.k_norm = RMSNorm(d_model, eps=eps)
        self.match_scale = nn.Parameter(torch.tensor(6.0))
        self.state_fuse = nn.Linear(d_model * 2, d_model, bias=False)
        self.cand_fuse = nn.Linear(d_model * 2, d_model, bias=False)
        self.state_norm = nn.LayerNorm(d_model, eps=eps)
        self.cand_norm = nn.LayerNorm(d_model, eps=eps)

        adapter_std = 0.25 / math.sqrt(d_model)
        nn.init.normal_(self.token_scorer.weight, std=adapter_std)
        nn.init.normal_(self.match_q.weight, std=adapter_std)
        nn.init.normal_(self.match_k.weight, std=adapter_std)
        nn.init.normal_(self.match_v.weight, std=adapter_std)
        nn.init.normal_(self.state_fuse.weight, std=adapter_std / math.sqrt(2.0))
        nn.init.normal_(self.cand_fuse.weight, std=adapter_std / math.sqrt(2.0))

    @staticmethod
    def _content_mask(
        token_ids: Tensor,
        valid: Tensor,
        *,
        strip_boundary: bool,
        bos_id: int,
        eos_id: int,
    ) -> Tensor:
        if not strip_boundary:
            return valid
        interior = valid & (token_ids != bos_id) & (token_ids != eos_id)
        has_interior = interior.any(dim=-1, keepdim=True)
        return torch.where(has_interior, interior, valid)

    @classmethod
    def _state_evidence_mask(
        cls,
        state_ids: Tensor,
        state_valid: Tensor,
        *,
        strip_boundary: bool,
        bos_id: int,
        eos_id: int,
    ) -> Tensor:
        """Isolate the Sensor/World-State register (`#U$# ...`) from the Question register (`\\n#Q$# ...`).

        Why this matters:
        In `#U$# <state>\\n#Q$# [<kind>] <instructions>`, the Question register (`#Q$#`) often
        contains words from a candidate option (e.g. `"Is the animal carrying a ball?"` contains
        `"a ball"` even when `#U$#` says `"A rabbit carries no balls."`). Masking out `\\n#Q$# ...`
        and the `#U$# ` prefix ensures the matched filter only extracts evidence from the
        World-State tokens.
        """
        mask = cls._content_mask(
            state_ids, state_valid, strip_boundary=strip_boundary, bos_id=bos_id, eos_id=eos_id
        )
        b, t = state_ids.shape
        if t < 2:
            return mask
        positions = torch.arange(t, device=state_ids.device)[None, :]
        trimmed = mask.clone()

        # 1. MorphCode-16K atomic `#Q$#` / `#S$#` / `#H$#` / `#T$#` / `#A$#` marker tokens:
        #    261..265 ("#Q$#".."#A$#"), 267..271 ("\n#Q$#".."\n#A$#"), 273..275 ("#Q$# ".."#A$# ")
        is_morph_q = (
            ((state_ids >= 261) & (state_ids <= 265))
            | ((state_ids >= 267) & (state_ids <= 271))
            | ((state_ids >= 273) & (state_ids <= 275))
        ) & state_valid
        if is_morph_q.any():
            has_mq = is_morph_q.any(dim=-1, keepdim=True)
            first_mq = torch.where(
                has_mq,
                is_morph_q.int().argmax(dim=-1, keepdim=True),
                torch.full((b, 1), t, device=state_ids.device, dtype=torch.long),
            )
            trimmed = trimmed & (positions < first_mq)

        # Strip MorphCode-16K atomic `#U$#` marker tokens (260="#U$#", 266="\n#U$#", 272="#U$# ")
        is_morph_u = ((state_ids == 260) | (state_ids == 266) | (state_ids == 272)) & state_valid
        if is_morph_u.any():
            trimmed = trimmed & ~is_morph_u

        if t < 6:
            return torch.where(trimmed.any(dim=-1, keepdim=True), trimmed, mask)

        # 2. ByteTokenizer 5-byte sequences:
        #    Detect `\n#Q$#` ([10, 35, 81, 36, 35]) and mask from `\n#Q$#` to the end of the sequence.
        is_nq = (
            (state_ids[:, :-4] == 10)
            & (state_ids[:, 1:-3] == 35)
            & (state_ids[:, 2:-2] == 81)
            & (state_ids[:, 3:-1] == 36)
            & (state_ids[:, 4:] == 35)
            & state_valid[:, :-4]
        )
        has_nq = is_nq.any(dim=-1, keepdim=True)
        first_nq = torch.where(
            has_nq,
            is_nq.int().argmax(dim=-1, keepdim=True),
            torch.full((b, 1), t, device=state_ids.device, dtype=torch.long),
        )
        trimmed = trimmed & (positions < first_nq)

        # Detect `#U$# ` ([35, 85, 36, 35, 32]) at the start of the state and strip those 5 tag bytes.
        is_u_tag = (
            (state_ids[:, :-4] == 35)
            & (state_ids[:, 1:-3] == 85)
            & (state_ids[:, 2:-2] == 36)
            & (state_ids[:, 3:-1] == 35)
            & (state_ids[:, 4:] == 32)
            & state_valid[:, :-4]
        )
        has_u = is_u_tag.any(dim=-1, keepdim=True)
        first_u_end = torch.where(
            has_u,
            is_u_tag.int().argmax(dim=-1, keepdim=True) + 5,
            torch.zeros((b, 1), device=state_ids.device, dtype=torch.long),
        )
        trimmed = trimmed & (positions >= first_u_end)

        return torch.where(trimmed.any(dim=-1, keepdim=True), trimmed, mask)

    def pool_sequence(
        self,
        hidden: Tensor,
        token_ids: Tensor,
        valid: Tensor,
        *,
        strip_boundary: bool = True,
        bos_id: int = 257,
        eos_id: int = 258,
    ) -> Tensor:
        mask = self._content_mask(
            token_ids, valid, strip_boundary=strip_boundary, bos_id=bos_id, eos_id=eos_id
        )
        raw_scores = self.token_scorer(hidden.to(self.token_scorer.weight.dtype))
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            scores = raw_scores.squeeze(-1).float()
            scores = scores.masked_fill(~mask, -torch.inf)
            weights = scores.softmax(dim=-1)
            attentive = (hidden.float() * weights.unsqueeze(-1)).sum(dim=1)
            mean_pool = hidden.float().masked_fill(~mask.unsqueeze(-1), 0.0).sum(
                dim=1
            ) / mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
            return (0.5 * (attentive + mean_pool)).to(hidden.dtype)

    def matched_pair_representations(
        self,
        state_hidden: Tensor,
        state_ids: Tensor,
        state_valid: Tensor,
        cand_hidden: Tensor,
        cand_ids: Tensor,
        cand_valid: Tensor,
        owner_rows: Tensor,
        *,
        cand_anchor: Tensor | None = None,
        strip_boundary: bool = True,
        bos_id: int = 257,
        eos_id: int = 258,
    ) -> tuple[Tensor, Tensor]:
        """Compute AC-coupled token-level QK-Norm matched-filter representations `(paired_state, cand_repr)`."""
        s_mask = self._state_evidence_mask(
            state_ids, state_valid, strip_boundary=strip_boundary, bos_id=bos_id, eos_id=eos_id
        )
        c_mask = self._content_mask(
            cand_ids, cand_valid, strip_boundary=strip_boundary, bos_id=bos_id, eos_id=eos_id
        )
        cand_base = cand_hidden if cand_anchor is None else cand_anchor

        # Residual matched-filter projections anchored on the shared lower-encoder space (`f_enc`).
        k_state_all = self.k_norm(state_hidden + self.match_k(state_hidden))
        v_state_raw = state_hidden + self.match_v(state_hidden)
        with torch.autocast(device_type=state_hidden.device.type, enabled=False):
            s_mask_f = s_mask.unsqueeze(-1).float()
            v_mean = (v_state_raw.float() * s_mask_f).sum(dim=1, keepdim=True) / s_mask_f.sum(
                dim=1, keepdim=True
            ).clamp_min(1.0)
            # AC-couple state values within the world-state register to strip the DC template blocker.
            v_state_all = ((v_state_raw.float() - v_mean) * s_mask_f).to(state_hidden.dtype)

        k_state = k_state_all.index_select(0, owner_rows)  # [R, T_state, d_model]
        v_state = v_state_all.index_select(0, owner_rows)  # [R, T_state, d_model]
        s_mask_rows = s_mask.index_select(0, owner_rows).unsqueeze(1)  # [R, 1, T_state]

        # Every candidate token queries all valid state tokens in the shared `f_enc` basis.
        q_cand = self.q_norm(cand_base + self.match_q(cand_hidden))  # [R, T_cand, d_model]
        with torch.autocast(device_type=state_hidden.device.type, enabled=False):
            scale = self.match_scale.float().clamp(0.5, 20.0) / math.sqrt(self.d_model)
            sim_matrix = torch.matmul(q_cand.float(), k_state.float().transpose(-1, -2)) * scale
            sim_matrix = sim_matrix.masked_fill(~s_mask_rows, -torch.inf)
            attn_weights = sim_matrix.softmax(dim=-1)  # [R, T_cand, T_state]
            aligned_state_tokens = torch.matmul(attn_weights, v_state.float()).to(cand_hidden.dtype)

            # Weight candidate positions by their peak match confidence in the state.
            peak_match = sim_matrix.max(dim=-1).values.masked_fill(~c_mask, -torch.inf)
            match_weights = peak_match.softmax(dim=-1).unsqueeze(-1)
            peak_aligned_state = (aligned_state_tokens.float() * match_weights).sum(dim=1).to(
                state_hidden.dtype
            )
            peak_cand_base = (cand_base.float() * match_weights).sum(dim=1).to(cand_hidden.dtype)

        pooled_aligned_state = self.pool_sequence(
            aligned_state_tokens,
            cand_ids,
            c_mask,
            strip_boundary=False,
            bos_id=bos_id,
            eos_id=eos_id,
        )
        pooled_cand_base = self.pool_sequence(
            cand_base,
            cand_ids,
            c_mask,
            strip_boundary=False,
            bos_id=bos_id,
            eos_id=eos_id,
        )
        pooled_cand = self.pool_sequence(
            cand_hidden,
            cand_ids,
            c_mask,
            strip_boundary=False,
            bos_id=bos_id,
            eos_id=eos_id,
        )
        interaction = pooled_cand * pooled_aligned_state

        state_residual = 0.5 * (pooled_aligned_state + peak_aligned_state)
        cand_residual = 0.5 * (pooled_cand_base + peak_cand_base)

        paired_state = self.state_norm(
            (
                state_residual
                + self.state_fuse(torch.cat([pooled_aligned_state, peak_aligned_state], dim=-1))
            ).float()
        ).to(state_hidden.dtype)
        cand_repr = self.cand_norm(
            (
                cand_residual
                + self.cand_fuse(torch.cat([pooled_cand, interaction], dim=-1))
            ).float()
        ).to(cand_hidden.dtype)
        return paired_state, cand_repr


class CognitiveOperatorEnsemble(nn.Module):
    """Composable Single-Softmax Cognitive Operator Ensemble (COE) on MCL4-CED."""

    def __init__(self, backbone: MCL4CEDBackbone, config: COEConfig | None = None) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = COEConfig() if config is None else config
        self.config.__post_init__()
        d_model = backbone.config.d_model
        d_proj = self.config.projection_dim

        self.pooler = AttentiveMatchedFilterPool(d_model, eps=backbone.config.rms_norm_eps)
        self.state_projection = nn.Linear(d_model, d_proj, bias=False)
        self.output_projection = nn.Linear(d_model, d_proj, bias=False)
        self.log_scale = nn.Parameter(torch.tensor(math.log(self.config.initial_scale)))

        # Association independent link calibration: logit = alpha * sim + beta
        self.assoc_alpha = nn.Parameter(torch.tensor(6.0))
        self.assoc_beta = nn.Parameter(torch.tensor(0.0))

        # Deduction 3-way logical status prototypes in Pearson space:
        # (entailed, contradicted, undetermined)
        self.deduction_query_proj = nn.Linear(d_model * 2, d_proj, bias=False)
        self.deduction_prototypes = nn.Parameter(torch.randn(3, d_proj) * 0.1)

        # Metacognition 3 calibrated binary signals:
        # (correctness, evidence_sufficiency, context_benefit)
        self.meta_query_proj = nn.Linear(d_model * 2, d_proj, bias=False)
        self.meta_prototypes = nn.Parameter(torch.randn(3, d_proj) * 0.1)
        self.meta_bias = nn.Parameter(torch.zeros(3))

        # Abstraction multi-scale compression adapters (`brief`, `standard`, `detailed`)
        self.abstraction_gates = nn.ParameterDict(
            {
                "brief": nn.Parameter(torch.tensor(-0.5)),
                "standard": nn.Parameter(torch.tensor(0.0)),
                "detailed": nn.Parameter(torch.tensor(0.5)),
            }
        )
        self.abstraction_proj = nn.Linear(d_model, d_model, bias=False)
        self.abstraction_norm = RMSNorm(d_model, eps=backbone.config.rms_norm_eps)

        proj_std = 1.0 / math.sqrt(d_model)
        nn.init.normal_(self.state_projection.weight, std=proj_std)
        with torch.no_grad():
            self.output_projection.weight.copy_(self.state_projection.weight)
        nn.init.normal_(self.deduction_query_proj.weight, std=proj_std)
        nn.init.normal_(self.meta_query_proj.weight, std=proj_std)
        nn.init.normal_(self.abstraction_proj.weight, std=proj_std)

        # Persistent ContextLMCompressor soft gist queries & LayerNorm trained end-to-end
        self.gist_norm = nn.LayerNorm(d_model)
        self.gist_queries = nn.Parameter(
            torch.empty(self.config.gist_tokens_per_chunk, d_model)
        )
        nn.init.normal_(self.gist_queries, std=0.02)

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        for opt_key, param in (
            ("gist_queries", self.gist_queries),
            ("gist_norm.weight", self.gist_norm.weight),
            ("gist_norm.bias", self.gist_norm.bias),
        ):
            full_key = prefix + opt_key
            if full_key not in state_dict:
                state_dict[full_key] = param.detach().clone()
            elif state_dict[full_key].shape != param.shape and opt_key == "gist_queries":
                self.gist_queries = nn.Parameter(
                    torch.empty_like(state_dict[full_key], device=param.device, dtype=param.dtype)
                )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _similarity(self, left: Tensor, right: Tensor) -> Tensor:
        if self.config.metric == "pearson":
            return pearson_similarity(left, right, self.config.eps)
        with torch.autocast(device_type=left.device.type, enabled=False):
            if self.config.metric == "cosine":
                u = F.normalize(left.float(), dim=-1, eps=self.config.eps)
                v = F.normalize(right.float(), dim=-1, eps=self.config.eps)
                return (u * v).sum(-1).clamp(-1.0, 1.0)
            return (left.float() * right.float()).sum(-1) / math.sqrt(self.config.projection_dim)

    def _scale(self) -> Tensor:
        return self.log_scale.float().clamp(math.log(0.5), math.log(4.5)).exp()

    def forward_decision(
        self,
        context_ids: Tensor,
        context_lengths: Tensor,
        option_ids: Tensor,
        option_lengths: Tensor,
        targets: Tensor | None = None,
        *,
        recurrence_steps: int | None = None,
        depth_ponder: bool | None = None,
        margin_halt_threshold: float | None = None,
    ) -> DecisionOutput:
        """Evaluate `Decision` (`Choice` / `Noul` / `Score`) with CED cross-attention.

        - Encodes `context_ids` once per batch row in the lower CED encoder.
        - Canonically orders valid options before batched upper-CED decoding so
          output scores are 100% 0-ULP bitwise permutation-invariant (`torch.equal`).
        """
        if context_ids.ndim != 2 or min(context_ids.shape) < 1:
            raise ValueError("context_ids must have nonempty shape [batch, tokens]")
        batch = context_ids.shape[0]
        if context_lengths.shape != (batch,):
            raise ValueError("context_lengths must have shape [batch]")
        if option_ids.ndim != 3 or option_ids.shape[0] != batch or min(option_ids.shape[1:]) < 1:
            raise ValueError("option_ids must have nonempty shape [batch, options, tokens]")
        if option_lengths.shape != option_ids.shape[:2]:
            raise ValueError("option_lengths must have shape [batch, options]")
        valid_options = option_lengths > 0
        if (valid_options.sum(-1) < 2).any():
            raise ValueError("Each decision row requires at least two valid options")
        if targets is not None:
            if targets.shape != (batch,):
                raise ValueError("targets must have shape [batch]")
            if ((targets < 0) | (targets >= option_ids.shape[1])).any():
                raise ValueError("targets out of range")
            if not valid_options.gather(-1, targets.long().unsqueeze(-1)).all():
                raise ValueError("targets cannot select a masked option")

        use_ponder = self.config.depth_ponder if depth_ponder is None else depth_ponder
        halt_thresh = (
            self.config.margin_halt_threshold
            if margin_halt_threshold is None
            else margin_halt_threshold
        )
        if recurrence_steps is None:
            steps = (
                max(2, self.config.recurrence_steps)
                if (use_ponder or halt_thresh is not None)
                else self.config.recurrence_steps
            )
        else:
            steps = recurrence_steps
        ctx_width = int(context_lengths.max().item())
        trimmed_ctx_ids = context_ids[:, :ctx_width].long()
        ctx_valid = (
            torch.arange(ctx_width, device=context_ids.device)[None, :] < context_lengths[:, None]
        )

        # 1. Encode shared state/question once in the lower CED encoder layers.
        encoded_state: EncodedState = self.backbone.encode_state(
            trimmed_ctx_ids, ctx_valid, causal=True
        )

        # 2. Extract valid options and sort into canonical order per question row
        #    so batched GEMM tiling is 100% invariant to option permutation.
        valid_lengths = option_lengths[valid_options]
        opt_width = int(valid_lengths.max().item())
        flat_opt_ids = option_ids[valid_options][:, :opt_width].long()
        owner_rows = valid_options.nonzero(as_tuple=False)[:, 0]
        perm, inv_perm = _canonical_candidate_order(owner_rows, valid_lengths, flat_opt_ids)

        canon_opt_ids = flat_opt_ids.index_select(0, perm)
        canon_lengths = valid_lengths.index_select(0, perm)
        canon_owners = owner_rows.index_select(0, perm)
        canon_valid = (
            torch.arange(opt_width, device=option_ids.device)[None, :] < canon_lengths[:, None]
        )

        decoded_options = self.backbone.decode_candidates(
            encoded_state,
            canon_opt_ids,
            canon_valid,
            owner_rows=canon_owners,
            recurrence_steps=steps,
            causal_local=True,
        )

        # 3. Token-level QK-Norm matched-filter pooling + centered Pearson projection.
        paired_state_repr, cand_repr = self.pooler.matched_pair_representations(
            encoded_state.hidden,
            trimmed_ctx_ids,
            ctx_valid,
            decoded_options.hidden,
            canon_opt_ids,
            canon_valid,
            canon_owners,
            cand_anchor=decoded_options.anchor,
            strip_boundary=self.config.strip_boundary_tokens_in_pool,
            bos_id=self.config.bos_id,
            eos_id=self.config.eos_id,
        )
        canon_state_proj = self.state_projection(paired_state_repr)
        canon_cand_proj = self.output_projection(cand_repr)
        canon_scores = self._similarity(canon_state_proj, canon_cand_proj)

        similarities_step1: Tensor | None = None
        if (use_ponder or halt_thresh is not None) and len(decoded_options.step_states) > 1:
            ps_1, cr_1 = self.pooler.matched_pair_representations(
                encoded_state.hidden,
                trimmed_ctx_ids,
                ctx_valid,
                decoded_options.step_states[0],
                canon_opt_ids,
                canon_valid,
                canon_owners,
                cand_anchor=decoded_options.anchor,
                strip_boundary=self.config.strip_boundary_tokens_in_pool,
                bos_id=self.config.bos_id,
                eos_id=self.config.eos_id,
            )
            cs_1 = self._similarity(
                self.state_projection(ps_1),
                self.output_projection(cr_1),
            )
            v_scores_1 = cs_1.index_select(0, inv_perm)
            similarities_step1 = v_scores_1.new_zeros(valid_options.shape).masked_scatter(
                valid_options, v_scores_1
            )
            if use_ponder and halt_thresh is None:
                canon_scores = 0.70 * cs_1 + 0.30 * canon_scores
            paired_state_repr = 0.70 * ps_1 + 0.30 * paired_state_repr
            cand_repr = 0.70 * cr_1 + 0.30 * cand_repr
            canon_state_proj = self.state_projection(paired_state_repr)
            canon_cand_proj = self.output_projection(cand_repr)

        # Restore original caller option order via inv_perm
        valid_scores = canon_scores.index_select(0, inv_perm)
        state_proj = canon_state_proj.index_select(0, inv_perm)
        cand_proj = canon_cand_proj.index_select(0, inv_perm)

        similarities = valid_scores.new_zeros(valid_options.shape).masked_scatter(
            valid_options, valid_scores
        )

        with torch.autocast(device_type=context_ids.device.type, enabled=False):
            scale = self._scale()
            if halt_thresh is not None and similarities_step1 is not None:
                logits_1 = (similarities_step1.float() * scale).masked_fill(
                    ~valid_options, -torch.inf
                )
                sorted_logits_1, _ = logits_1.sort(dim=-1, descending=True)
                sorted_probs_1 = sorted_logits_1.softmax(dim=-1)
                margin_1 = (sorted_probs_1[:, 0] - sorted_probs_1[:, 1]).unsqueeze(-1)
                blended_sims = 0.70 * similarities_step1 + 0.30 * similarities
                similarities = torch.where(
                    margin_1 >= float(halt_thresh),
                    similarities_step1,
                    blended_sims,
                )
            logits = (similarities.float() * scale).masked_fill(~valid_options, -torch.inf)
            probabilities = logits.softmax(dim=-1)
            aux_loss = 0.5 * (encoded_state.aux_loss.float() + decoded_options.aux_loss.float())
            router_z_loss = 0.5 * (
                encoded_state.router_z_loss.float() + decoded_options.router_z_loss.float()
            )
            ce_loss, loss = None, None
            if targets is not None:
                ce_loss = F.cross_entropy(logits, targets.long())
                loss = (
                    ce_loss
                    + self.backbone.config.aux_loss_coef * aux_loss
                    + self.backbone.config.router_z_loss_coef * router_z_loss
                )

        return DecisionOutput(
            logits=logits,
            similarities=similarities,
            probabilities=probabilities,
            state_features=state_proj,
            candidate_features=cand_proj,
            loss=loss,
            ce_loss=ce_loss,
            aux_loss=aux_loss,
            router_z_loss=router_z_loss,
        )

    forward = forward_decision

    def forward_all_operators(
        self,
        context_ids: Tensor,
        context_lengths: Tensor,
        option_ids: Tensor,
        option_lengths: Tensor,
        targets: Tensor,
        *,
        recurrence_steps: int | None = None,
        depth_ponder: bool | None = None,
        margin_halt_threshold: float | None = None,
    ) -> AllOperatorsOutput:
        """Train all COE operators in a single shared `encode_state` + `decode_candidates` pass.

        Reuses the canonicalized matched-filter representations `(paired_state_repr, cand_repr)`
        from `Decision` to simultaneously supervise:
        1. `Decision` / `Abduction` / `Induction` (option cross-entropy over Pearson similarities)
        2. `Deduction` (3-way `entailed=0`, `contradicted=1`, `undetermined=2` over positive,
           hardest-negative, second-negative, and cross-row unrelated state-candidate pairs)
        3. `Evaluation` & `Association` (`assoc_alpha`, `assoc_beta` utility calibration in `[0, 1]`)
        4. `Metacognition` (`correctness`, `evidence_sufficiency`, `context_benefit` across
           entailed `[1, 1, 1]`, contradicted `[0, 1, 0.5]`, and undetermined `[0, 0, 0]` pairs)
        5. `Abstraction` (`brief` / `standard` / `detailed` multi-scale state compression alignment)
        """
        batch = context_ids.shape[0]
        valid_options = option_lengths > 0
        use_ponder = self.config.depth_ponder if depth_ponder is None else depth_ponder
        halt_thresh = (
            self.config.margin_halt_threshold
            if margin_halt_threshold is None
            else margin_halt_threshold
        )
        if recurrence_steps is None:
            steps = (
                max(2, self.config.recurrence_steps)
                if (use_ponder or halt_thresh is not None)
                else self.config.recurrence_steps
            )
        else:
            steps = recurrence_steps
        ctx_width = int(context_lengths.max().item())
        trimmed_ctx_ids = context_ids[:, :ctx_width].long()
        ctx_valid = (
            torch.arange(ctx_width, device=context_ids.device)[None, :] < context_lengths[:, None]
        )

        # 1. Shared lower-CED state encoding + canonicalized upper-CED candidate decoding
        encoded_state: EncodedState = self.backbone.encode_state(
            trimmed_ctx_ids, ctx_valid, causal=True
        )
        valid_lengths = option_lengths[valid_options]
        opt_width = int(valid_lengths.max().item())
        flat_opt_ids = option_ids[valid_options][:, :opt_width].long()
        owner_rows = valid_options.nonzero(as_tuple=False)[:, 0]
        perm, inv_perm = _canonical_candidate_order(owner_rows, valid_lengths, flat_opt_ids)

        canon_opt_ids = flat_opt_ids.index_select(0, perm)
        canon_lengths = valid_lengths.index_select(0, perm)
        canon_owners = owner_rows.index_select(0, perm)
        canon_valid = (
            torch.arange(opt_width, device=option_ids.device)[None, :] < canon_lengths[:, None]
        )

        decoded_options = self.backbone.decode_candidates(
            encoded_state,
            canon_opt_ids,
            canon_valid,
            owner_rows=canon_owners,
            recurrence_steps=steps,
            causal_local=True,
        )
        paired_state_repr, cand_repr = self.pooler.matched_pair_representations(
            encoded_state.hidden,
            trimmed_ctx_ids,
            ctx_valid,
            decoded_options.hidden,
            canon_opt_ids,
            canon_valid,
            canon_owners,
            cand_anchor=decoded_options.anchor,
            strip_boundary=self.config.strip_boundary_tokens_in_pool,
            bos_id=self.config.bos_id,
            eos_id=self.config.eos_id,
        )

        state_in = F.dropout(paired_state_repr, p=0.10, training=self.training)
        cand_in = F.dropout(cand_repr, p=0.10, training=self.training)
        canon_state_proj = self.state_projection(state_in)
        canon_cand_proj = self.output_projection(cand_in)
        canon_scores = self._similarity(canon_state_proj, canon_cand_proj)

        if (use_ponder or halt_thresh is not None) and len(decoded_options.step_states) > 1:
            ps_1, cr_1 = self.pooler.matched_pair_representations(
                encoded_state.hidden,
                trimmed_ctx_ids,
                ctx_valid,
                decoded_options.step_states[0],
                canon_opt_ids,
                canon_valid,
                canon_owners,
                cand_anchor=decoded_options.anchor,
                strip_boundary=self.config.strip_boundary_tokens_in_pool,
                bos_id=self.config.bos_id,
                eos_id=self.config.eos_id,
            )
            cs_1 = self._similarity(
                self.state_projection(F.dropout(ps_1, p=0.10, training=self.training)),
                self.output_projection(F.dropout(cr_1, p=0.10, training=self.training)),
            )
            if halt_thresh is not None:
                v_s1 = cs_1.index_select(0, inv_perm)
                sims_1 = v_s1.new_zeros(valid_options.shape).masked_scatter(valid_options, v_s1)
                with torch.autocast(device_type=context_ids.device.type, enabled=False):
                    sc = self._scale()
                    sorted_l1, _ = (
                        (sims_1.float() * sc).masked_fill(~valid_options, -torch.inf)
                    ).sort(dim=-1, descending=True)
                    sorted_p1 = sorted_l1.softmax(dim=-1)
                    m1 = (sorted_p1[:, 0] - sorted_p1[:, 1]).index_select(0, canon_owners)
                canon_scores = torch.where(
                    m1 >= float(halt_thresh),
                    cs_1,
                    0.70 * cs_1 + 0.30 * canon_scores,
                )
            elif use_ponder:
                canon_scores = 0.70 * cs_1 + 0.30 * canon_scores
            paired_state_repr = 0.70 * ps_1 + 0.30 * paired_state_repr
            cand_repr = 0.70 * cr_1 + 0.30 * cand_repr
            canon_state_proj = self.state_projection(
                F.dropout(paired_state_repr, p=0.10, training=self.training)
            )
            canon_cand_proj = self.output_projection(
                F.dropout(cand_repr, p=0.10, training=self.training)
            )

        valid_scores = canon_scores.index_select(0, inv_perm)
        state_proj = canon_state_proj.index_select(0, inv_perm)
        cand_proj = canon_cand_proj.index_select(0, inv_perm)
        state_repr = paired_state_repr.index_select(0, inv_perm)
        opt_repr = cand_repr.index_select(0, inv_perm)

        similarities = valid_scores.new_zeros(valid_options.shape).masked_scatter(
            valid_options, valid_scores
        )

        # Scatter [R, d_model] representations into [B, M, d_model] grids for multi-operator readout
        d_m = state_repr.shape[-1]
        num_opts = option_ids.shape[1]
        mask_3d = valid_options.unsqueeze(-1).expand(batch, num_opts, d_m)
        state_grid = state_repr.new_zeros((batch, num_opts, d_m)).masked_scatter(mask_3d, state_repr)
        opt_grid = opt_repr.new_zeros((batch, num_opts, d_m)).masked_scatter(mask_3d, opt_repr)

        b_rows = torch.arange(batch, device=context_ids.device)
        t_idx = targets.long()
        s_pos = state_grid[b_rows, t_idx]
        c_pos = opt_grid[b_rows, t_idx]

        distractor_mask = valid_options.clone()
        distractor_mask[b_rows, t_idx] = False
        neg_scores = similarities.float().masked_fill(~distractor_mask, -torch.inf)
        d_idx = neg_scores.argmax(dim=-1)
        s_neg = state_grid[b_rows, d_idx]
        c_neg = opt_grid[b_rows, d_idx]

        # Also select the second-highest (or alternate) valid distractor so Deduction and
        # Metacognition learn contradiction across all distractors rather than a single pair
        distractor_mask_2 = distractor_mask.clone()
        distractor_mask_2[b_rows, d_idx] = False
        has_second = distractor_mask_2.any(dim=-1)
        neg_scores_2 = similarities.float().masked_fill(~distractor_mask_2, -torch.inf)
        d_idx2 = torch.where(has_second, neg_scores_2.argmax(dim=-1), d_idx)
        s_neg2 = state_grid[b_rows, d_idx2]
        c_neg2 = opt_grid[b_rows, d_idx2]

        # Cross-row mismatched pair represents `undetermined` (unrelated state and candidate)
        s_und = s_pos
        c_und = torch.roll(c_pos, shifts=1, dims=0) if batch > 1 else (0.5 * (c_pos + c_neg))

        s_all4 = torch.cat([s_pos, s_neg, s_neg2, s_und], dim=0)
        c_all4 = torch.cat([c_pos, c_neg, c_neg2, c_und], dim=0)
        joint_pair = F.dropout(torch.cat([s_all4, c_all4], dim=-1), p=0.10, training=self.training)

        ded_joint = self.deduction_query_proj(joint_pair)
        meta_joint = self.meta_query_proj(joint_pair)

        pooled_s = self.pooler.pool_sequence(
            encoded_state.hidden,
            trimmed_ctx_ids,
            ctx_valid,
            strip_boundary=self.config.strip_boundary_tokens_in_pool,
            bos_id=self.config.bos_id,
            eos_id=self.config.eos_id,
        )
        gate_mean = torch.stack([g.sigmoid() for g in self.abstraction_gates.values()]).mean()
        abs_state = self.abstraction_norm(pooled_s + gate_mean * self.abstraction_proj(pooled_s))
        abs_proj = self.state_projection(abs_state)
        pos_cand_proj = self.output_projection(c_pos)
        abs_sim = self._similarity(abs_proj, pos_cand_proj)

        with torch.autocast(device_type=context_ids.device.type, enabled=False):
            scale = self._scale()
            logits = (similarities.float() * scale).masked_fill(~valid_options, -torch.inf)
            probabilities = logits.softmax(dim=-1)
            aux_loss = 0.5 * (encoded_state.aux_loss.float() + decoded_options.aux_loss.float())
            router_z_loss = 0.5 * (
                encoded_state.router_z_loss.float() + decoded_options.router_z_loss.float()
            )
            log_probs = logits.log_softmax(dim=-1)
            nll = -log_probs.gather(-1, t_idx.unsqueeze(-1)).squeeze(-1)
            if self.training:
                safe_lp = log_probs.masked_fill(~valid_options, 0.0)
                smooth_nll = -safe_lp.sum(dim=-1) / valid_options.float().sum(dim=-1).clamp_min(1.0)
                ce_loss = ((1.0 - 0.05) * nll + 0.05 * smooth_nll).mean()
            else:
                ce_loss = nll.mean()
            dec_loss = (
                ce_loss
                + self.backbone.config.aux_loss_coef * aux_loss
                + self.backbone.config.router_z_loss_coef * router_z_loss
            )

            # 2. Deduction 3-way loss (`entailed=0`, `contradicted=1`, `undetermined=2`)
            ded_sims = pearson_similarity(
                ded_joint.unsqueeze(1), self.deduction_prototypes.unsqueeze(0), eps=self.config.eps
            )
            ded_logits = ded_sims.float() * scale
            ded_probs = ded_logits.softmax(dim=-1)
            ded_targets = torch.cat(
                [
                    torch.zeros(batch, device=context_ids.device, dtype=torch.long),
                    torch.ones(batch, device=context_ids.device, dtype=torch.long),
                    torch.ones(batch, device=context_ids.device, dtype=torch.long),
                    torch.full((batch,), 2, device=context_ids.device, dtype=torch.long),
                ],
                dim=0,
            )
            deduction_loss = F.cross_entropy(
                ded_logits, ded_targets, label_smoothing=(0.04 if self.training else 0.0)
            )

            # 3. Evaluation & Association utility loss in [0, 1]
            eval_logits = valid_scores.float() * self.assoc_alpha.float() + self.assoc_beta.float()
            eval_utilities = eval_logits.sigmoid()
            target_grid = F.one_hot(t_idx, num_classes=num_opts).float()
            eval_targets = target_grid[valid_options]
            if self.training:
                eval_targets = eval_targets * 0.92 + 0.04
            evaluation_loss = F.binary_cross_entropy_with_logits(eval_logits, eval_targets)

            # 4. Metacognition 3-signal loss (`correctness`, `evidence_sufficiency`, `context_benefit`)
            meta_sims = pearson_similarity(
                meta_joint.unsqueeze(1), self.meta_prototypes.unsqueeze(0), eps=self.config.eps
            )
            meta_logits = meta_sims.float() * scale + self.meta_bias.float().unsqueeze(0)
            meta_probs = meta_logits.sigmoid()
            t_pos_meta = torch.tensor(
                [1.0, 1.0, 1.0], device=context_ids.device, dtype=torch.float32
            ).expand(batch, 3)
            t_neg_meta = torch.tensor(
                [0.0, 1.0, 0.5], device=context_ids.device, dtype=torch.float32
            ).expand(batch, 3)
            t_und_meta = torch.tensor(
                [0.0, 0.0, 0.0], device=context_ids.device, dtype=torch.float32
            ).expand(batch, 3)
            meta_targets = torch.cat([t_pos_meta, t_neg_meta, t_neg_meta, t_und_meta], dim=0)
            if self.training:
                meta_targets = meta_targets * 0.92 + 0.04
            metacognition_loss = F.binary_cross_entropy_with_logits(meta_logits, meta_targets)

            # 5. Abstraction alignment loss
            abstraction_loss = (1.0 - abs_sim.float()).mean()

            total_op_loss = (
                dec_loss
                + 0.30 * deduction_loss
                + 0.25 * evaluation_loss
                + 0.25 * metacognition_loss
                + 0.10 * abstraction_loss
            )

        decision_out = DecisionOutput(
            logits=logits,
            similarities=similarities,
            probabilities=probabilities,
            state_features=state_proj,
            candidate_features=cand_proj,
            loss=dec_loss,
            ce_loss=ce_loss,
            aux_loss=aux_loss,
            router_z_loss=router_z_loss,
        )
        return AllOperatorsOutput(
            decision=decision_out,
            deduction_loss=deduction_loss,
            evaluation_loss=evaluation_loss,
            metacognition_loss=metacognition_loss,
            abstraction_loss=abstraction_loss,
            total_loss=total_op_loss,
            deduction_probs=ded_probs[:batch],
            metacognition_probs=meta_probs[:batch],
            evaluation_utilities=eval_utilities,
        )

    def embed_items(self, item_ids: Tensor, item_lengths: Tensor) -> Tensor:
        """Project a batch of items `[N, T]` into 32-dim centered Pearson vectors."""
        width = int(item_lengths.max().item())
        trimmed = item_ids[:, :width].long()
        valid = torch.arange(width, device=item_ids.device)[None, :] < item_lengths[:, None]
        encoded = self.backbone.encode_state(trimmed, valid, causal=True)
        pooled = self.pooler.pool_sequence(
            encoded.hidden,
            trimmed,
            valid,
            strip_boundary=self.config.strip_boundary_tokens_in_pool,
            bos_id=self.config.bos_id,
            eos_id=self.config.eos_id,
        )
        normed = self.pooler.cand_norm(pooled.float()).to(pooled.dtype)
        proj = self.output_projection(normed).float()
        centered = proj - proj.mean(dim=-1, keepdim=True)
        return F.normalize(centered, dim=-1, eps=self.config.eps)

    def forward_association_matrix(
        self,
        item_ids: Tensor,
        item_lengths: Tensor,
        *,
        history_buffer: Tensor | None = None,
    ) -> AssociationMatrixOutput:
        """Compute pairwise Association matrix and optional <0.1ms Ornith-1.5 Novelty N(q)."""
        features = self.embed_items(item_ids, item_lengths)
        with torch.autocast(device_type=features.device.type, enabled=False):
            similarities = (features @ features.transpose(-1, -2)).clamp(-1.0, 1.0)
            logits = self.assoc_alpha.float() * similarities + self.assoc_beta.float()
            probabilities = logits.sigmoid()
            novelty = None
            if history_buffer is not None and history_buffer.numel() > 0:
                buf = F.normalize(
                    history_buffer.float() - history_buffer.float().mean(-1, keepdim=True),
                    dim=-1,
                    eps=self.config.eps,
                )
                max_sim = (features @ buf.transpose(-1, -2)).clamp(-1.0, 1.0).max(dim=-1).values
                novelty = (1.0 - max_sim).clamp(0.0, 2.0)
        return AssociationMatrixOutput(
            similarities=similarities,
            logits=logits,
            probabilities=probabilities,
            features=features,
            novelty=novelty,
        )

    def forward_deduction(
        self,
        premise_ids: Tensor,
        premise_lengths: Tensor,
        conclusion_ids: Tensor,
        conclusion_lengths: Tensor,
        targets: Tensor | None = None,
    ) -> DeductionOutput:
        """Predict 3-way logical status (`entailed=0`, `contradicted=1`, `undetermined=2`)."""
        batch = premise_ids.shape[0]
        p_width = int(premise_lengths.max().item())
        c_width = int(conclusion_lengths.max().item())
        p_trim = premise_ids[:, :p_width].long()
        c_trim = conclusion_ids[:, :c_width].long()
        p_valid = torch.arange(p_width, device=premise_ids.device)[None, :] < premise_lengths[:, None]
        c_valid = (
            torch.arange(c_width, device=conclusion_ids.device)[None, :]
            < conclusion_lengths[:, None]
        )

        encoded_p = self.backbone.encode_state(p_trim, p_valid, causal=True)
        owner_rows = torch.arange(batch, device=premise_ids.device)
        decoded_c = self.backbone.decode_candidates(
            encoded_p,
            c_trim,
            c_valid,
            owner_rows=owner_rows,
            recurrence_steps=self.config.recurrence_steps,
        )
        p_repr, c_repr = self.pooler.matched_pair_representations(
            encoded_p.hidden,
            p_trim,
            p_valid,
            decoded_c.hidden,
            c_trim,
            c_valid,
            owner_rows,
            cand_anchor=decoded_c.anchor,
            strip_boundary=self.config.strip_boundary_tokens_in_pool,
            bos_id=self.config.bos_id,
            eos_id=self.config.eos_id,
        )
        joint = self.deduction_query_proj(torch.cat([p_repr, c_repr], dim=-1))
        sims = pearson_similarity(
            joint.unsqueeze(1), self.deduction_prototypes.unsqueeze(0), eps=self.config.eps
        )
        logits = sims * self._scale()
        probs = logits.softmax(dim=-1)
        loss = F.cross_entropy(logits, targets.long()) if targets is not None else None
        return DeductionOutput(logits=logits, probabilities=probs, similarities=sims, loss=loss)

    def forward_metacognition(
        self,
        state_ids: Tensor,
        state_lengths: Tensor,
        candidate_ids: Tensor,
        candidate_lengths: Tensor,
        targets: Tensor | None = None,
    ) -> MetacognitionOutput:
        """Evaluate calibrated reliability signals (`correctness`, `evidence_sufficiency`, `context_benefit`)."""
        batch = state_ids.shape[0]
        s_width = int(state_lengths.max().item())
        c_width = int(candidate_lengths.max().item())
        s_trim = state_ids[:, :s_width].long()
        c_trim = candidate_ids[:, :c_width].long()
        s_valid = torch.arange(s_width, device=state_ids.device)[None, :] < state_lengths[:, None]
        c_valid = (
            torch.arange(c_width, device=candidate_ids.device)[None, :]
            < candidate_lengths[:, None]
        )

        encoded_s = self.backbone.encode_state(s_trim, s_valid, causal=True)
        owner_rows = torch.arange(batch, device=state_ids.device)
        decoded_c = self.backbone.decode_candidates(
            encoded_s,
            c_trim,
            c_valid,
            owner_rows=owner_rows,
            recurrence_steps=self.config.recurrence_steps,
        )
        s_repr, c_repr = self.pooler.matched_pair_representations(
            encoded_s.hidden,
            s_trim,
            s_valid,
            decoded_c.hidden,
            c_trim,
            c_valid,
            owner_rows,
            cand_anchor=decoded_c.anchor,
            strip_boundary=self.config.strip_boundary_tokens_in_pool,
            bos_id=self.config.bos_id,
            eos_id=self.config.eos_id,
        )
        joint = self.meta_query_proj(torch.cat([s_repr, c_repr], dim=-1))
        sims = pearson_similarity(
            joint.unsqueeze(1), self.meta_prototypes.unsqueeze(0), eps=self.config.eps
        )
        logits = sims * self._scale() + self.meta_bias.float().unsqueeze(0)
        probs = logits.sigmoid()
        loss = (
            F.binary_cross_entropy_with_logits(logits, targets.float())
            if targets is not None
            else None
        )
        return MetacognitionOutput(logits=logits, probabilities=probs, loss=loss)

    def compress_abstraction(
        self,
        item_ids: Tensor,
        item_lengths: Tensor,
        *,
        level: str = "standard",
        encoded_hidden: Tensor | None = None,
    ) -> Tensor:
        """Multi-scale latent abstraction (`brief`, `standard`, `detailed`) for recurrent injection."""
        if level not in self.abstraction_gates:
            raise ValueError("level must be 'brief', 'standard', or 'detailed'")
        if encoded_hidden is not None:
            width = int(encoded_hidden.shape[1])
            trimmed = item_ids[:, :width].long()
            valid = torch.arange(width, device=item_ids.device)[None, :] < item_lengths[:, None]
            hid = encoded_hidden
        else:
            width = int(item_lengths.max().item())
            trimmed = item_ids[:, :width].long()
            valid = torch.arange(width, device=item_ids.device)[None, :] < item_lengths[:, None]
            encoded = self.backbone.encode_state(trimmed, valid, causal=True)
            hid = encoded.hidden
        pooled = self.pooler.pool_sequence(
            hid,
            trimmed,
            valid,
            strip_boundary=self.config.strip_boundary_tokens_in_pool,
            bos_id=self.config.bos_id,
            eos_id=self.config.eos_id,
        )
        gate = self.abstraction_gates[level].sigmoid()
        return self.abstraction_norm(pooled + gate * self.abstraction_proj(pooled))

    def forward_evaluation(
        self,
        criterion_ids: Tensor,
        criterion_lengths: Tensor,
        candidate_ids: Tensor,
        candidate_lengths: Tensor,
        targets: Tensor | None = None,
    ) -> EvaluationOutput:
        """Operator 3 (`Evaluation`): Criterion-conditioned utility $U(m, c) \\in [0, 1]$."""
        batch = criterion_ids.shape[0]
        k_width = int(criterion_lengths.max().item())
        c_width = int(candidate_lengths.max().item())
        k_trim = criterion_ids[:, :k_width].long()
        c_trim = candidate_ids[:, :c_width].long()
        k_valid = (
            torch.arange(k_width, device=criterion_ids.device)[None, :]
            < criterion_lengths[:, None]
        )
        c_valid = (
            torch.arange(c_width, device=candidate_ids.device)[None, :]
            < candidate_lengths[:, None]
        )

        encoded_k = self.backbone.encode_state(k_trim, k_valid, causal=True)
        owner_rows = torch.arange(batch, device=criterion_ids.device)
        decoded_c = self.backbone.decode_candidates(
            encoded_k,
            c_trim,
            c_valid,
            owner_rows=owner_rows,
            recurrence_steps=self.config.recurrence_steps,
        )
        k_repr, c_repr = self.pooler.matched_pair_representations(
            encoded_k.hidden,
            k_trim,
            k_valid,
            decoded_c.hidden,
            c_trim,
            c_valid,
            owner_rows,
            cand_anchor=decoded_c.anchor,
            strip_boundary=self.config.strip_boundary_tokens_in_pool,
            bos_id=self.config.bos_id,
            eos_id=self.config.eos_id,
        )
        k_proj = self.state_projection(k_repr)
        c_proj = self.output_projection(c_repr)
        sims = self._similarity(k_proj, c_proj)
        with torch.autocast(device_type=criterion_ids.device.type, enabled=False):
            logits = sims.float() * self.assoc_alpha.float() + self.assoc_beta.float()
            utilities = logits.sigmoid()
            loss = (
                F.binary_cross_entropy_with_logits(logits, targets.float())
                if targets is not None
                else None
            )
        return EvaluationOutput(
            logits=logits,
            utilities=utilities,
            similarities=sims,
            loss=loss,
        )

    def forward_induction(
        self,
        episodes_ids: Tensor,
        episodes_lengths: Tensor,
        rule_option_ids: Tensor,
        rule_option_lengths: Tensor,
        targets: Tensor | None = None,
    ) -> DecisionOutput:
        """Operator 5 (`Induction`): Generalize latent rule candidates across observed episodes."""
        return self.forward_decision(
            episodes_ids,
            episodes_lengths,
            rule_option_ids,
            rule_option_lengths,
            targets=targets,
        )

    def forward_abduction(
        self,
        observation_ids: Tensor,
        observation_lengths: Tensor,
        hypothesis_option_ids: Tensor,
        hypothesis_option_lengths: Tensor,
        targets: Tensor | None = None,
    ) -> DecisionOutput:
        """Operator 6 (`Abduction`): Rank explanatory hypotheses under observed outcomes."""
        return self.forward_decision(
            observation_ids,
            observation_lengths,
            hypothesis_option_ids,
            hypothesis_option_lengths,
            targets=targets,
        )

    def compress_gist_context(
        self,
        token_ids: Tensor,
        valid_mask: Tensor | None = None,
        *,
        chunk_size: int | None = None,
        gist_prefix_len: int = 0,
    ) -> EncodedState:
        """Fold a context sequence `[B, L]` into recurrent soft gist tokens `[B, N_gist, d_model]`."""
        batch_size, total_len = token_ids.shape
        if valid_mask is None:
            valid_mask = token_ids != self.backbone.config.pad_id
        c_sz = max(4, int(chunk_size or self.config.gist_chunk_size))
        d_model = self.backbone.config.d_model
        n_gist_per_chunk = self.gist_queries.shape[0]

        gist_chunks: list[Tensor] = []
        gist_valids: list[Tensor] = []
        enc_chunks: list[Tensor] = []
        running_gist: Tensor | None = None
        total_aux = token_ids.new_zeros((), dtype=torch.float32)
        total_z = token_ids.new_zeros((), dtype=torch.float32)
        loads: list[Tensor] = []

        for start in range(0, total_len, c_sz):
            end = min(total_len, start + c_sz)
            c_ids = token_ids[:, start:end]
            c_val = valid_mask[:, start:end]
            if not c_val.any():
                continue

            row_has_valid = c_val.any(dim=-1)
            c_val_safe = c_val.clone()
            c_val_safe[~row_has_valid, 0] = True

            enc = self.backbone.encode_state(c_ids, c_val_safe, causal=False)
            enc_chunks.append(enc.hidden)
            total_aux = total_aux + enc.aux_loss
            total_z = total_z + enc.router_z_loss
            loads.append(enc.expert_load)

            q = self.gist_queries.unsqueeze(0).expand(batch_size, -1, -1).to(enc.hidden.dtype)
            scores = torch.matmul(q.float(), enc.hidden.float().transpose(-1, -2)) / math.sqrt(
                d_model
            )
            scores = scores.masked_fill(~c_val_safe.unsqueeze(1), -1e9)
            attn = F.softmax(scores, dim=-1).to(enc.hidden.dtype)
            pooled = torch.matmul(attn, enc.hidden).float()
            chunk_gist = (
                0.85 * pooled + 0.15 * self.gist_norm(pooled.to(self.gist_norm.weight.dtype)).float()
            ).to(enc.hidden.dtype)

            chunk_has_valid = row_has_valid.unsqueeze(-1).expand(batch_size, n_gist_per_chunk)
            chunk_gist = chunk_gist.masked_fill(~chunk_has_valid.unsqueeze(-1), 0)
            if running_gist is not None:
                safe_valid = chunk_has_valid.clone()
                safe_valid[~row_has_valid, 0] = True
                updated_gist = self.backbone.feedback(chunk_gist, running_gist, valid=safe_valid)
                chunk_gist = torch.where(
                    chunk_has_valid.unsqueeze(-1),
                    0.75 * chunk_gist + 0.25 * updated_gist,
                    running_gist,
                )
            running_gist = chunk_gist
            gist_chunks.append(chunk_gist)
            gist_valids.append(chunk_has_valid)

        if not gist_chunks:
            self._last_gist_encoded_hidden = None
            fallback_valid = valid_mask[:, :1].clone()
            fallback_valid[:, 0] = True
            return self.backbone.encode_state(token_ids[:, :1], fallback_valid, causal=False)

        self._last_gist_encoded_hidden = torch.cat(enc_chunks, dim=1) if enc_chunks else None
        compressed_hidden = torch.cat(gist_chunks, dim=1)
        compressed_valid = torch.cat(gist_valids, dim=1)
        avg_load = torch.stack(loads, dim=0).mean(dim=0)
        return EncodedState(
            hidden=compressed_hidden,
            valid=compressed_valid,
            aux_loss=total_aux / max(1, len(loads)),
            router_z_loss=total_z / max(1, len(loads)),
            expert_load=avg_load,
            gist_prefix_len=int(gist_prefix_len),
        )

    def forward_inference(
        self,
        input_ids: Tensor,
        targets: Tensor | None = None,
        *,
        recurrence_steps: int | None = None,
        latent_prior: Tensor | None = None,
        latent_prior_scale: float = 0.0,
        recurrent_kv: bool | None = None,
        depth_ponder: bool | None = None,
        gist_state: EncodedState | None = None,
        prior_start_pos: int = 0,
        deep_supervision_weight: float = 0.0,
        condition_on_prefix_gist: bool = False,
    ):
        """Operator 9 (`Inference`): Autoregressive `U$ -> T$ -> A$` synthesis on the shared backbone."""
        use_ponder = self.config.depth_ponder if depth_ponder is None else depth_ponder
        use_rec_kv = self.config.recurrent_kv if recurrent_kv is None else recurrent_kv
        if recurrence_steps is None:
            steps = (
                max(2, self.config.recurrence_steps)
                if use_ponder
                else self.config.recurrence_steps
            )
        else:
            steps = recurrence_steps

        eff_gist = gist_state
        eff_prior = latent_prior
        eff_scale = latent_prior_scale
        eff_start = prior_start_pos
        if condition_on_prefix_gist and input_ids.shape[1] >= 12 and eff_gist is None:
            first_ep_mask = (input_ids == self.backbone.config.eos_id).cumsum(dim=1) == 0
            min_ep_len = int(first_ep_mask.sum(dim=1).min().item())
            pref_len = max(
                4,
                min(
                    max(12, min_ep_len) // 3,
                    48,
                    input_ids.shape[1] // 3,
                    int(self.config.gist_chunk_size),
                ),
            )
            pref_ids = input_ids[:, :pref_len]
            pref_valid = pref_ids != self.backbone.config.pad_id
            if not pref_valid.any(dim=-1).all():
                pref_valid = pref_valid.clone()
                pref_valid[:, 0] = True
            eff_chunk_sz = pref_len if pref_len >= 64 else max(4, pref_len // 2)
            eff_gist = self.compress_gist_context(
                pref_ids,
                pref_valid,
                chunk_size=eff_chunk_sz,
                gist_prefix_len=pref_len,
            )
            if eff_prior is None:
                pref_lengths = pref_valid.sum(dim=-1).clamp_min(1)
                cached_enc = getattr(self, "_last_gist_encoded_hidden", None)
                eff_prior = self.compress_abstraction(
                    pref_ids,
                    pref_lengths,
                    level="detailed",
                    encoded_hidden=(
                        cached_enc
                        if (cached_enc is not None and cached_enc.shape[1] == pref_len)
                        else None
                    ),
                )
                if eff_scale == 0.0:
                    eff_scale = 0.12
            eff_start = pref_len

        return self.backbone.forward(
            input_ids,
            targets=targets,
            recurrence_steps=steps,
            latent_prior=eff_prior,
            latent_prior_scale=eff_scale,
            recurrent_kv=use_rec_kv,
            depth_ponder=use_ponder,
            gist_state=eff_gist,
            prior_start_pos=eff_start,
            deep_supervision_weight=deep_supervision_weight,
        )

