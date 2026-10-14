"""Bounded Markov Chain Local-4 (MCL4) attention router and sparse QKV readout.

Unifies mcattn/fast_markov.py and oracle/markov_local4.py:
- 4 mandatory local anchor tokens {q, q-1, q-2, q-3} prevent Layer-0 byte starvation.
- Up to 12 bounded Markov-walked positions (16 total attended tokens per query)
  execute 3 relational graph hops per layer while eliminating softmax dilution.
- Bounded beam propagation (`_coalesce`) merges duplicate visited destinations
  differentiably, and a straight-through visitation-mass surrogate trains the
  four routing projections without altering forward QKV values.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def coalesce_candidates(ids: Tensor, mass: Tensor) -> tuple[Tensor, Tensor]:
    """Sum duplicate token visitation masses and return sorted unique IDs with zero padding."""
    sorted_ids, order = ids.sort(dim=-1, stable=True)
    sorted_mass = mass.gather(-1, order)
    boundary = torch.ones_like(sorted_ids, dtype=torch.bool)
    boundary[..., 1:] = sorted_ids[..., 1:] != sorted_ids[..., :-1]
    groups = boundary.long().cumsum(-1) - 1
    merged_ids = torch.zeros_like(ids).scatter(-1, groups, sorted_ids)
    merged_mass = torch.zeros_like(mass).scatter_add(-1, groups, sorted_mass)
    return merged_ids, merged_mass


def _gather_nodes(nodes: Tensor, ids: Tensor) -> Tensor:
    """Gather [B, H, N, D] node features at [B, H, ...] indices via embedding lookup."""
    b, h, n, d = nodes.shape
    offsets = torch.arange(b * h, device=ids.device).reshape(b, h, *([1] * (ids.ndim - 2))) * n
    return F.embedding(ids + offsets, nodes.reshape(b * h * n, d))


def _supported(allowed: Tensor, ids: Tensor) -> Tensor:
    """Gather query-specific support [B, 1 or H, Q, N] at candidate IDs."""
    if allowed.shape[1] == 1 and ids.shape[1] > 1:
        allowed = allowed.expand(-1, ids.shape[1], -1, -1)
    return allowed.gather(-1, ids.flatten(3)).reshape_as(ids)


class MarkovLocal4Router(nn.Module):
    """Bounded Markov graph walker with 4 mandatory local anchors."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        router_dim: int = 16,
        graph_degree: int = 4,
        num_walkers: int = 4,
        steps: int = 3,
        attend_tokens: int = 16,
        local_tokens: int = 4,
        beam_size: int = 8,
        restart: float = 0.1,
        query_chunk_size: int = 512,
    ) -> None:
        super().__init__()
        if d_model <= 0 or n_heads <= 0 or d_model % n_heads != 0:
            raise ValueError("d_model must be a positive multiple of n_heads")
        if local_tokens < 0 or local_tokens > attend_tokens:
            raise ValueError("local_tokens must be between 0 and attend_tokens")
        self.d_model = d_model
        self.n_heads = n_heads
        self.router_dim = router_dim
        self.graph_degree = graph_degree
        self.num_walkers = num_walkers
        self.steps = steps
        self.attend_tokens = attend_tokens
        self.local_tokens = local_tokens
        self.beam_size = beam_size
        self.restart = restart
        self.query_chunk_size = query_chunk_size

        for name in ("route_query", "route_state", "route_key", "start_query"):
            setattr(self, name, nn.Linear(d_model, n_heads * router_dim, bias=False))

    def _project(self, x: Tensor, name: str) -> Tensor:
        b, n, _ = x.shape
        out = getattr(self, name)(x).view(b, n, self.n_heads, self.router_dim).transpose(1, 2)
        return out.float().contiguous() if out.is_cuda else out.float()

    def project_routes(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return (
            self._project(x, "route_query"),
            self._project(x, "route_state"),
            self._project(x, "route_key"),
            self._project(x, "start_query"),
        )

    @torch.no_grad()
    def build_graph(self, keys: Tensor, key_valid: Tensor) -> tuple[Tensor, Tensor]:
        """Construct bounded-degree causal token graph (7 local offsets + top-k semantic earlier)."""
        b, h, n, _ = keys.shape
        with torch.autocast(device_type=keys.device.type, enabled=False):
            sim_dtype = torch.bfloat16 if keys.is_cuda else torch.float32
            normalized = F.normalize(keys.float(), dim=-1).to(sim_dtype)
            similarity = normalized @ normalized.transpose(-1, -2)
            position = torch.arange(n, device=keys.device)
            earlier = position[None, :] < position[:, None]
            eligible = earlier[None, None] & key_valid[:, None, None, :] & key_valid[:, None, :, None]
            similarity = similarity.masked_fill(~eligible, -torch.inf)
            semantic_k = min(n, self.graph_degree)
            semantic = torch.topk(similarity, k=semantic_k, dim=-1).indices
            semantic_valid = eligible.expand(b, h, -1, -1).gather(-1, semantic)

            offsets = torch.tensor([0, -1, 1, -2, 2, -4, 4], device=keys.device)
            raw = position[:, None] + offsets
            local = raw.clamp(0, n - 1).expand(b, h, -1, -1)
            node_valid = key_valid[:, None, :, None].expand(b, h, -1, 1)
            dest_valid = _gather_nodes(node_valid, local).squeeze(-1)
            local_valid = ((raw >= 0) & (raw < n))[None, None] & dest_valid & node_valid
            local_valid[..., 0] = True

            indices = torch.cat((local, semantic), dim=-1)
            valid = torch.cat((local_valid, semantic_valid), dim=-1)
            degree = indices.shape[-1]
            earlier_slot = torch.ones(degree, degree, device=keys.device, dtype=torch.bool).tril(-1)
            duplicate = (
                (indices.unsqueeze(-1) == indices.unsqueeze(-2))
                & earlier_slot
                & valid.unsqueeze(-2)
            ).any(-1)
            return indices, valid & ~duplicate

    def _allowed_nodes(self, key_valid: Tensor, positions: Tensor, causal: bool) -> Tensor:
        n = key_valid.shape[-1]
        allowed = key_valid[:, None, None, :].expand(-1, 1, positions.numel(), -1)
        if causal:
            allowed = allowed & (torch.arange(n, device=positions.device) <= positions[:, None])
        fallback = torch.arange(n, device=positions.device) == positions[:, None]
        return allowed | (~allowed.any(-1, keepdim=True) & fallback)

    def sparse_seeds(self, starts: Tensor, keys: Tensor, allowed: Tensor) -> tuple[Tensor, Tensor]:
        k = min(self.num_walkers, keys.shape[-2])
        if keys.is_cuda:
            with torch.no_grad():
                scores_bf16 = (
                    starts.to(torch.bfloat16) @ keys.to(torch.bfloat16).transpose(-1, -2)
                ).masked_fill(~allowed, float("-inf"))
                ids = torch.topk(scores_bf16, k=k, dim=-1).indices
            with torch.autocast(device_type=keys.device.type, enabled=False):
                sel_keys = _gather_nodes(keys.float(), ids)
                top_scores = torch.matmul(
                    starts.float().unsqueeze(-2), sel_keys.transpose(-1, -2)
                ).squeeze(-2) / math.sqrt(self.router_dim)
                top_scores = top_scores.masked_fill(~_supported(allowed, ids), -torch.inf)
                return ids, top_scores.softmax(-1)
        with torch.autocast(device_type=keys.device.type, enabled=False):
            scores = (starts.float() @ keys.float().transpose(-1, -2)) / math.sqrt(self.router_dim)
            scores = scores.masked_fill(~allowed, -torch.inf)
            top_scores, ids = torch.topk(scores, k=k, dim=-1)
            return ids, top_scores.softmax(-1)

    def transitions_at_sources(
        self,
        queries: Tensor,
        states: Tensor,
        indices: Tensor,
        valid: Tensor,
        allowed: Tensor,
        source_ids: Tensor,
        neighbor_key_rows: Tensor,
    ) -> tuple[Tensor, Tensor]:
        destinations = _gather_nodes(indices, source_ids)
        permitted = (
            _gather_nodes(valid, source_ids)
            & _supported(allowed, destinations)
            & _supported(allowed, source_ids).unsqueeze(-1)
        )
        if queries.is_cuda:
            q_bf = queries.to(torch.bfloat16)
            source_states = _gather_nodes(states.to(torch.bfloat16), source_ids)
            neighbor_keys = _gather_nodes(neighbor_key_rows, source_ids).reshape(
                *destinations.shape, self.router_dim
            )
            condition = torch.tanh(q_bf.unsqueeze(-2) + source_states)
            scores = (
                torch.matmul(condition.unsqueeze(-2), neighbor_keys.transpose(-1, -2))
                .squeeze(-2)
                .float()
                / math.sqrt(self.router_dim)
            )
            scores = scores.masked_fill(~permitted, -torch.inf)
            scores = torch.where(permitted.any(-1, keepdim=True), scores, torch.zeros_like(scores))
            return destinations, scores.softmax(-1).masked_fill(~permitted, 0)
        with torch.autocast(device_type=queries.device.type, enabled=False):
            source_states = _gather_nodes(states.float(), source_ids)
            neighbor_keys = _gather_nodes(neighbor_key_rows, source_ids).reshape(
                *destinations.shape, self.router_dim
            )
            condition = torch.tanh(queries.float().unsqueeze(-2) + source_states)
            scores = torch.matmul(
                condition.unsqueeze(-2), neighbor_keys.transpose(-1, -2)
            ).squeeze(-2) / math.sqrt(self.router_dim)
            scores = scores.masked_fill(~permitted, -torch.inf)
            scores = torch.where(permitted.any(-1, keepdim=True), scores, torch.zeros_like(scores))
            return destinations, scores.softmax(-1).masked_fill(~permitted, 0)

    def bounded_visits(
        self,
        queries: Tensor,
        states: Tensor,
        n_tokens: int,
        indices: Tensor,
        valid: Tensor,
        allowed: Tensor,
        seed_ids: Tensor,
        seed_mass: Tensor,
        neighbor_key_rows: Tensor,
        *,
        coalesce_final: bool = True,
    ) -> tuple[Tensor, Tensor]:
        current_ids, current_mass = seed_ids, seed_mass
        visited_ids, visited_mass = [seed_ids], [seed_mass]
        for _ in range(self.steps):
            destinations, probabilities = self.transitions_at_sources(
                queries, states, indices, valid, allowed, current_ids, neighbor_key_rows
            )
            candidate_ids = torch.cat((destinations.flatten(-2), seed_ids), dim=-1)
            edge_mass = (1.0 - self.restart) * current_mass.unsqueeze(-1) * probabilities
            candidate_mass = torch.cat((edge_mass.flatten(-2), self.restart * seed_mass), dim=-1)
            if n_tokens <= 2 * candidate_ids.shape[-1]:
                merged_mass = candidate_mass.new_zeros(*candidate_mass.shape[:-1], n_tokens)
                merged_mass = merged_mass.scatter_add(-1, candidate_ids, candidate_mass)
                merged_ids = torch.arange(n_tokens, device=queries.device).expand_as(merged_mass)
            else:
                merged_ids, merged_mass = coalesce_candidates(candidate_ids, candidate_mass)
            width = min(self.beam_size, n_tokens, merged_mass.shape[-1])
            chosen = merged_mass.argsort(dim=-1, descending=True, stable=True)[..., :width]
            current_ids = merged_ids.gather(-1, chosen)
            current_mass = merged_mass.gather(-1, chosen)
            current_mass = current_mass / current_mass.sum(-1, keepdim=True).clamp_min(
                torch.finfo(current_mass.dtype).tiny
            )
            visited_ids.append(current_ids)
            visited_mass.append(current_mass)
        all_ids = torch.cat(visited_ids, dim=-1)
        all_mass = torch.cat(visited_mass, dim=-1) / (self.steps + 1)
        if not coalesce_final:
            return all_ids, all_mass
        return coalesce_candidates(all_ids, all_mass)

    def select_subset(
        self, ids: Tensor, mass: Tensor, positions: Tensor, allowed: Tensor
    ) -> Tensor:
        """Reserve up to `local_tokens` local anchors, then rank distinct Markov-visited tokens."""
        query = positions.view(1, 1, -1, 1)
        zero = ids == 0
        unique = ~zero | (zero.long().cumsum(-1) == 1)
        supported = _supported(allowed, ids)
        anchors = unique & supported & (ids <= query) & (ids > query - self.local_tokens)
        routed = (mass > 0) & supported & ~anchors
        ranked = mass.masked_fill(~routed, -torch.inf).argsort(
            dim=-1, descending=True, stable=True
        )[..., : min(self.attend_tokens, ids.shape[-1])]
        available = self.attend_tokens - anchors.sum(-1, keepdim=True)
        take = torch.arange(ranked.shape[-1], device=ids.device) < available
        take = take & routed.gather(-1, ranked)
        return anchors | torch.zeros_like(anchors).scatter(-1, ranked, take)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        x_norm: Tensor,
        key_valid: Tensor,
        *,
        causal: bool = True,
        dropout_p: float = 0.0,
    ) -> Tensor:
        """Route and compute sparse QKV attention over 4 local anchors + 12 Markov walkers."""
        b, h, n, head_dim = q.shape
        queries, states, keys, starts = self.project_routes(x_norm)
        indices, valid = self.build_graph(keys, key_valid)
        nk_dtype = torch.bfloat16 if keys.is_cuda else torch.float32
        neighbor_key_rows = _gather_nodes(keys.to(nk_dtype), indices).flatten(-2)
        chunks = []
        for start in range(0, n, self.query_chunk_size):
            end = min(start + self.query_chunk_size, n)
            positions = torch.arange(start, end, device=q.device)
            allowed = self._allowed_nodes(key_valid, positions, causal=causal)
            seed_ids, seed_mass = self.sparse_seeds(starts[:, :, start:end], keys, allowed)
            ids, mass = self.bounded_visits(
                queries[:, :, start:end],
                states,
                n,
                indices,
                valid,
                allowed,
                seed_ids,
                seed_mass,
                neighbor_key_rows,
                coalesce_final=(self.local_tokens <= 0),
            )
            if self.local_tokens > 0:
                anchors = (
                    positions[:, None] - torch.arange(self.local_tokens, device=q.device)
                ).clamp_min(0)
                anchors = anchors.expand(b, h, -1, -1)
                ids, mass = coalesce_candidates(
                    torch.cat((ids, anchors), dim=-1),
                    torch.cat((mass, torch.zeros_like(anchors, dtype=mass.dtype)), dim=-1),
                )
            q_chunk = q[:, :, start:end]
            use_dense = (
                q.is_cuda
                and self.training
                and torch.is_grad_enabled()
                and q_chunk.shape[-2] > 1
                and n <= 2 * ids.shape[-1] * head_dim
            )
            if use_dense:
                scores = torch.matmul(q_chunk, k.transpose(-1, -2)).gather(-1, ids)
            else:
                selected_k = _gather_nodes(k, ids)
                scores = torch.matmul(q_chunk.unsqueeze(-2), selected_k.transpose(-1, -2)).squeeze(
                    -2
                )
            scores = scores.float() / math.sqrt(head_dim)
            hard_mask = self.select_subset(ids, mass, positions, allowed)
            with torch.autocast(device_type=q.device.type, enabled=False):
                hard_weights = scores.masked_fill(~hard_mask, -torch.inf).softmax(-1)
                if self.training and torch.is_grad_enabled():
                    soft_allowed = (mass > 0) & _supported(allowed, ids)
                    soft_logits = (
                        scores.detach() + mass.clamp_min(torch.finfo(torch.float32).tiny).log()
                    )
                    soft_weights = soft_logits.masked_fill(~soft_allowed, -torch.inf).softmax(-1)
                    weights = hard_weights + (soft_weights - soft_weights.detach())
                else:
                    weights = hard_weights
            if dropout_p > 0.0 and self.training:
                weights = F.dropout(weights, p=dropout_p, training=True)
            if use_dense:
                cand_w = weights.to(v.dtype)
                dense_w = cand_w.new_zeros(*weights.shape[:-1], n).scatter_add(-1, ids, cand_w)
                attended = torch.matmul(dense_w, v)
            else:
                selected_v = _gather_nodes(v, ids)
                attended = torch.matmul(weights.to(v.dtype).unsqueeze(-2), selected_v).squeeze(-2)
            chunks.append(attended)
        return torch.cat(chunks, dim=2)
