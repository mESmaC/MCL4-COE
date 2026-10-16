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

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

_MCL4_CUDA_EXT: Any = None
_MCL4_CUDA_EXT_ATTEMPTED: bool = False

_MCL4_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void coalesce_topk_warp_kernel(
    const int64_t* __restrict__ ids,
    const float* __restrict__ mass,
    int64_t* __restrict__ out_ids,
    float* __restrict__ out_mass,
    int32_t* __restrict__ in_to_k,
    float* __restrict__ out_denom,
    int R,
    int C,
    int N,
    int K,
    bool record_grad
) {
    extern __shared__ float smem[];
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp_id;

    float* row_smem = smem + warp_id * N;
    for (int i = lane; i < N; i += 32) {
        row_smem[i] = 0.0f;
    }
    __syncwarp();

    if (row >= R) return;

    const int64_t* r_ids = ids + (size_t)row * C;
    const float* r_mass = mass + (size_t)row * C;

    int local_id[6];
    float local_m[6];
    int n_items = 0;

    for (int i = lane; i < C; i += 32) {
        int id = (int)r_ids[i];
        float m = r_mass[i];
        if (id >= 0 && id < N && m > 0.0f) {
            atomicAdd(&row_smem[id], m);
        }
    }
    __syncwarp();

    for (int i = lane; i < C; i += 32) {
        int id = (int)r_ids[i];
        float m = 0.0f;
        if (id >= 0 && id < N) {
            m = atomicExch(&row_smem[id], 0.0f);
        }
        local_id[n_items] = (id >= 0 && id < N) ? id : 0;
        local_m[n_items] = m;
        n_items++;
    }
    __syncwarp();

    int64_t* o_ids = out_ids + (size_t)row * K;
    float* o_mass = out_mass + (size_t)row * K;
    float mass_sum = 0.0f;

    for (int k = 0; k < K; ++k) {
        float best_m = -1.0f;
        int best_id = 0x7fffffff;
        int best_slot = -1;
        for (int s = 0; s < n_items; ++s) {
            float m = local_m[s];
            int id = local_id[s];
            if (m > best_m || (m == best_m && id < best_id)) {
                best_m = m;
                best_id = id;
                best_slot = s;
            }
        }
        float w_m = best_m;
        int w_id = best_id;
        int w_lane = lane;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float o_m = __shfl_xor_sync(0xffffffff, w_m, offset);
            int o_id = __shfl_xor_sync(0xffffffff, w_id, offset);
            int o_lane = __shfl_xor_sync(0xffffffff, w_lane, offset);
            if (o_m > w_m || (o_m == w_m && o_id < w_id) || (o_m == w_m && o_id == w_id && o_lane < w_lane)) {
                w_m = o_m;
                w_id = o_id;
                w_lane = o_lane;
            }
        }
        if (lane == w_lane && best_slot >= 0) {
            local_m[best_slot] = -2.0f;
        }
        if (lane == 0) {
            float final_m = (w_m > 0.0f) ? w_m : 0.0f;
            int final_id = (w_m > 0.0f) ? w_id : 0;
            o_ids[k] = (int64_t)final_id;
            o_mass[k] = final_m;
            mass_sum += final_m;
            if (record_grad && w_m > 0.0f) {
                row_smem[final_id] = (float)(k + 1);
            }
        }
    }
    __syncwarp();

    float denom = __shfl_sync(0xffffffff, mass_sum, 0);
    if (denom < 1e-38f) denom = 1e-38f;
    if (lane < K) {
        o_mass[lane] /= denom;
    }
    if (record_grad) {
        if (lane == 0) {
            out_denom[row] = denom;
        }
        int32_t* r_map = in_to_k + (size_t)row * C;
        for (int i = lane; i < C; i += 32) {
            int id = (int)r_ids[i];
            int k_plus_1 = (id >= 0 && id < N) ? (int)row_smem[id] : 0;
            r_map[i] = k_plus_1 - 1;
        }
        __syncwarp();
        if (lane < K) {
            int fid = (int)o_ids[lane];
            if (fid >= 0 && fid < N) row_smem[fid] = 0.0f;
        }
    }
}

__global__ void coalesce_topk_bwd_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ out_mass,
    const int32_t* __restrict__ in_to_k,
    const float* __restrict__ out_denom,
    float* __restrict__ grad_in,
    int R,
    int C,
    int K
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_id >= R) return;

    const float* g_out = grad_out + (size_t)warp_id * K;
    const float* o_m = out_mass + (size_t)warp_id * K;
    const int32_t* r_map = in_to_k + (size_t)warp_id * C;
    float* g_in = grad_in + (size_t)warp_id * C;
    float denom = out_denom[warp_id];

    float dot = 0.0f;
    if (lane < K) {
        dot = g_out[lane] * o_m[lane];
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        dot += __shfl_xor_sync(0xffffffff, dot, offset);
    }
    float g_k[16];
    #pragma unroll
    for (int k = 0; k < 16; ++k) {
        g_k[k] = (k < K) ? ((g_out[k] - dot) / denom) : 0.0f;
    }
    for (int i = lane; i < C; i += 32) {
        int k = r_map[i];
        g_in[i] = (k >= 0 && k < K) ? g_k[k] : 0.0f;
    }
}

__global__ void coalesce_sorted_small_warp_kernel(
    const int64_t* __restrict__ ids,
    const float* __restrict__ mass,
    int64_t* __restrict__ merged_ids,
    float* __restrict__ merged_mass,
    int64_t* __restrict__ orig_groups,
    int R,
    int C
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_id >= R) return;

    const int64_t* r_ids = ids + (size_t)warp_id * C;
    const float* r_mass = mass + (size_t)warp_id * C;
    int64_t* m_ids = merged_ids + (size_t)warp_id * C;
    float* m_mass = merged_mass + (size_t)warp_id * C;
    int64_t* o_grp = orig_groups + (size_t)warp_id * C;

    int my_id = (lane < C) ? (int)r_ids[lane] : 0x7fffffff;
    float my_m = (lane < C) ? r_mass[lane] : 0.0f;
    if (lane < C) {
        m_ids[lane] = 0;
        m_mass[lane] = 0.0f;
    }

    unsigned int active_mask = (C >= 32) ? 0xffffffffu : ((1u << C) - 1u);
    int same_mask = 0;
    for (int src = 0; src < C; ++src) {
        int other_id = __shfl_sync(0xffffffff, my_id, src);
        if (other_id == my_id) {
            same_mask |= (1 << src);
        }
    }
    int first_lane = __ffs(same_mask) - 1;
    bool is_rep = (lane < C) && (lane == first_lane);

    unsigned int rep_mask = __ballot_sync(0xffffffff, is_rep) & active_mask;
    int my_group = 0;
    for (int src = 0; src < C; ++src) {
        int other_id = __shfl_sync(0xffffffff, my_id, src);
        if ((rep_mask & (1u << src)) && (other_id < my_id)) {
            my_group++;
        }
    }
    float sum_m = 0.0f;
    for (int src = 0; src < C; ++src) {
        float other_m = __shfl_sync(0xffffffff, my_m, src);
        if (same_mask & (1 << src)) {
            sum_m += other_m;
        }
    }
    if (lane < C) {
        o_grp[lane] = (int64_t)my_group;
        if (is_rep) {
            m_ids[my_group] = (int64_t)my_id;
            m_mass[my_group] = sum_m;
        }
    }
}

__global__ void select_subset_small_warp_kernel(
    const int64_t* __restrict__ ids,
    const float* __restrict__ mass,
    const bool* __restrict__ supported,
    const int64_t* __restrict__ positions,
    bool* __restrict__ out_mask,
    int R,
    int Q,
    int C,
    int local_tokens,
    int attend_tokens
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_id >= R) return;

    int q_idx = warp_id % Q;
    int q_pos = (int)positions[q_idx];

    const int64_t* r_ids = ids + (size_t)warp_id * C;
    const float* r_mass = mass + (size_t)warp_id * C;
    const bool* r_sup = supported + (size_t)warp_id * C;
    bool* o_mask = out_mask + (size_t)warp_id * C;

    int my_id = (lane < C) ? (int)r_ids[lane] : 0;
    float my_m = (lane < C) ? r_mass[lane] : 0.0f;
    bool my_sup = (lane < C) ? r_sup[lane] : false;

    unsigned int active_mask = (C >= 32) ? 0xffffffffu : ((1u << C) - 1u);
    unsigned int zero_mask = __ballot_sync(0xffffffff, (lane < C) && (my_id == 0)) & active_mask;
    int first_zero = __ffs(zero_mask) - 1;
    bool is_unique = (lane < C) && ((my_id != 0) || (lane == first_zero));

    bool is_anchor = is_unique && my_sup && (my_id <= q_pos) && (my_id > q_pos - local_tokens);
    bool is_routed = (lane < C) && (my_m > 0.0f) && my_sup && (!is_anchor);

    int num_anchors = __popc(__ballot_sync(0xffffffff, is_anchor) & active_mask);
    int available = attend_tokens - num_anchors;

    int my_rank = 0;
    for (int src = 0; src < C; ++src) {
        float other_m = __shfl_sync(0xffffffff, my_m, src);
        bool other_routed = __shfl_sync(0xffffffff, (int)is_routed, src) != 0;
        if (other_routed) {
            if (other_m > my_m || (other_m == my_m && src < lane)) {
                my_rank++;
            }
        }
    }
    if (lane < C) {
        o_mask[lane] = is_anchor || (is_routed && (my_rank < available));
    }
}

std::vector<torch::Tensor> coalesce_topk_fwd_cuda(torch::Tensor ids, torch::Tensor mass, int n_tokens, int k, bool record_grad) {
    auto sizes = ids.sizes().vec();
    int C = sizes.back();
    int R = ids.numel() / C;
    sizes.back() = k;
    auto out_ids = torch::empty(sizes, ids.options());
    auto out_mass = torch::empty(sizes, mass.options());
    auto in_to_k = record_grad ? torch::empty(ids.sizes(), ids.options().dtype(torch::kInt32)) : torch::empty({0}, ids.options().dtype(torch::kInt32));
    auto out_denom = record_grad ? torch::empty({R}, mass.options()) : torch::empty({0}, mass.options());

    int warps_per_block = 4;
    int threads = warps_per_block * 32;
    int blocks = (R + warps_per_block - 1) / warps_per_block;
    size_t smem_bytes = warps_per_block * n_tokens * sizeof(float);

    coalesce_topk_warp_kernel<<<blocks, threads, smem_bytes>>>(
        ids.data_ptr<int64_t>(),
        mass.data_ptr<float>(),
        out_ids.data_ptr<int64_t>(),
        out_mass.data_ptr<float>(),
        record_grad ? in_to_k.data_ptr<int32_t>() : nullptr,
        record_grad ? out_denom.data_ptr<float>() : nullptr,
        R, C, n_tokens, k, record_grad
    );
    return {out_ids, out_mass, in_to_k, out_denom};
}

torch::Tensor coalesce_topk_bwd_cuda(torch::Tensor grad_out, torch::Tensor out_mass, torch::Tensor in_to_k, torch::Tensor out_denom) {
    int C = in_to_k.size(-1);
    int K = out_mass.size(-1);
    int R = out_mass.numel() / K;
    auto grad_in = torch::empty(in_to_k.sizes(), out_mass.options());

    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int blocks = (R + warps_per_block - 1) / warps_per_block;

    coalesce_topk_bwd_kernel<<<blocks, threads>>>(
        grad_out.data_ptr<float>(),
        out_mass.data_ptr<float>(),
        in_to_k.data_ptr<int32_t>(),
        out_denom.data_ptr<float>(),
        grad_in.data_ptr<float>(),
        R, C, K
    );
    return grad_in;
}

std::vector<torch::Tensor> coalesce_sorted_small_cuda(torch::Tensor ids, torch::Tensor mass) {
    int C = ids.size(-1);
    int R = ids.numel() / C;
    auto merged_ids = torch::empty_like(ids);
    auto merged_mass = torch::empty_like(mass);
    auto orig_groups = torch::empty_like(ids);

    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int blocks = (R + warps_per_block - 1) / warps_per_block;

    coalesce_sorted_small_warp_kernel<<<blocks, threads>>>(
        ids.data_ptr<int64_t>(),
        mass.data_ptr<float>(),
        merged_ids.data_ptr<int64_t>(),
        merged_mass.data_ptr<float>(),
        orig_groups.data_ptr<int64_t>(),
        R, C
    );
    return {merged_ids, merged_mass, orig_groups};
}

torch::Tensor select_subset_small_cuda(torch::Tensor ids, torch::Tensor mass, torch::Tensor supported, torch::Tensor positions, int local_tokens, int attend_tokens) {
    int C = ids.size(-1);
    int Q = positions.numel();
    int R = ids.numel() / C;
    auto out_mask = torch::empty(ids.sizes(), ids.options().dtype(torch::kBool));

    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int blocks = (R + warps_per_block - 1) / warps_per_block;

    select_subset_small_warp_kernel<<<blocks, threads>>>(
        ids.data_ptr<int64_t>(),
        mass.data_ptr<float>(),
        supported.data_ptr<bool>(),
        positions.data_ptr<int64_t>(),
        out_mask.data_ptr<bool>(),
        R, Q, C, local_tokens, attend_tokens
    );
    return out_mask;
}
"""

_MCL4_CPP_SRC = """
std::vector<torch::Tensor> coalesce_topk_fwd_cuda(torch::Tensor ids, torch::Tensor mass, int n_tokens, int k, bool record_grad);
torch::Tensor coalesce_topk_bwd_cuda(torch::Tensor grad_out, torch::Tensor out_mass, torch::Tensor in_to_k, torch::Tensor out_denom);
std::vector<torch::Tensor> coalesce_sorted_small_cuda(torch::Tensor ids, torch::Tensor mass);
torch::Tensor select_subset_small_cuda(torch::Tensor ids, torch::Tensor mass, torch::Tensor supported, torch::Tensor positions, int local_tokens, int attend_tokens);
"""


def _get_mcl4_cuda_ext() -> Any:
    """Lazy-compile or load cached CUDA warp extension for MarkovLocal4Router."""
    global _MCL4_CUDA_EXT, _MCL4_CUDA_EXT_ATTEMPTED
    if _MCL4_CUDA_EXT_ATTEMPTED:
        return _MCL4_CUDA_EXT
    _MCL4_CUDA_EXT_ATTEMPTED = True
    if not torch.cuda.is_available():
        return None
    try:
        from torch.utils.cpp_extension import load_inline

        _MCL4_CUDA_EXT = load_inline(
            name="mcl4_fast_ext_v3",
            cpp_sources=_MCL4_CPP_SRC,
            cuda_sources=_MCL4_CUDA_SRC,
            functions=[
                "coalesce_topk_fwd_cuda",
                "coalesce_topk_bwd_cuda",
                "coalesce_sorted_small_cuda",
                "select_subset_small_cuda",
            ],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    except Exception:
        _MCL4_CUDA_EXT = None
    return _MCL4_CUDA_EXT


class _CoalesceTopKAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, ids: Tensor, mass: Tensor, n_tokens: int, k: int) -> tuple[Tensor, Tensor]:
        ext = _get_mcl4_cuda_ext()
        req = bool(mass.requires_grad)
        out_ids, out_mass, in_to_k, out_denom = ext.coalesce_topk_fwd_cuda(
            ids.contiguous(), mass.float().contiguous(), int(n_tokens), int(k), req
        )
        if req:
            ctx.save_for_backward(out_mass, in_to_k, out_denom)
        return out_ids, out_mass.to(mass.dtype)

    @staticmethod
    def backward(ctx: Any, grad_ids: Tensor | None, grad_mass: Tensor) -> tuple[None, Tensor, None, None]:
        ext = _get_mcl4_cuda_ext()
        out_mass, in_to_k, out_denom = ctx.saved_tensors
        grad_in = ext.coalesce_topk_bwd_cuda(
            grad_mass.float().contiguous(), out_mass, in_to_k, out_denom
        )
        return None, grad_in.to(grad_mass.dtype), None, None


class _CoalesceSmallAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, ids: Tensor, mass: Tensor) -> tuple[Tensor, Tensor]:
        ext = _get_mcl4_cuda_ext()
        m_ids, m_mass, o_grp = ext.coalesce_sorted_small_cuda(
            ids.contiguous(), mass.float().contiguous()
        )
        if mass.requires_grad:
            ctx.save_for_backward(o_grp)
        return m_ids, m_mass.to(mass.dtype)

    @staticmethod
    def backward(ctx: Any, grad_ids: Tensor | None, grad_mass: Tensor) -> tuple[None, Tensor]:
        (o_grp,) = ctx.saved_tensors
        return None, grad_mass.gather(-1, o_grp)


def coalesce_candidates(ids: Tensor, mass: Tensor) -> tuple[Tensor, Tensor]:
    """Sum duplicate token visitation masses and return sorted unique IDs with zero padding."""
    if ids.is_cuda and ids.shape[-1] <= 32 and _get_mcl4_cuda_ext() is not None:
        return _CoalesceSmallAutograd.apply(ids, mass)
    if ids.is_cuda:
        sorted_ids_i32, order = ids.int().sort(dim=-1, stable=False)
        sorted_ids = sorted_ids_i32.long()
    else:
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
        query_chunk_size: int = 1024,
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

            offsets = torch.tensor(
                [0, -1, 1, -2, 2, -3, -4, 4, -8, -16, -32, -64], device=keys.device
            )
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
        ext = _get_mcl4_cuda_ext() if queries.is_cuda else None
        for step_i in range(self.steps):
            is_last = step_i == (self.steps - 1)
            ctx = (
                torch.enable_grad()
                if (is_last and self.training and torch.is_grad_enabled())
                else torch.no_grad()
            )
            with ctx:
                c_mass_in = current_mass.detach() if is_last else current_mass
                s_mass_in = seed_mass if is_last else seed_mass.detach()
                destinations, probabilities = self.transitions_at_sources(
                    queries, states, indices, valid, allowed, current_ids, neighbor_key_rows
                )
                candidate_ids = torch.cat((destinations.flatten(-2), seed_ids), dim=-1)
                edge_mass = (1.0 - self.restart) * c_mass_in.unsqueeze(-1) * probabilities
                candidate_mass = torch.cat((edge_mass.flatten(-2), self.restart * s_mass_in), dim=-1)
                width = min(self.beam_size, n_tokens, candidate_ids.shape[-1])
                if (
                    ext is not None
                    and n_tokens <= 2048
                    and width <= 16
                    and candidate_ids.shape[-1] <= 192
                ):
                    current_ids, current_mass = _CoalesceTopKAutograd.apply(
                        candidate_ids, candidate_mass, n_tokens, width
                    )
                else:
                    if n_tokens <= 2 * candidate_ids.shape[-1]:
                        merged_mass = candidate_mass.new_zeros(*candidate_mass.shape[:-1], n_tokens)
                        merged_mass = merged_mass.scatter_add(-1, candidate_ids, candidate_mass)
                        merged_ids = torch.arange(n_tokens, device=queries.device).expand_as(
                            merged_mass
                        )
                    else:
                        merged_ids, merged_mass = coalesce_candidates(candidate_ids, candidate_mass)
                    width = min(self.beam_size, n_tokens, merged_mass.shape[-1])
                    if queries.is_cuda:
                        current_mass, chosen = torch.topk(
                            merged_mass, k=width, dim=-1, largest=True, sorted=True
                        )
                        current_ids = merged_ids.gather(-1, chosen)
                    else:
                        chosen = merged_mass.argsort(dim=-1, descending=True, stable=True)[
                            ..., :width
                        ]
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
        ext = _get_mcl4_cuda_ext() if ids.is_cuda else None
        if ext is not None and ids.shape[-1] <= 32:
            sup = _supported(allowed, ids)
            return ext.select_subset_small_cuda(
                ids.contiguous(),
                mass.float().contiguous(),
                sup.contiguous(),
                positions.long().contiguous(),
                int(self.local_tokens),
                int(self.attend_tokens),
            )
        query = positions.view(1, 1, -1, 1)
        zero = ids == 0
        unique = ~zero | (zero.long().cumsum(-1) == 1)
        supported = _supported(allowed, ids)
        anchors = unique & supported & (ids <= query) & (ids > query - self.local_tokens)
        routed = (mass > 0) & supported & ~anchors
        k_sel = min(self.attend_tokens, ids.shape[-1])
        if ids.is_cuda:
            ranked = torch.topk(
                mass.masked_fill(~routed, -torch.inf), k=k_sel, dim=-1, largest=True, sorted=True
            ).indices
        else:
            ranked = mass.masked_fill(~routed, -torch.inf).argsort(
                dim=-1, descending=True, stable=True
            )[..., :k_sel]
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
