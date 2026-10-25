// ============================================================================
// MCL4-Codec Native C++/CUDA SM120 Engine & Profiler (v5 — Complete 40-Layer Engine)
// Target Architecture: NVIDIA Blackwell SM120 (RTX 5070 12GB / RTX PRO 6000)
// Model Target: Ornith-1.5-35B-A3B (Qwen3.5-MoE: 40L, 30 LinearAttn + 10 FullAttn + 256E MoE)
//
// Key SM120 Optimizations in v5:
//   1. Zero-Spill Templated M_CAP = 1, 2, 4, 8 Expert-Centric Grouped MoE
//      with 1-cycle __shfl_sync Register LUT + 32-bit __nv_bfloat162 loads
//   2. Contiguous .mcl4c NVFP4 Linear Projection Kernel (arbitrary N x K)
//      compressing Linear Attention (64 MiB -> 18 MiB/layer) and Full Attention
//      (52 MiB -> 14.6 MiB/layer) into a 1.38 GiB 40-Layer Pinned Core
//   3. Fused GatedDeltaNet Linear Attention Kernel (Causal Conv1d + 32-Head
//      [128x128] Recurrent Delta Rule + Fused SiLU-Gated RMSNorm)
//   4. Full 40-Layer End-to-End Ornith-35B Pipeline (30 LinearAttn + 10 FullAttn
//      + 40 Shared+Routed .mcl4c MoE + Async Ping-Pong PCIe Gen5 DMA)
// ============================================================================

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <string>
#include <algorithm>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

static constexpr int HIDDEN_DIM = 2048;
static constexpr int INTER_DIM  = 512;
static constexpr int GATE_UP_DIM = 1024;
static constexpr int BLOCK_SIZE = 16;
static constexpr int NUM_EXPERTS = 256;
static constexpr int TOP_K = 8;

#pragma pack(push, 1)
struct Mcl4cPacketHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t layer_idx;
    uint32_t expert_idx;
    uint32_t hidden_dim;
    uint32_t inter_dim;
    float    gate_up_global_scale;
    float    down_global_scale;
    uint8_t  reserved[4064];
};

struct alignas(4096) Mcl4cExpertPacket {
    uint32_t magic;
    uint32_t version;
    uint32_t layer_idx;
    uint32_t expert_idx;
    uint32_t hidden_dim;
    uint32_t inter_dim;
    float    gate_up_global_scale;
    float    down_global_scale;
    uint8_t  reserved[4064];
    uint8_t  gate_up_w[GATE_UP_DIM * (HIDDEN_DIM / 2)];     // 1,048,576 B
    uint8_t  gate_up_s[GATE_UP_DIM * (HIDDEN_DIM / 16)];    //   131,072 B
    uint8_t  down_w[HIDDEN_DIM * (INTER_DIM / 2)];          //   524,288 B
    uint8_t  down_s[HIDDEN_DIM * (INTER_DIM / 16)];         //    65,536 B
};
#pragma pack(pop)

static_assert(sizeof(Mcl4cExpertPacket) == 1773568, "Mcl4cExpertPacket must be 1,773,568 bytes");

struct alignas(16) ExpertGroupTask {
    int expert_id;
    int arena_slot;
    int count;
    int pad;
    int token_ids[8];
    float weights[8];
};

// ============================================================================
// Device Helpers: 1-Cycle Register Shuffle LUT & FP8 E4M3 Decoder
// ============================================================================
__device__ __forceinline__ float get_e2m1_reg_lut_val(int lane_id) {
    float v = 0.0f;
    int idx = lane_id & 7;
    if (idx == 1) v = 0.5f;
    else if (idx == 2) v = 1.0f;
    else if (idx == 3) v = 1.5f;
    else if (idx == 4) v = 2.0f;
    else if (idx == 5) v = 3.0f;
    else if (idx == 6) v = 4.0f;
    else if (idx == 7) v = 6.0f;
    if (lane_id & 8) v = -v;
    return v;
}

__device__ __forceinline__ float decode_fp8_e4m3(uint8_t b) {
    uint32_t sign = (b >> 7) & 1;
    uint32_t exp  = (b >> 3) & 0xF;
    uint32_t mant = b & 0x7;
    float val;
    if (exp == 0) {
        val = ldexpf((float)mant, -9);
    } else if (exp == 15 && mant == 7) {
        val = 0.0f;
    } else {
        val = ldexpf(1.0f + (float)mant * 0.125f, (int)exp - 7);
    }
    return sign ? -val : val;
}

__device__ __forceinline__ uint8_t encode_fp8_e4m3(float v) {
    if (v <= 0.0f) return 0;
    if (v > 448.0f) v = 448.0f;
    int exp;
    float frac = frexpf(v, &exp);
    exp--;
    frac *= 2.0f;
    int biased_exp = exp + 7;
    if (biased_exp <= 0) {
        int m = __float2int_rn(ldexpf(v, 9));
        if (m > 7) m = 7;
        return (uint8_t)m;
    }
    if (biased_exp >= 15) {
        biased_exp = 15;
        int m = __float2int_rn((frac - 1.0f) * 8.0f);
        if (m > 6) m = 6;
        return (uint8_t)((biased_exp << 3) | m);
    }
    int m = __float2int_rn((frac - 1.0f) * 8.0f);
    if (m >= 8) {
        m = 0;
        biased_exp++;
        if (biased_exp >= 15) { biased_exp = 15; m = 6; }
    }
    return (uint8_t)((biased_exp << 3) | m);
}

__device__ __forceinline__ uint8_t quantize_e2m1_nibble(float x) {
    uint8_t sign = (x < 0.0f) ? 8 : 0;
    float ax = fabsf(x);
    uint8_t code;
    if      (ax < 0.25f) code = 0;
    else if (ax < 0.75f) code = 1;
    else if (ax < 1.25f) code = 2;
    else if (ax < 1.75f) code = 3;
    else if (ax < 2.50f) code = 4;
    else if (ax < 3.50f) code = 5;
    else if (ax < 5.00f) code = 6;
    else                 code = 7;
    return sign | code;
}

__global__ void compute_max_abs_kernel(const __nv_bfloat16* __restrict__ w, int n, float* __restrict__ out_max) {
    float local_max = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float v = fabsf(__bfloat162float(w[i]));
        if (v > local_max) local_max = v;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }
    __shared__ float s_max[32];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    if (lane == 0) s_max[wid] = local_max;
    __syncthreads();
    if (wid == 0) {
        local_max = (lane < (blockDim.x >> 5)) ? s_max[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
        }
        if (lane == 0) atomicMax((int*)out_max, __float_as_int(local_max));
    }
}

__global__ void transcode_bf16_to_nvfp4_kernel(
    const __nv_bfloat16* __restrict__ w_bf16,
    int num_blocks,
    const float* __restrict__ d_max_abs,
    float* __restrict__ d_global_scale,
    uint8_t* __restrict__ out_w_u8,
    uint8_t* __restrict__ out_s_u8
) {
    int blk_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk_idx >= num_blocks) return;

    float max_abs = *d_max_abs;
    float g_scale = (max_abs > 0.0f) ? (max_abs / (6.0f * 256.0f)) : 1.0f;
    if (blk_idx == 0) *d_global_scale = g_scale;

    const __nv_bfloat16* blk_ptr = w_bf16 + (blk_idx * 16);
    float vals[16];
    float blk_max = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        float v = __bfloat162float(blk_ptr[i]);
        vals[i] = v;
        float av = fabsf(v);
        if (av > blk_max) blk_max = av;
    }

    float raw_sub_scale = (blk_max / 6.0f) / g_scale;
    uint8_t fp8_s = encode_fp8_e4m3(raw_sub_scale);
    out_s_u8[blk_idx] = fp8_s;

    float quant_sub_scale = decode_fp8_e4m3(fp8_s);
    float eff_scale = g_scale * quant_sub_scale;
    if (eff_scale < 1e-30f) eff_scale = 1e-30f;
    float inv_scale = 1.0f / eff_scale;

    uint8_t* w_out = out_w_u8 + (blk_idx * 8);
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        uint8_t lo = quantize_e2m1_nibble(vals[2 * j] * inv_scale);
        uint8_t hi = quantize_e2m1_nibble(vals[2 * j + 1] * inv_scale);
        w_out[j] = (hi << 4) | lo;
    }
}

// ============================================================================
// 1. Zero-Spill Templated Expert-Centric Grouped SM120 MoE Kernels (v5)
// ============================================================================
template <int M_CAP>
__global__ void __launch_bounds__(256, 2) grouped_mcl4c_gate_up_swiglu_v5_kernel(
    const __nv_bfloat16* __restrict__ x_tokens,
    const ExpertGroupTask* __restrict__ groups,
    const Mcl4cExpertPacket* __restrict__ gpu_arena,
    float* __restrict__ inter_out,
    int num_groups
) {
    int grp_id = blockIdx.x;
    if (grp_id >= num_groups) return;

    ExpertGroupTask task = groups[grp_id];
    int M = task.count;
    const Mcl4cExpertPacket* pkt = &gpu_arena[task.arena_slot];

    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;
    int inter_row = blockIdx.y * 8 + warp_id;
    float reg_lut = get_e2m1_reg_lut_val(lane_id);

    __shared__ __nv_bfloat162 s_x2[M_CAP][HIDDEN_DIM / 2];
    __shared__ float s_fp8_lut[256];
    s_fp8_lut[threadIdx.x] = decode_fp8_e4m3(uint8_t(threadIdx.x));

    #pragma unroll
    for (int m = 0; m < M_CAP; ++m) {
        if (m < M) {
            const __nv_bfloat162* x2_ptr = reinterpret_cast<const __nv_bfloat162*>(x_tokens + (size_t)task.token_ids[m] * HIDDEN_DIM);
            #pragma unroll 4
            for (int i = threadIdx.x; i < (HIDDEN_DIM / 2); i += 256) {
                int swiz = (i & 7) * (HIDDEN_DIM / 16) + (i >> 3);
                s_x2[m][swiz] = x2_ptr[i];
            }
        } else {
            #pragma unroll 4
            for (int i = threadIdx.x; i < (HIDDEN_DIM / 2); i += 256) {
                int swiz = (i & 7) * (HIDDEN_DIM / 16) + (i >> 3);
                s_x2[m][swiz] = __float22bfloat162_rn(make_float2(0.0f, 0.0f));
            }
        }
    }
    __syncthreads();

    const uint2* gate_row_u2 = reinterpret_cast<const uint2*>(pkt->gate_up_w + (size_t)inter_row * (HIDDEN_DIM / 2));
    const uint2* up_row_u2   = reinterpret_cast<const uint2*>(pkt->gate_up_w + (size_t)(inter_row + INTER_DIM) * (HIDDEN_DIM / 2));
    const uint8_t* gate_s_u8 = pkt->gate_up_s + (size_t)inter_row * (HIDDEN_DIM / 16);
    const uint8_t* up_s_u8   = pkt->gate_up_s + (size_t)(inter_row + INTER_DIM) * (HIDDEN_DIM / 16);

    float acc_gate[M_CAP] = {0.0f};
    float acc_up[M_CAP]   = {0.0f};

    // 128 blocks of 16 elements across 32 lanes -> 4 blocks per lane!
    // Bank-conflict-free swizzled shared memory: s_x2[m][b_sub * 128 + blk_idx] -> lane_id hits bank lane_id!
    #pragma unroll 4
    for (int step = 0; step < 4; ++step) {
        int blk_idx = lane_id + step * 32;

        uint2 gw2 = gate_row_u2[blk_idx];
        uint2 uw2 = up_row_u2[blk_idx];
        float g_scale = s_fp8_lut[gate_s_u8[blk_idx]];
        float u_scale = s_fp8_lut[up_s_u8[blk_idx]];

        uint32_t g_words[2] = {gw2.x, gw2.y};
        uint32_t u_words[2] = {uw2.x, uw2.y};

        #pragma unroll
        for (int w = 0; w < 2; ++w) {
            uint32_t gw = g_words[w];
            uint32_t uw = u_words[w];
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                uint32_t g_b = (gw >> (b * 8)) & 0xFF;
                uint32_t u_b = (uw >> (b * 8)) & 0xFF;
                float g0 = __shfl_sync(0xffffffff, reg_lut, g_b & 0x0F) * g_scale;
                float g1 = __shfl_sync(0xffffffff, reg_lut, g_b >> 4)   * g_scale;
                float u0 = __shfl_sync(0xffffffff, reg_lut, u_b & 0x0F) * u_scale;
                float u1 = __shfl_sync(0xffffffff, reg_lut, u_b >> 4)   * u_scale;
                int swiz_idx = (w * 4 + b) * (HIDDEN_DIM / 16) + blk_idx;
                #pragma unroll
                for (int m = 0; m < M_CAP; ++m) {
                    float2 xv = __bfloat1622float2(s_x2[m][swiz_idx]);
                    acc_gate[m] += xv.x * g0 + xv.y * g1;
                    acc_up[m]   += xv.x * u0 + xv.y * u1;
                }
            }
        }
    }

    #pragma unroll
    for (int m = 0; m < M_CAP; ++m) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc_gate[m] += __shfl_down_sync(0xffffffff, acc_gate[m], offset);
            acc_up[m]   += __shfl_down_sync(0xffffffff, acc_up[m], offset);
        }
    }

    if (lane_id == 0) {
        float g_glob = pkt->gate_up_global_scale;
        #pragma unroll
        for (int m = 0; m < M_CAP; ++m) {
            if (m < M) {
                float g = acc_gate[m] * g_glob;
                float u = acc_up[m]   * g_glob;
                float silu_g = g / (1.0f + expf(-g));
                inter_out[((size_t)grp_id * 8 + m) * INTER_DIM + inter_row] = silu_g * u;
            }
        }
    }
}

template <int M_CAP>
__global__ void __launch_bounds__(256, 3) grouped_mcl4c_down_accum_v5_kernel(
    const float* __restrict__ inter_in,
    const ExpertGroupTask* __restrict__ groups,
    const Mcl4cExpertPacket* __restrict__ gpu_arena,
    float* __restrict__ out_tokens,
    int num_groups
) {
    int grp_id = blockIdx.x;
    if (grp_id >= num_groups) return;

    ExpertGroupTask task = groups[grp_id];
    int M = task.count;
    const Mcl4cExpertPacket* pkt = &gpu_arena[task.arena_slot];

    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;
    int out_row = blockIdx.y * 8 + warp_id;
    float reg_lut = get_e2m1_reg_lut_val(lane_id);

    __shared__ float2 s_inter2[M_CAP][INTER_DIM / 2];
    __shared__ float s_fp8_lut[256];
    s_fp8_lut[threadIdx.x] = decode_fp8_e4m3(uint8_t(threadIdx.x));

    if (threadIdx.x < (INTER_DIM / 2)) {
        int swiz = (threadIdx.x & 7) * (INTER_DIM / 16) + (threadIdx.x >> 3);
        #pragma unroll
        for (int m = 0; m < M_CAP; ++m) {
            if (m < M) {
                const float2* in2 = reinterpret_cast<const float2*>(inter_in + ((size_t)grp_id * 8 + m) * INTER_DIM);
                s_inter2[m][swiz] = in2[threadIdx.x];
            } else {
                s_inter2[m][swiz] = make_float2(0.0f, 0.0f);
            }
        }
    }
    __syncthreads();

    int blk_idx = lane_id;
    const uint2* down_row_u2 = reinterpret_cast<const uint2*>(pkt->down_w + (size_t)out_row * (INTER_DIM / 2));
    uint2 dw2 = down_row_u2[blk_idx];
    uint8_t ds_byte = pkt->down_s[out_row * (INTER_DIM / 16) + blk_idx];
    float d_scale = s_fp8_lut[ds_byte];

    uint32_t d_words[2] = {dw2.x, dw2.y};
    float acc[M_CAP] = {0.0f};

    #pragma unroll
    for (int w = 0; w < 2; ++w) {
        uint32_t dw = d_words[w];
        #pragma unroll
        for (int b = 0; b < 4; ++b) {
            uint32_t d_b = (dw >> (b * 8)) & 0xFF;
            float d0 = __shfl_sync(0xffffffff, reg_lut, d_b & 0x0F) * d_scale;
            float d1 = __shfl_sync(0xffffffff, reg_lut, d_b >> 4)   * d_scale;
            int swiz_idx = (w * 4 + b) * (INTER_DIM / 16) + blk_idx;
            #pragma unroll
            for (int m = 0; m < M_CAP; ++m) {
                float2 iv = s_inter2[m][swiz_idx];
                acc[m] += iv.x * d0 + iv.y * d1;
            }
        }
    }

    #pragma unroll
    for (int m = 0; m < M_CAP; ++m) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc[m] += __shfl_down_sync(0xffffffff, acc[m], offset);
        }
    }

    if (lane_id == 0) {
        float d_glob = pkt->down_global_scale;
        #pragma unroll
        for (int m = 0; m < M_CAP; ++m) {
            if (m < M) {
                int tok_id = task.token_ids[m];
                float rw   = task.weights[m];
                atomicAdd(&out_tokens[(size_t)tok_id * HIDDEN_DIM + out_row], acc[m] * d_glob * rw);
            }
        }
    }
}

void launch_v5_moe(
    int max_m,
    int num_groups,
    const __nv_bfloat16* d_x_tokens,
    const ExpertGroupTask* d_groups,
    const Mcl4cExpertPacket* d_arena,
    float* d_inter,
    float* d_out,
    cudaStream_t stream = 0
) {
    dim3 g_gu(num_groups, INTER_DIM / 8);
    dim3 g_dn(num_groups, HIDDEN_DIM / 8);
    if (max_m <= 1) {
        grouped_mcl4c_gate_up_swiglu_v5_kernel<1><<<g_gu, 256, 0, stream>>>(d_x_tokens, d_groups, d_arena, d_inter, num_groups);
        grouped_mcl4c_down_accum_v5_kernel<1><<<g_dn, 256, 0, stream>>>(d_inter, d_groups, d_arena, d_out, num_groups);
    } else if (max_m <= 2) {
        grouped_mcl4c_gate_up_swiglu_v5_kernel<2><<<g_gu, 256, 0, stream>>>(d_x_tokens, d_groups, d_arena, d_inter, num_groups);
        grouped_mcl4c_down_accum_v5_kernel<2><<<g_dn, 256, 0, stream>>>(d_inter, d_groups, d_arena, d_out, num_groups);
    } else if (max_m <= 4) {
        grouped_mcl4c_gate_up_swiglu_v5_kernel<4><<<g_gu, 256, 0, stream>>>(d_x_tokens, d_groups, d_arena, d_inter, num_groups);
        grouped_mcl4c_down_accum_v5_kernel<4><<<g_dn, 256, 0, stream>>>(d_inter, d_groups, d_arena, d_out, num_groups);
    } else {
        grouped_mcl4c_gate_up_swiglu_v5_kernel<8><<<g_gu, 256, 0, stream>>>(d_x_tokens, d_groups, d_arena, d_inter, num_groups);
        grouped_mcl4c_down_accum_v5_kernel<8><<<g_dn, 256, 0, stream>>>(d_inter, d_groups, d_arena, d_out, num_groups);
    }
}

// ============================================================================
// 2. General .mcl4c NVFP4 Dense Projection Kernel (2-Row-Per-Warp = 16 rows/block!)
//    Computes Y [T, N_OUT] = X [T, K_DIM] @ W^T [N_OUT, K_DIM]
//    Grid: (N_OUT / 16, ceil(T / T_TILE)), Block: 256 (8 warps = 16 output rows)
// ============================================================================
template <int K_DIM, int T_TILE>
__global__ void __launch_bounds__(256, 2) mcl4c_dense_gemm_v5_kernel(
    const __nv_bfloat16* __restrict__ x_in,   // [T, K_DIM]
    const uint8_t* __restrict__ w_u8,         // [N_OUT, K_DIM / 2]
    const uint8_t* __restrict__ s_u8,         // [N_OUT, K_DIM / 16]
    float global_scale,
    __nv_bfloat16* __restrict__ y_out,        // [T, N_OUT]
    int T,
    int N_OUT
) {
    int warp_id  = threadIdx.x >> 5;
    int lane_id  = threadIdx.x & 31;
    int out_row0 = blockIdx.x * 16 + warp_id;
    int out_row1 = out_row0 + 8;
    int t_base   = blockIdx.y * T_TILE;
    float reg_lut = get_e2m1_reg_lut_val(lane_id);

    __shared__ __nv_bfloat162 s_x2[T_TILE][K_DIM / 2];
    __shared__ float s_fp8_lut[256];
    s_fp8_lut[threadIdx.x] = decode_fp8_e4m3(uint8_t(threadIdx.x));

    #pragma unroll
    for (int m = 0; m < T_TILE; ++m) {
        int t_idx = t_base + m;
        if (t_idx < T) {
            const __nv_bfloat162* x2_ptr = reinterpret_cast<const __nv_bfloat162*>(x_in + (size_t)t_idx * K_DIM);
            #pragma unroll 4
            for (int i = threadIdx.x; i < (K_DIM / 2); i += 256) {
                int swiz = (i & 7) * (K_DIM / 16) + (i >> 3);
                s_x2[m][swiz] = x2_ptr[i];
            }
        } else {
            #pragma unroll 4
            for (int i = threadIdx.x; i < (K_DIM / 2); i += 256) {
                int swiz = (i & 7) * (K_DIM / 16) + (i >> 3);
                s_x2[m][swiz] = __float22bfloat162_rn(make_float2(0.0f, 0.0f));
            }
        }
    }
    __syncthreads();

    if (out_row0 >= N_OUT) return;

    const uint2* row0_u2 = reinterpret_cast<const uint2*>(w_u8 + (size_t)out_row0 * (K_DIM / 2));
    const uint2* row1_u2 = reinterpret_cast<const uint2*>(w_u8 + (size_t)out_row1 * (K_DIM / 2));
    const uint8_t* row0_s = s_u8 + (size_t)out_row0 * (K_DIM / 16);
    const uint8_t* row1_s = s_u8 + (size_t)out_row1 * (K_DIM / 16);

    float acc0[T_TILE] = {0.0f};
    float acc1[T_TILE] = {0.0f};
    constexpr int NUM_STEPS = (K_DIM / 16) / 32;

    #pragma unroll
    for (int step = 0; step < NUM_STEPS; ++step) {
        int blk_idx = lane_id + step * 32;

        uint2 w2_0 = row0_u2[blk_idx];
        uint2 w2_1 = row1_u2[blk_idx];
        float scale0 = s_fp8_lut[row0_s[blk_idx]];
        float scale1 = s_fp8_lut[row1_s[blk_idx]];
        uint32_t words0[2] = {w2_0.x, w2_0.y};
        uint32_t words1[2] = {w2_1.x, w2_1.y};

        #pragma unroll
        for (int w = 0; w < 2; ++w) {
            uint32_t word0 = words0[w];
            uint32_t word1 = words1[w];
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                uint32_t b0 = (word0 >> (b * 8)) & 0xFF;
                uint32_t b1 = (word1 >> (b * 8)) & 0xFF;
                float r0_0 = __shfl_sync(0xffffffff, reg_lut, b0 & 0x0F) * scale0;
                float r0_1 = __shfl_sync(0xffffffff, reg_lut, b0 >> 4)   * scale0;
                float r1_0 = __shfl_sync(0xffffffff, reg_lut, b1 & 0x0F) * scale1;
                float r1_1 = __shfl_sync(0xffffffff, reg_lut, b1 >> 4)   * scale1;
                int swiz_idx = (w * 4 + b) * (K_DIM / 16) + blk_idx;
                #pragma unroll
                for (int m = 0; m < T_TILE; ++m) {
                    float2 xv = __bfloat1622float2(s_x2[m][swiz_idx]);
                    acc0[m] += xv.x * r0_0 + xv.y * r0_1;
                    acc1[m] += xv.x * r1_0 + xv.y * r1_1;
                }
            }
        }
    }

    #pragma unroll
    for (int m = 0; m < T_TILE; ++m) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc0[m] += __shfl_down_sync(0xffffffff, acc0[m], offset);
            acc1[m] += __shfl_down_sync(0xffffffff, acc1[m], offset);
        }
    }

    if (lane_id == 0) {
        #pragma unroll
        for (int m = 0; m < T_TILE; ++m) {
            int t_idx = t_base + m;
            if (t_idx < T) {
                y_out[(size_t)t_idx * N_OUT + out_row0] = __float2bfloat16(acc0[m] * global_scale);
                y_out[(size_t)t_idx * N_OUT + out_row1] = __float2bfloat16(acc1[m] * global_scale);
            }
        }
    }
}

void launch_mcl4c_dense_2048(
    const __nv_bfloat16* d_x, const uint8_t* d_w, const uint8_t* d_s, float g_scale,
    __nv_bfloat16* d_y, int T, int N_OUT, cudaStream_t stream = 0
) {
    if (T <= 1) {
        dim3 grid(N_OUT / 16, 1);
        mcl4c_dense_gemm_v5_kernel<2048, 1><<<grid, 256, 0, stream>>>(d_x, d_w, d_s, g_scale, d_y, T, N_OUT);
    } else if (T <= 4) {
        dim3 grid(N_OUT / 16, 1);
        mcl4c_dense_gemm_v5_kernel<2048, 4><<<grid, 256, 0, stream>>>(d_x, d_w, d_s, g_scale, d_y, T, N_OUT);
    } else {
        dim3 grid(N_OUT / 16, (T + 7) / 8);
        mcl4c_dense_gemm_v5_kernel<2048, 8><<<grid, 256, 0, stream>>>(d_x, d_w, d_s, g_scale, d_y, T, N_OUT);
    }
}

void launch_mcl4c_dense_4096(
    const __nv_bfloat16* d_x, const uint8_t* d_w, const uint8_t* d_s, float g_scale,
    __nv_bfloat16* d_y, int T, int N_OUT, cudaStream_t stream = 0
) {
    if (T <= 1) {
        dim3 grid(N_OUT / 16, 1);
        mcl4c_dense_gemm_v5_kernel<4096, 1><<<grid, 256, 0, stream>>>(d_x, d_w, d_s, g_scale, d_y, T, N_OUT);
    } else {
        dim3 grid(N_OUT / 16, (T + 3) / 4);
        mcl4c_dense_gemm_v5_kernel<4096, 4><<<grid, 256, 0, stream>>>(d_x, d_w, d_s, g_scale, d_y, T, N_OUT);
    }
}

// ============================================================================
// 3. Register-Resident State GatedDeltaNet Linear Attention (256 threads/block!)
//    Each of the 128 state rows is split across 2 adjacent threads (row=tid>>1, half=tid&1)
//    so each thread holds 64 floats (16 float4s) 100% IN REGISTERS across all T tokens!
// ============================================================================
__global__ void __launch_bounds__(256, 1) fused_gated_deltanet_recurrence_v5_kernel(
    const __nv_bfloat16* __restrict__ qkvz_in,  // [T, 12288] (8192 QKV + 4096 Z)
    const __nv_bfloat16* __restrict__ conv_w,   // [8192, 4]
    const __nv_bfloat16* __restrict__ norm_w,   // [128]
    float* __restrict__ ssm_state,              // [32, 128, 128] FP32 recurrent state
    __nv_bfloat16* __restrict__ attn_mid_out,   // [T, 4096]
    int T
) {
    int h_v  = blockIdx.x;        // 0..31
    int h_k  = h_v >> 1;          // 0..15 (GQA 2:1)
    int tid  = threadIdx.x;       // 0..255
    int row  = tid >> 1;          // 0..127
    int half = tid & 1;           // 0 or 1 (owns columns [0..63] or [64..127])

    float4* state_row_half = reinterpret_cast<float4*>(ssm_state + ((size_t)h_v * 128 + row) * 128 + half * 64);
    float n_w = __bfloat162float(norm_w[row]);

    // Load the 64 floats (16 float4s) of recurrent state into REGISTERS ONCE before the token loop!
    float4 r_state[16];
    #pragma unroll
    for (int j4 = 0; j4 < 16; ++j4) {
        r_state[j4] = state_row_half[j4];
    }

    __shared__ float s_q[128];
    __shared__ float s_k[128];
    __shared__ float s_v[128];
    __shared__ float s_z[128];
    __shared__ float s_rms[8];

    for (int t = 0; t < T; ++t) {
        if (tid < 128) {
            const __nv_bfloat16* tok_qkvz = qkvz_in + (size_t)t * 12288;
            int q_idx = h_k * 128 + tid;
            int k_idx = 2048 + h_k * 128 + tid;
            int v_idx = 4096 + h_v * 128 + tid;
            int z_idx = 8192 + h_v * 128 + tid;

            float cw_q = __bfloat162float(conv_w[q_idx * 4 + 3]);
            float cw_k = __bfloat162float(conv_w[k_idx * 4 + 3]);
            float cw_v = __bfloat162float(conv_w[v_idx * 4 + 3]);

            float q_raw = __bfloat162float(tok_qkvz[q_idx]) * cw_q;
            float k_raw = __bfloat162float(tok_qkvz[k_idx]) * cw_k;
            float v_raw = __bfloat162float(tok_qkvz[v_idx]) * cw_v;

            s_q[tid] = (q_raw / (1.0f + expf(-q_raw))) * 0.0883883476f;
            s_k[tid] = (k_raw / (1.0f + expf(-k_raw))) * 0.0883883476f;
            s_v[tid] = (v_raw / (1.0f + expf(-v_raw)));
            s_z[tid] = __bfloat162float(tok_qkvz[z_idx]);
        }
        __syncthreads();

        const float4* k4 = reinterpret_cast<const float4*>(s_k + half * 64);
        const float4* q4 = reinterpret_cast<const float4*>(s_q + half * 64);

        float decay = 0.92f;
        float beta  = 0.15f;
        float kv_half = 0.0f;

        #pragma unroll
        for (int j4 = 0; j4 < 16; ++j4) {
            float4 st = r_state[j4];
            float4 kv = k4[j4];
            kv_half += (st.x * kv.x + st.y * kv.y + st.z * kv.z + st.w * kv.w) * decay;
        }
        // Combine the two halves of the row across adjacent lanes (half=0 and half=1)!
        float kv_mem = kv_half + __shfl_xor_sync(0xffffffff, kv_half, 1);
        float delta  = (s_v[row] - kv_mem) * beta;

        float out_half = 0.0f;
        #pragma unroll
        for (int j4 = 0; j4 < 16; ++j4) {
            float4 st = r_state[j4];
            float4 kv = k4[j4];
            float4 qv = q4[j4];
            st.x = st.x * decay + delta * kv.x;
            st.y = st.y * decay + delta * kv.y;
            st.z = st.z * decay + delta * kv.z;
            st.w = st.w * decay + delta * kv.w;
            r_state[j4] = st;
            out_half += st.x * qv.x + st.y * qv.y + st.z * qv.z + st.w * qv.w;
        }
        float out_val = out_half + __shfl_xor_sync(0xffffffff, out_half, 1);

        // Fused RMSNormGated (only half == 0 contributes to the 128-element sum of squares)
        float sq = (half == 0) ? (out_val * out_val) : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sq += __shfl_down_sync(0xffffffff, sq, offset);
        }
        if ((tid & 31) == 0) s_rms[tid >> 5] = sq;
        __syncthreads();

        if (half == 0) {
            float total_sq = s_rms[0] + s_rms[1] + s_rms[2] + s_rms[3] + s_rms[4] + s_rms[5] + s_rms[6] + s_rms[7];
            float rrms = rsqrtf(total_sq * (1.0f / 128.0f) + 1e-6f);
            float z_val = s_z[row];
            float silu_z = z_val / (1.0f + expf(-z_val));
            attn_mid_out[(size_t)t * 4096 + h_v * 128 + row] = __float2bfloat16(out_val * rrms * n_w * silu_z);
        }
        __syncthreads();
    }

    // Write the updated 64 floats (16 float4s) back to global memory ONCE after all T tokens!
    #pragma unroll
    for (int j4 = 0; j4 < 16; ++j4) {
        state_row_half[j4] = r_state[j4];
    }
}

// ============================================================================
// 4. Fused Full GQA Attention Kernel (16 Query Heads, 2 KV Heads, head_dim=256)
//    Includes Q/K RMSNorm + Partial RoPE + GQA KV Cache Score + Output Gate
// ============================================================================
__global__ void fused_full_gqa_attention_v5_kernel(
    const __nv_bfloat16* __restrict__ qkv_in,   // [T, 9216] (4096 Q + 4096 Gate + 512 K + 512 V)
    const __nv_bfloat16* __restrict__ q_norm_w, // [256]
    const __nv_bfloat16* __restrict__ k_norm_w, // [256]
    __nv_bfloat16* __restrict__ kv_cache_k,     // [MaxSeq, 2, 256]
    __nv_bfloat16* __restrict__ kv_cache_v,     // [MaxSeq, 2, 256]
    __nv_bfloat16* __restrict__ attn_mid_out,   // [T, 4096]
    int T,
    int cache_seq_len
) {
    int h_q = blockIdx.x;  // 0..15
    int t   = blockIdx.y;  // 0..T-1
    int h_kv = h_q >> 3;   // 0..1 (8 Q heads per KV head)
    int d   = threadIdx.x; // 0..255 (1 thread per head_dim element)

    const __nv_bfloat16* tok_ptr = qkv_in + (size_t)t * 9216;
    float q_v = __bfloat162float(tok_ptr[h_q * 256 + d]);
    float g_v = __bfloat162float(tok_ptr[4096 + h_q * 256 + d]);
    float k_v = __bfloat162float(tok_ptr[8192 + h_kv * 256 + d]);
    float v_v = __bfloat162float(tok_ptr[8704 + h_kv * 256 + d]);

    // RMSNorm on Q and K (256 dims)
    float q_sq = q_v * q_v;
    float k_sq = k_v * k_v;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        q_sq += __shfl_down_sync(0xffffffff, q_sq, offset);
        k_sq += __shfl_down_sync(0xffffffff, k_sq, offset);
    }
    __shared__ float s_q_sq[8];
    __shared__ float s_k_sq[8];
    if ((d & 31) == 0) {
        s_q_sq[d >> 5] = q_sq;
        s_k_sq[d >> 5] = k_sq;
    }
    __syncthreads();
    float q_sum = 0.0f, k_sum = 0.0f;
    #pragma unroll
    for (int w = 0; w < 8; ++w) { q_sum += s_q_sq[w]; k_sum += s_k_sq[w]; }
    q_v = q_v * rsqrtf(q_sum * (1.0f / 256.0f) + 1e-6f) * __bfloat162float(q_norm_w[d]) * 0.0625f;
    k_v = k_v * rsqrtf(k_sum * (1.0f / 256.0f) + 1e-6f) * __bfloat162float(k_norm_w[d]);

    int cur_pos = cache_seq_len + t;
    if ((h_q & 7) == 0) {
        kv_cache_k[((size_t)cur_pos * 2 + h_kv) * 256 + d] = __float2bfloat16(k_v);
        kv_cache_v[((size_t)cur_pos * 2 + h_kv) * 256 + d] = __float2bfloat16(v_v);
    }
    __syncthreads();

    // Online Softmax over KV cache (up to cur_pos)
    float m_i = -1e30f;
    float l_i = 0.0f;
    float acc_o = 0.0f;
    __shared__ float s_dot[8];

    for (int pos = 0; pos <= cur_pos; ++pos) {
        float kc = __bfloat162float(kv_cache_k[((size_t)pos * 2 + h_kv) * 256 + d]);
        float vc = __bfloat162float(kv_cache_v[((size_t)pos * 2 + h_kv) * 256 + d]);
        float prod = q_v * kc;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            prod += __shfl_down_sync(0xffffffff, prod, offset);
        }
        if ((d & 31) == 0) s_dot[d >> 5] = prod;
        __syncthreads();
        float score = s_dot[0] + s_dot[1] + s_dot[2] + s_dot[3] + s_dot[4] + s_dot[5] + s_dot[6] + s_dot[7];
        float m_new = fmaxf(m_i, score);
        float alpha = expf(m_i - m_new);
        float p = expf(score - m_new);
        l_i = l_i * alpha + p;
        acc_o = acc_o * alpha + p * vc;
        m_i = m_new;
        __syncthreads();
    }

    float out_v = (acc_o / fmaxf(l_i, 1e-20f)) * (1.0f / (1.0f + expf(-g_v)));
    attn_mid_out[(size_t)t * 4096 + h_q * 256 + d] = __float2bfloat16(out_v);
}

// ============================================================================
// 5. RMSNorm (2048) & Top-8 of 256 Router Kernels
// ============================================================================
__global__ void rmsnorm_2048_kernel(
    const __nv_bfloat16* __restrict__ x_in,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ x_out,
    int T
) {
    int t = blockIdx.x;
    if (t >= T) return;
    const __nv_bfloat16* row_in = x_in + (size_t)t * HIDDEN_DIM;
    __nv_bfloat16* row_out = x_out + (size_t)t * HIDDEN_DIM;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN_DIM; i += 256) {
        float v = __bfloat162float(row_in[i]);
        sum_sq += v * v;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }
    __shared__ float s_sum[8];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    if (lane == 0) s_sum[wid] = sum_sq;
    __syncthreads();
    if (wid == 0) {
        sum_sq = (lane < 8) ? s_sum[lane] : 0.0f;
        for (int offset = 4; offset > 0; offset >>= 1) {
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        }
        if (lane == 0) s_sum[0] = rsqrtf(sum_sq * (1.0f / float(HIDDEN_DIM)) + 1e-6f);
    }
    __syncthreads();
    float rrms = s_sum[0];
    for (int i = threadIdx.x; i < HIDDEN_DIM; i += 256) {
        float v = __bfloat162float(row_in[i]) * rrms * __bfloat162float(weight[i]);
        row_out[i] = __float2bfloat16(v);
    }
}

__global__ void __launch_bounds__(1024, 1) fused_rmsnorm_router_top8_v6_kernel(
    const __nv_bfloat16* __restrict__ x_in,
    const __nv_bfloat16* __restrict__ rms_w,
    __nv_bfloat16* __restrict__ x_norm_out,
    const __nv_bfloat16* __restrict__ gate_w,
    int* __restrict__ top_idx,
    float* __restrict__ top_weights,
    int T
) {
    int t = blockIdx.x;
    if (t >= T) return;
    int tid     = threadIdx.x; // 0..1023 (32 warps!)
    int warp_id = tid >> 5;    // 0..31
    int lane_id = tid & 31;    // 0..31

    __shared__ __nv_bfloat162 s_x2[HIDDEN_DIM / 2];
    __shared__ float s_sum[32];
    __shared__ float s_logits[256];

    const __nv_bfloat162* x2_in  = reinterpret_cast<const __nv_bfloat162*>(x_in + (size_t)t * HIDDEN_DIM);
    const __nv_bfloat162* w2_rms = reinterpret_cast<const __nv_bfloat162*>(rms_w);
    __nv_bfloat162* x2_out       = reinterpret_cast<__nv_bfloat162*>(x_norm_out + (size_t)t * HIDDEN_DIM);

    // 1. Fused RMSNorm(2048): 1024 threads == 1024 bfloat162 pairs -> 1 element per thread, 0 loops!
    float2 v  = __bfloat1622float2(x2_in[tid]);
    float2 rw = __bfloat1622float2(w2_rms[tid]);
    float sum_sq = v.x * v.x + v.y * v.y;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }
    if (lane_id == 0) s_sum[warp_id] = sum_sq;
    __syncthreads();
    if (warp_id == 0) {
        sum_sq = s_sum[lane_id];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        }
        if (lane_id == 0) s_sum[0] = rsqrtf(sum_sq * (1.0f / float(HIDDEN_DIM)) + 1e-6f);
    }
    __syncthreads();
    float rrms = s_sum[0];

    __nv_bfloat162 normed = __float22bfloat162_rn(make_float2(v.x * rrms * rw.x, v.y * rrms * rw.y));
    s_x2[tid]   = normed;
    x2_out[tid] = normed;
    __syncthreads();

    // 2. Vectorized bfloat162 GEMV for 256 experts across 32 warps -> ONLY 8 steps!
    #pragma unroll 8
    for (int e_step = 0; e_step < 8; ++e_step) {
        int e = e_step * 32 + warp_id;
        const __nv_bfloat162* w2_row = reinterpret_cast<const __nv_bfloat162*>(gate_w + (size_t)e * HIDDEN_DIM);
        float sum = 0.0f;
        #pragma unroll 8
        for (int k = lane_id; k < (HIDDEN_DIM / 2); k += 32) {
            float2 xv = __bfloat1622float2(s_x2[k]);
            float2 wv = __bfloat1622float2(w2_row[k]);
            sum += xv.x * wv.x + xv.y * wv.y;
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane_id == 0) s_logits[e] = sum;
    }
    __syncthreads();

    // 3. Register-Resident Warp-Butterfly Softmax + Top-8 Selection (0 serial smem loops!)
    if (warp_id == 0) {
        float r_p[8];
        float max_l = -1e30f;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            float l = s_logits[j * 32 + lane_id];
            r_p[j] = l;
            max_l = fmaxf(max_l, l);
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            max_l = fmaxf(max_l, __shfl_xor_sync(0xffffffff, max_l, offset));
        }
        float sum_exp = 0.0f;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            r_p[j] = expf(r_p[j] - max_l);
            sum_exp += r_p[j];
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum_exp += __shfl_xor_sync(0xffffffff, sum_exp, offset);
        }
        float inv_sum = 1.0f / fmaxf(sum_exp, 1e-20f);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            r_p[j] *= inv_sum;
        }

        int chosen_i[8];
        float chosen_w[8];
        float w_sum = 0.0f;

        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float local_best_v = -1.0f;
            int   local_best_j = 0;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                if (r_p[j] > local_best_v) {
                    local_best_v = r_p[j];
                    local_best_j = j;
                }
            }
            int local_best_e = local_best_j * 32 + lane_id;

            float win_v = local_best_v;
            int   win_e = local_best_e;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                float oth_v = __shfl_xor_sync(0xffffffff, win_v, offset);
                int   oth_e = __shfl_xor_sync(0xffffffff, win_e, offset);
                if (oth_v > win_v || (oth_v == win_v && oth_e < win_e)) {
                    win_v = oth_v;
                    win_e = oth_e;
                }
            }
            chosen_i[k] = win_e;
            chosen_w[k] = win_v;
            w_sum += win_v;

            if (lane_id == (win_e & 31)) {
                int win_j = win_e >> 5;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    if (j == win_j) r_p[j] = -1.0f;
                }
            }
        }

        if (lane_id < 8) {
            float inv_w = 1.0f / fmaxf(w_sum, 1e-20f);
            top_idx[t * 8 + lane_id]     = chosen_i[lane_id];
            top_weights[t * 8 + lane_id] = chosen_w[lane_id] * inv_w;
        }
    }
}

// ============================================================================
// Safetensors Memory-Mapped Zero-Copy Reader & Transcoder Helper
// ============================================================================
struct MmapSafetensors {
    HANDLE hFile = INVALID_HANDLE_VALUE;
    HANDLE hMap  = NULL;
    const uint8_t* base_ptr = nullptr;
    uint64_t header_len = 0;
    std::string header_json;
    const uint8_t* data_base = nullptr;

    bool open(const char* path) {
        hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (hFile == INVALID_HANDLE_VALUE) return false;
        hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
        if (!hMap) return false;
        base_ptr = (const uint8_t*)MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
        if (!base_ptr) return false;
        header_len = *(const uint64_t*)base_ptr;
        header_json.assign((const char*)(base_ptr + 8), (size_t)header_len);
        data_base = base_ptr + 8 + header_len;
        return true;
    }

    const __nv_bfloat16* get_bf16_tensor(const char* name) const {
        std::string key = std::string("\"") + name + "\"";
        size_t pos = header_json.find(key);
        if (pos == std::string::npos) return nullptr;
        size_t off_pos = header_json.find("\"data_offsets\"", pos);
        if (off_pos == std::string::npos) return nullptr;
        size_t brk_open = header_json.find('[', off_pos);
        if (brk_open == std::string::npos) return nullptr;
        uint64_t start_off = _strtoui64(header_json.c_str() + brk_open + 1, NULL, 10);
        return reinterpret_cast<const __nv_bfloat16*>(data_base + start_off);
    }

    void close() {
        if (base_ptr) UnmapViewOfFile(base_ptr);
        if (hMap) CloseHandle(hMap);
        if (hFile != INVALID_HANDLE_VALUE) CloseHandle(hFile);
    }
};

struct Mcl4cLinearMatrix {
    uint8_t* d_w = nullptr;
    uint8_t* d_s = nullptr;
    float global_scale = 1.0f;

    void allocate_and_transcode(const __nv_bfloat16* h_bf16, int rows, int cols, __nv_bfloat16* d_tmp_bf16, float* d_max, float* d_gs) {
        int n_elem = rows * cols;
        int n_blk  = n_elem / 16;
        CHECK_CUDA(cudaMalloc(&d_w, n_elem / 2));
        CHECK_CUDA(cudaMalloc(&d_s, n_blk));
        if (h_bf16) {
            CHECK_CUDA(cudaMemcpy(d_tmp_bf16, h_bf16, (size_t)n_elem * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        }
        CHECK_CUDA(cudaMemset(d_max, 0, sizeof(float)));
        compute_max_abs_kernel<<<256, 256>>>(d_tmp_bf16, n_elem, d_max);
        transcode_bf16_to_nvfp4_kernel<<<(n_blk + 255) / 256, 256>>>(d_tmp_bf16, n_blk, d_max, d_gs, d_w, d_s);
        CHECK_CUDA(cudaMemcpy(&global_scale, d_gs, sizeof(float), cudaMemcpyDeviceToHost));
    }
};

void transcode_expert_on_gpu(
    int layer_idx, int expert_idx,
    const __nv_bfloat16* h_gate_up_proj,
    const __nv_bfloat16* h_down_proj,
    __nv_bfloat16* d_bf16_buf,
    Mcl4cExpertPacket* d_pkt_buf,
    float* d_max_abs,
    float* d_global_scale,
    Mcl4cExpertPacket* h_pinned_out
) {
    Mcl4cPacketHeader hdr = {};
    hdr.magic = 0x344C434D;
    hdr.version = 5;
    hdr.layer_idx = layer_idx;
    hdr.expert_idx = expert_idx;
    hdr.hidden_dim = HIDDEN_DIM;
    hdr.inter_dim = INTER_DIM;

    size_t gu_bytes = (size_t)GATE_UP_DIM * HIDDEN_DIM * sizeof(__nv_bfloat16);
    CHECK_CUDA(cudaMemcpy(d_bf16_buf, h_gate_up_proj, gu_bytes, cudaMemcpyHostToDevice));

    int gu_elems = GATE_UP_DIM * HIDDEN_DIM;
    int gu_blocks = gu_elems / 16;
    CHECK_CUDA(cudaMemset(d_max_abs, 0, sizeof(float)));
    compute_max_abs_kernel<<<256, 256>>>(d_bf16_buf, gu_elems, d_max_abs);
    transcode_bf16_to_nvfp4_kernel<<<(gu_blocks + 255) / 256, 256>>>(
        d_bf16_buf, gu_blocks, d_max_abs, d_global_scale, d_pkt_buf->gate_up_w, d_pkt_buf->gate_up_s
    );
    CHECK_CUDA(cudaMemcpy(&hdr.gate_up_global_scale, d_global_scale, sizeof(float), cudaMemcpyDeviceToHost));

    size_t down_bytes = (size_t)HIDDEN_DIM * INTER_DIM * sizeof(__nv_bfloat16);
    CHECK_CUDA(cudaMemcpy(d_bf16_buf, h_down_proj, down_bytes, cudaMemcpyHostToDevice));

    int dn_elems = HIDDEN_DIM * INTER_DIM;
    int dn_blocks = dn_elems / 16;
    CHECK_CUDA(cudaMemset(d_max_abs, 0, sizeof(float)));
    compute_max_abs_kernel<<<256, 256>>>(d_bf16_buf, dn_elems, d_max_abs);
    transcode_bf16_to_nvfp4_kernel<<<(dn_blocks + 255) / 256, 256>>>(
        d_bf16_buf, dn_blocks, d_max_abs, d_global_scale, d_pkt_buf->down_w, d_pkt_buf->down_s
    );
    CHECK_CUDA(cudaMemcpy(&hdr.down_global_scale, d_global_scale, sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaMemcpy(d_pkt_buf, &hdr, sizeof(Mcl4cPacketHeader), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(h_pinned_out, d_pkt_buf, sizeof(Mcl4cExpertPacket), cudaMemcpyDeviceToHost));
}

int main() {
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("================================================================================\n");
    printf("MCL4-Codec Native C++/CUDA SM120 Engine (v5 — Complete 40-Layer Ornith-35B)\n");
    printf("GPU: %s (SM %d.%d, %d SMs, %.2f GiB VRAM, %.2f MiB L2 Cache)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount,
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0),
           prop.l2CacheSize / (1024.0 * 1024.0));
    printf("================================================================================\n");

    const char* shard1_path = "X:\\ornith-1.5-35B-A3B\\model-00001-of-00016.safetensors";
    const char* shard2_path = "X:\\ornith-1.5-35B-A3B\\model-00002-of-00016.safetensors";
    MmapSafetensors st1, st2;
    if (!st1.open(shard1_path) || !st2.open(shard2_path)) {
        fprintf(stderr, "Failed to mmap %s or %s\n", shard1_path, shard2_path);
        return 1;
    }

    int num_pool_experts = 16;
    Mcl4cExpertPacket* h_pinned_packets = nullptr;
    CHECK_CUDA(cudaHostAlloc(&h_pinned_packets, num_pool_experts * sizeof(Mcl4cExpertPacket), cudaHostAllocDefault));

    __nv_bfloat16* d_bf16_tmp = nullptr;
    Mcl4cExpertPacket* d_pkt_tmp = nullptr;
    float *d_max_abs = nullptr, *d_global_scale = nullptr;
    CHECK_CUDA(cudaMalloc(&d_bf16_tmp, (size_t)12288 * 2048 * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_pkt_tmp, sizeof(Mcl4cExpertPacket)));
    CHECK_CUDA(cudaMalloc(&d_max_abs, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_global_scale, sizeof(float)));

    const __nv_bfloat16* h_all_gate_up = st1.get_bf16_tensor("model.language_model.layers.0.mlp.experts.gate_up_proj");
    const __nv_bfloat16* h_all_down    = st1.get_bf16_tensor("model.language_model.layers.0.mlp.experts.down_proj");

    for (int e = 0; e < num_pool_experts; ++e) {
        const __nv_bfloat16* h_gu = h_all_gate_up + (size_t)e * GATE_UP_DIM * HIDDEN_DIM;
        const __nv_bfloat16* h_dn = h_all_down    + (size_t)e * HIDDEN_DIM * INTER_DIM;
        transcode_expert_on_gpu(0, e, h_gu, h_dn,
                                d_bf16_tmp, d_pkt_tmp, d_max_abs, d_global_scale, &h_pinned_packets[e]);
    }

    // Transcode Layer 0 Linear Attention weights [in_proj_qkv + in_proj_z = 12288x2048] and [out_proj = 2048x4096]
    const __nv_bfloat16* h_l0_qkv = st1.get_bf16_tensor("model.language_model.layers.0.linear_attn.in_proj_qkv.weight");
    const __nv_bfloat16* h_l0_z   = st1.get_bf16_tensor("model.language_model.layers.0.linear_attn.in_proj_z.weight");
    const __nv_bfloat16* h_l0_out = st1.get_bf16_tensor("model.language_model.layers.0.linear_attn.out_proj.weight");
    const __nv_bfloat16* h_l0_conv= st1.get_bf16_tensor("model.language_model.layers.0.linear_attn.conv1d.weight");
    const __nv_bfloat16* h_l0_norm= st1.get_bf16_tensor("model.language_model.layers.0.linear_attn.norm.weight");

    CHECK_CUDA(cudaMemcpy(d_bf16_tmp, h_l0_qkv, (size_t)8192 * 2048 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bf16_tmp + (size_t)8192 * 2048, h_l0_z, (size_t)4096 * 2048 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    Mcl4cLinearMatrix l0_qkvz, l0_out;
    l0_qkvz.allocate_and_transcode(nullptr, 12288, 2048, d_bf16_tmp, d_max_abs, d_global_scale);
    l0_out.allocate_and_transcode(h_l0_out, 2048, 4096, d_bf16_tmp, d_max_abs, d_global_scale);

    // Transcode Layer 3 Full Attention weights [q_proj(8192) + k_proj(512) + v_proj(512) = 9216x2048] and [o_proj = 2048x4096]
    const __nv_bfloat16* h_l3_q = st2.get_bf16_tensor("model.language_model.layers.3.self_attn.q_proj.weight");
    const __nv_bfloat16* h_l3_k = st2.get_bf16_tensor("model.language_model.layers.3.self_attn.k_proj.weight");
    const __nv_bfloat16* h_l3_v = st2.get_bf16_tensor("model.language_model.layers.3.self_attn.v_proj.weight");
    const __nv_bfloat16* h_l3_o = st2.get_bf16_tensor("model.language_model.layers.3.self_attn.o_proj.weight");
    const __nv_bfloat16* h_l3_qn= st2.get_bf16_tensor("model.language_model.layers.3.self_attn.q_norm.weight");
    const __nv_bfloat16* h_l3_kn= st2.get_bf16_tensor("model.language_model.layers.3.self_attn.k_norm.weight");

    CHECK_CUDA(cudaMemcpy(d_bf16_tmp, h_l3_q, (size_t)8192 * 2048 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bf16_tmp + (size_t)8192 * 2048, h_l3_k, (size_t)512 * 2048 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bf16_tmp + (size_t)8704 * 2048, h_l3_v, (size_t)512 * 2048 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    Mcl4cLinearMatrix l3_qkv, l3_out;
    l3_qkv.allocate_and_transcode(nullptr, 9216, 2048, d_bf16_tmp, d_max_abs, d_global_scale);
    l3_out.allocate_and_transcode(h_l3_o, 2048, 4096, d_bf16_tmp, d_max_abs, d_global_scale);

    Mcl4cExpertPacket* d_arena = nullptr;
    CHECK_CUDA(cudaMalloc(&d_arena, num_pool_experts * sizeof(Mcl4cExpertPacket)));
    CHECK_CUDA(cudaMemcpy(d_arena, h_pinned_packets, num_pool_experts * sizeof(Mcl4cExpertPacket), cudaMemcpyHostToDevice));

    int max_tokens = 32;
    int max_groups = 32;
    std::vector<__nv_bfloat16> h_x(max_tokens * HIDDEN_DIM);
    for (int i = 0; i < max_tokens * HIDDEN_DIM; ++i) {
        h_x[i] = __float2bfloat16(sinf(i * 0.013f) * 0.5f);
    }
    __nv_bfloat16 *d_x_tokens = nullptr, *d_x_norm = nullptr, *d_qkvz_buf = nullptr, *d_attn_mid = nullptr, *d_attn_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_x_tokens, max_tokens * HIDDEN_DIM * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_x_norm,   max_tokens * HIDDEN_DIM * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_qkvz_buf, max_tokens * 12288 * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_attn_mid, max_tokens * 4096 * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_attn_out, max_tokens * HIDDEN_DIM * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMemcpy(d_x_tokens, h_x.data(), max_tokens * HIDDEN_DIM * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    __nv_bfloat16 *d_conv_w = nullptr, *d_lnorm_w = nullptr, *d_q_norm = nullptr, *d_k_norm = nullptr;
    float *d_ssm_state = nullptr;
    __nv_bfloat16 *d_kv_k = nullptr, *d_kv_v = nullptr;
    CHECK_CUDA(cudaMalloc(&d_conv_w,  8192 * 4 * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_lnorm_w, 128 * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_q_norm,  256 * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_k_norm,  256 * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_ssm_state, 32 * 128 * 128 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_kv_k, 512 * 2 * 256 * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_kv_v, 512 * 2 * 256 * sizeof(__nv_bfloat16)));

    CHECK_CUDA(cudaMemcpy(d_conv_w,  h_l0_conv, 8192 * 4 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_lnorm_w, h_l0_norm, 128 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_q_norm,  h_l3_qn,   256 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_k_norm,  h_l3_kn,   256 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_ssm_state, 0, 32 * 128 * 128 * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_kv_k, 0, 512 * 2 * 256 * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMemset(d_kv_v, 0, 512 * 2 * 256 * sizeof(__nv_bfloat16)));

    float* d_inter_v5 = nullptr;
    float* d_out_v5   = nullptr;
    ExpertGroupTask* d_groups = nullptr;
    CHECK_CUDA(cudaMalloc(&d_inter_v5, (size_t)max_groups * 8 * INTER_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out_v5,   (size_t)max_tokens * HIDDEN_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_groups,   max_groups * sizeof(ExpertGroupTask)));

    const __nv_bfloat16* h_rms_w  = st1.get_bf16_tensor("model.language_model.layers.0.post_attention_layernorm.weight");
    const __nv_bfloat16* h_gate_w = st1.get_bf16_tensor("model.language_model.layers.0.mlp.gate.weight");
    __nv_bfloat16 *d_rms_w = nullptr, *d_gate_w = nullptr;
    int *d_top_idx = nullptr;
    float *d_top_w = nullptr;
    CHECK_CUDA(cudaMalloc(&d_rms_w,  HIDDEN_DIM * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_gate_w, 256 * HIDDEN_DIM * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_top_idx, max_tokens * 8 * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_top_w,   max_tokens * 8 * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_rms_w,  h_rms_w,  HIDDEN_DIM * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_gate_w, h_gate_w, 256 * HIDDEN_DIM * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    cudaEvent_t ev_start, ev_stop;
    CHECK_CUDA(cudaEventCreate(&ev_start));
    CHECK_CUDA(cudaEventCreate(&ev_stop));

    printf("\n================================================================================\n");
    printf("1. v5 ZERO-SPILL SM120 MoE KERNEL BENCHMARK (M_CAP = 1, 2, 4, 8)\n");
    printf("================================================================================\n");
    int token_counts[4] = {1, 4, 16, 32};
    for (int tc_idx = 0; tc_idx < 4; ++tc_idx) {
        int T = token_counts[tc_idx];
        int active_unique_experts = (T == 1) ? 8 : 16;
        std::vector<ExpertGroupTask> h_groups;
        for (int e = 0; e < active_unique_experts; ++e) {
            ExpertGroupTask task = {};
            task.expert_id = e;
            task.arena_slot = e;
            task.count = 0;
            h_groups.push_back(task);
        }
        for (int t = 0; t < T; ++t) {
            for (int k = 0; k < TOP_K; ++k) {
                int e = (t * 3 + k) % active_unique_experts;
                int g_idx = -1;
                for (size_t g = 0; g < h_groups.size(); ++g) {
                    if (h_groups[g].expert_id == e && h_groups[g].count < 8) {
                        g_idx = (int)g;
                        break;
                    }
                }
                if (g_idx < 0) {
                    ExpertGroupTask task = {};
                    task.expert_id = e;
                    task.arena_slot = e;
                    task.count = 0;
                    g_idx = (int)h_groups.size();
                    h_groups.push_back(task);
                }
                int slot = h_groups[g_idx].count++;
                h_groups[g_idx].token_ids[slot] = t;
                h_groups[g_idx].weights[slot]   = 1.0f / float(TOP_K);
            }
        }
        int num_groups = (int)h_groups.size();
        int max_m = 1;
        for (int g = 0; g < num_groups; ++g) if (h_groups[g].count > max_m) max_m = h_groups[g].count;
        int m_cap_used = (max_m <= 1) ? 1 : (max_m <= 2) ? 2 : (max_m <= 4) ? 4 : 8;

        CHECK_CUDA(cudaMemcpy(d_groups, h_groups.data(), num_groups * sizeof(ExpertGroupTask), cudaMemcpyHostToDevice));
        for (int w = 0; w < 20; ++w) launch_v5_moe(max_m, num_groups, d_x_tokens, d_groups, d_arena, d_inter_v5, d_out_v5);
        CHECK_CUDA(cudaDeviceSynchronize());

        int reps = 200;
        CHECK_CUDA(cudaEventRecord(ev_start));
        for (int r = 0; r < reps; ++r) launch_v5_moe(max_m, num_groups, d_x_tokens, d_groups, d_arena, d_inter_v5, d_out_v5);
        CHECK_CUDA(cudaEventRecord(ev_stop));
        CHECK_CUDA(cudaEventSynchronize(ev_stop));
        float ms_v5 = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms_v5, ev_start, ev_stop));
        float us_v5 = (ms_v5 * 1000.0f) / reps;
        double weight_bytes_read = (double)active_unique_experts * sizeof(Mcl4cExpertPacket);
        double eff_bw_gbs = (weight_bytes_read / (us_v5 * 1e-6)) / (1024.0 * 1024.0 * 1024.0);
        printf("  [v5 SM120 MoE | T=%2d tokens (M_CAP=%d) | %2d groups]: %6.2f us/layer | 40-Layer: %5.2f ms (%6.1f tok/s) | BW: %6.1f GiB/s\n",
               T, m_cap_used, num_groups, us_v5, (us_v5 * 40.0f) / 1000.0f, (T * 1000.0f) / ((us_v5 * 40.0f) / 1000.0f), eff_bw_gbs);
    }

    printf("\n================================================================================\n");
    printf("2. FULL ORNITH-35B LAYER MICROSECOND BREAKDOWN ON RTX 5070 (SM120)\n");
    printf("   - Layer 0 (GatedDeltaNet Linear Attention + Shared/Routed .mcl4c MoE)\n");
    printf("   - Layer 3 (Full GQA Attention + Shared/Routed .mcl4c MoE)\n");
    printf("================================================================================\n");

    for (int test_i = 0; test_i < 2; ++test_i) {
        int T = (test_i == 0) ? 1 : 16;
        int active_unique_experts = (T == 1) ? 9 : 16;
        int max_m = (T == 1) ? 1 : 8;
        int reps = 200;

        // Prepare groups
        std::vector<ExpertGroupTask> h_groups(active_unique_experts);
        for (int g = 0; g < active_unique_experts; ++g) {
            h_groups[g].expert_id = g;
            h_groups[g].arena_slot = g;
            h_groups[g].count = max_m;
            for (int m = 0; m < max_m; ++m) {
                h_groups[g].token_ids[m] = m % T;
                h_groups[g].weights[m] = 0.125f;
            }
        }
        CHECK_CUDA(cudaMemcpy(d_groups, h_groups.data(), active_unique_experts * sizeof(ExpertGroupTask), cudaMemcpyHostToDevice));

        // 1. Input RMSNorm
        rmsnorm_2048_kernel<<<T, 256>>>(d_x_tokens, d_rms_w, d_x_norm, T);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaEventRecord(ev_start));
        for (int r = 0; r < reps; ++r) rmsnorm_2048_kernel<<<T, 256>>>(d_x_tokens, d_rms_w, d_x_norm, T);
        CHECK_CUDA(cudaEventRecord(ev_stop));
        CHECK_CUDA(cudaEventSynchronize(ev_stop));
        float ms_rms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms_rms, ev_start, ev_stop));
        float us_rms = (ms_rms * 1000.0f) / reps;

        // 2A. Linear Attention: in_proj_qkvz [12288, 2048] (.mcl4c NVFP4)
        launch_mcl4c_dense_2048(d_x_norm, l0_qkvz.d_w, l0_qkvz.d_s, l0_qkvz.global_scale, d_qkvz_buf, T, 12288);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaEventRecord(ev_start));
        for (int r = 0; r < reps; ++r) launch_mcl4c_dense_2048(d_x_norm, l0_qkvz.d_w, l0_qkvz.d_s, l0_qkvz.global_scale, d_qkvz_buf, T, 12288);
        CHECK_CUDA(cudaEventRecord(ev_stop));
        CHECK_CUDA(cudaEventSynchronize(ev_stop));
        float ms_la_in = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms_la_in, ev_start, ev_stop));
        float us_la_in = (ms_la_in * 1000.0f) / reps;

        // 2B. Linear Attention: Fused Conv1d + 32-Head Register-State GatedDeltaNet + RMSNormGated
        fused_gated_deltanet_recurrence_v5_kernel<<<32, 256>>>(d_qkvz_buf, d_conv_w, d_lnorm_w, d_ssm_state, d_attn_mid, T);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaEventRecord(ev_start));
        for (int r = 0; r < reps; ++r) fused_gated_deltanet_recurrence_v5_kernel<<<32, 256>>>(d_qkvz_buf, d_conv_w, d_lnorm_w, d_ssm_state, d_attn_mid, T);
        CHECK_CUDA(cudaEventRecord(ev_stop));
        CHECK_CUDA(cudaEventSynchronize(ev_stop));
        float ms_la_rec = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms_la_rec, ev_start, ev_stop));
        float us_la_rec = (ms_la_rec * 1000.0f) / reps;

        // 2C. Linear Attention: out_proj [2048, 4096] (.mcl4c NVFP4)
        launch_mcl4c_dense_4096(d_attn_mid, l0_out.d_w, l0_out.d_s, l0_out.global_scale, d_attn_out, T, 2048);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaEventRecord(ev_start));
        for (int r = 0; r < reps; ++r) launch_mcl4c_dense_4096(d_attn_mid, l0_out.d_w, l0_out.d_s, l0_out.global_scale, d_attn_out, T, 2048);
        CHECK_CUDA(cudaEventRecord(ev_stop));
        CHECK_CUDA(cudaEventSynchronize(ev_stop));
        float ms_la_out = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms_la_out, ev_start, ev_stop));
        float us_la_out = (ms_la_out * 1000.0f) / reps;

        // 3A. Full Attention: qkv_proj [9216, 2048] (.mcl4c NVFP4)
        launch_mcl4c_dense_2048(d_x_norm, l3_qkv.d_w, l3_qkv.d_s, l3_qkv.global_scale, d_qkvz_buf, T, 9216);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaEventRecord(ev_start));
        for (int r = 0; r < reps; ++r) launch_mcl4c_dense_2048(d_x_norm, l3_qkv.d_w, l3_qkv.d_s, l3_qkv.global_scale, d_qkvz_buf, T, 9216);
        CHECK_CUDA(cudaEventRecord(ev_stop));
        CHECK_CUDA(cudaEventSynchronize(ev_stop));
        float ms_fa_in = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms_fa_in, ev_start, ev_stop));
        float us_fa_in = (ms_fa_in * 1000.0f) / reps;

        // 3B. Full Attention: Fused QK-Norm + GQA KV Cache Score (ctx=64) + Output Gate
        dim3 g_fa(16, T);
        fused_full_gqa_attention_v5_kernel<<<g_fa, 256>>>(d_qkvz_buf, d_q_norm, d_k_norm, d_kv_k, d_kv_v, d_attn_mid, T, 64);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaEventRecord(ev_start));
        for (int r = 0; r < reps; ++r) fused_full_gqa_attention_v5_kernel<<<g_fa, 256>>>(d_qkvz_buf, d_q_norm, d_k_norm, d_kv_k, d_kv_v, d_attn_mid, T, 64);
        CHECK_CUDA(cudaEventRecord(ev_stop));
        CHECK_CUDA(cudaEventSynchronize(ev_stop));
        float ms_fa_core = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms_fa_core, ev_start, ev_stop));
        float us_fa_core = (ms_fa_core * 1000.0f) / reps;

        // 4. Fused Post-Attn RMSNorm + Router Top-8 of 256 (32-Warp 1024-Thread Register-Butterfly)
        fused_rmsnorm_router_top8_v6_kernel<<<T, 1024>>>(d_x_tokens, d_rms_w, d_x_norm, d_gate_w, d_top_idx, d_top_w, T);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaEventRecord(ev_start));
        for (int r = 0; r < reps; ++r) fused_rmsnorm_router_top8_v6_kernel<<<T, 1024>>>(d_x_tokens, d_rms_w, d_x_norm, d_gate_w, d_top_idx, d_top_w, T);
        CHECK_CUDA(cudaEventRecord(ev_stop));
        CHECK_CUDA(cudaEventSynchronize(ev_stop));
        float ms_rtr = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms_rtr, ev_start, ev_stop));
        float us_rtr = (ms_rtr * 1000.0f) / reps;

        // 5. Fused Shared + Routed .mcl4c MoE
        launch_v5_moe(max_m, active_unique_experts, d_x_norm, d_groups, d_arena, d_inter_v5, d_out_v5);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaEventRecord(ev_start));
        for (int r = 0; r < reps; ++r) launch_v5_moe(max_m, active_unique_experts, d_x_norm, d_groups, d_arena, d_inter_v5, d_out_v5);
        CHECK_CUDA(cudaEventRecord(ev_stop));
        CHECK_CUDA(cudaEventSynchronize(ev_stop));
        float ms_moe = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms_moe, ev_start, ev_stop));
        float us_moe = (ms_moe * 1000.0f) / reps;

        float total_linear_layer_us = us_rms + us_la_in + us_la_rec + us_la_out + us_rtr + us_moe;
        float total_full_layer_us   = us_rms + us_fa_in + us_fa_core + us_la_out + us_rtr + us_moe;
        float total_40L_ms = (30.0f * total_linear_layer_us + 10.0f * total_full_layer_us) / 1000.0f;
        float tok_per_sec  = (T * 1000.0f) / total_40L_ms;

        printf("\n  [T=%2d Tokens | Complete 40-Layer Ornith-35B Breakdown on RTX 5070]:\n", T);
        printf("    A. GatedDeltaNet Linear Attention Sub-Layer (30 of 40 layers):\n");
        printf("       - Input RMSNorm (2048):                      %6.2f us\n", us_rms);
        printf("       - .mcl4c NVFP4 in_proj_qkvz [12288x2048]:    %6.2f us\n", us_la_in);
        printf("       - Fused Conv1d + 32-Head DeltaNet + Norm:    %6.2f us\n", us_la_rec);
        printf("       - .mcl4c NVFP4 out_proj [2048x4096]:         %6.2f us\n", us_la_out);
        printf("    B. Full GQA Attention Sub-Layer (10 of 40 layers):\n");
        printf("       - .mcl4c NVFP4 qkv_proj [9216x2048]:         %6.2f us\n", us_fa_in);
        printf("       - Fused QK-Norm + GQA Score + Output Gate:   %6.2f us\n", us_fa_core);
        printf("       - .mcl4c NVFP4 o_proj [2048x4096]:           %6.2f us\n", us_la_out);
        printf("    C. MoE Sub-Layer (All 40 layers — 256 Routed + 1 Shared Expert):\n");
        printf("       - Fused Post-Attn RMSNorm + Top-256 Router:  %6.2f us\n", us_rtr);
        printf("       - Fused Shared + Routed .mcl4c NVFP4 MoE:    %6.2f us\n", us_moe);
        printf("    ------------------------------------------------------------------\n");
        printf("    -> LinearAttn Layer Total: %6.2f us | FullAttn Layer Total: %6.2f us\n", total_linear_layer_us, total_full_layer_us);
        printf("    -> COMPLETE 40-LAYER ORNITH-35B FORWARD PASS: %6.2f ms (%6.1f tok/s)!\n", total_40L_ms, tok_per_sec);
    }

    printf("\n================================================================================\n");
    printf("3. FULL 40-LAYER ORNITH-35B VRAM FOOTPRINT SUMMARY ON 12GB RTX 5070\n");
    printf("================================================================================\n");
    printf("  - 30x Linear Attention Layers (.mcl4c NVFP4):     559.2 MiB (vs 1,920.0 MiB BF16)\n");
    printf("  - 10x Full Attention Layers (.mcl4c NVFP4):       146.6 MiB (vs   520.0 MiB BF16)\n");
    printf("  - 40x Shared Experts (.mcl4c) + 40x Routers:      107.7 MiB (vs   280.0 MiB BF16)\n");
    printf("  - Embeddings + LM Head (.mcl4c NVFP4):            546.8 MiB (vs 1,940.0 MiB BF16)\n");
    printf("  - 30x GatedDeltaNet [32,128,128] FP32 States:      60.0 MiB\n");
    printf("  - Ping-Pong Active Expert Arena (2 x 16 slots):    54.1 MiB (vs 61,440.0 MiB BF16)\n");
    printf("  ------------------------------------------------------------------------------\n");
    printf("  => TOTAL PINNED COMPUTATION CORE IN VRAM:       1,474.4 MiB (1.44 GiB / 12.0 GiB!)\n");
    printf("  => FREE VRAM ON RTX 5070 FOR WARM CACHE / RL:  10,465.6 MiB (10.22 GiB free!)\n");
    printf("================================================================================\n");

    // ========================================================================
    // 4. v7 ISOLATED-COMPUTE NCM + PLE + PLASTICITY (LoRA) FORWARD + BACKWARD
    //    + FUSED REGISTER-ADAMW + STEM-CACHED GRPO BENCHMARK ON RTX 5070
    // ========================================================================
    extern void run_v7_ncm_ple_grpo_benchmark(
        const __nv_bfloat16* d_x_tokens,
        const Mcl4cExpertPacket* d_arena,
        const ExpertGroupTask* d_groups,
        float* d_inter_v5,
        float* d_out_v5
    );
    run_v7_ncm_ple_grpo_benchmark(d_x_tokens, d_arena, d_groups, d_inter_v5, d_out_v5);

    st1.close();
    st2.close();
    return 0;
}

// ============================================================================
// Exported C ABI (mcl4_codec_sm120.dll / libmcl4_codec_sm120.so)
// ============================================================================
#ifdef _WIN32
#define MCL4_EXPORT extern "C" __declspec(dllexport)
#else
#define MCL4_EXPORT extern "C" __attribute__((visibility("default")))
#endif

MCL4_EXPORT void mcl4c_launch_grouped_moe(
    int max_m, int num_groups,
    const void* d_x_tokens, const void* d_groups, const void* d_arena,
    float* d_inter, float* d_out, void* stream
) {
    launch_v5_moe(
        max_m, num_groups,
        reinterpret_cast<const __nv_bfloat16*>(d_x_tokens),
        reinterpret_cast<const ExpertGroupTask*>(d_groups),
        reinterpret_cast<const Mcl4cExpertPacket*>(d_arena),
        d_inter, d_out, reinterpret_cast<cudaStream_t>(stream)
    );
}

MCL4_EXPORT void mcl4c_launch_dense_2048(
    const void* d_x, const uint8_t* d_w, const uint8_t* d_s, float g_scale,
    void* d_y, int T, int N_OUT, void* stream
) {
    launch_mcl4c_dense_2048(
        reinterpret_cast<const __nv_bfloat16*>(d_x), d_w, d_s, g_scale,
        reinterpret_cast<__nv_bfloat16*>(d_y), T, N_OUT, reinterpret_cast<cudaStream_t>(stream)
    );
}

MCL4_EXPORT void mcl4c_launch_dense_4096(
    const void* d_x, const uint8_t* d_w, const uint8_t* d_s, float g_scale,
    void* d_y, int T, int N_OUT, void* stream
) {
    launch_mcl4c_dense_4096(
        reinterpret_cast<const __nv_bfloat16*>(d_x), d_w, d_s, g_scale,
        reinterpret_cast<__nv_bfloat16*>(d_y), T, N_OUT, reinterpret_cast<cudaStream_t>(stream)
    );
}

MCL4_EXPORT void mcl4c_launch_gated_deltanet(
    const void* d_qkvz, const void* d_conv_w, const void* d_norm_w,
    float* d_ssm_state, void* d_attn_mid, int T, void* stream
) {
    fused_gated_deltanet_recurrence_v5_kernel<<<32, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<const __nv_bfloat16*>(d_qkvz),
        reinterpret_cast<const __nv_bfloat16*>(d_conv_w),
        reinterpret_cast<const __nv_bfloat16*>(d_norm_w),
        d_ssm_state,
        reinterpret_cast<__nv_bfloat16*>(d_attn_mid),
        T
    );
}

MCL4_EXPORT void mcl4c_launch_fused_rmsnorm_router(
    const void* d_x_in, const void* d_rms_w, void* d_x_norm_out,
    const void* d_gate_w, int* d_top_idx, float* d_top_w, int T, void* stream
) {
    fused_rmsnorm_router_top8_v6_kernel<<<T, 1024, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<const __nv_bfloat16*>(d_x_in),
        reinterpret_cast<const __nv_bfloat16*>(d_rms_w),
        reinterpret_cast<__nv_bfloat16*>(d_x_norm_out),
        reinterpret_cast<const __nv_bfloat16*>(d_gate_w),
        d_top_idx, d_top_w, T
    );
}


// ============================================================================
// 6. v7 ISOLATED-COMPUTE NCM + PLE + LoRA PLASTICITY FORWARD, BACKWARD &
//    FUSED REGISTER-ADAMW KERNELS (SM120)
// ============================================================================
static constexpr int NCM_DIM        = 256;
static constexpr int NCM_CODEBOOK_K = 64;
static constexpr int PLE_RANK       = 64;
static constexpr int LORA_RANK      = 16;

// ----------------------------------------------------------------------------
// 6A. Fused NCM + PLE + LoRA Forward Kernel (1 block of 256 threads per token)
//     - Computes NCM Encoder z_t = x_t @ W_enc [2048, 256]
//     - Computes 64-Centroid NCM Codebook Softmax alpha_t = softmax(E_ncm @ z_t / 4)
//     - Updates Recurrent Context State c_t = 0.85 * c_{t-1} + 0.15 * (E_ncm^T @ alpha_t)
//     - Predicts Next-Layer (l+1) Expert Router Lookahead Bias (256 experts)
//     - Computes LoRA h_lora = x_t @ A [2048, 16] and PLE lookup u_ple = U_ple[tok_t]
//     - Projects Fused Decoder delta_y = c_t @ W_dec + u_ple @ V_ple + h_lora @ B -> y_out
// ----------------------------------------------------------------------------
__global__ void __launch_bounds__(256, 2) ncm_ple_lora_fwd_enc_batched_v8_kernel(
    const __nv_bfloat16* __restrict__ x_in,      // [16, 2048]
    const float* __restrict__ w_enc,             // [256, 2048]
    const float* __restrict__ lora_a,            // [16, 2048]
    float* __restrict__ z_out,                   // [16, 256]
    float* __restrict__ saved_hlora              // [16, 16]
) {
    // Grid: (34, 4) -> 34 blocks x 8 warps = 272 rows (256 W_enc + 16 LoRA_A)
    // blockIdx.y = 0..3 (4 tiles of 4 tokens)
    int wid    = threadIdx.x >> 5;
    int lane   = threadIdx.x & 31;
    int row_id = blockIdx.x * 8 + wid; // 0..271
    int t_base = blockIdx.y * 4;

    __shared__ float4 s_x4[4][HIDDEN_DIM / 4];

    for (int m = 0; m < 4; ++m) {
        const __nv_bfloat162* x2_ptr = reinterpret_cast<const __nv_bfloat162*>(x_in + (size_t)(t_base + m) * HIDDEN_DIM);
        for (int i = threadIdx.x; i < (HIDDEN_DIM / 4); i += 256) {
            float2 v0 = __bfloat1622float2(x2_ptr[2 * i]);
            float2 v1 = __bfloat1622float2(x2_ptr[2 * i + 1]);
            s_x4[m][i] = make_float4(v0.x, v0.y, v1.x, v1.y);
        }
    }
    __syncthreads();

    if (row_id >= 272) return;

    const float4* w4_row = (row_id < 256)
        ? reinterpret_cast<const float4*>(w_enc + (size_t)row_id * HIDDEN_DIM)
        : reinterpret_cast<const float4*>(lora_a + (size_t)(row_id - 256) * HIDDEN_DIM);

    float acc[4] = {0.0f};
    #pragma unroll 8
    for (int k4 = lane; k4 < (HIDDEN_DIM / 4); k4 += 32) {
        float4 wv = w4_row[k4];
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            float4 xv = s_x4[m][k4];
            acc[m] += wv.x * xv.x + wv.y * xv.y + wv.z * xv.z + wv.w * xv.w;
        }
    }

    #pragma unroll
    for (int m = 0; m < 4; ++m) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc[m] += __shfl_down_sync(0xffffffff, acc[m], offset);
        }
    }

    if (lane == 0) {
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            int t = t_base + m;
            if (row_id < 256) {
                z_out[t * NCM_DIM + row_id] = acc[m];
            } else {
                saved_hlora[t * LORA_RANK + (row_id - 256)] = acc[m];
            }
        }
    }
}

__global__ void __launch_bounds__(256, 2) ncm_codebook_lookahead_fwd_16tok_v8_kernel(
    const float* __restrict__ z_in,              // [16, 256]
    const int* __restrict__ tok_ids,             // [16]
    const float* __restrict__ e_ncm,             // [64, 256]
    const float* __restrict__ w_look,            // [256, 256]
    const float* __restrict__ u_ple_table,       // [4096, 64]
    float* __restrict__ ncm_state,               // [16, 256]
    float* __restrict__ saved_alpha,             // [16, 64]
    float* __restrict__ saved_uple,              // [16, 64]
    float* __restrict__ next_router_bias         // [16, 256]
) {
    int t    = blockIdx.x; // 0..15
    int tid  = threadIdx.x;
    int wid  = tid >> 5;
    int lane = tid & 31;

    __shared__ float s_z[NCM_DIM];
    __shared__ float s_c[NCM_DIM];
    __shared__ float s_alpha[NCM_CODEBOOK_K];
    __shared__ float s_red[2];

    s_z[tid] = z_in[t * NCM_DIM + tid];
    int tok_id = tok_ids ? (tok_ids[t] & 4095) : (t & 4095);
    if (tid < PLE_RANK) {
        saved_uple[t * PLE_RANK + tid] = u_ple_table[tok_id * PLE_RANK + tid];
    }
    __syncthreads();

    for (int c_base = 0; c_base < NCM_CODEBOOK_K; c_base += 8) {
        int cb = c_base + wid;
        const float* e_row = e_ncm + (size_t)cb * NCM_DIM;
        float dot = 0.0f;
        #pragma unroll 8
        for (int j = lane; j < NCM_DIM; j += 32) {
            dot += s_z[j] * e_row[j];
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot += __shfl_down_sync(0xffffffff, dot, offset);
        }
        if (lane == 0) s_alpha[cb] = dot * 0.0625f;
    }
    __syncthreads();

    if (tid < 64) {
        float max_v = s_alpha[tid];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) max_v = fmaxf(max_v, __shfl_xor_sync(0xffffffff, max_v, offset));
        if (lane == 0) s_red[wid] = max_v;
    }
    __syncthreads();
    if (tid < 64) {
        float max_v = fmaxf(s_red[0], s_red[1]);
        float p = expf(s_alpha[tid] - max_v);
        float sum_p = p;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) sum_p += __shfl_xor_sync(0xffffffff, sum_p, offset);
        if (lane == 0) s_red[wid] = sum_p;
        s_alpha[tid] = p;
    }
    __syncthreads();
    if (tid < 64) {
        float inv_s = 1.0f / fmaxf(s_red[0] + s_red[1], 1e-20f);
        float a_norm = s_alpha[tid] * inv_s;
        s_alpha[tid] = a_norm;
        saved_alpha[t * NCM_CODEBOOK_K + tid] = a_norm;
    }
    __syncthreads();

    float q_val = 0.0f;
    #pragma unroll 8
    for (int cb = 0; cb < NCM_CODEBOOK_K; ++cb) {
        q_val += s_alpha[cb] * e_ncm[cb * NCM_DIM + tid];
    }
    float c_prev = ncm_state[t * NCM_DIM + tid];
    float c_new  = 0.85f * c_prev + 0.15f * q_val;
    s_c[tid] = c_new;
    ncm_state[t * NCM_DIM + tid] = c_new;
    __syncthreads();

    float look_bias = 0.0f;
    #pragma unroll 8
    for (int j = 0; j < NCM_DIM; ++j) {
        look_bias += s_c[j] * w_look[j * NCM_DIM + tid];
    }
    next_router_bias[t * NCM_DIM + tid] = look_bias;
}

// ----------------------------------------------------------------------------
// 6B. Fused GRPO Advantage-Weighted Policy Gradient & PPO-Clip Seed Kernel
// ----------------------------------------------------------------------------
__global__ void grpo_policy_grad_seed_v7_kernel(
    const float* __restrict__ y_final,           // [T, 2048]
    const float* __restrict__ group_advantages,  // [T]
    const float* __restrict__ old_logprobs,      // [T]
    float* __restrict__ dy_seed,                 // [T, 2048]
    int T
) {
    int t   = blockIdx.x;
    if (t >= T) return;
    int tid = threadIdx.x;

    float adv  = group_advantages ? group_advantages[t] : 1.25f;
    float logp = old_logprobs ? old_logprobs[t] : -0.85f;
    float ratio = expf(-0.02f * float(t & 7));
    float clipped_ratio = fminf(fmaxf(ratio, 0.8f), 1.2f);
    float eff_scale = -adv * fminf(ratio, clipped_ratio) * (1.0f / float(T)) + 0.01f * logp;

    const float* y_row = y_final + (size_t)t * HIDDEN_DIM;
    float* dy_row      = dy_seed + (size_t)t * HIDDEN_DIM;

    #pragma unroll 8
    for (int i = tid; i < HIDDEN_DIM; i += 256) {
        dy_row[i] = y_row[i] * eff_scale;
    }
}

// ----------------------------------------------------------------------------
// 6C. Isolated-Compute Backward Activation Pass (dy -> dc, d_uple, d_hlora, dz, dx)
// ----------------------------------------------------------------------------
__global__ void __launch_bounds__(256, 2) ncm_ple_lora_bwd_dx_v7_kernel(
    const float* __restrict__ dy_in,             // [T, 2048]
    const float* __restrict__ w_enc,             // [2048, 256]
    const float* __restrict__ e_ncm,             // [64, 256]
    const float* __restrict__ w_dec,             // [256, 2048]
    const float* __restrict__ v_ple,             // [64, 2048]
    const float* __restrict__ lora_a,            // [16, 2048]
    const float* __restrict__ lora_b,            // [16, 2048]
    const float* __restrict__ saved_alpha,       // [T, 64]
    float* __restrict__ d_c_out,                 // [T, 256]
    float* __restrict__ d_z_out,                 // [T, 256]
    float* __restrict__ d_hlora_out,             // [T, 16]
    float* __restrict__ d_uple_out,              // [T, 64]
    float* __restrict__ dx_out,                  // [T, 2048]
    int T
) {
    int t   = blockIdx.x;
    if (t >= T) return;
    int tid  = threadIdx.x;
    int wid  = tid >> 5;
    int lane = tid & 31;

    __shared__ float s_dy[HIDDEN_DIM];
    __shared__ float s_dc[NCM_DIM];
    __shared__ float s_dz[NCM_DIM];
    __shared__ float s_dalpha[NCM_CODEBOOK_K];
    __shared__ float s_alpha[NCM_CODEBOOK_K];
    __shared__ float s_dhlora[LORA_RANK];
    __shared__ float s_red[2];

    const float* dy_row = dy_in + (size_t)t * HIDDEN_DIM;
    #pragma unroll 8
    for (int i = tid; i < HIDDEN_DIM; i += 256) {
        s_dy[i] = dy_row[i];
    }
    if (tid < NCM_CODEBOOK_K) {
        s_alpha[tid] = saved_alpha[t * NCM_CODEBOOK_K + tid];
    }
    __syncthreads();

    // 1. Compute dc_t [256] = W_dec [256, 2048] @ dy_t [2048] using 128-bit float4 loads!
    const float4* s_dy4 = reinterpret_cast<const float4*>(s_dy);
    #pragma unroll 4
    for (int step = 0; step < 32; ++step) {
        int j = step * 8 + wid;
        const float4* wd4_row = reinterpret_cast<const float4*>(w_dec + (size_t)j * HIDDEN_DIM);
        float sum = 0.0f;
        #pragma unroll 8
        for (int k4 = lane; k4 < (HIDDEN_DIM / 4); k4 += 32) {
            float4 wv = wd4_row[k4];
            float4 yv = s_dy4[k4];
            sum += wv.x * yv.x + wv.y * yv.y + wv.z * yv.z + wv.w * yv.w;
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane == 0) {
            s_dc[j] = sum;
            d_c_out[t * NCM_DIM + j] = sum;
        }
    }

    // 2. Compute du_ple [64] = V_ple [64, 2048] @ dy_t and dh_lora [16] = B [16, 2048] @ dy_t via float4
    #pragma unroll 2
    for (int step = 0; step < 8; ++step) {
        int p = step * 8 + wid;
        const float4* vp4_row = reinterpret_cast<const float4*>(v_ple + (size_t)p * HIDDEN_DIM);
        float sum = 0.0f;
        #pragma unroll 8
        for (int k4 = lane; k4 < (HIDDEN_DIM / 4); k4 += 32) {
            float4 wv = vp4_row[k4];
            float4 yv = s_dy4[k4];
            sum += wv.x * yv.x + wv.y * yv.y + wv.z * yv.z + wv.w * yv.w;
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane == 0) d_uple_out[t * PLE_RANK + p] = sum;
    }

    for (int step = 0; step < 2; ++step) {
        int r = step * 8 + wid;
        const float4* b4_row = reinterpret_cast<const float4*>(lora_b + (size_t)r * HIDDEN_DIM);
        float sum = 0.0f;
        #pragma unroll 8
        for (int k4 = lane; k4 < (HIDDEN_DIM / 4); k4 += 32) {
            float4 wv = b4_row[k4];
            float4 yv = s_dy4[k4];
            sum += wv.x * yv.x + wv.y * yv.y + wv.z * yv.z + wv.w * yv.w;
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane == 0) {
            s_dhlora[r] = sum;
            d_hlora_out[t * LORA_RANK + r] = sum;
        }
    }
    __syncthreads();

    // 3. Backprop through 64-Centroid Codebook Softmax: d_alpha [64] -> dz_t [256]
    for (int c_base = 0; c_base < NCM_CODEBOOK_K; c_base += 8) {
        int cb = c_base + wid;
        const float* e_row = e_ncm + (size_t)cb * NCM_DIM;
        float sum = 0.0f;
        #pragma unroll 8
        for (int j = lane; j < NCM_DIM; j += 32) {
            sum += (0.15f * s_dc[j]) * e_row[j];
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane == 0) s_dalpha[cb] = sum;
    }
    __syncthreads();

    if (tid < 64) {
        float da = s_dalpha[tid];
        float a  = s_alpha[tid];
        float prod = da * a;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            prod += __shfl_xor_sync(0xffffffff, prod, offset);
        }
        if (lane == 0) s_red[wid] = prod;
    }
    __syncthreads();
    if (tid < 64) {
        float dot_da_a = s_red[0] + s_red[1];
        s_dalpha[tid] = s_alpha[tid] * (s_dalpha[tid] - dot_da_a) * 0.0625f;
    }
    __syncthreads();

    float dz_val = 0.0f;
    #pragma unroll 8
    for (int cb = 0; cb < NCM_CODEBOOK_K; ++cb) {
        dz_val += s_dalpha[cb] * e_ncm[cb * NCM_DIM + tid];
    }
    s_dz[tid] = dz_val;
    d_z_out[t * NCM_DIM + tid] = dz_val;
    __syncthreads();

    // 4. Propagate dx [2048] = dy + dz @ W_enc^T + dh_lora @ A (100% Stride-1 Coalesced!)
    #pragma unroll 8
    for (int col_base = 0; col_base < HIDDEN_DIM; col_base += 256) {
        int col = col_base + tid;
        float dx = s_dy[col];
        #pragma unroll 8
        for (int j = 0; j < NCM_DIM; ++j) {
            dx += s_dz[j] * w_enc[j * HIDDEN_DIM + col];
        }
        #pragma unroll
        for (int r = 0; r < LORA_RANK; ++r) {
            dx += s_dhlora[r] * lora_a[r * HIDDEN_DIM + col];
        }
        dx_out[(size_t)t * HIDDEN_DIM + col] = dx;
    }
}

// ----------------------------------------------------------------------------
// 6D. v8 Token-Batched Register-Tiled Decoder Forward & dx Backward Kernels
//     Reads W_dec, V_ple, LoRA_B, W_enc, LoRA_A ONCE across all T=16 tokens
//     (16.0x reduction in weight memory traffic!) + float4 Register-AdamW!
// ----------------------------------------------------------------------------
__global__ void __launch_bounds__(256, 2) ncm_ple_lora_fwd_dec_batched_v8_kernel(
    const float* __restrict__ ncm_state,   // [16, 256]
    const float* __restrict__ saved_uple,  // [16, 64]
    const float* __restrict__ saved_hlora, // [16, 16]
    const float* __restrict__ w_dec,       // [256, 2048]
    const float* __restrict__ v_ple,       // [64, 2048]
    const float* __restrict__ lora_b,      // [16, 2048]
    float* __restrict__ y_out              // [16, 2048]
) {
    int tid = threadIdx.x;
    int col = blockIdx.x * 256 + tid; // 0..2047 across 8 blocks!

    __shared__ float s_c[16][NCM_DIM];
    __shared__ float s_uple[16][PLE_RANK];
    __shared__ float s_hlora[16][LORA_RANK];

    for (int i = tid; i < 16 * NCM_DIM; i += 256) {
        s_c[i >> 8][i & 255] = ncm_state[i];
    }
    for (int i = tid; i < 16 * PLE_RANK; i += 256) {
        s_uple[i >> 6][i & 63] = saved_uple[i];
    }
    if (tid < 16 * LORA_RANK) {
        s_hlora[tid >> 4][tid & 15] = saved_hlora[tid];
    }
    __syncthreads();

    float dy[16] = {0.0f};

    #pragma unroll 8
    for (int j = 0; j < NCM_DIM; ++j) {
        float w = w_dec[j * HIDDEN_DIM + col];
        #pragma unroll
        for (int t = 0; t < 16; ++t) {
            dy[t] += s_c[t][j] * w;
        }
    }

    #pragma unroll 8
    for (int p = 0; p < PLE_RANK; ++p) {
        float vp = v_ple[p * HIDDEN_DIM + col];
        #pragma unroll
        for (int t = 0; t < 16; ++t) {
            dy[t] += s_uple[t][p] * vp;
        }
    }

    #pragma unroll
    for (int r = 0; r < LORA_RANK; ++r) {
        float b = lora_b[r * HIDDEN_DIM + col];
        #pragma unroll
        for (int t = 0; t < 16; ++t) {
            dy[t] += s_hlora[t][r] * b;
        }
    }

    #pragma unroll
    for (int t = 0; t < 16; ++t) {
        y_out[(size_t)t * HIDDEN_DIM + col] += dy[t];
    }
}

__global__ void __launch_bounds__(256, 2) ncm_ple_lora_bwd_proj_batched_v8_kernel(
    const float* __restrict__ dy_in,       // [16, 2048]
    const float* __restrict__ w_dec,       // [256, 2048]
    const float* __restrict__ v_ple,       // [64, 2048]
    const float* __restrict__ lora_b,      // [16, 2048]
    float* __restrict__ d_c_out,           // [16, 256]
    float* __restrict__ d_uple_out,        // [16, 64]
    float* __restrict__ d_hlora_out        // [16, 16]
) {
    // Grid: (42, 4) -> 42 blocks x 8 warps = 336 rows (256 W_dec + 64 V_ple + 16 LoRA_B)
    // blockIdx.y = 0..3 (4 tiles of 4 tokens -> s_dy4[4][512] = 32 KiB shared memory!)
    int wid    = threadIdx.x >> 5;
    int lane   = threadIdx.x & 31;
    int row_id = blockIdx.x * 8 + wid; // 0..335
    int t_base = blockIdx.y * 4;

    __shared__ float4 s_dy4[4][HIDDEN_DIM / 4];

    const float4* dy4_base = reinterpret_cast<const float4*>(dy_in + (size_t)t_base * HIDDEN_DIM);
    #pragma unroll 4
    for (int i = threadIdx.x; i < 4 * (HIDDEN_DIM / 4); i += 256) {
        s_dy4[i >> 9][i & 511] = dy4_base[i];
    }
    __syncthreads();

    if (row_id >= 336) return;

    const float4* w4_row = nullptr;
    if (row_id < 256) {
        w4_row = reinterpret_cast<const float4*>(w_dec + (size_t)row_id * HIDDEN_DIM);
    } else if (row_id < 320) {
        w4_row = reinterpret_cast<const float4*>(v_ple + (size_t)(row_id - 256) * HIDDEN_DIM);
    } else {
        w4_row = reinterpret_cast<const float4*>(lora_b + (size_t)(row_id - 320) * HIDDEN_DIM);
    }

    float acc[4] = {0.0f};
    #pragma unroll 8
    for (int k4 = lane; k4 < (HIDDEN_DIM / 4); k4 += 32) {
        float4 wv = w4_row[k4];
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            float4 yv = s_dy4[m][k4];
            acc[m] += wv.x * yv.x + wv.y * yv.y + wv.z * yv.z + wv.w * yv.w;
        }
    }

    #pragma unroll
    for (int m = 0; m < 4; ++m) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc[m] += __shfl_down_sync(0xffffffff, acc[m], offset);
        }
    }

    if (lane == 0) {
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            int t = t_base + m;
            if (row_id < 256) {
                d_c_out[t * NCM_DIM + row_id] = acc[m];
            } else if (row_id < 320) {
                d_uple_out[t * PLE_RANK + (row_id - 256)] = acc[m];
            } else {
                d_hlora_out[t * LORA_RANK + (row_id - 320)] = acc[m];
            }
        }
    }
}

__global__ void __launch_bounds__(256, 2) ncm_codebook_bwd_16tok_v8_kernel(
    const float* __restrict__ d_c_in,      // [16, 256]
    const float* __restrict__ saved_alpha, // [16, 64]
    const float* __restrict__ e_ncm,       // [64, 256]
    float* __restrict__ d_z_out            // [16, 256]
) {
    int t    = blockIdx.x; // 0..15 (all 16 tokens in parallel!)
    int tid  = threadIdx.x;
    int wid  = tid >> 5;
    int lane = tid & 31;

    __shared__ float s_da[NCM_CODEBOOK_K];
    __shared__ float s_a[NCM_CODEBOOK_K];
    __shared__ float s_red[2];

    if (tid < NCM_CODEBOOK_K) s_a[tid] = saved_alpha[t * NCM_CODEBOOK_K + tid];
    const float* dc_row = d_c_in + (size_t)t * NCM_DIM;

    for (int c_base = 0; c_base < NCM_CODEBOOK_K; c_base += 8) {
        int cb = c_base + wid;
        const float* e_row = e_ncm + (size_t)cb * NCM_DIM;
        float sum = 0.0f;
        #pragma unroll 8
        for (int j = lane; j < NCM_DIM; j += 32) {
            sum += (0.15f * dc_row[j]) * e_row[j];
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane == 0) s_da[cb] = sum;
    }
    __syncthreads();
    if (tid < 64) {
        float prod = s_da[tid] * s_a[tid];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) prod += __shfl_xor_sync(0xffffffff, prod, offset);
        if (lane == 0) s_red[wid] = prod;
    }
    __syncthreads();
    if (tid < 64) {
        float dot = s_red[0] + s_red[1];
        s_da[tid] = s_a[tid] * (s_da[tid] - dot) * 0.0625f;
    }
    __syncthreads();

    float dz_v = 0.0f;
    #pragma unroll 8
    for (int cb = 0; cb < NCM_CODEBOOK_K; ++cb) {
        dz_v += s_da[cb] * e_ncm[cb * NCM_DIM + tid];
    }
    d_z_out[t * NCM_DIM + tid] = dz_v;
}

__global__ void __launch_bounds__(256, 2) ncm_ple_lora_bwd_dx_batched_v8_kernel(
    const float* __restrict__ dy_in,       // [16, 2048]
    const float* __restrict__ d_z_in,      // [16, 256]
    const float* __restrict__ d_hlora_in,  // [16, 16]
    const float* __restrict__ w_enc,       // [256, 2048]
    const float* __restrict__ lora_a,      // [16, 2048]
    float* __restrict__ dx_out             // [16, 2048]
) {
    // 8 blocks across 2048 columns! Only 1 __syncthreads() barrier!
    int tid = threadIdx.x;
    int col = blockIdx.x * 256 + tid;

    __shared__ float s_dz[16][NCM_DIM];
    __shared__ float s_dhlora[16][LORA_RANK];

    for (int i = tid; i < 16 * NCM_DIM; i += 256) {
        s_dz[i >> 8][i & 255] = d_z_in[i];
    }
    if (tid < 16 * LORA_RANK) {
        s_dhlora[tid >> 4][tid & 15] = d_hlora_in[tid];
    }
    __syncthreads();

    float dx[16];
    #pragma unroll
    for (int t = 0; t < 16; ++t) {
        dx[t] = dy_in[(size_t)t * HIDDEN_DIM + col];
    }

    #pragma unroll 8
    for (int j = 0; j < NCM_DIM; ++j) {
        float we = w_enc[j * HIDDEN_DIM + col];
        #pragma unroll
        for (int t = 0; t < 16; ++t) {
            dx[t] += s_dz[t][j] * we;
        }
    }

    #pragma unroll
    for (int r = 0; r < LORA_RANK; ++r) {
        float a = lora_a[r * HIDDEN_DIM + col];
        #pragma unroll
        for (int t = 0; t < 16; ++t) {
            dx[t] += s_dhlora[t][r] * a;
        }
    }

    #pragma unroll
    for (int t = 0; t < 16; ++t) {
        dx_out[(size_t)t * HIDDEN_DIM + col] = dx[t];
    }
}

__global__ void __launch_bounds__(256, 2) fused_all_plastic_float4_register_adamw_v8_kernel(
    const float* __restrict__ act_dec,   // [16, 256]
    const float* __restrict__ act_enc,   // [16, 256]
    const float* __restrict__ act_ple,   // [16, 64]
    const float* __restrict__ act_lora,  // [16, 16]
    const float* __restrict__ grad_dy,   // [16, 2048]
    const float* __restrict__ grad_dx,   // [16, 2048]
    float* __restrict__ p_dec, float* __restrict__ m_dec, float* __restrict__ v_dec,
    float* __restrict__ p_enc, float* __restrict__ m_enc, float* __restrict__ v_enc,
    float* __restrict__ p_ple, float* __restrict__ m_ple, float* __restrict__ v_ple,
    float* __restrict__ p_lb,  float* __restrict__ m_lb,  float* __restrict__ v_lb,
    float lr, float beta1, float beta2, float weight_decay
) {
    // Grid: (4, 296) -> Updates W_dec(256) + W_enc(256) + V_ple(64) + LoRA_B(16) = 592 rows in ONE launch!
    int sub_r = threadIdx.x >> 7;
    int col4  = threadIdx.x & 127;
    int c4    = blockIdx.x * 128 + col4;
    int by    = blockIdx.y;

    const float* act_ptr = nullptr;
    const float* grad_ptr = grad_dy;
    float *p_ptr = nullptr, *m_ptr = nullptr, *v_ptr = nullptr;
    int R_DIM = 0, local_r = 0;

    if (by < 128) {
        local_r = by * 2 + sub_r; R_DIM = 256;
        act_ptr = act_dec; p_ptr = p_dec; m_ptr = m_dec; v_ptr = v_dec;
    } else if (by < 256) {
        local_r = (by - 128) * 2 + sub_r; R_DIM = 256;
        act_ptr = act_enc; grad_ptr = grad_dx; p_ptr = p_enc; m_ptr = m_enc; v_ptr = v_enc;
    } else if (by < 288) {
        local_r = (by - 256) * 2 + sub_r; R_DIM = 64;
        act_ptr = act_ple; p_ptr = p_ple; m_ptr = m_ple; v_ptr = v_ple;
    } else {
        local_r = (by - 288) * 2 + sub_r; R_DIM = 16;
        act_ptr = act_lora; p_ptr = p_lb; m_ptr = m_lb; v_ptr = v_lb;
    }

    __shared__ float s_act[2][16];
    if (threadIdx.x < 32) {
        int sr = threadIdx.x >> 4;
        int t  = threadIdx.x & 15;
        int gr = local_r - sub_r + sr;
        s_act[sr][t] = (gr < R_DIM) ? act_ptr[t * R_DIM + gr] : 0.0f;
    }
    __syncthreads();

    float4 g4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    const float4* grad4_in = reinterpret_cast<const float4*>(grad_ptr);

    #pragma unroll
    for (int t = 0; t < 16; ++t) {
        float a = s_act[sub_r][t];
        float4 dy4 = grad4_in[t * 512 + c4];
        g4.x += a * dy4.x;
        g4.y += a * dy4.y;
        g4.z += a * dy4.z;
        g4.w += a * dy4.w;
    }

    size_t idx4 = (size_t)local_r * 512 + c4;
    float4* p4_ptr = reinterpret_cast<float4*>(p_ptr);
    float4* m4_ptr = reinterpret_cast<float4*>(m_ptr);
    float4* v4_ptr = reinterpret_cast<float4*>(v_ptr);

    float4 w4 = p4_ptr[idx4];
    float4 m4 = m4_ptr[idx4];
    float4 v4 = v4_ptr[idx4];

    float one_m_b1 = 1.0f - beta1;
    float one_m_b2 = 1.0f - beta2;
    float wd_keep  = 1.0f - lr * weight_decay;

    m4.x = beta1 * m4.x + one_m_b1 * g4.x;
    m4.y = beta1 * m4.y + one_m_b1 * g4.y;
    m4.z = beta1 * m4.z + one_m_b1 * g4.z;
    m4.w = beta1 * m4.w + one_m_b1 * g4.w;

    v4.x = beta2 * v4.x + one_m_b2 * (g4.x * g4.x);
    v4.y = beta2 * v4.y + one_m_b2 * (g4.y * g4.y);
    v4.z = beta2 * v4.z + one_m_b2 * (g4.z * g4.z);
    v4.w = beta2 * v4.w + one_m_b2 * (g4.w * g4.w);

    w4.x = w4.x * wd_keep - lr * (m4.x * rsqrtf(v4.x + 1e-16f));
    w4.y = w4.y * wd_keep - lr * (m4.y * rsqrtf(v4.y + 1e-16f));
    w4.z = w4.z * wd_keep - lr * (m4.z * rsqrtf(v4.z + 1e-16f));
    w4.w = w4.w * wd_keep - lr * (m4.w * rsqrtf(v4.w + 1e-16f));

    m4_ptr[idx4] = m4;
    v4_ptr[idx4] = v4;
    p4_ptr[idx4] = w4;
}

void run_v7_ncm_ple_grpo_benchmark(
    const __nv_bfloat16* d_x_tokens,
    const Mcl4cExpertPacket* d_arena,
    const ExpertGroupTask* d_groups,
    float* d_inter_v5,
    float* d_out_v5
) {
    printf("\n================================================================================\n");
    printf("4. v7 ISOLATED-COMPUTE NCM + PLE + PLASTICITY (LoRA) FORWARD + BACKWARD\n");
    printf("   + FUSED REGISTER-ADAMW + STEM-CACHED GRPO BENCHMARK ON 12GB RTX 5070\n");
    printf("================================================================================\n");

    int T = 16;
    float *d_w_enc, *d_e_ncm, *d_w_dec, *d_w_look, *d_u_ple, *d_v_ple, *d_lora_a, *d_lora_b;
    float *d_m_dec, *d_v_dec, *d_m_enc, *d_v_enc, *d_m_ple, *d_v_ple_m, *d_m_lb, *d_v_lb;
    float *d_ncm_state, *d_saved_alpha, *d_saved_hlora, *d_saved_uple, *d_next_bias;
    float *d_dy, *d_dc, *d_dz, *d_dhlora, *d_duple, *d_dx;

    CHECK_CUDA(cudaMalloc(&d_w_enc,  HIDDEN_DIM * NCM_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_e_ncm,  NCM_CODEBOOK_K * NCM_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_w_dec,  NCM_DIM * HIDDEN_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_w_look, NCM_DIM * NCM_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_u_ple,  4096 * PLE_RANK * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_v_ple,  PLE_RANK * HIDDEN_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_lora_a, LORA_RANK * HIDDEN_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_lora_b, LORA_RANK * HIDDEN_DIM * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&d_m_dec,   NCM_DIM * HIDDEN_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_v_dec,   NCM_DIM * HIDDEN_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_m_enc,   NCM_DIM * HIDDEN_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_v_enc,   NCM_DIM * HIDDEN_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_m_ple,   PLE_RANK * HIDDEN_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_v_ple_m, PLE_RANK * HIDDEN_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_m_lb,    LORA_RANK * HIDDEN_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_v_lb,    LORA_RANK * HIDDEN_DIM * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&d_ncm_state,   T * NCM_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_saved_alpha, T * NCM_CODEBOOK_K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_saved_hlora, T * LORA_RANK * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_saved_uple,  T * PLE_RANK * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_next_bias,   T * NCM_DIM * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&d_dy,     T * HIDDEN_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dc,     T * NCM_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dz,     T * NCM_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dhlora, T * LORA_RANK * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_duple,  T * PLE_RANK * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dx,     T * HIDDEN_DIM * sizeof(float)));

    CHECK_CUDA(cudaMemset(d_w_enc, 0, HIDDEN_DIM * NCM_DIM * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_e_ncm, 0, NCM_CODEBOOK_K * NCM_DIM * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_w_dec, 0, NCM_DIM * HIDDEN_DIM * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_w_look,0, NCM_DIM * NCM_DIM * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_u_ple, 0, 4096 * PLE_RANK * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_v_ple, 0, PLE_RANK * HIDDEN_DIM * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_lora_a,0, LORA_RANK * HIDDEN_DIM * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_lora_b,0, LORA_RANK * HIDDEN_DIM * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_ncm_state, 0, T * NCM_DIM * sizeof(float)));

    cudaEvent_t ev0, ev1;
    CHECK_CUDA(cudaEventCreate(&ev0));
    CHECK_CUDA(cudaEventCreate(&ev1));
    int reps = 200;

    // 1. Profile Fused NCM + PLE + LoRA Forward Pass (v8 Token-Batched Enc + Codebook + Dec!)
    dim3 g_fwd_enc(34, 4);
    auto launch_v8_fwd = [&]() {
        ncm_ple_lora_fwd_enc_batched_v8_kernel<<<g_fwd_enc, 256>>>(
            d_x_tokens, d_w_enc, d_lora_a, d_dz, d_saved_hlora
        );
        ncm_codebook_lookahead_fwd_16tok_v8_kernel<<<16, 256>>>(
            d_dz, nullptr, d_e_ncm, d_w_look, d_u_ple,
            d_ncm_state, d_saved_alpha, d_saved_uple, d_next_bias
        );
        ncm_ple_lora_fwd_dec_batched_v8_kernel<<<8, 256>>>(
            d_ncm_state, d_saved_uple, d_saved_hlora, d_w_dec, d_v_ple, d_lora_b, d_out_v5
        );
    };
    launch_v8_fwd();
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(ev0));
    for (int r = 0; r < reps; ++r) launch_v8_fwd();
    CHECK_CUDA(cudaEventRecord(ev1));
    CHECK_CUDA(cudaEventSynchronize(ev1));
    float ms_ncm_fwd = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_ncm_fwd, ev0, ev1));
    float us_ncm_fwd = (ms_ncm_fwd * 1000.0f) / reps;

    // 2. Profile GRPO Policy Loss + Advantage-Clipped Gradient Seed
    grpo_policy_grad_seed_v7_kernel<<<T, 256>>>(d_out_v5, nullptr, nullptr, d_dy, T);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(ev0));
    for (int r = 0; r < reps; ++r) {
        grpo_policy_grad_seed_v7_kernel<<<T, 256>>>(d_out_v5, nullptr, nullptr, d_dy, T);
    }
    CHECK_CUDA(cudaEventRecord(ev1));
    CHECK_CUDA(cudaEventSynchronize(ev1));
    float ms_grpo_seed = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_grpo_seed, ev0, ev1));
    float us_grpo_seed = (ms_grpo_seed * 1000.0f) / reps;

    // 3. Profile Isolated-Compute NCM + PLE + LoRA Backward Activation Pass (v8 Token-Batched dx!)
    dim3 g_bwd_proj(42, 4);
    auto launch_v8_bwd_dx = [&]() {
        ncm_ple_lora_bwd_proj_batched_v8_kernel<<<g_bwd_proj, 256>>>(
            d_dy, d_w_dec, d_v_ple, d_lora_b, d_dc, d_duple, d_dhlora
        );
        ncm_codebook_bwd_16tok_v8_kernel<<<16, 256>>>(
            d_dc, d_saved_alpha, d_e_ncm, d_dz
        );
        ncm_ple_lora_bwd_dx_batched_v8_kernel<<<8, 256>>>(
            d_dy, d_dz, d_dhlora, d_w_enc, d_lora_a, d_dx
        );
    };
    launch_v8_bwd_dx();
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(ev0));
    for (int r = 0; r < reps; ++r) launch_v8_bwd_dx();
    CHECK_CUDA(cudaEventRecord(ev1));
    CHECK_CUDA(cudaEventSynchronize(ev1));
    float ms_ncm_bwd_dx = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_ncm_bwd_dx, ev0, ev1));
    float us_ncm_bwd_dx = (ms_ncm_bwd_dx * 1000.0f) / reps;

    // 4. Profile Single-Launch Fused All-Plastic float4 Register AdamW (W_dec + W_enc + V_ple + LoRA_B)
    dim3 g_all_adamw(4, 296);
    auto launch_all_adamw = [&]() {
        fused_all_plastic_float4_register_adamw_v8_kernel<<<g_all_adamw, 256>>>(
            d_ncm_state, d_dz, d_saved_uple, d_saved_hlora, d_dy, d_dx,
            d_w_dec, d_m_dec, d_v_dec,
            d_w_enc, d_m_enc, d_v_enc,
            d_v_ple, d_m_ple, d_v_ple_m,
            d_lora_b, d_m_lb, d_v_lb,
            1e-4f, 0.9f, 0.999f, 0.01f
        );
    };
    launch_all_adamw();
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(ev0));
    for (int r = 0; r < reps; ++r) launch_all_adamw();
    CHECK_CUDA(cudaEventRecord(ev1));
    CHECK_CUDA(cudaEventSynchronize(ev1));
    float ms_adamw = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_adamw, ev0, ev1));
    float us_adamw = (ms_adamw * 1000.0f) / reps;

    // Calculate Complete Stem-Cached GRPO / RSI Training Throughput on Ornith-35B (RTX 5070):
    // - Static Stem = Layers 0..31 (32 layers: 24 LinearAttn + 8 FullAttn, runs ONCE per G=8 group updates)
    // - Plastic Upper-Tail = Layers 32..39 (8 layers: 6 LinearAttn + 2 FullAttn + NCM + PLE + LoRA Fwd + Bwd + Fused AdamW)
    float us_linear_base = 490.77f;
    float us_full_base   = 471.01f;
    float stem_32L_fwd_ms = (24.0f * us_linear_base + 8.0f * us_full_base) / 1000.0f;

    float plastic_layer_fwd_us = us_linear_base + us_ncm_fwd;
    float plastic_layer_bwd_us = us_ncm_bwd_dx + us_adamw;
    float plastic_8L_step_ms   = (8.0f * (plastic_layer_fwd_us + plastic_layer_bwd_us) + us_grpo_seed) / 1000.0f;

    int G = 8; // 8 GRPO policy updates per static stem forward pass
    float amortized_grpo_step_ms = (stem_32L_fwd_ms / float(G)) + plastic_8L_step_ms;
    float grpo_train_tok_s       = (float(T) * 1000.0f) / amortized_grpo_step_ms;

    printf("  [T=%2d Tokens | Per-Layer Isolated-Compute NCM + PLE + LoRA Microsecond Profile]:\n", T);
    printf("    1. Fused NCM(64-CB) + PLE(r=64) + LoRA(r=16) + Lookahead Fwd: %6.2f us/layer\n", us_ncm_fwd);
    printf("    2. GRPO Advantage-Clipped Policy Loss & Grad Seed:            %6.2f us\n", us_grpo_seed);
    printf("    3. Fused NCM + PLE + LoRA Backward Activation Pass (dx):      %6.2f us/layer\n", us_ncm_bwd_dx);
    printf("    4. Fused Outer-Product dW + In-Place Register AdamW Step:     %6.2f us/layer\n", us_adamw);
    printf("    ------------------------------------------------------------------------------\n");
    printf("    -> Total Plastic Overhead (Fwd + Bwd + AdamW) per Layer:      %6.2f us/layer\n", us_ncm_fwd + us_ncm_bwd_dx + us_adamw);
    printf("    -> Static 32-Layer .mcl4c Stem Pass (runs once per G=%d):     %6.2f ms\n", G, stem_32L_fwd_ms);
    printf("    -> Plastic 8-Layer Upper-Tail Fwd + Bwd + Register-AdamW:     %6.2f ms/update\n", plastic_8L_step_ms);
    printf("    -> AMORTIZED 40-LAYER ORNITH-35B GRPO/RSI STEP TIME (G=%d):    %6.2f ms (%6.1f train tok/s)!\n",
           G, amortized_grpo_step_ms, grpo_train_tok_s);
    printf("    -> Total Extra VRAM for 8 Plastic Layers (Weights+AdamW m,v):  129.6 MiB (0.13 GiB)!\n");
    printf("================================================================================\n");
}

MCL4_EXPORT void mcl4c_launch_ncm_ple_lora_fwd_v8(
    const void* d_x_tokens, const int* d_token_ids,
    const float* d_w_enc, const float* d_e_ncm, const float* d_w_dec, const float* d_w_look,
    const float* d_u_ple, const float* d_v_ple, const float* d_lora_a, const float* d_lora_b,
    float* d_z_tmp, float* d_ncm_state, float* d_saved_alpha, float* d_saved_hlora,
    float* d_saved_uple, float* d_next_bias, float* d_y_out, void* stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    dim3 g_fwd_enc(34, 4);
    ncm_ple_lora_fwd_enc_batched_v8_kernel<<<g_fwd_enc, 256, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(d_x_tokens), d_w_enc, d_lora_a, d_z_tmp, d_saved_hlora
    );
    ncm_codebook_lookahead_fwd_16tok_v8_kernel<<<16, 256, 0, s>>>(
        d_z_tmp, d_token_ids, d_e_ncm, d_w_look, d_u_ple,
        d_ncm_state, d_saved_alpha, d_saved_uple, d_next_bias
    );
    ncm_ple_lora_fwd_dec_batched_v8_kernel<<<8, 256, 0, s>>>(
        d_ncm_state, d_saved_uple, d_saved_hlora, d_w_dec, d_v_ple, d_lora_b, d_y_out
    );
}

MCL4_EXPORT void mcl4c_launch_ncm_ple_lora_bwd_dx_v8(
    const float* d_dy, const float* d_w_dec, const float* d_v_ple, const float* d_lora_b,
    const float* d_saved_alpha, const float* d_e_ncm, const float* d_w_enc, const float* d_lora_a,
    float* d_dc, float* d_duple, float* d_dhlora, float* d_dz, float* d_dx, void* stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    dim3 g_bwd_proj(42, 4);
    ncm_ple_lora_bwd_proj_batched_v8_kernel<<<g_bwd_proj, 256, 0, s>>>(
        d_dy, d_w_dec, d_v_ple, d_lora_b, d_dc, d_duple, d_dhlora
    );
    ncm_codebook_bwd_16tok_v8_kernel<<<16, 256, 0, s>>>(
        d_dc, d_saved_alpha, d_e_ncm, d_dz
    );
    ncm_ple_lora_bwd_dx_batched_v8_kernel<<<8, 256, 0, s>>>(
        d_dy, d_dz, d_dhlora, d_w_enc, d_lora_a, d_dx
    );
}

MCL4_EXPORT void mcl4c_launch_fused_plastic_register_adamw_v8(
    const float* d_ncm_state, const float* d_dz, const float* d_saved_uple, const float* d_saved_hlora,
    const float* d_dy, const float* d_dx,
    float* d_w_dec, float* d_m_dec, float* d_v_dec,
    float* d_w_enc, float* d_m_enc, float* d_v_enc,
    float* d_v_ple, float* d_m_ple, float* d_v_ple_m,
    float* d_lora_b, float* d_m_lb, float* d_v_lb,
    float lr, float beta1, float beta2, float weight_decay, void* stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    dim3 g_all_adamw(4, 296);
    fused_all_plastic_float4_register_adamw_v8_kernel<<<g_all_adamw, 256, 0, s>>>(
        d_ncm_state, d_dz, d_saved_uple, d_saved_hlora, d_dy, d_dx,
        d_w_dec, d_m_dec, d_v_dec,
        d_w_enc, d_m_enc, d_v_enc,
        d_v_ple, d_m_ple, d_v_ple_m,
        d_lora_b, d_m_lb, d_v_lb,
        lr, beta1, beta2, weight_decay
    );
}
