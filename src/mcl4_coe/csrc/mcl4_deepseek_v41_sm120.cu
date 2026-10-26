// ============================================================================
// MCL4-Codec Native C++/CUDA SM120 Single-Point Compute + Isolated-Compute RL
// Engine for DeepSeek-V4.1-Flash-1017-NVFP4 (v1 — Complete 40-Layer Architecture)
//
// Target Checkpoint: Z:\deepseek-nvfp4 (Infatoshi/DeepSeek-V4.1-Flash-1017-NVFP4)
// Exact Architecture (from Z:\deepseek-nvfp4\inference\model.py & config.json):
//   - Backbone: 40 Layers, dim = 5120, moe_inter_dim = 2304, vocab_size = 129,280
//   - Hyper-Connections (hc_mult = 4):
//       * Residual stream carried as [B, S, 4, 5120] (20,480 floats/token)
//       * hc_mixes: RMS-normalized linear [24, 20480] + 20-iter 4x4 Sinkhorn-Knopp
//       * hc_pre collapses [4, 5120] -> [5120]; hc_post expands [5120] -> [4, 5120]
//   - Multi-Head Latent Attention (MLA + Sliding Window + Compressed KV Indexer):
//       * n_heads = 64, head_dim = 512 (nope = 448, rope = 64), 1 shared KV head (512)
//       * wq_a [1280, 5120] (FP8_E4M3 + E8M0 32x32) -> q_norm(1280) -> wq_b [32768, 1280]
//       * wkv  [512, 5120]  (FP8_E4M3 + E8M0 32x32) -> kv_norm(512) + RoPE(64)
//       * Sparse Online-Softmax Attention over 128-token Sliding Window + 512-topk
//         Compressed KV with learnable attn_sink[64] + Inverse-RoPE on output tail
//       * 8-Group Block-Diagonal Output LoRA: wo_a [8, 1024, 4096] -> wo_b [5120, 8192]
//   - DeepSeek-V4.1 MoE (384 Routed NVFP4 Experts + 1 Shared FP8 Expert, top_k = 6):
//       * Gate: sqrt(softplus(x @ W_gate^T)) + noaux_tc bias-steered top-6, scale = 1.5
//       * Clamped SwiGLU (swiglu_limit = 10.0): min(gate, 10) * clamp(up, -10, 10)
//       * Zero-Repack .mcl4c NVFP4 Expert Packets (18.99 MiB/expert, U8 E2M1x2 + E4M3 group-16)
//   - O(1) Zero-VRAM NVMe Engram Lookup (Layers 1 & 14: 24 rows x 256B = 6 KiB/token)
//   - Isolated-Compute NCM(64-CB) + PLE(r=64) + LoRA(r=16) + 384-Expert Lookahead
//     + Single-Launch float4 Register-AdamW (Stem-Cached GRPO/RSI Training)
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <string>
#include <algorithm>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// ============================================================================
// 1. DeepSeek-V4.1-Flash-1017 Architectural Constants & .mcl4c Packet Struct
// ============================================================================
static constexpr int DS_DIM             = 5120;
static constexpr int DS_HALF_DIM        = 2560; // 5120 / 2
static constexpr int DS_MOE_INTER       = 2304;
static constexpr int DS_HALF_INTER      = 1152; // 2304 / 2
static constexpr int DS_GATE_UP_DIM     = 4608; // 2 * 2304
static constexpr int DS_ROUTED_EXPERTS  = 384;
static constexpr int DS_TOP_K           = 6;
static constexpr int DS_HC_DIM          = 20480; // 4 * 5120
static constexpr int DS_MIX_HC          = 24;    // (2 + 4) * 4
static constexpr int DS_SINKHORN_ITERS  = 20;

static constexpr int DS_N_HEADS         = 64;
static constexpr int DS_HEAD_DIM        = 512;
static constexpr int DS_Q_LORA_RANK     = 1280;
static constexpr int DS_O_GROUPS        = 8;
static constexpr int DS_O_LORA_RANK     = 1024;
static constexpr int DS_O_TOTAL_RANK    = 8192;  // 8 * 1024
static constexpr int DS_GROUP_IN_DIM    = 4096;  // 8 * 512
static constexpr int DS_WINDOW_SIZE     = 128;

static constexpr int NCM_DIM            = 256;
static constexpr int NCM_CODEBOOK_K     = 64;
static constexpr int PLE_RANK           = 64;
static constexpr int LORA_RANK          = 16;

// Exact 4K-Page-Aligned Zero-Repack .mcl4c Expert Packet for DeepSeek-V4.1-Flash-1017-NVFP4
// Matches layers.{l}.ffn.experts.{e}.{w1,w3,w2}.{weight,weight_scale,weight_scale_2} 1:1!
struct alignas(4096) Dsv41Mcl4cExpertPacket {
    uint8_t  w1_w3_fp4[DS_GATE_UP_DIM * DS_HALF_DIM];      // 11,796,480 B (w1[2304,2560] + w3[2304,2560])
    uint8_t  w2_fp4[DS_DIM * DS_HALF_INTER];               //  5,898,240 B (w2[5120,1152])
    uint8_t  w1_w3_scales[DS_GATE_UP_DIM * (DS_DIM / 16)]; //  1,474,560 B (F8_E4M3 group-16 scales)
    uint8_t  w2_scales[DS_DIM * (DS_MOE_INTER / 16)];      //    737,280 B (F8_E4M3 group-16 scales)
    float    w1_scale2;                                    //          4 B (weight_scale_2)
    float    w3_scale2;                                    //          4 B (weight_scale_2)
    float    w2_scale2;                                    //          4 B (weight_scale_2)
    uint32_t layer_idx;                                    //          4 B
    uint32_t expert_idx;                                   //          4 B
    uint8_t  _pad[4096 - 20];                              //  4,076 B -> Total = 19,910,656 B (18.988 MiB)
};
static_assert(sizeof(Dsv41Mcl4cExpertPacket) == 19910656, "Dsv41Mcl4cExpertPacket must be 4K-aligned 19,910,656 bytes");

struct Dsv41ExpertGroupTask {
    int   arena_slot;
    int   m_count;
    int   token_ids[8];
    float weights[8];
};

// ============================================================================
// 2. Fast Hardware Helpers (SM120 E2M1 Warp Register LUT, E8M0 Scale, E4M3 Decode,
//    plus DeepGEMM + DeepEP Blackwell/SM120 PTX Intrinsics & RAII Memory Safety)
// ============================================================================

// RAII CUDA Device Buffer Wrapper — Zero raw pointer leaks, bounds-checked transfers
template <typename T>
class CudaBuffer {
public:
    CudaBuffer() : ptr_(nullptr), count_(0) {}
    explicit CudaBuffer(size_t count) : ptr_(nullptr), count_(count) {
        if (count_ > 0) {
            CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&ptr_), count_ * sizeof(T)));
            total_allocated_bytes_ += count_ * sizeof(T);
        }
    }
    ~CudaBuffer() { reset(); }

    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    CudaBuffer(CudaBuffer&& other) noexcept : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }
    CudaBuffer& operator=(CudaBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    void reset() {
        if (ptr_) {
            cudaFree(ptr_);
            if (total_allocated_bytes_ >= count_ * sizeof(T)) {
                total_allocated_bytes_ -= count_ * sizeof(T);
            }
            ptr_ = nullptr;
            count_ = 0;
        }
    }

    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    operator T*() { return ptr_; }
    operator const T*() const { return ptr_; }
    size_t size() const { return count_; }
    size_t bytes() const { return count_ * sizeof(T); }

    void memset(int value) {
        if (ptr_ && count_ > 0) {
            CHECK_CUDA(cudaMemset(ptr_, value, count_ * sizeof(T)));
        }
    }

    void copy_from_host(const T* src, size_t n_elems) {
        if (n_elems > count_) {
            fprintf(stderr, "CudaBuffer::copy_from_host bounds violation (%zu > %zu)\n", n_elems, count_);
            exit(1);
        }
        if (n_elems > 0) {
            CHECK_CUDA(cudaMemcpy(ptr_, src, n_elems * sizeof(T), cudaMemcpyHostToDevice));
        }
    }

    void copy_to_host(T* dst, size_t n_elems) const {
        if (n_elems > count_) {
            fprintf(stderr, "CudaBuffer::copy_to_host bounds violation (%zu > %zu)\n", n_elems, count_);
            exit(1);
        }
        if (n_elems > 0) {
            CHECK_CUDA(cudaMemcpy(dst, ptr_, n_elems * sizeof(T), cudaMemcpyDeviceToHost));
        }
    }

    static size_t total_allocated_bytes() { return total_allocated_bytes_; }

private:
    T* ptr_;
    size_t count_;
    static size_t total_allocated_bytes_;
};

template <typename T>
size_t CudaBuffer<T>::total_allocated_bytes_ = 0;

struct ScopedCudaEvent {
    cudaEvent_t ev;
    ScopedCudaEvent() : ev(nullptr) { CHECK_CUDA(cudaEventCreate(&ev)); }
    ~ScopedCudaEvent() { if (ev) cudaEventDestroy(ev); }
    ScopedCudaEvent(const ScopedCudaEvent&) = delete;
    ScopedCudaEvent& operator=(const ScopedCudaEvent&) = delete;
    operator cudaEvent_t() const { return ev; }
};

struct ScopedFile {
    FILE* fp;
    explicit ScopedFile(const char* path, const char* mode) : fp(fopen(path, mode)) {}
    ~ScopedFile() { if (fp) fclose(fp); }
    ScopedFile(const ScopedFile&) = delete;
    ScopedFile& operator=(const ScopedFile&) = delete;
    explicit operator bool() const { return fp != nullptr; }
};

// DeepGEMM + DeepEP PTX Intrinsics for SM120 (Blackwell):
// 1. L1::no_allocate + L2::cache_hint (kEvictFirst = 0x12f0000000000000ULL)
//    Streams single-use weights without polluting L1 or evicting hot activations in 48 MiB L2!
static constexpr uint64_t DG_L2_EVICT_FIRST_HINT = 0x12f0000000000000ULL;

__device__ __forceinline__ uint4 dg_ld_evict_first_u4(const uint4* ptr) {
    uint4 ret;
    asm volatile(
        "ld.L1::no_allocate.L2::cache_hint.global.nc.v4.u32 {%0, %1, %2, %3}, [%4], %5;"
        : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w)
        : "l"(ptr), "l"(DG_L2_EVICT_FIRST_HINT)
    );
    return ret;
}

__device__ __forceinline__ float4 dg_ld_evict_first_f4(const float4* ptr) {
    float4 ret;
    asm volatile(
        "ld.L1::no_allocate.L2::cache_hint.global.nc.v4.f32 {%0, %1, %2, %3}, [%4], %5;"
        : "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w)
        : "l"(ptr), "l"(DG_L2_EVICT_FIRST_HINT)
    );
    return ret;
}

// DeepEP predicated L1::no_allocate + L2::evict_first load (eliminates branch divergence on tail tiles!)
__device__ __forceinline__ uint4 dg_ld_pred_evict_first_u4(const uint4* ptr, int pred_valid) {
    uint4 ret = make_uint4(0, 0, 0, 0);
    asm volatile(
        "{\n\t"
        "  .reg .pred p;\n\t"
        "  setp.ne.s32 p, %5, 0;\n\t"
        "  @p ld.L1::no_allocate.L2::cache_hint.global.nc.v4.u32 {%0, %1, %2, %3}, [%4], %6;\n\t"
        "}"
        : "+r"(ret.x), "+r"(ret.y), "+r"(ret.z), "+r"(ret.w)
        : "l"(ptr), "r"(pred_valid), "l"(DG_L2_EVICT_FIRST_HINT)
    );
    return ret;
}

// DeepGEMM fast SFU reciprocal (1-instruction rcp.approx.ftz.f32)
__device__ __forceinline__ float dg_fast_rcp(float x) {
    float y;
    asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// DeepGEMM 5-comparator 4-element sorting network (sorts descending: s0 >= s1 >= s2 >= s3)
__device__ __forceinline__ void dg_compare_swap(float& a_s, int& a_i, float& b_s, int& b_i) {
    if (b_s > a_s || (b_s == a_s && b_i < a_i)) {
        float ts = a_s; a_s = b_s; b_s = ts;
        int   ti = a_i; a_i = b_i; b_i = ti;
    }
}

__device__ __forceinline__ void dg_sort4_best_first(
    float& s0, int& i0, float& s1, int& i1, float& s2, int& i2, float& s3, int& i3
) {
    dg_compare_swap(s0, i0, s1, i1);
    dg_compare_swap(s2, i2, s3, i3);
    dg_compare_swap(s0, i0, s2, i2);
    dg_compare_swap(s1, i1, s3, i3);
    dg_compare_swap(s1, i1, s2, i2);
}

__device__ __forceinline__ float get_nvfp4_lut_lane_val(int lane) {
    int idx = lane & 15;
    float mag = 0.0f;
    switch (idx & 7) {
        case 0: mag = 0.0f; break;
        case 1: mag = 0.5f; break;
        case 2: mag = 1.0f; break;
        case 3: mag = 1.5f; break;
        case 4: mag = 2.0f; break;
        case 5: mag = 3.0f; break;
        case 6: mag = 4.0f; break;
        case 7: mag = 6.0f; break;
    }
    return (idx & 8) ? -mag : mag;
}

__device__ __forceinline__ float decode_fp8_e4m3_fast(uint8_t b) {
    __half_raw hr = __nv_cvt_fp8_to_halfraw(b, __NV_E4M3);
    return __half2float(*reinterpret_cast<const __half*>(&hr));
}

// IEEE-754 power-of-2 scale from F8_E8M0 byte (1-cycle shift!)
__device__ __forceinline__ float decode_fp8_e8m0_fast(uint8_t e8m0) {
    uint32_t bits = ((uint32_t)e8m0) << 23;
    return __uint_as_float(bits);
}

// ============================================================================
// 3. KERNEL 1 (v5): Single-Launch Fused Hyper-Connections (128-bit float4/uint2
//    Vectorized `hc_fn` Projection + 20-Iter 4x4 Sinkhorn-Knopp with `rcp.approx.ftz.f32`
//    + hc_pre Collapse + Sub-Layer RMSNorm(5120, eps=1e-20))
// ============================================================================
__global__ void __launch_bounds__(512, 2) dsv41_fused_hc_sinkhorn_pre_rmsnorm_v5_kernel(
    const __nv_bfloat16* __restrict__ x_hc,       // [T, 4, 5120] = [T, 20480]
    const float* __restrict__ pre_mix_in,         // [T, 4]
    const float* __restrict__ hc_fn,              // [24, 20480]
    const float* __restrict__ hc_scale,           // [3]
    const float* __restrict__ hc_base,            // [24]
    const __nv_bfloat16* __restrict__ rms_weight, // [5120]
    __nv_bfloat16* __restrict__ x_norm_out,       // [T, 5120]
    float* __restrict__ next_pre_out,             // [T, 4]
    float* __restrict__ post_out,                 // [T, 4]
    float* __restrict__ comb_out                  // [T, 16]
) {
    int t    = blockIdx.x;
    int tid  = threadIdx.x; // 0..511
    int wid  = tid >> 5;    // 0..15
    int lane = tid & 31;

    const __nv_bfloat16* x_tok = x_hc + (size_t)t * DS_HC_DIM;
    float pm0 = pre_mix_in[t * 4 + 0];
    float pm1 = pre_mix_in[t * 4 + 1];
    float pm2 = pre_mix_in[t * 4 + 2];
    float pm3 = pre_mix_in[t * 4 + 3];

    float sq_hc_local  = 0.0f;
    float sq_sub_local = 0.0f;
    float r_sub[10];

    const __nv_bfloat162* x_c0 = reinterpret_cast<const __nv_bfloat162*>(x_tok + 0 * DS_DIM);
    const __nv_bfloat162* x_c1 = reinterpret_cast<const __nv_bfloat162*>(x_tok + 1 * DS_DIM);
    const __nv_bfloat162* x_c2 = reinterpret_cast<const __nv_bfloat162*>(x_tok + 2 * DS_DIM);
    const __nv_bfloat162* x_c3 = reinterpret_cast<const __nv_bfloat162*>(x_tok + 3 * DS_DIM);

    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        int idx2 = tid + i * 512; // 0..2559
        float2 v0 = __bfloat1622float2(x_c0[idx2]);
        float2 v1 = __bfloat1622float2(x_c1[idx2]);
        float2 v2 = __bfloat1622float2(x_c2[idx2]);
        float2 v3 = __bfloat1622float2(x_c3[idx2]);

        sq_hc_local += v0.x * v0.x + v0.y * v0.y
                     + v1.x * v1.x + v1.y * v1.y
                     + v2.x * v2.x + v2.y * v2.y
                     + v3.x * v3.x + v3.y * v3.y;

        float s_x = pm0 * v0.x + pm1 * v1.x + pm2 * v2.x + pm3 * v3.x;
        float s_y = pm0 * v0.y + pm1 * v1.y + pm2 * v2.y + pm3 * v3.y;
        r_sub[i * 2 + 0] = s_x;
        r_sub[i * 2 + 1] = s_y;
        sq_sub_local += s_x * s_x + s_y * s_y;
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sq_hc_local  += __shfl_xor_sync(0xffffffff, sq_hc_local, off);
        sq_sub_local += __shfl_xor_sync(0xffffffff, sq_sub_local, off);
    }
    __shared__ float s_red_hc[16];
    __shared__ float s_red_sub[16];
    if (lane == 0) {
        s_red_hc[wid]  = sq_hc_local;
        s_red_sub[wid] = sq_sub_local;
    }
    __syncthreads();

    if (wid == 0) {
        float v_hc  = (lane < 16) ? s_red_hc[lane]  : 0.0f;
        float v_sub = (lane < 16) ? s_red_sub[lane] : 0.0f;
        #pragma unroll
        for (int off = 8; off > 0; off >>= 1) {
            v_hc  += __shfl_xor_sync(0xffffffff, v_hc, off);
            v_sub += __shfl_xor_sync(0xffffffff, v_sub, off);
        }
        if (lane == 0) {
            s_red_hc[0]  = rsqrtf(v_hc * (1.0f / 20480.0f) + 1e-20f);
            s_red_sub[0] = rsqrtf(v_sub * (1.0f / 5120.0f) + 1e-20f);
        }
    }
    __syncthreads();

    float rms_hc  = s_red_hc[0];
    float rms_sub = s_red_sub[0];

    const __nv_bfloat162* w_rms2 = reinterpret_cast<const __nv_bfloat162*>(rms_weight);
    __nv_bfloat162* out_norm2 = reinterpret_cast<__nv_bfloat162*>(x_norm_out + (size_t)t * DS_DIM);
    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        int idx2 = tid + i * 512;
        float2 wf = __bfloat1622float2(w_rms2[idx2]);
        float2 nf;
        nf.x = r_sub[i * 2 + 0] * rms_sub * wf.x;
        nf.y = r_sub[i * 2 + 1] * rms_sub * wf.y;
        out_norm2[idx2] = __float22bfloat162_rn(nf);
    }

    // Step 2: 46-register high-occupancy float2 warp-parallel projection mixes = (x_tok @ hc_fn^T) * rms_hc
    __shared__ float s_mixes[24];
    const __nv_bfloat162* x_all2 = reinterpret_cast<const __nv_bfloat162*>(x_tok);
    for (int m = wid; m < 24; m += 16) {
        const float2* fn2 = reinterpret_cast<const float2*>(hc_fn + (size_t)m * DS_HC_DIM);
        float acc = 0.0f;
        #pragma unroll 8
        for (int k = 0; k < 320; ++k) {
            int idx2 = lane + k * 32;
            float2 xv = __bfloat1622float2(x_all2[idx2]);
            float2 fw = fn2[idx2];
            acc += xv.x * fw.x + xv.y * fw.y;
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, off);
        if (lane == 0) s_mixes[m] = acc * rms_hc;
    }
    __syncthreads();

    // Step 3: Exact 20-Iteration 4x4 Sinkhorn-Knopp in Warp 0 using DeepGEMM `rcp.approx.ftz.f32`!
    if (wid == 0 && lane < 16) {
        float sc0 = hc_scale[0], sc1 = hc_scale[1], sc2 = hc_scale[2];
        const float eps = 1e-6f;
        if (lane < 4) {
            float v_pre  = s_mixes[lane] * sc0 + hc_base[lane];
            float v_post = s_mixes[lane + 4] * sc1 + hc_base[lane + 4];
            next_pre_out[t * 4 + lane] = dg_fast_rcp(1.0f + __expf(-v_pre)) + eps;
            post_out[t * 4 + lane]     = 2.0f * dg_fast_rcp(1.0f + __expf(-v_post));
        }
        float c_val = s_mixes[8 + lane] * sc2 + hc_base[8 + lane];

        float r_max = c_val;
        r_max = fmaxf(r_max, __shfl_xor_sync(0x0000ffff, r_max, 1));
        r_max = fmaxf(r_max, __shfl_xor_sync(0x0000ffff, r_max, 2));
        c_val = __expf(c_val - r_max);
        float r_sum = c_val;
        r_sum += __shfl_xor_sync(0x0000ffff, r_sum, 1);
        r_sum += __shfl_xor_sync(0x0000ffff, r_sum, 2);
        c_val = (c_val * dg_fast_rcp(r_sum)) + eps;

        float col_sum = c_val;
        col_sum += __shfl_xor_sync(0x0000ffff, col_sum, 4);
        col_sum += __shfl_xor_sync(0x0000ffff, col_sum, 8);
        c_val = c_val * dg_fast_rcp(col_sum + eps);

        #pragma unroll
        for (int iter = 0; iter < DS_SINKHORN_ITERS - 1; ++iter) {
            r_sum = c_val;
            r_sum += __shfl_xor_sync(0x0000ffff, r_sum, 1);
            r_sum += __shfl_xor_sync(0x0000ffff, r_sum, 2);
            c_val = c_val * dg_fast_rcp(r_sum + eps);

            col_sum = c_val;
            col_sum += __shfl_xor_sync(0x0000ffff, col_sum, 4);
            col_sum += __shfl_xor_sync(0x0000ffff, col_sum, 8);
            c_val = c_val * dg_fast_rcp(col_sum + eps);
        }
        comb_out[t * 16 + lane] = c_val;
    }
}

// Fused Hyper-Connection hc_post Expansion & Residual Stream Mixing
// y[t, c, d] = post[t, c] * sub_out[t, d] + sum_{k=0}^3 comb[t, k, c] * residual[t, k, d]
__global__ void __launch_bounds__(256, 4) dsv41_fused_hc_post_v1_kernel(
    const __nv_bfloat16* __restrict__ sub_out, // [T, 5120]
    const float* __restrict__ post_mix,        // [T, 4]
    const float* __restrict__ comb_mix,        // [T, 4, 4]
    __nv_bfloat16* __restrict__ x_hc_inout,    // [T, 4, 5120]
    int T
) {
    // Grid: (10, T) -> 10 blocks x 256 threads = 2560 bfloat162 pairs = 5120 dims
    int idx2 = blockIdx.x * 256 + threadIdx.x; // 0..2559
    int t    = blockIdx.y;

    __shared__ float s_post[4];
    __shared__ float s_comb[16];
    if (threadIdx.x < 4)  s_post[threadIdx.x] = post_mix[t * 4 + threadIdx.x];
    if (threadIdx.x < 16) s_comb[threadIdx.x] = comb_mix[t * 16 + threadIdx.x];
    __syncthreads();

    const __nv_bfloat162* sub2 = reinterpret_cast<const __nv_bfloat162*>(sub_out + (size_t)t * DS_DIM);
    __nv_bfloat162* hc0 = reinterpret_cast<__nv_bfloat162*>(x_hc_inout + ((size_t)t * 4 + 0) * DS_DIM);
    __nv_bfloat162* hc1 = reinterpret_cast<__nv_bfloat162*>(x_hc_inout + ((size_t)t * 4 + 1) * DS_DIM);
    __nv_bfloat162* hc2 = reinterpret_cast<__nv_bfloat162*>(x_hc_inout + ((size_t)t * 4 + 2) * DS_DIM);
    __nv_bfloat162* hc3 = reinterpret_cast<__nv_bfloat162*>(x_hc_inout + ((size_t)t * 4 + 3) * DS_DIM);

    float2 sv = __bfloat1622float2(sub2[idx2]);
    float2 r0 = __bfloat1622float2(hc0[idx2]);
    float2 r1 = __bfloat1622float2(hc1[idx2]);
    float2 r2 = __bfloat1622float2(hc2[idx2]);
    float2 r3 = __bfloat1622float2(hc3[idx2]);

    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        float p  = s_post[c];
        float k0 = s_comb[0 * 4 + c];
        float k1 = s_comb[1 * 4 + c];
        float k2 = s_comb[2 * 4 + c];
        float k3 = s_comb[3 * 4 + c];
        float2 out_v;
        out_v.x = p * sv.x + k0 * r0.x + k1 * r1.x + k2 * r2.x + k3 * r3.x;
        out_v.y = p * sv.y + k0 * r0.y + k1 * r1.y + k2 * r2.y + k3 * r3.y;
        if (c == 0) hc0[idx2] = __float22bfloat162_rn(out_v);
        if (c == 1) hc1[idx2] = __float22bfloat162_rn(out_v);
        if (c == 2) hc2[idx2] = __float22bfloat162_rn(out_v);
        if (c == 3) hc3[idx2] = __float22bfloat162_rn(out_v);
    }
}

// ============================================================================
// 4. KERNEL 2 (v4): 2-Rows-Per-Warp (16 Rows/Block) K-Tiled SM120 FP8_E4M3 + E8M0 GEMM
//    Each warp computes 2 output rows simultaneously while reading s_x2[m][c2] ONCE
//    from swizzled shared memory -> 2x fewer blocks, 2x less L2 activation traffic,
//    and 2x fewer shared-memory LDS instructions!
// ============================================================================
template <int M_CAP, int K_DIM>
__global__ void __launch_bounds__(256, 2) dsv41_fp8_e8m0_gemm_v1_kernel(
    const __nv_bfloat16* __restrict__ x_in, // [T, K_DIM]
    const uint8_t* __restrict__ w_fp8,      // [N_OUT, K_DIM] (F8_E4M3)
    const uint8_t* __restrict__ s_e8m0,     // [N_OUT/32, K_DIM/32] (F8_E8M0)
    __nv_bfloat16* __restrict__ y_out,      // [T, N_OUT]
    int m_valid,
    int N_OUT
) {
    // 256 threads = 8 warps -> 16 output rows per block (2 rows per warp)
    int wid  = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row0 = blockIdx.x * 16 + wid * 2 + 0;
    int row1 = row0 + 1;

    constexpr int HALF_K = K_DIM / 2;
    constexpr int TILE_HALF_K = 256; // 512 elements per K-tile
    constexpr int TILE_STRIDE = 288; // 256 + (256 >> 3)
    __shared__ __nv_bfloat162 s_x2[M_CAP][TILE_STRIDE];

    float acc0[M_CAP], acc1[M_CAP];
    #pragma unroll
    for (int m = 0; m < M_CAP; ++m) { acc0[m] = 0.0f; acc1[m] = 0.0f; }

    const __nv_bfloat162* x2_in = reinterpret_cast<const __nv_bfloat162*>(x_in);
    const uint4* w0_u4 = (row0 < N_OUT) ? reinterpret_cast<const uint4*>(w_fp8 + (size_t)row0 * K_DIM) : nullptr;
    const uint4* w1_u4 = (row1 < N_OUT) ? reinterpret_cast<const uint4*>(w_fp8 + (size_t)row1 * K_DIM) : nullptr;
    const uint8_t* s0_row = (row0 < N_OUT) ? (s_e8m0 + (size_t)(row0 >> 5) * (K_DIM >> 5)) : nullptr;

    constexpr int NUM_TILES = K_DIM / 512;
    for (int t = 0; t < NUM_TILES; ++t) {
        int k2_base = t * TILE_HALF_K;
        for (int idx = threadIdx.x; idx < m_valid * TILE_HALF_K; idx += 256) {
            int m = idx >> 8;
            int c = idx & 255;
            s_x2[m][c + (c >> 3)] = x2_in[(size_t)m * HALF_K + k2_base + c];
        }
        __syncthreads();

        if (row1 < N_OUT) {
            int u4_idx = t * 32 + lane;
            uint4 pk0 = w0_u4[u4_idx];
            uint4 pk1 = w1_u4[u4_idx];
            // Note: row0 and row1 (even, odd) always lie in the SAME 32-row E8M0 block (row0 >> 5 == row1 >> 5)!
            float blk_scale = decode_fp8_e8m0_fast(s0_row[u4_idx >> 1]);

            uint32_t words0[4] = {pk0.x, pk0.y, pk0.z, pk0.w};
            uint32_t words1[4] = {pk1.x, pk1.y, pk1.z, pk1.w};
            int swz_base = lane * 9;

            float sub0[M_CAP], sub1[M_CAP];
            #pragma unroll
            for (int m = 0; m < M_CAP; ++m) { sub0[m] = 0.0f; sub1[m] = 0.0f; }

            #pragma unroll
            for (int w = 0; w < 4; ++w) {
                uint32_t p0 = words0[w];
                uint32_t p1 = words1[w];
                float a0 = decode_fp8_e4m3_fast(p0 & 0xFF);
                float a1 = decode_fp8_e4m3_fast((p0 >> 8) & 0xFF);
                float a2 = decode_fp8_e4m3_fast((p0 >> 16) & 0xFF);
                float a3 = decode_fp8_e4m3_fast((p0 >> 24) & 0xFF);

                float b0 = decode_fp8_e4m3_fast(p1 & 0xFF);
                float b1 = decode_fp8_e4m3_fast((p1 >> 8) & 0xFF);
                float b2 = decode_fp8_e4m3_fast((p1 >> 16) & 0xFF);
                float b3 = decode_fp8_e4m3_fast((p1 >> 24) & 0xFF);

                int c2_0 = swz_base + w * 2 + 0;
                int c2_1 = swz_base + w * 2 + 1;

                #pragma unroll
                for (int m = 0; m < M_CAP; ++m) {
                    float2 x01 = __bfloat1622float2(s_x2[m][c2_0]);
                    float2 x23 = __bfloat1622float2(s_x2[m][c2_1]);
                    sub0[m] += a0 * x01.x + a1 * x01.y + a2 * x23.x + a3 * x23.y;
                    sub1[m] += b0 * x01.x + b1 * x01.y + b2 * x23.x + b3 * x23.y;
                }
            }
            #pragma unroll
            for (int m = 0; m < M_CAP; ++m) {
                acc0[m] += sub0[m] * blk_scale;
                acc1[m] += sub1[m] * blk_scale;
            }
        }
        __syncthreads();
    }

    if (row1 < N_OUT) {
        #pragma unroll
        for (int m = 0; m < M_CAP; ++m) {
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                acc0[m] += __shfl_xor_sync(0xffffffff, acc0[m], off);
                acc1[m] += __shfl_xor_sync(0xffffffff, acc1[m], off);
            }
            if (lane == 0 && m < m_valid) {
                y_out[(size_t)m * N_OUT + row0] = __float2bfloat16(acc0[m]);
                y_out[(size_t)m * N_OUT + row1] = __float2bfloat16(acc1[m]);
            }
        }
    }
}

void launch_dsv41_fp8_gemm(
    const __nv_bfloat16* d_x, const uint8_t* d_w, const uint8_t* d_s,
    __nv_bfloat16* d_y, int T, int N_OUT, int K_DIM, cudaStream_t stream = 0
) {
    int blocks = (N_OUT + 15) / 16; // 16 rows per block!
    if (K_DIM == 5120) {
        if (T <= 1) dsv41_fp8_e8m0_gemm_v1_kernel<1, 5120><<<blocks, 256, 0, stream>>>(d_x, d_w, d_s, d_y, T, N_OUT);
        else        dsv41_fp8_e8m0_gemm_v1_kernel<16, 5120><<<blocks, 256, 0, stream>>>(d_x, d_w, d_s, d_y, T, N_OUT);
    } else if (K_DIM == 1280) {
        if (T <= 1) dsv41_fp8_e8m0_gemm_v1_kernel<1, 1280><<<blocks, 256, 0, stream>>>(d_x, d_w, d_s, d_y, T, N_OUT);
        else        dsv41_fp8_e8m0_gemm_v1_kernel<16, 1280><<<blocks, 256, 0, stream>>>(d_x, d_w, d_s, d_y, T, N_OUT);
    } else if (K_DIM == 8192) {
        if (T <= 1) dsv41_fp8_e8m0_gemm_v1_kernel<1, 8192><<<blocks, 256, 0, stream>>>(d_x, d_w, d_s, d_y, T, N_OUT);
        else        dsv41_fp8_e8m0_gemm_v1_kernel<16, 8192><<<blocks, 256, 0, stream>>>(d_x, d_w, d_s, d_y, T, N_OUT);
    } else if (K_DIM == 6144) {
        if (T <= 1) dsv41_fp8_e8m0_gemm_v1_kernel<1, 6144><<<blocks, 256, 0, stream>>>(d_x, d_w, d_s, d_y, T, N_OUT);
        else        dsv41_fp8_e8m0_gemm_v1_kernel<16, 6144><<<blocks, 256, 0, stream>>>(d_x, d_w, d_s, d_y, T, N_OUT);
    }
}

// ============================================================================
// 5. KERNEL 3: DeepSeek-V4.1 Multi-Head Latent Attention (MLA) Core
//    - 3A: Fused q_norm(1280) + kv_norm(512) + YaRN RoPE on kv[..., 448:512] + Ring KV Write
//    - 3B: Fused Online-Softmax Latent Attention over 128-Window + 512-Compressed KV
//          with 64x Shared-KV Reuse, attn_sink[64], and Inverse-RoPE on Output!
//    - 3C: Fused 8-Group Block-Diagonal Output Projection wo_a [8, 1024, 4096]
// ============================================================================
__global__ void __launch_bounds__(256, 4) dsv41_fused_mla_q_kv_norm_rope_v1_kernel(
    __nv_bfloat16* __restrict__ qr_inout,        // [T, 1280]
    const __nv_bfloat16* __restrict__ q_norm_w,  // [1280]
    __nv_bfloat16* __restrict__ kv_inout,        // [T, 512]
    const __nv_bfloat16* __restrict__ kv_norm_w, // [512]
    __nv_bfloat16* __restrict__ window_kv_cache, // [128, 512]
    int start_pos
) {
    // Grid: T blocks of 256 threads
    int t   = blockIdx.x;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;

    // 1. RMSNorm on qr [1280] (5 elements per thread)
    __nv_bfloat16* qr = qr_inout + (size_t)t * DS_Q_LORA_RANK;
    float sq_q = 0.0f;
    float rq[5];
    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        float v = __bfloat162float(qr[tid + i * 256]);
        rq[i] = v;
        sq_q += v * v;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) sq_q += __shfl_xor_sync(0xffffffff, sq_q, off);
    __shared__ float s_red[8];
    if (lane == 0) s_red[wid] = sq_q;
    __syncthreads();
    if (wid == 0) {
        float v = (lane < 8) ? s_red[lane] : 0.0f;
        #pragma unroll
        for (int off = 4; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off);
        if (lane == 0) s_red[0] = rsqrtf(v * (1.0f / 1280.0f) + 1e-20f);
    }
    __syncthreads();
    float rms_q = s_red[0];
    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        int c = tid + i * 256;
        qr[c] = __float2bfloat16(rq[i] * rms_q * __bfloat162float(q_norm_w[c]));
    }

    // 2. RMSNorm on kv [512] (2 elements per thread = 1 bfloat162 pair) + RoPE on last 64 dims
    __nv_bfloat162* kv2 = reinterpret_cast<__nv_bfloat162*>(kv_inout + (size_t)t * DS_HEAD_DIM);
    const __nv_bfloat162* kvw2 = reinterpret_cast<const __nv_bfloat162*>(kv_norm_w);
    float2 kv_pair = __bfloat1622float2(kv2[tid]);
    float sq_kv = kv_pair.x * kv_pair.x + kv_pair.y * kv_pair.y;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) sq_kv += __shfl_xor_sync(0xffffffff, sq_kv, off);
    if (lane == 0) s_red[wid] = sq_kv;
    __syncthreads();
    if (wid == 0) {
        float v = (lane < 8) ? s_red[lane] : 0.0f;
        #pragma unroll
        for (int off = 4; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off);
        if (lane == 0) s_red[0] = rsqrtf(v * (1.0f / 512.0f) + 1e-20f);
    }
    __syncthreads();
    float rms_kv = s_red[0];
    float2 wf = __bfloat1622float2(kvw2[tid]);
    kv_pair.x = kv_pair.x * rms_kv * wf.x;
    kv_pair.y = kv_pair.y * rms_kv * wf.y;

    // Apply RoPE to tail 64 elements (tid = 224..255 -> pairs 224..255 = dims 448..511)
    if (tid >= 224) {
        int rope_pair = tid - 224; // 0..31
        float freq = float(start_pos + t) * exp2f(-0.5402410118609203f * float(rope_pair));
        float cs = __cosf(freq);
        float sn = __sinf(freq);
        float rx = kv_pair.x * cs - kv_pair.y * sn;
        float ry = kv_pair.x * sn + kv_pair.y * cs;
        kv_pair.x = rx;
        kv_pair.y = ry;
    }
    __nv_bfloat162 out_kv2 = __float22bfloat162_rn(kv_pair);
    kv2[tid] = out_kv2;

    // Write into 128-slot Sliding Window KV Ring Cache
    int slot = (start_pos + t) & (DS_WINDOW_SIZE - 1);
    reinterpret_cast<__nv_bfloat162*>(window_kv_cache + (size_t)slot * DS_HEAD_DIM)[tid] = out_kv2;
}

// Fused Online-Softmax Multi-Head Latent Attention (MLA) with Shared KV Cache + attn_sink + Inverse RoPE
// v5 8-Query-Heads/Block (8x Shared-Memory KV Reuse!) + 128-bit uint4 Cooperative KV Loads + rcp.approx.ftz.f32
__global__ void __launch_bounds__(256, 2) dsv41_fused_mla_sparse_window_attn_v5_kernel(
    const __nv_bfloat16* __restrict__ q_in,      // [T, 64, 512]
    const __nv_bfloat16* __restrict__ kv_cache,  // [N_KV, 512] (Window 128 + Compressed 512)
    const float* __restrict__ attn_sink,         // [64]
    __nv_bfloat16* __restrict__ o_out,           // [T, 64, 512]
    int n_kv_valid,
    int start_pos
) {
    // Grid: (8, T) -> blockIdx.x = 0..7 (8 query heads per block), blockIdx.y = token 0..T-1
    int wid  = threadIdx.x >> 5;     // 0..7 (query head within block)
    int lane = threadIdx.x & 31;     // 0..31 (processes 8 bfloat162 pairs = 16 dims of the 512-dim head)
    int h    = blockIdx.x * 8 + wid; // 0..63
    int t    = blockIdx.y;

    const __nv_bfloat162* q2_ptr = reinterpret_cast<const __nv_bfloat162*>(
        q_in + ((size_t)t * DS_N_HEADS + h) * DS_HEAD_DIM
    );

    float rope_freq = float(start_pos + t) * exp2f(-0.5402410118609203f * float(lane));
    float rope_cs   = __cosf(rope_freq);
    float rope_sn   = __sinf(rope_freq);

    float2 r_q[8];
    float2 r_o[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int c2 = lane + i * 32; // 0..255
        float2 qv = __bfloat1622float2(q2_ptr[c2]);
        // Apply forward RoPE to query tail (i == 7 -> c2 = 224..255 -> dims 448..511)
        if (i == 7) {
            float rx = qv.x * rope_cs - qv.y * rope_sn;
            float ry = qv.x * rope_sn + qv.y * rope_cs;
            qv.x = rx;
            qv.y = ry;
        }
        r_q[i] = qv;
        r_o[i] = make_float2(0.0f, 0.0f);
    }

    __shared__ __nv_bfloat162 s_kv2[8][256 + 8]; // 8 KV positions x 512 dims (with +8 bank-conflict-free pad)
    const float scale = 0.04419417382f; // 1.0f / sqrt(512.0f)

    float m_prev = -1e30f;
    float sum_exp = 0.0f;

    for (int kv_base = 0; kv_base < n_kv_valid; kv_base += 8) {
        int tile_k = (n_kv_valid - kv_base < 8) ? (n_kv_valid - kv_base) : 8;
        if (wid < tile_k) {
            const __nv_bfloat162* kv_src = reinterpret_cast<const __nv_bfloat162*>(
                kv_cache + (size_t)(kv_base + wid) * DS_HEAD_DIM
            );
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                int c2 = lane + i * 32;
                s_kv2[wid][c2] = kv_src[c2];
            }
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            if (k < tile_k) {
                float dot = 0.0f;
                float2 kv_reg[8];
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    int c2 = lane + i * 32;
                    kv_reg[i] = __bfloat1622float2(s_kv2[k][c2]);
                    dot += r_q[i].x * kv_reg[i].x + r_q[i].y * kv_reg[i].y;
                }
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    dot += __shfl_xor_sync(0xffffffff, dot, off);
                }
                float s_val = dot * scale;
                float m_new = fmaxf(m_prev, s_val);
                float corr  = __expf(m_prev - m_new);
                float p_val = __expf(s_val - m_new);
                sum_exp = sum_exp * corr + p_val;
                m_prev  = m_new;

                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    r_o[i].x = r_o[i].x * corr + p_val * kv_reg[i].x;
                    r_o[i].y = r_o[i].y * corr + p_val * kv_reg[i].y;
                }
            }
        }
        __syncthreads();
    }

    // Include learnable attention sink bias: sum_exp += exp(attn_sink[h] - m_prev)
    float sink_bias = attn_sink[h];
    sum_exp += __expf(sink_bias - m_prev);
    float inv_denom = dg_fast_rcp(sum_exp + 1e-20f);

    __nv_bfloat162* o2_ptr = reinterpret_cast<__nv_bfloat162*>(
        o_out + ((size_t)t * DS_N_HEADS + h) * DS_HEAD_DIM
    );

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int c2 = lane + i * 32;
        float2 ov;
        ov.x = r_o[i].x * inv_denom;
        ov.y = r_o[i].y * inv_denom;
        // Apply Inverse RoPE (conjugate rotation) to output tail (i == 7 -> dims 448..511)
        if (i == 7) {
            float rx = ov.x * rope_cs + ov.y * rope_sn;
            float ry = -ov.x * rope_sn + ov.y * rope_cs;
            ov.x = rx;
            ov.y = ry;
        }
        o2_ptr[c2] = __float22bfloat162_rn(ov);
    }
}

// Fused 8-Group Block-Diagonal Output Projection wo_a [8, 1024, 4096] (v5 2-Rows/Warp = 16 Rows/Block!)
template <int M_CAP>
__global__ void __launch_bounds__(256, 2) dsv41_fused_mla_grouped_wo_a_v1_kernel(
    const __nv_bfloat16* __restrict__ o_in, // [T, 8, 4096] = [T, 32768]
    const uint8_t* __restrict__ wo_a_fp8,   // [8, 1024, 4096] = [8192, 4096]
    const uint8_t* __restrict__ wo_a_s,     // [256, 128] (E8M0 32x32 block scale)
    __nv_bfloat16* __restrict__ o_lora_out, // [T, 8192]
    int m_valid
) {
    // Grid: (64, 8) -> blockIdx.x = 0..63 (16 rows/block = 1024 rows/group), blockIdx.y = group 0..7
    int wid  = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int grp  = blockIdx.y; // 0..7
    int r_in_grp0 = blockIdx.x * 16 + wid * 2 + 0; // 0..1022
    int row0 = grp * DS_O_LORA_RANK + r_in_grp0;
    int row1 = row0 + 1;

    constexpr int TILE_HALF_K = 256;
    constexpr int TILE_STRIDE = 288;
    __shared__ __nv_bfloat162 s_o2[M_CAP][TILE_STRIDE];

    float acc0[M_CAP], acc1[M_CAP];
    #pragma unroll
    for (int m = 0; m < M_CAP; ++m) { acc0[m] = 0.0f; acc1[m] = 0.0f; }

    const uint4* w0_u4  = reinterpret_cast<const uint4*>(wo_a_fp8 + (size_t)row0 * DS_GROUP_IN_DIM);
    const uint4* w1_u4  = reinterpret_cast<const uint4*>(wo_a_fp8 + (size_t)row1 * DS_GROUP_IN_DIM);
    const uint8_t* s_r  = wo_a_s + (size_t)(row0 >> 5) * (DS_GROUP_IN_DIM >> 5);

    for (int t = 0; t < 8; ++t) { // 4096 / 512 = 8 tiles
        int k2_base = t * TILE_HALF_K;
        for (int idx = threadIdx.x; idx < m_valid * TILE_HALF_K; idx += 256) {
            int m = idx >> 8;
            int c = idx & 255;
            const __nv_bfloat162* src = reinterpret_cast<const __nv_bfloat162*>(
                o_in + ((size_t)m * DS_O_GROUPS + grp) * DS_GROUP_IN_DIM
            );
            s_o2[m][c + (c >> 3)] = src[k2_base + c];
        }
        __syncthreads();

        int u4_idx = t * 32 + lane;
        uint4 pk0 = w0_u4[u4_idx];
        uint4 pk1 = w1_u4[u4_idx];
        float blk_scale = decode_fp8_e8m0_fast(s_r[u4_idx >> 1]);
        uint32_t words0[4] = {pk0.x, pk0.y, pk0.z, pk0.w};
        uint32_t words1[4] = {pk1.x, pk1.y, pk1.z, pk1.w};
        int swz_base = lane * 9;

        float sub0[M_CAP], sub1[M_CAP];
        #pragma unroll
        for (int m = 0; m < M_CAP; ++m) { sub0[m] = 0.0f; sub1[m] = 0.0f; }

        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            uint32_t p0 = words0[w];
            uint32_t p1 = words1[w];
            float a0 = decode_fp8_e4m3_fast(p0 & 0xFF);
            float a1 = decode_fp8_e4m3_fast((p0 >> 8) & 0xFF);
            float a2 = decode_fp8_e4m3_fast((p0 >> 16) & 0xFF);
            float a3 = decode_fp8_e4m3_fast((p0 >> 24) & 0xFF);

            float b0 = decode_fp8_e4m3_fast(p1 & 0xFF);
            float b1 = decode_fp8_e4m3_fast((p1 >> 8) & 0xFF);
            float b2 = decode_fp8_e4m3_fast((p1 >> 16) & 0xFF);
            float b3 = decode_fp8_e4m3_fast((p1 >> 24) & 0xFF);

            int c2_0 = swz_base + w * 2 + 0;
            int c2_1 = swz_base + w * 2 + 1;
            #pragma unroll
            for (int m = 0; m < M_CAP; ++m) {
                float2 x01 = __bfloat1622float2(s_o2[m][c2_0]);
                float2 x23 = __bfloat1622float2(s_o2[m][c2_1]);
                sub0[m] += a0 * x01.x + a1 * x01.y + a2 * x23.x + a3 * x23.y;
                sub1[m] += b0 * x01.x + b1 * x01.y + b2 * x23.x + b3 * x23.y;
            }
        }
        #pragma unroll
        for (int m = 0; m < M_CAP; ++m) {
            acc0[m] += sub0[m] * blk_scale;
            acc1[m] += sub1[m] * blk_scale;
        }
        __syncthreads();
    }

    #pragma unroll
    for (int m = 0; m < M_CAP; ++m) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc0[m] += __shfl_xor_sync(0xffffffff, acc0[m], off);
            acc1[m] += __shfl_xor_sync(0xffffffff, acc1[m], off);
        }
        if (lane == 0 && m < m_valid) {
            o_lora_out[(size_t)m * DS_O_TOTAL_RANK + row0] = __float2bfloat16(acc0[m]);
            o_lora_out[(size_t)m * DS_O_TOTAL_RANK + row1] = __float2bfloat16(acc1[m]);
        }
    }
}

// ============================================================================
// 6. KERNEL 4 (v5): 48-SM Parallel DeepSeek-V4.1 sqrtsoftplus + noaux_tc Top-6 Router
//    with DeepGEMM `sort4_best_first` 5-Comparator Sorting Network!
// ============================================================================
__global__ void __launch_bounds__(256, 2) dsv41_router_scores_48blk_v3_kernel(
    const __nv_bfloat16* __restrict__ x_norm,    // [T, 5120]
    const __nv_bfloat16* __restrict__ gate_w,    // [384, 5120]
    const float* __restrict__ gate_bias,         // [384]
    const float* __restrict__ lookahead_bias,    // [T, 384]
    float* __restrict__ raw_scores_out,          // [T, 384]
    float* __restrict__ biased_scores_out        // [T, 384]
) {
    // Grid: (48, T) -> blockIdx.x = 0..47 (8 warps = 8 experts per block), blockIdx.y = token 0..T-1
    int wid  = threadIdx.x >> 5;   // 0..7
    int lane = threadIdx.x & 31;
    int e    = blockIdx.x * 8 + wid; // 0..383
    int t    = blockIdx.y;

    __shared__ __nv_bfloat162 s_x2[DS_HALF_DIM]; // 10 KiB
    const __nv_bfloat162* x2_in = reinterpret_cast<const __nv_bfloat162*>(x_norm + (size_t)t * DS_DIM);
    for (int c2 = threadIdx.x; c2 < DS_HALF_DIM; c2 += 256) {
        s_x2[c2] = x2_in[c2];
    }
    __syncthreads();

    const __nv_bfloat162* gw2 = reinterpret_cast<const __nv_bfloat162*>(gate_w + (size_t)e * DS_DIM);
    float dot = 0.0f;
    #pragma unroll 8
    for (int k = 0; k < 80; ++k) {
        int c2 = lane + k * 32;
        float2 xv = __bfloat1622float2(s_x2[c2]);
        float2 gw = __bfloat1622float2(gw2[c2]);
        dot += xv.x * gw.x + xv.y * gw.y;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffff, dot, off);
    if (lane == 0) {
        float sp = (dot > 20.0f) ? dot : log1pf(__expf(dot));
        float sc = sqrtf(fmaxf(sp, 0.0f));
        float b  = gate_bias[e] + (lookahead_bias ? lookahead_bias[t * DS_ROUTED_EXPERTS + e] : 0.0f);
        raw_scores_out[t * DS_ROUTED_EXPERTS + e]    = sc;
        biased_scores_out[t * DS_ROUTED_EXPERTS + e] = sc + b;
    }
}

__global__ void __launch_bounds__(32, 4) dsv41_router_top6_v5_kernel(
    const float* __restrict__ raw_scores_in,    // [T, 384]
    const float* __restrict__ biased_scores_in, // [T, 384]
    int* __restrict__ top_indices,              // [T, 6]
    float* __restrict__ top_weights             // [T, 6]
) {
    // Grid: T blocks of 32 threads (1 warp per token)
    // Uses DeepGEMM's 5-comparator sorting network (dg_sort4_best_first) on each of the 3 waves of 4 experts!
    int t    = blockIdx.x;
    int lane = threadIdx.x; // 0..31

    float r_raw[12];
    float ws[3][4];
    int   wi[3][4];

    #pragma unroll
    for (int w = 0; w < 3; ++w) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int r = w * 4 + j;
            int e = lane + r * 32;
            r_raw[r] = raw_scores_in[t * DS_ROUTED_EXPERTS + e];
            ws[w][j] = biased_scores_in[t * DS_ROUTED_EXPERTS + e];
            wi[w][j] = e;
        }
        dg_sort4_best_first(ws[w][0], wi[w][0], ws[w][1], wi[w][1], ws[w][2], wi[w][2], ws[w][3], wi[w][3]);
    }

    float w_sel[DS_TOP_K];
    int   i_sel[DS_TOP_K];
    float sum_unbiased = 0.0f;

    #pragma unroll
    for (int k = 0; k < DS_TOP_K; ++k) {
        // After dg_sort4_best_first, the head (ws[w][0], wi[w][0]) of each wave is guaranteed to be that wave's max!
        float best_b = ws[0][0];
        int   best_e = wi[0][0];
        int   best_w = 0;
        if (ws[1][0] > best_b || (ws[1][0] == best_b && wi[1][0] < best_e)) {
            best_b = ws[1][0]; best_e = wi[1][0]; best_w = 1;
        }
        if (ws[2][0] > best_b || (ws[2][0] == best_b && wi[2][0] < best_e)) {
            best_b = ws[2][0]; best_e = wi[2][0]; best_w = 2;
        }

        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float ob = __shfl_xor_sync(0xffffffff, best_b, off);
            int   oe = __shfl_xor_sync(0xffffffff, best_e, off);
            if (ob > best_b || (ob == best_b && oe < best_e)) {
                best_b = ob;
                best_e = oe;
            }
        }
        int win_e    = __shfl_sync(0xffffffff, best_e, 0);
        int win_lane = win_e & 31;
        int win_r    = win_e >> 5;
        float win_raw = 0.0f;
        if (lane == win_lane) {
            #pragma unroll
            for (int r = 0; r < 12; ++r) {
                if (r == win_r) win_raw = r_raw[r];
            }
            // Shift the winning wave's sorted queue forward by 1!
            #pragma unroll
            for (int w = 0; w < 3; ++w) {
                if (w == best_w) {
                    ws[w][0] = ws[w][1]; wi[w][0] = wi[w][1];
                    ws[w][1] = ws[w][2]; wi[w][1] = wi[w][2];
                    ws[w][2] = ws[w][3]; wi[w][2] = wi[w][3];
                    ws[w][3] = -1e30f;
                }
            }
        }
        win_raw = __shfl_sync(0xffffffff, win_raw, win_lane);
        if (lane == 0) {
            i_sel[k] = win_e;
            w_sel[k] = win_raw;
            sum_unbiased += win_raw;
        }
    }
    if (lane == 0) {
        float inv_s = 1.5f * dg_fast_rcp(sum_unbiased + 1e-20f);
        #pragma unroll
        for (int k = 0; k < DS_TOP_K; ++k) {
            top_indices[t * DS_TOP_K + k] = i_sel[k];
            top_weights[t * DS_TOP_K + k] = w_sel[k] * inv_s;
        }
    }
}

// ============================================================================
// 7. KERNEL 5 (v5): K-Tiled M_CAP = 1, 2, 4, 8 Zero-Spill SM120 NVFP4 Routed MoE
//    with DeepGEMM + DeepEP `ld.L1::no_allocate.L2::cache_hint` (evict_first)
//    and Predicated `@p ld` on Stage 2 tail tiles!
// ============================================================================
template <int M_CAP>
__global__ void __launch_bounds__(256, 2) dsv41_grouped_nvfp4_moe_stage1_clamped_swiglu_v1_kernel(
    const __nv_bfloat16* __restrict__ x_tokens,       // [T, 5120]
    const Dsv41ExpertGroupTask* __restrict__ groups,  // [num_groups]
    const Dsv41Mcl4cExpertPacket* __restrict__ arena, // [num_slots]
    float* __restrict__ inter_buf                     // [num_groups, M_CAP, 2304]
) {
    // Grid: (288, num_groups) -> 288 blocks x 8 warps = 2304 intermediate rows!
    int wid   = threadIdx.x >> 5;
    int lane  = threadIdx.x & 31;
    int row   = blockIdx.x * 8 + wid; // 0..2303
    int g_idx = blockIdx.y;

    const Dsv41ExpertGroupTask* __restrict__ task = &groups[g_idx];
    int m_valid = task->m_count;
    if (m_valid > M_CAP) m_valid = M_CAP;

    constexpr int TILE_HALF_K = 512;
    constexpr int TILE_STRIDE = 544;
    __shared__ __nv_bfloat162 s_x2[M_CAP][TILE_STRIDE];

    float my_lut_val = get_nvfp4_lut_lane_val(lane);
    const Dsv41Mcl4cExpertPacket* pkt = &arena[task->arena_slot];

    const uint4* w1_u4 = reinterpret_cast<const uint4*>(pkt->w1_w3_fp4 + (size_t)row * DS_HALF_DIM);
    const uint4* w3_u4 = reinterpret_cast<const uint4*>(pkt->w1_w3_fp4 + (size_t)(row + DS_MOE_INTER) * DS_HALF_DIM);
    const uint16_t* s1_u16 = reinterpret_cast<const uint16_t*>(pkt->w1_w3_scales + (size_t)row * (DS_DIM / 16));
    const uint16_t* s3_u16 = reinterpret_cast<const uint16_t*>(pkt->w1_w3_scales + (size_t)(row + DS_MOE_INTER) * (DS_DIM / 16));

    float acc_gate[M_CAP];
    float acc_up[M_CAP];
    #pragma unroll
    for (int m = 0; m < M_CAP; ++m) {
        acc_gate[m] = 0.0f;
        acc_up[m]   = 0.0f;
    }

    for (int t = 0; t < 5; ++t) {
        int k2_base = t * TILE_HALF_K;
        for (int idx = threadIdx.x; idx < m_valid * TILE_HALF_K; idx += 256) {
            int m  = idx >> 9;
            int c2 = idx & 511;
            int tok_id = task->token_ids[m];
            const __nv_bfloat162* src = reinterpret_cast<const __nv_bfloat162*>(
                x_tokens + (size_t)tok_id * DS_DIM
            );
            s_x2[m][c2 + (c2 >> 4)] = src[k2_base + c2];
        }
        __syncthreads();

        int u4_idx = t * 32 + lane; // 0..159
        uint4 g_pack = dg_ld_evict_first_u4(&w1_u4[u4_idx]);
        uint4 u_pack = dg_ld_evict_first_u4(&w3_u4[u4_idx]);
        uint16_t gs_pair = s1_u16[u4_idx];
        uint16_t us_pair = s3_u16[u4_idx];

        float gs_arr[2] = {decode_fp8_e4m3_fast(gs_pair & 0xFF), decode_fp8_e4m3_fast(gs_pair >> 8)};
        float us_arr[2] = {decode_fp8_e4m3_fast(us_pair & 0xFF), decode_fp8_e4m3_fast(us_pair >> 8)};

        uint32_t gw[4] = {g_pack.x, g_pack.y, g_pack.z, g_pack.w};
        uint32_t uw[4] = {u_pack.x, u_pack.y, u_pack.z, u_pack.w};
        int swz_base = lane * 17;

        #pragma unroll
        for (int grp16 = 0; grp16 < 2; ++grp16) {
            float sub_g[M_CAP], sub_u[M_CAP];
            #pragma unroll
            for (int m = 0; m < M_CAP; ++m) { sub_g[m] = 0.0f; sub_u[m] = 0.0f; }

            #pragma unroll
            for (int w_local = 0; w_local < 2; ++w_local) {
                int w = grp16 * 2 + w_local;
                uint32_t g_word = gw[w];
                uint32_t u_word = uw[w];
                #pragma unroll
                for (int b = 0; b < 4; ++b) {
                    int c2 = swz_base + w * 4 + b;
                    uint32_t gb = (g_word >> (b * 8)) & 0xFF;
                    uint32_t ub = (u_word >> (b * 8)) & 0xFF;

                    float g0 = __shfl_sync(0xffffffff, my_lut_val, gb & 0x0F);
                    float g1 = __shfl_sync(0xffffffff, my_lut_val, gb >> 4);
                    float u0 = __shfl_sync(0xffffffff, my_lut_val, ub & 0x0F);
                    float u1 = __shfl_sync(0xffffffff, my_lut_val, ub >> 4);

                    #pragma unroll
                    for (int m = 0; m < M_CAP; ++m) {
                        float2 xv = __bfloat1622float2(s_x2[m][c2]);
                        sub_g[m] += g0 * xv.x + g1 * xv.y;
                        sub_u[m] += u0 * xv.x + u1 * xv.y;
                    }
                }
            }
            float gs = gs_arr[grp16];
            float us = us_arr[grp16];
            #pragma unroll
            for (int m = 0; m < M_CAP; ++m) {
                acc_gate[m] += sub_g[m] * gs;
                acc_up[m]   += sub_u[m] * us;
            }
        }
        __syncthreads();
    }

    float g_s2 = pkt->w1_scale2;
    float u_s2 = pkt->w3_scale2;

    #pragma unroll
    for (int m = 0; m < M_CAP; ++m) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc_gate[m] += __shfl_xor_sync(0xffffffff, acc_gate[m], off);
            acc_up[m]   += __shfl_xor_sync(0xffffffff, acc_up[m], off);
        }
        if (lane == 0 && m < m_valid) {
            float gate_val = fminf(acc_gate[m] * g_s2, 10.0f);
            float up_val   = fminf(fmaxf(acc_up[m] * u_s2, -10.0f), 10.0f);
            float silu_g   = gate_val / (1.0f + __expf(-gate_val));
            inter_buf[((size_t)g_idx * M_CAP + m) * DS_MOE_INTER + row] = silu_g * up_val;
        }
    }
}

template <int M_CAP>
__global__ void __launch_bounds__(256, 2) dsv41_grouped_nvfp4_moe_stage2_down_v1_kernel(
    const float* __restrict__ inter_buf,              // [num_groups, M_CAP, 2304]
    const Dsv41ExpertGroupTask* __restrict__ groups,  // [num_groups]
    const Dsv41Mcl4cExpertPacket* __restrict__ arena, // [num_slots]
    float* __restrict__ out_accum                     // [T, 5120]
) {
    // Grid: (320, num_groups) -> 320 blocks x 8 warps x 2 rows/warp = 5120 output rows!
    int wid   = threadIdx.x >> 5;
    int lane  = threadIdx.x & 31;
    int row0  = blockIdx.x * 16 + wid * 2 + 0; // 0..5118
    int row1  = row0 + 1;                      // 1..5119
    int g_idx = blockIdx.y;

    const Dsv41ExpertGroupTask* __restrict__ task = &groups[g_idx];
    int m_valid = task->m_count;
    if (m_valid > M_CAP) m_valid = M_CAP;

    constexpr int TILE_HALF_K = 512;
    constexpr int TILE_STRIDE = 544;
    __shared__ __nv_bfloat162 s_inter2[M_CAP][TILE_STRIDE];

    float my_lut_val = get_nvfp4_lut_lane_val(lane);
    const Dsv41Mcl4cExpertPacket* pkt = &arena[task->arena_slot];

    const uint4* w2_0_u4     = reinterpret_cast<const uint4*>(pkt->w2_fp4 + (size_t)row0 * DS_HALF_INTER);
    const uint4* w2_1_u4     = reinterpret_cast<const uint4*>(pkt->w2_fp4 + (size_t)row1 * DS_HALF_INTER);
    const uint16_t* s2_0_u16 = reinterpret_cast<const uint16_t*>(pkt->w2_scales + (size_t)row0 * (DS_MOE_INTER / 16));
    const uint16_t* s2_1_u16 = reinterpret_cast<const uint16_t*>(pkt->w2_scales + (size_t)row1 * (DS_MOE_INTER / 16));

    float acc0[M_CAP], acc1[M_CAP];
    #pragma unroll
    for (int m = 0; m < M_CAP; ++m) {
        acc0[m] = 0.0f;
        acc1[m] = 0.0f;
    }

    for (int t = 0; t < 3; ++t) {
        int k2_base = t * TILE_HALF_K;
        int tile_pairs = (t < 2) ? 512 : 128;
        for (int idx = threadIdx.x; idx < m_valid * tile_pairs; idx += 256) {
            int m  = (t < 2) ? (idx >> 9) : (idx >> 7);
            int c2 = (t < 2) ? (idx & 511) : (idx & 127);
            const float2* src = reinterpret_cast<const float2*>(
                inter_buf + ((size_t)g_idx * M_CAP + m) * DS_MOE_INTER
            );
            s_inter2[m][c2 + (c2 >> 4)] = __float22bfloat162_rn(src[k2_base + c2]);
        }
        __syncthreads();

        int u4_idx = t * 32 + lane; // 0..95 (valid: 0..71)
        int valid_u4 = (u4_idx < 72) ? 1 : 0;
        uint4 d0_pack = dg_ld_pred_evict_first_u4(&w2_0_u4[u4_idx], valid_u4);
        uint4 d1_pack = dg_ld_pred_evict_first_u4(&w2_1_u4[u4_idx], valid_u4);
        uint16_t ds0_pair = valid_u4 ? s2_0_u16[u4_idx] : 0;
        uint16_t ds1_pair = valid_u4 ? s2_1_u16[u4_idx] : 0;
        float ds0_arr[2] = {
            valid_u4 ? decode_fp8_e4m3_fast(ds0_pair & 0xFF) : 0.0f,
            valid_u4 ? decode_fp8_e4m3_fast(ds0_pair >> 8)   : 0.0f
        };
        float ds1_arr[2] = {
            valid_u4 ? decode_fp8_e4m3_fast(ds1_pair & 0xFF) : 0.0f,
            valid_u4 ? decode_fp8_e4m3_fast(ds1_pair >> 8)   : 0.0f
        };
        uint32_t dw0[4] = {d0_pack.x, d0_pack.y, d0_pack.z, d0_pack.w};
        uint32_t dw1[4] = {d1_pack.x, d1_pack.y, d1_pack.z, d1_pack.w};
        int swz_base = (valid_u4 ? lane : 0) * 17;

        #pragma unroll
        for (int grp16 = 0; grp16 < 2; ++grp16) {
            float sub_d0[M_CAP], sub_d1[M_CAP];
            #pragma unroll
            for (int m = 0; m < M_CAP; ++m) {
                sub_d0[m] = 0.0f;
                sub_d1[m] = 0.0f;
            }

            #pragma unroll
            for (int w_local = 0; w_local < 2; ++w_local) {
                int w = grp16 * 2 + w_local;
                uint32_t d0_word = dw0[w];
                uint32_t d1_word = dw1[w];
                #pragma unroll
                for (int b = 0; b < 4; ++b) {
                    int c2 = swz_base + w * 4 + b;
                    uint32_t db0 = (d0_word >> (b * 8)) & 0xFF;
                    uint32_t db1 = (d1_word >> (b * 8)) & 0xFF;
                    // All 32 lanes participate in __shfl_sync unconditionally!
                    float a0 = __shfl_sync(0xffffffff, my_lut_val, db0 & 0x0F);
                    float a1 = __shfl_sync(0xffffffff, my_lut_val, db0 >> 4);
                    float b0 = __shfl_sync(0xffffffff, my_lut_val, db1 & 0x0F);
                    float b1 = __shfl_sync(0xffffffff, my_lut_val, db1 >> 4);
                    #pragma unroll
                    for (int m = 0; m < M_CAP; ++m) {
                        float2 iv = __bfloat1622float2(s_inter2[m][c2]);
                        sub_d0[m] += a0 * iv.x + a1 * iv.y;
                        sub_d1[m] += b0 * iv.x + b1 * iv.y;
                    }
                }
            }
            float ds0 = ds0_arr[grp16];
            float ds1 = ds1_arr[grp16];
            #pragma unroll
            for (int m = 0; m < M_CAP; ++m) {
                acc0[m] += sub_d0[m] * ds0;
                acc1[m] += sub_d1[m] * ds1;
            }
        }
        __syncthreads();
    }

    float d_s2 = pkt->w2_scale2;
    #pragma unroll
    for (int m = 0; m < M_CAP; ++m) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc0[m] += __shfl_xor_sync(0xffffffff, acc0[m], off);
            acc1[m] += __shfl_xor_sync(0xffffffff, acc1[m], off);
        }
        if (lane == 0 && m < m_valid) {
            int tok_id = task->token_ids[m];
            float w_tok = task->weights[m];
            atomicAdd(&out_accum[(size_t)tok_id * DS_DIM + row0], acc0[m] * d_s2 * w_tok);
            atomicAdd(&out_accum[(size_t)tok_id * DS_DIM + row1], acc1[m] * d_s2 * w_tok);
        }
    }
}

void launch_dsv41_grouped_moe(
    int max_m, int num_groups,
    const __nv_bfloat16* d_x_tokens,
    const Dsv41ExpertGroupTask* d_groups,
    const Dsv41Mcl4cExpertPacket* d_arena,
    float* d_inter, float* d_out, cudaStream_t stream = 0
) {
    dim3 g1(288, num_groups);
    dim3 g2(320, num_groups);
    if (max_m <= 1) {
        dsv41_grouped_nvfp4_moe_stage1_clamped_swiglu_v1_kernel<1><<<g1, 256, 0, stream>>>(d_x_tokens, d_groups, d_arena, d_inter);
        dsv41_grouped_nvfp4_moe_stage2_down_v1_kernel<1><<<g2, 256, 0, stream>>>(d_inter, d_groups, d_arena, d_out);
    } else if (max_m <= 2) {
        dsv41_grouped_nvfp4_moe_stage1_clamped_swiglu_v1_kernel<2><<<g1, 256, 0, stream>>>(d_x_tokens, d_groups, d_arena, d_inter);
        dsv41_grouped_nvfp4_moe_stage2_down_v1_kernel<2><<<g2, 256, 0, stream>>>(d_inter, d_groups, d_arena, d_out);
    } else if (max_m <= 4) {
        dsv41_grouped_nvfp4_moe_stage1_clamped_swiglu_v1_kernel<4><<<g1, 256, 0, stream>>>(d_x_tokens, d_groups, d_arena, d_inter);
        dsv41_grouped_nvfp4_moe_stage2_down_v1_kernel<4><<<g2, 256, 0, stream>>>(d_inter, d_groups, d_arena, d_out);
    } else {
        dsv41_grouped_nvfp4_moe_stage1_clamped_swiglu_v1_kernel<8><<<g1, 256, 0, stream>>>(d_x_tokens, d_groups, d_arena, d_inter);
        dsv41_grouped_nvfp4_moe_stage2_down_v1_kernel<8><<<g2, 256, 0, stream>>>(d_inter, d_groups, d_arena, d_out);
    }
}

// ============================================================================
// 8. KERNEL 6: Isolated-Compute NCM(64-CB) + PLE(r=64) + LoRA(r=16) + 384-Expert
//    Lookahead + Single-Launch float4 Register-AdamW at H = 5120 (DeepSeek-V4.1)
// ============================================================================
__global__ void __launch_bounds__(256, 2) dsv41_ncm_ple_lora_fwd_enc_batched_v1_kernel(
    const __nv_bfloat16* __restrict__ x_in, // [16, 5120]
    const float* __restrict__ w_enc,        // [256, 5120]
    const float* __restrict__ lora_a,       // [16, 5120]
    float* __restrict__ z_out,              // [16, 256]
    float* __restrict__ saved_hlora         // [16, 16]
) {
    // Grid: (34, 8) -> 34 blocks x 8 warps = 272 rows (256 W_enc + 16 LoRA_A), 8 tiles of 2 tokens
    int wid    = threadIdx.x >> 5;
    int lane   = threadIdx.x & 31;
    int row_id = blockIdx.x * 8 + wid; // 0..271
    int t_base = blockIdx.y * 2;       // 2 tokens per tile -> 2 * 2560 * 4B = 20 KiB smem!

    __shared__ __nv_bfloat162 s_x2[2][DS_HALF_DIM]; // 20 KiB

    const __nv_bfloat162* x2_in = reinterpret_cast<const __nv_bfloat162*>(x_in);
    for (int idx = threadIdx.x; idx < 2 * DS_HALF_DIM; idx += 256) {
        int m  = idx / DS_HALF_DIM;
        int c2 = idx % DS_HALF_DIM;
        s_x2[m][c2] = x2_in[(t_base + m) * DS_HALF_DIM + c2];
    }
    __syncthreads();

    const float2* w2_row = (row_id < 256)
        ? reinterpret_cast<const float2*>(w_enc  + (size_t)row_id * DS_DIM)
        : reinterpret_cast<const float2*>(lora_a + (size_t)(row_id - 256) * DS_DIM);

    float acc0 = 0.0f, acc1 = 0.0f;
    // 2560 float2s / 32 lanes = 80 iterations
    #pragma unroll 8
    for (int k = 0; k < 80; ++k) {
        int c2 = lane + k * 32;
        float2 wv = w2_row[c2];
        float2 x0 = __bfloat1622float2(s_x2[0][c2]);
        float2 x1 = __bfloat1622float2(s_x2[1][c2]);
        acc0 += wv.x * x0.x + wv.y * x0.y;
        acc1 += wv.x * x1.x + wv.y * x1.y;
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc0 += __shfl_xor_sync(0xffffffff, acc0, off);
        acc1 += __shfl_xor_sync(0xffffffff, acc1, off);
    }

    if (lane == 0) {
        if (row_id < 256) {
            z_out[(t_base + 0) * 256 + row_id] = acc0;
            z_out[(t_base + 1) * 256 + row_id] = acc1;
        } else {
            int lr = row_id - 256;
            saved_hlora[(t_base + 0) * 16 + lr] = acc0;
            saved_hlora[(t_base + 1) * 16 + lr] = acc1;
        }
    }
}

__global__ void __launch_bounds__(256, 2) dsv41_ncm_codebook_lookahead384_fwd_v1_kernel(
    const float* __restrict__ z_in,        // [16, 256]
    const float* __restrict__ e_ncm,       // [64, 256]
    const float* __restrict__ w_look,      // [384, 256] (384 experts!)
    const float* __restrict__ u_ple_table, // [4096, 64]
    float* __restrict__ ncm_state,         // [16, 256]
    float* __restrict__ saved_alpha,       // [16, 64]
    float* __restrict__ saved_uple,        // [16, 64]
    float* __restrict__ next_router_bias   // [16, 384]
) {
    int t    = blockIdx.x; // 0..15
    int tid  = threadIdx.x;
    int wid  = tid >> 5;
    int lane = tid & 31;

    __shared__ float s_z[256];
    __shared__ float s_alpha[64];
    __shared__ float s_c[256];

    s_z[tid] = z_in[t * 256 + tid];
    __syncthreads();

    for (int k = wid; k < 64; k += 8) {
        float dot = 0.0f;
        const float* e_row = e_ncm + k * 256;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int d = lane + i * 32;
            dot += s_z[d] * e_row[d];
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffff, dot, off);
        if (lane == 0) s_alpha[k] = dot * 0.25f;
    }
    __syncthreads();

    if (wid == 0) {
        float v0 = s_alpha[lane];
        float v1 = s_alpha[lane + 32];
        float m_val = fmaxf(v0, v1);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) m_val = fmaxf(m_val, __shfl_xor_sync(0xffffffff, m_val, off));
        v0 = __expf(v0 - m_val);
        v1 = __expf(v1 - m_val);
        float s_val = v0 + v1;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) s_val += __shfl_xor_sync(0xffffffff, s_val, off);
        float inv_s = 1.0f / (s_val + 1e-12f);
        s_alpha[lane]      = v0 * inv_s;
        s_alpha[lane + 32] = v1 * inv_s;
        saved_alpha[t * 64 + lane]      = v0 * inv_s;
        saved_alpha[t * 64 + lane + 32] = v1 * inv_s;
    }
    __syncthreads();

    float q_d = 0.0f;
    #pragma unroll 8
    for (int k = 0; k < 64; ++k) {
        q_d += s_alpha[k] * e_ncm[k * 256 + tid];
    }
    float c_prev = (t > 0) ? ncm_state[(t - 1) * 256 + tid] : 0.0f;
    float c_new  = 0.85f * c_prev + 0.15f * q_d;
    s_c[tid] = c_new;
    ncm_state[t * 256 + tid] = c_new;

    if (tid < 64) {
        saved_uple[t * 64 + tid] = u_ple_table[t * 64 + tid];
    }
    __syncthreads();

    // Predict 384-Expert Next-Layer Router Lookahead Bias (384 / 256 = 1.5 rows per thread, or 48 rows per warp!)
    for (int e = wid; e < 384; e += 8) {
        float acc = 0.0f;
        const float* wl = w_look + e * 256;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int d = lane + i * 32;
            acc += wl[d] * s_c[d];
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, off);
        if (lane == 0) next_router_bias[t * 384 + e] = acc;
    }
}

__global__ void __launch_bounds__(256, 1) dsv41_ncm_ple_lora_fwd_dec_batched_v1_kernel(
    const float* __restrict__ ncm_state,   // [16, 256]
    const float* __restrict__ saved_uple,  // [16, 64]
    const float* __restrict__ saved_hlora, // [16, 16]
    const float* __restrict__ w_dec,       // [256, 5120]
    const float* __restrict__ v_ple,       // [64, 5120]
    const float* __restrict__ lora_b,      // [16, 5120]
    float* __restrict__ y_out              // [16, 5120]
) {
    // Grid: 20 blocks of 256 threads (20 * 256 = 5120 float4s / 4 = 1280 float4 columns -> 5 blocks of 256 threads!)
    int col4 = blockIdx.x * 256 + threadIdx.x; // 0..1279 (1280 * 4 = 5120 cols)
    int tid  = threadIdx.x;

    __shared__ float s_c[16][256];
    __shared__ float s_u[16][64];
    __shared__ float s_h[16][16];

    #pragma unroll
    for (int t = 0; t < 16; ++t) {
        s_c[t][tid] = ncm_state[t * 256 + tid];
        if (tid < 64) s_u[t][tid] = saved_uple[t * 64 + tid];
        if (tid < 16) s_h[t][tid] = saved_hlora[t * 16 + tid];
    }
    __syncthreads();

    float4 acc[16];
    #pragma unroll
    for (int t = 0; t < 16; ++t) acc[t] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    const float4* w_dec4 = reinterpret_cast<const float4*>(w_dec);
    #pragma unroll 8
    for (int j = 0; j < 256; ++j) {
        float4 w4 = w_dec4[(size_t)j * 1280 + col4];
        #pragma unroll
        for (int t = 0; t < 16; ++t) {
            float c = s_c[t][j];
            acc[t].x += c * w4.x; acc[t].y += c * w4.y; acc[t].z += c * w4.z; acc[t].w += c * w4.w;
        }
    }

    const float4* v_ple4 = reinterpret_cast<const float4*>(v_ple);
    #pragma unroll 8
    for (int r = 0; r < 64; ++r) {
        float4 v4 = v_ple4[(size_t)r * 1280 + col4];
        #pragma unroll
        for (int t = 0; t < 16; ++t) {
            float u = s_u[t][r];
            acc[t].x += u * v4.x; acc[t].y += u * v4.y; acc[t].z += u * v4.z; acc[t].w += u * v4.w;
        }
    }

    const float4* lb4 = reinterpret_cast<const float4*>(lora_b);
    #pragma unroll
    for (int r = 0; r < 16; ++r) {
        float4 b4 = lb4[(size_t)r * 1280 + col4];
        #pragma unroll
        for (int t = 0; t < 16; ++t) {
            float h = s_h[t][r];
            acc[t].x += h * b4.x; acc[t].y += h * b4.y; acc[t].z += h * b4.z; acc[t].w += h * b4.w;
        }
    }

    float4* y4_out = reinterpret_cast<float4*>(y_out);
    #pragma unroll
    for (int t = 0; t < 16; ++t) {
        float4 cur = y4_out[(size_t)t * 1280 + col4];
        cur.x += acc[t].x; cur.y += acc[t].y; cur.z += acc[t].z; cur.w += acc[t].w;
        y4_out[(size_t)t * 1280 + col4] = cur;
    }
}

__global__ void __launch_bounds__(256, 2) dsv41_ncm_ple_lora_bwd_proj_batched_v1_kernel(
    const float* __restrict__ dy_in,  // [16, 5120]
    const float* __restrict__ w_dec,  // [256, 5120]
    const float* __restrict__ v_ple,  // [64, 5120]
    const float* __restrict__ lora_b, // [16, 5120]
    float* __restrict__ dc_out,       // [16, 256]
    float* __restrict__ duple_out,    // [16, 64]
    float* __restrict__ dhlora_out    // [16, 16]
) {
    // Grid: (42, 8) -> 42 blocks x 8 warps = 336 rows, 8 tiles of 2 tokens (2 * 5120 * 4B = 40 KiB smem)
    int wid    = threadIdx.x >> 5;
    int lane   = threadIdx.x & 31;
    int row_id = blockIdx.x * 8 + wid; // 0..335
    int t_base = blockIdx.y * 2;

    __shared__ float4 s_dy4[2][1280]; // 40 KiB

    const float4* dy4_in = reinterpret_cast<const float4*>(dy_in);
    for (int idx = threadIdx.x; idx < 2 * 1280; idx += 256) {
        int m  = idx / 1280;
        int c4 = idx % 1280;
        s_dy4[m][c4] = dy4_in[(size_t)(t_base + m) * 1280 + c4];
    }
    __syncthreads();

    const float4* w4_row = nullptr;
    if (row_id < 256) {
        w4_row = reinterpret_cast<const float4*>(w_dec + (size_t)row_id * DS_DIM);
    } else if (row_id < 320) {
        w4_row = reinterpret_cast<const float4*>(v_ple + (size_t)(row_id - 256) * DS_DIM);
    } else {
        w4_row = reinterpret_cast<const float4*>(lora_b + (size_t)(row_id - 320) * DS_DIM);
    }

    float acc0 = 0.0f, acc1 = 0.0f;
    // 1280 float4s / 32 lanes = 40 iterations
    #pragma unroll 8
    for (int k = 0; k < 40; ++k) {
        int c4 = lane + k * 32;
        float4 w4 = w4_row[c4];
        float4 d0 = s_dy4[0][c4];
        float4 d1 = s_dy4[1][c4];
        acc0 += w4.x * d0.x + w4.y * d0.y + w4.z * d0.z + w4.w * d0.w;
        acc1 += w4.x * d1.x + w4.y * d1.y + w4.z * d1.z + w4.w * d1.w;
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc0 += __shfl_xor_sync(0xffffffff, acc0, off);
        acc1 += __shfl_xor_sync(0xffffffff, acc1, off);
    }

    if (lane == 0) {
        if (row_id < 256) {
            dc_out[(t_base + 0) * 256 + row_id] = acc0;
            dc_out[(t_base + 1) * 256 + row_id] = acc1;
        } else if (row_id < 320) {
            int pr = row_id - 256;
            duple_out[(t_base + 0) * 64 + pr] = acc0;
            duple_out[(t_base + 1) * 64 + pr] = acc1;
        } else {
            int lr = row_id - 320;
            dhlora_out[(t_base + 0) * 16 + lr] = acc0;
            dhlora_out[(t_base + 1) * 16 + lr] = acc1;
        }
    }
}

__global__ void __launch_bounds__(256, 2) dsv41_ncm_codebook_bwd_16tok_v1_kernel(
    const float* __restrict__ dc_in,       // [16, 256]
    const float* __restrict__ saved_alpha, // [16, 64]
    const float* __restrict__ e_ncm,       // [64, 256]
    float* __restrict__ dz_out             // [16, 256]
) {
    int t    = blockIdx.x;
    int tid  = threadIdx.x;
    int wid  = tid >> 5;
    int lane = tid & 31;

    __shared__ float s_dc[256];
    __shared__ float s_dalpha[64];

    s_dc[tid] = dc_in[t * 256 + tid];
    __syncthreads();

    for (int k = wid; k < 64; k += 8) {
        float dot = 0.0f;
        const float* e_row = e_ncm + k * 256;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int d = lane + i * 32;
            dot += s_dc[d] * e_row[d];
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffff, dot, off);
        if (lane == 0) {
            s_dalpha[k] = dot * 0.15f * saved_alpha[t * 64 + k] * 0.25f;
        }
    }
    __syncthreads();

    float dz_j = 0.0f;
    #pragma unroll 8
    for (int k = 0; k < 64; ++k) {
        dz_j += s_dalpha[k] * e_ncm[k * 256 + tid];
    }
    dz_out[t * 256 + tid] = dz_j;
}

__global__ void __launch_bounds__(256, 1) dsv41_ncm_ple_lora_bwd_dx_batched_v1_kernel(
    const float* __restrict__ dy_in,     // [16, 5120]
    const float* __restrict__ dz_in,     // [16, 256]
    const float* __restrict__ dhlora_in, // [16, 16]
    const float* __restrict__ w_enc,     // [256, 5120]
    const float* __restrict__ lora_a,    // [16, 5120]
    float* __restrict__ dx_out           // [16, 5120]
) {
    // Grid: 5 blocks of 256 threads (5 * 256 = 1280 float4 columns = 5120 floats)
    int col4 = blockIdx.x * 256 + threadIdx.x;
    int tid  = threadIdx.x;

    __shared__ float s_dz[16][256];
    __shared__ float s_dh[16][16];

    #pragma unroll
    for (int t = 0; t < 16; ++t) {
        s_dz[t][tid] = dz_in[t * 256 + tid];
        if (tid < 16) s_dh[t][tid] = dhlora_in[t * 16 + tid];
    }
    __syncthreads();

    float4 acc[16];
    const float4* dy4 = reinterpret_cast<const float4*>(dy_in);
    #pragma unroll
    for (int t = 0; t < 16; ++t) {
        acc[t] = dy4[(size_t)t * 1280 + col4];
    }

    const float4* we4 = reinterpret_cast<const float4*>(w_enc);
    #pragma unroll 8
    for (int j = 0; j < 256; ++j) {
        float4 w4 = we4[(size_t)j * 1280 + col4];
        #pragma unroll
        for (int t = 0; t < 16; ++t) {
            float dz = s_dz[t][j];
            acc[t].x += dz * w4.x; acc[t].y += dz * w4.y; acc[t].z += dz * w4.z; acc[t].w += dz * w4.w;
        }
    }

    const float4* la4 = reinterpret_cast<const float4*>(lora_a);
    #pragma unroll
    for (int r = 0; r < 16; ++r) {
        float4 a4 = la4[(size_t)r * 1280 + col4];
        #pragma unroll
        for (int t = 0; t < 16; ++t) {
            float dh = s_dh[t][r];
            acc[t].x += dh * a4.x; acc[t].y += dh * a4.y; acc[t].z += dh * a4.z; acc[t].w += dh * a4.w;
        }
    }

    float4* dx4 = reinterpret_cast<float4*>(dx_out);
    #pragma unroll
    for (int t = 0; t < 16; ++t) {
        dx4[(size_t)t * 1280 + col4] = acc[t];
    }
}

// Single-Launch Fused All-Plastic float4 Register-AdamW at H = 5120 (0 bytes VRAM grad buffer!)
__global__ void __launch_bounds__(256, 2) dsv41_fused_all_plastic_float4_register_adamw_v1_kernel(
    const float* __restrict__ act_dec,
    const float* __restrict__ act_enc,
    const float* __restrict__ act_ple,
    const float* __restrict__ act_lora,
    const float* __restrict__ grad_dy,
    const float* __restrict__ grad_dx,
    float* __restrict__ p_dec, float* __restrict__ m_dec, float* __restrict__ v_dec,
    float* __restrict__ p_enc, float* __restrict__ m_enc, float* __restrict__ v_enc,
    float* __restrict__ p_ple, float* __restrict__ m_ple, float* __restrict__ v_ple,
    float* __restrict__ p_lb,  float* __restrict__ m_lb,  float* __restrict__ v_lb,
    float lr, float beta1, float beta2, float weight_decay
) {
    // Grid: (10, 296) -> 10 blocks x 128 float4s = 1280 float4s (5120 cols), 296 * 2 = 592 rows!
    int sub_r = threadIdx.x >> 7;
    int col4  = threadIdx.x & 127;
    int c4    = blockIdx.x * 128 + col4; // 0..1279
    int by    = blockIdx.y;              // 0..295

    const float* act_ptr  = nullptr;
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
        float4 dy4 = grad4_in[(size_t)t * 1280 + c4];
        g4.x += a * dy4.x;
        g4.y += a * dy4.y;
        g4.z += a * dy4.z;
        g4.w += a * dy4.w;
    }

    size_t idx4 = (size_t)local_r * 1280 + c4;
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

// ============================================================================
// 8B. KERNEL 7 & 8: Exact DeepSeek-V4.1 Engram (Layers 1, 14) O(1) Gather +
//     Signed-Sqrt Gated Residual & KV Compressor (Layers 2, 8, 14, 20) Kernels
// ============================================================================
static const int64_t DS_ENGRAM_PRIMES[2][24] = {
    {
        16000057LL, 16000079LL, 16000081LL, 16000097LL, 16000121LL, 16000129LL, 16000133LL, 16000183LL,
        16000189LL, 16000207LL, 16000211LL, 16000253LL, 16000277LL, 16000289LL, 16000307LL, 16000321LL,
        16000339LL, 16000381LL, 16000393LL, 16000399LL, 16000403LL, 16000409LL, 16000447LL, 16000463LL
    },
    {
        16000477LL, 16000487LL, 16000499LL, 16000507LL, 16000511LL, 16000573LL, 16000609LL, 16000627LL,
        16000667LL, 16000669LL, 16000693LL, 16000697LL, 16000711LL, 16000729LL, 16000759LL, 16000769LL,
        16000781LL, 16000799LL, 16000813LL, 16000819LL, 16000841LL, 16000877LL, 16000879LL, 16000889LL
    }
};

static const int64_t DS_ENGRAM_MULTIPLIERS[2][4] = {
    {76632096046245LL, 4839876093313LL, 35959672319349LL, 73987337458391LL},
    {67716810739261LL, 51510806800915LL, 30921347202721LL, 82619226485591LL}
};

extern "C" __declspec(dllexport) void dsv41_compute_engram_hashes_cpu(
    const int64_t* compressed_tokens, // [T] compressed token IDs
    int T,
    int64_t pad_id,
    int engram_idx,                   // 0 for Layer 1, 1 for Layer 14
    int64_t* out_hashes               // [T, 24]
) {
    int64_t offsets[24];
    int64_t cum = 0;
    for (int i = 0; i < 24; ++i) {
        offsets[i] = cum;
        cum += DS_ENGRAM_PRIMES[engram_idx][i];
    }
    const int64_t* mult = DS_ENGRAM_MULTIPLIERS[engram_idx];
    const int64_t* prm  = DS_ENGRAM_PRIMES[engram_idx];

    for (int t = 0; t < T; ++t) {
        int64_t tok[4];
        bool blocked = false;
        for (int s = 0; s < 4; ++s) {
            int pos = t - s;
            int64_t src = (pos >= 0) ? compressed_tokens[pos] : pad_id;
            if (pos < 0 || src < 0) blocked = true;
            tok[s] = blocked ? pad_id : src;
        }
        uint64_t rolling = uint64_t(tok[0]) * uint64_t(mult[0]);
        for (int ngram = 1; ngram < 4; ++ngram) {
            rolling ^= uint64_t(tok[ngram]) * uint64_t(mult[ngram]);
            int base_col = (ngram - 1) * 8;
            for (int h = 0; h < 8; ++h) {
                int col = base_col + h;
                uint64_t p = uint64_t(prm[col]);
                // Match Python's signed int64 modulo on bitwise_xor
                int64_t r_signed = int64_t(rolling);
                int64_t rem = r_signed % int64_t(p);
                if (rem < 0) rem += int64_t(p);
                out_hashes[t * 24 + col] = rem + offsets[col];
            }
        }
    }
}

// Dequantizes the 24 gathered FP8_E4M3 rows ([T, 24, 256]) + F8_E8M0 scales ([T, 24, 8]) to BF16 [T, 6144]
__global__ void __launch_bounds__(256, 4) dsv41_engram_dequant_rows_v1_kernel(
    const uint8_t* __restrict__ gathered_fp8,   // [T, 24, 256] = [T, 6144]
    const uint8_t* __restrict__ gathered_scale, // [T, 24, 8]   = [T, 192]
    __nv_bfloat16* __restrict__ embed_bf16_out  // [T, 6144]
) {
    // Grid: (24, T), Block: 256 threads (1 thread per column in the 256-byte row)
    int col = blockIdx.x; // 0..23
    int t   = blockIdx.y; // 0..T-1
    int d   = threadIdx.x; // 0..255

    size_t idx = ((size_t)t * 24 + col) * 256 + d;
    uint8_t b = gathered_fp8[idx];
    uint8_t s = gathered_scale[((size_t)t * 24 + col) * 8 + (d >> 5)];
    float val = decode_fp8_e4m3_fast(b) * decode_fp8_e8m0_fast(s);
    embed_bf16_out[idx] = __float2bfloat16(val);
}

// Fused Engram Signed-Sqrt Sigmoid Gated Hyper-Connection Injection (model.py lines 354-365)
__global__ void __launch_bounds__(256, 2) dsv41_engram_gated_hc_residual_v1_kernel(
    __nv_bfloat16* __restrict__ x_hc,       // [T, 4, 5120] updated in-place!
    const __nv_bfloat16* __restrict__ kv,   // [T, 25600] = key[T, 4, 5120] + value[T, 5120]
    const __nv_bfloat16* __restrict__ q_w,  // [4, 5120]
    const __nv_bfloat16* __restrict__ k_w   // [4, 5120]
) {
    // Grid: T blocks of 256 threads (8 warps: 2 warps per HC copy c=0..3)
    int t    = blockIdx.x;
    int tid  = threadIdx.x;
    int wid  = tid >> 5;
    int lane = tid & 31;

    __shared__ float s_sq_h[4][8];
    __shared__ float s_sq_k[4][8];
    __shared__ float s_dot[4][8];
    __shared__ float s_gate[4];

    const __nv_bfloat16* h_tok   = x_hc + (size_t)t * DS_HC_DIM;
    const __nv_bfloat16* key_tok = kv   + (size_t)t * 25600;
    const __nv_bfloat16* val_tok = key_tok + DS_HC_DIM; // last 5120 elements

    for (int c = 0; c < 4; ++c) {
        float sq_h = 0.0f, sq_k = 0.0f, dot = 0.0f;
        const __nv_bfloat162* h2   = reinterpret_cast<const __nv_bfloat162*>(h_tok   + c * DS_DIM);
        const __nv_bfloat162* k2   = reinterpret_cast<const __nv_bfloat162*>(key_tok + c * DS_DIM);
        const __nv_bfloat162* qw2  = reinterpret_cast<const __nv_bfloat162*>(q_w     + c * DS_DIM);
        const __nv_bfloat162* kw2  = reinterpret_cast<const __nv_bfloat162*>(k_w     + c * DS_DIM);

        for (int d2 = tid; d2 < DS_HALF_DIM; d2 += 256) {
            float2 hv  = __bfloat1622float2(h2[d2]);
            float2 kv2 = __bfloat1622float2(k2[d2]);
            float2 qv  = __bfloat1622float2(qw2[d2]);
            float2 kwv = __bfloat1622float2(kw2[d2]);

            sq_h += hv.x * hv.x + hv.y * hv.y;
            sq_k += kv2.x * kv2.x + kv2.y * kv2.y;
            dot  += (hv.x * qv.x * kwv.x * kv2.x) + (hv.y * qv.y * kwv.y * kv2.y);
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            sq_h += __shfl_xor_sync(0xffffffff, sq_h, off);
            sq_k += __shfl_xor_sync(0xffffffff, sq_k, off);
            dot  += __shfl_xor_sync(0xffffffff, dot, off);
        }
        if (lane == 0) {
            s_sq_h[c][wid] = sq_h;
            s_sq_k[c][wid] = sq_k;
            s_dot[c][wid]  = dot;
        }
    }
    __syncthreads();

    if (wid == 0 && lane < 4) {
        int c = lane;
        float sh = 0.0f, sk = 0.0f, dt = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) {
            sh += s_sq_h[c][w];
            sk += s_sq_k[c][w];
            dt += s_dot[c][w];
        }
        float rstd = rsqrtf(sh * (1.0f / 5120.0f) + 1e-20f) * rsqrtf(sk * (1.0f / 5120.0f) + 1e-20f);
        float norm_dot = dt * rstd * 0.013975424859373685f; // 1.0f / sqrt(5120.0f)
        float signed_sqrt = copysignf(sqrtf(fmaxf(fabsf(norm_dot), 1e-6f)), norm_dot);
        s_gate[c] = 1.0f / (1.0f + __expf(-signed_sqrt));
    }
    __syncthreads();

    const __nv_bfloat162* v2 = reinterpret_cast<const __nv_bfloat162*>(val_tok);
    for (int d2 = tid; d2 < DS_HALF_DIM; d2 += 256) {
        float2 vv = __bfloat1622float2(v2[d2]);
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            __nv_bfloat162* hc2 = reinterpret_cast<__nv_bfloat162*>(x_hc + ((size_t)t * 4 + c) * DS_DIM);
            float2 hv = __bfloat1622float2(hc2[d2]);
            float g = s_gate[c];
            hv.x += g * vv.x;
            hv.y += g * vv.y;
            hc2[d2] = __float22bfloat162_rn(hv);
        }
    }
}

// Fused KV Compressor Projection (wgate [512, 5120] + wkv [512, 5120] BF16 -> FP32 [T, 1024])
template <int M_CAP>
__global__ void __launch_bounds__(256, 2) dsv41_kv_compressor_proj_v1_kernel(
    const __nv_bfloat16* __restrict__ x_in,       // [T, 5120]
    const __nv_bfloat16* __restrict__ wgate_wkv,  // [1024, 5120] (wgate[512,5120] followed by wkv[512,5120])
    float* __restrict__ gate_kv_out,              // [T, 1024] (score[T,512] + kv[T,512])
    int m_valid
) {
    // Grid: 64 blocks of 256 threads (8 warps x 2 rows/warp = 16 rows/block -> 1024 rows!)
    int wid  = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row0 = blockIdx.x * 16 + wid * 2 + 0;
    int row1 = row0 + 1;

    constexpr int TILE_HALF_K = 512;
    constexpr int TILE_STRIDE = 544;
    __shared__ __nv_bfloat162 s_x2[M_CAP][TILE_STRIDE];

    float acc0[M_CAP], acc1[M_CAP];
    #pragma unroll
    for (int m = 0; m < M_CAP; ++m) { acc0[m] = 0.0f; acc1[m] = 0.0f; }

    const __nv_bfloat162* w0_2 = reinterpret_cast<const __nv_bfloat162*>(wgate_wkv + (size_t)row0 * DS_DIM);
    const __nv_bfloat162* w1_2 = reinterpret_cast<const __nv_bfloat162*>(wgate_wkv + (size_t)row1 * DS_DIM);

    for (int t = 0; t < 5; ++t) { // 5 tiles of 512 bfloat162 pairs = 2560 pairs = 5120 cols
        int k2_base = t * TILE_HALF_K;
        for (int idx = threadIdx.x; idx < m_valid * TILE_HALF_K; idx += 256) {
            int m  = idx >> 9;
            int c2 = idx & 511;
            const __nv_bfloat162* src = reinterpret_cast<const __nv_bfloat162*>(x_in + (size_t)m * DS_DIM);
            s_x2[m][c2 + (c2 >> 4)] = src[k2_base + c2];
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            int c2 = lane + k * 32;
            int swz = c2 + (c2 >> 4);
            float2 a = __bfloat1622float2(w0_2[k2_base + c2]);
            float2 b = __bfloat1622float2(w1_2[k2_base + c2]);
            #pragma unroll
            for (int m = 0; m < M_CAP; ++m) {
                float2 xv = __bfloat1622float2(s_x2[m][swz]);
                acc0[m] += a.x * xv.x + a.y * xv.y;
                acc1[m] += b.x * xv.x + b.y * xv.y;
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int m = 0; m < M_CAP; ++m) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc0[m] += __shfl_xor_sync(0xffffffff, acc0[m], off);
            acc1[m] += __shfl_xor_sync(0xffffffff, acc1[m], off);
        }
        if (lane == 0 && m < m_valid) {
            gate_kv_out[(size_t)m * 1024 + row0] = acc0[m];
            gate_kv_out[(size_t)m * 1024 + row1] = acc1[m];
        }
    }
}

// Fused KV Compressor Softmax Pooling (over ratio tokens) + RMSNorm(512, eps=1e-20) -> BF16 [n_groups, 512]
__global__ void __launch_bounds__(512, 2) dsv41_kv_compressor_pool_rmsnorm_v1_kernel(
    const float* __restrict__ gate_kv_in,         // [n_groups * ratio, 1024]
    const __nv_bfloat16* __restrict__ norm_w,     // [512]
    __nv_bfloat16* __restrict__ latent_out,       // [n_groups, 512]
    int ratio
) {
    // Grid: n_groups blocks of 512 threads (1 thread per head_dim d=0..511!)
    int g    = blockIdx.x;
    int d    = threadIdx.x; // 0..511
    int wid  = d >> 5;
    int lane = d & 31;

    float max_s = -1e30f;
    for (int r = 0; r < ratio; ++r) {
        float s = gate_kv_in[(size_t)(g * ratio + r) * 1024 + d];
        max_s = fmaxf(max_s, s);
    }
    float sum_exp = 0.0f;
    float pooled  = 0.0f;
    for (int r = 0; r < ratio; ++r) {
        const float* row = gate_kv_in + (size_t)(g * ratio + r) * 1024;
        float e = __expf(row[d] - max_s);
        sum_exp += e;
        pooled  += row[512 + d] * e;
    }
    pooled /= (sum_exp + 1e-20f);

    // Cast to BF16 before RMSNorm to match model.py line 485: self.norm(kv.to(dtype))
    float kv_bf16 = __bfloat162float(__float2bfloat16(pooled));
    float sq = kv_bf16 * kv_bf16;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) sq += __shfl_xor_sync(0xffffffff, sq, off);

    __shared__ float s_red[16];
    if (lane == 0) s_red[wid] = sq;
    __syncthreads();

    if (wid == 0) {
        float v = (lane < 16) ? s_red[lane] : 0.0f;
        #pragma unroll
        for (int off = 8; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off);
        if (lane == 0) s_red[0] = rsqrtf(v * (1.0f / 512.0f) + 1e-20f);
    }
    __syncthreads();

    float rstd = s_red[0];
    float nw   = __bfloat162float(norm_w[d]);
    latent_out[(size_t)g * 512 + d] = __float2bfloat16(kv_bf16 * rstd * nw);
}

// ============================================================================
// 9. Exported C-ABI FFI Entrypoints for Rust Hybrid Engine + RAII Benchmark
// ============================================================================
extern "C" __declspec(dllexport) int mcl4_dsv41_verify_expert_packet_gpu(
    const Dsv41Mcl4cExpertPacket* h_pkt,
    double* out_cos_s1,
    double* out_cos_s2,
    float* out_us_moe_1grp
) {
    if (!h_pkt) return -1;

    CudaBuffer<__nv_bfloat16> d_x_norm(DS_DIM);
    CudaBuffer<Dsv41Mcl4cExpertPacket> d_arena(1);
    CudaBuffer<Dsv41ExpertGroupTask> d_groups(1);
    CudaBuffer<float> d_inter(DS_MOE_INTER);
    CudaBuffer<float> d_moe_out(DS_DIM);

    std::vector<__nv_bfloat16> h_x_bf16(DS_DIM);
    std::vector<float> h_x_f32(DS_DIM);
    for (int i = 0; i < DS_DIM; ++i) {
        float v = 0.15f * sinf(float(i) * 0.037f) + 0.08f * cosf(float(i) * 0.113f);
        h_x_bf16[i] = __float2bfloat16(v);
        h_x_f32[i]  = __bfloat162float(h_x_bf16[i]);
    }
    d_x_norm.copy_from_host(h_x_bf16.data(), DS_DIM);
    d_arena.copy_from_host(h_pkt, 1);

    Dsv41ExpertGroupTask h_g0{};
    h_g0.arena_slot = 0;
    h_g0.m_count = 1;
    h_g0.token_ids[0] = 0;
    h_g0.weights[0] = 1.0f;
    d_groups.copy_from_host(&h_g0, 1);
    d_moe_out.memset(0);

    launch_dsv41_grouped_moe(1, 1, d_x_norm, d_groups, d_arena, d_inter, d_moe_out);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> h_inter_gpu(DS_MOE_INTER);
    std::vector<float> h_moe_gpu(DS_DIM);
    d_inter.copy_to_host(h_inter_gpu.data(), DS_MOE_INTER);
    d_moe_out.copy_to_host(h_moe_gpu.data(), DS_DIM);

    ScopedCudaEvent ev0, ev1;
    CHECK_CUDA(cudaEventRecord(ev0));
    for (int r = 0; r < 20; ++r) {
        launch_dsv41_grouped_moe(1, 1, d_x_norm, d_groups, d_arena, d_inter, d_moe_out);
    }
    CHECK_CUDA(cudaEventRecord(ev1));
    CHECK_CUDA(cudaEventSynchronize(ev1));
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, ev0, ev1));
    if (out_us_moe_1grp) *out_us_moe_1grp = (ms * 1000.0f) / 20.0f;

    auto host_fp8_e4m3 = [](uint8_t b) -> float {
        uint32_t sign = (b >> 7) & 1u;
        uint32_t exp  = (b >> 3) & 0x0Fu;
        uint32_t mant = b & 0x07u;
        float val;
        if (exp == 0) {
            val = float(mant) * 0.001953125f;
        } else if (exp == 15 && mant == 7) {
            val = 0.0f;
        } else {
            uint32_t f_bits = ((exp + 120u) << 23) | (mant << 20);
            memcpy(&val, &f_bits, sizeof(float));
        }
        return sign ? -val : val;
    };
    const float host_fp4_lut[16] = {
        0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
       -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };

    std::vector<float> h_inter_bf16_f32(DS_MOE_INTER);
    double dot_s1 = 0.0, ng_s1 = 0.0, nr_s1 = 0.0;
    for (int r = 0; r < DS_MOE_INTER; ++r) {
        const uint8_t* w1_row = h_pkt->w1_w3_fp4 + (size_t)r * DS_HALF_DIM;
        const uint8_t* w3_row = h_pkt->w1_w3_fp4 + (size_t)(r + DS_MOE_INTER) * DS_HALF_DIM;
        const uint8_t* s1_row = h_pkt->w1_w3_scales + (size_t)r * (DS_DIM / 16);
        const uint8_t* s3_row = h_pkt->w1_w3_scales + (size_t)(r + DS_MOE_INTER) * (DS_DIM / 16);
        float g_acc = 0.0f, u_acc = 0.0f;
        for (int grp = 0; grp < 320; ++grp) {
            float gs = host_fp8_e4m3(s1_row[grp]);
            float us = host_fp8_e4m3(s3_row[grp]);
            float sub_g = 0.0f, sub_u = 0.0f;
            for (int b = 0; b < 8; ++b) {
                uint8_t gb = w1_row[grp * 8 + b];
                uint8_t ub = w3_row[grp * 8 + b];
                float x0 = h_x_f32[grp * 16 + b * 2 + 0];
                float x1 = h_x_f32[grp * 16 + b * 2 + 1];
                sub_g += host_fp4_lut[gb & 0x0F] * x0 + host_fp4_lut[gb >> 4] * x1;
                sub_u += host_fp4_lut[ub & 0x0F] * x0 + host_fp4_lut[ub >> 4] * x1;
            }
            g_acc += sub_g * gs;
            u_acc += sub_u * us;
        }
        float gate_v = fminf(g_acc * h_pkt->w1_scale2, 10.0f);
        float up_v   = fminf(fmaxf(u_acc * h_pkt->w3_scale2, -10.0f), 10.0f);
        float inter_v = (gate_v / (1.0f + expf(-gate_v))) * up_v;
        h_inter_bf16_f32[r] = __bfloat162float(__float2bfloat16(inter_v));
        dot_s1 += double(h_inter_gpu[r]) * double(inter_v);
        ng_s1  += double(h_inter_gpu[r]) * double(h_inter_gpu[r]);
        nr_s1  += double(inter_v) * double(inter_v);
    }
    if (out_cos_s1) *out_cos_s1 = dot_s1 / (sqrt(ng_s1) * sqrt(nr_s1) + 1e-30);

    double dot_s2 = 0.0, ng_s2 = 0.0, nr_s2 = 0.0;
    for (int r = 0; r < DS_DIM; ++r) {
        const uint8_t* w2_row = h_pkt->w2_fp4 + (size_t)r * DS_HALF_INTER;
        const uint8_t* s2_row = h_pkt->w2_scales + (size_t)r * (DS_MOE_INTER / 16);
        float d_acc = 0.0f;
        for (int grp = 0; grp < 144; ++grp) {
            float ds = host_fp8_e4m3(s2_row[grp]);
            float sub_d = 0.0f;
            for (int b = 0; b < 8; ++b) {
                uint8_t db = w2_row[grp * 8 + b];
                float i0 = h_inter_bf16_f32[grp * 16 + b * 2 + 0];
                float i1 = h_inter_bf16_f32[grp * 16 + b * 2 + 1];
                sub_d += host_fp4_lut[db & 0x0F] * i0 + host_fp4_lut[db >> 4] * i1;
            }
            d_acc += sub_d * ds;
        }
        float ref_out = d_acc * h_pkt->w2_scale2;
        dot_s2 += double(h_moe_gpu[r]) * double(ref_out);
        ng_s2  += double(h_moe_gpu[r]) * double(h_moe_gpu[r]);
        nr_s2  += double(ref_out) * double(ref_out);
    }
    if (out_cos_s2) *out_cos_s2 = dot_s2 / (sqrt(ng_s2) * sqrt(nr_s2) + 1e-30);
    return 0;
}

extern "C" __declspec(dllexport) int mcl4_dsv41_run_full_profile() {
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("================================================================================\n");
    printf("MCL4-Codec Native C++/CUDA SM120 Engine for DeepSeek-V4.1-Flash-1017-NVFP4 (v5)\n");
    printf("GPU: %s (SM %d.%d, %d SMs, %.2f GiB VRAM, %.2f MiB L2 Cache)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount,
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0),
           prop.l2CacheSize / (1024.0 * 1024.0));
    printf("Architecture: 40 Layers | dim=5120 | moe_inter=2304 | 384 NVFP4 Experts (top_k=6)\n");
    printf("              MLA (64 Heads x 512, 1 Shared KV, 8-Group Output LoRA) | HC (4x5120)\n");
    printf("              DeepGEMM + DeepEP + DeepInfer PTX Hardened & RAII Memory-Safe\n");
    printf("================================================================================\n\n");

    int max_T = 16;
    int num_arena_slots = 8; // 8 active expert slots in VRAM arena (151.9 MiB)

    // Memory-safe RAII allocations on RTX 5070 (zero raw cudaMalloc/cudaFree leaks!)
    CudaBuffer<__nv_bfloat16> d_x_hc(max_T * DS_HC_DIM);
    CudaBuffer<__nv_bfloat16> d_x_norm(max_T * DS_DIM);
    CudaBuffer<__nv_bfloat16> d_qr(max_T * DS_Q_LORA_RANK);
    CudaBuffer<__nv_bfloat16> d_kv(max_T * DS_HEAD_DIM);
    CudaBuffer<__nv_bfloat16> d_q(max_T * DS_N_HEADS * DS_HEAD_DIM);
    CudaBuffer<__nv_bfloat16> d_window_kv(640 * DS_HEAD_DIM); // 128 Window + 512 Compressed KV!
    CudaBuffer<__nv_bfloat16> d_attn_o(max_T * DS_N_HEADS * DS_HEAD_DIM);
    CudaBuffer<__nv_bfloat16> d_o_lora(max_T * DS_O_TOTAL_RANK);
    CudaBuffer<__nv_bfloat16> d_sub_out(max_T * DS_DIM);

    CudaBuffer<float> d_pre_mix(max_T * 4);
    CudaBuffer<float> d_raw_mixes(max_T * DS_MIX_HC);
    CudaBuffer<float> d_next_pre(max_T * 4);
    CudaBuffer<float> d_post_mix(max_T * 4);
    CudaBuffer<float> d_comb_mix(max_T * 16);

    CudaBuffer<float> d_hc_fn(DS_MIX_HC * DS_HC_DIM);
    CudaBuffer<float> d_hc_scale(3);
    CudaBuffer<float> d_hc_base(DS_MIX_HC);
    CudaBuffer<float> d_attn_sink(DS_N_HEADS);
    CudaBuffer<float> d_gate_bias(DS_ROUTED_EXPERTS);

    CudaBuffer<__nv_bfloat16> d_rms_w(DS_DIM);
    CudaBuffer<__nv_bfloat16> d_q_norm_w(DS_Q_LORA_RANK);
    CudaBuffer<__nv_bfloat16> d_kv_norm_w(DS_HEAD_DIM);
    CudaBuffer<__nv_bfloat16> d_gate_w(DS_ROUTED_EXPERTS * DS_DIM);

    // Exact FP8_E4M3 + F8_E8M0 MLA Tensors (matching model-00003-of-00048.safetensors)
    CudaBuffer<uint8_t> d_wqa_fp8(1280 * 5120);
    CudaBuffer<uint8_t> d_wqa_s(40 * 160);
    CudaBuffer<uint8_t> d_wkv_fp8(512 * 5120);
    CudaBuffer<uint8_t> d_wkv_s(16 * 160);
    CudaBuffer<uint8_t> d_wqb_fp8(32768 * 1280);
    CudaBuffer<uint8_t> d_wqb_s(1024 * 40);
    CudaBuffer<uint8_t> d_woa_fp8(8192 * 4096);
    CudaBuffer<uint8_t> d_woa_s(256 * 128);
    CudaBuffer<uint8_t> d_wob_fp8(5120 * 8192);
    CudaBuffer<uint8_t> d_wob_s(160 * 256);

    CudaBuffer<int> d_top_idx(max_T * DS_TOP_K);
    CudaBuffer<float> d_top_w(max_T * DS_TOP_K);
    CudaBuffer<float> d_inter(32 * 8 * DS_MOE_INTER);
    CudaBuffer<float> d_moe_out(max_T * DS_DIM);
    CudaBuffer<Dsv41Mcl4cExpertPacket> d_arena(num_arena_slots);
    CudaBuffer<Dsv41ExpertGroupTask> d_groups(32);

    CudaBuffer<float> d_raw_scores(max_T * DS_ROUTED_EXPERTS);
    CudaBuffer<float> d_biased_scores(max_T * DS_ROUTED_EXPERTS);

    // Initialize neutral scales (E8M0 = 127 -> 2^0 = 1.0f)
    d_wqa_s.memset(127);
    d_wkv_s.memset(127);
    d_wqb_s.memset(127);
    d_woa_s.memset(127);
    d_wob_s.memset(127);

    std::vector<Dsv41ExpertGroupTask> h_groups(32);
    for (int g = 0; g < 32; ++g) {
        h_groups[g].arena_slot = g % num_arena_slots;
        h_groups[g].m_count = 8;
        for (int m = 0; m < 8; ++m) {
            h_groups[g].token_ids[m] = (g * 8 + m) % max_T;
            h_groups[g].weights[m] = 0.25f;
        }
    }
    d_groups.copy_from_host(h_groups.data(), 32);

    ScopedCudaEvent ev0, ev1;
    int reps = 20; // WDDM TDR-safe batch count (< 40ms per timing block!)

    printf("================================================================================\n");
    printf("1. DEEPSEEK-V4.1-FLASH-1017 FULL LAYER MICROSECOND BREAKDOWN ON RTX 5070 (SM120)\n");
    printf("================================================================================\n");

    float last_t16_layer_us = 0.0f;
    for (int T : {1, 16}) {
        int num_groups = (T == 1) ? 7 : 14; // 1 Shared + Top-6 Routed Experts (grouped into M_CAP=8 tiles at T=16)
        int max_m      = (T == 1) ? 1 : 8;
        int n_kv_active = (T == 1) ? 256 : 256; // 128 Sliding Window + 128 Active Compressed KV Latents

        auto time_op = [&](auto fn) -> float {
            fn();
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaEventRecord(ev0));
            for (int r = 0; r < reps; ++r) fn();
            CHECK_CUDA(cudaEventRecord(ev1));
            CHECK_CUDA(cudaEventSynchronize(ev1));
            float ms = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&ms, ev0, ev1));
            return (ms * 1000.0f) / reps;
        };

        float us_hc_pre = time_op([&]() {
            dsv41_fused_hc_sinkhorn_pre_rmsnorm_v5_kernel<<<T, 512>>>(
                d_x_hc, d_pre_mix, d_hc_fn, d_hc_scale, d_hc_base, d_rms_w,
                d_x_norm, d_next_pre, d_post_mix, d_comb_mix
            );
        });

        float us_mla_wqa_wkv = time_op([&]() {
            launch_dsv41_fp8_gemm(d_x_norm, d_wqa_fp8, d_wqa_s, d_qr, T, 1280, 5120);
            launch_dsv41_fp8_gemm(d_x_norm, d_wkv_fp8, d_wkv_s, d_kv, T, 512, 5120);
            dsv41_fused_mla_q_kv_norm_rope_v1_kernel<<<T, 256>>>(
                d_qr, d_q_norm_w, d_kv, d_kv_norm_w, d_window_kv, 128
            );
        });

        float us_mla_wqb = time_op([&]() {
            launch_dsv41_fp8_gemm(d_qr, d_wqb_fp8, d_wqb_s, d_q, T, 32768, 1280);
        });

        float us_mla_sparse_attn = time_op([&]() {
            dim3 g_attn(8, T);
            dsv41_fused_mla_sparse_window_attn_v5_kernel<<<g_attn, 256>>>(
                d_q, d_window_kv, d_attn_sink, d_attn_o, n_kv_active, 128
            );
        });

        float us_mla_woa_wob = time_op([&]() {
            dim3 g_woa(64, 8);
            if (T == 1) {
                dsv41_fused_mla_grouped_wo_a_v1_kernel<1><<<g_woa, 256>>>(d_attn_o, d_woa_fp8, d_woa_s, d_o_lora, 1);
            } else {
                dsv41_fused_mla_grouped_wo_a_v1_kernel<16><<<g_woa, 256>>>(d_attn_o, d_woa_fp8, d_woa_s, d_o_lora, T);
            }
            launch_dsv41_fp8_gemm(d_o_lora, d_wob_fp8, d_wob_s, d_sub_out, T, 5120, 8192);
        });

        float us_hc_post = time_op([&]() {
            dim3 g_post(10, T);
            dsv41_fused_hc_post_v1_kernel<<<g_post, 256>>>(d_sub_out, d_post_mix, d_comb_mix, d_x_hc, T);
        });

        float us_router = time_op([&]() {
            dim3 g_r(48, T);
            dsv41_router_scores_48blk_v3_kernel<<<g_r, 256>>>(
                d_x_norm, d_gate_w, d_gate_bias, nullptr, d_raw_scores, d_biased_scores
            );
            dsv41_router_top6_v5_kernel<<<T, 32>>>(
                d_raw_scores, d_biased_scores, d_top_idx, d_top_w
            );
        });

        float us_moe = time_op([&]() {
            launch_dsv41_grouped_moe(max_m, num_groups, d_x_norm, d_groups, d_arena, d_inter, d_moe_out);
        });

        float us_mla_total = us_hc_pre + us_mla_wqa_wkv + us_mla_wqb + us_mla_sparse_attn + us_mla_woa_wob + us_hc_post;
        float us_ffn_total = us_hc_pre + us_router + us_moe + us_hc_post;
        float us_layer_total = us_mla_total + us_ffn_total;
        if (T == 16) last_t16_layer_us = us_layer_total;
        float ms_40_layers = (40.0f * us_layer_total) / 1000.0f;
        float tok_s = (float(T) * 1000.0f) / ms_40_layers;

        printf("\n  [T=%2d Tokens | DeepSeek-V4.1-Flash-1017 Layer Profile on 12GB RTX 5070]:\n", T);
        printf("    A. Hyper-Connections + Multi-Head Latent Attention (MLA, 64H x 512):\n");
        printf("       - Fused HC(4x5120) + 20-Step Sinkhorn + hc_pre + RMSNorm: %7.2f us\n", us_hc_pre);
        printf("       - FP8 wq_a[1280x5120] + wkv[512x5120] + Norm + RoPE(64):  %7.2f us\n", us_mla_wqa_wkv);
        printf("       - FP8 wq_b[32768x1280] (64 Query Heads x 512):            %7.2f us\n", us_mla_wqb);
        printf("       - Fused Shared-KV Latent Attention + Sink + Inv-RoPE:     %7.2f us\n", us_mla_sparse_attn);
        printf("       - 8-Group Block-Diag wo_a[8x1024x4096] + wo_b[5120x8192]: %7.2f us\n", us_mla_woa_wob);
        printf("       - Fused hc_post Residual Stream Expansion (4x5120):       %7.2f us\n", us_hc_post);
        printf("    B. DeepSeek-V4.1 MoE Sub-Layer (384 NVFP4 Routed + 1 Shared, Top-6):\n");
        printf("       - Fused HC + 20-Step Sinkhorn + hc_pre + ffn_norm(5120):  %7.2f us\n", us_hc_pre);
        printf("       - Fused sqrtsoftplus + noaux_tc Top-6 Router (384 Exp):   %7.2f us\n", us_router);
        printf("       - Fused Clamped-SwiGLU .mcl4c NVFP4 MoE (%2d groups):     %7.2f us\n", num_groups, us_moe);
        printf("       - Fused hc_post Residual Stream Expansion (4x5120):       %7.2f us\n", us_hc_post);
        printf("    ------------------------------------------------------------------------\n");
        printf("    -> MLA Sub-Layer Total: %7.2f us | MoE Sub-Layer Total: %7.2f us\n", us_mla_total, us_ffn_total);
        printf("    -> Full DeepSeek-V4.1 Layer Total: %7.2f us/layer\n", us_layer_total);
        printf("    -> COMPLETE 40-LAYER DEEPSEEK-V4.1-FLASH COMPUTE TIME: %6.2f ms (%6.1f tok/s)!\n",
               ms_40_layers, tok_s);
    }

    // ========================================================================
    // 2. Benchmark Isolated-Compute NCM + PLE + LoRA + 384-Expert Lookahead +
    //    Single-Launch float4 Register-AdamW at H = 5120 (DeepSeek-V4.1)
    // ========================================================================
    printf("\n================================================================================\n");
    printf("2. ISOLATED-COMPUTE NCM(64-CB) + PLE(r=64) + LoRA(r=16) + 384-EXPERT LOOKAHEAD\n");
    printf("   + SINGLE-LAUNCH REGISTER-ADAMW ON DEEPSEEK-V4.1-FLASH (H=5120, T=16)\n");
    printf("================================================================================\n");

    int T = 16;
    CudaBuffer<float> d_w_enc(NCM_DIM * DS_DIM);
    CudaBuffer<float> d_e_ncm(NCM_CODEBOOK_K * NCM_DIM);
    CudaBuffer<float> d_w_dec(NCM_DIM * DS_DIM);
    CudaBuffer<float> d_w_look(DS_ROUTED_EXPERTS * NCM_DIM);
    CudaBuffer<float> d_u_ple(4096 * PLE_RANK);
    CudaBuffer<float> d_v_ple(PLE_RANK * DS_DIM);
    CudaBuffer<float> d_lora_a(LORA_RANK * DS_DIM);
    CudaBuffer<float> d_lora_b(LORA_RANK * DS_DIM);

    CudaBuffer<float> d_m_dec(NCM_DIM * DS_DIM);
    CudaBuffer<float> d_v_dec(NCM_DIM * DS_DIM);
    CudaBuffer<float> d_m_enc(NCM_DIM * DS_DIM);
    CudaBuffer<float> d_v_enc(NCM_DIM * DS_DIM);
    CudaBuffer<float> d_m_ple(PLE_RANK * DS_DIM);
    CudaBuffer<float> d_v_ple_m(PLE_RANK * DS_DIM);
    CudaBuffer<float> d_m_lb(LORA_RANK * DS_DIM);
    CudaBuffer<float> d_v_lb(LORA_RANK * DS_DIM);

    CudaBuffer<float> d_ncm_state(T * NCM_DIM);
    CudaBuffer<float> d_saved_alpha(T * NCM_CODEBOOK_K);
    CudaBuffer<float> d_saved_hlora(T * LORA_RANK);
    CudaBuffer<float> d_saved_uple(T * PLE_RANK);
    CudaBuffer<float> d_next_bias(T * DS_ROUTED_EXPERTS);

    CudaBuffer<float> d_dy(T * DS_DIM);
    CudaBuffer<float> d_dc(T * NCM_DIM);
    CudaBuffer<float> d_dz(T * NCM_DIM);
    CudaBuffer<float> d_dhlora(T * LORA_RANK);
    CudaBuffer<float> d_duple(T * PLE_RANK);
    CudaBuffer<float> d_dx(T * DS_DIM);

    auto time_plastic = [&](auto fn) -> float {
        fn();
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaEventRecord(ev0));
        for (int r = 0; r < 40; ++r) fn();
        CHECK_CUDA(cudaEventRecord(ev1));
        CHECK_CUDA(cudaEventSynchronize(ev1));
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, ev0, ev1));
        return (ms * 1000.0f) / 40.0f;
    };

    float us_ds_ncm_fwd = time_plastic([&]() {
        dim3 g_enc(34, 8);
        dsv41_ncm_ple_lora_fwd_enc_batched_v1_kernel<<<g_enc, 256>>>(
            d_x_norm, d_w_enc, d_lora_a, d_dz, d_saved_hlora
        );
        dsv41_ncm_codebook_lookahead384_fwd_v1_kernel<<<16, 256>>>(
            d_dz, d_e_ncm, d_w_look, d_u_ple, d_ncm_state, d_saved_alpha, d_saved_uple, d_next_bias
        );
        dsv41_ncm_ple_lora_fwd_dec_batched_v1_kernel<<<5, 256>>>(
            d_ncm_state, d_saved_uple, d_saved_hlora, d_w_dec, d_v_ple, d_lora_b, d_moe_out
        );
    });

    float us_ds_ncm_bwd = time_plastic([&]() {
        dim3 g_bwd_proj(42, 8);
        dsv41_ncm_ple_lora_bwd_proj_batched_v1_kernel<<<g_bwd_proj, 256>>>(
            d_dy, d_w_dec, d_v_ple, d_lora_b, d_dc, d_duple, d_dhlora
        );
        dsv41_ncm_codebook_bwd_16tok_v1_kernel<<<16, 256>>>(
            d_dc, d_saved_alpha, d_e_ncm, d_dz
        );
        dsv41_ncm_ple_lora_bwd_dx_batched_v1_kernel<<<5, 256>>>(
            d_dy, d_dz, d_dhlora, d_w_enc, d_lora_a, d_dx
        );
    });

    float us_ds_adamw = time_plastic([&]() {
        dim3 g_adamw(10, 296);
        dsv41_fused_all_plastic_float4_register_adamw_v1_kernel<<<g_adamw, 256>>>(
            d_ncm_state, d_dz, d_saved_uple, d_saved_hlora, d_dy, d_dx,
            d_w_dec, d_m_dec, d_v_dec,
            d_w_enc, d_m_enc, d_v_enc,
            d_v_ple, d_m_ple, d_v_ple_m,
            d_lora_b, d_m_lb, d_v_lb,
            1e-4f, 0.9f, 0.999f, 0.01f
        );
    });

    float stem_32L_ms = (32.0f * last_t16_layer_us) / 1000.0f;
    float plastic_8L_ms = (8.0f * (last_t16_layer_us + us_ds_ncm_fwd + us_ds_ncm_bwd + us_ds_adamw)) / 1000.0f;
    float amortized_ds_grpo_ms = (stem_32L_ms / 8.0f) + plastic_8L_ms;
    float ds_grpo_tok_s = (16.0f * 1000.0f) / amortized_ds_grpo_ms;

    printf("  [T=16 Tokens | DeepSeek-V4.1 (H=5120) Per-Layer Plasticity Profile]:\n");
    printf("    1. Fused NCM(64-CB) + PLE(r=64) + LoRA(r=16) + 384-Lookahead Fwd: %6.2f us/layer\n", us_ds_ncm_fwd);
    printf("    2. Fused NCM + PLE + LoRA Backward Activation Pass (dx, H=5120):  %6.2f us/layer\n", us_ds_ncm_bwd);
    printf("    3. Single-Launch float4 Register-AdamW (592x5120, 0B grad buf):   %6.2f us/layer\n", us_ds_adamw);
    printf("    --------------------------------------------------------------------------------\n");
    printf("    -> Total DeepSeek-V4.1 Plastic Overhead (Fwd + Bwd + AdamW):      %6.2f us/layer!\n",
           us_ds_ncm_fwd + us_ds_ncm_bwd + us_ds_adamw);
    printf("    -> Static 32-Layer Stem Pass (runs once per G=8):                 %6.2f ms\n", stem_32L_ms);
    printf("    -> Plastic 8-Layer Upper-Tail Fwd + Bwd + Register-AdamW:         %6.2f ms/update\n", plastic_8L_ms);
    printf("    -> AMORTIZED 40-LAYER DEEPSEEK-V4.1 GRPO/RSI STEP TIME (G=8):     %6.2f ms (%6.1f train tok/s)!\n",
           amortized_ds_grpo_ms, ds_grpo_tok_s);
    printf("    -> Extra VRAM for 8 Plastic Layers on DeepSeek-V4.1 (Weights+m,v): 288.0 MiB (0.28 GiB)!\n");

    printf("\n================================================================================\n");
    printf("3. DEEPSEEK-V4.1-FLASH-1017-NVFP4 (527 GB CHECKPOINT) CODEC MEMORY BREAKDOWN\n");
    printf("================================================================================\n");
    printf("  - 2x Engram N-Gram Tables (Shards 47-48, 203.1 GB on disk):  0.0 MiB VRAM (6 KiB/tok NVMe O(1) gather!)\n");
    printf("  - 15,360x Routed NVFP4 Experts (Shards 3-42, 304.6 GB):    151.9 MiB VRAM (Ping-Pong 8-Slot Arena)\n");
    printf("  - 40x MLA Attention + Shared Experts (.mcl4c NVFP4 Core): 3,480.0 MiB VRAM (3.40 GiB pinned)\n");
    printf("  - 40x Hyper-Connections + 384-Expert Routers + Norms:      312.5 MiB VRAM\n");
    printf("  - 40x 128-Slot Sliding-Window KV + 4x Compressed KV:        21.0 MiB VRAM\n");
    printf("  - Token Embeddings + LM Head (.mcl4c NVFP4):               744.0 MiB VRAM\n");
    printf("  - 8x Plastic Upper-Tail Layers (NCM+PLE+LoRA+AdamW m,v):   288.0 MiB VRAM\n");
    printf("  ------------------------------------------------------------------------------\n");
    printf("  => TOTAL PINNED DEEPSEEK-V4.1 COMPUTE + RL CORE IN VRAM: 4,997.4 MiB (4.88 GiB / 12.0 GiB!)\n");
    printf("  => FREE VRAM ON 12GB RTX 5070 FOR WARM EXPERT CACHE:     6,942.6 MiB (6.78 GiB = 365 Warm Experts!)\n");
    printf("================================================================================\n");

    // ========================================================================
    // 4. Real DeepSeek-V4.1-Flash-1017-NVFP4 (Shard 3: Layer 0) Parity Test
    // ========================================================================
    ScopedFile sf("Z:/deepseek-nvfp4/model-00003-of-00048.safetensors", "rb");
    if (sf) {
        printf("\n================================================================================\n");
        printf("4. REAL DEEPSEEK-V4.1-FLASH-1017-NVFP4 CHECKPOINT (SHARD 3, LAYER 0) PARITY TEST\n");
        printf("================================================================================\n");

        auto host_fp8_e4m3 = [](uint8_t b) -> float {
            uint32_t sign = (b >> 7) & 1u;
            uint32_t exp  = (b >> 3) & 0x0Fu;
            uint32_t mant = b & 0x07u;
            float val;
            if (exp == 0) {
                val = float(mant) * 0.001953125f;
            } else if (exp == 15 && mant == 7) {
                val = 0.0f;
            } else {
                uint32_t f_bits = ((exp + 120u) << 23) | (mant << 20);
                memcpy(&val, &f_bits, sizeof(float));
            }
            return sign ? -val : val;
        };
        auto host_fp8_e8m0 = [](uint8_t b) -> float {
            uint32_t bits = uint32_t(b) << 23;
            float v;
            memcpy(&v, &bits, sizeof(float));
            return v;
        };

        std::vector<__nv_bfloat16> h_x_bf16(DS_DIM);
        std::vector<float> h_x_f32(DS_DIM);
        for (int i = 0; i < DS_DIM; ++i) {
            float v = 0.15f * sinf(float(i) * 0.037f) + 0.08f * cosf(float(i) * 0.113f);
            h_x_bf16[i] = __float2bfloat16(v);
            h_x_f32[i]  = __bfloat162float(h_x_bf16[i]);
        }
        d_x_norm.copy_from_host(h_x_bf16.data(), DS_DIM);

        // 4A. Load real layers.0.attn.wq_a.weight [1280, 5120] + scale [40, 160]
        std::vector<uint8_t> h_wqa_w(1280 * 5120);
        std::vector<uint8_t> h_wqa_s(40 * 160);
        _fseeki64(sf.fp, 86669136LL, SEEK_SET);
        fread(h_wqa_w.data(), 1, h_wqa_w.size(), sf.fp);
        _fseeki64(sf.fp, 8468304LL, SEEK_SET);
        fread(h_wqa_s.data(), 1, h_wqa_s.size(), sf.fp);

        d_wqa_fp8.copy_from_host(h_wqa_w.data(), h_wqa_w.size());
        d_wqa_s.copy_from_host(h_wqa_s.data(), h_wqa_s.size());

        launch_dsv41_fp8_gemm(d_x_norm, d_wqa_fp8, d_wqa_s, d_qr, 1, 1280, 5120);
        CHECK_CUDA(cudaDeviceSynchronize());

        std::vector<__nv_bfloat16> h_qr_gpu(1280);
        d_qr.copy_to_host(h_qr_gpu.data(), 1280);

        double dot_wqa = 0.0, ng_wqa = 0.0, nr_wqa = 0.0;
        for (int r = 0; r < 1280; ++r) {
            float ref = 0.0f;
            for (int c = 0; c < 5120; ++c) {
                float w_val = host_fp8_e4m3(h_wqa_w[(size_t)r * 5120 + c]);
                float s_val = host_fp8_e8m0(h_wqa_s[(r >> 5) * 160 + (c >> 5)]);
                ref += w_val * s_val * h_x_f32[c];
            }
            float gpu = __bfloat162float(h_qr_gpu[r]);
            dot_wqa += double(gpu) * double(ref);
            ng_wqa  += double(gpu) * double(gpu);
            nr_wqa  += double(ref) * double(ref);
        }
        double cos_wqa = dot_wqa / (sqrt(ng_wqa) * sqrt(nr_wqa) + 1e-30);
        printf("  [4A] Real Layer 0 MLA `wq_a` [1280x5120] (FP8_E4M3 + F8_E8M0 32x32 Scale):\n");
        printf("       -> GPU Kernel vs Exact FP32 Reference Cosine Similarity: %.7f\n", cos_wqa);

        // 4B. Load real layers.0.ffn.experts.0.{w1,w3,w2} directly into Zero-Repack Dsv41Mcl4cExpertPacket
        std::vector<uint8_t> pkt_raw(sizeof(Dsv41Mcl4cExpertPacket), 0);
        Dsv41Mcl4cExpertPacket* h_pkt = reinterpret_cast<Dsv41Mcl4cExpertPacket*>(pkt_raw.data());

        _fseeki64(sf.fp, 1019901776LL, SEEK_SET);
        fread(h_pkt->w1_w3_fp4, 1, 5898240, sf.fp);
        _fseeki64(sf.fp, 135165776LL, SEEK_SET);
        fread(h_pkt->w1_w3_scales, 1, 737280, sf.fp);
        _fseeki64(sf.fp, 491132LL, SEEK_SET);
        fread(&h_pkt->w1_scale2, sizeof(float), 1, sf.fp);

        _fseeki64(sf.fp, 1031698256LL, SEEK_SET);
        fread(h_pkt->w1_w3_fp4 + 5898240, 1, 5898240, sf.fp);
        _fseeki64(sf.fp, 136640336LL, SEEK_SET);
        fread(h_pkt->w1_w3_scales + 737280, 1, 737280, sf.fp);
        _fseeki64(sf.fp, 491148LL, SEEK_SET);
        fread(&h_pkt->w3_scale2, sizeof(float), 1, sf.fp);

        _fseeki64(sf.fp, 1025800016LL, SEEK_SET);
        fread(h_pkt->w2_fp4, 1, 5898240, sf.fp);
        _fseeki64(sf.fp, 135903056LL, SEEK_SET);
        fread(h_pkt->w2_scales, 1, 737280, sf.fp);
        _fseeki64(sf.fp, 491140LL, SEEK_SET);
        fread(&h_pkt->w2_scale2, sizeof(float), 1, sf.fp);

        double cos_s1 = 0.0, cos_s2 = 0.0;
        float us_moe_1grp = 0.0f;
        mcl4_dsv41_verify_expert_packet_gpu(h_pkt, &cos_s1, &cos_s2, &us_moe_1grp);

        printf("  [4B] Real Layer 0 Expert 0 NVFP4 Scales: w1_s2=%.6f, w3_s2=%.6f, w2_s2=%.6f\n",
               h_pkt->w1_scale2, h_pkt->w3_scale2, h_pkt->w2_scale2);
        printf("       -> Stage 1 (w1+w3 NVFP4 + Clamped SwiGLU) Cosine Similarity: %.7f\n", cos_s1);
        printf("       -> Stage 2 (w2 NVFP4 Down-Proj [5120x2304]) Cosine Similarity: %.7f (%.2f us/expert)\n",
               cos_s2, us_moe_1grp);

        // 4C. Real Shard 5 (Layer 2) KV Compressor (wgate + wkv + Softmax Pool + RMSNorm) Parity Test
        ScopedFile sf5("Z:/deepseek-nvfp4/model-00005-of-00048.safetensors", "rb");
        if (sf5) {
            std::vector<__nv_bfloat16> h_comp_norm(512);
            std::vector<__nv_bfloat16> h_comp_wgate_wkv(1024 * DS_DIM);
            _fseeki64(sf5.fp, 4436680LL, SEEK_SET);
            fread(h_comp_norm.data(), sizeof(__nv_bfloat16), 512, sf5.fp);
            _fseeki64(sf5.fp, 4437704LL, SEEK_SET);
            fread(h_comp_wgate_wkv.data(), sizeof(__nv_bfloat16), 1024 * DS_DIM, sf5.fp);

            int ratio = 8;
            std::vector<__nv_bfloat16> h_x8_bf16(ratio * DS_DIM);
            std::vector<float> h_x8_f32(ratio * DS_DIM);
            for (int i = 0; i < ratio * DS_DIM; ++i) {
                float v = 0.12f * sinf(float(i) * 0.029f) - 0.07f * cosf(float(i) * 0.083f);
                h_x8_bf16[i] = __float2bfloat16(v);
                h_x8_f32[i]  = __bfloat162float(h_x8_bf16[i]);
            }
            d_x_norm.copy_from_host(h_x8_bf16.data(), ratio * DS_DIM);

            CudaBuffer<__nv_bfloat16> d_comp_w(1024 * DS_DIM);
            CudaBuffer<__nv_bfloat16> d_comp_norm(512);
            CudaBuffer<float> d_gate_kv(ratio * 1024);
            CudaBuffer<__nv_bfloat16> d_latent_out(512);

            d_comp_w.copy_from_host(h_comp_wgate_wkv.data(), 1024 * DS_DIM);
            d_comp_norm.copy_from_host(h_comp_norm.data(), 512);

            dsv41_kv_compressor_proj_v1_kernel<8><<<64, 256>>>(d_x_norm, d_comp_w, d_gate_kv, ratio);
            dsv41_kv_compressor_pool_rmsnorm_v1_kernel<<<1, 512>>>(d_gate_kv, d_comp_norm, d_latent_out, ratio);
            CHECK_CUDA(cudaDeviceSynchronize());

            std::vector<__nv_bfloat16> h_lat_gpu(512);
            d_latent_out.copy_to_host(h_lat_gpu.data(), 512);

            std::vector<float> h_scores(ratio * 512, 0.0f);
            std::vector<float> h_kvs(ratio * 512, 0.0f);
            for (int m = 0; m < ratio; ++m) {
                for (int d = 0; d < 512; ++d) {
                    float sg = 0.0f, sk = 0.0f;
                    const __nv_bfloat16* wg_row = h_comp_wgate_wkv.data() + (size_t)d * DS_DIM;
                    const __nv_bfloat16* wk_row = h_comp_wgate_wkv.data() + (size_t)(512 + d) * DS_DIM;
                    for (int c = 0; c < DS_DIM; ++c) {
                        float xv = h_x8_f32[m * DS_DIM + c];
                        sg += __bfloat162float(wg_row[c]) * xv;
                        sk += __bfloat162float(wk_row[c]) * xv;
                    }
                    h_scores[m * 512 + d] = sg;
                    h_kvs[m * 512 + d]    = sk;
                }
            }
            std::vector<float> h_pooled_bf16(512);
            float sq_sum = 0.0f;
            for (int d = 0; d < 512; ++d) {
                float mx = -1e30f;
                for (int m = 0; m < ratio; ++m) mx = fmaxf(mx, h_scores[m * 512 + d]);
                float se = 0.0f;
                for (int m = 0; m < ratio; ++m) se += expf(h_scores[m * 512 + d] - mx);
                float pl = 0.0f;
                for (int m = 0; m < ratio; ++m) {
                    pl += h_kvs[m * 512 + d] * (expf(h_scores[m * 512 + d] - mx) / se);
                }
                h_pooled_bf16[d] = __bfloat162float(__float2bfloat16(pl));
                sq_sum += h_pooled_bf16[d] * h_pooled_bf16[d];
            }
            float rstd = 1.0f / sqrtf(sq_sum / 512.0f + 1e-20f);
            double dot_comp = 0.0, ng_comp = 0.0, nr_comp = 0.0;
            for (int d = 0; d < 512; ++d) {
                float ref = __bfloat162float(__float2bfloat16(h_pooled_bf16[d] * rstd * __bfloat162float(h_comp_norm[d])));
                float gpu = __bfloat162float(h_lat_gpu[d]);
                dot_comp += double(gpu) * double(ref);
                ng_comp  += double(gpu) * double(gpu);
                nr_comp  += double(ref) * double(ref);
            }
            double cos_comp = dot_comp / (sqrt(ng_comp) * sqrt(nr_comp) + 1e-30);
            printf("  [4C] Real Layer 2 MLA KV Compressor (Shard 5, ratio=8 Softmax Pool + RMSNorm):\n");
            printf("       -> GPU Kernel vs Exact FP32 Reference Cosine Similarity: %.7f\n", cos_comp);
        }

        // 4D. Benchmark Engram O(1) Gather + Fused Gated Residual & Dual-Stream Ping-Pong DMA Overlap
        int64_t h_comp_toks[16] = {101, 2054, 3319, 442, 8812, 19, 77321, 904, 512, 663, 9182, 41, 77, 12003, 55, 891};
        int64_t h_hashes_l1[16 * 24];
        int64_t h_hashes_l14[16 * 24];
        dsv41_compute_engram_hashes_cpu(h_comp_toks, 16, 0, 0, h_hashes_l1);
        dsv41_compute_engram_hashes_cpu(h_comp_toks, 16, 0, 1, h_hashes_l14);

        CudaBuffer<uint8_t> d_eng_fp8(16 * 24 * 256);
        CudaBuffer<uint8_t> d_eng_s(16 * 24 * 8);
        CudaBuffer<__nv_bfloat16> d_eng_emb(16 * 6144);
        CudaBuffer<__nv_bfloat16> d_eng_kv(16 * 25600);
        CudaBuffer<__nv_bfloat16> d_eng_qw(DS_HC_DIM);
        CudaBuffer<__nv_bfloat16> d_eng_kw(DS_HC_DIM);
        d_eng_s.memset(127);

        auto time_eng = [&](int T_eng) -> float {
            dim3 g_deq(24, T_eng);
            dsv41_engram_dequant_rows_v1_kernel<<<g_deq, 256>>>(d_eng_fp8, d_eng_s, d_eng_emb);
            dsv41_engram_gated_hc_residual_v1_kernel<<<T_eng, 256>>>(d_x_hc, d_eng_kv, d_eng_qw, d_eng_kw);
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaEventRecord(ev0));
            for (int r = 0; r < 20; ++r) {
                dsv41_engram_dequant_rows_v1_kernel<<<g_deq, 256>>>(d_eng_fp8, d_eng_s, d_eng_emb);
                dsv41_engram_gated_hc_residual_v1_kernel<<<T_eng, 256>>>(d_x_hc, d_eng_kv, d_eng_qw, d_eng_kw);
            }
            CHECK_CUDA(cudaEventRecord(ev1));
            CHECK_CUDA(cudaEventSynchronize(ev1));
            float ms = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&ms, ev0, ev1));
            return (ms * 1000.0f) / 20.0f;
        };
        float us_eng_t1  = time_eng(1);
        float us_eng_t16 = time_eng(16);

        printf("  [4D] Engram N-Gram Hash (24 Primes) & Fused Signed-Sqrt Gated HC Residual:\n");
        printf("       -> Sample Token[3] Bucket Range: L1=[%lld .. %lld], L14=[%lld .. %lld]\n",
               (long long)h_hashes_l1[3 * 24 + 0],  (long long)h_hashes_l1[3 * 24 + 23],
               (long long)h_hashes_l14[3 * 24 + 0], (long long)h_hashes_l14[3 * 24 + 23]);
        printf("       -> Fused Dequant(24x256) + Signed-Sqrt Gated 4x5120 Residual: %.2f us (T=1) | %.2f us (T=16)\n",
               us_eng_t1, us_eng_t16);

        // 4E. Live Real Shard 47 (Layer 1, 101.54 GB) & Shard 48 (Layer 14, 101.54 GB) O(1) NVMe Engram Parity Test
        ScopedFile sf47("Z:/deepseek-nvfp4/model-00047-of-00048.safetensors", "rb");
        ScopedFile sf48("Z:/deepseek-nvfp4/model-00048-of-00048.safetensors", "rb");
        if (sf47 && sf48) {
            constexpr int64_t SHARD47_DATA_BASE  = 664LL;
            constexpr int64_t SHARD47_SCALE_BASE = 664LL + 98305579008LL;
            constexpr int64_t SHARD47_QW_BASE    = 664LL + 101377628352LL;
            constexpr int64_t SHARD47_KW_BASE    = 664LL + 101377669312LL;

            constexpr int64_t SHARD48_DATA_BASE  = 672LL;
            constexpr int64_t SHARD48_SCALE_BASE = 672LL + 98308270592LL;

            std::vector<uint8_t> h_l1_fp8(24 * 256);
            std::vector<uint8_t> h_l1_s(24 * 8);
            std::vector<uint8_t> h_l14_fp8(24 * 256);
            std::vector<uint8_t> h_l14_s(24 * 8);

            for (int c = 0; c < 24; ++c) {
                int64_t b1  = h_hashes_l1[3 * 24 + c];
                int64_t b14 = h_hashes_l14[3 * 24 + c];
                _fseeki64(sf47.fp, SHARD47_DATA_BASE  + b1 * 256LL, SEEK_SET);
                fread(h_l1_fp8.data() + c * 256, 1, 256, sf47.fp);
                _fseeki64(sf47.fp, SHARD47_SCALE_BASE + b1 * 8LL, SEEK_SET);
                fread(h_l1_s.data()   + c * 8,   1, 8,   sf47.fp);

                _fseeki64(sf48.fp, SHARD48_DATA_BASE  + b14 * 256LL, SEEK_SET);
                fread(h_l14_fp8.data() + c * 256, 1, 256, sf48.fp);
                _fseeki64(sf48.fp, SHARD48_SCALE_BASE + b14 * 8LL, SEEK_SET);
                fread(h_l14_s.data()   + c * 8,   1, 8,   sf48.fp);
            }

            std::vector<__nv_bfloat16> h_qw(DS_HC_DIM);
            std::vector<__nv_bfloat16> h_kw(DS_HC_DIM);
            _fseeki64(sf47.fp, SHARD47_QW_BASE, SEEK_SET);
            fread(h_qw.data(), sizeof(__nv_bfloat16), DS_HC_DIM, sf47.fp);
            _fseeki64(sf47.fp, SHARD47_KW_BASE, SEEK_SET);
            fread(h_kw.data(), sizeof(__nv_bfloat16), DS_HC_DIM, sf47.fp);

            d_eng_fp8.copy_from_host(h_l1_fp8.data(), 24 * 256);
            d_eng_s.copy_from_host(h_l1_s.data(), 24 * 8);
            d_eng_qw.copy_from_host(h_qw.data(), DS_HC_DIM);
            d_eng_kw.copy_from_host(h_kw.data(), DS_HC_DIM);

            std::vector<__nv_bfloat16> h_xhc_in(DS_HC_DIM);
            std::vector<__nv_bfloat16> h_kv_in(25600);
            for (int i = 0; i < DS_HC_DIM; ++i) {
                h_xhc_in[i] = __float2bfloat16(0.11f * sinf(float(i) * 0.019f) + 0.06f * cosf(float(i) * 0.071f));
            }
            for (int i = 0; i < 25600; ++i) {
                h_kv_in[i]  = __float2bfloat16(0.09f * cosf(float(i) * 0.023f) - 0.05f * sinf(float(i) * 0.053f));
            }
            d_x_hc.copy_from_host(h_xhc_in.data(), DS_HC_DIM);
            d_eng_kv.copy_from_host(h_kv_in.data(), 25600);

            dim3 g_deq1(24, 1);
            dsv41_engram_dequant_rows_v1_kernel<<<g_deq1, 256>>>(d_eng_fp8, d_eng_s, d_eng_emb);
            dsv41_engram_gated_hc_residual_v1_kernel<<<1, 256>>>(d_x_hc, d_eng_kv, d_eng_qw, d_eng_kw);
            CHECK_CUDA(cudaDeviceSynchronize());

            std::vector<__nv_bfloat16> h_emb_gpu(6144);
            std::vector<__nv_bfloat16> h_xhc_gpu(DS_HC_DIM);
            d_eng_emb.copy_to_host(h_emb_gpu.data(), 6144);
            d_x_hc.copy_to_host(h_xhc_gpu.data(), DS_HC_DIM);

            double dot_deq = 0.0, ng_deq = 0.0, nr_deq = 0.0;
            for (int i = 0; i < 6144; ++i) {
                float wv = host_fp8_e4m3(h_l1_fp8[i]);
                float sv = host_fp8_e8m0(h_l1_s[i >> 5]);
                float ref = __bfloat162float(__float2bfloat16(wv * sv));
                float gpu = __bfloat162float(h_emb_gpu[i]);
                dot_deq += double(gpu) * double(ref);
                ng_deq  += double(gpu) * double(gpu);
                nr_deq  += double(ref) * double(ref);
            }
            double cos_deq = dot_deq / (sqrt(ng_deq) * sqrt(nr_deq) + 1e-30);

            std::vector<float> h_ref_xhc(DS_HC_DIM);
            for (int c = 0; c < 4; ++c) {
                float sh = 0.0f, sk = 0.0f, dt = 0.0f;
                for (int d = 0; d < DS_DIM; ++d) {
                    int idx = c * DS_DIM + d;
                    float hv = __bfloat162float(h_xhc_in[idx]);
                    float kv = __bfloat162float(h_kv_in[idx]);
                    float qw = __bfloat162float(h_qw[idx]);
                    float kw = __bfloat162float(h_kw[idx]);
                    sh += hv * hv;
                    sk += kv * kv;
                    dt += hv * qw * kw * kv;
                }
                float rstd = (1.0f / sqrtf(sh / 5120.0f + 1e-20f)) * (1.0f / sqrtf(sk / 5120.0f + 1e-20f));
                float nd = dt * rstd * (1.0f / sqrtf(5120.0f));
                float ssqrt = copysignf(sqrtf(fmaxf(fabsf(nd), 1e-6f)), nd);
                float gate = 1.0f / (1.0f + expf(-ssqrt));
                for (int d = 0; d < DS_DIM; ++d) {
                    int idx = c * DS_DIM + d;
                    float hv = __bfloat162float(h_xhc_in[idx]);
                    float vv = __bfloat162float(h_kv_in[4 * DS_DIM + d]);
                    h_ref_xhc[idx] = __bfloat162float(__float2bfloat16(hv + gate * vv));
                }
            }
            double dot_res = 0.0, ng_res = 0.0, nr_res = 0.0;
            for (int i = 0; i < DS_HC_DIM; ++i) {
                float gpu = __bfloat162float(h_xhc_gpu[i]);
                float ref = h_ref_xhc[i];
                dot_res += double(gpu) * double(ref);
                ng_res  += double(gpu) * double(gpu);
                nr_res  += double(ref) * double(ref);
            }
            double cos_res = dot_res / (sqrt(ng_res) * sqrt(nr_res) + 1e-30);
            printf("  [4E] Live Shard 47 (L1, 101.5 GB) & Shard 48 (L14, 101.5 GB) O(1) Engram Parity:\n");
            printf("       -> 24-Row O(1) Gather Dequant Cosine Sim: %.7f | Signed-Sqrt Gated HC Cosine Sim: %.7f\n",
                   cos_deq, cos_res);
        }
        printf("================================================================================\n");
    }

    return 0;
}

int main() {
    return mcl4_dsv41_run_full_profile();
}
