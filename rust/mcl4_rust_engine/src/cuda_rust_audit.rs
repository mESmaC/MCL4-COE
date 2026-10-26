// ============================================================================
// MCL4-Codec `cuda-rust` (`cuda-oxide`) Integration Module:
//   1. `cuda_target_spec`: SM120 / SM120a Target & PTX ISA Floor Specification
//      (adapted from `cuda-oxide/crates/cuda-target-spec/src/lib.rs`)
//   2. `access`: Compile-Time Warp Coalescing & Shared-Memory Bank-Conflict Proofs
//      (adapted from `cuda-oxide/crates/cuda-device/src/access.rs`)
//   3. `ptx_parse`: Structural PTX Parser & Hardware Instruction / Spill Auditor
//      (adapted from `cuda-oxide/crates/ptx-parse/src/lib.rs`)
// ============================================================================

use std::fs;
use std::io;
use std::path::Path;

pub const WARP_LANES: usize = 32;
pub const GLOBAL_CACHE_LINE_BYTES: usize = 128;
pub const L1_SECTOR_BYTES: usize = 32;
pub const SHARED_BANK_COUNT: usize = 32;
pub const SHARED_BANK_WIDTH_BYTES: usize = 4;

// ============================================================================
// 1. `cuda-target-spec` — Target Architecture & PTX ISA Floor Validation
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetVariant {
    Base,
    Accelerated, // `a` suffix (e.g. `sm_120a`) unlocking architecture-specific Tensor Core MMA
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PtxVersion {
    pub major: u8,
    pub minor: u8,
}

impl PtxVersion {
    pub const fn new(major: u8, minor: u8) -> Self {
        Self { major, minor }
    }

    pub const fn at_least(self, floor: Self) -> bool {
        (self.major > floor.major) || (self.major == floor.major && self.minor >= floor.minor)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SmTarget {
    pub sm_major: u8,
    pub sm_minor: u8,
    pub variant: TargetVariant,
}

impl SmTarget {
    pub const SM_120: Self = Self {
        sm_major: 12,
        sm_minor: 0,
        variant: TargetVariant::Base,
    };

    pub const SM_120A: Self = Self {
        sm_major: 12,
        sm_minor: 0,
        variant: TargetVariant::Accelerated,
    };

    pub const fn arch_name(self) -> &'static str {
        match (self.sm_major, self.sm_minor, self.variant) {
            (12, 0, TargetVariant::Base) => "sm_120",
            (12, 0, TargetVariant::Accelerated) => "sm_120a",
            _ => "sm_unknown",
        }
    }

    pub const fn min_ptx_version(self) -> PtxVersion {
        match (self.sm_major, self.sm_minor) {
            (12, 0) => PtxVersion::new(8, 7),
            _ => PtxVersion::new(7, 0),
        }
    }

    pub const fn supports_mma_m16n8k16_bf16(self) -> bool {
        self.sm_major >= 8
    }

    pub const fn supports_sm120a_accelerated_features(self) -> bool {
        self.sm_major == 12 && self.sm_minor == 0 && matches!(self.variant, TargetVariant::Accelerated)
    }
}

// Compile-time target assertions for RTX 5070 (`sm_120a`, PTX 8.7+)
const _: () = assert!(SmTarget::SM_120A.supports_mma_m16n8k16_bf16());
const _: () = assert!(SmTarget::SM_120A.supports_sm120a_accelerated_features());
const _: () = assert!(SmTarget::SM_120A.min_ptx_version().major == 8 && SmTarget::SM_120A.min_ptx_version().minor == 7);

// ============================================================================
// 2. `cuda-device::access` — Compile-Time Warp Memory Coalescing Proofs
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AccessPlan<
    const STRIDE: usize,
    const ELEM_BYTES: usize,
    const LINE_BYTES: usize = GLOBAL_CACHE_LINE_BYTES,
>;

impl<const STRIDE: usize, const ELEM_BYTES: usize, const LINE_BYTES: usize>
    AccessPlan<STRIDE, ELEM_BYTES, LINE_BYTES>
{
    pub const fn new() -> Self {
        assert!(ELEM_BYTES > 0, "element size must be non-zero");
        assert!(LINE_BYTES > 0, "cache-line size must be non-zero");
        Self
    }

    pub const fn bytes_moved(&self) -> usize {
        WARP_LANES * ELEM_BYTES
    }

    pub const fn span_bytes(&self) -> usize {
        if STRIDE == 0 {
            ELEM_BYTES
        } else {
            (WARP_LANES - 1) * STRIDE * ELEM_BYTES + ELEM_BYTES
        }
    }

    pub const fn lines_touched(&self) -> usize {
        let span = self.span_bytes();
        let line_span = (span + LINE_BYTES - 1) / LINE_BYTES;
        if line_span < WARP_LANES {
            line_span
        } else {
            WARP_LANES
        }
    }

    pub const fn ideal_lines(&self) -> usize {
        if STRIDE == 0 {
            (ELEM_BYTES + LINE_BYTES - 1) / LINE_BYTES
        } else {
            let moved = self.bytes_moved();
            (moved + LINE_BYTES - 1) / LINE_BYTES
        }
    }

    pub const fn wasted_lines(&self) -> usize {
        self.lines_touched().saturating_sub(self.ideal_lines())
    }

    pub const fn is_unit_stride(&self) -> bool {
        STRIDE == 1
    }

    pub const fn is_fully_coalesced(&self) -> bool {
        self.wasted_lines() == 0 && (STRIDE == 0 || STRIDE == 1)
    }
}

/// Checks whether a 2D shared-memory row stride (`ROW_STRIDE_WORDS` in 32-bit words)
/// allows a 4-thread `m16n8k16` group (`group = 0..7`, `thread = 0..3`) to read
/// `s_tile[group][c_base + thread]` across all 32 lanes without 32-way bank conflicts.
pub const fn m16n8k16_smem_stride_conflict_degree(row_stride_words: usize) -> usize {
    // In mma.sync.aligned.m16n8k16:
    // lane = group * 4 + thread (group = 0..7, thread = 0..3).
    // Address in 32-bit words for lane is: group * row_stride_words + thread.
    // Bank index is (group * row_stride_words + thread) % 32.
    let mut max_hits = 0usize;
    let mut bank = 0usize;
    while bank < SHARED_BANK_COUNT {
        let mut hits = 0usize;
        let mut lane = 0usize;
        while lane < WARP_LANES {
            let group = lane >> 2;
            let thread = lane & 3;
            let word_addr = group * row_stride_words + thread;
            if (word_addr & 31) == bank {
                hits += 1;
            }
            lane += 1;
        }
        if hits > max_hits {
            max_hits = hits;
        }
        bank += 1;
    }
    max_hits
}

// Compile-time coalescing & bank-conflict proofs for MCL4-Codec v7 SM120a kernels:
// 1. `dg_ld_l2_128b_u4` / `dg_ld_evict_first_u4` / `w4_row[t*32 + lane]` (16-byte uint4/float4, stride=1):
pub const PLAN_U4_COALESCED: AccessPlan<1, 16> = AccessPlan::new();
const _: () = assert!(PLAN_U4_COALESCED.is_fully_coalesced());
const _: () = assert!(PLAN_U4_COALESCED.lines_touched() == 4);
const _: () = assert!(PLAN_U4_COALESCED.wasted_lines() == 0);

// 2. `w0_u2[t*32 + lane]` in `dsv41_fp8_e8m0_gemm_v1_kernel` (8-byte uint2, stride=1):
pub const PLAN_U2_COALESCED: AccessPlan<1, 8> = AccessPlan::new();
const _: () = assert!(PLAN_U2_COALESCED.is_fully_coalesced());
const _: () = assert!(PLAN_U2_COALESCED.lines_touched() == 2);
const _: () = assert!(PLAN_U2_COALESCED.wasted_lines() == 0);

// 3. `w_dec[j * 5120 + col]` in `dsv41_ncm_ple_lora_fwd_dec_batched_v1_kernel` (4-byte f32, stride=1):
pub const PLAN_F32_COALESCED: AccessPlan<1, 4> = AccessPlan::new();
const _: () = assert!(PLAN_F32_COALESCED.is_fully_coalesced());
const _: () = assert!(PLAN_F32_COALESCED.lines_touched() == 1);
const _: () = assert!(PLAN_F32_COALESCED.wasted_lines() == 0);

// 4. Shared memory `s_A[16][68]` / `s_B[8][68]` (stride = 68 words = 64 + 4 pad) and `s_A[16][132]` (stride = 132 = 128 + 4 pad):
//    Without +4 padding (stride = 64 or 128, which is 0 mod 32), all 8 groups hit the SAME 4 banks (8-way bank conflict)!
//    With +4 padding (stride = 68 or 132, which is 4 mod 32), each group shifts by +4 banks -> 0 bank conflicts (degree = 1)!
const _: () = assert!(m16n8k16_smem_stride_conflict_degree(64) == 8);  // Unpadded = 8-way conflict
const _: () = assert!(m16n8k16_smem_stride_conflict_degree(68) == 1);  // +4 Padded = 100% conflict-free!
const _: () = assert!(m16n8k16_smem_stride_conflict_degree(132) == 1); // +4 Padded = 100% conflict-free!

// ============================================================================
// 3. `ptx-parse` — Structural SM120a PTX Module Parser & Instruction Auditor
// ============================================================================

#[derive(Clone, Debug, Default)]
pub struct PtxKernelAudit {
    pub name: String,
    pub reg_b32: u32,
    pub reg_f32: u32,
    pub reg_b64: u32,
    pub reg_pred: u32,
    pub shared_bytes: u64,
    pub local_spill_bytes: u64,
    pub mma_sync_count: usize,
    pub cvt_bf16x2_count: usize,
    pub ex2_approx_count: usize,
    pub rcp_approx_count: usize,
    pub ld_l2_128b_count: usize,
    pub ld_evict_first_count: usize,
}

#[derive(Clone, Debug, Default)]
pub struct PtxModuleAudit {
    pub ptx_version: (u8, u8),
    pub target: String,
    pub address_size: u8,
    pub kernels: Vec<PtxKernelAudit>,
    pub total_mma_sync: usize,
    pub total_cvt_bf16x2: usize,
    pub total_ex2_approx: usize,
    pub total_rcp_approx: usize,
    pub total_ld_l2_128b: usize,
    pub total_ld_evict_first: usize,
    pub total_local_spill_bytes: u64,
}

pub fn parse_and_audit_ptx_str(src: &str) -> PtxModuleAudit {
    let mut audit = PtxModuleAudit::default();
    let mut current_kernel: Option<PtxKernelAudit> = None;

    for raw_line in src.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with("//") {
            continue;
        }

        if line.starts_with(".version ") {
            let ver_str = line.trim_start_matches(".version ").trim();
            if let Some((maj, min)) = ver_str.split_once('.') {
                let min_clean = min.split_whitespace().next().unwrap_or("0");
                audit.ptx_version = (
                    maj.parse::<u8>().unwrap_or(0),
                    min_clean.parse::<u8>().unwrap_or(0),
                );
            }
            continue;
        }

        if line.starts_with(".target ") {
            audit.target = line
                .trim_start_matches(".target ")
                .trim()
                .trim_end_matches(',')
                .to_string();
            continue;
        }

        if line.starts_with(".address_size ") {
            let sz = line.trim_start_matches(".address_size ").trim();
            audit.address_size = sz.parse::<u8>().unwrap_or(0);
            continue;
        }

        if let Some(rest) = line.strip_prefix(".visible .entry ") {
            if let Some(prev) = current_kernel.take() {
                audit.kernels.push(prev);
            }
            let kname = rest
                .split('(')
                .next()
                .unwrap_or("")
                .trim()
                .to_string();
            current_kernel = Some(PtxKernelAudit {
                name: kname,
                ..Default::default()
            });
            continue;
        }

        if let Some(ref mut k) = current_kernel {
            // Check `.reg` bank declarations: e.g. `.reg .b32 %r<124>;`
            if line.starts_with(".reg ") {
                if let (Some(lt), Some(gt)) = (line.find('<'), line.find('>')) {
                    if gt > lt + 1 {
                        let count = line[lt + 1..gt].parse::<u32>().unwrap_or(0);
                        if line.contains(".b32") || line.contains(".u32") || line.contains(".s32") {
                            k.reg_b32 += count;
                        } else if line.contains(".f32") {
                            k.reg_f32 += count;
                        } else if line.contains(".b64") || line.contains(".u64") || line.contains(".s64") {
                            k.reg_b64 += count;
                        } else if line.contains(".pred") {
                            k.reg_pred += count;
                        }
                    }
                }
            } else if line.starts_with(".shared ") {
                if let (Some(lb), Some(rb)) = (line.find('['), line.find(']')) {
                    if rb > lb + 1 {
                        let count = line[lb + 1..rb].parse::<u64>().unwrap_or(0);
                        let elem_sz = if line.contains(".b64") || line.contains(".u64") || line.contains(".f64") {
                            8
                        } else if line.contains(".b32") || line.contains(".u32") || line.contains(".f32") {
                            4
                        } else if line.contains(".b16") || line.contains(".u16") {
                            2
                        } else {
                            1
                        };
                        k.shared_bytes += count * elem_sz;
                    }
                }
            } else if line.starts_with(".local ") {
                if let (Some(lb), Some(rb)) = (line.find('['), line.find(']')) {
                    if rb > lb + 1 {
                        let count = line[lb + 1..rb].parse::<u64>().unwrap_or(0);
                        k.local_spill_bytes += count;
                        audit.total_local_spill_bytes += count;
                    }
                }
            } else {
                if line.contains("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32") {
                    k.mma_sync_count += 1;
                    audit.total_mma_sync += 1;
                }
                if line.contains("cvt.rn.bf16x2.f32") {
                    k.cvt_bf16x2_count += 1;
                    audit.total_cvt_bf16x2 += 1;
                }
                if line.contains("ex2.approx.f32") {
                    k.ex2_approx_count += 1;
                    audit.total_ex2_approx += 1;
                }
                if line.contains("rcp.approx.ftz.f32") {
                    k.rcp_approx_count += 1;
                    audit.total_rcp_approx += 1;
                }
                if line.contains("ld.global.nc.L2::128B") {
                    k.ld_l2_128b_count += 1;
                    audit.total_ld_l2_128b += 1;
                }
                if line.contains("ld.L1::no_allocate.L2::cache_hint") {
                    k.ld_evict_first_count += 1;
                    audit.total_ld_evict_first += 1;
                }
            }
        }
    }

    if let Some(prev) = current_kernel.take() {
        audit.kernels.push(prev);
    }

    audit
}

pub fn parse_and_audit_ptx_file(path: &Path) -> io::Result<PtxModuleAudit> {
    let content = fs::read_to_string(path)?;
    Ok(parse_and_audit_ptx_str(&content))
}
