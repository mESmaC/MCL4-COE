//! Intel DC P4608 (`ICDPC5ED2ORA6.4T`) Dual-Controller (`X:` + `Z:`) 4K-Aligned
//! Single-Expert Sharder (`.mcl4e`) & Zero-Overhead 1-Syscall Streamer.
//!
//! Architecture:
//!   - Each `Dsv41Mcl4cExpertPacket` is `19,910,656` bytes (`4,861 × 4096 B`), an exact
//!     multiple of the 4096-byte physical NVMe sector size with `#[repr(C, align(4096))]`.
//!   - Instead of performing 9 scattered non-4K-aligned seeks per expert inside an 11.1 GB
//!     monolithic `.safetensors` file on a single drive, this module splits experts into
//!     standalone 4K-aligned `.mcl4e` files striped across BOTH independent NVMe controllers
//!     of the Intel P4608 board by layer parity (`layer & 1`):
//!       * Even `layer_id` (`L % 2 == 0`) -> `X:\deepseek_v41_experts\L{layer:02}\E{expert:03}.mcl4e` (Controller 0)
//!       * Odd  `layer_id` (`L % 2 == 1`) -> `Z:\deepseek_v41_experts\L{layer:02}\E{expert:03}.mcl4e` (Controller 1)
//!   - Why Layer-Parity Striping + Per-Expert `.mcl4e` Files Is Optimal for NCM Lookahead:
//!       * While Layer `L` executes/streams on Controller 0 (`X:`), our 384-Expert NCM Lookahead
//!         prefetches Layer `L+1` on Controller 1 (`Z:`) with ZERO NVMe controller contention!
//!       * Every `.mcl4e` file is read in ONE contiguous `seek_read(dst, 0)` syscall using cached
//!         file handles (`FILE_FLAG_SEQUENTIAL_SCAN`), eliminating 8 out of 9 syscalls, all JSON/HashMap
//!         lookups, and Windows `CreateFileW` filter-driver overhead!

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
#[cfg(not(windows))]
use std::io::Read;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::Instant;

#[cfg(windows)]
use std::os::windows::fs::{FileExt, OpenOptionsExt};

use crate::{Dsv41Mcl4cExpertPacket, SafetensorsShardIndex, DS_PACKET_BYTES};

#[cfg(windows)]
const FILE_FLAG_SEQUENTIAL_SCAN: u32 = 0x0800_0000;
#[cfg(windows)]
const FILE_FLAG_NO_BUFFERING: u32 = 0x2000_0000;

/// Manages striped 4K-aligned single-expert `.mcl4e` shards across the two NVMe controllers
/// of the Intel DC P4608 (`X:` = Controller 0 for Even Layers, `Z:` = Controller 1 for Odd Layers).
pub struct P4608DualDriveExpertStore {
    pub ctrl0_root: PathBuf, // X:\deepseek_v41_experts (Even layers: L % 2 == 0)
    pub ctrl1_root: PathBuf, // Z:\deepseek_v41_experts (Odd  layers: L % 2 == 1)
    handle_cache: RwLock<HashMap<(usize, usize), Arc<File>>>,
}

impl P4608DualDriveExpertStore {
    pub fn new(ctrl0_root: impl Into<PathBuf>, ctrl1_root: impl Into<PathBuf>) -> Self {
        Self {
            ctrl0_root: ctrl0_root.into(),
            ctrl1_root: ctrl1_root.into(),
            handle_cache: RwLock::new(HashMap::with_capacity(256)),
        }
    }

    pub fn default_p4608() -> Self {
        Self::new(r"X:\deepseek_v41_experts", r"Z:\deepseek_v41_experts")
    }

    /// O(1) deterministic path resolution striping even layers (`L % 2 == 0`) to Controller 0 (`X:`)
    /// and odd layers (`L % 2 == 1`) to Controller 1 (`Z:`), so Layer `L` execution and Layer `L+1`
    /// NCM-Lookahead prefetching always hit opposite physical NVMe controllers!
    #[inline]
    pub fn expert_path(&self, layer: usize, expert: usize) -> PathBuf {
        let root = if (layer & 1) == 0 {
            &self.ctrl0_root
        } else {
            &self.ctrl1_root
        };
        root.join(format!("L{:02}", layer))
            .join(format!("E{:03}.mcl4e", expert))
    }

    /// Checks if the 4K-aligned `.mcl4e` shard file already exists with exact `19,910,656` byte size.
    #[inline]
    pub fn is_expert_sharded(&self, layer: usize, expert: usize) -> bool {
        let p = self.expert_path(layer, expert);
        match fs::metadata(&p) {
            Ok(m) => m.is_file() && m.len() == (DS_PACKET_BYTES as u64),
            Err(_) => false,
        }
    }

    #[inline]
    fn get_or_open_handle(&self, layer: usize, expert: usize) -> io::Result<Arc<File>> {
        if let Some(f) = self.handle_cache.read().unwrap().get(&(layer, expert)) {
            return Ok(Arc::clone(f));
        }
        let path = self.expert_path(layer, expert);
        #[cfg(windows)]
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN)
            .open(&path)?;
        #[cfg(not(windows))]
        let file = File::open(&path)?;

        let arc_f = Arc::new(file);
        let mut cache = self.handle_cache.write().unwrap();
        if cache.len() >= 2048 {
            cache.clear();
        }
        cache.insert((layer, expert), Arc::clone(&arc_f));
        Ok(arc_f)
    }

    /// Deterministic path for the 4K-aligned per-layer Router + Norm + Hyper-Connection bundle
    /// (`META.mcl4m`, `5,910,528 B = 1,443 × 4096 B`), striped by layer parity (`X:` even, `Z:` odd).
    #[inline]
    pub fn layer_meta_path(&self, layer: usize) -> PathBuf {
        let root = if (layer & 1) == 0 {
            &self.ctrl0_root
        } else {
            &self.ctrl1_root
        };
        root.join(format!("L{:02}", layer)).join("META.mcl4m")
    }

    /// Exact byte size of `META.mcl4m` (`1,443 × 4096 = 5,910,528` bytes).
    pub const LAYER_META_BYTES: usize = 5_910_528;

    #[inline]
    pub fn is_layer_meta_sharded(&self, layer: usize) -> bool {
        let p = self.layer_meta_path(layer);
        match fs::metadata(&p) {
            Ok(m) => m.is_file() && m.len() == (Self::LAYER_META_BYTES as u64),
            Err(_) => false,
        }
    }

    /// Writes `L{layer:02}\META.mcl4m` (`5,910,528 B = 1,443 × 4096 B`) containing:
    ///   - `[0 .. 1,966,080)`: `hc_ffn_fn` (`24 * 20480 * f32`)
    ///   - `[1,966,080 .. 5,898,240)`: `ffn.gate.weight` (`384 * 5120 * u16`)
    ///   - `[5,898,240 .. 5,908,480)`: `ffn_norm.weight` (`5120 * u16`)
    ///   - `[5,908,480 .. 5,910,016)`: `ffn.gate.bias` (`384 * f32`)
    ///   - `[5,910,016 .. 5,910,112)`: `hc_ffn_base` (`24 * f32`)
    ///   - `[5,910,112 .. 5,910,124)`: `hc_ffn_scale` (`3 * f32`)
    ///   - `[5,910,124 .. 5,910,528)`: zero-padding to 4096B sector boundary
    pub fn write_layer_meta_shard(
        &self,
        index: &SafetensorsShardIndex,
        layer: usize,
    ) -> io::Result<PathBuf> {
        let out_path = self.layer_meta_path(layer);
        if let Some(parent) = out_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut buf = vec![0u8; Self::LAYER_META_BYTES];
        index.read_tensor_exact(&format!("layers.{}.hc_ffn_fn", layer), &mut buf[0..1_966_080])?;
        index.read_tensor_exact(&format!("layers.{}.ffn.gate.weight", layer), &mut buf[1_966_080..5_898_240])?;
        index.read_tensor_exact(&format!("layers.{}.ffn_norm.weight", layer), &mut buf[5_898_240..5_908_480])?;
        index.read_tensor_exact(&format!("layers.{}.ffn.gate.bias", layer), &mut buf[5_908_480..5_910_016])?;
        index.read_tensor_exact(&format!("layers.{}.hc_ffn_base", layer), &mut buf[5_910_016..5_910_112])?;
        index.read_tensor_exact(&format!("layers.{}.hc_ffn_scale", layer), &mut buf[5_910_112..5_910_124])?;

        let mut f = File::create(&out_path)?;
        f.write_all(&buf)?;
        Ok(out_path)
    }

    /// Reads `L{layer:02}\META.mcl4m` in a single contiguous read and unpacks into caller slices.
    pub fn read_layer_meta_single_syscall(
        &self,
        layer: usize,
        fn_dst: &mut [u8],
        gw_dst: &mut [u8],
        nw_dst: &mut [u8],
        gb_dst: &mut [u8],
        bs_dst: &mut [u8],
        sc_dst: &mut [u8],
    ) -> io::Result<()> {
        let path = self.layer_meta_path(layer);
        #[cfg(windows)]
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(FILE_FLAG_SEQUENTIAL_SCAN)
            .open(&path)?;
        #[cfg(not(windows))]
        let mut file = File::open(&path)?;

        let mut buf = vec![0u8; Self::LAYER_META_BYTES];
        #[cfg(windows)]
        {
            let mut read_total = 0usize;
            while read_total < Self::LAYER_META_BYTES {
                let n = file.seek_read(&mut buf[read_total..], read_total as u64)?;
                if n == 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "Unexpected EOF reading META.mcl4m",
                    ));
                }
                read_total += n;
            }
        }
        #[cfg(not(windows))]
        {
            file.read_exact(&mut buf)?;
        }

        fn_dst.copy_from_slice(&buf[0..1_966_080]);
        gw_dst.copy_from_slice(&buf[1_966_080..5_898_240]);
        nw_dst.copy_from_slice(&buf[5_898_240..5_908_480]);
        gb_dst.copy_from_slice(&buf[5_908_480..5_910_016]);
        bs_dst.copy_from_slice(&buf[5_910_016..5_910_112]);
        sc_dst.copy_from_slice(&buf[5_910_112..5_910_124]);
        Ok(())
    }

    /// Ensures `META.mcl4m` is sharded for all requested layers across `X:` (Even) and `Z:` (Odd).
    pub fn ensure_layers_meta_sharded(
        &self,
        index: &Arc<SafetensorsShardIndex>,
        layers: &[usize],
    ) -> io::Result<(usize, usize, f64)> {
        let t0 = Instant::now();
        let mut missing = Vec::new();
        let mut cached = 0usize;
        for &l in layers {
            if self.is_layer_meta_sharded(l) {
                cached += 1;
            } else {
                missing.push(l);
            }
        }
        if !missing.is_empty() {
            thread::scope(|s| -> io::Result<()> {
                let mut handles = Vec::new();
                for &l in &missing {
                    let idx_clone = Arc::clone(index);
                    let store_ref = self;
                    handles.push(s.spawn(move || -> io::Result<()> {
                        store_ref.write_layer_meta_shard(&idx_clone, l).map(|_| ())
                    }));
                }
                for h in handles {
                    h.join().unwrap()?;
                }
                Ok(())
            })?;
        }
        let dt_ms = t0.elapsed().as_secs_f64() * 1000.0;
        Ok((missing.len(), cached, dt_ms))
    }

    /// Exact byte size of `SHARED.mcl4s` (`8,649 × 4096 = 35,426,304` bytes = `33.79 MiB`).
    pub const SHARED_EXPERT_BYTES: usize = 35_426_304;
    /// Exact byte size of `ATTN.mcl4a` (`31,426 × 4096 = 128,720,896` bytes = `122.76 MiB`).
    pub const LAYER_ATTN_BYTES: usize = 128_720_896;

    #[inline]
    pub fn layer_shared_path(&self, layer: usize) -> PathBuf {
        let root = if (layer & 1) == 0 {
            &self.ctrl0_root
        } else {
            &self.ctrl1_root
        };
        root.join(format!("L{:02}", layer)).join("SHARED.mcl4s")
    }

    #[inline]
    pub fn layer_attn_path(&self, layer: usize) -> PathBuf {
        let root = if (layer & 1) == 0 {
            &self.ctrl0_root
        } else {
            &self.ctrl1_root
        };
        root.join(format!("L{:02}", layer)).join("ATTN.mcl4a")
    }

    #[inline]
    pub fn is_layer_shared_sharded(&self, layer: usize) -> bool {
        let p = self.layer_shared_path(layer);
        match fs::metadata(&p) {
            Ok(m) => m.is_file() && m.len() == (Self::SHARED_EXPERT_BYTES as u64),
            Err(_) => false,
        }
    }

    #[inline]
    pub fn is_layer_attn_sharded(&self, layer: usize) -> bool {
        let p = self.layer_attn_path(layer);
        match fs::metadata(&p) {
            Ok(m) => m.is_file() && m.len() == (Self::LAYER_ATTN_BYTES as u64),
            Err(_) => false,
        }
    }

    pub fn write_layer_shared_shard(
        &self,
        index: &SafetensorsShardIndex,
        layer: usize,
    ) -> io::Result<PathBuf> {
        let out_path = self.layer_shared_path(layer);
        if let Some(parent) = out_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut buf = vec![0u8; Self::SHARED_EXPERT_BYTES];
        let pfx = format!("layers.{}.ffn.shared_experts", layer);
        index.read_tensor_exact(&format!("{}.w1.weight", pfx), &mut buf[0..11_796_480])?;
        index.read_tensor_exact(&format!("{}.w3.weight", pfx), &mut buf[11_796_480..23_592_960])?;
        index.read_tensor_exact(&format!("{}.w2.weight", pfx), &mut buf[23_592_960..35_389_440])?;
        index.read_tensor_exact(&format!("{}.w1.scale", pfx), &mut buf[35_389_440..35_400_960])?;
        index.read_tensor_exact(&format!("{}.w3.scale", pfx), &mut buf[35_400_960..35_412_480])?;
        index.read_tensor_exact(&format!("{}.w2.scale", pfx), &mut buf[35_412_480..35_424_000])?;

        let mut f = File::create(&out_path)?;
        f.write_all(&buf)?;
        Ok(out_path)
    }

    pub fn write_layer_attn_shard(
        &self,
        index: &SafetensorsShardIndex,
        layer: usize,
    ) -> io::Result<PathBuf> {
        let out_path = self.layer_attn_path(layer);
        if let Some(parent) = out_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut buf = vec![0u8; Self::LAYER_ATTN_BYTES];
        let pfx = format!("layers.{}", layer);
        let mut off = 0usize;
        let mut read_into = |name: &str, len: usize, b: &mut [u8]| -> io::Result<()> {
            index.read_tensor_exact(name, &mut b[off..off + len])?;
            off += len;
            Ok(())
        };
        read_into(&format!("{}.hc_attn_fn", pfx), 1_966_080, &mut buf)?;
        read_into(&format!("{}.hc_attn_base", pfx), 96, &mut buf)?;
        read_into(&format!("{}.hc_attn_scale", pfx), 12, &mut buf)?;
        read_into(&format!("{}.attn_norm.weight", pfx), 10_240, &mut buf)?;
        read_into(&format!("{}.attn.q_norm.weight", pfx), 2_560, &mut buf)?;
        read_into(&format!("{}.attn.kv_norm.weight", pfx), 1_024, &mut buf)?;
        read_into(&format!("{}.attn.attn_sink", pfx), 256, &mut buf)?;
        read_into(&format!("{}.attn.wq_a.weight", pfx), 6_553_600, &mut buf)?;
        read_into(&format!("{}.attn.wq_a.scale", pfx), 6_400, &mut buf)?;
        read_into(&format!("{}.attn.wq_b.weight", pfx), 41_943_040, &mut buf)?;
        read_into(&format!("{}.attn.wq_b.scale", pfx), 40_960, &mut buf)?;
        read_into(&format!("{}.attn.wkv.weight", pfx), 2_621_440, &mut buf)?;
        read_into(&format!("{}.attn.wkv.scale", pfx), 2_560, &mut buf)?;
        read_into(&format!("{}.attn.wo_a.weight", pfx), 33_554_432, &mut buf)?;
        read_into(&format!("{}.attn.wo_a.scale", pfx), 32_768, &mut buf)?;
        read_into(&format!("{}.attn.wo_b.weight", pfx), 41_943_040, &mut buf)?;
        read_into(&format!("{}.attn.wo_b.scale", pfx), 40_960, &mut buf)?;

        let mut f = File::create(&out_path)?;
        f.write_all(&buf)?;
        Ok(out_path)
    }

    pub fn ensure_layers_shared_and_attn_sharded(
        &self,
        index: &Arc<SafetensorsShardIndex>,
        layers: &[usize],
    ) -> io::Result<(usize, usize, f64)> {
        let t0 = Instant::now();
        let mut missing_sh = Vec::new();
        let mut missing_at = Vec::new();
        let mut cached = 0usize;
        for &l in layers {
            if self.is_layer_shared_sharded(l) {
                cached += 1;
            } else {
                missing_sh.push(l);
            }
            if self.is_layer_attn_sharded(l) {
                cached += 1;
            } else {
                missing_at.push(l);
            }
        }
        let new_cnt = missing_sh.len() + missing_at.len();
        if new_cnt > 0 {
            thread::scope(|s| -> io::Result<()> {
                let mut handles = Vec::new();
                for &l in &missing_sh {
                    let idx_clone = Arc::clone(index);
                    let store_ref = self;
                    handles.push(s.spawn(move || -> io::Result<()> {
                        store_ref.write_layer_shared_shard(&idx_clone, l).map(|_| ())
                    }));
                }
                for &l in &missing_at {
                    let idx_clone = Arc::clone(index);
                    let store_ref = self;
                    handles.push(s.spawn(move || -> io::Result<()> {
                        store_ref.write_layer_attn_shard(&idx_clone, l).map(|_| ())
                    }));
                }
                for h in handles {
                    h.join().unwrap()?;
                }
                Ok(())
            })?;
        }
        let dt_ms = t0.elapsed().as_secs_f64() * 1000.0;
        Ok((new_cnt, cached, dt_ms))
    }

    /// Writes a single `Dsv41Mcl4cExpertPacket` (`19,910,656 B = 4,861 × 4096 B`) to its
    /// striped `.mcl4e` destination on `X:` or `Z:`.
    pub fn write_expert_shard(
        &self,
        index: &SafetensorsShardIndex,
        layer: usize,
        expert: usize,
        scratch_pkt: &mut Dsv41Mcl4cExpertPacket,
    ) -> io::Result<PathBuf> {
        let out_path = self.expert_path(layer, expert);
        if let Some(parent) = out_path.parent() {
            fs::create_dir_all(parent)?;
        }

        // Assemble the 9 tensors from the source .safetensors shard into the 4K-aligned packet
        index.stream_expert_packet(layer, expert, scratch_pkt)?;
        scratch_pkt._pad.fill(0);

        let raw_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                (scratch_pkt as *const Dsv41Mcl4cExpertPacket) as *const u8,
                DS_PACKET_BYTES,
            )
        };

        let mut f = File::create(&out_path)?;
        f.write_all(raw_bytes)?;
        f.flush()?;
        Ok(out_path)
    }

    /// Ensures a list of `(layer, expert)` pairs are sharded into `.mcl4e` files across `X:` and `Z:`,
    /// extracting any missing files in parallel and pre-warming their OS file handles.
    pub fn ensure_experts_sharded(
        &self,
        index: &Arc<SafetensorsShardIndex>,
        targets: &[(usize, usize)],
    ) -> io::Result<(usize, usize, f64)> {
        let t0 = Instant::now();
        let mut missing = Vec::new();
        let mut cached = 0usize;

        for &(l, e) in targets {
            if self.is_expert_sharded(l, e) {
                cached += 1;
            } else {
                missing.push((l, e));
            }
        }

        if !missing.is_empty() {
            for &(l, e) in &missing {
                if let Some(parent) = self.expert_path(l, e).parent() {
                    fs::create_dir_all(parent)?;
                }
            }

            let num_workers = missing.len().min(16).max(1);
            let chunk_size = (missing.len() + num_workers - 1) / num_workers;

            thread::scope(|s| -> io::Result<()> {
                let mut handles = Vec::new();
                for chunk in missing.chunks(chunk_size) {
                    let idx_clone = Arc::clone(index);
                    let store_ref = self;
                    handles.push(s.spawn(move || -> io::Result<()> {
                        let mut pkt = Dsv41Mcl4cExpertPacket::new_zeroed_box();
                        for &(l, e) in chunk {
                            store_ref.write_expert_shard(&idx_clone, l, e, &mut pkt)?;
                        }
                        Ok(())
                    }));
                }
                for h in handles {
                    h.join().unwrap()?;
                }
                Ok(())
            })?;
        }

        // Pre-warm open file handles for small active batches (`<= 64`)
        if targets.len() <= 64 {
            for &(l, e) in targets {
                let _ = self.get_or_open_handle(l, e)?;
            }
        }

        let dt_ms = t0.elapsed().as_secs_f64() * 1000.0;
        Ok((missing.len(), cached, dt_ms))
    }

    /// High-throughput dual-controller parallel sharder across a list of layers and experts per layer.
    /// Interleaves `(Even Layer -> X:, Odd Layer -> Z:)` tasks across 16 worker threads so both
    /// Intel DC P4608 NVMe controllers write concurrently at maximum bandwidth.
    pub fn shard_layers_parallel(
        &self,
        index: &Arc<SafetensorsShardIndex>,
        layers: &[usize],
        experts_per_layer: &[usize],
    ) -> io::Result<(usize, usize, f64)> {
        let t0 = Instant::now();

        // 1. Ensure all layer META.mcl4m, SHARED.mcl4s, and ATTN.mcl4a bundles exist across X: and Z:
        let (meta_new, meta_cached, meta_ms) = self.ensure_layers_meta_sharded(index, layers)?;
        let (sa_new, sa_cached, sa_ms) = self.ensure_layers_shared_and_attn_sharded(index, layers)?;
        println!(
            "    [P4608 META.mcl4m + SHARED.mcl4s + ATTN.mcl4a] Layers={}: meta(new={}, cached={}, {:.2} ms) | shared+attn(new={}, cached={}, {:.2} ms)",
            layers.len(),
            meta_new,
            meta_cached,
            meta_ms,
            sa_new,
            sa_cached,
            sa_ms
        );

        // 2. Interleave (layer, expert) pairs across layers so adjacent items in `missing`
        // alternate between Even layers (`X:`) and Odd layers (`Z:`)!
        let mut missing = Vec::with_capacity(layers.len() * experts_per_layer.len());
        let mut cached = 0usize;
        for &l in layers {
            let dir_path = if (l & 1) == 0 {
                self.ctrl0_root.join(format!("L{:02}", l))
            } else {
                self.ctrl1_root.join(format!("L{:02}", l))
            };
            fs::create_dir_all(&dir_path)?;
        }

        for &e in experts_per_layer {
            for &l in layers {
                if self.is_expert_sharded(l, e) {
                    cached += 1;
                } else {
                    missing.push((l, e));
                }
            }
        }

        let total_new = missing.len();
        if total_new > 0 {
            let num_workers = total_new.min(16).max(1);
            // Round-robin assignment across 16 workers preserves Even/Odd controller balance
            thread::scope(|s| -> io::Result<()> {
                let mut handles = Vec::with_capacity(num_workers);
                let missing_ref = &missing;
                for worker_id in 0..num_workers {
                    let idx_clone = Arc::clone(index);
                    let store_ref = self;
                    handles.push(s.spawn(move || -> io::Result<()> {
                        let mut pkt = Dsv41Mcl4cExpertPacket::new_zeroed_box();
                        let mut idx = worker_id;
                        while idx < missing_ref.len() {
                            let (l, e) = missing_ref[idx];
                            store_ref.write_expert_shard(&idx_clone, l, e, &mut pkt)?;
                            idx += num_workers;
                        }
                        Ok(())
                    }));
                }
                for h in handles {
                    h.join().unwrap()?;
                }
                Ok(())
            })?;
        }

        let dt_ms = t0.elapsed().as_secs_f64() * 1000.0;
        Ok((total_new, cached, dt_ms))
    }

    /// Reads a single 4K-aligned `.mcl4e` expert shard (`19,910,656` B) from `X:` or `Z:`
    /// in a single contiguous read at `offset = 0` (replacing 9 scattered seeks + 9 HashMap lookups).
    pub fn read_expert_single_syscall(
        &self,
        layer: usize,
        expert: usize,
        dst_pkt: &mut Dsv41Mcl4cExpertPacket,
    ) -> io::Result<()> {
        let file = self.get_or_open_handle(layer, expert)?;
        let dst_ptr = (dst_pkt as *mut Dsv41Mcl4cExpertPacket) as *mut u8;
        let dst_bytes: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(dst_ptr, DS_PACKET_BYTES) };

        #[cfg(windows)]
        {
            let mut read_total = 0usize;
            while read_total < DS_PACKET_BYTES {
                let n = file.seek_read(&mut dst_bytes[read_total..], read_total as u64)?;
                if n == 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "Unexpected EOF reading .mcl4e shard",
                    ));
                }
                read_total += n;
            }
            return Ok(());
        }

        #[cfg(not(windows))]
        {
            let mut f = file.as_ref();
            f.read_exact(dst_bytes)?;
            Ok(())
        }
    }

    /// Streams a batch of `(layer, expert)` packets concurrently across BOTH Intel P4608
    /// NVMe controllers (`X:` and `Z:`) using 1-syscall `.mcl4e` reads!
    pub fn stream_experts_striped(
        &self,
        targets: &[(usize, usize)],
        out_packets: &mut [Dsv41Mcl4cExpertPacket],
    ) -> io::Result<(f64, usize, usize)> {
        assert_eq!(targets.len(), out_packets.len());
        let mut ctrl0_x = 0usize;
        let mut ctrl1_z = 0usize;
        for &(l, _) in targets {
            if (l & 1) == 0 {
                ctrl0_x += 1;
            } else {
                ctrl1_z += 1;
            }
        }

        let t0 = Instant::now();
        thread::scope(|s| -> io::Result<()> {
            let mut handles = Vec::with_capacity(targets.len());
            for (&(layer, expert), pkt_slot) in targets.iter().zip(out_packets.iter_mut()) {
                let store_ref = self;
                handles.push(s.spawn(move || -> io::Result<()> {
                    store_ref.read_expert_single_syscall(layer, expert, pkt_slot)
                }));
            }
            for h in handles {
                h.join().unwrap()?;
            }
            Ok(())
        })?;
        let dt_ms = t0.elapsed().as_secs_f64() * 1000.0;
        Ok((dt_ms, ctrl0_x, ctrl1_z))
    }
}

/// Confidence-Adaptive Top-K Router Filter for High-Throughput Inference & Rollouts:
/// Given sorted descending router weights `w_sorted[0..6]` (which sum to `route_scale = 1.5`),
/// returns the active expert count `k_active` in `[k_min..=6]` once cumulative probability mass
/// reaches `confidence_threshold` (e.g., `0.90` = 90% of total routing weight), renormalizing
/// the surviving weights so output magnitude is preserved.
#[inline]
pub fn adaptive_topk_filter(
    indices: &mut [usize; 6],
    weights: &mut [f32; 6],
    confidence_threshold: f32,
    k_min: usize,
) -> usize {
    let total_w: f32 = weights.iter().sum();
    if total_w <= 1e-12 || confidence_threshold >= 0.9999 {
        return 6;
    }
    let target_w = total_w * confidence_threshold;
    let mut cum_w = 0.0f32;
    let mut k_active = 6usize;
    let min_k = k_min.clamp(1, 6);

    for k in 0..6 {
        cum_w += weights[k];
        if (k + 1) >= min_k && cum_w >= target_w {
            k_active = k + 1;
            break;
        }
    }

    if k_active < 6 && cum_w > 1e-12 {
        let scale = total_w / cum_w;
        for k in 0..k_active {
            weights[k] *= scale;
        }
        for k in k_active..6 {
            weights[k] = 0.0;
            indices[k] = 0;
        }
    }
    k_active
}
