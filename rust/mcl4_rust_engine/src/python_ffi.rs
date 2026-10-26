//! C-ABI FFI Exports (`mcl4_rust_loader.dll`) for Zero-GIL Python Integration.
//!
//! Replaces Python's `json.loads` header parsing (~435 ms) and single-threaded GIL-bound
//! `seek()`/`readinto()` loops with native multi-threaded Rust + Intel P4608 Dual-Controller
//! (`X:` + `Z:`) `.mcl4e` 1-syscall streaming (~38 ms index scan, 8.3–13.4 GiB/s expert streaming).

use std::path::Path;
use std::slice;
use std::str;
use std::sync::Arc;

use crate::p4608_expert_sharder::P4608DualDriveExpertStore;
use crate::{Dsv41Mcl4cExpertPacket, SafetensorsShardIndex};

pub struct RustEngineHandle {
    pub index: Arc<SafetensorsShardIndex>,
    pub p4608_store: P4608DualDriveExpertStore,
}

#[no_mangle]
pub unsafe extern "C" fn mcl4_rust_engine_create(
    ckpt_dir_utf8: *const u8,
    ckpt_dir_len: usize,
) -> *mut RustEngineHandle {
    if ckpt_dir_utf8.is_null() || ckpt_dir_len == 0 {
        return std::ptr::null_mut();
    }
    let dir_bytes = slice::from_raw_parts(ckpt_dir_utf8, ckpt_dir_len);
    let dir_str = match str::from_utf8(dir_bytes) {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };
    let index = match SafetensorsShardIndex::scan_directory(Path::new(dir_str)) {
        Ok(idx) => Arc::new(idx),
        Err(_) => return std::ptr::null_mut(),
    };
    let handle = Box::new(RustEngineHandle {
        index,
        p4608_store: P4608DualDriveExpertStore::default_p4608(),
    });
    Box::into_raw(handle)
}

#[no_mangle]
pub unsafe extern "C" fn mcl4_rust_engine_destroy(handle: *mut RustEngineHandle) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

#[no_mangle]
pub unsafe extern "C" fn mcl4_rust_engine_num_shards(handle: *const RustEngineHandle) -> usize {
    if handle.is_null() {
        return 0;
    }
    let h = &*handle;
    h.index.shard_files.len()
}

#[no_mangle]
pub unsafe extern "C" fn mcl4_rust_engine_num_tensors(handle: *const RustEngineHandle) -> usize {
    if handle.is_null() {
        return 0;
    }
    let h = &*handle;
    h.index.tensors.len()
}

#[no_mangle]
pub unsafe extern "C" fn mcl4_rust_engine_num_layers(handle: *const RustEngineHandle) -> usize {
    if handle.is_null() {
        return 0;
    }
    let h = &*handle;
    h.index.available_layers.len()
}

/// Streams `count` `(layer, expert)` packets in parallel across BOTH Intel P4608 NVMe controllers
/// (`X:` for even layers, `Z:` for odd layers) using 1-syscall `.mcl4e` shards directly into `out_packets`.
#[no_mangle]
pub unsafe extern "C" fn mcl4_rust_engine_stream_experts(
    handle: *const RustEngineHandle,
    layers_ptr: *const u32,
    experts_ptr: *const u32,
    count: usize,
    out_packets_ptr: *mut Dsv41Mcl4cExpertPacket,
    out_elapsed_ms: *mut f64,
) -> i32 {
    if handle.is_null()
        || layers_ptr.is_null()
        || experts_ptr.is_null()
        || out_packets_ptr.is_null()
        || count == 0
    {
        return -1;
    }
    let h = &*handle;
    let layers = slice::from_raw_parts(layers_ptr, count);
    let experts = slice::from_raw_parts(experts_ptr, count);
    let out_packets = slice::from_raw_parts_mut(out_packets_ptr, count);

    let targets: Vec<(usize, usize)> = layers
        .iter()
        .zip(experts.iter())
        .map(|(&l, &e)| (l as usize, e as usize))
        .collect();

    if h.p4608_store.ensure_experts_sharded(&h.index, &targets).is_err() {
        return -2;
    }

    match h.p4608_store.stream_experts_striped(&targets, out_packets) {
        Ok((dt_ms, _, _)) => {
            if !out_elapsed_ms.is_null() {
                *out_elapsed_ms = dt_ms;
            }
            0
        }
        Err(_) => -3,
    }
}

/// O(1) NVMe gather of 24 hashed Engram rows (`24 * 256` FP8 + `24 * 8` E8M0) in native Rust.
#[no_mangle]
pub unsafe extern "C" fn mcl4_rust_engine_gather_engram_rows(
    handle: *const RustEngineHandle,
    layer: u32,
    bucket_ids_24_ptr: *const i64,
    out_fp8_ptr: *mut u8,
    out_scales_ptr: *mut u8,
) -> i32 {
    if handle.is_null()
        || bucket_ids_24_ptr.is_null()
        || out_fp8_ptr.is_null()
        || out_scales_ptr.is_null()
    {
        return -1;
    }
    let h = &*handle;
    let buckets = &*(bucket_ids_24_ptr as *const [i64; 24]);
    let out_fp8 = &mut *(out_fp8_ptr as *mut [u8; 24 * 256]);
    let out_scales = &mut *(out_scales_ptr as *mut [u8; 24 * 8]);

    match h.index.gather_engram_token_rows(layer as usize, buckets, out_fp8, out_scales) {
        Ok(()) => 0,
        Err(_) => -2,
    }
}
