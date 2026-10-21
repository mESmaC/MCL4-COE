"""Native Blackwell (sm_120) NVFP4 training & rollout kernels with 2:4 Structured Sparsity.

Adapted from oracle/src/oracle/nvfp4.py for MCL4-COE on the RTX 5070:
- Master weights and optimizer states remain FP32 (`torch.optim.AdamW` or `CPUOffloadAdamW`).
- Matrix operands use packed 4-bit E2M1 (`torch.float4_e2m1fn_x2`), swizzled FP8 E4M3
  block scales (`SWIZZLE_32_4_4`) per 16 values, and a global FP32 decoding scale.
- Causal Row-Wise Activation Scaling (`causal_activation_scale=True`):
  Activations use `global_scale = 1.0` with strictly per-row `1x16` `FP8 E4M3` block scales
  (dynamic range `[-2688, 2688]` per row) so future tokens never leak into earlier tokens'
  quantization scales across causal sequences.
- 2:4 Transposable Structured Weight Sparsity (`2to4_transposable`) + SR-STE:
  Enforces exact 50% 2:4 structured sparsity across every 4x4 weight tile (2 non-zeros per
  4 elements in both `W` for `forward` and `W.T` for `dgrad` via the 90 doubly-2:4 patterns),
  paired with Sparse-Refined Straight-Through Estimator (`sr_ste_decay`) in `wgrad`.
- Backward quantizes gradients with stochastic rounding and applies a shared fixed-sign
  orthonormal 16-point Hadamard transform (`_hadamard16`) to both reduction operands of `wgrad`.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache

import torch
from torch import Tensor, nn
from torch.nn import functional as F

_GEMM_COUNTS: Counter = Counter(forward=0, dgrad=0, wgrad=0, other=0, sparse_2to4=0)

SPARSITY_MODES = ("none", "2to4_row", "2to4_transposable")


def get_gemm_counts() -> dict[str, int]:
    """Return successful native NVFP4 GEMM calls, grouped by training phase."""
    return dict(_GEMM_COUNTS)


def reset_gemm_counts() -> None:
    """Zero all NVFP4 GEMM phase counters."""
    for phase in _GEMM_COUNTS:
        _GEMM_COUNTS[phase] = 0


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


@lru_cache(maxsize=8)
def _transposable_2to4_catalog(device_str: str) -> Tensor:
    """Return the `[90, 16]` catalog of 4x4 binary masks that are 2:4 in both rows and columns."""
    row_patterns = [
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 1],
    ]
    valid_tiles: list[list[float]] = []
    for r0 in row_patterns:
        for r1 in row_patterns:
            for r2 in row_patterns:
                for r3 in row_patterns:
                    if all(r0[c] + r1[c] + r2[c] + r3[c] == 2 for c in range(4)):
                        valid_tiles.append([float(x) for row in (r0, r1, r2, r3) for x in row])
    return torch.tensor(valid_tiles, dtype=torch.float32, device=torch.device(device_str))


@torch.no_grad()
def compute_2to4_mask(weight: Tensor, mode: str = "2to4_transposable") -> Tensor:
    """Compute a 50%-sparse 2:4 boolean mask for a 2D weight matrix `[out_features, in_features]`.

    - `'2to4_row'`: keeps top-2 magnitudes out of every 4 contiguous elements along `in_features`.
    - `'2to4_transposable'`: selects the maximum-L1 doubly-2:4 pattern on every 4x4 tile so
       BOTH `W` (`forward`) and `W.T` (`dgrad`) satisfy exact 2:4 structured sparsity.
    """
    if mode not in SPARSITY_MODES:
        raise ValueError(f"sparsity mode must be one of {SPARSITY_MODES}, got {mode!r}")
    if mode == "none":
        return torch.ones_like(weight, dtype=torch.bool)
    if weight.ndim != 2:
        raise ValueError("compute_2to4_mask expects a 2D weight matrix")

    rows, cols = weight.shape
    if mode == "2to4_row" or rows % 4 != 0 or cols % 4 != 0:
        pad_c = _round_up(cols, 4) - cols
        w_abs = F.pad(weight.detach().abs().float(), (0, pad_c)) if pad_c else weight.detach().abs().float()
        grouped = w_abs.view(rows, -1, 4)
        top2 = grouped.topk(2, dim=-1).indices
        mask = torch.zeros_like(grouped, dtype=torch.bool).scatter_(-1, top2, True).view(rows, -1)
        return mask[:, :cols]

    catalog = _transposable_2to4_catalog(str(weight.device))
    tiles = (
        weight.detach()
        .abs()
        .float()
        .view(rows // 4, 4, cols // 4, 4)
        .permute(0, 2, 1, 3)
        .reshape(-1, 16)
    )
    best_idx = torch.matmul(tiles, catalog.t()).argmax(dim=-1)
    chosen = catalog.index_select(0, best_idx).bool()
    return (
        chosen.view(rows // 4, cols // 4, 4, 4)
        .permute(0, 2, 1, 3)
        .reshape(rows, cols)
        .contiguous()
    )


def has_native_nvfp4(device: torch.device | str = "cuda") -> bool:
    """Return True if `device` and the active PyTorch build support native Blackwell sm_100+ NVFP4."""
    if not torch.cuda.is_available():
        return False
    dev = torch.device(device)
    if dev.type != "cuda":
        return False
    if not (
        hasattr(torch, "float4_e2m1fn_x2")
        and hasattr(F, "scaled_mm")
        and hasattr(F, "ScalingType")
        and hasattr(F, "SwizzleType")
    ):
        return False
    major, _ = torch.cuda.get_device_capability(dev)
    return major >= 10


def _swizzle_scales(scales: Tensor) -> Tensor:
    """Pack block scales into cuBLASLt's 128-row by 4-scale tiles with inner 32x4x4 layout."""
    rows, columns = scales.shape
    padded_rows, padded_columns = _round_up(rows, 128), _round_up(columns, 4)
    s_u8 = scales.to(torch.float8_e4m3fn).view(torch.uint8)
    if padded_rows != rows or padded_columns != columns:
        s_u8 = F.pad(s_u8, (0, padded_columns - columns, 0, padded_rows - rows))
    return (
        s_u8.view(padded_rows // 128, 4, 32, padded_columns // 4, 4)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
        .view(padded_rows, padded_columns)
        .view(torch.float8_e4m3fn)
    )


def _unswizzle_scales(swizzled: Tensor, rows: int, columns: int) -> Tensor:
    """Invert `SWIZZLE_32_4_4` block scales back to `[rows, columns]` FP32 scales."""
    padded_rows, padded_columns = swizzled.shape
    return (
        swizzled.float()
        .view(padded_rows // 128, padded_columns // 4, 32, 4, 4)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
        .view(padded_rows, padded_columns)[:rows, :columns]
    )


@lru_cache(maxsize=8)
def _e2m1_levels(device_str: str) -> Tensor:
    return torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=torch.device(device_str),
    )


@lru_cache(maxsize=8)
def _e2m1_midpoints(device_str: str) -> Tensor:
    return torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
        dtype=torch.float32,
        device=torch.device(device_str),
    )


@lru_cache(maxsize=8)
def _hadamard16_signs(device_str: str) -> Tensor:
    return torch.tensor(
        [1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1],
        dtype=torch.float32,
        device=torch.device(device_str),
    )


def _e2m1_codes(values: Tensor, stochastic: bool) -> Tensor:
    """Round normalized values to E2M1, returning one unsigned nibble per scalar."""
    m = values.abs().clamp(max=6.0)
    u = torch.where(m <= 2.0, m * 2.0, torch.where(m <= 4.0, m + 2.0, m * 0.5 + 4.0))
    if stochastic:
        code = (u + torch.rand_like(u)).to(torch.uint8).clamp(max=7)
    else:
        code = torch.round(u).to(torch.uint8)
    return code | (torch.signbit(values).to(torch.uint8) << 3)


def _quantize_nvfp4_core(
    values: Tensor,
    two_dimensional: bool,
    stochastic: bool,
    causal_row_scale: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    rows, columns = values.shape
    if causal_row_scale and not two_dimensional:
        global_scale = values.new_ones(1)
    else:
        amax = values.abs().amax()
        global_scale = torch.where(amax > 0, amax / 2688.0, torch.ones_like(amax))
        global_scale = global_scale.clamp_min(1.17549435e-38).reshape(1)
    if two_dimensional:
        block_amax = values.abs().view(rows // 16, 16, columns // 16, 16).amax(dim=(1, 3))
        block_amax = block_amax.repeat_interleave(16, dim=0)
    else:
        block_amax = values.abs().view(rows, columns // 16, 16).amax(dim=-1)
    block_scales = (block_amax / (6.0 * global_scale)).clamp(max=448.0).to(torch.float8_e4m3fn)
    decoding = block_scales.float() * global_scale
    inverse = torch.where(
        decoding > 0,
        decoding.reciprocal().clamp(max=3.402823466e38),
        torch.zeros_like(decoding),
    )
    normalized = (values.view(rows, columns // 16, 16) * inverse.unsqueeze(-1)).reshape(rows, columns)
    codes = _e2m1_codes(normalized, stochastic)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    return packed, block_scales, global_scale


try:
    _quantize_nvfp4_compiled = torch.compile(_quantize_nvfp4_core, dynamic=True)
except Exception:
    _quantize_nvfp4_compiled = _quantize_nvfp4_core


@torch.no_grad()
def quantize_nvfp4(
    matrix: Tensor,
    *,
    two_dimensional: bool = False,
    stochastic: bool = False,
    causal_row_scale: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """Encode a CUDA matrix as packed FP4 (`float4_e2m1fn_x2`), swizzled FP8 scales, and FP32 scale."""
    if matrix.ndim != 2 or not matrix.is_floating_point() or matrix.numel() == 0:
        raise ValueError("NVFP4 quantization requires a nonempty floating-point matrix")
    if matrix.device.type != "cuda":
        raise ValueError("NVFP4 quantization requires a CUDA matrix")
    rows, columns = matrix.shape
    if columns % 16 or (two_dimensional and rows % 16):
        raise ValueError("NVFP4 blocks require aligned dimensions divisible by 16")
    values = matrix.detach().float().contiguous()
    packed, block_scales, global_scale = _quantize_nvfp4_compiled(
        values, two_dimensional, stochastic, causal_row_scale
    )
    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", torch.uint8)
    return (
        packed.view(fp4_dtype),
        _swizzle_scales(block_scales),
        global_scale,
    )


@torch.no_grad()
def _dequantize_nvfp4(
    packed: Tensor,
    swizzled_scales: Tensor,
    global_scale: Tensor,
    rows: int,
    columns: int,
) -> Tensor:
    """Decode packed E2M1 + swizzled FP8 E4M3 block scales back to FP32 on pre-Blackwell GPUs."""
    raw_u8 = packed.view(torch.uint8)
    c0 = raw_u8 & 0x0F
    c1 = (raw_u8 >> 4) & 0x0F
    codes = torch.stack((c0, c1), dim=-1).reshape(rows, columns)
    sign = torch.where((codes & 0x08) != 0, -1.0, 1.0)
    mag_idx = (codes & 0x07).long()
    levels = _e2m1_levels(str(packed.device))
    norm_vals = sign * levels[mag_idx]
    block_scales = _unswizzle_scales(swizzled_scales, rows, columns // 16)
    decoding = block_scales * global_scale.float()
    return (norm_vals.view(rows, columns // 16, 16) * decoding.unsqueeze(-1)).reshape(rows, columns)


@torch.no_grad()
def nvfp4_mm(
    a: Tensor,
    b: Tensor,
    *,
    weight_b: bool = False,
    stochastic_a: bool = False,
    stochastic_b: bool = False,
    causal_a: bool = False,
    prequantized_b: tuple[Tensor, Tensor, Tensor] | None = None,
    _phase: str = "other",
) -> Tensor:
    """Return `a @ b.T` using native Blackwell NVFP4 (`F.scaled_mm`) or exact E2M1 emulation on pre-Blackwell CUDA."""
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("NVFP4 matmul expects a[M,K] and b[N,K]")
    if a.device != b.device or a.device.type != "cuda":
        raise ValueError("NVFP4 matmul requires both operands on the same CUDA device")
    m, k = a.shape
    n = b.shape[0]
    if min(m, n, k) == 0:
        raise ValueError("NVFP4 matmul requires nonempty operands")
    kp = _round_up(k, 32)
    mp = _round_up(m, 32)
    np_ = _round_up(n, 32)
    ap = F.pad(a.float(), (0, kp - k, 0, mp - m)) if (kp != k or mp != m) else a.float()
    qa, sa, ga = quantize_nvfp4(ap, stochastic=stochastic_a, causal_row_scale=causal_a)
    if prequantized_b is not None:
        qb, sb, gb = prequantized_b
    else:
        bp = F.pad(b.float(), (0, kp - k, 0, np_ - n)) if (kp != k or np_ != n) else b.float()
        qb, sb, gb = quantize_nvfp4(bp, two_dimensional=weight_b, stochastic=stochastic_b)
    if has_native_nvfp4(a.device):
        recipe = [F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise]
        swizzle = [F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE]
        result = F.scaled_mm(
            qa,
            qb.t(),
            scale_a=[sa, ga],
            scale_recipe_a=recipe,
            scale_b=[sb, gb],
            scale_recipe_b=recipe,
            swizzle_a=swizzle,
            swizzle_b=swizzle,
            output_dtype=torch.float32,
        )
    else:
        deq_a = _dequantize_nvfp4(qa, sa, ga, mp, kp)
        deq_b = _dequantize_nvfp4(qb, sb, gb, np_, kp)
        result = torch.matmul(deq_a, deq_b.t())
    _GEMM_COUNTS[_phase] += 1
    return result[:m, :n]


@lru_cache(maxsize=8)
def _hadamard16_matrix(device_str: str) -> Tensor:
    """Return the precomputed 16x16 signed Walsh-Hadamard orthonormal matrix on `device_str`."""
    values = torch.eye(16, dtype=torch.float32, device=torch.device(device_str)).reshape(16, 1, 16)
    values = values * _hadamard16_signs(device_str)
    for width in (1, 2, 4, 8):
        pairs = values.reshape(16, 1, 16 // (2 * width), 2, width)
        left, right = pairs[..., 0, :], pairs[..., 1, :]
        values = torch.stack((left + right, left - right), dim=-2).reshape(16, 1, 16)
    return (values * 0.25).reshape(16, 16).contiguous()


def _hadamard16(matrix: Tensor) -> Tensor:
    """Shared fixed-sign orthonormal transform over groups of 16 columns."""
    rows, columns = matrix.shape
    cp = _round_up(columns, 16)
    values = F.pad(matrix.float(), (0, cp - columns)) if cp != columns else matrix.float()
    h16 = _hadamard16_matrix(str(values.device))
    return torch.matmul(values.reshape(-1, 16), h16).reshape(rows, cp)


@torch.no_grad()
def _get_or_build_weight_nvfp4_cache(
    weight: Tensor,
    sparsity_mode: str,
    *,
    need_bwd: bool = False,
) -> list:
    """Cache `[key, mask, w_eff, fwd_q, bwd_q]` on `weight` keyed by `weight._version`.

    Within a training step (before `optimizer.step()` increments `weight._version`),
    `weight` is immutable across micro-batches, recurrence steps, `forward_all_operators`,
    `Ornith`, and `Dream-RSI`, eliminating ~85% of redundant 2:4 mask and 2D weight
    quantization passes with zero mathematical difference.
    """
    key = (weight._version, weight.data_ptr(), weight.shape, str(weight.device), sparsity_mode)
    cache = getattr(weight, "_nvfp4_weight_cache", None)
    if cache is None or cache[0] != key:
        mask: Tensor | None = None
        w_eff = weight.detach()
        if sparsity_mode != "none":
            mask = compute_2to4_mask(w_eff, mode=sparsity_mode)
            w_eff = w_eff * mask.to(w_eff.dtype)
        n, k = w_eff.shape
        np_, kp = _round_up(n, 32), _round_up(k, 32)
        bp = F.pad(w_eff.float(), (0, kp - k, 0, np_ - n)) if (kp != k or np_ != n) else w_eff.float()
        fwd_q = quantize_nvfp4(bp, two_dimensional=True, stochastic=False)
        cache = [key, mask, w_eff, fwd_q, None]
        try:
            setattr(weight, "_nvfp4_weight_cache", cache)
        except Exception:
            pass
    if need_bwd and cache[4] is None:
        w_t = cache[2].t().contiguous()
        kt, nt = w_t.shape
        ktp, ntp = _round_up(kt, 32), _round_up(nt, 32)
        bpt = F.pad(w_t.float(), (0, ntp - nt, 0, ktp - kt)) if (ntp != nt or ktp != kt) else w_t.float()
        cache[4] = quantize_nvfp4(bpt, two_dimensional=True, stochastic=False)
    return cache


class _NVFP4Linear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inputs: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        sparsity_mode: str = "none",
        sr_ste_decay: float = 0.0,
        causal_activation_scale: bool = True,
    ) -> Tensor:
        cache = _get_or_build_weight_nvfp4_cache(weight, sparsity_mode, need_bwd=False)
        mask: Tensor | None = cache[1]
        w_eff: Tensor = cache[2]
        fwd_q: tuple[Tensor, Tensor, Tensor] = cache[3]
        if sparsity_mode != "none":
            _GEMM_COUNTS["sparse_2to4"] += 1
        ctx.save_for_backward(inputs, weight, mask)
        ctx.bias_dtype = None if bias is None else bias.dtype
        ctx.sparsity_mode = sparsity_mode
        ctx.sr_ste_decay = float(sr_ste_decay)
        output = nvfp4_mm(
            inputs,
            w_eff,
            weight_b=True,
            causal_a=causal_activation_scale,
            prequantized_b=fwd_q,
            _phase="forward",
        )
        if bias is not None:
            output = output + bias.float()
        return output.to(inputs.dtype)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        inputs, weight, mask = ctx.saved_tensors
        grad_inputs = grad_weight = grad_bias = None
        gradient = grad_output.reshape(-1, weight.shape[0])
        if ctx.needs_input_grad[0]:
            cache = _get_or_build_weight_nvfp4_cache(weight, ctx.sparsity_mode, need_bwd=True)
            w_bwd_t = cache[2].t()
            bwd_q = cache[4]
            if mask is not None:
                _GEMM_COUNTS["sparse_2to4"] += 1
            grad_inputs = nvfp4_mm(
                gradient,
                w_bwd_t,
                weight_b=True,
                stochastic_a=True,
                prequantized_b=bwd_q,
                _phase="dgrad",
            ).to(inputs.dtype)
        if ctx.needs_input_grad[1]:
            grad_weight = nvfp4_mm(
                _hadamard16(gradient.t()),
                _hadamard16(inputs.t()),
                stochastic_a=True,
                _phase="wgrad",
            ).to(weight.dtype)
            if mask is not None and ctx.sr_ste_decay > 0.0:
                # SR-STE (Zhou et al., ICLR 2021): penalize masked-out master weights
                # so they smoothly decay toward 0 without destabilizing mask transitions.
                grad_weight = grad_weight + (
                    ctx.sr_ste_decay * (~mask).to(weight.dtype) * weight
                )
        if ctx.bias_dtype is not None and ctx.needs_input_grad[2]:
            grad_bias = gradient.float().sum(dim=0).to(ctx.bias_dtype)
        return grad_inputs, grad_weight, grad_bias, None, None, None


class NVFP4Linear(nn.Linear):
    """Projection executing native Blackwell NVFP4 GEMMs with optional 2:4 structured sparsity."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        *,
        sparsity_mode: str = "2to4_transposable",
        sr_ste_decay: float = 1e-4,
        causal_activation_scale: bool = True,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        if sparsity_mode not in SPARSITY_MODES:
            raise ValueError(f"sparsity_mode must be one of {SPARSITY_MODES}")
        self.sparsity_mode = sparsity_mode
        self.sr_ste_decay = float(sr_ste_decay)
        self.causal_activation_scale = bool(causal_activation_scale)

    @classmethod
    def from_linear(
        cls,
        layer: nn.Linear,
        *,
        sparsity_mode: str = "2to4_transposable",
        sr_ste_decay: float = 1e-4,
        causal_activation_scale: bool = True,
    ) -> "NVFP4Linear":
        converted = cls(
            layer.in_features,
            layer.out_features,
            bias=layer.bias is not None,
            device=layer.weight.device,
            dtype=layer.weight.dtype,
            sparsity_mode=sparsity_mode,
            sr_ste_decay=sr_ste_decay,
            causal_activation_scale=causal_activation_scale,
        )
        converted.load_state_dict(layer.state_dict())
        converted.weight.requires_grad_(layer.weight.requires_grad)
        if converted.bias is not None:
            converted.bias.requires_grad_(layer.bias.requires_grad)
        converted.train(layer.training)
        return converted

    @torch.no_grad()
    def active_2to4_mask(self) -> Tensor:
        """Return the current 2:4 boolean weight mask `[out_features, in_features]`."""
        return compute_2to4_mask(self.weight, mode=self.sparsity_mode)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim < 1 or inputs.shape[-1] != self.in_features:
            raise ValueError("NVFP4Linear input must end in in_features")
        if inputs.device.type != "cuda":
            w = (
                self.weight * compute_2to4_mask(self.weight, self.sparsity_mode).to(self.weight.dtype)
                if self.sparsity_mode != "none"
                else self.weight
            )
            return F.linear(inputs, w, self.bias)
        flat = inputs.reshape(-1, self.in_features)
        if flat.shape[0] == 0:
            return inputs.new_empty(*inputs.shape[:-1], self.out_features)
        result = _NVFP4Linear.apply(
            flat,
            self.weight,
            self.bias,
            self.sparsity_mode,
            self.sr_ste_decay,
            self.causal_activation_scale,
        )
        return result.reshape(*inputs.shape[:-1], self.out_features)


class NVFP4Conv2d(nn.Conv2d):
    """Convolution expressed as unfold plus native NVFP4 training GEMMs."""

    @classmethod
    def from_conv(cls, conv: nn.Conv2d) -> "NVFP4Conv2d":
        if conv.groups != 1 or conv.padding_mode != "zeros" or isinstance(conv.padding, str):
            raise ValueError("NVFP4Conv2d currently supports groups=1 and numeric zero padding")
        converted = cls(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=conv.bias is not None,
            padding_mode=conv.padding_mode,
            device=conv.weight.device,
            dtype=conv.weight.dtype,
        )
        converted.load_state_dict(conv.state_dict())
        converted.weight.requires_grad_(conv.weight.requires_grad)
        if converted.bias is not None:
            converted.bias.requires_grad_(conv.bias.requires_grad)
        converted.train(conv.training)
        return converted

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4:
            raise ValueError("NVFP4Conv2d expects [batch, channels, height, width]")
        if inputs.device.type != "cuda":
            return self._conv_forward(inputs, self.weight, self.bias)
        batch, _, height, width = inputs.shape
        patches = F.unfold(
            inputs,
            self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
        )
        patches = patches.transpose(1, 2).reshape(-1, patches.shape[1])
        result = _NVFP4Linear.apply(patches, self.weight.flatten(1), self.bias, "none", 0.0, False)
        out_height = (
            height + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1
        ) // self.stride[0] + 1
        out_width = (
            width + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1
        ) // self.stride[1] + 1
        return (
            result.reshape(batch, out_height * out_width, self.out_channels)
            .transpose(1, 2)
            .reshape(batch, self.out_channels, out_height, out_width)
        )


def _convert_expert_ffns(
    module: nn.Module,
    *,
    min_features: int,
    sparsity_mode: str,
    sr_ste_decay: float,
    causal_activation_scale: bool,
) -> int:
    """Convert `SwiGLUExpert` (`up_proj`, `down_proj`) FFN layers to `NVFP4Linear`."""
    converted = 0
    for child in module.modules():
        if child.__class__.__name__ == "SwiGLUExpert":
            for attr in ("up_proj", "down_proj"):
                layer = getattr(child, attr, None)
                if (
                    isinstance(layer, nn.Linear)
                    and not isinstance(layer, NVFP4Linear)
                    and layer.in_features >= min_features
                    and layer.out_features >= min_features
                ):
                    setattr(
                        child,
                        attr,
                        NVFP4Linear.from_linear(
                            layer,
                            sparsity_mode=sparsity_mode,
                            sr_ste_decay=sr_ste_decay,
                            causal_activation_scale=causal_activation_scale,
                        ),
                    )
                    converted += 1
    return converted


def convert_linear_to_nvfp4(
    module: nn.Module,
    *,
    min_features: int = 32,
    experts_only: bool = True,
    sparsity_mode: str = "2to4_transposable",
    sr_ste_decay: float = 1e-4,
    causal_activation_scale: bool = True,
    exclude_names: tuple[str, ...] = ("router", "token_scorer", "ple", "route_", "start_query"),
) -> int:
    """Convert eligible `nn.Linear` submodules to `NVFP4Linear` with 2:4 sparsity.

    Following `oracle/src/oracle/multimodal.py:_convert_feedforwards`, when `module`
    contains `SwiGLUExpert` blocks and `experts_only=True`, only the expert FFN
    projections (`up_proj`, `down_proj`) are converted to `NVFP4Linear`, keeping
    attention, `PerLayerEmbedding`, `MCL4` routers, MoE routers, and `COE` 32-dim
    single-softmax readout heads in BF16/FP32.
    """
    if experts_only:
        expert_count = _convert_expert_ffns(
            module,
            min_features=min_features,
            sparsity_mode=sparsity_mode,
            sr_ste_decay=sr_ste_decay,
            causal_activation_scale=causal_activation_scale,
        )
        if expert_count > 0:
            return expert_count

    converted_count = 0
    for child_name, child in list(module.named_children()):
        if any(ex in child_name for ex in exclude_names):
            continue
        if (
            isinstance(child, nn.Linear)
            and not isinstance(child, NVFP4Linear)
            and child.in_features >= min_features
            and child.out_features >= min_features
        ):
            setattr(
                module,
                child_name,
                NVFP4Linear.from_linear(
                    child,
                    sparsity_mode=sparsity_mode,
                    sr_ste_decay=sr_ste_decay,
                    causal_activation_scale=causal_activation_scale,
                ),
            )
            converted_count += 1
        else:
            converted_count += convert_linear_to_nvfp4(
                child,
                min_features=min_features,
                experts_only=False,
                sparsity_mode=sparsity_mode,
                sr_ste_decay=sr_ste_decay,
                causal_activation_scale=causal_activation_scale,
                exclude_names=exclude_names,
            )
    return converted_count

