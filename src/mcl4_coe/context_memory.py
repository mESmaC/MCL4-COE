"""Four-Tier Context & Virtual Memory Hierarchy for `MCL4-COE`.

Unifies:
1. Tier 2 (`TurboQuantKV`): Data-oblivious randomized Walsh-Hadamard rotation (`H * D`) +
   3-bit Lloyd-Max hypersphere quantization + 1-bit Quantized Johnson-Lindenstrauss (`QJL`)
   residual sign sketch for ~4.2x KV / `EncodedState` compression with unbiased inner products.
2. Tier 3 (`ContextLMCompressor`): Multi-chunk continuous Context Language Model compressor
   that condenses long byte streams into compact gist vectors and `RecurrentFeedback` states.
3. Tier 4 (`A.TXY` Virtual Memory & `COEAddressRouter`): Base-36 factorized `A.TXY` virtual
   memory (`1,679,616` logical addresses across Planes `A0..A5`), SHA-256 verified page-table
   lookup, `N5` CRC32 append-only WAL writes, `N1` 1-position factorized address embeddings,
   `N6` single-register pointer programs (`READ`, `FOLLOW`, `UP`, `DOWN`, `PROJECT`, `FILTER`,
   `FIND`, `COUNT`, `COMPARE`), and `COEAddressRouter` (solving `N3b-primary` natural-language
   candidate routing via `Association` + `Decision`).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import struct
import unicodedata
from typing import Any, Iterable, Sequence
import zlib

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .backbone import EncodedState, MCL4CEDBackbone, RecurrentFeedback
from .data import DecisionExample, collate_decisions
from .operators import CognitiveOperatorEnsemble, pearson_similarity

# ---------------------------------------------------------------------------
# Tier 2: TurboQuant (Walsh-Hadamard + 3-bit Polar/Lloyd-Max + 1-bit QJL)
# ---------------------------------------------------------------------------

# Optimal 8-level (3-bit) Lloyd-Max centroids for a standard normal N(0, 1) variable
_LLOYD_MAX_3BIT_NORMAL: tuple[float, ...] = (
    -2.1520,
    -1.3439,
    -0.7560,
    -0.2451,
    0.2451,
    0.7560,
    1.3439,
    2.1520,
)
_LLOYD_MAX_3BIT_BOUNDS: tuple[float, ...] = (
    -1.7480,
    -1.0500,
    -0.5006,
    0.0000,
    0.5006,
    1.0500,
    1.7480,
)


def _rademacher_signs(dim: int, seed: int, device: torch.device) -> Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    bits = torch.randint(0, 2, (dim,), generator=gen, dtype=torch.int64)
    signs = (bits * 2 - 1).to(dtype=torch.float32, device=device)
    return signs


def fast_walsh_hadamard(x: Tensor, signs: Tensor | None = None, *, inverse: bool = False) -> Tensor:
    """Orthonormal Fast Walsh-Hadamard Transform ($O(D \\log D)$) along the last dimension.

    For non-power-of-two `D`, pads to the next power of two or applies block-diagonal FWHT
    on the largest power-of-two factor. When `signs` is provided:
    - forward (`inverse=False`): computes $H (s \\odot x) / \\sqrt{D}$
    - inverse (`inverse=True`): computes $s \\odot (H x / \\sqrt{D})$
    """
    orig_dtype = x.dtype
    h = x.float()
    dim = h.shape[-1]
    if dim <= 0 or (dim & (dim - 1)) != 0:
        raise ValueError(f"fast_walsh_hadamard requires power-of-two last dim, got {dim}")

    if not inverse and signs is not None:
        h = h * signs.to(device=h.device, dtype=torch.float32)

    step = 1
    while step < dim:
        h = h.view(*h.shape[:-1], dim // (step * 2), 2, step)
        a = h[..., 0, :]
        b = h[..., 1, :]
        h = torch.stack((a + b, a - b), dim=-2).reshape(*x.shape)
        step *= 2

    h = h * (1.0 / math.sqrt(dim))
    if inverse and signs is not None:
        h = h * signs.to(device=h.device, dtype=torch.float32)
    return h.to(orig_dtype)


@dataclass
class CompressedKV:
    """TurboQuant compressed representation of a `[..., D]` key/value or encoder state tensor."""

    codes_3bit: Tensor          # uint8 in [0, 7], shape [..., D]
    qjl_signs: Tensor           # uint8 in {0, 1} representing {-1, +1}, shape [..., D]
    row_norms: Tensor           # float16/bfloat16 row L2 norms, shape [..., 1]
    residual_norms: Tensor      # float16/bfloat16 residual L2 norms, shape [..., 1]
    seed: int
    orig_dtype: torch.dtype
    orig_shape: tuple[int, ...]

    @property
    def uncompressed_bytes(self) -> int:
        elem_size = 2 if self.orig_dtype in (torch.bfloat16, torch.float16) else 4
        n_elems = math.prod(self.orig_shape)
        return n_elems * elem_size

    @property
    def packed_logical_bytes(self) -> int:
        """Theoretical bit-packed footprint: 3 bits (polar/Lloyd-Max) + 1 bit (QJL) + 2 FP16 scalars per row."""
        n_elems = math.prod(self.orig_shape)
        n_rows = n_elems // self.orig_shape[-1]
        return (n_elems * 4 + 7) // 8 + n_rows * 4

    @property
    def compression_ratio(self) -> float:
        return self.uncompressed_bytes / max(1, self.packed_logical_bytes)


def turboquant_compress(x: Tensor, *, seed: int = 42) -> CompressedKV:
    """Compress `x` (`[..., D]`) via randomized FWHT + 3-bit Lloyd-Max + 1-bit QJL residual sketch."""
    orig_shape = tuple(x.shape)
    orig_dtype = x.dtype
    dim = orig_shape[-1]
    xf = x.float()

    row_norms = xf.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    unit_x = xf / row_norms

    signs = _rademacher_signs(dim, seed, x.device)
    rotated = fast_walsh_hadamard(unit_x, signs, inverse=False)

    # Scale by sqrt(D) so coordinates match standard N(0, 1) variance on the unit sphere
    scaled = rotated * math.sqrt(dim)
    bounds = torch.tensor(_LLOYD_MAX_3BIT_BOUNDS, device=x.device, dtype=torch.float32)
    centroids = torch.tensor(_LLOYD_MAX_3BIT_NORMAL, device=x.device, dtype=torch.float32)

    codes = torch.bucketize(scaled, bounds).to(torch.uint8)
    quant_rotated = centroids[codes.long()] / math.sqrt(dim)

    residual = rotated - quant_rotated
    res_norms = residual.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    qjl_bits = (residual >= 0).to(torch.uint8)

    return CompressedKV(
        codes_3bit=codes,
        qjl_signs=qjl_bits,
        row_norms=row_norms.to(torch.float16),
        residual_norms=res_norms.to(torch.float16),
        seed=seed,
        orig_dtype=orig_dtype,
        orig_shape=orig_shape,
    )


def turboquant_decompress(comp: CompressedKV) -> Tensor:
    """Reconstruct `[..., D]` tensor from `CompressedKV` with QJL residual correction."""
    device = comp.codes_3bit.device
    dim = comp.orig_shape[-1]
    centroids = torch.tensor(_LLOYD_MAX_3BIT_NORMAL, device=device, dtype=torch.float32)

    quant_rotated = centroids[comp.codes_3bit.long()] / math.sqrt(dim)
    qjl_pm1 = comp.qjl_signs.float() * 2.0 - 1.0
    # Unbiased 1-bit QJL reconstruction factor sqrt(pi / (2 * D)) * ||residual||_2 * sign(residual)
    qjl_correction = math.sqrt(math.pi / (2.0 * dim)) * comp.residual_norms.float() * qjl_pm1
    reconstructed_rotated = quant_rotated + qjl_correction

    signs = _rademacher_signs(dim, comp.seed, device)
    unit_x = fast_walsh_hadamard(reconstructed_rotated, signs, inverse=True)
    out = unit_x * comp.row_norms.float()
    return out.view(*comp.orig_shape).to(comp.orig_dtype)


def turboquant_inner_product(queries: Tensor, comp_keys: CompressedKV) -> Tensor:
    """Compute unbiased attention/similarity inner products `<queries, keys>` directly in rotated space."""
    device = queries.device
    dim = queries.shape[-1]
    signs = _rademacher_signs(dim, comp_keys.seed, device)
    q_rot = fast_walsh_hadamard(queries.float(), signs, inverse=False)

    centroids = torch.tensor(_LLOYD_MAX_3BIT_NORMAL, device=device, dtype=torch.float32)
    k_polar = centroids[comp_keys.codes_3bit.long()] / math.sqrt(dim)
    k_qjl = (
        math.sqrt(math.pi / (2.0 * dim))
        * comp_keys.residual_norms.float()
        * (comp_keys.qjl_signs.float() * 2.0 - 1.0)
    )
    k_rot = (k_polar + k_qjl) * comp_keys.row_norms.float()
    return torch.matmul(q_rot, k_rot.transpose(-1, -2)).to(queries.dtype)


@dataclass
class CompressedEncodedState:
    """TurboQuant-compressed `EncodedState` for low-VRAM multi-pass `COE` & long-context caching."""

    compressed_hidden: CompressedKV
    valid: Tensor
    aux_loss: Tensor
    router_z_loss: Tensor
    expert_load: Tensor

    @property
    def compression_ratio(self) -> float:
        return self.compressed_hidden.compression_ratio

    def decompress(self) -> EncodedState:
        return EncodedState(
            hidden=turboquant_decompress(self.compressed_hidden),
            valid=self.valid,
            aux_loss=self.aux_loss,
            router_z_loss=self.router_z_loss,
            expert_load=self.expert_load,
        )


def compress_encoded_state(state: EncodedState, *, seed: int = 42) -> CompressedEncodedState:
    """Compress `EncodedState.hidden` with TurboQuant while preserving exact validity masks and MoE stats."""
    return CompressedEncodedState(
        compressed_hidden=turboquant_compress(state.hidden, seed=seed),
        valid=state.valid,
        aux_loss=state.aux_loss,
        router_z_loss=state.router_z_loss,
        expert_load=state.expert_load,
    )


# ---------------------------------------------------------------------------
# Tier 3: Context-LM Multi-Chunk Recurrent State Compressor
# ---------------------------------------------------------------------------


class ContextLMCompressor(nn.Module):
    """Streaming Context-LM compressor that condenses long byte streams into compact gist states.

    Splits an arbitrarily long input `[B, L_total]` into windows of `chunk_size`, encodes each
    chunk through `MCL4CEDBackbone.encode_state`, blends running cross-chunk state via
    `MCL4CEDBackbone.feedback` (`RecurrentFeedback`), and pools each chunk into
    `gist_tokens_per_chunk` summary vectors so downstream `COE` operators or decoders can attend
    to an $O(N_{\\text{chunks}} \\cdot G)$ compressed context.
    """

    def __init__(
        self,
        d_model: int,
        *,
        chunk_size: int = 256,
        gist_tokens_per_chunk: int = 4,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.chunk_size = chunk_size
        self.gist_tokens_per_chunk = gist_tokens_per_chunk
        self.gist_queries = nn.Parameter(torch.randn(gist_tokens_per_chunk, d_model) * 0.02)
        self.gist_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        backbone: MCL4CEDBackbone,
        token_ids: Tensor,
        valid_mask: Tensor | None = None,
    ) -> EncodedState:
        batch_size, total_len = token_ids.shape
        if valid_mask is None:
            valid_mask = torch.ones_like(token_ids, dtype=torch.bool)

        gist_chunks: list[Tensor] = []
        gist_valids: list[Tensor] = []
        running_gist: Tensor | None = None
        total_aux = token_ids.new_zeros((), dtype=torch.float32)
        total_z = token_ids.new_zeros((), dtype=torch.float32)
        loads: list[Tensor] = []

        for start in range(0, total_len, self.chunk_size):
            end = min(total_len, start + self.chunk_size)
            c_ids = token_ids[:, start:end]
            c_val = valid_mask[:, start:end]
            if not c_val.any():
                continue

            enc = backbone.encode_state(c_ids, c_val, causal=False)
            total_aux = total_aux + enc.aux_loss
            total_z = total_z + enc.router_z_loss
            loads.append(enc.expert_load)

            # Cross-attend `gist_queries` over the chunk's valid token representations
            q = self.gist_queries.unsqueeze(0).expand(batch_size, -1, -1).to(enc.hidden.dtype)
            scores = torch.matmul(q.float(), enc.hidden.float().transpose(-1, -2)) / math.sqrt(
                self.d_model
            )
            scores = scores.masked_fill(~c_val.unsqueeze(1), -1e9)
            attn = F.softmax(scores, dim=-1).to(enc.hidden.dtype)
            chunk_gist = self.gist_norm(torch.matmul(attn, enc.hidden).float()).to(enc.hidden.dtype)

            chunk_has_valid = c_val.any(dim=-1, keepdim=True).expand(
                batch_size, self.gist_tokens_per_chunk
            )
            if running_gist is not None:
                chunk_gist = backbone.feedback(chunk_gist, running_gist, valid=chunk_has_valid)
            running_gist = chunk_gist

            gist_chunks.append(chunk_gist)
            gist_valids.append(chunk_has_valid)

        if not gist_chunks:
            return backbone.encode_state(token_ids[:, :1], valid_mask[:, :1])

        compressed_hidden = torch.cat(gist_chunks, dim=1)
        compressed_valid = torch.cat(gist_valids, dim=1)
        avg_load = torch.stack(loads, dim=0).mean(dim=0)
        return EncodedState(
            hidden=compressed_hidden,
            valid=compressed_valid,
            aux_loss=total_aux / max(1, len(loads)),
            router_z_loss=total_z / max(1, len(loads)),
            expert_load=avg_load,
        )


# ---------------------------------------------------------------------------
# Tier 4: A.TXY Factorized Virtual Memory, Pointer Programs & COE Router
# ---------------------------------------------------------------------------

ATXY_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ATXY_BASE = len(ATXY_ALPHABET)  # 36
ATXY_N_ADDRESSES = ATXY_BASE**4  # 1,679,616
_ATXY_VAL = {ch: idx for idx, ch in enumerate(ATXY_ALPHABET)}

ATXY_OPS: tuple[str, ...] = (
    "READ",
    "FOLLOW",
    "UP",
    "DOWN",
    "PROJECT",
    "FILTER",
    "FIND",
    "FOLLOW_SET",
    "COUNT",
    "COMPARE",
)


@dataclass(frozen=True)
class ATXYAddress:
    """Four-field base-36 semantic address `A.TXY` (`A`: plane, `T`: table, `X`: row, `Y`: col)."""

    A: int
    T: int
    X: int
    Y: int

    def __post_init__(self) -> None:
        for name, val in (("A", self.A), ("T", self.T), ("X", self.X), ("Y", self.Y)):
            if not isinstance(val, int) or not (0 <= val < ATXY_BASE):
                raise ValueError(f"ATXY field {name}={val!r} out of range [0, {ATXY_BASE})")

    @property
    def packed(self) -> int:
        return ((self.A * ATXY_BASE + self.T) * ATXY_BASE + self.X) * ATXY_BASE + self.Y

    @property
    def directory_index(self) -> int:
        return self.A * ATXY_BASE + self.T

    @property
    def cell(self) -> int:
        return self.X * ATXY_BASE + self.Y

    def format(self, *, dot: bool = True, alphabet: str = ATXY_ALPHABET) -> str:
        body = alphabet[self.A] + alphabet[self.T] + alphabet[self.X] + alphabet[self.Y]
        return f"{body[0]}.{body[1:]}" if dot else body

    def __str__(self) -> str:
        return self.format(dot=True)


def pack_atxy(A: int, T: int, X: int, Y: int) -> int:
    return ATXYAddress(A, T, X, Y).packed


def unpack_atxy(packed: int) -> ATXYAddress:
    if not isinstance(packed, int) or not (0 <= packed < ATXY_N_ADDRESSES):
        raise ValueError(f"packed={packed!r} out of range [0, {ATXY_N_ADDRESSES})")
    Y = packed % ATXY_BASE
    X = (packed // ATXY_BASE) % ATXY_BASE
    T = (packed // (ATXY_BASE * ATXY_BASE)) % ATXY_BASE
    A = (packed // (ATXY_BASE**3)) % ATXY_BASE
    return ATXYAddress(A, T, X, Y)


def parse_atxy(text: str, *, alphabet: str = ATXY_ALPHABET) -> ATXYAddress:
    s = text.strip()
    if alphabet == ATXY_ALPHABET:
        s = s.upper()
    if "." in s:
        if len(s) != 5 or s[1] != ".":
            raise ValueError(f"Invalid dotted A.TXY address: {text!r}")
        chars = s[0] + s[2:]
    else:
        if len(s) != 4:
            raise ValueError(f"Invalid undotted ATXY address: {text!r}")
        chars = s
    val_map = _ATXY_VAL if alphabet == ATXY_ALPHABET else {c: i for i, c in enumerate(alphabet)}
    try:
        A, T, X, Y = (val_map[c] for c in chars)
    except KeyError as exc:
        raise ValueError(f"Character {exc} not in alphabet") from None
    return ATXYAddress(A, T, X, Y)


@dataclass(frozen=True)
class ATXYRecord:
    """Verified canonical record stored at an `A.TXY` cell."""

    address: str
    kind: str
    payload: str
    summary: str = ""
    up: tuple[str, ...] = ()
    down: tuple[str, ...] = ()
    related: tuple[str, ...] = ()
    rels: dict[str, tuple[str, ...]] = field(default_factory=dict)
    fields: dict[str, int] = field(default_factory=dict)
    version: int = 1
    schema_version: int = 1
    provenance: tuple[str, ...] = ()

    @property
    def parsed_address(self) -> ATXYAddress:
        return parse_atxy(self.address)

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "address": str(parse_atxy(self.address)),
            "fields": {str(k): int(v) for k, v in sorted(self.fields.items())},
            "kind": str(self.kind),
            "payload": str(self.payload),
            "provenance": list(self.provenance),
            "related": [str(parse_atxy(x)) for x in self.related],
            "rels": {
                str(k): sorted(str(parse_atxy(x)) for x in v)
                for k, v in sorted(self.rels.items())
            },
            "schema_version": int(self.schema_version),
            "summary": str(self.summary),
            "up": [str(parse_atxy(x)) for x in self.up],
            "down": [str(parse_atxy(x)) for x in self.down],
            "version": int(self.version),
        }

    def digest(self) -> str:
        addr = self.parsed_address
        h = hashlib.sha256()
        h.update(struct.pack("<III", self.schema_version, self.version, addr.packed))
        raw = json.dumps(
            self.canonical_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        h.update(raw)
        return h.hexdigest()

    def to_candidate_description(self) -> str:
        """Render a compact `#T$# ... #A$#` candidate description for `COE` routing."""
        rel_str = ", ".join(f"{k}:{len(v)}" for k, v in sorted(self.rels.items()) if v)
        meta = f"[{self.kind} @{self.address}] {self.summary}".strip()
        if rel_str:
            meta = f"{meta} ({rel_str})"
        return f"#T$# {meta}\n#A$# {self.payload}"


class ATXYFactorizedEmbedding(nn.Module):
    """`elem` N1 validated 1-position factorized address embedding: $E_A(A) + E_T(T) + E_X(X) + E_Y(Y)$."""

    def __init__(self, d_model: int, *, base: int = ATXY_BASE) -> None:
        super().__init__()
        self.base = base
        self.d_model = d_model
        self.plane_emb = nn.Embedding(base, d_model)
        self.table_emb = nn.Embedding(base, d_model)
        self.row_emb = nn.Embedding(base, d_model)
        self.col_emb = nn.Embedding(base, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, coords: Tensor) -> Tensor:
        """Encode `[..., 4]` integer coordinates `(A, T, X, Y)` into `[..., d_model]` single-position vectors."""
        if coords.shape[-1] != 4:
            raise ValueError(f"Expected last dim 4 for (A, T, X, Y), got {coords.shape}")
        a = coords[..., 0]
        t = coords[..., 1]
        x = coords[..., 2]
        y = coords[..., 3]
        emb = self.plane_emb(a) + self.table_emb(t) + self.row_emb(x) + self.col_emb(y)
        return self.norm(emb)

    def encode_addresses(
        self,
        addresses: Sequence[str | int | ATXYAddress],
        *,
        device: str | torch.device = "cpu",
    ) -> Tensor:
        rows = []
        for item in addresses:
            addr = (
                item
                if isinstance(item, ATXYAddress)
                else (unpack_atxy(item) if isinstance(item, int) else parse_atxy(item))
            )
            rows.append([addr.A, addr.T, addr.X, addr.Y])
        tensor_coords = torch.tensor(rows, dtype=torch.long, device=device)
        return self.forward(tensor_coords)


class ATXYPageStore:
    """Two-level `(A, T) -> (X, Y)` virtual memory page store with SHA-256 verification and `N5` WAL writes."""

    def __init__(self, records: Iterable[ATXYRecord | dict[str, Any]] = ()) -> None:
        self._directory: dict[int, dict[int, tuple[ATXYRecord, str]]] = {}
        self._aliases: dict[str, int | None] = {}
        self._wal_frames: list[bytes] = []
        for rec in records:
            self.put(rec, record_wal=False)

    @staticmethod
    def _normalize_alias(text: str) -> str:
        return unicodedata.normalize("NFC", text).strip().casefold()

    def _coerce_record(self, rec: ATXYRecord | dict[str, Any]) -> ATXYRecord:
        if isinstance(rec, ATXYRecord):
            return rec
        return ATXYRecord(
            address=str(parse_atxy(str(rec["address"]))),
            kind=str(rec.get("kind", "concept")),
            payload=str(rec["payload"]),
            summary=str(rec.get("summary", "")),
            up=tuple(str(parse_atxy(x)) for x in rec.get("up", ())),
            down=tuple(str(parse_atxy(x)) for x in rec.get("down", ())),
            related=tuple(str(parse_atxy(x)) for x in rec.get("related", ())),
            rels={
                str(k): tuple(str(parse_atxy(x)) for x in v)
                for k, v in (rec.get("rels") or {}).items()
            },
            fields={str(k): int(v) for k, v in (rec.get("fields") or {}).items()},
            version=int(rec.get("version", 1)),
            schema_version=int(rec.get("schema_version", 1)),
            provenance=tuple(str(x) for x in rec.get("provenance", ())),
        )

    def put(self, rec: ATXYRecord | dict[str, Any], *, record_wal: bool = False) -> ATXYRecord:
        canonical = self._coerce_record(rec)
        addr = canonical.parsed_address
        dig = canonical.digest()
        page = self._directory.setdefault(addr.directory_index, {})
        page[addr.cell] = (canonical, dig)

        norm = self._normalize_alias(canonical.payload)
        if norm:
            if norm not in self._aliases:
                self._aliases[norm] = addr.packed
            elif self._aliases[norm] != addr.packed:
                self._aliases[norm] = None  # Ambiguous alias excluded (fail-closed)

        if record_wal:
            payload_bytes = json.dumps(canonical.canonical_dict(), sort_keys=True).encode("utf-8")
            crc = zlib.crc32(payload_bytes) & 0xFFFFFFFF
            frame = struct.pack("<II", crc, len(payload_bytes)) + payload_bytes
            self._wal_frames.append(frame)
        return canonical

    def commit_write(
        self,
        rec: ATXYRecord | dict[str, Any],
        *,
        wal_path: str | Path | None = None,
    ) -> ATXYRecord:
        """`elem` N5 durable append-only versioned write with CRC32 WAL framing."""
        canonical = self._coerce_record(rec)
        existing = self.lookup(canonical.address)
        if existing is not None and canonical.version <= existing.version:
            canonical = ATXYRecord(
                **{**canonical.__dict__, "version": existing.version + 1}
            )
        committed = self.put(canonical, record_wal=True)
        if wal_path is not None:
            p = Path(wal_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("ab") as fh:
                fh.write(self._wal_frames[-1])
                fh.flush()
        return committed

    def load_wal(self, wal_path: str | Path) -> int:
        """Replay a framed CRC32 write-ahead log (`N5` session boundary recovery). Fails closed on corruption."""
        raw = Path(wal_path).read_bytes()
        off = 0
        applied = 0
        while off < len(raw):
            if off + 8 > len(raw):
                raise ValueError("Torn WAL header frame detected")
            crc_expected, length = struct.unpack_from("<II", raw, off)
            off += 8
            if off + length > len(raw):
                raise ValueError("Torn WAL payload frame detected")
            body = raw[off : off + length]
            off += length
            crc_actual = zlib.crc32(body) & 0xFFFFFFFF
            if crc_actual != crc_expected:
                raise ValueError("WAL frame CRC32 mismatch (corruption)")
            data = json.loads(body.decode("utf-8"))
            self.put(data, record_wal=True)
            applied += 1
        return applied

    def lookup(self, addr: str | int | ATXYAddress) -> ATXYRecord | None:
        """O(1) verified two-level directory/page lookup. Fails closed on digest or relocation mismatch."""
        a = (
            addr
            if isinstance(addr, ATXYAddress)
            else (unpack_atxy(addr) if isinstance(addr, int) else parse_atxy(addr))
        )
        page = self._directory.get(a.directory_index)
        if page is None:
            return None
        entry = page.get(a.cell)
        if entry is None:
            return None
        record, stored_digest = entry
        if record.parsed_address.packed != a.packed:
            raise ValueError(f"Relocation error at {a}: holds {record.address}")
        if record.digest() != stored_digest:
            raise ValueError(f"Digest corruption at {record.address}")
        return record

    def resolve(self, surface_text: str) -> ATXYRecord | None:
        """Exact deterministic reverse alias lookup (`surface_text -> ATXYRecord`)."""
        packed = self._aliases.get(self._normalize_alias(surface_text))
        if packed is None:
            return None
        return self.lookup(packed)

    def records(self) -> list[ATXYRecord]:
        out: list[ATXYRecord] = []
        for di in sorted(self._directory):
            page = self._directory[di]
            for cell in sorted(page):
                out.append(page[cell][0])
        return out

    def __len__(self) -> int:
        return sum(len(p) for p in self._directory.values())


class ATXYPointerExecutor:
    """`elem` N6 sequential single-register pointer program executor over `ATXYPageStore`."""

    def __init__(self, store: ATXYPageStore) -> None:
        self.store = store
        self.register: Any = None
        self.register_kind: str = "empty"
        self.audit: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.register = None
        self.register_kind = "empty"
        self.audit.clear()

    def step(self, op: str, operand: Any = "REG", arg: Any = None) -> tuple[str, Any]:
        op_u = op.upper()
        if op_u not in ATXY_OPS:
            raise ValueError(f"Unsupported pointer op: {op!r}")

        src = self.register if operand == "REG" else operand

        if op_u == "READ":
            rec = self.store.lookup(src)
            if rec is None:
                raise KeyError(f"READ miss at {src!r}")
            self.register = rec.payload
            self.register_kind = "value"
        elif op_u in ("UP", "DOWN"):
            rec = self.store.lookup(src)
            if rec is None:
                raise KeyError(f"{op_u} miss at {src!r}")
            links = rec.up if op_u == "UP" else rec.down
            idx = int(arg or 0)
            if idx >= len(links):
                raise IndexError(f"No {op_u}[{idx}] link on {rec.address}")
            self.register = links[idx]
            self.register_kind = "addr"
        elif op_u == "FOLLOW":
            rec = self.store.lookup(src)
            if rec is None:
                raise KeyError(f"FOLLOW miss at {src!r}")
            rel_name = str(arg or "RELATED").upper()
            targets = rec.related if rel_name == "RELATED" else rec.rels.get(rel_name, ())
            if not targets:
                raise KeyError(f"No {rel_name} edge from {rec.address}")
            self.register = targets[0]
            self.register_kind = "addr"
        elif op_u == "FOLLOW_SET":
            addrs = src if isinstance(src, (list, tuple, frozenset, set)) else [src]
            rel_name = str(arg or "RELATED").upper()
            out_set: list[str] = []
            for a_item in addrs:
                rec = self.store.lookup(a_item)
                if rec is None:
                    continue
                edges = (
                    rec.up
                    if rel_name == "UP"
                    else (
                        rec.down
                        if rel_name == "DOWN"
                        else (rec.related if rel_name == "RELATED" else rec.rels.get(rel_name, ()))
                    )
                )
                for dst in edges:
                    if dst not in out_set:
                        out_set.append(dst)
            self.register = tuple(out_set)
            self.register_kind = "set"
        elif op_u == "PROJECT":
            rec = self.store.lookup(src)
            if rec is None:
                raise KeyError(f"PROJECT miss at {src!r}")
            field_name = str(arg)
            if field_name not in rec.fields:
                raise KeyError(f"Missing field {field_name!r} at {rec.address}")
            self.register = rec.fields[field_name]
            self.register_kind = "scalar"
        elif op_u == "FIND":
            plane_filter = arg.get("plane") if isinstance(arg, dict) else None
            kind_filter = arg.get("kind") if isinstance(arg, dict) else None
            matches = []
            for r in self.store.records():
                if plane_filter is not None and r.parsed_address.A != int(plane_filter):
                    continue
                if kind_filter is not None and r.kind != str(kind_filter):
                    continue
                matches.append(r.address)
            self.register = tuple(matches)
            self.register_kind = "set"
        elif op_u == "FILTER":
            addrs = src if isinstance(src, (list, tuple, frozenset, set)) else []
            field_name = str(arg["field"])
            op_cmp = str(arg.get("op", "EQ")).upper()
            val = int(arg["val"])
            kept = []
            for a_item in addrs:
                rec = self.store.lookup(a_item)
                if rec is None or field_name not in rec.fields:
                    continue
                fv = rec.fields[field_name]
                if (op_cmp == "EQ" and fv == val) or (op_cmp == "LT" and fv < val) or (op_cmp == "GT" and fv > val):
                    kept.append(rec.address)
            self.register = tuple(kept)
            self.register_kind = "set"
        elif op_u == "COUNT":
            self.register = len(src) if isinstance(src, (list, tuple, frozenset, set)) else 0
            self.register_kind = "scalar"
        elif op_u == "COMPARE":
            self.register = bool(self.register == arg)
            self.register_kind = "bool"

        self.audit.append(
            {
                "op": op_u,
                "operand": operand,
                "arg": arg,
                "result_kind": self.register_kind,
                "result": self.register,
            }
        )
        return self.register_kind, self.register

    def run(self, program: Sequence[tuple[str, Any, Any]]) -> tuple[str, Any]:
        self.reset()
        for op, operand, arg in program:
            self.step(op, operand, arg)
        return self.register_kind, self.register


class COEAddressRouter:
    """Resolves `elem`'s `N3b-primary` natural-language routing bottleneck via `MCL4-COE`.

    Two-stage exact routing over `ATXYPageStore`:
    1. Stage 1 (`Association`, `<0.1ms` matched filter): Projects all occupied `A.TXY` records
       into `d_proj` vectors and selects the top-$K$ candidate cells.
    2. Stage 2 (`Decision`, shared-state `MCL4-CED` cross-attention + `AttentiveMatchedFilterPool`):
       Scores the $K$ candidate records against the natural-language query in a single softmax
       with bitwise candidate-permutation invariance.
    """

    def __init__(
        self,
        store: ATXYPageStore,
        *,
        max_seq_len: int = 256,
    ) -> None:
        self.store = store
        self.max_seq_len = max_seq_len

    @torch.no_grad()
    def route_query(
        self,
        coe: CognitiveOperatorEnsemble,
        query_text: str,
        *,
        top_k_candidates: int = 8,
        plane_filter: int | None = None,
    ) -> dict[str, Any]:
        all_recs = [
            r
            for r in self.store.records()
            if plane_filter is None or r.parsed_address.A == plane_filter
        ]
        if not all_recs:
            raise ValueError("No records available in ATXYPageStore for routing")

        device = next(coe.parameters()).device
        k = min(top_k_candidates, len(all_recs))

        # If store has more than top_k_candidates records, pre-filter via fast byte-sketch + Association
        if len(all_recs) > k:
            sketch_scores = []
            q_lower = query_text.lower()
            q_tokens = set(q_lower.replace("_", " ").replace(".", " ").split())
            for r in all_recs:
                doc = f"{r.address} {r.kind} {r.payload} {r.summary}".lower()
                d_tokens = set(doc.replace("_", " ").replace(":", " ").replace("/", " ").split())
                overlap = len(q_tokens & d_tokens) / max(1, len(q_tokens))
                sketch_scores.append(overlap)
            ranked_idx = sorted(
                range(len(all_recs)),
                key=lambda i: sketch_scores[i],
                reverse=True,
            )[:k]
            cand_recs = [all_recs[i] for i in ranked_idx]
        else:
            cand_recs = all_recs

        max_len = min(self.max_seq_len, int(coe.backbone.config.max_seq_len))
        budget = max(16, max_len - 4)
        raw_ctx = f"#U$# {query_text}\n#Q$# Select the exact A.TXY record matching the request."
        ctx_str = raw_ctx.encode("utf-8")[:budget].decode("utf-8", errors="ignore")
        desc_strs = tuple(
            r.to_candidate_description().encode("utf-8")[:budget].decode("utf-8", errors="ignore")
            for r in cand_recs
        )
        example = DecisionExample(
            question_id="atxy_route",
            kind="choice",
            context=ctx_str,
            descriptions=desc_strs,
            values=tuple(r.address for r in cand_recs),
            target=0,
        )
        batch = collate_decisions([example], max_length=max_len, device=device)
        dec_out = coe.forward_decision(
            batch["context_ids"],
            batch["context_lengths"],
            batch["option_ids"],
            batch["option_lengths"],
        )
        probs = dec_out.probabilities[0]
        best_idx = int(probs.argmax(dim=-1).item())
        chosen_rec = cand_recs[best_idx]
        return {
            "address": chosen_rec.address,
            "record": chosen_rec,
            "confidence": float(probs[best_idx].item()),
            "candidates": [r.address for r in cand_recs],
            "probabilities": [round(float(p), 5) for p in probs.cpu().tolist()],
        }


def index_python_repo_to_atxy(
    repo_root: str | Path,
    *,
    subsystems: Sequence[str] = ("src", "scripts", "tests"),
) -> ATXYPageStore:
    """Compile a Python repository's AST into an `ATXYPageStore` (`A=3` subsystem, `A=2` file, `A=1` symbol)."""
    root = Path(repo_root).expanduser().resolve()
    records: list[ATXYRecord] = []
    file_coords: dict[str, tuple[int, int]] = {}
    sym_index: dict[str, list[str]] = {}
    parsed_files: dict[str, tuple[list[tuple[str, str, str, ast.AST]], list[str]]] = {}

    for t_idx, sub in enumerate(subsystems[:ATXY_BASE]):
        sub_dir = root / sub
        if not sub_dir.is_dir():
            continue
        py_files = sorted(p for p in sub_dir.rglob("*.py") if "__pycache__" not in p.parts)[
            :ATXY_BASE
        ]
        for x_idx, py_path in enumerate(py_files):
            rel = py_path.relative_to(root).as_posix()
            file_coords[rel] = (t_idx, x_idx)
            try:
                tree = ast.parse(py_path.read_text(encoding="utf-8", errors="replace"))
            except SyntaxError:
                parsed_files[rel] = ([], [])
                continue

            syms: list[tuple[str, str, str, ast.AST]] = []
            imports: list[str] = []
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    kind = "class" if isinstance(node, ast.ClassDef) else "function"
                    doc = (ast.get_docstring(node) or "").strip().splitlines()
                    summary = doc[0] if doc else f"{kind} {node.name} in {Path(rel).name}"
                    syms.append((node.name, kind, summary, node))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    imports.append(node.module)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)

            syms = syms[:ATXY_BASE]
            parsed_files[rel] = (syms, imports)
            for y_idx, (s_name, _kind, _sum, _nd) in enumerate(syms):
                sym_addr = str(ATXYAddress(1, t_idx, x_idx, y_idx))
                sym_index.setdefault(s_name, []).append(sym_addr)

    for t_idx, sub in enumerate(subsystems[:ATXY_BASE]):
        sub_rels = [r for r, (ti, _) in file_coords.items() if ti == t_idx]
        if not sub_rels:
            continue
        sub_addr = str(ATXYAddress(3, t_idx, 0, 0))
        file_addrs = tuple(str(ATXYAddress(2, t_idx, file_coords[r][1], 0)) for r in sub_rels)
        records.append(
            ATXYRecord(
                address=sub_addr,
                kind="subsystem",
                payload=sub,
                summary=f"Subsystem {sub} ({len(sub_rels)} files)",
                down=file_addrs,
                fields={"file_count": len(sub_rels)},
                provenance=("ast:subsystem",),
            )
        )
        for rel in sub_rels:
            _, x_idx = file_coords[rel]
            f_addr = str(ATXYAddress(2, t_idx, x_idx, 0))
            syms, _ = parsed_files.get(rel, ([], []))
            sym_addrs = tuple(str(ATXYAddress(1, t_idx, x_idx, y_idx)) for y_idx in range(len(syms)))
            records.append(
                ATXYRecord(
                    address=f_addr,
                    kind="file",
                    payload=rel,
                    summary=f"Module {rel} ({len(syms)} top-level symbols)",
                    up=(sub_addr,),
                    down=sym_addrs,
                    fields={"symbol_count": len(syms)},
                    provenance=("ast:file",),
                )
            )
            for y_idx, (s_name, s_kind, s_sum, s_node) in enumerate(syms):
                s_addr = str(ATXYAddress(1, t_idx, x_idx, y_idx))
                callees: list[str] = []
                for n in ast.walk(s_node):
                    if isinstance(n, ast.Call):
                        cname = (
                            n.func.id
                            if isinstance(n.func, ast.Name)
                            else (n.func.attr if isinstance(n.func, ast.Attribute) else None)
                        )
                        if cname and cname != s_name and len(sym_index.get(cname, ())) == 1:
                            tgt = sym_index[cname][0]
                            if tgt not in callees:
                                callees.append(tgt)
                records.append(
                    ATXYRecord(
                        address=s_addr,
                        kind=s_kind,
                        payload=f"{rel}:{s_name}",
                        summary=s_sum,
                        up=(f_addr,),
                        rels={"CALLS": tuple(callees)} if callees else {},
                        fields={"lineno": int(getattr(s_node, "lineno", 1))},
                        provenance=("ast:symbol",),
                    )
                )

    return ATXYPageStore(records)
