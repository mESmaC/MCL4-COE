"""Full-Scale Resumable Training Runner & Checkpoint Manager for `MCL4-COE`.

Provides:
- `TrainingRunConfig`: Declarative configuration for multi-track curriculum training,
  `CPUOffloadAdamW`, optional native Blackwell `NVFP4`, checkpoint intervals, and
  live `#U$# -> #T$# -> #A$#` trace sampling.
- `MCL4COETrainer`: Executes training steps over `MultiTrackMixtureDataset` +
  `CurriculumMixtureSampler` + `HostBatchPrefetcher` + `TypedDecisionShardStreamer`,
  emitting real-time `metrics.jsonl`, `samples.jsonl`, `run_manifest.json`, and
  self-describing checkpoint receipts (`step_XXXXX.meta.json` + `step_XXXXX.pt`).
- `list_runs`, `load_run_metrics`, `list_checkpoints`, `compare_checkpoints`, and
  `evaluate_checkpoint_prompt` for the Web UI and CLI.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import gc
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable

import torch
import torch.nn.functional as F

from .backbone import MCL4CEDBackbone, MCL4CEDConfig
from .data import ByteTokenizer
from .ensemble import CognitiveSchedule, MultiPassController, compute_joint_coe_loss
from .nvfp4 import convert_linear_to_nvfp4, get_gemm_counts, reset_gemm_counts
from .operators import COEConfig, CognitiveOperatorEnsemble
from .optim import CPUOffloadAdamW, model_state_cpu, prepare_model
from .ornith_rl import OrnithSelfImprover, parameter_count
from .pipeline import (
    CurriculumMixtureSampler,
    CurriculumMixtureSchedule,
    MultiTrackMixtureDataset,
    TypedDecisionShardStreamer,
    ensure_sharded_tracks_ready,
)


def _safe_replace(tmp: Path, path: Path) -> None:
    """Replace `path` with `tmp` using short retries to avoid Windows [WinError 5] reader locks."""
    for attempt in range(8):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.015 * (attempt + 1))
    # Fallback if a reader handle persists: copy bytes in place and remove tmp
    try:
        path.write_bytes(tmp.read_bytes())
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{threading.get_ident()}.tmp")
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with tmp.open("wb") as handle:
        handle.write(raw)
        handle.flush()
    _safe_replace(tmp, path)


_REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_runs_root(path: str | Path = r"X:\MCL4-COE\runs") -> Path:
    """Resolve `runs_root`, falling back to `<repo>/runs` when the `X:` drive is not mounted."""
    p = Path(path).expanduser()
    if str(p).upper().startswith("X:") and not Path("X:\\").exists():
        return (_REPO_ROOT / "runs").resolve()
    return p.resolve()


def resolve_shards_root(path: str | Path = r"X:\MCL4-COE\data_shards") -> Path:
    """Resolve `shards_root`, falling back to `<repo>/data_shards` when the `X:` drive is not mounted."""
    p = Path(path).expanduser()
    if str(p).upper().startswith("X:") and not Path("X:\\").exists():
        return (_REPO_ROOT / "data_shards").resolve()
    return p.resolve()


def resolve_corpora_root(path: str | Path = r"X:\corpora") -> Path:
    """Resolve `corpora_root`, falling back to `<repo>/corpora` when the `X:` drive is not mounted."""
    p = Path(path).expanduser()
    if str(p).upper().startswith("X:") and not Path("X:\\").exists():
        return (_REPO_ROOT / "corpora").resolve()
    return p.resolve()


@dataclass
class TrainingRunConfig:
    """Configuration for an `MCL4-COE` training run."""

    run_id: str = "run_default"
    runs_root: str = r"X:\MCL4-COE\runs"
    shards_root: str = r"X:\MCL4-COE\data_shards"
    corpora_root: str = r"X:\corpora"
    resume_checkpoint: str | None = None

    def __post_init__(self) -> None:
        self.runs_root = str(resolve_runs_root(self.runs_root))
        self.shards_root = str(resolve_shards_root(self.shards_root))
        self.corpora_root = str(resolve_corpora_root(self.corpora_root))

    # Model architecture
    d_model: int = 128
    n_heads: int = 4
    n_encoder_layers: int = 2
    n_decoder_layers: int = 2
    ple_dim: int = 32
    expert_hidden_dim: int = 256
    num_experts: int = 4
    top_k: int = 2
    max_seq_len: int = 2048
    projection_dim: int = 32
    recurrence_steps: int = 1
    use_nvfp4: bool = False
    nvfp4_sparsity_mode: str = "2to4_transposable"
    nvfp4_sr_ste_decay: float = 1e-4
    cpu_offload_optimizer: bool = False

    # Optimization & batching
    total_steps: int = 50
    batch_size: int = 8
    micro_batch_size: int = 4
    decision_batch_size: int = 4
    lr: float = 2.5e-3
    warmup_ratio: float = 0.08
    min_lr_ratio: float = 0.12
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    seed: int = 42

    # Curriculum schedule
    curriculum_mode: str = "cosine"
    start_weights: dict[str, float] = field(
        default_factory=lambda: {
            "pretrain_lm": 0.65,
            "uta_trace": 0.25,
            "typed_decision": 0.10,
        }
    )
    end_weights: dict[str, float] = field(
        default_factory=lambda: {
            "pretrain_lm": 0.15,
            "uta_trace": 0.50,
            "typed_decision": 0.35,
        }
    )

    # Ornith-1.0 / 1.5 Self-Scaffolding & Self-Improvement + Dream-RSI loop
    enable_ornith_rl: bool = True
    ornith_every: int = 2
    ornith_weight: float = 0.15
    enable_dream_rsi: bool = True
    dream_every: int = 2
    dream_max_rounds: int = 3
    dream_max_calls: int = 4
    dream_workers: int = 2
    dream_revisions: int = 3

    # Checkpointing & live telemetry
    checkpoint_every: int = 10
    sample_every: int = 10
    sample_prompt: str = "How do plants use capillary action to transport water?"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingRunConfig:
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


def upcycle_state_dict_experts(
    state_dict: dict[str, torch.Tensor],
    old_num_experts: int,
    new_num_experts: int,
    *,
    expert_loads: list[float] | None = None,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """Upcycle a `SparseMoE` checkpoint from `old_num_experts` to `new_num_experts` via mitosis.

    Clones the highest-utilization parent experts into the new slots `[old_num_experts .. new_num_experts-1]`
    with symmetric $\\pm \\delta$ orthogonal/Gaussian perturbations ($\\|\\delta\\| = 10^{-3} \\sigma_W$) so the
    ensemble mean $\\frac{1}{2}(f(x; W_p - \\delta) + f(x; W_p + \\delta))$ is preserved to second order
    at the split step while breaking gradient symmetry immediately on the next step.
    """
    if new_num_experts <= old_num_experts:
        return state_dict

    out = {k: v.clone() for k, v in state_dict.items()}
    if expert_loads and len(expert_loads) == old_num_experts:
        parent_order = sorted(
            range(old_num_experts),
            key=lambda i: float(expert_loads[i]),
            reverse=True,
        )
    else:
        parent_order = list(range(old_num_experts))

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    router_suffix = ".moe.router.weight"
    moe_prefixes = [k[: -len(".router.weight")] for k in out if k.endswith(router_suffix)]

    for prefix in sorted(moe_prefixes):
        r_key = f"{prefix}.router.weight"
        r_weight = out[r_key]
        extra_rows = []
        for new_e in range(old_num_experts, new_num_experts):
            parent_e = parent_order[(new_e - old_num_experts) % old_num_experts]
            extra_rows.append(r_weight[parent_e : parent_e + 1].clone())
            for proj in ("up_proj.weight", "down_proj.weight"):
                p_key = f"{prefix}.experts.{parent_e}.{proj}"
                c_key = f"{prefix}.experts.{new_e}.{proj}"
                w_p = out[p_key]
                std = float(w_p.float().std().item())
                delta = (
                    torch.randn(w_p.shape, generator=gen, dtype=torch.float32) * (1e-3 * max(std, 1e-5))
                ).to(w_p.dtype)
                out[p_key] = w_p - delta
                out[c_key] = w_p + delta
        out[r_key] = torch.cat([r_weight, *extra_rows], dim=0)

    return out


class MCL4COETrainer:
    """Resumable multi-track trainer with live JSONL telemetry and checkpoint receipts."""

    def __init__(
        self,
        config: TrainingRunConfig,
        *,
        stop_event: threading.Event | None = None,
        on_step_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.config = config
        self.stop_event = stop_event or threading.Event()
        self.on_step_callback = on_step_callback
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = ByteTokenizer()

        self.run_dir = Path(config.runs_root).expanduser().resolve() / config.run_id
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.manifest_path = self.run_dir / "run_manifest.json"
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.samples_path = self.run_dir / "samples.jsonl"

        self._resume_payload: dict[str, Any] | None = None
        if config.resume_checkpoint:
            self._prepare_resume_geometry(config.resume_checkpoint)

        torch.manual_seed(config.seed)
        self._build_dataset_and_samplers()
        self._build_model_and_optimizer()

        self.current_step = 0
        self.start_time = time.time()
        self.cumulative_wall_s = 0.0
        self.cumulative_tokens = 0
        self.latest_metrics: dict[str, Any] = {}

        if config.resume_checkpoint:
            self.load_checkpoint(config.resume_checkpoint)

    def _prepare_resume_geometry(self, checkpoint_path: str | Path) -> None:
        """Align backbone dimensions with the checkpoint while allowing `num_experts` upcycling."""
        p = Path(checkpoint_path).expanduser().resolve()
        if p.name.endswith(".meta.json"):
            meta = json.loads(p.read_text(encoding="utf-8"))
            pt_candidate = Path(meta["pt_file"])
            p = pt_candidate if pt_candidate.exists() else p.with_name(pt_candidate.name)
        payload = torch.load(p, map_location="cpu", weights_only=False)
        self._resume_payload = payload
        b_cfg = payload.get("backbone_config", {})
        for key in (
            "d_model",
            "n_heads",
            "n_encoder_layers",
            "n_decoder_layers",
            "ple_dim",
            "expert_hidden_dim",
        ):
            if key in b_cfg:
                setattr(self.config, key, int(b_cfg[key]))
        ckpt_experts = int(b_cfg.get("num_experts", self.config.num_experts))
        if self.config.num_experts < ckpt_experts:
            self.config.num_experts = ckpt_experts
        if self.config.top_k > self.config.num_experts:
            self.config.top_k = self.config.num_experts

    def _build_dataset_and_samplers(self) -> None:
        from .data import HostBatchPrefetcher, collate_decisions

        shards_root = Path(self.config.shards_root).expanduser().resolve()
        track_receipts = ensure_sharded_tracks_ready(
            shards_root,
            corpora_root=self.config.corpora_root,
            seq_len=min(2048, self.config.max_seq_len),
        )
        if not track_receipts:
            raise FileNotFoundError(
                f"No index_receipt.json files found under {shards_root}. "
                "Run `python scripts/build_shards.py` first."
            )

        self.dataset = MultiTrackMixtureDataset(track_receipts)
        if self.config.max_seq_len < self.dataset.seq_len:
            self.config.max_seq_len = self.dataset.seq_len

        start_w = {k: v for k, v in self.config.start_weights.items() if k in self.dataset.track_names}
        end_w = {k: v for k, v in self.config.end_weights.items() if k in self.dataset.track_names}
        self.schedule = CurriculumMixtureSchedule(
            start_weights=start_w,
            end_weights=end_w,
            mode=self.config.curriculum_mode,
        )
        self.sampler = CurriculumMixtureSampler(
            self.dataset,
            self.schedule,
            seed=self.config.seed,
        )
        self.prefetcher = HostBatchPrefetcher(
            self.dataset,
            micro_batch_size=min(self.config.micro_batch_size, self.config.batch_size),
            pin_memory=(self.device.type == "cuda"),
        )

        self.decision_streamer: TypedDecisionShardStreamer | None = None
        self.decision_batches: list[dict[str, torch.Tensor]] = []
        self._decision_cursor = 0
        dec_cache = shards_root / "typed_decision" / "decision_examples_cache.pt"
        ts_dir = Path(self.config.corpora_root) / "text" / "tasksource-jev-typed-decisions"
        if dec_cache.is_file() or ts_dir.is_dir():
            dec_seq_len = min(384, self.config.max_seq_len)
            self.decision_streamer = TypedDecisionShardStreamer(
                self.config.corpora_root,
                max_seq_len=dec_seq_len,
                max_options=6,
                cache_path=dec_cache,
                max_cached_examples=512,
            )
            pool = list(self.decision_streamer.iter_decision_examples())
            bsz = max(1, self.config.decision_batch_size)
            for i in range(0, len(pool) - bsz + 1, bsz):
                collated = collate_decisions(pool[i : i + bsz], dec_seq_len, device="cpu")
                if self.device.type == "cuda":
                    collated = {k: v.pin_memory() for k, v in collated.items()}
                self.decision_batches.append(collated)

    def _next_decision_batch(self) -> dict[str, torch.Tensor] | None:
        if not self.decision_batches:
            return None
        batch_cpu = self.decision_batches[self._decision_cursor % len(self.decision_batches)]
        self._decision_cursor += 1
        return {k: v.to(self.device, non_blocking=True) for k, v in batch_cpu.items()}

    def _build_model_and_optimizer(self) -> None:
        self.backbone_cfg = MCL4CEDConfig(
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            n_encoder_layers=self.config.n_encoder_layers,
            n_decoder_layers=self.config.n_decoder_layers,
            ple_dim=self.config.ple_dim,
            expert_hidden_dim=self.config.expert_hidden_dim,
            num_experts=self.config.num_experts,
            top_k=self.config.top_k,
            max_seq_len=self.config.max_seq_len,
        )
        self.coe_cfg = COEConfig(
            projection_dim=self.config.projection_dim,
            initial_scale=5.0,
            recurrence_steps=self.config.recurrence_steps,
        )
        self.model = CognitiveOperatorEnsemble(
            MCL4CEDBackbone(self.backbone_cfg),
            self.coe_cfg,
        )
        if self.config.use_nvfp4 and self.device.type == "cuda":
            reset_gemm_counts()
            self.nvfp4_layers_converted = convert_linear_to_nvfp4(
                self.model,
                min_features=32,
                experts_only=True,
                sparsity_mode=self.config.nvfp4_sparsity_mode,
                sr_ste_decay=self.config.nvfp4_sr_ste_decay,
                causal_activation_scale=True,
            )
        else:
            self.nvfp4_layers_converted = 0

        if self.config.cpu_offload_optimizer:
            prepare_model(
                self.model,
                device=self.device,
                bf16_working_weights=(self.device.type == "cuda" and not self.config.use_nvfp4),
            )
        else:
            self.model.to(self.device)

        decay_params = [p for p in self.model.parameters() if p.requires_grad and p.ndim >= 2]
        nodecay_params = [p for p in self.model.parameters() if p.requires_grad and p.ndim < 2]
        param_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        if self.config.cpu_offload_optimizer:
            self.optimizer: Any = CPUOffloadAdamW(
                param_groups,
                lr=self.config.lr,
                betas=(0.9, 0.95),
                weight_decay=self.config.weight_decay,
                transfer_mode="batched",
            )
        else:
            use_fused = (self.device.type == "cuda") and (not self.config.use_nvfp4)
            self.optimizer = torch.optim.AdamW(
                param_groups,
                lr=self.config.lr,
                betas=(0.9, 0.95),
                fused=use_fused,
            )
        self.controller = MultiPassController(
            self.model,
            schedule=CognitiveSchedule(
                operators=("abduction", "association", "decision", "deduction", "evaluation", "metacognition"),
                recurrence_steps=self.config.recurrence_steps,
            ),
        )
        from .dream_rsi import DreamRSIEngine

        dream_engine = (
            DreamRSIEngine(
                max_rounds=self.config.dream_max_rounds,
                workers=self.config.dream_workers,
                max_calls=self.config.dream_max_calls,
                revisions_per_cycle=self.config.dream_revisions,
                out_dir=self.run_dir / "dream",
                seed=self.config.seed,
            )
            if self.config.enable_dream_rsi
            else None
        )
        self.ornith_improver = OrnithSelfImprover(
            projection_dim=self.config.projection_dim,
            max_seq_len=min(512, self.config.max_seq_len),
            seed=self.config.seed,
            enable_dream_rsi=self.config.enable_dream_rsi,
            dream_every=self.config.dream_every,
            dream_engine=dream_engine,
        )

    def _write_manifest(self, status: str) -> None:
        payload = {
            "run_id": self.config.run_id,
            "status": status,
            "created_at": self.start_time,
            "updated_at": time.time(),
            "current_step": self.current_step,
            "total_steps": self.config.total_steps,
            "device": str(self.device),
            "gpu_name": torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU",
            "parameter_count": parameter_count(self.model),
            "num_experts": self.model.backbone.config.num_experts,
            "nvfp4_layers": self.nvfp4_layers_converted,
            "dataset_seq_len": self.dataset.seq_len,
            "dataset_total_blocks": len(self.dataset),
            "track_blocks": dict(zip(self.dataset.track_names, self.dataset.track_block_counts)),
            "config": self.config.to_dict(),
            "latest_metrics": self.latest_metrics,
        }
        _atomic_json(self.manifest_path, payload)

    @torch.no_grad()
    def _accumulate_per_track_losses_gpu(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        track_ids: torch.Tensor,
        accum_sums: dict[str, Tensor],
        accum_counts: dict[str, Tensor],
    ) -> None:
        """Accumulate unmasked token cross-entropy per track on GPU without host sync."""
        b, seq_len, vocab = logits.shape
        token_losses = F.cross_entropy(
            logits.float().view(-1, vocab),
            labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(b, seq_len)
        valid_mask = (labels != -100).float()
        for t_idx, t_name in enumerate(self.dataset.track_names):
            row_mask = (track_ids == t_idx).unsqueeze(-1).float() * valid_mask
            key = f"loss_{t_name}"
            accum_sums[key] = accum_sums[key] + (token_losses * row_mask).sum()
            accum_counts[key] = accum_counts[key] + row_mask.sum()

    def save_checkpoint(
        self,
        tag: str | None = None,
        *,
        sample_uta: str | None = None,
    ) -> Path:
        """Save `.pt` checkpoint + lightweight `.meta.json` receipt for the Web UI."""
        step_name = tag or f"step_{self.current_step:05d}"
        pt_path = self.ckpt_dir / f"{step_name}.pt"
        meta_path = self.ckpt_dir / f"{step_name}.meta.json"

        if sample_uta is None:
            sample_uta = self.controller.generate_uta(
                self.config.sample_prompt,
                max_new_tokens=48,
            )
        meta = {
            "checkpoint_id": f"{self.config.run_id}/{step_name}",
            "run_id": self.config.run_id,
            "step": self.current_step,
            "total_steps": self.config.total_steps,
            "timestamp": time.time(),
            "pt_file": str(pt_path),
            "meta_file": str(meta_path),
            "parameter_count": parameter_count(self.model),
            "num_experts": self.model.backbone.config.num_experts,
            "use_nvfp4": self.config.use_nvfp4,
            "metrics": dict(self.latest_metrics),
            "sample_prompt": self.config.sample_prompt,
            "sample_uta": sample_uta,
            "config": self.config.to_dict(),
        }

        opt_state_cpu: dict[str, Any] = {}
        try:
            raw_opt = self.optimizer.state_dict()
            opt_state_cpu = {
                "param_groups": raw_opt.get("param_groups", []),
                "state": {
                    k: {
                        sk: (sv.detach().cpu() if isinstance(sv, torch.Tensor) else sv)
                        for sk, sv in v.items()
                    }
                    for k, v in raw_opt.get("state", {}).items()
                },
            }
        except Exception:
            pass

        state_payload = {
            "step": self.current_step,
            "config": self.config.to_dict(),
            "backbone_config": asdict(self.backbone_cfg),
            "coe_config": asdict(self.coe_cfg),
            "model_state": model_state_cpu(self.model),
            "optimizer_state": opt_state_cpu,
            "sampler_state": self.sampler.state_dict(),
            "cumulative_wall_s": self.cumulative_wall_s,
            "cumulative_tokens": self.cumulative_tokens,
            "latest_metrics": self.latest_metrics,
        }
        tmp_pt = pt_path.with_name(f"{pt_path.name}.{threading.get_ident()}.tmp")
        torch.save(state_payload, tmp_pt)
        _safe_replace(tmp_pt, pt_path)
        _atomic_json(meta_path, meta)
        return meta_path

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Restore model weights (upcycling experts if `num_experts` increased), optimizer state, and sampler state."""
        ckpt_resolved = Path(checkpoint_path).expanduser().resolve()
        if ckpt_resolved.name.endswith(".meta.json") and ckpt_resolved.is_file():
            meta = json.loads(ckpt_resolved.read_text(encoding="utf-8"))
            pt_candidate = Path(meta["pt_file"])
            ckpt_pt_path = pt_candidate if pt_candidate.exists() else ckpt_resolved.with_name(pt_candidate.name)
        else:
            ckpt_pt_path = ckpt_resolved

        if self._resume_payload is not None:
            payload = self._resume_payload
            self._resume_payload = None
        else:
            payload = torch.load(ckpt_pt_path, map_location="cpu", weights_only=False)

        b_cfg = payload.get("backbone_config", {})
        ckpt_experts = int(b_cfg.get("num_experts", self.config.num_experts))
        model_state = payload["model_state"]
        if self.config.num_experts > ckpt_experts:
            prev_loads = payload.get("latest_metrics", {}).get("expert_load")
            model_state = upcycle_state_dict_experts(
                model_state,
                ckpt_experts,
                self.config.num_experts,
                expert_loads=prev_loads,
                seed=self.config.seed,
            )

        self.model.load_state_dict(model_state, strict=False)
        if self.config.cpu_offload_optimizer:
            prepare_model(
                self.model,
                device=self.device,
                bf16_working_weights=(self.device.type == "cuda" and not self.config.use_nvfp4),
                preserve_master_weights=False,
            )
            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    if p in self.optimizer.state and "master" in self.optimizer.state[p]:
                        self.optimizer.state[p]["master"].copy_(
                            p.detach().to(device="cpu", dtype=torch.float32)
                        )
        else:
            self.model.to(self.device)
            if (
                self.config.num_experts == ckpt_experts
                and payload.get("optimizer_state")
            ):
                try:
                    self.optimizer.load_state_dict(payload["optimizer_state"])
                except Exception:
                    pass
        if "sampler_state" in payload:
            self.sampler.load_state_dict(payload["sampler_state"])
        self.current_step = int(payload.get("step", 0))
        self.initial_step = self.current_step
        self.cumulative_wall_s = float(payload.get("cumulative_wall_s", 0.0))
        self.cumulative_tokens = int(payload.get("cumulative_tokens", 0))
        self.latest_metrics = dict(payload.get("latest_metrics", {}))

        # Carry forward parent run's Dream-RSI world archive (`dream_state.json`) if present
        if (
            self.ornith_improver is not None
            and self.ornith_improver.dream_engine is not None
            and not self.ornith_improver.dream_engine.state.worlds
        ):
            parent_dream = ckpt_pt_path.parent.parent / "dream" / "dream_state.json"
            if parent_dream.is_file():
                try:
                    from .dream_rsi import DreamState

                    raw_d = json.loads(parent_dream.read_text(encoding="utf-8"))
                    self.ornith_improver.dream_engine.state = DreamState.from_dict(raw_d)
                    self.ornith_improver.dream_engine.save_artifacts()
                except Exception:
                    pass

        # If resuming in-place in an existing run dir, prune any steps > current_step
        for jsonl_file in (self.metrics_path, self.samples_path):
            if jsonl_file.is_file():
                try:
                    kept = []
                    for line in jsonl_file.read_text(encoding="utf-8").splitlines():
                        if not line.strip():
                            continue
                        obj = json.loads(line)
                        if int(obj.get("step", 0)) <= self.current_step:
                            kept.append(line)
                    jsonl_file.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
                except Exception:
                    pass

        # Seed a fresh upcycled/resumed run's metrics.jsonl with the checkpoint anchor point
        if not self.metrics_path.is_file() and self.latest_metrics and self.current_step > 0:
            anchor = dict(self.latest_metrics)
            anchor["step"] = self.current_step
            anchor["total_steps"] = self.config.total_steps
            with self.metrics_path.open("w", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(anchor, sort_keys=True) + "\n")

    def _compute_lr(self, step: int) -> float:
        base_step = getattr(self, "initial_step", 0)
        stage_steps = max(1, self.config.total_steps - base_step)
        local_step = max(1, step - base_step)
        # Ensure resumed stages without optimizer moments warm up smoothly over >= 12% of stage steps
        eff_warmup = max(self.config.warmup_ratio, 0.12 if base_step > 0 else 0.0)
        warmup_steps = max(1, int(round(stage_steps * eff_warmup)))
        if local_step <= warmup_steps:
            return self.config.lr * (local_step / warmup_steps)
        decay_span = max(1, stage_steps - warmup_steps)
        decay_progress = min(1.0, max(0.0, (local_step - warmup_steps) / decay_span))
        cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        return self.config.lr * (
            self.config.min_lr_ratio + (1.0 - self.config.min_lr_ratio) * cosine
        )

    def train(self) -> dict[str, Any]:
        """Run the training loop until `total_steps` or `stop_event` is set."""
        self._write_manifest("running")
        if self.device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
        status = "completed"
        try:
            start_step = self.current_step + 1
            if start_step > self.config.total_steps:
                self._write_manifest("completed")
                return self.latest_metrics

            base_step = getattr(self, "initial_step", 0)
            stage_steps = max(1, self.config.total_steps - base_step)
            progress0 = max(0.0, (start_step - 1 - base_step) / stage_steps)
            next_indices = self.sampler.take(self.config.batch_size, progress=progress0)
            self.prefetcher.schedule(next_indices)

            for step in range(start_step, self.config.total_steps + 1):
                if self.stop_event.is_set():
                    status = "stopped"
                    break

                t_step0 = time.perf_counter()
                self.current_step = step
                progress = min(1.0, max(0.0, (step - base_step) / stage_steps))
                cur_weights = self.schedule.weights_at(progress)
                w_dec = float(cur_weights.get("typed_decision", 0.15))

                lr = self._compute_lr(step)
                for group in self.optimizer.param_groups:
                    group["lr"] = lr

                cur_indices = next_indices
                quotas = self.sampler.allocate_quotas(self.config.batch_size, progress=progress)
                if step < self.config.total_steps:
                    next_indices = self.sampler.take(self.config.batch_size, progress=progress)

                micro_batches = self.prefetcher.get(cur_indices)
                if step < self.config.total_steps:
                    self.prefetcher.schedule(next_indices)

                dec_batch = self._next_decision_batch()
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)

                step_total_gpu = torch.zeros((), device=self.device, dtype=torch.float32)
                step_lm_gpu = torch.zeros((), device=self.device, dtype=torch.float32)
                step_dec_gpu = torch.zeros((), device=self.device, dtype=torch.float32)
                step_ded_gpu = torch.zeros((), device=self.device, dtype=torch.float32)
                step_eval_gpu = torch.zeros((), device=self.device, dtype=torch.float32)
                step_meta_gpu = torch.zeros((), device=self.device, dtype=torch.float32)
                step_abs_gpu = torch.zeros((), device=self.device, dtype=torch.float32)
                step_ornith_gpu = torch.zeros((), device=self.device, dtype=torch.float32)
                step_tokens_gpu = torch.zeros((), device=self.device, dtype=torch.long)
                step_expert_load_gpu = torch.zeros(
                    (self.model.backbone.config.num_experts,),
                    device=self.device,
                    dtype=torch.float32,
                )
                track_sums_gpu = {
                    f"loss_{t}": torch.zeros((), device=self.device, dtype=torch.float32)
                    for t in self.dataset.track_names
                }
                track_counts_gpu = {
                    f"loss_{t}": torch.zeros((), device=self.device, dtype=torch.float32)
                    for t in self.dataset.track_names
                }

                ornith_telemetry: dict[str, Any] = {}
                run_ornith = bool(
                    self.config.enable_ornith_rl
                    and (step % max(1, self.config.ornith_every) == 0 or step == 1)
                )

                use_amp = self.device.type == "cuda"
                num_mb = len(micro_batches)
                # Token-weighted micro-batch scaling eliminates gradient/metric variance when
                # sparse `0x8000`-masked `typed_decision` rows mix with dense `uta_trace` rows.
                mb_token_counts = [
                    max(1, int((mb["labels"] != -100).sum().item())) for mb in micro_batches
                ]
                total_mb_tokens = max(1, sum(mb_token_counts))

                for mb, mb_toks in zip(micro_batches, mb_token_counts):
                    inp = mb["input_ids"].to(self.device, non_blocking=True)
                    lbl = mb["labels"].to(self.device, non_blocking=True)
                    t_ids = mb["track_ids"].to(self.device, non_blocking=True)
                    step_tokens_gpu = step_tokens_gpu + mb_toks
                    mb_frac = float(mb_toks) / float(total_mb_tokens)

                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=torch.bfloat16,
                        enabled=use_amp,
                    ):
                        lm_out = self.model.forward_inference(inp, targets=lbl)
                        assert lm_out.loss is not None and lm_out.ce_loss is not None
                        lm_scale = (1.0 - w_dec) * mb_frac if dec_batch is not None else mb_frac
                        scaled_lm_loss = lm_out.loss * lm_scale

                    step_lm_gpu = step_lm_gpu + (lm_out.ce_loss.detach().float() * mb_frac)
                    step_total_gpu = step_total_gpu + scaled_lm_loss.detach().float()
                    step_expert_load_gpu = step_expert_load_gpu + (
                        lm_out.expert_load.detach().float().mean(dim=0) / num_mb
                    )
                    self._accumulate_per_track_losses_gpu(
                        lm_out.logits.detach(),
                        lbl,
                        t_ids,
                        track_sums_gpu,
                        track_counts_gpu,
                    )
                    del lm_out
                    scaled_lm_loss.backward()
                    del scaled_lm_loss

                if dec_batch is not None:
                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=torch.bfloat16,
                        enabled=use_amp,
                    ):
                        op_out = self.model.forward_all_operators(**dec_batch)
                        assert op_out.decision.ce_loss is not None
                        step_dec_gpu = op_out.decision.ce_loss.detach().float()
                        step_ded_gpu = op_out.deduction_loss.detach().float()
                        step_eval_gpu = op_out.evaluation_loss.detach().float()
                        step_meta_gpu = op_out.metacognition_loss.detach().float()
                        step_abs_gpu = op_out.abstraction_loss.detach().float()
                        scaled_op_loss = w_dec * op_out.total_loss
                    step_total_gpu = step_total_gpu + scaled_op_loss.detach().float()
                    del op_out
                    scaled_op_loss.backward()
                    del scaled_op_loss

                if run_ornith:
                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=torch.bfloat16,
                        enabled=use_amp,
                    ):
                        ornith_out = self.ornith_improver.step_self_improvement(self.model)
                        step_ornith_gpu = ornith_out.loss.detach().float()
                        scaled_ornith_loss = self.config.ornith_weight * ornith_out.loss
                        ornith_telemetry = {
                            "ornith_loss": round(float(step_ornith_gpu.item()), 5),
                            "ornith_task_reward": round(ornith_out.task_reward, 4),
                            "ornith_harness_reward": round(ornith_out.harness_reward, 4),
                            "ornith_rollout_reward": round(ornith_out.mean_rollout_reward, 4),
                            "ornith_novelty": round(ornith_out.novelty, 4),
                            "ornith_pass_rate": round(ornith_out.pass_rate, 4),
                            "scaffolds_cached": ornith_out.scaffolds_cached,
                            "novelty_buffer_size": ornith_out.novelty_buffer_size,
                            "dream_cycle": ornith_out.dream_cycle,
                            "dream_replay_score": round(ornith_out.dream_replay_score, 5),
                            "dream_initial_score": round(ornith_out.dream_initial_score, 5),
                            "dream_accepted_revisions": ornith_out.dream_accepted_revisions,
                            "dream_replays_dreamed": ornith_out.dream_replays_dreamed,
                            "dream_worlds_archived": ornith_out.dream_worlds_archived,
                            "dream_active_policy": ornith_out.dream_active_policy,
                        }
                    step_total_gpu = step_total_gpu + scaled_ornith_loss.detach().float()
                    del ornith_out
                    scaled_ornith_loss.backward()
                    del scaled_ornith_loss

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()

                # Single GPU-to-CPU sync point per step after optimizer.step()
                step_unmasked_tokens = int(step_tokens_gpu.item())
                step_lm_loss = float(step_lm_gpu.item())
                step_dec_loss = float(step_dec_gpu.item())
                step_ded_loss = float(step_ded_gpu.item())
                step_eval_loss = float(step_eval_gpu.item())
                step_meta_loss = float(step_meta_gpu.item())
                step_abs_loss = float(step_abs_gpu.item())
                step_total_loss = float(step_total_gpu.item())
                expert_loads = [round(float(x), 4) for x in step_expert_load_gpu.cpu().tolist()]
                step_elapsed = time.perf_counter() - t_step0
                self.cumulative_wall_s += step_elapsed
                self.cumulative_tokens += step_unmasked_tokens
                tok_per_s = step_unmasked_tokens / max(step_elapsed, 1e-6)

                avg_track_losses: dict[str, float] = {}
                for t_name in self.dataset.track_names:
                    key = f"loss_{t_name}"
                    cnt = float(track_counts_gpu[key].item())
                    if cnt > 0:
                        avg_track_losses[key] = float(track_sums_gpu[key].item()) / cnt
                    elif key in self.latest_metrics:
                        avg_track_losses[key] = self.latest_metrics[key]

                if not ornith_telemetry and self.latest_metrics:
                    for ok in (
                        "ornith_loss",
                        "ornith_task_reward",
                        "ornith_harness_reward",
                        "ornith_rollout_reward",
                        "ornith_novelty",
                        "ornith_pass_rate",
                        "scaffolds_cached",
                        "novelty_buffer_size",
                        "dream_cycle",
                        "dream_replay_score",
                        "dream_initial_score",
                        "dream_accepted_revisions",
                        "dream_replays_dreamed",
                        "dream_worlds_archived",
                        "dream_active_policy",
                    ):
                        if ok in self.latest_metrics:
                            ornith_telemetry[ok] = self.latest_metrics[ok]

                vram_alloc_mb = (
                    torch.cuda.memory_allocated(0) / (1024**2) if self.device.type == "cuda" else 0.0
                )
                vram_res_mb = (
                    torch.cuda.memory_reserved(0) / (1024**2) if self.device.type == "cuda" else 0.0
                )
                gemm_counts = (
                    get_gemm_counts()
                    if self.config.use_nvfp4
                    else {"forward": 0, "dgrad": 0, "wgrad": 0, "sparse_2to4": 0}
                )

                record: dict[str, Any] = {
                    "step": step,
                    "total_steps": self.config.total_steps,
                    "progress": round(progress, 4),
                    "lr": round(lr, 6),
                    "total_loss": round(step_total_loss, 5),
                    "lm_loss": round(step_lm_loss, 5),
                    "decision_loss": round(step_dec_loss, 5),
                    "deduction_loss": round(step_ded_loss, 5),
                    "evaluation_loss": round(step_eval_loss, 5),
                    "metacognition_loss": round(step_meta_loss, 5),
                    "abstraction_loss": round(step_abs_loss, 5),
                    "tok_per_s": round(tok_per_s, 1),
                    "step_tokens": step_unmasked_tokens,
                    "cumulative_tokens": self.cumulative_tokens,
                    "wall_time_s": round(self.cumulative_wall_s, 2),
                    "step_time_s": round(step_elapsed, 3),
                    "vram_allocated_mb": round(vram_alloc_mb, 1),
                    "vram_reserved_mb": round(vram_res_mb, 1),
                    "num_experts": self.model.backbone.config.num_experts,
                    "expert_load": expert_loads,
                    "quotas": quotas,
                    "weights": {k: round(v, 4) for k, v in cur_weights.items()},
                    "nvfp4_gemms": gemm_counts["forward"] + gemm_counts["dgrad"] + gemm_counts["wgrad"],
                    "nvfp4_sparse_2to4_gemms": gemm_counts.get("sparse_2to4", 0),
                    **ornith_telemetry,
                    **{k: round(v, 5) for k, v in avg_track_losses.items()},
                }
                self.latest_metrics = record
                with self.metrics_path.open("a", encoding="utf-8", newline="\n") as fh:
                    fh.write(json.dumps(record, sort_keys=True) + "\n")

                sample_trace: str | None = None
                if step % self.config.sample_every == 0 or step == 1 or step == self.config.total_steps:
                    sample_trace = self.controller.generate_uta(
                        self.config.sample_prompt,
                        max_new_tokens=48,
                    )
                    sample_entry = {
                        "step": step,
                        "prompt": self.config.sample_prompt,
                        "trace": sample_trace,
                        "total_loss": record["total_loss"],
                        "timestamp": time.time(),
                    }
                    with self.samples_path.open("a", encoding="utf-8", newline="\n") as sh:
                        sh.write(json.dumps(sample_entry, sort_keys=True) + "\n")

                if step % self.config.checkpoint_every == 0 or step == self.config.total_steps:
                    self.save_checkpoint(sample_uta=sample_trace)

                self._write_manifest("running")
                if self.on_step_callback is not None:
                    self.on_step_callback(record)

            if status == "stopped":
                self.save_checkpoint(tag=f"step_{self.current_step:05d}_stopped")
        except Exception:
            status = "failed"
            raise
        finally:
            self.prefetcher.close()
            self._write_manifest(status)
            if self.device.type == "cuda":
                gc.collect()
                torch.cuda.empty_cache()
        return self.latest_metrics


def list_runs(runs_root: str | Path = r"X:\MCL4-COE\runs") -> list[dict[str, Any]]:
    """Return all training runs sorted newest-first from `runs_root`."""
    root = resolve_runs_root(runs_root)
    if not root.is_dir():
        return []
    runs: list[dict[str, Any]] = []
    for sub in root.iterdir():
        manifest = sub / "run_manifest.json"
        if manifest.is_file():
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
                runs.append(data)
            except Exception:
                continue
    runs.sort(key=lambda r: float(r.get("updated_at", 0.0)), reverse=True)
    return runs


def delete_run(run_id: str, runs_root: str | Path = r"X:\MCL4-COE\runs") -> bool:
    """Delete a specific run directory and its checkpoints from `runs_root`."""
    import shutil

    root = resolve_runs_root(runs_root)
    target = (root / run_id).resolve()
    if target.parent != root or not target.is_dir():
        return False
    shutil.rmtree(target, ignore_errors=True)
    return not target.exists()


def clear_all_runs(
    runs_root: str | Path = r"X:\MCL4-COE\runs",
    *,
    exclude_run_id: str | None = None,
) -> int:
    """Remove all completed/historical run directories under `runs_root`."""
    import shutil

    root = resolve_runs_root(runs_root)
    if not root.is_dir():
        return 0
    removed = 0
    for sub in list(root.iterdir()):
        if not sub.is_dir():
            continue
        if exclude_run_id and sub.name == exclude_run_id:
            continue
        shutil.rmtree(sub, ignore_errors=True)
        if not sub.exists():
            removed += 1
    return removed


def load_run_metrics(
    run_id: str,
    runs_root: str | Path = r"X:\MCL4-COE\runs",
) -> dict[str, Any]:
    """Load `run_manifest.json`, `metrics.jsonl`, and `samples.jsonl` for a specific `run_id`."""
    run_dir = resolve_runs_root(runs_root) / run_id
    manifest_file = run_dir / "run_manifest.json"
    metrics_file = run_dir / "metrics.jsonl"
    samples_file = run_dir / "samples.jsonl"

    manifest = (
        json.loads(manifest_file.read_text(encoding="utf-8"))
        if manifest_file.is_file()
        else {"run_id": run_id}
    )
    points: list[dict[str, Any]] = []
    if metrics_file.is_file():
        for line in metrics_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    points.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    samples: list[dict[str, Any]] = []
    if samples_file.is_file():
        for line in samples_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    return {
        "manifest": manifest,
        "metrics": points,
        "samples": samples,
    }


def list_checkpoints(runs_root: str | Path = r"X:\MCL4-COE\runs") -> list[dict[str, Any]]:
    """List all saved checkpoint `.meta.json` receipts across all runs."""
    root = resolve_runs_root(runs_root)
    if not root.is_dir():
        return []
    ckpts: list[dict[str, Any]] = []
    for meta_file in root.rglob("*.meta.json"):
        try:
            entry = json.loads(meta_file.read_text(encoding="utf-8"))
            entry["meta_file"] = str(meta_file.resolve())
            pt_candidate = Path(entry.get("pt_file", ""))
            if not pt_candidate.exists():
                entry["pt_file"] = str(meta_file.with_name(pt_candidate.name).resolve())
            ckpts.append(entry)
        except Exception:
            continue
    ckpts.sort(key=lambda c: (float(c.get("timestamp", 0.0)), int(c.get("step", 0))), reverse=True)
    return ckpts


@torch.no_grad()
def evaluate_checkpoint_prompt(
    meta_or_pt_path: str | Path,
    prompt: str,
    *,
    max_new_tokens: int = 64,
) -> dict[str, Any]:
    """Load a saved checkpoint and generate a `#U$# -> #T$# -> #A$#` trace for interactive comparison."""
    p = Path(meta_or_pt_path).expanduser().resolve()
    if p.name.endswith(".meta.json"):
        meta = json.loads(p.read_text(encoding="utf-8"))
        pt_candidate = Path(meta["pt_file"])
        pt_path = pt_candidate if pt_candidate.exists() else p.with_name(pt_candidate.name)
    else:
        pt_path = p
        meta = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    b_cfg = MCL4CEDConfig(**payload["backbone_config"])
    c_cfg = COEConfig(**payload["coe_config"])
    model = CognitiveOperatorEnsemble(MCL4CEDBackbone(b_cfg), c_cfg)
    model.load_state_dict(payload["model_state"], strict=False)
    model.to(device).eval()

    controller = MultiPassController(model)
    t0 = time.perf_counter()
    trace = controller.generate_uta(prompt, max_new_tokens=max_new_tokens)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "checkpoint_id": meta.get("checkpoint_id", pt_path.stem),
        "run_id": meta.get("run_id", ""),
        "step": payload.get("step", 0),
        "prompt": prompt,
        "trace": trace,
        "latency_ms": round(elapsed_ms, 2),
        "metrics": payload.get("latest_metrics", {}),
    }
