"""Launch script and preset builder for MCL4-COE v7-Mitotic-3B (`16 -> 206` experts = `320.5M -> 3.01B`).

Architecture highlights:
- Option B Fine-Grained Specialist aspect ratio: `expert_hidden_dim = 0.5 * d_model` with `top_k = 4`
  active experts per token (`active FFN width = 2 * d_model`), giving 4 independent specialist axes
  per token and smooth multi-active density scaling (`dense_subset_k` from `k=4` up to full `3.01B`).
- Preset `balanced_768` (default):
  * `d_model = 768`, `n_heads = 12`, `n_encoder_layers = 8`, `n_decoder_layers = 8` (`16` MoE layers)
  * Single expert per layer: `3 * 768 * 384 = 884,736` (`~885K`) parameters (`14.16M` per 16-layer column)
  * Starts at `E = 16` (`320.5M` total, `150.6M` active) -> grows via mitosis to `E = 206` (`3.010B` total)
- Preset `ultra_fine_512`:
  * `d_model = 512`, `n_heads = 8`, `n_encoder_layers = 6`, `n_decoder_layers = 6` (`12` MoE layers)
  * Single expert per layer: `3 * 512 * 256 = 393,216` (`~393K`) parameters (`4.72M` per 12-layer column)
  * Starts at `E = 16` (`112.1M` total, `55.5M` active) -> grows via mitosis to `E = 628` (`3.000B` total)
- Shard-windowed token streaming + consumed-shard eviction (`clear_consumed_shards = True`):
  * Shuffles blocks within each shard sequentially (`Shard 0 -> Shard 1 -> ...`) and unlinks consumed
    `.bin` files from disk as training progresses through mitosis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from mcl4_coe import MORPHCODE_16K_VOCAB_SIZE
from mcl4_coe.backbone import MCL4CEDConfig
from mcl4_coe.operators import COEConfig
from mcl4_coe.trainer import MCL4COETrainer, TrainingRunConfig


V7_PRESETS: dict[str, dict[str, Any]] = {
    "balanced_768": {
        "d_model": 768,
        "n_heads": 12,
        "n_encoder_layers": 8,
        "n_decoder_layers": 8,
        "ple_dim": 192,
        "expert_hidden_dim": 384,  # 0.5 * d_model -> 884,736 params/expert/layer (14.16M across 16 layers)
        "num_experts": 16,         # 320.5M starting total params (150.6M active at top_k=4)
        "max_experts": 206,        # 3,010,275,536 (3.01B) terminal total params
        "top_k": 4,
        "mitosis_step_size": 10,
        "projection_dim": 192,
    },
    "ultra_fine_512": {
        "d_model": 512,
        "n_heads": 8,
        "n_encoder_layers": 6,
        "n_decoder_layers": 6,
        "ple_dim": 128,
        "expert_hidden_dim": 256,  # 0.5 * d_model -> 393,216 params/expert/layer (4.72M across 12 layers)
        "num_experts": 16,         # 112.1M starting total params (55.5M active at top_k=4)
        "max_experts": 628,        # 3,000,159,280 (3.00B) terminal total params
        "top_k": 4,
        "mitosis_step_size": 32,
        "projection_dim": 128,
    },
}


def compute_v7_parameter_profile(preset_name: str = "balanced_768") -> dict[str, Any]:
    """Compute exact parameter counts across mitotic stages and multi-active densities without allocating tensors."""
    if preset_name not in V7_PRESETS:
        raise ValueError(f"Unknown preset {preset_name!r}; choose from {sorted(V7_PRESETS)}")
    p = V7_PRESETS[preset_name]
    V = MORPHCODE_16K_VOCAB_SIZE
    d = int(p["d_model"])
    L_enc = int(p["n_encoder_layers"])
    L_dec = int(p["n_decoder_layers"])
    L = L_enc + L_dec
    d_ple = int(p["ple_dim"])
    d_ffn = int(p["expert_hidden_dim"])
    k_sparse = int(p["top_k"])
    e_start = int(p["num_experts"])
    e_max = int(p["max_experts"])
    p_dim = int(p["projection_dim"])

    tok_emb = V * d
    ple_per_layer = (V * d_ple + d_ple * d) + d
    attn_per_layer = 4 * d * d + 2 * (d // int(p["n_heads"])) + d
    router_shared_per_layer = (4 * d) * 32 + 32 + d
    backbone_norms = 2 * d
    coe_heads = (
        1
        + (d * d + d + 2 * (d * p_dim + p_dim))
        + (d * p_dim + p_dim + 4 * p_dim * p_dim + 2 * p_dim * d + d + 2 * d)
        + (2 * d * p_dim + p_dim + p_dim * 1 + 1 + 2 * d)
        + (d * p_dim + p_dim + p_dim * 5 + 5 + 3 * d)
        + (3 * d * p_dim + 3 * p_dim + p_dim * d + d + 2 * d)
    )
    shared_non_expert = (
        tok_emb
        + L * (ple_per_layer + attn_per_layer + router_shared_per_layer)
        + backbone_norms
        + coe_heads
    )
    single_expert_layer = 3 * d * d_fd if (d_fd := d_ffn) else 0
    router_per_expert_layer = 32 + 1
    per_expert_stack = L * (single_expert_layer + router_per_expert_layer)

    def total_at(e: int) -> int:
        return shared_non_expert + e * per_expert_stack

    active_sparse = shared_non_expert + k_sparse * L * single_expert_layer
    density_ladder = {}
    for k_eval in sorted({k_sparse, 8, 16, 32, 64, 128, e_max}):
        if k_eval <= e_max:
            act_k = shared_non_expert + k_eval * L * single_expert_layer
            density_ladder[f"k={k_eval}"] = {
                "active_params": act_k,
                "active_millions": round(act_k / 1e6, 2),
                "pct_of_3b": round(100.0 * act_k / total_at(e_max), 2),
            }

    return {
        "preset": preset_name,
        "d_model": d,
        "layers_total": L,
        "expert_hidden_dim": d_ffn,
        "top_k": k_sparse,
        "single_expert_per_layer_params": single_expert_layer,
        "single_expert_full_stack_params": per_expert_stack,
        "shared_backbone_and_coe_params": shared_non_expert,
        "start_experts": e_start,
        "start_total_params": total_at(e_start),
        "max_experts": e_max,
        "terminal_total_params": total_at(e_max),
        "sparse_active_params": active_sparse,
        "multi_active_density_ladder": density_ladder,
    }


def build_v7_config(
    *,
    preset: str = "balanced_768",
    run_id: str = "colab_mcl4_coe_v7_mitotic_3b",
    runs_root: str = "/content/runs" if Path("/content").is_dir() else r"X:\MCL4-COE\runs",
    shards_root: str = (
        "/content/data_shards_morph16k"
        if Path("/content").is_dir()
        else r"X:\MCL4-COE\data_shards_morph16k"
    ),
    corpora_root: str = (
        "/content/MCL4-COE/corpora" if Path("/content").is_dir() else r"X:\corpora"
    ),
    total_steps: int = 3800,
    mitosis_every: int = 180,
    batch_size: int = 16,
    micro_batch_size: int = 8,
    max_seq_len: int = 1024,
    clear_consumed_shards: bool = True,
    use_nvfp4: bool = False,
    cpu_offload_optimizer: bool = False,
    resume_checkpoint: str | None = None,
) -> TrainingRunConfig:
    """Construct the `TrainingRunConfig` for `MCL4-COE v7-Mitotic-3B`."""
    if preset not in V7_PRESETS:
        raise ValueError(f"Unknown preset {preset!r}; choose from {sorted(V7_PRESETS)}")
    p = V7_PRESETS[preset]
    return TrainingRunConfig(
        run_id=run_id,
        runs_root=runs_root,
        shards_root=shards_root,
        corpora_root=corpora_root,
        resume_checkpoint=resume_checkpoint,
        vocab_size=MORPHCODE_16K_VOCAB_SIZE,
        max_seq_len=max_seq_len,
        d_model=int(p["d_model"]),
        n_heads=int(p["n_heads"]),
        n_encoder_layers=int(p["n_encoder_layers"]),
        n_decoder_layers=int(p["n_decoder_layers"]),
        ple_dim=int(p["ple_dim"]),
        expert_hidden_dim=int(p["expert_hidden_dim"]),
        num_experts=int(p["num_experts"]),
        max_experts=int(p["max_experts"]),
        top_k=int(p["top_k"]),
        projection_dim=int(p["projection_dim"]),
        enable_mitosis=True,
        orthogonal_mitosis=True,
        mitosis_every=mitosis_every,
        mitosis_step_size=int(p["mitosis_step_size"]),
        clear_consumed_shards=clear_consumed_shards,
        mitotic_ortho_coef=0.02,
        sequence_diversity_coef=0.02,
        recurrent_coverage_penalty=0.25,
        recurrence_steps=2,
        recurrent_kv=True,
        depth_ponder=True,
        deep_supervision_weight=0.20,
        enable_gist_conditioning=True,
        gist_conditioning_prob=0.35,
        latent_prior_scale=0.35,
        use_nvfp4=use_nvfp4,
        cpu_offload_optimizer=cpu_offload_optimizer,
        total_steps=total_steps,
        batch_size=batch_size,
        micro_batch_size=micro_batch_size,
        decision_batch_size=4,
        lr=1.0e-3,
        warmup_ratio=0.06,
        min_lr_ratio=0.12,
        curriculum_mode="cosine",
        start_weights={"pretrain_lm": 0.50, "uta_trace": 0.35, "typed_decision": 0.15},
        end_weights={"pretrain_lm": 0.25, "uta_trace": 0.55, "typed_decision": 0.20},
        enable_ornith_rl=True,
        ornith_every=8,
        ornith_weight=0.20,
        ornith_warmup_ratio=0.50,
        enable_dream_rsi=True,
        dream_every=10,
        checkpoint_every=200,
        max_keep_checkpoints=2,
        eval_every=200,
        sample_every=200,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch MCL4-COE v7-Mitotic-3B training")
    parser.add_argument("--preset", default="balanced_768", choices=sorted(V7_PRESETS))
    parser.add_argument("--run-id", default="colab_mcl4_coe_v7_mitotic_3b")
    parser.add_argument(
        "--runs-root",
        default="/content/runs" if Path("/content").is_dir() else r"X:\MCL4-COE\runs",
    )
    parser.add_argument(
        "--shards-root",
        default="/content/data_shards_morph16k"
        if Path("/content").is_dir()
        else r"X:\MCL4-COE\data_shards_morph16k",
    )
    parser.add_argument(
        "--corpora-root",
        default="/content/MCL4-COE/corpora" if Path("/content").is_dir() else r"X:\corpora",
    )
    parser.add_argument("--total-steps", type=int, default=3800)
    parser.add_argument("--mitosis-every", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--hf-run-repo", default="mESmaC/mcl4-coe-runs")
    parser.add_argument("--no-clear-shards", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print exact parameter profile and exit")
    args = parser.parse_args()

    profile = compute_v7_parameter_profile(args.preset)
    print(json.dumps(profile, indent=2), flush=True)
    if args.dry_run:
        return

    cfg = build_v7_config(
        preset=args.preset,
        run_id=args.run_id,
        runs_root=args.runs_root,
        shards_root=args.shards_root,
        corpora_root=args.corpora_root,
        total_steps=args.total_steps,
        mitosis_every=args.mitosis_every,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        max_seq_len=args.max_seq_len,
        clear_consumed_shards=not args.no_clear_shards,
    )
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from colab_hf_cloud import run_cloud_training

    run_cloud_training(
        cfg,
        hf_run_repo=args.hf_run_repo or None,
        drive_root=None,
        resume_from_cloud=True,
    )


if __name__ == "__main__":
    main()
