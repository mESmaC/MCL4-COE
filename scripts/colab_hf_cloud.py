"""Cloud Pipeline for `MCL4-COE` on Google Colab + Hugging Face Pro + Google Drive.

Provides four end-to-end cloud workflows (callable from CLI or directly from Colab cells):
1. `build-cloud-shards`: Streams curated Hugging Face datasets directly into block-aligned
   `.bin` shards + `index_receipt.json` + `decision_examples_cache.pt` across all 3 tracks
   (`pretrain_lm`, `uta_trace`, `typed_decision`), with optional automatic push to a private
   Hugging Face Pro dataset repo (`--hf-dataset-repo`) and/or Google Drive (`--drive-root`).
2. `pull-shards`: Stages pre-built `.bin` shards from a Hugging Face dataset repo or Google
   Drive onto Colab's local NVMe SSD (`/content/data_shards`) for zero-bottleneck `mmap` I/O.
3. `train`: Runs `MCL4COETrainer` on Colab GPU (`A100` / `L4` / `T4`) with automatic checkpoint
   and telemetry syncing (`step_XXXXX.pt`, `step_XXXXX.meta.json`, `metrics.jsonl`, `dream_out/`)
   to Hugging Face Hub (`--hf-run-repo`) and/or Google Drive (`--drive-root`).
4. `ablate`: Executes the 4-Arm Context Hierarchy & Backbone Ablation Suite (`Arm A` Standard
   Attention vs. `Arm B` `MCL4-CED` vs. `Arm C` `+TurboQuant/Context-LM` vs. `Arm D` `+A.TXY`
   `COEAddressRouter`) and writes a publication-ready JSON + Markdown report.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mcl4_coe import (
    ATXYPageStore,
    BlockAlignedShardWriter,
    COEAddressRouter,
    COEConfig,
    CPUOffloadAdamW,
    CognitiveOperatorEnsemble,
    ContextLMCompressor,
    DecisionExample,
    MCL4CEDBackbone,
    MCL4CEDConfig,
    MCL4COETrainer,
    MultiTrackMixtureDataset,
    TrainingRunConfig,
    compress_encoded_state,
    ensure_sharded_tracks_ready,
    extract_pretrain_text,
    extract_typed_decision_uta,
    extract_uta_episodes,
    index_python_repo_to_atxy,
    list_checkpoints,
    parse_mcqa_decision_fields,
    prepare_model,
)
from mcl4_coe.curation import sanitize_trace_markers

# Default "Nemotron-Core" 3-Track Hugging Face streaming recipe (`gated: false`)
DEFAULT_HF_TRACK_SOURCES: dict[str, list[dict[str, Any]]] = {
    "pretrain_lm": [
        {"path": "nvidia/Nemotron-ClimbMix", "name": None, "split": "train"},
        {
            "path": "nvidia/Nemotron-Pretraining-Specialized-v1.1",
            "name": "Nemotron-Pretraining-Code-Concepts",
            "split": "train",
        },
        {
            "path": "nvidia/Nemotron-Pretraining-Specialized-v1.1",
            "name": "Nemotron-Pretraining-Formal-Logic",
            "split": "train",
        },
        {
            "path": "nvidia/Nemotron-Pretraining-Specialized-v1.1",
            "name": "Nemotron-Pretraining-Unconditional-Algorithmic",
            "split": "train",
        },
        {
            "path": "nvidia/Nemotron-Pretraining-Dataset-sample",
            "name": "Nemotron-CC-High-Quality-Synthetic",
            "split": "train",
        },
        {
            "path": "nvidia/Nemotron-Pretraining-Dataset-sample",
            "name": "Nemotron-CC-MATH",
            "split": "train",
        },
        {"path": "nvidia/Nemotron-MIND", "name": None, "split": "train"},
    ],
    "uta_trace": [
        {"path": "nvidia/Nemotron-Cascade-2-SFT-Data", "name": "math", "split": "train"},
        {"path": "nvidia/Nemotron-Cascade-2-SFT-Data", "name": "science", "split": "train"},
        {"path": "nvidia/Nemotron-Cascade-2-SFT-Data", "name": "instruction_following", "split": "train"},
        {"path": "nvidia/Nemotron-Cascade-2-SFT-Data", "name": "chat", "split": "train"},
        {"path": "nvidia/Nemotron-Post-Training-Dataset-v1", "name": "default", "split": "math"},
        {"path": "nvidia/Nemotron-Post-Training-Dataset-v1", "name": "default", "split": "stem"},
        {"path": "nvidia/Nemotron-Pretraining-Dataset-sample", "name": "Nemotron-SFT-Code", "split": "train"},
    ],
    "typed_decision": [
        {"path": "nvidia/Nemotron-RL-knowledge-mcqa", "name": "default", "split": "train"},
        {"path": "nvidia/Nemotron-CrossThink", "name": "default", "split": "train_qa"},
        {
            "path": "nvidia/Nemotron-Pretraining-Specialized-v1.1",
            "name": "Nemotron-Pretraining-Multiple-Choice",
            "split": "train",
        },
    ],
}


def resolve_hf_token(explicit_token: str | None = None) -> str | None:
    """Resolve `HF_TOKEN` from argument, environment, or Google Colab Secrets (`userdata`)."""
    if explicit_token:
        return explicit_token.strip()
    for env_key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        val = os.environ.get(env_key)
        if val and val.strip():
            return val.strip()
    try:
        from google.colab import userdata  # type: ignore[import-not-found]

        secret = userdata.get("HF_TOKEN")
        if secret and str(secret).strip():
            os.environ["HF_TOKEN"] = str(secret).strip()
            return str(secret).strip()
    except Exception:
        pass
    return None


def _convert_record_to_decision_example(
    record: dict[str, Any],
    idx: int,
    max_seq_len: int,
    max_options: int = 10,
) -> DecisionExample | None:
    """Convert a raw HF or Nemotron MCQA record into a bounded `DecisionExample` for `MCL4-CED`."""
    parsed = parse_mcqa_decision_fields(record)
    if parsed is None:
        return None
    state, kind, question, clean_opts, best_idx, rationale = parsed
    if len(clean_opts) < 2 or len(clean_opts) > max_options:
        return None

    budget = max(64, max_seq_len - 16)
    ctx_raw = f"#U$# {sanitize_trace_markers(state)}\n#Q$# [{kind}] {sanitize_trace_markers(question)}"
    ctx_str = ctx_raw.encode("utf-8")[:budget].decode("utf-8", errors="ignore")

    descs: list[str] = []
    for opt_i, opt_text in enumerate(clean_opts):
        if rationale and opt_i == best_idx:
            t_line = sanitize_trace_markers(rationale.splitlines()[0][:140])
            d_raw = f"#T$# {t_line}\n#A$# {opt_text}"
        else:
            d_raw = f"#T$# Evaluate option [{opt_i}] against state.\n#A$# {opt_text}"
        d_str = d_raw.encode("utf-8")[:budget].decode("utf-8", errors="ignore")
        descs.append(d_str)

    return DecisionExample(
        question_id=f"hf_dec_{idx:06d}",
        kind="binary" if len(clean_opts) == 2 else "choice",
        context=ctx_str,
        descriptions=tuple(descs),
        values=tuple(f"opt_{i}" for i in range(len(clean_opts))),
        target=best_idx,
    )


def build_cloud_shards(
    shards_root: str | Path,
    *,
    seq_len: int = 2048,
    max_blocks_per_shard: int = 2048,
    records_per_source: int = 4000,
    hf_token: str | None = None,
    hf_dataset_repo: str | None = None,
    drive_root: str | Path | None = None,
) -> dict[str, Any]:
    """Stream datasets from Hugging Face Hub directly into `.bin` shards + `index_receipt.json`."""
    token = resolve_hf_token(hf_token)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    out_root = Path(shards_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError:
        load_dataset = None

    summary: dict[str, Any] = {"shards_root": str(out_root), "tracks": {}}
    decision_pool: list[DecisionExample] = []

    if load_dataset is not None:
        for track, sources in DEFAULT_HF_TRACK_SOURCES.items():
            t_dir = out_root / track
            t_dir.mkdir(parents=True, exist_ok=True)
            writer = BlockAlignedShardWriter(
                t_dir,
                track=track,
                split="train",
                seq_len=seq_len,
                max_blocks_per_shard=max_blocks_per_shard,
            )
            track_accepted = 0
            track_rejected = 0

            for src in sources:
                ds_path = src["path"]
                ds_name = src.get("name")
                ds_split = src.get("split", "train")
                print(f"[build-cloud-shards] Streaming {ds_path} ({ds_name or 'default'}) -> {track} ...")
                try:
                    stream = load_dataset(
                        ds_path,
                        name=ds_name,
                        split=ds_split,
                        streaming=True,
                        token=token,
                    )
                except Exception as exc:
                    print(f"  [skip] {ds_path}: {exc}")
                    continue

                accepted = 0
                rejected = 0
                row_idx = 0
                for row in stream:
                    row_idx += 1
                    if row_idx > records_per_source:
                        break
                    if not isinstance(row, dict):
                        continue

                    if track == "pretrain_lm":
                        text, rej = extract_pretrain_text(row)
                        if rej or not text:
                            rejected += 1
                        else:
                            writer.add_pretrain_document(text)
                            accepted += 1
                    elif track == "uta_trace":
                        eps = extract_uta_episodes(row)
                        if not eps:
                            rejected += 1
                        else:
                            added_any = False
                            for u_s, t_s, a_s in eps:
                                if writer.add_uta_episode(u_s, t_s, a_s):
                                    added_any = True
                            if added_any:
                                accepted += 1
                            else:
                                rejected += 1
                    else:
                        uta = extract_typed_decision_uta(row)
                        dec_ex = _convert_record_to_decision_example(row, len(decision_pool), max_seq_len=min(512, seq_len))
                        if dec_ex is not None and len(decision_pool) < 4096:
                            decision_pool.append(dec_ex)
                        if uta is not None and writer.add_uta_episode(*uta):
                            accepted += 1
                        else:
                            # Fallback if a tasksource instruct record has inputs/targets
                            eps = extract_uta_episodes(row)
                            if eps and writer.add_uta_episode(*eps[0]):
                                accepted += 1
                            else:
                                rejected += 1

                if accepted > 0 or rejected > 0:
                    writer.record_provenance(
                        corpus=ds_path,
                        source_file=f"hf://{ds_path}/{ds_name or ds_split}",
                        row_start=1,
                        row_end=row_idx,
                        episodes_accepted=accepted,
                        episodes_rejected_stage0=rejected,
                    )
                    track_accepted += accepted
                    track_rejected += rejected
                    print(f"  -> accepted={accepted:,}, rejected_stage0={rejected:,}")

            writer.flush_shard()
            summary["tracks"][track] = {
                "accepted_episodes": track_accepted,
                "rejected_stage0": track_rejected,
                "num_blocks": writer.index_receipt.totals.get("num_blocks", 0),
                "num_tokens": writer.index_receipt.totals.get("num_tokens", 0),
                "num_shards": writer.index_receipt.totals.get("num_shards", 0),
            }

    if decision_pool:
        dec_cache = out_root / "typed_decision" / "decision_examples_cache.pt"
        dec_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(decision_pool, dec_cache)
        summary["decision_examples_cached"] = len(decision_pool)

    # Guarantee every track has at least 1 valid .bin shard (portable self-healing fallback)
    healed = ensure_sharded_tracks_ready(out_root, seq_len=seq_len)
    for t_name, idx_path in healed.items():
        raw_idx = json.loads(idx_path.read_text(encoding="utf-8"))
        summary["tracks"][t_name] = raw_idx.get("totals", {})

    if drive_root:
        drive_shards = Path(drive_root).expanduser().resolve() / "data_shards"
        drive_shards.mkdir(parents=True, exist_ok=True)
        shutil.copytree(out_root, drive_shards, dirs_exist_ok=True)
        summary["drive_mirror"] = str(drive_shards)

    if hf_dataset_repo:
        push_folder_to_hf(
            out_root,
            repo_id=hf_dataset_repo,
            repo_type="dataset",
            hf_token=token,
            commit_message=f"Update MCL4-COE sharded tracks (seq_len={seq_len})",
        )
        summary["hf_dataset_repo"] = hf_dataset_repo

    return summary


def push_folder_to_hf(
    local_dir: str | Path,
    *,
    repo_id: str,
    repo_type: str = "dataset",
    path_in_repo: str | None = None,
    hf_token: str | None = None,
    commit_message: str = "Sync MCL4-COE artifacts",
) -> str:
    """Create (if needed) and upload `local_dir` to a private Hugging Face repo."""
    from huggingface_hub import HfApi  # type: ignore[import-not-found]

    token = resolve_hf_token(hf_token)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=True, exist_ok=True)
    info = api.upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=str(Path(local_dir).expanduser().resolve()),
        path_in_repo=path_in_repo,
        commit_message=commit_message,
    )
    return str(info)


def pull_shards_to_local_nvme(
    local_shards_root: str | Path = "/content/data_shards",
    *,
    hf_dataset_repo: str | None = None,
    drive_root: str | Path | None = None,
    hf_token: str | None = None,
    seq_len: int = 2048,
) -> dict[str, Path]:
    """Stage `.bin` shards onto local NVMe from HF Hub or Google Drive (or heal portably)."""
    dst = Path(local_shards_root).expanduser().resolve()
    dst.mkdir(parents=True, exist_ok=True)
    token = resolve_hf_token(hf_token)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    if hf_dataset_repo:
        from huggingface_hub import snapshot_download  # type: ignore[import-not-found]

        print(f"[pull-shards] Downloading {hf_dataset_repo} -> {dst} via HF Hub ...")
        snapshot_download(
            repo_id=hf_dataset_repo,
            repo_type="dataset",
            local_dir=str(dst),
            token=token,
        )
    elif drive_root:
        src = Path(drive_root).expanduser().resolve() / "data_shards"
        if src.is_dir():
            print(f"[pull-shards] Staging {src} -> {dst} onto local NVMe ...")
            shutil.copytree(src, dst, dirs_exist_ok=True)

    return ensure_sharded_tracks_ready(dst, seq_len=seq_len)


def run_cloud_training(
    config: TrainingRunConfig,
    *,
    hf_run_repo: str | None = None,
    drive_root: str | Path | None = None,
    hf_token: str | None = None,
    resume_from_cloud: bool = True,
) -> dict[str, Any]:
    """Run `MCL4COETrainer` and automatically sync checkpoints + telemetry to HF Hub and/or Drive."""
    token = resolve_hf_token(hf_token)
    runs_root = Path(config.runs_root).expanduser().resolve()
    run_dir = runs_root / config.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Optionally restore prior run checkpoints from Drive or HF Hub before starting
    if resume_from_cloud:
        if drive_root:
            drive_run = Path(drive_root).expanduser().resolve() / "runs" / config.run_id
            if drive_run.is_dir():
                shutil.copytree(drive_run, run_dir, dirs_exist_ok=True)
        if hf_run_repo:
            try:
                from huggingface_hub import snapshot_download  # type: ignore[import-not-found]

                snapshot_download(
                    repo_id=hf_run_repo,
                    repo_type="model",
                    allow_patterns=[f"{config.run_id}/**"],
                    local_dir=str(runs_root),
                    token=token,
                )
            except Exception:
                pass
        existing_ckpts = list_checkpoints(run_dir)
        if existing_ckpts and not config.resume_checkpoint:
            latest_meta = existing_ckpts[-1]
            ckpt_pt = Path(str(latest_meta["checkpoint_path"]))
            if not ckpt_pt.is_file():
                ckpt_pt = run_dir / ckpt_pt.name
            if ckpt_pt.is_file():
                config.resume_checkpoint = str(ckpt_pt)
                print(f"[cloud-train] Resuming from checkpoint: {ckpt_pt.name} (step={latest_meta['step']})")

    last_synced_ckpt_count = len(list_checkpoints(run_dir))

    def _sync_run_artifacts(step: int) -> None:
        if drive_root:
            drive_run = Path(drive_root).expanduser().resolve() / "runs" / config.run_id
            drive_run.mkdir(parents=True, exist_ok=True)
            shutil.copytree(run_dir, drive_run, dirs_exist_ok=True)
        if hf_run_repo:
            try:
                push_folder_to_hf(
                    run_dir,
                    repo_id=hf_run_repo,
                    repo_type="model",
                    path_in_repo=config.run_id,
                    hf_token=token,
                    commit_message=f"Checkpoint sync for {config.run_id} at step {step}",
                )
            except Exception as exc:
                print(f"[cloud-train] Warning: HF upload skipped at step {step}: {exc}")

    def _on_step(metrics: dict[str, Any]) -> None:
        nonlocal last_synced_ckpt_count
        step = int(metrics.get("step", 0))
        loss = float(metrics.get("total_loss", 0.0))
        lm = float(metrics.get("lm_loss", 0.0))
        tps = float(metrics.get("tokens_per_sec", 0.0))
        if step == 1 or step % 5 == 0 or step == config.total_steps:
            print(
                f"  [step {step:>4}/{config.total_steps}] "
                f"total_loss={loss:.4f} | lm_loss={lm:.4f} | tok/s={tps:,.0f}"
            )
        cur_ckpts = len(list_checkpoints(run_dir))
        if cur_ckpts > last_synced_ckpt_count:
            last_synced_ckpt_count = cur_ckpts
            _sync_run_artifacts(step)

    trainer = MCL4COETrainer(config, on_step_callback=_on_step)
    manifest = trainer.run()
    _sync_run_artifacts(int(manifest.get("current_step", config.total_steps)))
    return manifest


def run_context_hierarchy_ablation(
    output_dir: str | Path,
    *,
    d_model: int = 192,
    n_heads: int = 6,
    steps: int = 24,
    seq_len: int = 256,
) -> dict[str, Any]:
    """Execute the 4-Arm Context Hierarchy & Backbone Ablation Suite on GPU/CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_path = Path(output_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    shards_root = REPO_ROOT / "data_shards"
    track_receipts = ensure_sharded_tracks_ready(shards_root, seq_len=seq_len)
    ds = MultiTrackMixtureDataset(track_receipts)

    arms = [
        ("Arm_A_Standard_Attention", "standard", False, False),
        ("Arm_B_MCL4_CED", "markov_local4", False, False),
        ("Arm_C_MCL4_TurboQuant_ContextLM", "markov_local4", True, False),
        ("Arm_D_MCL4_Full_4Tier_ATXY", "markov_local4", True, True),
    ]

    store: ATXYPageStore = index_python_repo_to_atxy(REPO_ROOT, subsystems=("src",))
    results: dict[str, Any] = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "d_model": d_model,
        "steps": steps,
        "seq_len": seq_len,
        "atxy_records_indexed": len(store),
        "arms": {},
    }

    for arm_name, attn_type, use_compression, use_atxy in arms:
        torch.manual_seed(42)
        cfg = MCL4CEDConfig(
            d_model=d_model,
            n_heads=n_heads,
            n_encoder_layers=2,
            n_decoder_layers=2,
            ple_dim=48,
            expert_hidden_dim=d_model * 2,
            num_experts=4,
            top_k=2,
            max_seq_len=seq_len,
            attention_type=attn_type,
        )
        model = CognitiveOperatorEnsemble(
            MCL4CEDBackbone(cfg),
            COEConfig(projection_dim=48, initial_scale=15.0, recurrence_steps=1),
        )
        prepare_model(model, device=device, bf16_working_weights=(device.type == "cuda"))
        opt = CPUOffloadAdamW(model.parameters(), lr=2.5e-3, weight_decay=0.01, transfer_mode="batched")

        ctx_compressor = (
            ContextLMCompressor(d_model=d_model, chunk_size=64, gist_tokens_per_chunk=8).to(device)
            if use_compression
            else None
        )

        t0 = time.perf_counter()
        first_loss = 0.0
        final_loss = 0.0
        kv_ratio = 1.0

        for step_idx in range(1, steps + 1):
            batch_items = [ds[(step_idx * 7 + b) % len(ds)] for b in range(4)]
            input_ids = torch.stack([b["input_ids"][:seq_len] for b in batch_items]).to(device)
            targets = torch.stack([b["labels"][:seq_len] for b in batch_items]).to(device)

            opt.zero_grad()
            lm_out = model.forward_inference(input_ids, targets=targets)
            loss = lm_out.loss if lm_out.loss is not None else lm_out.aux_loss

            if ctx_compressor is not None:
                valid = input_ids != cfg.pad_id
                gist_state = ctx_compressor(model.backbone, input_ids, valid)
                comp_kv = compress_encoded_state(gist_state)
                kv_ratio = (seq_len / max(1, gist_state.hidden.shape[1])) * comp_kv.compression_ratio
                loss = loss + 0.05 * gist_state.aux_loss

            loss.backward()
            opt.step()

            lv = float(loss.detach().item())
            if step_idx == 1:
                first_loss = lv
            final_loss = lv

        elapsed = time.perf_counter() - t0
        atxy_route_conf = 0.0
        atxy_selected = None
        if use_atxy:
            router = COEAddressRouter(store, max_seq_len=min(128, seq_len))
            r_out = router.route_query(
                model,
                "Locate TurboQuant compression and A.TXY virtual memory page store in mcl4_coe",
                top_k_candidates=6,
            )
            atxy_route_conf = float(r_out["confidence"])
            atxy_selected = str(r_out["address"])

        results["arms"][arm_name] = {
            "attention_type": attn_type,
            "first_loss": round(first_loss, 4),
            "final_loss": round(final_loss, 4),
            "effective_context_compression_ratio": round(kv_ratio, 2),
            "atxy_routed_address": atxy_selected,
            "atxy_route_confidence": round(atxy_route_conf, 4),
            "wall_time_sec": round(elapsed, 2),
        }
        print(
            f"[{arm_name}] loss: {first_loss:.4f} -> {final_loss:.4f} | "
            f"compression={kv_ratio:.1f}x | atxy={atxy_selected} ({elapsed:.2f}s)"
        )

    json_file = out_path / "context_hierarchy_ablation.json"
    json_file.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="MCL4-COE Cloud Pipeline (Colab + HF Pro + Drive)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build-cloud-shards", help="Stream HF corpora into .bin shards")
    p_build.add_argument("--shards-root", default="/content/data_shards")
    p_build.add_argument("--seq-len", type=int, default=2048)
    p_build.add_argument("--max-blocks-per-shard", type=int, default=2048)
    p_build.add_argument("--records-per-source", type=int, default=4000)
    p_build.add_argument("--hf-dataset-repo", default=None)
    p_build.add_argument("--drive-root", default=None)

    p_pull = sub.add_parser("pull-shards", help="Stage .bin shards onto local NVMe")
    p_pull.add_argument("--shards-root", default="/content/data_shards")
    p_pull.add_argument("--hf-dataset-repo", default=None)
    p_pull.add_argument("--drive-root", default=None)
    p_pull.add_argument("--seq-len", type=int, default=2048)

    p_train = sub.add_parser("train", help="Run MCL4COETrainer with cloud checkpoint sync")
    p_train.add_argument("--run-id", default="colab_mcl4_coe_v1")
    p_train.add_argument("--shards-root", default="/content/data_shards")
    p_train.add_argument("--runs-root", default="/content/runs")
    p_train.add_argument("--total-steps", type=int, default=200)
    p_train.add_argument("--checkpoint-every", type=int, default=50)
    p_train.add_argument("--d-model", type=int, default=384)
    p_train.add_argument("--n-heads", type=int, default=6)
    p_train.add_argument("--n-encoder-layers", type=int, default=4)
    p_train.add_argument("--n-decoder-layers", type=int, default=4)
    p_train.add_argument("--num-experts", type=int, default=8)
    p_train.add_argument("--batch-size", type=int, default=8)
    p_train.add_argument("--micro-batch-size", type=int, default=4)
    p_train.add_argument("--hf-run-repo", default=None)
    p_train.add_argument("--drive-root", default=None)

    p_ablate = sub.add_parser("ablate", help="Run 4-Arm Context Hierarchy & Backbone Ablation")
    p_ablate.add_argument("--output-dir", default="runs/ablations")
    p_ablate.add_argument("--d-model", type=int, default=192)
    p_ablate.add_argument("--n-heads", type=int, default=6)
    p_ablate.add_argument("--steps", type=int, default=24)
    p_ablate.add_argument("--seq-len", type=int, default=256)

    args = parser.parse_args()
    if args.cmd == "build-cloud-shards":
        res = build_cloud_shards(
            args.shards_root,
            seq_len=args.seq_len,
            max_blocks_per_shard=args.max_blocks_per_shard,
            records_per_source=args.records_per_source,
            hf_dataset_repo=args.hf_dataset_repo,
            drive_root=args.drive_root,
        )
        print(json.dumps(res, indent=2))
    elif args.cmd == "pull-shards":
        receipts = pull_shards_to_local_nvme(
            args.shards_root,
            hf_dataset_repo=args.hf_dataset_repo,
            drive_root=args.drive_root,
            seq_len=args.seq_len,
        )
        print(json.dumps({k: str(v) for k, v in receipts.items()}, indent=2))
    elif args.cmd == "train":
        cfg = TrainingRunConfig(
            run_id=args.run_id,
            shards_root=args.shards_root,
            runs_root=args.runs_root,
            total_steps=args.total_steps,
            checkpoint_every=args.checkpoint_every,
            sample_every=max(10, args.checkpoint_every // 2),
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_encoder_layers=args.n_encoder_layers,
            n_decoder_layers=args.n_decoder_layers,
            ple_dim=min(128, args.d_model // 2),
            expert_hidden_dim=args.d_model * 2,
            num_experts=args.num_experts,
            top_k=min(2, args.num_experts),
            batch_size=args.batch_size,
            micro_batch_size=args.micro_batch_size,
            enable_ornith_rl=True,
            ornith_every=10,
            enable_dream_rsi=True,
            dream_every=25,
        )
        manifest = run_cloud_training(
            cfg,
            hf_run_repo=args.hf_run_repo,
            drive_root=args.drive_root,
        )
        print(json.dumps(manifest, indent=2))
    elif args.cmd == "ablate":
        res = run_context_hierarchy_ablation(
            args.output_dir,
            d_model=args.d_model,
            n_heads=args.n_heads,
            steps=args.steps,
            seq_len=args.seq_len,
        )
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
