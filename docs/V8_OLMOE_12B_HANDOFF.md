# Handoff: Pre-Scaled `11.97B` (`12B`) Donor-Seeded MCL4-COE Run & OpenCode Inference UI

**Date:** October 19, 2026  
**Git Branch / Commit:** `main`  
**Hugging Face Backup:** [`mesmac/mcl4-coe-runs` (`colab_mcl4_coe_v8_olmoe_12b`)](https://huggingface.co/mesmac/mcl4-coe-runs/tree/main/colab_mcl4_coe_v8_olmoe_12b)

---

## 1. Completed Today

### A. OpenCode-Inspired Interactive Inference & Testing UI
- **Files:** [`src/mcl4_coe/inference_ui.py`](../src/mcl4_coe/inference_ui.py), [`scripts/launch_inference_ui.py`](../scripts/launch_inference_ui.py), [`tests/test_inference_ui.py`](../tests/test_inference_ui.py)
- Built a zero-external-dependency (`ThreadingHTTPServer`) dark monospace OpenCode-inspired inference frontend for interactive testing of inference-time features:
  - **Checkpoint Loader:** Discovers local checkpoints under `runs/` and remote runs on Hugging Face (`mesmac/mcl4-coe-runs`) with 1-click download and fast `bfloat16` weight loading (`~5s` for `12B`).
  - **Interactive `#U$# -> #T$# -> #A$#` Chat & Execution Modes:** Supports `Direct UTA`, `COE 9-Operator Loop` (multi-candidate generation + self-consistency + verification), and `2-Round Inference-Time Dream-RL`.
  - **Live $O(1)$ Routing & Recurrence Controls:** Adjusts `recurrence_steps`, `adaptive_recurrent_depth`, `top_k`, `dense_subset_k`, `combine_dense_and_sparse`, Tabu `recurrent_coverage_penalty`, and `release_mitotic_pairs` without reloading weights.
  - **MoE Telemetry & Test Panels:** Per-layer expert routing heatmaps, **Quarantined Whiteboard** fast-weight ingestion (`prose`/`code`/`web`) + sleep consolidation, **Windowless Demand-Paged Context** (`InfiniteContextPager`), and a **9-Operator MCQA Decision Probe**.

### B. Donor-Seeded `7.14B -> 11.97B` (`12B`) Architecture & Training (`colab_mcl4_coe_v8_olmoe_12b`)
- **Files:** [`src/mcl4_coe/donor_olmoe.py`](../src/mcl4_coe/donor_olmoe.py), [`src/mcl4_coe/backbone.py`](../src/mcl4_coe/backbone.py), [`src/mcl4_coe/data.py`](../src/mcl4_coe/data.py), [`src/mcl4_coe/trainer.py`](../src/mcl4_coe/trainer.py), [`configs/colab_mcl4_coe_v8_olmoe_12b.json`](../configs/colab_mcl4_coe_v8_olmoe_12b.json), [`scripts/launch_colab_v8_olmoe_12b.py`](../scripts/launch_colab_v8_olmoe_12b.py)
- **Donor Base:** `allenai/OLMoE-1B-7B-0125` (`Apache-2.0`, `6.92B` total / `1.3B` active, `E=64`, `top_k=8`, `d_model=2048`, `16` layers, pretrained on `5.1T` tokens) mapped losslessly into `RecursiveMoEBackbone` (`7,137,926,683` initial params including MCL4-COE heads and recurrent adapters).
- **Training Run (`2,800` steps on NVIDIA RTX PRO 6000 Blackwell `95 GiB` in `125.6 min`):**
  - **Stage 0 (`0–699`):** `E = 64` (`7.14B` params), donor alignment + recurrent `#U$# -> #T$# -> #A$#` warmup (`val_bpb: 2.1748 -> 1.8351`).
  - **Stage 1 (`700–1299`):** Mitosis #1 (`E: 64 -> 80`, `8.75B` params) + Dream-RSI Round 1 (`18/24` verified).
  - **Stage 2 (`1300–1899`):** Mitosis #2 (`E: 80 -> 96`, `10.36B` params) + Dream-RSI Round 2 (`21/24` verified).
  - **Stage 3 (`1900–2800`):** Mitosis #3 (`E: 96 -> 112`, **`11,971,337,755` (`11.97B`) params**) + Dream-RSI Rounds 3–4 (`20/24` verified), finishing at **`val_bpb = 1.8177` (step `2750`) / `1.8187` (step `2800`)** with `44.2 GiB` peak VRAM.

### C. Live `11.97B` Sampling & Inference Stack Fixes
- **Files:** [`src/mcl4_coe/ensemble.py`](../src/mcl4_coe/ensemble.py), [`src/mcl4_coe/operators.py`](../src/mcl4_coe/operators.py), [`src/mcl4_coe/inference_ui.py`](../src/mcl4_coe/inference_ui.py)
- Verified live `11.97B` sampling on `step_02800.pt` across free-form `#U$# -> #T$# -> #A$#` traces, `COE` multi-candidate verification, the 9-Operator Typed-COE MCQA Probe (`100%` on 4-choice probes), and Quarantined Whiteboard fast-weight ingestion (`+3.025` logit shift on targeted fact probes).
- Fixed cross-segment repetition penalty bleeding between `#T$#` and `#A$#`, premature `#T$#`/`#A$#` stop cutoffs on donor-seeded checkpoints, BPE `top_k`/`min_p` sampling, and `bfloat16` SVD + untied `lm_head` scaling in `QuarantinedWhiteboard`.

### D. Stage-2 Post-Mitosis Twin-Release + Memory-Adaptive Active Density RL & Dream-RSI (`steps 2,801 -> 3,200`)
- **Files:** [`src/mcl4_coe/trainer.py`](../src/mcl4_coe/trainer.py), [`scripts/launch_v8_olmoe_12b.py`](../scripts/launch_v8_olmoe_12b.py) (`build_v8_olmoe_12b_stage2_rl_config`, `--stage2-rl`)
- **Configuration (`steps 2,801 -> 3,200` on NVIDIA RTX PRO 6000 Blackwell `95 GiB`):**
  - **Twin Un-Tying (`release_mitotic_pairs = True`):** Explicitly un-tied all `16` mitotic twin pairs (`32` parent/child experts `[17, 96] .. [25, 111]`) and prioritized them inside the `64` plastic (`requires_grad=True`) experts/layer so every split child expert specializes independently.
  - **Memory-Adaptive Active Expert Density (`top_k: 8 -> 12 -> 16`):** Ramped active experts from `top_k = 8` (`~1.71B` active params, steps `2,801–2,899`) $\rightarrow$ `top_k = 12` (`~2.11B` active params, steps `2,900–3,049`) $\rightarrow$ `top_k = 16` (`~2.52B` active params, steps `3,050–3,200`) with `recurrence_steps = 2` (`recurrent_kv = True`, `depth_ponder = True`), sustaining `62.6 GiB / 95.0 GiB` allocated VRAM (`~6.4s/step`).
  - **High-Frequency ORNITH-RL & Dream-RSI (`operator_every = 2`, `ornith_every = 2`, `dream_every = 2`, `ornith_weight = 0.25`):** Executed `101` new Dream-RSI self-play cycles (`dream_cycle: 37 -> 138`, `106,405` replays dreamed, `7` accepted self-revisions, `dream_replay_score: 0.6910 -> 0.7017`).
- **Held-Out Evaluation Trajectory (`steps 2,800 -> 3,200`):**

| Metric | Pre-RL (`step 2,800`, `top_k=8`) | `step 2,900` (`top_k=12`) | `step 3,000` (`top_k=12`) | `step 3,100` (`top_k=16`) | **`step 3,200` (`top_k=16`)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Held-Out `val_mcqa_acc`** | `38.28%` | `39.06%` | `42.19%` | **`46.09%` (`+7.81%`)** | **`43.75%` (`+5.47%`)** |
| **Held-Out `val_mcqa_ce_loss`** | `1.2952` | `1.3019` | `1.2652` | `1.2150` | **`1.2119` (best)** |
| **Held-Out `val_bpb` (Overall)** | `1.8187` | `1.8413` | `1.8268` | `1.8129` | **`1.8100` (best)** |
| **`val_bpb_pretrain_lm`** | `0.7938` | `0.8136` | `0.7999` | `0.7859` | **`0.7836` (best)** |
| **`val_bpb_typed_decision`** | `1.9726` | `1.9851` | `1.9761` | `1.9635` | **`1.9607` (best)** |
| **`val_bpb_uta_trace`** | `5.2913` | `5.3128` | `5.2921` | `5.2778` | **`5.2728` (best)** |
| **ORNITH Pass Rate / Reward** | `31.6%` / `0.2608` | `99.6%` / `0.6868` | `99.7%` / `0.7155` | `99.6%` / `0.6670` | **`99.7%` / `0.7245` (best)** |
| **Expert Load Std (`112` experts)** | `0.00204` (`min 0.0044`) | `0.00112` | `0.00105` | `0.00109` | **`0.00104` (`min 0.0071`)** |

---

## 2. Remote Synchronization Status

- **GitHub (`origin/main`):** All source code, configs, tests, and full Stage-1 + Stage-2 run telemetry (`runs/colab_mcl4_coe_v8_olmoe_12b/run_manifest.json`, `metrics.jsonl`, `samples.jsonl`, `dream/dream_state.json`, `checkpoints/step_00250..03200.meta.json`) are committed and pushed.
- **Hugging Face Hub (`mesmac/mcl4-coe-runs` & `YYCore`):**
  - `colab_mcl4_coe_v8_olmoe_12b/run_manifest.json` (`status: "completed"`, `current_step: 3200`), `metrics.jsonl`, `samples.jsonl`, `dream/dream_state.json`, and `checkpoints/step_00250..03200.meta.json` are synced to `mesmac/mcl4-coe-runs`.
  - Once `YYCore` organization write permissions are checked on the Hugging Face access token (`https://huggingface.co/settings/tokens`) or old intermediate `.pt` files from `v6`/`v7` are pruned on `mesmac/mcl4-coe-runs`, `step_03100.pt` (`46.09%` MCQA peak) and `step_03200.pt` (`1.8100` `val_bpb` best) on `/content/runs/colab_mcl4_coe_v8_olmoe_12b/checkpoints/` can be pushed directly to `YYCore/mcl4-coe-runs`.

