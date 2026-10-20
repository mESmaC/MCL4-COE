# Handoff: Pre-Scaled `11.97B` (`12B`) Donor-Seeded MCL4-COE Run & OpenCode Inference UI

**Date:** October 19, 2026  
**Git Branch / Commit:** `main`  
**Hugging Face Backup:** [`YYCore/mcl4-coe-runs`](https://huggingface.co/YYCore/mcl4-coe-runs/tree/main/colab_mcl4_coe_v8_olmoe_12b) & [`mesmac/mcl4-coe-runs`](https://huggingface.co/mesmac/mcl4-coe-runs/tree/main/colab_mcl4_coe_v8_olmoe_12b)

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
- **Files:** [`src/mcl4_coe/donor_olmoe.py`](../src/mcl4_coe/donor_olmoe.py), [`src/mcl4_coe/backbone.py`](../src/mcl4_coe/backbone.py), [`src/mcl4_coe/data.py`](../src/mcl4_coe/data.py), [`src/mcl4_coe/trainer.py`](../src/mcl4_coe/trainer.py), [`configs/colab_mcl4_coe_v8_olmoe_12b.json`](../configs/colab_mcl4_coe_v8_olmoe_12b.json), [`scripts/launch_v8_olmoe_12b.py`](../scripts/launch_v8_olmoe_12b.py)
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
  - **High-Frequency ORNITH-RL & Dream-RSI (`operator_every = 2`, `ornith_every = 2`, `dream_every = 2`, `ornith_weight = 0.25`):** Reached `dream_cycle = 239` (`7` accepted self-revisions, `dream_replay_score = 0.7021`).
- **Held-Out Evaluation Trajectory (`steps 2,800 -> 3,200`):**

| Metric | Pre-RL (`step 2,800`, `top_k=8`) | `step 2,900` (`top_k=12`) | `step 3,000` (`top_k=12`) | `step 3,100` (`top_k=16`) | **`step 3,200` (`top_k=16`)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Held-Out `val_mcqa_acc`** | `38.28%` | `40.62%` | `45.31%` | **`47.66%` (`+9.38%`)** | **`46.88%` (`+8.60%`)** |
| **Held-Out `val_mcqa_ce_loss`** | `1.2952` | `1.2913` | `1.2642` | `1.1834` | **`1.1777` (best)** |
| **Held-Out `val_bpb` (Overall)** | `1.8187` | `1.8334` | `1.8259` | `1.8142` | **`1.8093` (best)** |
| **`val_bpb_pretrain_lm`** | `0.7938` | `0.8058` | `0.7978` | `0.7869` | **`0.7820` (best)** |
| **`val_bpb_typed_decision`** | `1.9726` | `1.9818` | `1.9756` | `1.9639` | **`1.9634` (best)** |
| **`val_bpb_uta_trace`** | `5.2913` | `5.3087` | `5.2952` | `5.2807` | **`5.2727` (best)** |
| **ORNITH Pass Rate / Reward** | `31.6%` / `0.2608` | `99.6%` / `0.6868` | `99.6%` / `0.6526` | `99.7%` / `0.6979` | **`99.7%` / `0.7291` (best)** |
| **Expert Load Std (`112` experts)** | `0.00204` (`min 0.0044`) | `0.00110` | `0.00105` | `0.00108` | **`0.00104` (`min 0.0071`)** |

---

## 2. Remote Synchronization Status

- **GitHub (`origin/main`):** All source code, configs, tests, and full Stage-1 + Stage-2 run telemetry (`runs/colab_mcl4_coe_v8_olmoe_12b/run_manifest.json`, `metrics.jsonl`, `samples.jsonl`, `dream/dream_state.json`, `checkpoints/step_00250..03200.meta.json`) are committed and pushed.
- **Hugging Face Hub (`YYCore/mcl4-coe-runs` & `mesmac/mcl4-coe-runs`):**
  - **[`YYCore/mcl4-coe-runs`](https://huggingface.co/YYCore/mcl4-coe-runs/tree/main/colab_mcl4_coe_v8_olmoe_12b):** `colab_mcl4_coe_v8_olmoe_12b/checkpoints/step_03000.pt` (`23.94 GB`), `step_03100.pt` (`23.94 GB` — `47.66%` MCQA peak), and `step_03200.pt` (`23.94 GB` — `46.88%` MCQA, `1.1777` MCQA CE loss, `1.8093` `val_bpb` best), plus all `.meta.json`, `metrics.jsonl`, `samples.jsonl`, `run_manifest.json`, and `dream/dream_state.json`.
  - **[`mesmac/mcl4-coe-runs`](https://huggingface.co/mesmac/mcl4-coe-runs/tree/main/colab_mcl4_coe_v8_olmoe_12b):** Pruned `131` superseded intermediate `.pt` files (`580.65 GB` freed) and backed up `colab_mcl4_coe_v8_olmoe_12b/checkpoints/step_02800.pt` (pre-RL baseline), `step_03100.pt` (`23.94 GB`), and `step_03200.pt` (`23.94 GB`), plus all `.meta.json` and run telemetry.


---

## 3. External Benchmark Evaluation (`MMLU`, `ARC-Challenge`, `ARC-Easy`)

Evaluated `colab_mcl4_coe_v8_olmoe_12b` (`step_03200.pt`, `11.97B` parameters, `E = 112` experts/layer across `16` layers) against the donor base model `allenai/OLMoE-1B-7B-0125` (`1.3B` active / `6.92B` total parameters, `E = 64`):

| Benchmark | `OLMoE-1B-7B-0125` (`6.92B`, `E=64`) | `v8-OLMoE-12B` (`11.97B`, `E=112`, Uncalibrated `top_k=8`) | `v8-OLMoE-12B` (`11.97B`, `E=112`, Lineage-Calibrated MC) | `v8-OLMoE-12B` (`11.97B`, `E=112`, Lineage-Calibrated + OLMES Hybrid) | $\Delta$ vs Base Model |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MMLU** *(5-shot, 57 subjects)* | `56.30%` | — | **`56.84%`** | **`56.84%`** | **`+0.54%`** |
| **ARC-Challenge** *(5-shot, full `N=1,172`)* | `67.50%` | `49.60%` | **`67.32%`** | **`69.03%`** *(`68.26%` @ $\lambda=0.15$)* | **`+1.53%`** *(`-0.18%` pure MC)* |
| **ARC-Easy** *(5-shot, full `N=2,376`)* | `84.40%` | — | **`84.09%`** | **`86.07%`** *(`85.19%` @ $\lambda=0.15$)* | **`+1.67%`** *(`-0.31%` pure MC)* |

### Root-Cause Discovery & Fix: Parent-Anchored Lineage-Calibrated Family Routing (`SparseMoE.configure_lineage_family_routing`)
- After three mitotic growth waves (`E: 64 -> 80 -> 96 -> 112`), each layer's `64` root donor experts split unevenly (`24` unsplit size-1 families, `36` once-split size-2 families, `4` twice-split size-4 families).
- With `norm_topk_prob = False` and un-tied mitotic pairs (`release_mitotic_pairs = True`), the `112`-way softmax denominator $Z_{112} \approx 1.85 \, Z_{64}$ cuts the routing weight of the `24` unsplit specialist experts in half (`~0.54x`) while correlated siblings from size-4 and size-2 families crowd out distinct root families in `top_k = 8` (`49.60%` on `ARC-Challenge`).
- **Fix (`MCL4CEDBackbone.set_lineage_family_routing(num_roots=64, top_families=8, child_share=0.25)` in [`src/mcl4_coe/backbone.py`](../src/mcl4_coe/backbone.py)):**
  - Traces each mitotic child (`e >= 64`) to its root donor expert (`r in [0..63]`) via `up_proj.weight` cosine similarity, applies a parent-anchored family logit prior $\log w_e$ so $\sum_{e \in \mathcal{F}_r} w_e = 1$ ($Z_{112} \equiv Z_{64}$), and selects the top `top_families = 8` distinct root families per token (`~18` active experts/layer), restoring **`67.32%` MC / `69.03%` Hybrid** on `ARC-Challenge`, **`84.09%` MC / `86.07%` Hybrid** on `ARC-Easy`, and **`56.84%`** on `MMLU`.
