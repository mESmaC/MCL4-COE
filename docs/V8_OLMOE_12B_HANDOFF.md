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

---

## 2. Remote Synchronization Status

- **GitHub (`origin/main`):** All source code, configs, tests, and lightweight run telemetry (`runs/colab_mcl4_coe_v8_olmoe_12b/run_manifest.json`, `metrics.jsonl`, `samples.jsonl`, `checkpoints/step_02800.meta.json`) are committed and pushed.
- **Hugging Face Hub (`mesmac/mcl4-coe-runs`):**
  - `colab_mcl4_coe_v8_olmoe_12b/checkpoints/step_02800.pt` (`22.30 GiB`) + `step_02800.meta.json`
  - `colab_mcl4_coe_v8_olmoe_12b/checkpoints/step_02750.pt` (`22.30 GiB`) + `step_02750.meta.json`
  - `colab_mcl4_coe_v8_olmoe_12b/run_manifest.json`, `metrics.jsonl`, `samples.jsonl`, `dream/dream_state.json`

---

## 3. Priority Note for Next Session: Un-Tying Twin Weights & Memory-Adaptive Active Experts

During today's `7.14B -> 11.97B` mitotic expansion (`E: 64 -> 80 -> 96 -> 112`), the final mitosis stage kept the 16 newest parent/child sibling pairs coupled (`mitotic_pairs` active: `0.5 * (expert_parent + expert_child)`) and held `top_k = 8` fixed across all stages.

**Next Iteration Action Items:**
1. **Un-Tie Twin (Mitotic Sibling) Weights (`release_mitotic_pairs`):**
   - Ensure twin weights (`mitotic_pairs`) are explicitly un-tied (`moe.release_mitotic_pairs()`) after a differentiation warmup window during training (and at inference time) so every split child expert operates as an independent specialist rather than remaining permanently averaged with its parent.
2. **Memory-Adaptive Active Expert Scaling (`top_k` / `dense_subset_k`):**
   - Adjust active experts (`top_k` and/or `dense_subset_k` with `combine_dense_and_sparse`) upward dynamically **based on what available hardware VRAM can support**, pairing our mitotic capacity expansion with higher active expert density whenever memory headroom allows (e.g., we used only `44.2 GiB / 95.0 GiB` on the RTX PRO 6000 at `11.97B`).
