# MCL4-COE `v6-Scale` Overnight Handoff & 6K Analysis Guide

**Run ID:** `colab_mcl4_coe_v6_scale`  
**Cloud Sync Repo (Hugging Face Model Repo):** `mESmaC/mcl4-coe-runs` (`colab_mcl4_coe_v6_scale/`)  
**Hardware:** Colab NVIDIA RTX PRO 6000 Blackwell Server Edition (`97.9 GB` VRAM)  
**Git Commits:** `fd223d8` (CUDA warp kernels + 1-hop router grad + cached prefix gist), `b3020cd` (in-place warm resume mitosis cadence)

---

## 1. Architecture & Scale Configuration (`v6-Scale`)

| Parameter | Value | Notes |
| :--- | :--- | :--- |
| `vocab_size` | `16,384` | Morphemic-16K BPE + 256 byte fallback + typed workspace sentinels (`#U$#`, `#T$#`, `#A$#`) |
| `max_seq_len` | `1,024` | `query_chunk_size = 1024` (single-pass MCL4 routing per layer) |
| `d_model` / `n_heads` | `768` / `12` | `head_dim = 64`, `markov_router_dim = 64` |
| `n_encoder_layers` / `n_decoder_layers` | `8` / `8` | `16` physical layers (`24` effective layers per pass at `recurrence_steps = 2`) |
| `ple_dim` | `256` | Per-Layer Embeddings across all 16 encoder/decoder layers |
| `expert_hidden_dim` / `top_k` | `1,536` / `2` | SwiGLU experts with tournament-preserve mitotic routing |
| **Bounded Mitosis (`E=8 -> 12`)** | `8 -> 10 -> 12` | Splits `+2` experts every `1,000` steps (`550.01M` $\rightarrow$ `663.28M` $\rightarrow$ `776.55M` params) |
| `batch_size` / `micro_batch_size` | `16` / `8` | `16,384` LM/UTA tokens + `decision_batch_size = 4` (`16` option sequences) per step |
| `total_steps` | `6,000` | Cosine LR (`8e-4` peak, `6%` warmup), Stage-2 Ornithopter + Dream-RSI activates at Step `3,900` (`0.65`) |
| `checkpoint_every` / `eval_every` | `250` / `200` | Auto-synced to `mESmaC/mcl4-coe-runs` (`colab_mcl4_coe_v6_scale/`) |

---

## 2. Dataset Shards (`data_shards_morph16k`)

Built and synced to `mESmaC/mcl4-coe-runs` (`data_shards_morph16k/`):
- **Total Volume:** **`66,184,192` tokens (`66.18M` tokens, `132.4 MB` uint16)** across `64,633` packed `1,024`-token blocks (`46` `.bin` shards):
  - `pretrain_lm`: `54,487` train blocks + `1,294` val blocks (`57.12M` tokens — arXiv C++ research code, streamed C/Rust/Python from `codeparrot/github-code`, FineMath-4+, Quantum/Physics/Math/CS textbooks, and repo source)
  - `uta_trace`: `6,533` train blocks + `639` val blocks (`7.34M` tokens — OpenMathInstruct-2, MetaMathQA, GSM8K, NuminaMath-CoT, Competition Math, plus repo-grounded tool/search/code traces)
  - `typed_decision`: `1,520` train blocks + `160` val blocks (`1.72M` tokens — symmetric 4-way MCQA from ARC-Challenge, OpenBookQA, SciQ, MMLU-Pro, MathQA)

---

## 3. Custom CUDA Warp Kernel & Memory Optimizations (`fd223d8`, `b3020cd`)

1. **`mcl4_fast_ext_v3` CUDA Warp Kernels (`src/mcl4_coe/markov_local4.py`):**
   - `coalesce_topk_warp_kernel` (`_CoalesceTopKAutograd` + `coalesce_topk_bwd_kernel`): Fuses duplicate ID probability mass coalescing + top-$k$ beam selection + normalization into a single shared-memory warp kernel (**`0.133 ms` vs `3.514 ms` — `26.3x` faster, `100%` bitwise-identical token IDs**).
   - `coalesce_sorted_small_warp_kernel` (`_CoalesceSmallAutograd`): Warp-shuffle sorted candidate coalescing for `C <= 32`.
   - `select_subset_small_warp_kernel`: Warp-register anchor reservation + routed top-$k$ subset selection.
2. **1-Hop Straight-Through Router Gradients & `query_chunk_size = 1024` (`src/mcl4_coe/markov_local4.py`):**
   - Runs intermediate walk prefix hops (`0 .. steps - 2`) under `torch.no_grad()` while keeping `seed_mass` and the final hop (`steps - 1`) fully differentiable (`0.0` forward diff).
   - Cuts per-layer activation memory from `1,628 MB` to `758 MB` (**saving `~20.9 GB` VRAM across 24 layers**) and doubles `micro_batch_size` from `4` to `8`.
3. **Cached Prefix Gist Encoder State (`src/mcl4_coe/operators.py`):**
   - Reuses `_last_gist_encoded_hidden` between `compress_gist_context` and `compress_abstraction`, eliminating 2 redundant 8-layer encoder passes (`16` transformer layers) on every gist-conditioned micro-batch.
4. **End-to-End Speedup:**
   - Step wall time dropped from **`3.42s – 3.67s`** to **`2.01s – 2.15s`** (**`~1.75x` faster**, **`~5,900 tok/s`**), cutting reserved VRAM from **`84.8 GB`** down to **`65.3 – 67.0 GB`** even at `E=10` (`663.28M` params).

---

## 4. Live Telemetry Snapshot Prior to Overnight Run (`Step 1 -> 1,123`)

| Step | Experts (`E`) | Params | `val_bpb` $\downarrow$ | `val_loss_pretrain_lm` $\downarrow$ | `val_loss_uta_trace` $\downarrow$ | `val_mcqa` (`n=512`) $\uparrow$ | `step_time_s` |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`1`** | `8` | `550.01M` | `4.7031` | `10.1382` | `10.1036` | `24.02%` | `4.306s` |
| **`200`** | `8` | `550.01M` | `2.2581` | `5.3704` | `4.3011` | `32.23%` | `3.436s` |
| **`400`** | `8` | `550.01M` | `2.1074` | `5.0135` | `4.1026` | `31.25%` | `3.387s` |
| **`500`** | `8` | `550.01M` | `1.9901` | `4.7320` | `3.8886` | **`37.30%`** | `3.635s` |
| **`600`** | `8` | `550.01M` | `1.9018` | `3.9899` | `4.0422` | `35.35%` | `2.025s` |
| **`800`** | `8` | `550.01M` | `1.8421` | `3.8163` | `3.9482` | `29.49%` | `2.027s` |
| **`1000`** | `8 -> 10` | **`663.28M`** | **`1.7877`** | **`3.7315`** | **`3.8164`** | `30.08%` | `2.068s` |

- **Step `1,001` Mitosis Event:** Expanded `E=8 -> E=10` (`550,011,115` $\rightarrow$ `663,282,827` parameters), splitting `[[5, 8], [4, 9]]`.
- **Step `2,000` Scheduled Mitosis Event:** Will expand `E=10 -> E=12` (`663.28M` $\rightarrow$ **`776.55M` parameters**).
- **Step `3,900` (`0.65` warmup ratio):** Stage-2 Ornithopter RL + Dream-RSI self-improvement activates through Step `6,000`.

---

## 5. How to Pick Up Tomorrow on the Laptop (6K Checkpoint Analysis)

1. **Pull latest repo changes:**
   ```bash
   git pull origin main
   ```
2. **Download the completed `colab_mcl4_coe_v6_scale` telemetry & `step_06000.pt` from Hugging Face Hub (`mESmaC/mcl4-coe-runs`):**
   ```python
   from huggingface_hub import snapshot_download

   # Pulls run_manifest.json, metrics.jsonl, samples.jsonl, and checkpoints
   snapshot_download(
       repo_id="mESmaC/mcl4-coe-runs",
       repo_type="model",
       allow_patterns=[
           "colab_mcl4_coe_v6_scale/run_manifest.json",
           "colab_mcl4_coe_v6_scale/metrics.jsonl",
           "colab_mcl4_coe_v6_scale/samples.jsonl",
           "colab_mcl4_coe_v6_scale/checkpoints/step_06000.*",
       ],
       local_dir="./runs",
   )
   ```
3. **6K Analysis Checklist for Tomorrow's Session:**
   - Inspect full `metrics.jsonl` convergence curve (`Step 1 -> 6,000`): `val_bpb`, per-track losses (`pretrain_lm`, `uta_trace`, `typed_decision`), and symmetric 4-way `val_mcqa`.
   - Verify the `Step 2,000` mitosis transition (`E=10 -> E=12`, `776.55M` parameters) and final 12-expert load distribution.
   - Evaluate Stage-2 Ornithopter + Dream-RSI self-improvement metrics (`Step 3,900 -> 6,000`: `ornith_pass_rate`, `dream_replay_score`, `dream_accepted_revisions`).
   - Run test-time recurrent depth scaling ($K \in \{1, 2, 3, 4\}$) and qualitative `#U$# -> #T$# -> #A$#` generation probes on `step_06000.pt`.
