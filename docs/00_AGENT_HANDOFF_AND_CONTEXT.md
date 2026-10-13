# 00. Agent Handoff & Complete Session Context (`Laptop -> RTX 5070 Desktop`)

> **Read This First When Resuming on the Desktop:**
> This document captures the full technical, architectural, and strategic context from our session on 2026-10-13 so we can pick up seamlessly on the **RTX 5070 desktop** without losing a single detail.

---

## 1. Who You Are Working With & How We Collaborate

* **Researcher / Founder:** Cameron (`mESmaC`), building **YY (`yycore.ai`)**.
* **Background:** 15 years of programming experience, self-taught embedded systems engineer with deep instincts for bit-packing, memory hierarchy, DMA/offloading, and custom transformer/kernel architectures (`Mirror`, `mini`, `mcattn`, `oracle`, `spalmer`, `stc`, `molor`, `xerces`, `CDAS`, `yyLab`).
* **Business & Financial Goal:**
  * Currently makes `$800`/week at his day job; only needs **`$500`/week (`~$2,100`/month)** net to replace his day job and do full-time independent AI research.
  * **Strict Constraint:** Zero sales calls (severe anxiety around synchronous sales/pitching). Everything must be **100% proof-led inbound** (benchmarks, interactive visualizations, open research posts on `yycore.ai`, `r/LocalLLaMA`, Hacker News, Hugging Face) + **self-serve checkout** (Polar.sh / Stripe for `yyLab Pro`, expert packs, and commercial SDK licenses) + **written non-dilutive research/compute grants** (Google TPU Research Cloud, Emergent Ventures).
* **Collaboration Style:**
  * Explain *why* things happen mathematically and mechanically (never dismiss ideas with generic "scaling laws = only money can win" platitudes).
  * Keep full architectural control from scratch (`Option A`): Cameron researches changes at the level of attention (`MCL4`), encoder-decoder structure (`CED`), single-softmax cognitive operators (`COE`), and self-improvement loops (`Ornith 1.0 / 1.5`).

---

## 2. Hardware Environment (Desktop vs. Laptop)

| Machine | Specs | Role |
| :--- | :--- | :--- |
| **Primary Desktop (Current Target)** | **NVIDIA GeForce RTX 5070 (`12 GB` GDDR7 VRAM, Blackwell `sm_120`)** + **`32 GB` DDR5 RAM** + **`10 TB` NVMe SSD** + `2 TB` disk | Primary training, native Windows **Blackwell `NVFP4`** (`~1 PFLOP` sparse 2:4 tensor cores for rollouts/search), `CPUOffloadAdamW` in 32GB DDR5 (`1B–4B` sparse params, `50M–300M` active), and 10TB NVMe replay/task buffer. |
| **Portable Laptop (Used Earlier Today)** | **NVIDIA GeForce RTX 2050 (`4 GB` VRAM, Ampere `sm_86`)** + **`32 GB` DDR5 RAM** | Used earlier today to audit `mini` and `mcattn`, build the block-aligned `U$ / T$ / A$` pipeline, and run the 1M-parameter baseline experiments. |

---

## 3. Key Technical Discoveries From Today's Session

### 3.1 What We Proved in `mini` (`https://github.com/mESmaC/mini.git`, commit `2e5358f`)
1. **Why Raw `Prose` & Raw Distillation Failed Previously:**
   * `mini/src/mini_moe/data.py` treats each line of a `.txt` file as an independent `<bos>...<eos>` document. Raw Project Gutenberg books had 350-line legal headers and 70-character hard line wraps (turning every 10-word line into a truncated fragment).
   * Raw teacher distillation traces had 300-byte system prompts, 105-byte `<message_metadata>` tags, tool call schemas, and 4,000-byte `<think>` blocks that got severed across 256-byte block boundaries, separating `U$` prompts from `A$` answers.
2. **The Fix (`scripts/prepare_clean_corpora.py` + `TokenBlockDataset.batch()`):**
   * Stripped Gutenberg boilerplate, unwrapped paragraphs, and deduplicated 18 books into `31,145` clean paragraphs (`15.24M` byte tokens).
   * Converted distillation traces into tool-free, compact **`U$ / T$ / A$`** micro-traces (`<= 256` byte tokens) written **1 complete trace per 256-token block** in `train.bin` / `val.bin`.
   * Added **`0x8000` high-bit prompt masking** in `uint16` `train.bin` so `TokenBlockDataset.batch()` keeps `<bos>U$ ...\n` in `input_ids` (`raw & 0x7FFF`) while masking both prompt tokens and `pad_id = 256` tail tokens as `-100` in `labels`.
3. **Training & Generation Results on the 4GB RTX 2050:**
   * **Stage 1 (`configs/pretrain_prose_1m.json`, 5 epochs = `780` steps):** Ran at **`~583,000` byte tokens/sec** (`220s` total, `2.33 GiB` VRAM), dropping `val_ce_loss` from `5.58` -> **`1.2222`** (`ppl = 3.39`), with all 12 experts (`3` layers $\times$ `4` experts) balanced at `~0.25`.
   * **Stage 2 (`configs/micro_traces_1m.json --weights-from runs\prose_1m\best.pt`, 5 epochs = `610` steps):** Ran at **`~500,000` byte tokens/sec** (`168s` total, **`796 MiB` peak VRAM**), dropping masked completion `val_ce_loss` from `2.932` -> **`1.1093`** (`ppl = 3.03`).
   * **Generation Diagnostics:**
     * Default `--temperature 0.8` on a byte vocabulary (`260` IDs) injects random noise at every character/digit. Running with `--temperature 0` (greedy) immediately produced clean `U$ -> T$ -> A$ -> <eos>` structure (`T$ The smallest number is 12.\nA$ 12`).
     * Identified why the 3-layer model guessed `12`: `65.5%` of `micro_traces_256` jumped straight from `U$` to `A$` without a `T$` calculation step, and a 3-layer model cannot compute multi-step arithmetic in a single forward pass without `T$`. **Rule for `MCL4-COE`:** 100% of reasoning traces must include a clean step-by-step `T$` scratchpad (with `"Verification:"` newline leaks stripped).

### 3.2 Why `stc` (`0.0625` Bits/Virtual Weight) Stalled—And How `MCL4-COE` Solves It
* `stc` tied many virtual weights to a small pool of physical bits (`0.0625` bits/virtual weight). That bounded the gradient update rank $\text{rank}(\nabla W)$ by the small physical state (causing destructive gradient collisions and slow learning) while paying the FLOP cost of the large virtual matrix.
* **The Fix:** Flip the ratio using memory hierarchy—keep **active FLOPs per token small** (`MCL4` + `top-2` fine-grained MoE in VRAM) while keeping **total parameters independent** (`PLE` + fine-grained MoE backed by `CPUOffloadAdamW` in 32GB DDR5 RAM).

### 3.3 Unifying `mcattn`, `oracle`, and `mini` into `MCL4-COE`
1. **`MCL4` (`markov_local_tokens: 4` from `mcattn` & `oracle/markov_local4.py`):**
   * Reserving **4 local anchors (`{q, q-1, q-2, q-3}`) + 12 bounded Markov-walked positions (`16` tokens attended total)** beat standard dense 256-token attention on TinyStories (**`1.1832` vs `1.2189` nats/byte**) on matched initial weights and batches.
   * Why: Eliminates softmax dilution across irrelevant context while executing **3 relational graph hops per layer**.
2. **`CED` (Causal Encoder-Decoder from `oracle/causal.py`) Fixes `mcattn.decision` (`AssociationModel`):**
   * In `mcattn`, `AssociationModel` (`Choice`, `Noul`, `Score` via centered Pearson similarity + single softmax) was stuck at `44.4%` (`12/27` majority baseline) because `context` (`state + question`) and `options` were encoded in separate passes (bi-encoder) with zero cross-attention.
   * In `MCL4-COE`, the **lower `CED` encoder layers** encode the shared `state` once into global `K/V`, and the **upper `CED` decoder layers** let each candidate option cross-attend directly to that state's global `K/V` via `MCL4` before pooling into the 32-dim Pearson softmax. This preserves **100% bitwise permutation invariance** across options while giving every option full attention access to the state.
3. **`COE` (9 Composable Single-Softmax Cognitive Operators from `oracle`):**
   * `Association`, `Decision` (`Choice`/`Noul`/`Score`), `Evaluation`, `Deduction`, `Induction`, `Abduction`, `Abstraction`, `Metacognition`, and `Inference` (`U$ -> T$ -> A$`), chained dynamically via `RecurrentFeedback` (`anchor + sigmoid(gate) * LayerNorm(state)`).
4. **Ornith-1.0 & Ornith-1.5 Self-Improvement Mapped Natively to `COE` (`docs/05_ORNITH_SELF_SCAFFOLDING_SPEC.md`):**
   * **Ornith-1.0 Mode (`U$ -> S$ -> COE -> T$ -> A$`):** 2-stage self-scaffolding where `S$` lives in the `CED` encoder (zero ChatML escaping bloat) and schedules the `COE` operators.
   * **Ornith-1.5 Mode (`Q$ -> H$/S$ -> COE -> T$ -> A$`):** 3-stage self-improvement where the model proposes frontier tasks $Q\$$ ($R_{\text{task}} = V \cdot D \cdot N$ targeting empirical pass rate $p^* = 0.2$), generates harnesses/scaffolds $H\$, S\$$ ($R_{\text{harness}} = C \cdot F \cdot H$), and solves them ($R_{\text{rollout}}$)—with every reward factor ($N(q)$ via `Association`, $V/C$ via `Deduction`, $F$ via `Evaluation`, and $p^*=0.2$ prior via `Metacognition`) powered directly by `COE`'s single-softmax heads!

---

## 4. Exact Code Locations Across Repositories to Pull From on the Desktop

When building `src/mcl4_coe/` on the desktop, pull/adapt from these three repositories (`https://github.com/mESmaC/mini.git` and `https://github.com/mESmaC/oracle.git`):

1. **From [`mESmaC/mini`](https://github.com/mESmaC/mini):**
   * `src/mini_moe/optim.py` -> `CPUOffloadAdamW` & `prepare_model`
   * `src/mini_moe/data.py` -> `ByteTokenizer`, `TokenBlockDataset` (with `0x8000` prompt masking), `EpochBlockSampler`, `HostBatchPrefetcher`
   * `src/mini_moe/expert_lifecycle.py` & `expert_morph.py` -> Bounded MoE expert growth/pruning (`ExpertLifecycle`)
   * `src/mini_moe/chat_rl.py` & `rl.py` -> Clipped GRPO, scaffold cache validation, and multi-stage rollout replay
   * `scripts/prepare_clean_corpora.py` -> Clean prose & block-aligned `U$ / T$ / A$` data pipeline

2. **From [`mESmaC/oracle`](https://github.com/mESmaC/oracle):**
   * `src/oracle/markov_local4.py`, `local4_selection.py`, `local4_readout.py` -> Bounded `MCL4` token router with `_coalesce` and 4 mandatory local anchors
   * `src/oracle/causal.py` -> `CED` (lower causal encoder layers -> upper global `K/V` decoder layers)
   * `src/oracle/ple.py` -> Gemma-style `PerLayerEmbedding`
   * `src/oracle/nvfp4.py` -> Native Windows Blackwell `cuBLASLt` packed `NVFP4` (`E2M1` + `FP8 E4M3` block scales, Hadamard transforms, stochastic rounding)
   * `src/oracle/reasoning.py` -> `RecurrentFeedback` (`anchor + gate.sigmoid() * LayerNorm(state)`)
   * `src/oracle/{association.py, decision.py, evaluation.py, abstraction.py, metacognition.py}` -> Typed cognitive operator schemas and Pearson/cosine single-softmax readouts

---

## 5. Immediate Next Step When You Sit Down at the RTX 5070 Desktop

1. Clone this repository (`https://github.com/mESmaC/MCL4-COE.git`) and `mini` / `oracle` if not already on the desktop.
2. **Execute Milestone 1 & 2:**
   * Scaffold `pyproject.toml` and `src/mcl4_coe/` (`backbone.py`, `markov_local4.py`, `ced.py`, `operators.py`, `optim.py`, `data.py`).
   * Run the **first validation benchmark** on the 5070: train the `CED`-contextualized `Association` / `Decision` (`Choice`, `Noul`, `Score`) head on top of the `MCL4-CED` backbone and verify that `CED` cross-attention jumps past `mcattn.decision`'s old `44.4%` bi-encoder ceiling to `>95%` accuracy while maintaining exact bitwise permutation invariance.
   * Next, wire up the **100%-`T$` scratchpad micro-traces** + **Ornith 1.0 / 1.5 self-improvement loop** (`src/mcl4_coe/ornith_rl.py`) using the 5070's native `NVFP4` rollout path.
