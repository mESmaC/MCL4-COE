"""Milestone 3: Composable Multi-Pass Controller & U$ -> T$ -> A$ Verification Loop.

Chains the 9 Cognitive Operators (`Decision`, `Association`, `Evaluation`, `Deduction`,
`Induction`, `Abduction`, `Abstraction`, `Metacognition`, `Inference`) across recurrent
passes over the shared `MCL4-CED` backbone, and provides:
1. `MultiPassController`: Executes an explicit `CognitiveSchedule` (e.g.,
   `abduction -> association -> decision -> deduction -> evaluation -> metacognition`)
   with in-network `Deduction` + `Metacognition` consistency vetoing.
2. `generate_uta`: Autoregressive `U$ -> T$ -> A$` micro-trace decoding that enforces
   mandatory `T$` scratchpad emission before `A$`.
3. `compute_joint_coe_loss`: Stage-2 joint multi-operator + `0x8000` prompt-masked
   `U$ -> T$ -> A$` supervised warming objective.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .data import ByteTokenizer
from .operators import (
    CognitiveOperatorEnsemble,
    DecisionOutput,
    DeductionOutput,
    EvaluationOutput,
    MetacognitionOutput,
)

VALID_OPERATORS: tuple[str, ...] = (
    "decision",
    "association",
    "evaluation",
    "deduction",
    "induction",
    "abduction",
    "abstraction",
    "metacognition",
    "inference",
)

DEFAULT_SCHEDULE: tuple[str, ...] = (
    "abduction",
    "association",
    "decision",
    "deduction",
    "evaluation",
    "metacognition",
)


@dataclass(frozen=True)
class CognitiveSchedule:
    """Validated sequence of COE operators to execute across recurrent passes."""

    operators: tuple[str, ...] = DEFAULT_SCHEDULE
    recurrence_steps: int = 2
    min_correctness_confidence: float = 0.40
    max_contradiction_prob: float = 0.35

    def __post_init__(self) -> None:
        if not self.operators:
            raise ValueError("CognitiveSchedule requires at least one operator")
        for op in self.operators:
            if op not in VALID_OPERATORS:
                raise ValueError(f"Unknown cognitive operator: {op!r}")
        if isinstance(self.recurrence_steps, bool) or self.recurrence_steps < 1:
            raise ValueError("recurrence_steps must be >= 1")
        if not 0.0 <= self.min_correctness_confidence <= 1.0:
            raise ValueError("min_correctness_confidence must be in [0, 1]")
        if not 0.0 <= self.max_contradiction_prob <= 1.0:
            raise ValueError("max_contradiction_prob must be in [0, 1]")

    @classmethod
    def from_scaffold_text(cls, text: str, *, recurrence_steps: int = 2) -> "CognitiveSchedule":
        """Parse an operator schedule from an `S$` scaffold string (falling back to default)."""
        lowered = text.lower()
        found = tuple(op for op in VALID_OPERATORS if op in lowered)
        return cls(
            operators=found if found else DEFAULT_SCHEDULE,
            recurrence_steps=recurrence_steps,
        )


@dataclass
class MultiPassOutput:
    """Result of a multi-pass `COE` ensemble execution and consistency check."""

    schedule: CognitiveSchedule
    selected_indices: Tensor
    decision: DecisionOutput
    deduction: DeductionOutput | None
    evaluation: EvaluationOutput | None
    metacognition: MetacognitionOutput | None
    vetoed: Tensor
    novelty: Tensor | None = None


@dataclass
class JointCOELossOutput:
    """Breakdown of the Stage-2 joint multi-operator + U$/T$/A$ training loss."""

    total_loss: Tensor
    lm_loss: Tensor | None
    decision_loss: Tensor | None
    deduction_loss: Tensor | None
    evaluation_loss: Tensor | None
    metacognition_loss: Tensor | None


@dataclass
class CandidateTraceScore:
    """Telemetry for a single candidate `#U$# -> #T$# -> #A$#` rollout evaluated by `COE`."""

    index: int
    recurrence_steps: int
    temperature: float
    text: str
    decision_similarity: float
    decision_prob: float
    evaluation_utility: float
    deduction_entailed: float
    deduction_contradicted: float
    metacognition_correctness: float
    metacognition_sufficiency: float
    structural_score: float
    composite_score: float
    vetoed: bool


@dataclass
class COEGenerationOutput:
    """Full output of `MultiPassController.generate_uta_with_coe` with `COE` in the loop."""

    text: str
    baseline_text: str
    selected_index: int
    improved_over_baseline: bool
    candidates: tuple[CandidateTraceScore, ...]
    abstraction_norm: float


def _score_trace_integrity(user_prompt: str, trace_text: str) -> float:
    """Evaluate structural completeness, AST validity (if code), and topical alignment of a `#U$# -> #T$# -> #A$#` trace."""
    import ast
    import re

    if "#T$#" not in trace_text or "#A$#" not in trace_text:
        return 0.0
    after_t = trace_text.split("#T$#", 1)[1]
    if "#A$#" not in after_t:
        return 0.0
    t_seg, a_seg = after_t.split("#A$#", 1)
    t_clean = t_seg.strip()
    a_clean = a_seg.replace("<eos>", "").strip()
    if len(t_clean) < 16 or len(a_clean) < 16:
        return 0.10

    score = 0.55
    # 1. Clean sentence / code termination on both #T$# and #A$#
    if t_clean[-1] in ".!?:;)\"'`]}":
        score += 0.10
    if a_clean[-1] in ".!?:;)\"'`]}":
        score += 0.15
    elif a_clean[-1].isalnum():
        # Penalize mid-word / mid-sentence cutoff in #A$#
        score -= 0.15

    # 2. Penalize verbatim duplication between #T$# and #A$#
    if a_clean == t_clean or a_clean.startswith(t_clean[:40]):
        score -= 0.25

    # 3. Prompt topical alignment and off-topic artifact check
    p_lower = user_prompt.lower()
    body_lower = (t_clean + " " + a_clean).lower()
    prompt_terms = {
        w
        for w in re.findall(r"[a-zA-Z_]{4,}", p_lower)
        if w not in {"write", "explain", "what", "when", "where", "which", "that", "this", "with", "from", "into", "your", "does", "have"}
    }
    if prompt_terms:
        hits = sum(1 for w in prompt_terms if w in body_lower)
        score += 0.15 * min(1.0, hits / max(1, min(4, len(prompt_terms))))

    # Penalize unrelated web/SQL/HTML boilerplate when prompt does not mention them
    for noise_tok in ("<!doctype", "<html", "addEventListener", "SELECT ", "INSERT INTO", "http://"):
        if noise_tok.lower() in body_lower and noise_tok.lower() not in p_lower:
            score -= 0.25

    # 4. Python AST check when prompt requests Python code or #A$# emits a function/class
    wants_python = "python" in p_lower or "def " in p_lower
    has_def = "def " in a_clean or "class " in a_clean
    if wants_python or has_def:
        code_block = a_clean
        if "```" in code_block:
            parts = code_block.split("```")
            if len(parts) >= 2:
                code_block = parts[1]
                if code_block.startswith("python\n"):
                    code_block = code_block[len("python\n") :]
        code_lines = [
            ln
            for ln in code_block.splitlines()
            if ln.startswith(("def ", "class ", "import ", "from ", "    ", "\t", "return ", "if ", "for ", "while "))
        ]
        if code_lines:
            candidate_src = "\n".join(code_lines)
            try:
                ast.parse(candidate_src)
                score += 0.20
            except SyntaxError:
                score -= 0.10

    return float(max(0.0, min(1.0, score)))


class MultiPassController(nn.Module):
    """Executes composable multi-pass COE schedules and `U$ -> T$ -> A$` verification."""

    def __init__(
        self,
        coe: CognitiveOperatorEnsemble,
        schedule: CognitiveSchedule | None = None,
    ) -> None:
        super().__init__()
        self.coe = coe
        self.schedule = CognitiveSchedule() if schedule is None else schedule
        self.tokenizer = ByteTokenizer()

    def forward(
        self,
        context_ids: Tensor,
        context_lengths: Tensor,
        option_ids: Tensor,
        option_lengths: Tensor,
        targets: Tensor | None = None,
        *,
        schedule: CognitiveSchedule | None = None,
        history_buffer: Tensor | None = None,
    ) -> MultiPassOutput:
        """Execute the scheduled COE operators with `RecurrentFeedback` and verify the winner."""
        active_sched = self.schedule if schedule is None else schedule

        # 1. Run multi-pass CED Decision (`Choice` / `Noul` / `Score` / `Abduction` / `Induction`).
        decision_out = self.coe.forward_decision(
            context_ids,
            context_lengths,
            option_ids,
            option_lengths,
            targets=targets,
            recurrence_steps=active_sched.recurrence_steps,
        )
        selected = decision_out.probabilities.argmax(dim=-1)
        batch_rows = torch.arange(context_ids.shape[0], device=context_ids.device)
        chosen_opt_ids = option_ids[batch_rows, selected]
        chosen_opt_lens = option_lengths[batch_rows, selected]

        # 2. Optional Association / Novelty N(q) check against task history buffer.
        novelty = None
        if "association" in active_sched.operators and history_buffer is not None:
            assoc_out = self.coe.forward_association_matrix(
                context_ids, context_lengths, history_buffer=history_buffer
            )
            novelty = assoc_out.novelty

        # 3. Deduction check: does the state logically entail the selected candidate?
        deduction_out = None
        if "deduction" in active_sched.operators:
            deduction_out = self.coe.forward_deduction(
                context_ids,
                context_lengths,
                chosen_opt_ids,
                chosen_opt_lens,
            )

        # 4. Evaluation check: criterion utility U(m, c) in [0, 1].
        evaluation_out = None
        if "evaluation" in active_sched.operators:
            evaluation_out = self.coe.forward_evaluation(
                context_ids,
                context_lengths,
                chosen_opt_ids,
                chosen_opt_lens,
            )

        # 5. Metacognition check: calibrated (`correctness`, `evidence_sufficiency`, `context_benefit`).
        meta_out = None
        if "metacognition" in active_sched.operators:
            meta_out = self.coe.forward_metacognition(
                context_ids,
                context_lengths,
                chosen_opt_ids,
                chosen_opt_lens,
            )

        # 6. In-Network Consistency Veto: flag predictions that fail Deduction or Metacognition.
        vetoed = torch.zeros(context_ids.shape[0], dtype=torch.bool, device=context_ids.device)
        if deduction_out is not None:
            # Index 1 of ("entailed", "contradicted", "undetermined") is "contradicted"
            contradicted_prob = deduction_out.probabilities[:, 1]
            vetoed = vetoed | (contradicted_prob > active_sched.max_contradiction_prob)
        if meta_out is not None:
            # Index 0 of ("correctness", "evidence_sufficiency", "context_benefit") is "correctness"
            correctness_prob = meta_out.probabilities[:, 0]
            vetoed = vetoed | (correctness_prob < active_sched.min_correctness_confidence)

        return MultiPassOutput(
            schedule=active_sched,
            selected_indices=selected,
            decision=decision_out,
            deduction=deduction_out,
            evaluation=evaluation_out,
            metacognition=meta_out,
            vetoed=vetoed,
            novelty=novelty,
        )

    @torch.no_grad()
    def generate_uta(
        self,
        user_prompt: str,
        *,
        max_new_tokens: int = 320,
        temperature: float = 0.0,
        recurrence_steps: int | None = None,
    ) -> str:
        """Synthesize a `#U$# -> #T$# -> #A$#` trace, guaranteeing complete `#T$#` and `#A$#` segments."""
        was_training = self.coe.training
        self.coe.eval()
        try:
            device = next(self.coe.parameters()).device
            # Seed the generation with `<bos>#U$# {user_prompt}\n#T$# ` so the model never skips `#T$#`.
            prefix = f"#U$# {user_prompt.strip()}\n#T$# "
            tokens = self.tokenizer.encode(prefix, add_bos=True, add_eos=False)
            prompt_len = len(tokens)
            max_seq = self.coe.backbone.config.max_seq_len
            a_tag_tokens = self.tokenizer.encode("\n#A$# ", add_bos=False, add_eos=False)
            has_emitted_a = False
            a_start_token_idx = -1
            thought_budget = (
                min(140, max(56, int(max_new_tokens * 0.35)))
                if max_new_tokens >= 128
                else max(12, max_new_tokens // 2)
            )
            grace_tokens = 32 if max_new_tokens >= 128 else 0
            hard_max_steps = max_new_tokens + grace_tokens
            use_amp = device.type == "cuda"

            step = 0
            while step < hard_max_steps and len(tokens) < max_seq:
                if not has_emitted_a and step >= thought_budget:
                    if max_new_tokens >= 128:
                        at_sentence = (
                            tokens[-1] in (ord("."), ord("!"), ord("?"), 10)
                            or (
                                len(tokens) >= 2
                                and tokens[-2] in (ord("."), ord("!"), ord("?"))
                                and tokens[-1] in (32, 10)
                            )
                        )
                        at_clause_word = (
                            step >= thought_budget + 28
                            and tokens[-1] in (32, ord(","), ord(";"), ord(":"))
                        )
                        force_transition = step >= min(
                            thought_budget + 44, max_new_tokens - 64
                        )
                        at_boundary = at_sentence or at_clause_word
                    else:
                        at_boundary = tokens[-1] in (32, 10, ord("."), ord(","), ord(";"), ord(":"))
                        force_transition = step >= min(
                            max(thought_budget, max_new_tokens - 12), thought_budget + 12
                        )
                    if at_boundary or force_transition:
                        if len(tokens) + len(a_tag_tokens) + 1 < max_seq:
                            if tokens[-1] in (32, ord(","), ord(";")) and max_new_tokens >= 128:
                                tokens.pop()
                                if tokens[-1] not in (ord("."), ord("!"), ord("?"), 10):
                                    tokens.append(ord("."))
                            tokens.extend(a_tag_tokens)
                            has_emitted_a = True
                            a_start_token_idx = len(tokens)
                            if len(tokens) >= max_seq:
                                break

                # Once past max_new_tokens (in grace window), stop cleanly as soon as #A$# hits a sentence boundary
                if has_emitted_a and step >= max_new_tokens:
                    if tokens[-1] in (ord("."), ord("!"), ord("?"), 10):
                        break

                inp = torch.tensor([tokens], dtype=torch.long, device=device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                    out = self.coe.forward_inference(inp, recurrence_steps=recurrence_steps)
                next_logits = out.logits[0, -1, :259].float()

                # Mask control bytes (<32 except newline=10), DEL..0xFF (127..255), pad_id (256), bos_id (257)
                next_logits[:10] = -torch.inf
                next_logits[11:32] = -torch.inf
                next_logits[127:256] = -torch.inf
                next_logits[self.tokenizer.pad_id] = -torch.inf
                next_logits[self.tokenizer.bos_id] = -torch.inf
                if not has_emitted_a or (
                    a_start_token_idx > 0 and len(tokens) - a_start_token_idx < 4
                ):
                    # Do not terminate with <eos> before emitting a non-empty #A$# answer
                    next_logits[self.tokenizer.eos_id] = -torch.inf

                # Prevent premature tag transition before minimum thought budget
                if not has_emitted_a and tokens[-1] == 10 and step < min(48, thought_budget):
                    next_logits[ord("#")] = -torch.inf

                # Permit paragraph breaks (\n\n) in multi-step reasoning traces while preventing >=3 consecutive newlines,
                # mid-line consecutive spaces, or single-letter space alternation ("t a e o").
                if len(tokens) >= 2 and tokens[-1] == 10 and tokens[-2] == 10:
                    next_logits[10] = -torch.inf
                elif tokens[-1] == 32:
                    # Check if we are in a leading-indentation run right after a newline
                    indent_run = 0
                    idx_back = len(tokens) - 1
                    while idx_back >= 0 and tokens[idx_back] == 32:
                        indent_run += 1
                        idx_back -= 1
                    is_leading_indent = (idx_back >= 0 and tokens[idx_back] == 10 and indent_run < 8)
                    if not is_leading_indent:
                        next_logits[32] = -torch.inf
                    next_logits[10] = -torch.inf
                elif (
                    len(tokens) >= 2
                    and tokens[-2] in (32, 10)
                    and tokens[-1] not in (ord("a"), ord("A"), ord("I"))
                ):
                    next_logits[32] -= 2.5
                    next_logits[10] -= 2.5

                # Antidoom span-loop guard: never penalize individual English letters inside words;
                # instead, detect and suppress only the exact trigger byte that restarts a repeated
                # 4-byte or 6-byte span in the generated window.
                gen_window = tokens[max(prompt_len, len(tokens) - 128) :]
                if len(gen_window) >= 6:
                    ctx5 = gen_window[-5:]
                    for i in range(len(gen_window) - 5):
                        if gen_window[i : i + 5] == ctx5:
                            trig_b = gen_window[i + 5]
                            if 32 <= trig_b < 127:
                                next_logits[trig_b] -= 6.0
                if len(gen_window) >= 4:
                    ctx3 = gen_window[-3:]
                    hits_3: dict[int, int] = {}
                    for i in range(len(gen_window) - 3):
                        if gen_window[i : i + 3] == ctx3:
                            trig_b = gen_window[i + 3]
                            hits_3[trig_b] = hits_3.get(trig_b, 0) + 1
                    for trig_b, cnt in hits_3.items():
                        if cnt >= 2 and 32 <= trig_b < 127:
                            next_logits[trig_b] -= 3.5 * cnt

                # Prevent XML/HTML tag openers ('<') at start of #T$# or #A$#, 3x identical character repeats, or 2x punctuation
                if step == 0 or (len(tokens) >= len(a_tag_tokens) and tokens[-len(a_tag_tokens) :] == a_tag_tokens):
                    next_logits[ord("<")] = -torch.inf
                if len(tokens) >= 3 and tokens[-1] == tokens[-2] == tokens[-3]:
                    next_logits[tokens[-1]] = -torch.inf
                elif len(tokens) >= 2 and tokens[-1] == tokens[-2] and not chr(min(126, max(32, tokens[-1]))).isalnum():
                    next_logits[tokens[-1]] = -torch.inf

                if temperature <= 0.0:
                    next_id = int(next_logits.argmax(dim=-1).item())
                else:
                    probs = (next_logits / temperature).softmax(dim=-1)
                    next_id = int(torch.multinomial(probs, num_samples=1).item())

                # Cleanly terminate if #A$# is complete and the model attempts to open a new episode marker after newline
                if (
                    has_emitted_a
                    and a_start_token_idx > 0
                    and len(tokens) - a_start_token_idx >= 8
                    and tokens[-1] == 10
                    and next_id in (ord("#"), ord("<"))
                ):
                    break

                # Also stop cleanly near the end of max_new_tokens if #A$# just completed a sentence before a newline
                if (
                    has_emitted_a
                    and a_start_token_idx > 0
                    and step >= int(max_new_tokens * 0.82)
                    and tokens[-1] in (ord("."), ord("!"), ord("?"))
                    and next_id == 10
                ):
                    break

                tokens.append(next_id)
                step += 1

                if not has_emitted_a and len(tokens) >= len(a_tag_tokens):
                    if tokens[-len(a_tag_tokens) :] == a_tag_tokens:
                        has_emitted_a = True
                        a_start_token_idx = len(tokens)
                if next_id == self.tokenizer.eos_id:
                    break

            decoded = self.tokenizer.decode(tokens).rstrip()
            if "\n#A$# " not in decoded:
                decoded = decoded + "\n#A$# "
            return decoded
        finally:
            self.coe.train(was_training)

    @torch.no_grad()
    def generate_uta_with_coe(
        self,
        user_prompt: str,
        *,
        max_new_tokens: int = 384,
        candidate_specs: tuple[tuple[int, float], ...] = (
            (1, 0.0),
            (2, 0.0),
            (1, 0.30),
            (2, 0.35),
            (2, 0.50),
        ),
        seed: int = 42,
    ) -> COEGenerationOutput:
        """Synthesize and verify `#U$# -> #T$# -> #A$#` with all 9 `COE` operators in the loop.

        1. Runs `Abstraction` (`compress_abstraction`) on `#U$# {user_prompt}`.
        2. Generates candidate `#T$# -> #A$#` trajectories across `candidate_specs` (`(recurrence_steps, temperature)`),
           where Candidate 0 (`r=1, temp=0.0`) is the baseline single-pass trace without `COE`.
        3. Evaluates all candidates through `Decision` (`Choice` / `Abduction` / `Induction` matched-filter Pearson similarity),
           `Deduction` (`entailed` vs. `contradicted`), `Evaluation` (`utility` $U(m, c)$), and
           `Metacognition` (`correctness`, `evidence_sufficiency`).
        4. Combines neural `COE` scores with `_score_trace_integrity` (sentence completeness, Python AST verification,
           and topical alignment), applies the `COE` consistency veto, and returns the winning trajectory + full telemetry.
        """
        from .data import DecisionExample, collate_decisions

        was_training = self.coe.training
        self.coe.eval()
        try:
            device = next(self.coe.parameters()).device
            raw_traces: list[tuple[int, float, str]] = []
            for idx, (r_steps, temp) in enumerate(candidate_specs):
                if temp > 0.0:
                    torch.manual_seed(seed + idx * 131)
                trace = self.generate_uta(
                    user_prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temp,
                    recurrence_steps=r_steps,
                )
                raw_traces.append((r_steps, temp, trace))

            baseline_text = raw_traces[0][2]

            # Build candidate bodies (everything from #T$# onward) for COE multi-operator evaluation
            cand_bodies: list[str] = []
            for _, _, tr in raw_traces:
                idx_t = tr.find("#T$#")
                body = tr[idx_t:].strip() if idx_t >= 0 else tr.strip()
                cand_bodies.append(body if body else "#T$# empty\n#A$# empty")

            max_seq = self.coe.backbone.config.max_seq_len
            ctx_text = (
                f"#U$# {user_prompt.strip()}\n"
                f"#Q$# [choice] Select the most accurate, complete, and logically entailed #T$# -> #A$# response."
            )
            ex = DecisionExample(
                question_id="coe_loop",
                kind="choice",
                context=ctx_text.encode("utf-8")[: max_seq - 4].decode("utf-8", errors="ignore"),
                descriptions=tuple(
                    b.encode("utf-8")[: max_seq - 4].decode("utf-8", errors="ignore")
                    for b in cand_bodies
                ),
                values=tuple(f"cand_{i}" for i in range(len(cand_bodies))),
                target=0,
            )
            batch = collate_decisions([ex], max_seq, device=device)
            ctx_ids = batch["context_ids"]
            ctx_lens = batch["context_lengths"]
            opt_ids = batch["option_ids"]
            opt_lens = batch["option_lengths"]

            # 1. Abstraction state compression norm
            abs_vec = self.coe.compress_abstraction(ctx_ids, ctx_lens, level="standard")
            abs_norm = float(abs_vec.float().norm(dim=-1).mean().item())

            # 2. Decision (`Choice` / `Abduction` / `Induction`) with recurrent upper-CED passes
            dec_out = self.coe.forward_decision(
                ctx_ids,
                ctx_lens,
                opt_ids,
                opt_lens,
                recurrence_steps=self.schedule.recurrence_steps,
            )
            dec_sims = dec_out.similarities[0].detach().float().cpu().tolist()
            dec_probs = dec_out.probabilities[0].detach().float().cpu().tolist()

            # 3. Batched Deduction, Evaluation, and Metacognition across all K candidates
            k_cands = len(cand_bodies)
            rep_ctx_ids = ctx_ids.expand(k_cands, -1).contiguous()
            rep_ctx_lens = ctx_lens.expand(k_cands).contiguous()
            flat_opt_ids = opt_ids[0]
            flat_opt_lens = opt_lens[0]

            ded_out = self.coe.forward_deduction(
                rep_ctx_ids, rep_ctx_lens, flat_opt_ids, flat_opt_lens
            )
            eval_out = self.coe.forward_evaluation(
                rep_ctx_ids, rep_ctx_lens, flat_opt_ids, flat_opt_lens
            )
            meta_out = self.coe.forward_metacognition(
                rep_ctx_ids, rep_ctx_lens, flat_opt_ids, flat_opt_lens
            )

            ded_probs = ded_out.probabilities.detach().float().cpu().tolist()
            eval_utils = eval_out.utilities.detach().float().cpu().tolist()
            meta_probs = meta_out.probabilities.detach().float().cpu().tolist()

            scored_candidates: list[CandidateTraceScore] = []
            for i, (r_steps, temp, tr) in enumerate(raw_traces):
                d_sim = float(dec_sims[i])
                d_prob = float(dec_probs[i])
                u_eval = float(eval_utils[i])
                p_ent = float(ded_probs[i][0])
                p_con = float(ded_probs[i][1])
                m_cor = float(meta_probs[i][0])
                m_suf = float(meta_probs[i][1])
                s_int = _score_trace_integrity(user_prompt, tr)

                is_vetoed = bool(
                    p_con > self.schedule.max_contradiction_prob
                    or m_cor < self.schedule.min_correctness_confidence
                    or s_int < 0.35
                )
                neural_score = (
                    0.35 * d_prob
                    + 0.25 * u_eval
                    + 0.20 * (p_ent - 0.75 * p_con)
                    + 0.15 * (0.7 * m_cor + 0.3 * m_suf)
                )
                comp_score = neural_score + 0.15 * s_int - (0.30 if is_vetoed else 0.0)
                scored_candidates.append(
                    CandidateTraceScore(
                        index=i,
                        recurrence_steps=r_steps,
                        temperature=temp,
                        text=tr,
                        decision_similarity=round(d_sim, 4),
                        decision_prob=round(d_prob, 4),
                        evaluation_utility=round(u_eval, 4),
                        deduction_entailed=round(p_ent, 4),
                        deduction_contradicted=round(p_con, 4),
                        metacognition_correctness=round(m_cor, 4),
                        metacognition_sufficiency=round(m_suf, 4),
                        structural_score=round(s_int, 4),
                        composite_score=round(comp_score, 4),
                        vetoed=is_vetoed,
                    )
                )

            best_cand = max(scored_candidates, key=lambda c: c.composite_score)
            return COEGenerationOutput(
                text=best_cand.text,
                baseline_text=baseline_text,
                selected_index=best_cand.index,
                improved_over_baseline=(best_cand.text != baseline_text),
                candidates=tuple(scored_candidates),
                abstraction_norm=round(abs_norm, 4),
            )
        finally:
            self.coe.train(was_training)


def compute_joint_coe_loss(
    coe: CognitiveOperatorEnsemble,
    *,
    lm_batch: tuple[Tensor, Tensor] | None = None,
    decision_batch: dict[str, Tensor] | None = None,
    deduction_batch: dict[str, Tensor] | None = None,
    evaluation_batch: dict[str, Tensor] | None = None,
    metacognition_batch: dict[str, Tensor] | None = None,
    weights: dict[str, float] | None = None,
) -> JointCOELossOutput:
    """Compute Stage-2 joint multi-task loss across `U$ -> T$ -> A$` LM and COE operators."""
    w = {
        "lm": 1.0,
        "decision": 1.0,
        "deduction": 0.5,
        "evaluation": 0.5,
        "metacognition": 0.5,
    }
    if weights is not None:
        w.update(weights)

    total: Tensor | None = None
    lm_loss = decision_loss = deduction_loss = evaluation_loss = metacognition_loss = None

    def _accum(current: Tensor | None, term: Tensor, scale: float) -> Tensor:
        scaled = term * scale
        return scaled if current is None else current + scaled

    if lm_batch is not None:
        inp, tgt = lm_batch
        lm_out = coe.forward_inference(inp, targets=tgt)
        assert lm_out.loss is not None
        lm_loss = lm_out.loss
        total = _accum(total, lm_loss, w["lm"])

    if decision_batch is not None:
        dec_out = coe.forward_decision(**decision_batch)
        assert dec_out.loss is not None
        decision_loss = dec_out.loss
        total = _accum(total, decision_loss, w["decision"])

    if deduction_batch is not None:
        ded_out = coe.forward_deduction(**deduction_batch)
        assert ded_out.loss is not None
        deduction_loss = ded_out.loss
        total = _accum(total, deduction_loss, w["deduction"])

    if evaluation_batch is not None:
        eval_out = coe.forward_evaluation(**evaluation_batch)
        assert eval_out.loss is not None
        evaluation_loss = eval_out.loss
        total = _accum(total, evaluation_loss, w["evaluation"])

    if metacognition_batch is not None:
        meta_out = coe.forward_metacognition(**metacognition_batch)
        assert meta_out.loss is not None
        metacognition_loss = meta_out.loss
        total = _accum(total, metacognition_loss, w["metacognition"])

    if total is None:
        raise ValueError("compute_joint_coe_loss requires at least one active batch")

    return JointCOELossOutput(
        total_loss=total,
        lm_loss=lm_loss,
        decision_loss=decision_loss,
        deduction_loss=deduction_loss,
        evaluation_loss=evaluation_loss,
        metacognition_loss=metacognition_loss,
    )
