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
    min_correctness_confidence: float = 0.5
    max_contradiction_prob: float = 0.5

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
            thought_budget = max(12, max_new_tokens // 2)
            use_amp = device.type == "cuda"

            step = 0
            while step < max_new_tokens and len(tokens) < max_seq:
                if not has_emitted_a and step >= thought_budget:
                    if max_new_tokens >= 128:
                        at_boundary = tokens[-1] in (ord("."), ord("!"), ord("?"), 10)
                        force_transition = step >= min(max(thought_budget, max_new_tokens - 64), thought_budget + 64)
                    else:
                        at_boundary = tokens[-1] in (32, 10, ord("."), ord(","), ord(";"), ord(":"))
                        force_transition = step >= min(max(thought_budget, max_new_tokens - 12), thought_budget + 12)
                    if at_boundary or force_transition:
                        if len(tokens) + len(a_tag_tokens) < max_seq:
                            tokens.extend(a_tag_tokens)
                            has_emitted_a = True
                            if len(tokens) >= max_seq:
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
                if not has_emitted_a or step < thought_budget + 4:
                    # Do not terminate with <eos> before emitting a non-empty #A$# answer
                    next_logits[self.tokenizer.eos_id] = -torch.inf

                # Prevent consecutive spaces or single-letter space alternation ("t a e o")
                if tokens[-1] in (32, 10):
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

                # Prevent 3x identical character repeats ("sss") or 2x identical punctuation
                if step == 0:
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
                tokens.append(next_id)
                step += 1

                if not has_emitted_a and len(tokens) >= len(a_tag_tokens):
                    if tokens[-len(a_tag_tokens) :] == a_tag_tokens:
                        has_emitted_a = True
                if next_id == self.tokenizer.eos_id:
                    break

            decoded = self.tokenizer.decode(tokens)
            if "\n#A$# " not in decoded:
                decoded = decoded.rstrip() + "\n#A$# "
            return decoded
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
