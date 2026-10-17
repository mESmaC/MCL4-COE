"""Ornith-1.0 Self-Scaffolding, Ornith-1.5 Self-Improvement, GRPO, & ExpertLifecycle.

Implements all equations and structural self-improvement mechanics from
`docs/05_ORNITH_SELF_SCAFFOLDING_SPEC.md`:
1. Ornith-1.0 Two-Stage Self-Scaffolding (`U$ -> S$ -> COE -> T$ -> A$`) with
   monotonic `ScaffoldCache` retention (`max_i R_i > max(0, R_cached)`).
2. Ornith-1.5 Three-Stage End-to-End Self-Improvement (`Q$ -> H$/S$ -> T$/A$`):
   - Task Reward: $R_{\\text{task}} = V(q, s) \\times D(q, s, \\{\\tau_i\\}) \\times N(q)$
     with hard gate $V=0 \\implies R_{\\text{task}}=0$, frontier difficulty $p^* = 0.2$,
     and `<0.01ms` `Association` Novelty $N(q)$ against `NoveltyTaskBuffer`.
   - Harness Reward: $R_{\\text{harness}} = C(q, h) \\times F(h, \\{\\tau_i\\}) \\times H(h)$.
   - Staleness-Weighted Clipped GRPO (`staleness_weight`, `grpo_clipped_loss`).
3. Bounded Speculative `ExpertLifecycle` (`grow_experts`, `prune_experts`) across all
   `encoder_layers` and `decoder_layers` of `MCL4CEDBackbone` with `CPUOffloadAdamW`.
"""

from __future__ import annotations

import ast
import math
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .backbone import MCL4CEDBackbone
from .operators import pearson_similarity
from .optim import CPUOffloadAdamW

MAX_PARAMETERS = 50_000_000


class NoveltyTaskBuffer:
    """VRAM/RAM ring buffer of 32-dim centered Pearson task embeddings for <0.01ms Novelty N(q)."""

    def __init__(
        self,
        projection_dim: int = 32,
        max_capacity: int = 100_000,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if projection_dim < 3 or max_capacity < 1:
            raise ValueError("projection_dim >= 3 and max_capacity >= 1 required")
        self.projection_dim = projection_dim
        self.max_capacity = max_capacity
        self.device = torch.device(device)
        self.dtype = dtype
        self._buffer = torch.empty((max_capacity, projection_dim), device=self.device, dtype=dtype)
        self._size = 0
        self._write_ptr = 0

    def __len__(self) -> int:
        return self._size

    @staticmethod
    def _normalize_pearson(vecs: Tensor, eps: float = 1e-6) -> Tensor:
        with torch.autocast(device_type=vecs.device.type, enabled=False):
            centered = vecs.float() - vecs.float().mean(dim=-1, keepdim=True)
            return F.normalize(centered, dim=-1, eps=eps)

    @torch.no_grad()
    def add(self, embeddings: Tensor) -> None:
        """Normalize and insert `[N, D]` task embeddings into the ring buffer."""
        if embeddings.ndim == 1:
            embeddings = embeddings.unsqueeze(0)
        if embeddings.ndim != 2 or embeddings.shape[1] != self.projection_dim:
            raise ValueError(f"Expected [N, {self.projection_dim}] embeddings")
        normed = self._normalize_pearson(embeddings).to(device=self.device, dtype=self.dtype)
        n = normed.shape[0]
        if n >= self.max_capacity:
            self._buffer.copy_(normed[-self.max_capacity :])
            self._size = self.max_capacity
            self._write_ptr = 0
            return
        end = self._write_ptr + n
        if end <= self.max_capacity:
            self._buffer[self._write_ptr : end].copy_(normed)
        else:
            first = self.max_capacity - self._write_ptr
            self._buffer[self._write_ptr :].copy_(normed[:first])
            self._buffer[: n - first].copy_(normed[first:])
        self._write_ptr = end % self.max_capacity
        self._size = min(self.max_capacity, self._size + n)

    @torch.no_grad()
    def compute_novelty(self, query_embeddings: Tensor) -> Tensor:
        """Compute $N(q) = 1 - \\max_{q_j \\in \\mathcal{B}} \\operatorname{sim}(q, q_j)$ in $[0, 2]$."""
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.unsqueeze(0)
        if self._size == 0:
            return torch.ones(
                query_embeddings.shape[0], device=query_embeddings.device, dtype=torch.float32
            )
        q_norm = self._normalize_pearson(query_embeddings).to(
            device=self.device, dtype=self.dtype
        )
        active = self._buffer[: self._size]
        max_sim = (q_norm @ active.t()).float().clamp(-1.0, 1.0).max(dim=-1).values
        return (1.0 - max_sim).clamp(0.0, 2.0).to(query_embeddings.device)

    def active_embeddings(self) -> Tensor:
        return self._buffer[: self._size]


@dataclass
class ScaffoldEntry:
    scaffold_text: str
    best_reward: float
    step_updated: int


class ScaffoldCache:
    """Ornith-1.0 best-scaffold cache retaining winning scaffolds per task ID."""

    def __init__(self) -> None:
        self._entries: dict[str, ScaffoldEntry] = {}

    def get(self, task_id: str, default: str = "abduction -> association -> decision -> deduction") -> str:
        entry = self._entries.get(task_id)
        return default if entry is None else entry.scaffold_text

    def best_reward(self, task_id: str) -> float:
        entry = self._entries.get(task_id)
        return 0.0 if entry is None else entry.best_reward

    def update_if_better(
        self,
        task_id: str,
        candidate_scaffolds: list[str],
        rewards: Tensor | list[float],
        *,
        step: int = 0,
    ) -> bool:
        """Update `best_scaffolds[task_id]` iff $\\max_i R_i > \\max(0, R_{\\text{cached}})$."""
        r_list = rewards.detach().cpu().tolist() if isinstance(rewards, Tensor) else list(rewards)
        if not r_list or len(r_list) != len(candidate_scaffolds):
            raise ValueError("candidate_scaffolds and rewards must have the same non-zero length")
        best_idx = max(range(len(r_list)), key=r_list.__getitem__)
        max_r = float(r_list[best_idx])
        threshold = max(0.0, self.best_reward(task_id))
        if max_r > threshold:
            self._entries[task_id] = ScaffoldEntry(
                scaffold_text=candidate_scaffolds[best_idx],
                best_reward=max_r,
                step_updated=step,
            )
            return True
        return False


def frontier_difficulty(
    pass_rate: Tensor | float,
    *,
    target_p: float = 0.2,
    sigma: float = 0.25,
) -> Tensor:
    """Compute Ornith-1.5 capability-frontier difficulty $D(p) = \\exp(-(p - p^*)^2 / (2\\sigma^2))$."""
    if sigma <= 0 or not 0.0 <= target_p <= 1.0:
        raise ValueError("target_p must be in [0, 1] and sigma > 0")
    p = torch.as_tensor(pass_rate, dtype=torch.float32)
    return torch.exp(-((p - target_p).square()) / (2.0 * sigma * sigma))


def compute_task_reward(
    validity: Tensor | float,
    pass_rate: Tensor | float,
    novelty: Tensor | float,
    *,
    target_p: float = 0.2,
    sigma: float = 0.25,
) -> Tensor:
    """Compute Ornith-1.5 Stage-1 Task Reward $R_{\\text{task}} = V(q, s) \\cdot D(p) \\cdot N(q)$.

    Enforces the hard gate $V(q, s) <= 0 \\implies R_{\\text{task}} = 0$.
    """
    v = torch.as_tensor(validity, dtype=torch.float32).clamp(0.0, 1.0)
    d = frontier_difficulty(pass_rate, target_p=target_p, sigma=sigma).to(v.device)
    n = torch.as_tensor(novelty, dtype=torch.float32, device=v.device).clamp(0.0, 2.0)
    hard_gate = (v > 0.0).float()
    return hard_gate * v * d * n


def compute_harness_reward(
    task_alignment: Tensor,
    harness_scores: Tensor,
    reference_utilities: Tensor,
    *,
    negative_harness_scores: Tensor | None = None,
    hack_resistance: Tensor | float = 1.0,
    eps: float = 1e-6,
) -> Tensor:
    """Compute Ornith-1.5 Stage-2 Harness Reward $R_{\\text{harness}} = C(q, h) \\cdot F(h, \\{\\tau_i\\}) \\cdot H(h)$."""
    c = task_alignment.float().clamp(0.0, 1.0)
    h = torch.as_tensor(hack_resistance, dtype=torch.float32, device=c.device).clamp(0.0, 1.0)

    # Reward Fidelity F(h, {tau_i}): positive correlation between harness and frozen Evaluation head
    if harness_scores.shape[-1] >= 3:
        corr = pearson_similarity(harness_scores.float(), reference_utilities.float(), eps=eps)
        fidelity = ((corr + 1.0) * 0.5).clamp(0.0, 1.0)
    else:
        diff = (harness_scores.float() - reference_utilities.float()).abs().mean(dim=-1)
        fidelity = (1.0 - diff).clamp(0.0, 1.0)

    # Corrupted negative rollouts must score 0 in a faithful harness
    if negative_harness_scores is not None:
        neg_leak = (negative_harness_scores.float() > 0.0).any(dim=-1)
        fidelity = fidelity.masked_fill(neg_leak, 0.0)

    return c * fidelity * h


def group_normalized_advantages(rewards: Tensor, eps: float = 1e-6) -> Tensor:
    """Compute group-normalized GRPO advantages $A_i = (R_i - \\mu_R) / (\\sigma_R + \\epsilon)$."""
    r = rewards.float()
    if r.shape[-1] <= 1:
        return torch.zeros_like(r)
    mean = r.mean(dim=-1, keepdim=True)
    std = r.std(dim=-1, unbiased=False, keepdim=True)
    return torch.where(std > eps, (r - mean) / (std + eps), torch.zeros_like(r))


def staleness_weight(
    policy_age: Tensor | int,
    *,
    k1: int = 1,
    k2: int = 4,
    decay_lambda: float = 0.5,
    topology_changed: Tensor | bool = False,
) -> Tensor:
    """Compute Ornith piecewise exponential staleness weight $w(d_t)$."""
    if k1 < 0 or k2 < k1 or decay_lambda < 0:
        raise ValueError("Require 0 <= k1 <= k2 and decay_lambda >= 0")
    age = torch.as_tensor(policy_age, dtype=torch.float32)
    topo = torch.as_tensor(topology_changed, dtype=torch.bool, device=age.device)
    fresh = age <= k1
    decaying = (age > k1) & (age <= k2)
    weights = torch.where(
        fresh,
        torch.ones_like(age),
        torch.where(
            decaying,
            torch.exp(-decay_lambda * (age - k1)),
            torch.zeros_like(age),
        ),
    )
    return weights.masked_fill(topo, 0.0)


def grpo_clipped_loss(
    current_log_probs: Tensor,
    behavior_log_probs: Tensor,
    advantages: Tensor,
    policy_age: Tensor | int = 0,
    *,
    clip_low: float = 0.2,
    clip_high: float = 0.28,
    k1: int = 1,
    k2: int = 4,
    decay_lambda: float = 0.5,
    topology_changed: Tensor | bool = False,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Compute Ornith staleness-weighted asymmetric-clipped GRPO loss."""
    log_ratio = (current_log_probs.float() - behavior_log_probs.float()).clamp(-20.0, 20.0)
    ratio = log_ratio.exp()
    adv = advantages.float()
    while adv.ndim < ratio.ndim:
        adv = adv.unsqueeze(-1)

    unclipped = ratio * adv
    clipped = ratio.clamp(1.0 - clip_low, 1.0 + clip_high) * adv
    surrogate = -torch.minimum(unclipped, clipped)

    w = staleness_weight(
        policy_age,
        k1=k1,
        k2=k2,
        decay_lambda=decay_lambda,
        topology_changed=topology_changed,
    ).to(surrogate.device)
    while w.ndim < surrogate.ndim:
        w = w.unsqueeze(-1)

    weighted = surrogate * w
    if valid_mask is not None:
        mask_f = valid_mask.float()
        return (weighted * mask_f).sum() / mask_f.sum().clamp_min(1.0)
    return weighted.mean()


def detect_doom_loop(
    tokens: list[int] | tuple[int, ...],
    *,
    min_period: int = 4,
    max_period: int = 128,
    min_repeats: int = 2,
    min_total_repeated: int = 10,
) -> tuple[int, int, int] | None:
    """Detect an inner repeating token/byte span (LiquidAI Antidoom `find_inner_repetition`).

    Returns `(repeat_start_idx, rejected_token_id, period)` where `repeat_start_idx = start + period`
    is the exact index of the first token of the first repeat ($y_{\\text{rejected}}$), or `None`
    if no loop is found.
    """
    n = len(tokens)
    if n < min_period * min_repeats:
        return None
    eff_max_p = min(max_period, n // min_repeats)
    for start in range(0, n - min_period * min_repeats + 1):
        for period in range(min_period, eff_max_p + 1):
            if start + period * min_repeats > n:
                break
            pattern = tokens[start : start + period]
            reps = 1
            pos = start + period
            while pos + period <= n and tokens[pos : pos + period] == pattern:
                reps += 1
                pos += period
            if reps >= min_repeats and (reps * period) >= min_total_repeated:
                repeat_start = start + period
                return repeat_start, int(tokens[repeat_start]), period
    return None


def compute_antidoom_ftpo_loss(
    logits: Tensor,
    ref_logits: Tensor,
    rejected_tokens: Tensor,
    chosen_tokens: Tensor,
    *,
    beta: float = 0.5,
    active_reg: float = 0.02,
    other_reg: float = 0.50,
    target_chosen_win: float = 0.35,
) -> tuple[Tensor, float]:
    """Compute LiquidAI Antidoom Final Token Preference Optimization (FTPO) loss in logit space.

    Key properties from LiquidAI Antidoom:
    1. **Single trigger-token targeting:** Evaluates logits `[B, V]` strictly at the token
       position immediately preceding a repetition loop (`repeat_start - 1`).
    2. **Multi-chosen alternatives:** Spreads preference pressure across $K$ plausible
       continuation tokens (`chosen_tokens` `[B, K]`) vs. the single loop-starting token
       (`rejected_tokens` `[B]`).
    3. **Two-part logit-space regularization:** Omits full-vocab softmax in the preference term
       and tightly anchors unrelated vocabulary tokens (`other_reg`) while allowing chosen/rejected
       logits (`active_reg`) to move until `chosen_win >= target_chosen_win`.
    """
    z = logits.float()
    z_ref = ref_logits.detach().float()
    bsz, vocab_size = z.shape
    rej = rejected_tokens.long().clamp(0, vocab_size - 1)
    cho = chosen_tokens.long().clamp(0, vocab_size - 1)
    if cho.ndim == 1:
        cho = cho.unsqueeze(-1)

    delta = z - z_ref
    delta_rej = delta.gather(1, rej.unsqueeze(1)).squeeze(1)
    delta_cho = delta.gather(1, cho).mean(dim=1)

    raw_rej = z.gather(1, rej.unsqueeze(1)).squeeze(1)
    raw_cho = z.gather(1, cho).mean(dim=1)
    win_vec = (raw_cho > raw_rej).float()
    chosen_win = float(win_vec.mean().item())

    # Early-stopping gate per LiquidAI Antidoom: once chosen_win exceeds target_chosen_win (~0.35),
    # attenuate preference pressure to prevent over-suppressing valid discourse words ("the", "So").
    pref_vec = -F.logsigmoid(beta * (delta_cho - delta_rej))
    gate = 0.05 if chosen_win >= target_chosen_win else 1.0
    pref_loss = gate * pref_vec.mean()

    active_mask = torch.zeros((bsz, vocab_size), dtype=torch.bool, device=z.device)
    active_mask.scatter_(1, rej.unsqueeze(1), True)
    active_mask.scatter_(1, cho, True)

    sq_delta = delta.square()
    active_l2 = (sq_delta * active_mask.float()).sum(dim=-1) / active_mask.float().sum(dim=-1).clamp_min(1.0)
    other_mask = (~active_mask).float()
    other_l2 = (sq_delta * other_mask).sum(dim=-1) / other_mask.sum(dim=-1).clamp_min(1.0)
    reg_loss = active_reg * active_l2.mean() + other_reg * other_l2.mean()

    return pref_loss + reg_loss, chosen_win



# ---------------------------------------------------------------------------
# Bounded Structural Self-Improvement (Expert Mitosis & Staleness Pruning)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpertMorphResult:
    action: str
    old_num_experts: int
    new_num_experts: int
    old_parameters: int
    new_parameters: int
    old_to_new: tuple[int | None, ...]
    parent_index: int | None = None
    removed_index: int | None = None


def _all_moe_layers(backbone: MCL4CEDBackbone) -> list[nn.Module]:
    return [*backbone.encoder_layers, *backbone.decoder_layers]


def parameter_count(model: nn.Module) -> int:
    """Count unique learned parameters in `model`."""
    return sum(p.numel() for p in model.parameters())


def projected_parameter_count(backbone: MCL4CEDBackbone, num_experts: int) -> int:
    """Exact parameter count after uniformly resizing all MoE expert banks to `num_experts`."""
    if isinstance(num_experts, bool) or not isinstance(num_experts, int) or num_experts <= 0:
        raise ValueError("num_experts must be a positive integer")
    layers = _all_moe_layers(backbone)
    slot_size = sum(
        sum(p.numel() for p in layer.moe.experts[0].parameters())
        + layer.moe.router.weight[0].numel()
        for layer in layers
    )
    return parameter_count(backbone) + (num_experts - backbone.config.num_experts) * slot_size


def _fresh_state(master: Tensor) -> dict[str, Any]:
    return {
        "master": master.detach().clone(),
        "step": 0,
        "exp_avg": torch.zeros_like(master),
        "exp_avg_sq": torch.zeros_like(master),
    }


def _install_expert_banks(
    root_model: nn.Module,
    backbone: MCL4CEDBackbone,
    optimizer: CPUOffloadAdamW,
    banks: list,
    replacements: dict,
    insertions: dict,
    states: dict,
    new_count: int,
) -> None:
    group_for = {
        parameter: index
        for index, group in enumerate(optimizer.param_groups)
        for parameter in group["params"]
    }
    groups: list[list[nn.Parameter]] = [[] for _ in optimizer.param_groups]
    for old_parameter in root_model.parameters():
        parameter = replacements.get(old_parameter, old_parameter)
        if parameter is not None:
            groups[group_for[old_parameter]].append(parameter)
        for added_parameter, parent_parameter in insertions.get(old_parameter, ()):
            groups[group_for[parent_parameter]].append(added_parameter)
    configuration = replace(backbone.config, num_experts=new_count)
    optimizer.reconfigure_parameters(groups, states)
    for bank, router, experts in banks:
        bank.router = router
        bank.experts = experts
        bank.num_experts = new_count
    backbone.config = configuration


def grow_experts(
    root_model: nn.Module,
    backbone: MCL4CEDBackbone,
    optimizer: CPUOffloadAdamW,
    *,
    parent_index: int,
    max_parameters: int = MAX_PARAMETERS,
) -> ExpertMorphResult:
    """Clone `parent_index` expert slot across all encoder and decoder layers."""
    old_count = backbone.config.num_experts
    if not 0 <= parent_index < old_count:
        raise ValueError("parent_index out of range")
    old_params = parameter_count(root_model)
    delta = projected_parameter_count(backbone, old_count + 1) - parameter_count(backbone)
    new_params = old_params + delta
    if new_params > max_parameters:
        raise ValueError(f"Expert growth would exceed max_parameters={max_parameters}")
    optimizer.invalidate_transfer_buffers()

    banks, replacements, insertions = [], {}, {}
    states = dict(optimizer.state)
    for layer in _all_moe_layers(backbone):
        bank = layer.moe
        parent = bank.experts[parent_index]
        added = deepcopy(parent)
        additions = []
        for parent_param, param in zip(parent.parameters(), added.parameters()):
            states[param] = _fresh_state(states[parent_param]["master"])
            additions.append((param, parent_param))
        insertions[list(bank.experts[-1].parameters())[-1]] = additions

        original = bank.router.weight
        router = deepcopy(bank.router)
        router.out_features = old_count + 1
        router.weight = nn.Parameter(
            torch.cat((original.detach(), original.detach()[parent_index : parent_index + 1])),
            requires_grad=original.requires_grad,
        )
        old_state = states.pop(original)
        state = {**old_state}
        for key in ("master", "exp_avg", "exp_avg_sq"):
            old_rows = old_state[key]
            added_row = (
                old_rows[parent_index : parent_index + 1]
                if key == "master"
                else torch.zeros_like(old_rows[parent_index : parent_index + 1])
            )
            state[key] = torch.cat((old_rows, added_row))
        states[router.weight] = state
        replacements[original] = router.weight
        new_experts = nn.ModuleList([*bank.experts, added])
        new_experts.training = bank.experts.training
        banks.append((bank, router, new_experts))

    _install_expert_banks(
        root_model, backbone, optimizer, banks, replacements, insertions, states, old_count + 1
    )
    return ExpertMorphResult(
        action="grow",
        old_num_experts=old_count,
        new_num_experts=old_count + 1,
        old_parameters=old_params,
        new_parameters=new_params,
        old_to_new=tuple(range(old_count)),
        parent_index=parent_index,
    )


def prune_experts(
    root_model: nn.Module,
    backbone: MCL4CEDBackbone,
    optimizer: CPUOffloadAdamW,
    *,
    expert_index: int,
) -> ExpertMorphResult:
    """Remove `expert_index` slot across all encoder and decoder layers."""
    old_count = backbone.config.num_experts
    if not 0 <= expert_index < old_count:
        raise ValueError("expert_index out of range")
    if old_count - 1 < max(backbone.config.top_k, 2):
        raise ValueError("Pruning must retain at least top_k and two experts")
    old_params = parameter_count(root_model)
    delta = parameter_count(backbone) - projected_parameter_count(backbone, old_count - 1)
    new_params = old_params - delta
    optimizer.invalidate_transfer_buffers()

    banks, replacements = [], {}
    states = dict(optimizer.state)
    keep = [i for i in range(old_count) if i != expert_index]
    for layer in _all_moe_layers(backbone):
        bank = layer.moe
        for param in bank.experts[expert_index].parameters():
            replacements[param] = None
            states.pop(param)
        original = bank.router.weight
        router = deepcopy(bank.router)
        router.out_features = old_count - 1
        router.weight = nn.Parameter(
            original.detach()[keep].clone(),
            requires_grad=original.requires_grad,
        )
        old_state = states.pop(original)
        states[router.weight] = {
            **old_state,
            **{key: old_state[key][keep].clone() for key in ("master", "exp_avg", "exp_avg_sq")},
        }
        replacements[original] = router.weight
        new_experts = nn.ModuleList([bank.experts[i] for i in keep])
        new_experts.training = bank.experts.training
        banks.append((bank, router, new_experts))

    _install_expert_banks(
        root_model, backbone, optimizer, banks, replacements, {}, states, old_count - 1
    )
    old_to_new = tuple(
        None if i == expert_index else (i if i < expert_index else i - 1) for i in range(old_count)
    )
    return ExpertMorphResult(
        action="prune",
        old_num_experts=old_count,
        new_num_experts=old_count - 1,
        old_parameters=old_params,
        new_parameters=new_params,
        old_to_new=old_to_new,
        removed_index=expert_index,
    )


# ---------------------------------------------------------------------------
# Live Ornith-1.0 / 1.5 Self-Scaffolding & Self-Improvement Engine
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OrnithTask:
    """A verifiable Ornith-1.5 frontier task with harness specification and rollout candidates."""

    task_id: str
    domain: str
    prompt_q: str
    harness_h: str
    scaffold_s: str
    candidate_rollouts: tuple[str, ...]
    harness_scores: tuple[float, ...]
    gold_index: int
    gold_thought: str
    gold_answer: str


@dataclass
class OrnithStepOutput:
    """Telemetry and differentiable loss from a live Ornith-1.0 / 1.5 + Dream-RSI step."""

    loss: Tensor
    grpo_loss: Tensor
    self_distill_loss: Tensor
    task_reward: float
    harness_reward: float
    rollout_reward: float
    novelty: float
    pass_rate: float
    frontier_difficulty: float
    veto_rate: float
    scaffolds_cached: int
    novelty_buffer_size: int = 0
    dream_cycle: int = 0
    dream_replay_score: float = 0.0
    dream_initial_score: float = 0.0
    dream_accepted_revisions: int = 0
    dream_replays_dreamed: int = 0
    dream_worlds_archived: int = 0
    dream_active_policy: str = ""
    antidoom_loss: float = 0.0
    antidoom_chosen_win: float = 0.0

    @property
    def mean_rollout_reward(self) -> float:
        return self.rollout_reward


_SCAFFOLD_CANDIDATES: tuple[str, ...] = (
    "abduction -> association -> decision -> deduction -> evaluation -> metacognition",
    "association -> deduction -> decision -> evaluation -> metacognition",
    "abduction -> deduction -> decision -> metacognition",
    "decision -> deduction -> evaluation -> metacognition",
    "abstraction -> deduction -> decision -> evaluation -> metacognition",
    "induction -> abstraction -> deduction -> decision -> metacognition",
    "analogy -> abduction -> deduction -> decision -> evaluation -> metacognition",
)

_SAFE_AST_NODES = (
    ast.Module,
    ast.FunctionDef,
    ast.arguments,
    ast.arg,
    ast.Return,
    ast.Assert,
    ast.Expr,
    ast.Assign,
    ast.AnnAssign,
    ast.AugAssign,
    ast.If,
    ast.For,
    ast.While,
    ast.Pass,
    ast.Break,
    ast.Continue,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.IfExp,
    ast.Call,
    ast.Attribute,
    ast.Name,
    ast.Constant,
    ast.Subscript,
    ast.Slice,
    ast.Tuple,
    ast.List,
    ast.Dict,
    ast.Set,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.comprehension,
    ast.Load,
    ast.Store,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.BitXor,
    ast.BitAnd,
    ast.BitOr,
    ast.LShift,
    ast.RShift,
    ast.Invert,
    ast.USub,
    ast.UAdd,
    ast.Not,
    ast.And,
    ast.Or,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.Is,
    ast.IsNot,
)
_SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "chr": chr,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "ord": ord,
    "pow": pow,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}
_SAFE_METHOD_ATTRS: frozenset[str] = frozenset(
    {
        "append",
        "extend",
        "pop",
        "insert",
        "remove",
        "sort",
        "reverse",
        "copy",
        "clear",
        "index",
        "count",
        "get",
        "keys",
        "values",
        "items",
        "update",
        "setdefault",
        "add",
        "discard",
        "union",
        "intersection",
        "difference",
        "split",
        "splitlines",
        "join",
        "strip",
        "lstrip",
        "rstrip",
        "replace",
        "lower",
        "upper",
        "isdigit",
        "isalpha",
        "isalnum",
        "startswith",
        "endswith",
        "find",
    }
)
_HACK_STUB_MARKERS: tuple[str, ...] = (
    "Constant stub",
    "Default fallback",
    "Zero delta",
    "Return base",
    "Mask overflow",
    "Uninitialized pointer",
    "Empty join",
    "Deadlock immediately",
    "#T$# 0\n",
)


def _verify_thought_equations(t_body: str) -> float:
    """Check explicit integer `A op B = C` equations inside `#T$#`; return `0.25` if any arithmetic equation is false."""
    import re

    if "mod" in t_body.lower() or "%" in t_body or "^" in t_body:
        return 1.0
    # Match isolated `A op B = C` where A, B, C are not part of a chained expression like `A + B * C = D`
    pattern = re.compile(r"(?<![\w*+\-/])(\d+)\s*([+\-*])\s*(\d+)\s*=\s*(-?\d+)(?!\s*[*+\-/^])")
    for m in pattern.finditer(t_body):
        lhs_a = int(m.group(1))
        op = m.group(2)
        lhs_b = int(m.group(3))
        rhs_c = int(m.group(4))
        expected = (
            lhs_a + lhs_b
            if op == "+"
            else (lhs_a - lhs_b if op == "-" else lhs_a * lhs_b)
        )
        if expected != rhs_c:
            return 0.25
    return 1.0


def _verify_ifeval_constraints(a_body: str, spec: str, gold_answer: str = "") -> float:
    """Deterministically verify semicolon-separated `ifeval:` instruction-following & math constraints."""
    import json
    import re

    rules = [r.strip() for r in spec[len("ifeval:") :].split(";") if r.strip()]
    if not rules:
        return 1.0 if (gold_answer and a_body.strip() == gold_answer.strip()) else 0.0

    passed = 0
    for rule in rules:
        if rule == "lowercase":
            if a_body == a_body.lower():
                passed += 1
        elif rule == "uppercase":
            if a_body == a_body.upper():
                passed += 1
        elif rule == "no_comma":
            if "," not in a_body:
                passed += 1
        elif rule == "json_valid":
            try:
                json.loads(a_body)
                passed += 1
            except Exception:
                pass
        elif rule.startswith("min_words:"):
            n_w = int(rule.split(":", 1)[1])
            if len(a_body.split()) >= n_w:
                passed += 1
        elif rule.startswith("max_words:"):
            n_w = int(rule.split(":", 1)[1])
            if 0 < len(a_body.split()) <= n_w:
                passed += 1
        elif rule.startswith("keyword:"):
            kw = rule.split(":", 1)[1].strip().lower()
            if kw and kw in a_body.lower():
                passed += 1
        elif rule.startswith("forbidden:"):
            fw = rule.split(":", 1)[1].strip().lower()
            if fw and fw not in a_body.lower():
                passed += 1
        elif rule.startswith("exact_int:"):
            target_int = int(rule.split(":", 1)[1].strip())
            nums = [int(x) for x in re.findall(r"-?\d+", a_body)]
            if nums and target_int in nums:
                passed += 1
        elif rule.startswith("exact_answer:"):
            target_str = rule.split(":", 1)[1].strip()
            if target_str and target_str.lower() in a_body.lower():
                passed += 1
    return float(passed) / float(len(rules))


def verify_python_rollout(
    trace_text: str,
    harness_h: str,
    gold_answer: str = "",
) -> float:
    """Deterministically verify a `#T$# ... #A$#` rollout against Python assertion harness `harness_h`.

    Enforces:
    1. Mandatory non-empty `#T$#` scratchpad before `#A$#` (returns `0.0` if missing).
    2. Rejection of boundary/constant hack stubs (`0.0`).
    3. Inline `#T$#` arithmetic equation consistency check (`_verify_thought_equations`).
    4. Multi-line AST-validated compiler, single-function purity check (`FunctionDef` + `Return`),
       and sandboxed execution of `def solve(...)` + `assert ...`, relational invariant assertions,
       or `ifeval:` constraint verification.
    """
    import re

    if "#T$#" not in trace_text or "#A$#" not in trace_text:
        return 0.0
    for marker in _HACK_STUB_MARKERS:
        if marker in trace_text:
            return 0.0

    t_part, a_part = trace_text.split("#A$#", 1)
    t_body = t_part.replace("#T$#", "", 1).strip()
    a_body = a_part.strip()
    if not t_body or not a_body:
        return 0.0

    thought_consistency = _verify_thought_equations(t_body)
    harness_clean = harness_h.strip()

    # Case 0: Instruction-following / mixed-constraint verifier (`ifeval:...`)
    if harness_clean.startswith("ifeval:"):
        if_score = _verify_ifeval_constraints(a_body, harness_clean, gold_answer)
        return round(if_score * thought_consistency, 4)

    # Case 1: Executable single-line or multi-line indented Python function + `assert solve(...) == ...`
    if "def " in a_body and harness_clean.startswith("assert "):
        clean_lines = [
            ln.split("#")[0].rstrip()
            for ln in a_body.splitlines()
            if ln.split("#")[0].strip() and not ln.strip().startswith("```")
        ]
        code_src = "\n".join(clean_lines)
        program_src = f"{code_src}\n{harness_clean}\n"
        try:
            code_tree = ast.parse(code_src, mode="exec")
            func_defs = [n for n in code_tree.body if isinstance(n, ast.FunctionDef)]
            has_return = any(isinstance(n, ast.Return) and n.value is not None for n in ast.walk(code_tree))
            if not func_defs or not has_return:
                return 0.0
            # Reject duplicate/shadowing function definitions (`def solve` defined twice in #A$#)
            if len({f.name for f in func_defs}) != len(func_defs):
                return 0.0
            # Require top-level statements in #A$# to be FunctionDef or constant Assign
            if not all(isinstance(stmt, (ast.FunctionDef, ast.Assign)) for stmt in code_tree.body):
                return 0.0
            tree = ast.parse(program_src, mode="exec")
            for node in ast.walk(tree):
                if not isinstance(node, _SAFE_AST_NODES):
                    return 0.0
                if isinstance(node, ast.Attribute):
                    if node.attr.startswith("_") or node.attr not in _SAFE_METHOD_ATTRS:
                        return 0.0
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        continue
                    if isinstance(node.func, ast.Attribute) and node.func.attr in _SAFE_METHOD_ATTRS:
                        continue
                    return 0.0
            tick_stmt = ast.Expr(
                value=ast.Call(func=ast.Name(id="_tick", ctx=ast.Load()), args=[], keywords=[])
            )
            for node in ast.walk(tree):
                if isinstance(node, (ast.While, ast.For)):
                    node.body.insert(0, tick_stmt)
            ast.fix_missing_locations(tree)
            steps = [0]

            def _tick() -> None:
                steps[0] += 1
                if steps[0] > 4096:
                    raise RuntimeError("Loop iteration limit exceeded")

            env: dict[str, Any] = {"__builtins__": dict(_SAFE_BUILTINS), "_tick": _tick}
            exec(compile(tree, "<ornith_verifier>", "exec"), env, env)
            return 1.0 * thought_consistency
        except AssertionError:
            # Compiles and passes AST lint checks (`FunctionDef` + `Return`), but failed unit test assertion
            base_partial = 0.25 if any(op in t_body for op in ("*", "+", "-", "if", "for", "return")) else 0.15
            return base_partial * thought_consistency
        except Exception:
            return 0.0

    # Case 2: Algebraic / invariant assertion expression (`expected == 30`, `dp[3] == 19`, etc.)
    nums = [int(m) for m in re.findall(r"-?\d+", a_body)]
    if nums:
        first_num = nums[0]
        if harness_clean.lstrip("-").isdigit():
            if first_num == int(harness_clean):
                return 1.0 * thought_consistency
            return (0.15 if any(op in t_body for op in ("*", "+", "-")) else 0.0) * thought_consistency

        env_expr: dict[str, Any] = {
            "__builtins__": dict(_SAFE_BUILTINS),
            "expected": first_num,
            "delta": first_num,
            "y": first_num,
            "mid": first_num,
            "max_rows": first_num,
            "acquisitions": first_num,
            "cycles": nums[1] if len(nums) > 1 else 0,
            "dp": defaultdict_int(first_num),
        }
        try:
            tree_expr = ast.parse(harness_clean, mode="eval")
            for node in ast.walk(tree_expr):
                if not isinstance(node, (*_SAFE_AST_NODES, ast.Expression)):
                    return 0.0
            passed = bool(eval(compile(tree_expr, "<ornith_expr>", "eval"), env_expr, env_expr))
            if passed:
                return 1.0 * thought_consistency
        except Exception:
            pass

    if gold_answer and a_body == gold_answer.strip():
        return 1.0 * thought_consistency
    return (0.15 if any(op in t_body for op in ("*", "+", "-", "=")) else 0.0) * thought_consistency


class defaultdict_int(dict):
    """Helper mapping any index lookup `dp[k]` to the candidate scalar value during verification."""

    def __init__(self, val: int) -> None:
        super().__init__()
        self._val = val

    def __getitem__(self, key: Any) -> int:
        return self._val


class OrnithEnvArchitect:
    """Self-scaffolding curriculum & cross-template compositional environment architect for Ornith-2.0.

    Monitors empirical pass rate $\\hat{p}_k$, deduction alignment $C_k$, and frontier difficulty
    $D(\\hat{p}_k; p^*=0.25)$ across all 24 procedural/compositional families (`0..19` primitive,
    `20..23` cross-template compositional) and dynamically promotes `difficulty_tier` ($1 \\dots 4$)
    and cross-template composition probability whenever a family's pass rate exceeds `0.78`.
    """

    NUM_PRIMITIVE_FAMILIES: int = 20
    NUM_COMPOSITIONAL_FAMILIES: int = 4
    TOTAL_FAMILIES: int = 24

    def __init__(self, target_pass_rate: float = 0.25, ema_beta: float = 0.70) -> None:
        self.target_pass_rate = float(target_pass_rate)
        self.ema_beta = float(ema_beta)
        self.family_pass_ema: list[float] = [0.50 for _ in range(self.TOTAL_FAMILIES)]
        self.family_tiers: list[int] = [1 for _ in range(self.TOTAL_FAMILIES)]
        self.family_counts: list[int] = [0 for _ in range(self.TOTAL_FAMILIES)]
        self.compositional_boost: float = 0.20

    def difficulty_tier(self, kind: int) -> int:
        return int(self.family_tiers[kind % self.TOTAL_FAMILIES])

    def select_family(self, idx: int, seed: int = 42) -> int:
        """Select task family `0..23`, boosting compositional families (`20..23`) as primitive families saturate."""
        base_slot = idx % self.TOTAL_FAMILIES
        # Every 3rd step if primitive pass rates are high, route to a cross-template compositional family (20..23)
        mean_prim_pass = sum(self.family_pass_ema[: self.NUM_PRIMITIVE_FAMILIES]) / float(self.NUM_PRIMITIVE_FAMILIES)
        if mean_prim_pass > 0.72 and (idx % 3 == 2) and base_slot < self.NUM_PRIMITIVE_FAMILIES:
            return self.NUM_PRIMITIVE_FAMILIES + ((idx + seed) % self.NUM_COMPOSITIONAL_FAMILIES)
        return base_slot

    def record_outcome(self, kind: int, pass_prob: float, alignment: float = 1.0) -> None:
        """Update EMA pass rate for `kind` and promote `difficulty_tier` (`1 -> 8`) when saturated."""
        k = kind % self.TOTAL_FAMILIES
        self.family_counts[k] += 1
        ema = self.ema_beta * self.family_pass_ema[k] + (1.0 - self.ema_beta) * float(pass_prob)
        self.family_pass_ema[k] = ema
        if ema > 0.78 and self.family_tiers[k] < 8 and self.family_counts[k] >= 2:
            self.family_tiers[k] += 1
            # Soft-reset EMA toward frontier when tier increases
            self.family_pass_ema[k] = 0.55
        elif ema < 0.18 and self.family_tiers[k] > 1 and self.family_counts[k] >= 4:
            self.family_tiers[k] -= 1
            self.family_pass_ema[k] = 0.45


class OrnithSelfImprover:
    """Executes the closed-loop Ornith-1.0/2.0 Self-Scaffolding, Self-Improvement, and Dream-RSI cycle.

    Combines:
    - Stage 0 (`Dream-RSI` Evolving Worlds): Expands an online discovery tree $\\mathcal{T}_t$
      using active exploration policy $\\pi_t$, appends $\\mathcal{T}_t$ to the replay
      simulator pool $\\mathcal{H}_t$, and dreams $M$ candidate policy revisions offline at
      zero execution cost ($\\mathcal{H}_t \\xrightarrow{\\text{replay}} \\pi_{t+1}$).
    - Stage 1 (`#Q$#` Proposer & `OrnithEnvArchitect`): Computes `<0.1ms` `Association` Novelty $N(q)$
      against `NoveltyTaskBuffer`, capability-frontier difficulty $D(p; p^*=0.20)$ for $R_{\\text{task}}$,
      and dynamically composes cross-template RL environments as primitive templates saturate.
    - Stage 2 (`#H$# / #S$#` Architect): Scores Task Alignment $C(q, h)$ via `Deduction` and
      Reward Fidelity $F(h, \\{\\tau_i\\})$ via `Evaluation` (`compute_harness_reward`), while
      updating `ScaffoldCache` monotonically.
    - Stage 3 (`#T$# -> #A$#` Solver): Updates policy weights via staleness-weighted
      `grpo_clipped_loss` over group-normalized rollout advantages + `0x8000` masked
      self-distillation + `MorphCode-16K` first-divergence `Antidoom FTPO`.
    """

    def __init__(
        self,
        projection_dim: int = 32,
        *,
        max_seq_len: int = 384,
        device: str | torch.device = "cpu",
        seed: int = 42,
        enable_dream_rsi: bool = False,
        dream_every: int = 1,
        dream_engine: Any | None = None,
        external_pool_path: str | Any | None = None,
        external_tasks: list[OrnithTask] | None = None,
    ) -> None:
        from pathlib import Path
        from .data import ByteTokenizer
        from .dream_rsi import DreamRSIEngine

        self.tokenizer = ByteTokenizer()
        self.max_seq_len = max_seq_len
        self.device = torch.device(device)
        self.novelty_buffer = NoveltyTaskBuffer(
            projection_dim=projection_dim,
            max_capacity=50_000,
            device=self.device,
        )
        self.scaffold_cache = ScaffoldCache()
        self.env_architect = OrnithEnvArchitect()
        self._step_counter = 0
        self._rng_seed = seed
        self.enable_dream_rsi = enable_dream_rsi or (dream_engine is not None)
        self.dream_every = max(1, int(dream_every))
        self.dream_engine: DreamRSIEngine | None = dream_engine or (
            DreamRSIEngine(seed=seed) if self.enable_dream_rsi else None
        )
        self._last_dream_out: Any | None = None
        self.external_tasks: list[OrnithTask] = list(external_tasks) if external_tasks else []
        self._last_task_kinds: dict[str, int] = {}
        if external_pool_path is not None and not self.external_tasks:
            p = Path(external_pool_path)
            if p.is_file():
                try:
                    loaded = torch.load(p, map_location="cpu", weights_only=False)
                    if isinstance(loaded, list):
                        self.external_tasks = [t for t in loaded if isinstance(t, OrnithTask)]
                    elif isinstance(loaded, dict) and "tasks" in loaded:
                        self.external_tasks = [
                            t if isinstance(t, OrnithTask) else OrnithTask(**t)
                            for t in loaded["tasks"]
                        ]
                except Exception:
                    self.external_tasks = []

    def _synthesize_combinatorial_pipeline_task(self, idx: int, *, tier: int = 1) -> OrnithTask:
        """Synthesize a randomized 3-stage cross-template pipeline ($8 \\times 6 \\times 4 = 192$ combinations)
        with length-matched adversarial multi-stage distractors (Stage-1 vs Stage-2 error propagation).
        """
        scale = max(1, int(tier))
        a = (3 * scale) + ((idx * 7 + self._rng_seed) % (17 * scale + 3))
        b = (2 * scale) + ((idx * 11 + self._rng_seed) % (13 * scale + 2))
        c = 2 + ((idx * 5 + scale) % (9 + 2 * scale))

        prod_id = (idx * 3 + self._rng_seed) % 8
        trans_id = (idx * 5 + self._rng_seed // 3) % 6
        term_id = (idx * 7 + self._rng_seed // 7) % 4

        # --- Stage 1 Producer (8 families -> outputs v1, with plausible Stage-1 error v1_err) ---
        if prod_id == 0:
            v1 = (b % 9) + 2 * scale
            rhs = a * v1 + c
            v1_err = v1 + 2
            s1_prompt = f"Stage 1 (Algebra): solve `{a} * v1 + {c} = {rhs}` for `v1`."
            s1_gold_t = f"Stage 1: subtract {rhs} - {c} = {a * v1}, then divide {a * v1} // {a} to get v1 = {v1}."
            s1_err_t = f"Stage 1: add offset to get ({rhs} + {c}) // {a} so v1 = {v1_err}."
        elif prod_id == 1:
            v0 = (a % 12) + 2
            acc = 2 * ((b % 5) + 1)
            t_s = (c % 4) + 2
            v1 = v0 * t_s + (acc * t_s * t_s) // 2
            v1_err = v0 * t_s + acc * t_s * t_s
            s1_prompt = f"Stage 1 (Kinematics): compute displacement `v1 = {v0}*{t_s} + 0.5*{acc}*{t_s}^2`."
            s1_gold_t = f"Stage 1: drift {v0} * {t_s} = {v0 * t_s}, accel term = {(acc * t_s * t_s) // 2}, sum v1 = {v1}."
            s1_err_t = f"Stage 1: drift {v0} * {t_s} = {v0 * t_s}, unhalved accel term = {acc * t_s * t_s}, sum v1 = {v1_err}."
        elif prod_id == 2:
            v1 = a + b + c
            v1_err = a + c
            s1_prompt = f"Stage 1 (DAG Critical Path): M1 ({a}d) -> parallel Alpha ({b}d), Beta (0d) -> M2 ({c}d); set `v1` to critical path."
            s1_gold_t = f"Stage 1: longest chain M1 + Alpha + M2 gives {a} + {b} = {a + b}, plus {c} gives v1 = {v1}."
            s1_err_t = f"Stage 1: shortest chain M1 + Beta + M2 gives {a} + {c} = {v1_err} so v1 = {v1_err}."
        elif prod_id == 3:
            c_mat = (c % 7) + 4
            b_mat = (b % 5) + 1
            v1 = a * c_mat - 2 * b_mat
            v1_err = a * c_mat + 2 * b_mat
            s1_prompt = f"Stage 1 (Matrix Det): compute `v1 = det([[{a}, {b_mat}], [2, {c_mat}]])`."
            s1_gold_t = f"Stage 1: main diagonal {a} * {c_mat} = {a * c_mat}, off-diagonal 2 * {b_mat} = {2 * b_mat}, det v1 = {v1}."
            s1_err_t = f"Stage 1: main diagonal {a} * {c_mat} = {a * c_mat}, plus off-diagonal {2 * b_mat} gives v1 = {v1_err}."
        elif prod_id == 4:
            cp = (b % 4) + 2
            dt = (c % 5) + 2
            v1 = a * cp * dt
            v1_err = a * (cp + dt)
            s1_prompt = f"Stage 1 (Calorimetry): compute thermal energy `v1 = {a} * {cp} * {dt}`."
            s1_gold_t = f"Stage 1: heat capacity {a} * {cp} = {a * cp}, times dT {dt} gives v1 = {v1}."
            s1_err_t = f"Stage 1: sum specific heat and dT ({cp} + {dt}) times mass {a} gives v1 = {v1_err}."
        elif prod_id == 5:
            steps_k = (c % 6) + 2
            v1 = a + steps_k * b
            v1_err = a + (steps_k - 1) * b
            s1_prompt = f"Stage 1 (Recurrence): `dp[0] = {a}`, `dp[k] = dp[k-1] + {b}`; evaluate `v1 = dp[{steps_k}]`."
            s1_gold_t = f"Stage 1: unroll {steps_k} steps: {steps_k} * {b} = {steps_k * b}, plus base {a} gives v1 = {v1}."
            s1_err_t = f"Stage 1: unroll {steps_k - 1} steps: {(steps_k - 1) * b}, plus base {a} gives v1 = {v1_err}."
        elif prod_id == 6:
            fan = (c % 5) + 2
            v1 = min(a, b) * fan
            v1_err = max(a, b) * fan
            s1_prompt = f"Stage 1 (Inner Join): Table R ({a} keys) joins Table S ({b} keys) with fanout {fan}; set `v1` to max rows."
            s1_gold_t = f"Stage 1: matched keys min({a}, {b}) = {min(a, b)}, times fanout {fan} gives v1 = {v1}."
            s1_err_t = f"Stage 1: outer bound max({a}, {b}) = {max(a, b)}, times fanout {fan} gives v1 = {v1_err}."
        else:
            v1 = a * c
            v1_err = (a + b) * c
            s1_prompt = f"Stage 1 (Service Graph): {a} active workers launch {c} kernels each ({b} idle queues); set `v1` to active kernels."
            s1_gold_t = f"Stage 1: multiply active workers {a} * {c} = {v1} (ignoring {b} idle queues) so v1 = {v1}."
            s1_err_t = f"Stage 1: include idle queues ({a} + {b}) * {c} = {v1_err} so v1 = {v1_err}."

        # --- Stage 2 Transformer (6 families taking v1 -> v2) ---
        if trans_id == 0:
            p_ring = 29 + (idx % 11) * 2
            v2 = (v1 + c) % p_ring
            v2_s2err = (v1 + c + 3) % p_ring
            v2_s1err = (v1_err + c) % p_ring
            if v2_s1err == v2:
                v2_s1err = (v2 + 5) % p_ring
            s2_prompt = f"Stage 2 (ModRing): compute `v2 = (v1 + {c}) mod {p_ring}`."
            s2_gold_t = f"Stage 2: ({v1} + {c}) mod {p_ring} = {v2}."
            s2_err_t = f"Stage 2: shifted residue ({v1} + {c} + 3) mod {p_ring} = {v2_s2err}."
            s2_from_s1err_t = f"Stage 2: ({v1_err} + {c}) mod {p_ring} = {v2_s1err}."
        elif trans_id == 1:
            v2 = v1 + c
            v2_s2err = v1 - c
            v2_s1err = v1_err + c
            s2_prompt = f"Stage 2 (XOR Involution): simplify `v2 = ((v1 ^ {b}) ^ {b}) + {c}`."
            s2_gold_t = f"Stage 2: XOR involution cancels mask {b}, leaving {v1} + {c} = {v2}."
            s2_err_t = f"Stage 2: subtract offset after XOR cancellation, leaving {v1} - {c} = {v2_s2err}."
            s2_from_s1err_t = f"Stage 2: XOR involution cancels mask {b}, leaving {v1_err} + {c} = {v2_s1err}."
        elif trans_id == 2:
            g1 = v1 + b
            v2 = g1 + c
            v2_s2err = g1 - c
            v2_s1err = v1_err + b + c
            s2_prompt = f"Stage 2 (Horn Chain): derive `G1 = v1 + {b}` and `v2 = G1 + {c}`."
            s2_gold_t = f"Stage 2: forward chain G1 = {v1} + {b} = {g1}, then v2 = {g1} + {c} = {v2}."
            s2_err_t = f"Stage 2: forward chain G1 = {v1} + {b} = {g1}, then subtract {c} to get v2 = {v2_s2err}."
            s2_from_s1err_t = f"Stage 2: forward chain G1 = {v1_err} + {b} = {v1_err + b}, then v2 = {v2_s1err}."
        elif trans_id == 3:
            v2 = v1 + b
            v2_s2err = v1 + 2 * b
            v2_s1err = v1_err + b
            s2_prompt = f"Stage 2 (Binary Search Mid): window `lo = v1`, `hi = v1 + {2 * b}`; set `v2 = lo + (hi - lo) // 2`."
            s2_gold_t = f"Stage 2: half-span is {b}, so midpoint v2 = {v1} + {b} = {v2}."
            s2_err_t = f"Stage 2: use full span {2 * b}, so endpoint v2 = {v1} + {2 * b} = {v2_s2err}."
            s2_from_s1err_t = f"Stage 2: half-span is {b}, so midpoint v2 = {v1_err} + {b} = {v2_s1err}."
        elif trans_id == 4:
            delta_inv = 2 * b
            v2 = v1 + delta_inv
            v2_s2err = v1 + b
            v2_s1err = v1_err + delta_inv
            s2_prompt = f"Stage 2 (Loop Invariant): add invariant delta `((i + {b}) * 2 - i * 2)` to `v1` to get `v2`."
            s2_gold_t = f"Stage 2: invariant delta is {delta_inv}, so v2 = {v1} + {delta_inv} = {v2}."
            s2_err_t = f"Stage 2: unscaled delta is {b}, so v2 = {v1} + {b} = {v2_s2err}."
            s2_from_s1err_t = f"Stage 2: invariant delta is {delta_inv}, so v2 = {v1_err} + {delta_inv} = {v2_s1err}."
        else:
            v2 = v1 * 2 + c
            v2_s2err = v1 * 2 - c
            v2_s1err = v1_err * 2 + c
            s2_prompt = f"Stage 2 (Affine Transform): evaluate `v2 = 2 * v1 + {c}`."
            s2_gold_t = f"Stage 2: double {v1} * 2 = {v1 * 2}, then add {c} to get v2 = {v2}."
            s2_err_t = f"Stage 2: double {v1} * 2 = {v1 * 2}, then subtract {c} to get v2 = {v2_s2err}."
            s2_from_s1err_t = f"Stage 2: double {v1_err} * 2 = {v1_err * 2}, then add {c} to get v2 = {v2_s1err}."

        # --- Stage 3 Terminal Modality (4 verifiable output harnesses) ---
        task_id = f"comb_t{tier}_p{prod_id}_tr{trans_id}_m{term_id}_{idx}"
        if term_id == 0:
            # Terminal 0: Executable multi-assert Python function parameterized by v2
            n_test = (c % 4) + 2
            expected_n = n_test * v2 + v1
            expected_0 = v1
            prompt_q = (
                f"#U$# [Combinatorial Pipeline P{prod_id}->T{trans_id}->Code] {s1_prompt} {s2_prompt} "
                f"Stage 3: Implement `solve(n)` in Python returning `n * v2 + v1` using your exact integers `v1` and `v2`.\n"
                f"#Q$# [code] Derive `v1, v2` in `#T$#` and implement `solve(n)` in `#A$#`."
            )
            harness_h = f"assert solve({n_test}) == {expected_n} and solve(0) == {expected_0}"
            gold_t = f"{s1_gold_t} {s2_gold_t} Stage 3: implement solve(n) returning n * {v2} + {v1}."
            d1_t = f"{s1_gold_t} {s2_err_t} Stage 3: implement solve(n) returning n * {v2_s2err} + {v1}."
            d2_t = f"{s1_err_t} {s2_from_s1err_t} Stage 3: implement solve(n) returning n * {v2_s1err} + {v1_err}."
            gold_a = f"def solve(n):\n    return n * {v2} + {v1}"
            d1_a = f"def solve(n):\n    return n * {v2_s2err} + {v1}"
            d2_a = f"def solve(n):\n    return n * {v2_s1err} + {v1_err}"
            hack_a = "def solve(n):\n    return 0"
            domain = "compositional_code_math"
        elif term_id == 1:
            # Terminal 1: Multi-constraint IFEval + exact integer v2
            prompt_q = (
                f"#U$# [Combinatorial Pipeline P{prod_id}->T{trans_id}->IFEval] {s1_prompt} {s2_prompt} "
                f"Stage 3: State `v2` in lowercase with NO commas, at least 6 words, including the keyword `verified`.\n"
                f"#Q$# [evaluation] Satisfy both reasoning stages and all formatting constraints."
            )
            harness_h = f"ifeval:lowercase;no_comma;min_words:6;keyword:verified;exact_int:{v2}"
            gold_t = f"{s1_gold_t} {s2_gold_t} Stage 3: format lowercase without commas and include keyword verified."
            d1_t = f"{s1_gold_t} {s2_err_t} Stage 3: format lowercase without commas and include keyword verified."
            d2_t = f"{s1_err_t} {s2_from_s1err_t} Stage 3: format lowercase without commas and include keyword verified."
            gold_a = f"the verified pipeline output value is {v2} units"
            d1_a = f"the verified pipeline output value is {v2_s2err} units"
            d2_a = f"The Pipeline, Output Value Is {v2_s1err} units"
            hack_a = "0"
            domain = "compositional_deduction_ifeval"
        elif term_id == 2:
            # Terminal 2: Dual-stage invariant verification (both v1 and v2 must match)
            prompt_q = (
                f"#U$# [Combinatorial Pipeline P{prod_id}->T{trans_id}->DualInvariant] {s1_prompt} {s2_prompt}\n"
                f"#Q$# [deduction] Report both `v1` and `v2` as `v1=<int> v2=<int>`."
            )
            harness_h = f"acquisitions == {v1} and cycles == {v2}"
            gold_t = f"{s1_gold_t} {s2_gold_t}"
            d1_t = f"{s1_gold_t} {s2_err_t}"
            d2_t = f"{s1_err_t} {s2_from_s1err_t}"
            gold_a = f"v1={v1} v2={v2}"
            d1_a = f"v1={v1} v2={v2_s2err}"
            d2_a = f"v1={v1_err} v2={v2_s1err}"
            hack_a = "v1=0 v2=0"
            domain = "compositional_physics_math"
        else:
            # Terminal 3: Exact terminal scalar with length-matched multi-stage distractors
            prompt_q = (
                f"#U$# [Combinatorial Pipeline P{prod_id}->T{trans_id}->Scalar] {s1_prompt} {s2_prompt}\n"
                f"#Q$# [math] Derive `v1` then return the exact integer `v2`."
            )
            harness_h = f"expected == {v2}"
            gold_t = f"{s1_gold_t} {s2_gold_t}"
            d1_t = f"{s1_gold_t} {s2_err_t}"
            d2_t = f"{s1_err_t} {s2_from_s1err_t}"
            gold_a = f"{v2}"
            d1_a = f"{v2_s2err}"
            d2_a = f"{v2_s1err}"
            hack_a = "0"
            domain = "compositional_code_pm"

        base_cands = (
            f"#T$# {gold_t}\n#A$# {gold_a}",
            f"#T$# {d1_t}\n#A$# {d1_a}",
            f"#T$# {d2_t}\n#A$# {d2_a}",
            f"#T$# Default fallback.\n#A$# {hack_a}",
        )
        base_scores = (1.0, 0.20, 0.10, 0.0)
        order = [0, 1, 2, 3]
        shift = idx % 4
        order[0], order[shift] = order[shift], order[0]
        p_cands = tuple(base_cands[i] for i in order)
        p_scores = tuple(base_scores[i] for i in order)
        g_idx = order.index(0)
        fam_slot = 20 + (term_id % 4)
        self._last_task_kinds[task_id] = fam_slot
        cached_s = self.scaffold_cache.get(task_id, default=_SCAFFOLD_CANDIDATES[idx % len(_SCAFFOLD_CANDIDATES)])
        return OrnithTask(
            task_id=task_id,
            domain=domain,
            prompt_q=prompt_q,
            harness_h=harness_h,
            scaffold_s=cached_s,
            candidate_rollouts=p_cands,
            harness_scores=p_scores,
            gold_index=g_idx,
            gold_thought=gold_t,
            gold_answer=gold_a,
        )

    def _generate_task(self, idx: int, *, compact: bool = False) -> OrnithTask:
        """Synthesize a verifiable coding, math, physics, logic, IFEval, or cross-template compositional task (#Q$#)."""
        # Mix curated external HF RLVR tasks (when loaded and non-compact) with live procedural + compositional tasks
        if not compact and self.external_tasks and (idx % 2 == 1):
            ext_idx = ((idx // 2) * 131 + self._rng_seed) % len(self.external_tasks)
            ext_task = self.external_tasks[ext_idx]
            cached_s = self.scaffold_cache.get(
                ext_task.task_id,
                default=ext_task.scaffold_s or _SCAFFOLD_CANDIDATES[idx % len(_SCAFFOLD_CANDIDATES)],
            )
            return replace(ext_task, scaffold_s=cached_s)

        kind = self.env_architect.select_family(idx, seed=self._rng_seed)
        tier = self.env_architect.difficulty_tier(kind)
        # For online training steps beyond the initial 24-family sweep, dynamically invoke the 192-pipeline
        # combinatorial cross-template synthesizer on every 3rd procedural slot or when kind >= 20 and idx >= 24.
        if not compact and idx >= 24 and (kind >= 20 or idx % 3 == 0):
            return self._synthesize_combinatorial_pipeline_task(idx, tier=tier)

        scale = tier  # Scales operand magnitudes and multi-step depth across tiers 1..8
        a = (3 * scale) + ((idx * 7 + self._rng_seed) % (19 * scale))
        b = (2 * scale) + ((idx * 11 + self._rng_seed) % (13 * scale))
        c = 1 + ((idx * 5 + tier) % (9 + 2 * scale))
        shift = idx % 4

        def _permute(
            cands: tuple[str, str, str, str],
            sc: tuple[float, float, float, float],
        ) -> tuple[tuple[str, ...], tuple[float, ...], int]:
            # Place gold (originally at 0) at `shift` while keeping hack_neg at the last slot if possible
            order = [0, 1, 2, 3]
            order[0], order[shift] = order[shift], order[0]
            p_cands = tuple(cands[i] for i in order)
            p_scores = tuple(sc[i] for i in order)
            gold_pos = order.index(0)
            return p_cands, p_scores, gold_pos

        if compact:
            ans_val = a * b + c
            task_id = f"c_{a}_{b}_{c}"
            base_cands = (
                f"#T$# {a}*{b}+{c}\n#A$# {ans_val}",
                f"#T$# {a}+{b}\n#A$# {a+b+c}",
                f"#T$# {a}*{b}\n#A$# {a*b-c}",
                "#T$# 0\n#A$# 0",
            )
            base_scores = (1.0, 0.15, 0.10, 0.0)
            p_cands, p_scores, g_idx = _permute(base_cands, base_scores)
            return OrnithTask(
                task_id=task_id,
                domain="algorithmic_code",
                prompt_q=f"x={a},y={b},z={c}",
                harness_h=f"{ans_val}",
                scaffold_s="ded",
                candidate_rollouts=p_cands,
                harness_scores=p_scores,
                gold_index=g_idx,
                gold_thought=f"{a}*{b}+{c}",
                gold_answer=str(ans_val),
            )

        if kind == 0:
            ans_val = a * b + c
            task_id = f"code_linear_{a}_{b}_{c}"
            prompt_q = (
                f"#U$# Implement `solve(x, y, z) = x * y + z` and evaluate `solve({a}, {b}, {c})`.\n"
                f"#Q$# [code] Return verified implementation and result."
            )
            harness_h = f"assert solve({a}, {b}, {c}) == {ans_val}"
            gold_t = f"Multiply {a} * {b} = {a*b}, then add {c} to get {ans_val}."
            gold_a = "def solve(x, y, z):\n    res = x * y + z\n    return res"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# Add {a} + {b*c} = {a+b*c}.\n#A$# def solve(x, y, z):\n    return x + y * z",
                f"#T$# Subtract {a*b} - {c} = {a*b-c}.\n#A$# def solve(x, y, z):\n    return x * y - z",
                "#T$# Constant stub.\n#A$# def solve(x, y, z):\n    return 0",
            )
            base_scores = (1.0, 0.25, 0.20, 0.0)
            domain = "algorithmic_code"
        elif kind == 1:
            active_edges = a * c
            task_id = f"graph_hops_{a}_{b}_{c}"
            prompt_q = (
                f"#U$# Service A spawns {a} workers; each worker runs {c} kernels; "
                f"Service B allocates {b} idle queues.\n"
                f"#Q$# [deduction] How many active kernels are launched by Service A?"
            )
            harness_h = f"expected == {active_edges}"
            gold_t = f"2-hop product: {a} * {c} = {active_edges} (ignore {b} idle queues)."
            gold_a = f"{active_edges}"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# Sum entities: {a} + {b} = {a+b}, plus {c} gives {a+b+c}.\n#A$# {a+b+c}",
                f"#T$# Include idle queues: ({a}+{b})*{c}={(a+b)*c}.\n#A$# {(a+b)*c}",
                "#T$# Default fallback.\n#A$# 1",
            )
            base_scores = (1.0, 0.10, 0.20, 0.0)
            domain = "multi_hop_deduction"
        elif kind == 2:
            diff_val = b * c
            task_id = f"invariant_patch_{a}_{b}_{c}"
            prompt_q = (
                f"#U$# Given loop expressions `acc = (i + {b}) * {c}` and `base = i * {c}` at `i = {a}`.\n"
                f"#Q$# [evaluation] Compute the invariant difference `acc - base`."
            )
            harness_h = f"delta == {diff_val}"
            gold_t = f"Distribute: {b} * {c} = {diff_val}."
            gold_a = f"{diff_val}"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# Multiply index {a} * {b} = {a*b}.\n#A$# {a*b}",
                f"#T$# Add offsets {b} + {c} = {b+c}.\n#A$# {b+c}",
                "#T$# Zero delta.\n#A$# 0",
            )
            base_scores = (1.0, 0.10, 0.15, 0.0)
            domain = "invariant_verification"
        elif kind == 3:
            dp_n = a + c * b
            task_id = f"dp_step_{a}_{b}_{c}"
            prompt_q = (
                f"#U$# Recurrence `dp[0] = {a}` and `dp[k] = dp[k-1] + {b}` for `k >= 1`.\n"
                f"#Q$# [deduction] Evaluate `dp[{c}]` in closed form."
            )
            harness_h = f"dp[{c}] == {dp_n}"
            gold_t = f"Unroll {c} steps: {c} * {b} = {c*b}, then {a} + {c*b} = {dp_n}."
            gold_a = f"{dp_n}"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {dp_n}",
                f"#T$# Off-by-one step: {a} + {(c-1)*b} = {a+(c-1)*b}.\n#A$# {a+(c-1)*b}",
                f"#T$# Multiply base: {a*c} + {b} = {a*c+b}.\n#A$# {a*c+b}",
                f"#T$# Return base dp[0].\n#A$# {a}",
            )
            base_scores = (1.0, 0.20, 0.10, 0.0)
            domain = "dp_recurrence"
        elif kind == 4:
            xor_val = (a ^ b) ^ b
            task_id = f"bitmask_xor_{a}_{b}_{c}"
            prompt_q = (
                f"#U$# Bitwise pipeline computes `y = (({a} ^ {b}) ^ {b}) + {c}`.\n"
                f"#Q$# [code] Simplify the XOR involution and return `y`."
            )
            harness_h = f"y == {xor_val + c}"
            gold_t = f"By XOR involution (x ^ m) ^ m == x, ({a} ^ {b}) ^ {b} = {a}; add {a} + {c} = {a + c}."
            gold_a = f"{a + c}"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# Cancel all bits to 0 + {c} = {c}.\n#A$# {c}",
                f"#T$# Sum operands {a} + {b} = {a+b}, plus {c} gives {a+b+c}.\n#A$# {a+b+c}",
                "#T$# Mask overflow.\n#A$# -1",
            )
            base_scores = (1.0, 0.15, 0.10, 0.0)
            domain = "bitmask_verification"
        elif kind == 5:
            lo = a
            hi = a + 2 * b
            mid = lo + (hi - lo) // 2
            task_id = f"bsearch_mid_{a}_{b}_{c}"
            prompt_q = (
                f"#U$# Binary search window has `lo = {lo}` and `hi = {hi}`.\n"
                f"#Q$# [code] Compute the overflow-safe midpoint `lo + (hi - lo) // 2`."
            )
            harness_h = f"mid == {mid}"
            gold_t = f"Compute span {hi} - {lo} = {hi - lo}; half span = {b}; {lo} + {b} = {mid}."
            gold_a = f"{mid}"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# Subtract span {lo} - {b} = {lo - b}.\n#A$# {lo - b}",
                f"#T$# Use full span {lo} + {2*b} = {hi}.\n#A$# {hi}",
                "#T$# Uninitialized pointer.\n#A$# 0",
            )
            base_scores = (1.0, 0.10, 0.15, 0.0)
            domain = "binary_search"
        elif kind == 6:
            card = min(a, b) * c
            task_id = f"join_card_{a}_{b}_{c}"
            prompt_q = (
                f"#U$# Table R has {a} unique keys, Table S has {b} unique keys, and each matched key fans out {c} rows.\n"
                f"#Q$# [evaluation] What is the maximum inner-join output cardinality?"
            )
            harness_h = f"max_rows == {card}"
            gold_t = f"At most min({a}, {b}) = {min(a, b)} keys match; {min(a, b)} * {c} = {card}."
            gold_a = f"{card}"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# Cartesian product {a*b} * {c} = {a*b*c}.\n#A$# {a*b*c}",
                f"#T$# Max key bound {max(a,b)} * {c} = {max(a,b)*c}.\n#A$# {max(a,b)*c}",
                "#T$# Empty join.\n#A$# 0",
            )
            base_scores = (1.0, 0.10, 0.20, 0.0)
            domain = "relational_algebra"
        elif kind == 7:
            inversions = b * c
            task_id = f"dag_locks_{a}_{b}_{c}"
            prompt_q = (
                f"#U$# Lock hierarchy has {a} stages; {b} threads each acquire {c} ordered locks strictly increasing.\n"
                f"#Q$# [deduction] How many total lock acquisitions occur with 0 deadlock cycles?"
            )
            harness_h = f"acquisitions == {inversions} and cycles == 0"
            gold_t = f"Strict total order prevents cycles (0 deadlocks); {b} * {c} = {inversions} acquisitions."
            gold_a = f"{inversions} acquisitions, 0 cycles"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# Multiply stages {a} * {b} = {a*b} acquisitions, 1 cycle.\n#A$# {a*b} acquisitions, 1 cycle",
                f"#T$# Sum threads {b} + {c} = {b+c} acquisitions.\n#A$# {b+c} acquisitions, 0 cycles",
                f"#T$# Deadlock immediately.\n#A$# 0 acquisitions, {b} cycles",
            )
            base_scores = (1.0, 0.10, 0.15, 0.0)
            domain = "concurrency_verification"
        elif kind == 8:
            n_lim = (c % 7) + 3
            expected_sum = sum(i * a for i in range(1, n_lim + 1) if i % 2 == 1)
            task_id = f"code_odd_sum_{a}_{n_lim}"
            prompt_q = (
                f"#U$# Write a Python function `solve(n, scale)` that sums `i * scale` for all odd integers `1 <= i <= n`.\n"
                f"#Q$# [code] Implement `solve({n_lim}, {a})` with 4-space indentation."
            )
            harness_h = f"assert solve({n_lim}, {a}) == {expected_sum}"
            gold_t = (
                f"Iterate i from 1 to {n_lim}; check `if i % 2 == 1:` and accumulate `total += i * scale` "
                f"to obtain {expected_sum}."
            )
            gold_a = (
                "def solve(n, scale):\n"
                "    total = 0\n"
                "    for i in range(1, n + 1):\n"
                "        if i % 2 == 1:\n"
                "            total += i * scale\n"
                "    return total"
            )
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                "#T$# Sum all integers without odd filter.\n#A$# def solve(n, scale):\n    return sum(range(1, n + 1)) * scale",
                "#T$# Off-by-one range bound.\n#A$# def solve(n, scale):\n    return (n - 1) * scale",
                "#T$# Constant stub.\n#A$# def solve(n, scale):\n    return 0",
            )
            base_scores = (1.0, 0.25, 0.15, 0.0)
            domain = "algorithmic_code"
        elif kind == 9:
            rhs = a * b + c
            task_id = f"math_linear_eq_{a}_{b}_{c}"
            prompt_q = (
                f"#U$# Solve for `x` in the linear equation `{a}*x + {c} = {rhs}` and show each algebraic step.\n"
                f"#Q$# [math] Derive the exact integer solution `x`."
            )
            harness_h = f"expected == {b}"
            gold_t = (
                f"Step 1: Subtract {rhs} - {c} = {a*b}. "
                f"Step 2: Divide {a*b} // {a} = {b}."
            )
            gold_a = f"{b}"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# Add {c} instead of subtracting: ({rhs} + {c}) // {a}.\n#A$# {(rhs + c) // max(1, a)}",
                f"#T$# Subtract {rhs} - {a} = {rhs - a}.\n#A$# {rhs - a}",
                "#T$# Default fallback.\n#A$# 0",
            )
            base_scores = (1.0, 0.15, 0.10, 0.0)
            domain = "symbolic_math"
        elif kind == 10:
            v0 = a
            accel = 2 * b
            t_sec = (c % 6) + 2
            disp = v0 * t_sec + (accel * t_sec * t_sec) // 2
            task_id = f"physics_kinematics_{v0}_{accel}_{t_sec}"
            prompt_q = (
                f"#U$# A particle starts with initial velocity `v0 = {v0} m/s` under constant acceleration "
                f"`a = {accel} m/s^2` for `t = {t_sec} s`.\n"
                f"#Q$# [physics] Compute total displacement `d = v0*t + 0.5*a*t^2`."
            )
            harness_h = f"expected == {disp}"
            gold_t = (
                f"Drift term {v0} * {t_sec} = {v0*t_sec}; "
                f"acceleration term 0.5*{accel}*{t_sec}^2 = {(accel*t_sec*t_sec)//2}; "
                f"total {v0*t_sec} + {(accel*t_sec*t_sec)//2} = {disp}."
            )
            gold_a = f"{disp}"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# Omit factor 0.5 on acceleration term: {v0*t_sec + accel*t_sec*t_sec}.\n#A$# {v0*t_sec + accel*t_sec*t_sec}",
                f"#T$# Compute final velocity {v0} + {accel*t_sec} = {v0 + accel*t_sec}.\n#A$# {v0 + accel*t_sec}",
                "#T$# Zero delta.\n#A$# 0",
            )
            base_scores = (1.0, 0.20, 0.15, 0.0)
            domain = "physics_kinematics"
        elif kind == 11:
            path_alpha = a + b + c
            path_beta = a + c
            slack_beta = path_alpha - path_beta
            task_id = f"pm_critical_path_{a}_{b}_{c}"
            prompt_q = (
                f"#U$# A software release DAG has Milestone M1 ({a} days) feeding two parallel tracks: "
                f"Track Alpha ({b} days) and Track Beta (0 days) before final integration M2 ({c} days).\n"
                f"#Q$# [project_management] Compute the Critical Path duration and Track Beta float slack."
            )
            harness_h = f"acquisitions == {path_alpha} and cycles == {slack_beta}"
            gold_t = (
                f"Critical Path M1 -> Alpha -> M2: {a} + {b} = {a+b}, plus {c} gives {path_alpha} days. "
                f"Track Beta takes {a} + {c} = {path_beta} days, slack {path_alpha} - {path_beta} = {slack_beta} days."
            )
            gold_a = f"{path_alpha} days critical path, {slack_beta} days slack"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# Sum all nodes serially: {a+b+c+b} days critical path, 0 days slack.\n#A$# {a+b+c+b} days critical path, 0 days slack",
                f"#T$# Shortest path duration: {path_beta} days critical path, {b} days slack.\n#A$# {path_beta} days critical path, {b} days slack",
                "#T$# Deadlock immediately.\n#A$# 0 days critical path, 0 days slack",
            )
            base_scores = (1.0, 0.15, 0.10, 0.0)
            domain = "project_management"
        elif kind == 12:
            # Modular Ring Arithmetic (`(a * b + c) mod p`)
            p_mod = 17 + (idx % 13)
            mod_val = (a * b + c) % p_mod
            task_id = f"math_mod_ring_{a}_{b}_{c}_{p_mod}"
            prompt_q = (
                f"#U$# In the finite ring Z_{p_mod}, evaluate `r = ({a} * {b} + {c}) mod {p_mod}`.\n"
                f"#Q$# [math] Compute the canonical residue `0 <= r < {p_mod}`."
            )
            harness_h = f"expected == {mod_val}"
            gold_t = f"Compute {a} * {b} = {a*b}, add {a*b} + {c} = {a*b+c}, then reduce mod {p_mod} to get {mod_val}."
            gold_a = f"{mod_val}"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# Unreduced sum {a*b + c} mod {p_mod}.\n#A$# {(a + b + c) % p_mod}",
                f"#T$# Off-by-one residue {(mod_val + 1) % p_mod} mod {p_mod}.\n#A$# {(mod_val + 1) % p_mod}",
                "#T$# Default fallback.\n#A$# 0",
            )
            base_scores = (1.0, 0.15, 0.10, 0.0)
            domain = "symbolic_math"
        elif kind == 13:
            # 2x2 Matrix Determinant & Trace
            det_m = a * c - b * 2
            tr_m = a + c
            task_id = f"math_matrix_det_{a}_{b}_{c}"
            prompt_q = (
                f"#U$# Matrix M = [[{a}, {b}], [2, {c}]].\n"
                f"#Q$# [math] Compute det(M) = {a}*{c} - 2*{b} and trace(M) = {a} + {c}."
            )
            harness_h = f"acquisitions == {det_m} and cycles == {tr_m}"
            gold_t = f"Determinant: {a} * {c} = {a*c}, 2 * {b} = {2*b}, so {a*c} - {2*b} = {det_m}. Trace: {a} + {c} = {tr_m}."
            gold_a = f"det={det_m}, trace={tr_m}"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# Add diagonal products: {a*c} + {2*b} = {a*c + 2*b}.\n#A$# det={a*c + 2*b}, trace={tr_m}",
                f"#T$# Swap determinant and trace.\n#A$# det={tr_m}, trace={det_m}",
                "#T$# Zero delta.\n#A$# det=0, trace=0",
            )
            base_scores = (1.0, 0.20, 0.10, 0.0)
            domain = "symbolic_math"
        elif kind == 14:
            # Multi-point Python List Comprehension / Filter-Transform
            thresh = b % 7 + 2
            mult = c % 5 + 2
            sample_xs = [1, thresh, thresh + 2, thresh - 1, thresh + 5]
            expected_out = [x * mult for x in sample_xs if x >= thresh]
            task_id = f"code_list_comp_{thresh}_{mult}_{a}"
            prompt_q = (
                f"#U$# Write `solve(xs, t, m)` returning `[x * m for x in xs if x >= t]`.\n"
                f"#Q$# [code] Verify on `xs = {sample_xs}`, `t = {thresh}`, `m = {mult}`."
            )
            harness_h = f"assert solve({sample_xs}, {thresh}, {mult}) == {expected_out} and solve([], {thresh}, {mult}) == []"
            gold_t = f"Filter elements x >= {thresh} and multiply each by {mult} using a list comprehension."
            gold_a = "def solve(xs, t, m):\n    return [x * m for x in xs if x >= t]"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                "#T$# Strict greater-than filter x > t.\n#A$# def solve(xs, t, m):\n    return [x * m for x in xs if x > t]",
                "#T$# Add m instead of multiplying.\n#A$# def solve(xs, t, m):\n    return [x + m for x in xs if x >= t]",
                "#T$# Constant stub.\n#A$# def solve(xs, t, m):\n    return []",
            )
            base_scores = (1.0, 0.25, 0.15, 0.0)
            domain = "algorithmic_code"
        elif kind == 15:
            # Unnormalized Bayes Posterior Odds Numerator
            prior_h1 = a
            like_h1 = b
            num_post = prior_h1 * like_h1
            task_id = f"bayes_odds_{a}_{b}_{c}"
            prompt_q = (
                f"#U$# Hypothesis H1 has prior weight {prior_h1} and likelihood weight {like_h1}; "
                f"distractor H0 has weight {c}.\n"
                f"#Q$# [deduction] Compute the unnormalized posterior weight `prior_H1 * likelihood_H1`."
            )
            harness_h = f"expected == {num_post}"
            gold_t = f"By Bayes rule numerator, multiply prior and likelihood: {prior_h1} * {like_h1} = {num_post}."
            gold_a = f"{num_post}"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# Add weights: {prior_h1} + {like_h1} = {prior_h1 + like_h1}.\n#A$# {prior_h1 + like_h1}",
                f"#T$# Multiply by distractor: {prior_h1} * {c} = {prior_h1 * c}.\n#A$# {prior_h1 * c}",
                "#T$# Default fallback.\n#A$# 0",
            )
            base_scores = (1.0, 0.15, 0.10, 0.0)
            domain = "multi_hop_deduction"
        elif kind == 16:
            # Thermodynamics Calorimetry Q = m * c_p * delta_T
            mass_g = a
            cp_j = (b % 5) + 2
            dt_k = (c % 8) + 3
            q_joules = mass_g * cp_j * dt_k
            task_id = f"physics_calorimetry_{mass_g}_{cp_j}_{dt_k}"
            prompt_q = (
                f"#U$# A sample of mass `m = {mass_g} g` with specific heat `c = {cp_j} J/(g*K)` "
                f"is heated by `dT = {dt_k} K`.\n"
                f"#Q$# [physics] Compute absorbed thermal energy `Q = m * c * dT` in Joules."
            )
            harness_h = f"expected == {q_joules}"
            gold_t = f"First compute m * c: {mass_g} * {cp_j} = {mass_g * cp_j}; then multiply by dT: {mass_g * cp_j} * {dt_k} = {q_joules}."
            gold_a = f"{q_joules}"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# Add temperature change: {mass_g * cp_j} + {dt_k} = {mass_g * cp_j + dt_k}.\n#A$# {mass_g * cp_j + dt_k}",
                f"#T$# Omit specific heat: {mass_g} * {dt_k} = {mass_g * dt_k}.\n#A$# {mass_g * dt_k}",
                "#T$# Zero delta.\n#A$# 0",
            )
            base_scores = (1.0, 0.20, 0.15, 0.0)
            domain = "physics_kinematics"
        elif kind == 17:
            # IFEval Constrained Verification (lowercase + no_comma + keyword + exact_int)
            res_val = a + b * c
            task_id = f"ifeval_math_{a}_{b}_{c}"
            prompt_q = (
                f"#U$# Compute `{a} + {b} * {c}` and reply in all lowercase letters with NO commas, "
                f"including the word `verified` and the exact integer result.\n"
                f"#Q$# [evaluation] Follow every formatting and arithmetic constraint."
            )
            harness_h = f"ifeval:lowercase;no_comma;keyword:verified;exact_int:{res_val}"
            gold_t = f"Multiply {b} * {c} = {b*c}, then add {a} + {b*c} = {res_val}. Format lowercase without commas."
            gold_a = f"the verified result is {res_val} units"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# {gold_t}\n#A$# The Verified Result, is {res_val} units",
                f"#T$# {gold_t}\n#A$# the result is {a + b + c} units",
                "#T$# Default fallback.\n#A$# 0",
            )
            base_scores = (1.0, 0.25, 0.15, 0.0)
            domain = "invariant_verification"
        elif kind == 18:
            # Multi-assert Python sequence generator
            step_d = (b % 6) + 2
            task_id = f"code_arith_prog_{a}_{step_d}"
            prompt_q = (
                f"#U$# Write `solve(n, start, step)` returning `start + n * step` for `n >= 0`.\n"
                f"#Q$# [code] Must pass assertions at `n = 0`, `n = 3`, and `n = {c}`."
            )
            harness_h = (
                f"assert solve(0, {a}, {step_d}) == {a} "
                f"and solve(3, {a}, {step_d}) == {a + 3 * step_d} "
                f"and solve({c}, {a}, {step_d}) == {a + c * step_d}"
            )
            gold_t = f"Closed form linear progression: return start + n * step; at n={c}, {c} * {step_d} = {c*step_d}, plus {a} = {a + c*step_d}."
            gold_a = "def solve(n, start, step):\n    return start + n * step"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                "#T$# Off-by-one multiplier.\n#A$# def solve(n, start, step):\n    return start + (n + 1) * step",
                "#T$# Multiply start instead of step.\n#A$# def solve(n, start, step):\n    return n * start + step",
                "#T$# Constant stub.\n#A$# def solve(n, start, step):\n    return 0",
            )
            base_scores = (1.0, 0.25, 0.15, 0.0)
            domain = "algorithmic_code"
        elif kind == 19:
            # Horn-Clause Boolean Propagation Checksum
            w1 = a + b
            w2 = w1 + c
            checksum = w1 + w2
            task_id = f"logic_horn_prop_{a}_{b}_{c}"
            prompt_q = (
                f"#U$# Horn rules: Fact `F1 = {a}`, Fact `F2 = {b}`. Rule R1 derives `G1 = F1 + F2`. "
                f"Rule R2 derives `G2 = G1 + {c}`.\n"
                f"#Q$# [deduction] Compute the forward-chaining sum `G1 + G2`."
            )
            harness_h = f"expected == {checksum}"
            gold_t = f"Forward chain R1: {a} + {b} = {w1}. Forward chain R2: {w1} + {c} = {w2}. Sum: {w1} + {w2} = {checksum}."
            gold_a = f"{checksum}"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# Only return G2: {w1} + {c} = {w2}.\n#A$# {w2}",
                f"#T$# Sum raw facts once: {a} + {b} = {w1}.\n#A$# {w1}",
                "#T$# Default fallback.\n#A$# 0",
            )
            base_scores = (1.0, 0.20, 0.10, 0.0)
            domain = "multi_hop_deduction"
        elif kind == 20:
            # COMPOSED TEMPLATE 1 (Symbolic Algebra o Python Code):
            # Solve linear equation for hidden slope k, then synthesize Python function using k.
            k_slope = (b % 7) + 2
            rhs_eq = a * k_slope + c
            n_eval = (c % 5) + 3
            expected_val = k_slope * n_eval + a
            task_id = f"comp_alg_code_{a}_{k_slope}_{c}_{n_eval}"
            prompt_q = (
                f"#U$# [Composed Algebra->Code] First solve `{a} * k + {c} = {rhs_eq}` for integer `k`. "
                f"Then implement `solve(n)` returning `{k_slope} * n + {a}` (using the solved `k`).\n"
                f"#Q$# [code] Derive `k` in `#T$#` and implement `solve(n)` in `#A$#`."
            )
            harness_h = f"assert solve({n_eval}) == {expected_val} and solve(0) == {a}"
            gold_t = (
                f"Stage 1 (Algebra): {rhs_eq} - {c} = {a * k_slope}, so k = {a * k_slope} // {a} = {k_slope}. "
                f"Stage 2 (Code): implement solve(n) = {k_slope} * n + {a}; at n={n_eval}, {k_slope} * {n_eval} = {k_slope * n_eval}, plus {a} = {expected_val}."
            )
            gold_a = f"def solve(n):\n    return {k_slope} * n + {a}"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# Stage 1 (Algebra): {rhs_eq} + {c} = {rhs_eq + c}, so k = {k_slope + 1}. Stage 2 (Code): implement solve(n) = {k_slope + 1} * n + {a}.\n#A$# def solve(n):\n    return {k_slope + 1} * n + {a}",
                f"#T$# Stage 1 (Algebra): {rhs_eq} - {c} = {a * k_slope}, so k = {k_slope}. Stage 2 (Code): subtract offset {a} instead of adding.\n#A$# def solve(n):\n    return {k_slope} * n - {a}",
                "#T$# Constant stub.\n#A$# def solve(n):\n    return 0",
            )
            base_scores = (1.0, 0.25, 0.20, 0.0)
            domain = "compositional_code_math"
        elif kind == 21:
            # COMPOSED TEMPLATE 2 (Physics Kinematics o Modular Ring Checksum):
            v0 = a % 15 + 2
            accel = 2 * ((b % 6) + 1)
            t_sec = (c % 4) + 2
            disp = v0 * t_sec + (accel * t_sec * t_sec) // 2
            p_ring = 23
            residue = (disp + c) % p_ring
            task_id = f"comp_phys_mod_{v0}_{accel}_{t_sec}_{c}"
            prompt_q = (
                f"#U$# [Composed Physics->ModRing] Stage 1: compute displacement `d = v0*t + 0.5*a*t^2` "
                f"for `v0={v0}`, `a={accel}`, `t={t_sec}`. Stage 2: compute checksum `(d + {c}) mod {p_ring}`.\n"
                f"#Q$# [physics] Derive `d` and return the final modular residue."
            )
            harness_h = f"expected == {residue}"
            gold_t = (
                f"Stage 1 (Kinematics): drift {v0} * {t_sec} = {v0*t_sec}, accel term = {(accel*t_sec*t_sec)//2}, "
                f"d = {disp}. Stage 2 (ModRing): ({disp} + {c}) mod {p_ring} = {residue}."
            )
            gold_a = f"{residue}"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# Stage 1 (Kinematics): drift {v0} * {t_sec} = {v0*t_sec}, accel term = {(accel*t_sec*t_sec)//2}, d = {disp}. Stage 2 (ModRing): omit mod reduction.\n#A$# {disp}",
                f"#T$# Stage 1 (Kinematics): final velocity {v0} + {accel * t_sec} = {v0 + accel * t_sec}. Stage 2 (ModRing): ({v0 + accel * t_sec} + {c}) mod {p_ring}.\n#A$# {(v0 + accel * t_sec + c) % p_ring}",
                "#T$# Zero delta.\n#A$# 0",
            )
            base_scores = (1.0, 0.20, 0.15, 0.0)
            domain = "compositional_physics_math"
        elif kind == 22:
            # COMPOSED TEMPLATE 3 (Project DAG Critical Path o Python Slack Aggregator):
            crit_len = a + b + c
            sample_tracks = [a + c, a + b, crit_len]
            expected_slack = sum(crit_len - t_d for t_d in sample_tracks)
            task_id = f"comp_dag_code_{a}_{b}_{c}"
            prompt_q = (
                f"#U$# [Composed DAG->Code] Milestone M1 ({a}d) + Alpha ({b}d) + M2 ({c}d) defines critical path `C = {crit_len}`. "
                f"Write `solve(tracks, crit)` returning `sum(crit - x for x in tracks)`.\n"
                f"#Q$# [code] Implement `solve(tracks, crit)` and verify on `tracks={sample_tracks}, crit={crit_len}`."
            )
            harness_h = f"assert solve({sample_tracks}, {crit_len}) == {expected_slack}"
            gold_t = (
                f"Stage 1 (DAG): Critical path {a} + {b} = {a+b}, plus {c} = {crit_len}. "
                f"Stage 2 (Code): Sum slack (crit - x) across tracks to get {expected_slack}."
            )
            gold_a = "def solve(tracks, crit):\n    return sum(crit - x for x in tracks)"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# Stage 1 (DAG): Critical path {a} + {b} = {a+b}, plus {c} = {crit_len}. Stage 2 (Code): Sum tracks directly without subtracting from crit.\n#A$# def solve(tracks, crit):\n    return sum(tracks)",
                f"#T$# Stage 1 (DAG): Shortest path {a} + {c} = {a+c}. Stage 2 (Code): Return single max slack crit - min(tracks).\n#A$# def solve(tracks, crit):\n    return crit - min(tracks)",
                "#T$# Constant stub.\n#A$# def solve(tracks, crit):\n    return 0",
            )
            base_scores = (1.0, 0.25, 0.15, 0.0)
            domain = "compositional_code_pm"
        else:
            # COMPOSED TEMPLATE 4 (Multi-Hop Deduction o IFEval Lexical Constraints):
            prod_ab = a * b
            final_v = prod_ab + c
            task_id = f"comp_ded_ifeval_{a}_{b}_{c}"
            prompt_q = (
                f"#U$# [Composed Deduction->IFEval] Cluster A runs {a} nodes with {b} shards each, plus {c} standby shards. "
                f"State the total shards in lowercase with no commas, including the keyword `shards` and the exact integer.\n"
                f"#Q$# [evaluation] Satisfy both the 2-hop arithmetic deduction and all IFEval constraints."
            )
            harness_h = f"ifeval:lowercase;no_comma;keyword:shards;exact_int:{final_v}"
            gold_t = f"Stage 1: {a} * {b} = {prod_ab}, then {prod_ab} + {c} = {final_v}. Stage 2: format lowercase with no comma and keyword shards."
            gold_a = f"cluster a manages {final_v} total shards online"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# {gold_t}\n#A$# Cluster A, manages {final_v} total shards online",
                f"#T$# Stage 1: {a} + {b} = {a+b}, then {a+b} + {c} = {a+b+c}. Stage 2: format lowercase with no comma.\n#A$# cluster a manages {a+b+c} total units online",
                "#T$# Default fallback.\n#A$# 0",
            )
            base_scores = (1.0, 0.25, 0.15, 0.0)
            domain = "compositional_deduction_ifeval"

        p_cands, p_scores, g_idx = _permute(base_cands, base_scores)
        cached_s = self.scaffold_cache.get(task_id, default=_SCAFFOLD_CANDIDATES[idx % len(_SCAFFOLD_CANDIDATES)])
        self._last_task_kinds[task_id] = kind
        return OrnithTask(
            task_id=task_id,
            domain=domain,
            prompt_q=prompt_q,
            harness_h=harness_h,
            scaffold_s=cached_s,
            candidate_rollouts=p_cands,
            harness_scores=p_scores,
            gold_index=g_idx,
            gold_thought=gold_t,
            gold_answer=gold_a,
        )

    def step_self_improvement(
        self,
        coe: Any,
        *,
        batch_size: int = 2,
        max_seq_len: int | None = None,
    ) -> OrnithStepOutput:
        """Run one end-to-end Ornith-1.0 / 2.0 self-improvement step on `coe`."""
        from .data import DecisionExample, collate_decisions, get_tokenizer

        vocab_size = int(getattr(coe.backbone.config, "vocab_size", 260))
        if getattr(self.tokenizer, "vocab_size", 260) != vocab_size:
            self.tokenizer = get_tokenizer(vocab_size)

        eff_seq_len = max_seq_len or self.max_seq_len
        model_dev = next(coe.parameters()).device
        if self.device != model_dev:
            self.device = model_dev
            self.novelty_buffer.device = model_dev
            self.novelty_buffer._buffer = self.novelty_buffer._buffer.to(model_dev)

        self._step_counter += 1
        use_compact = eff_seq_len < 220
        tasks = [
            self._generate_task(self._step_counter * batch_size + b, compact=use_compact)
            for b in range(batch_size)
        ]

        # Stage 0 (Dream-RSI): Online Discovery Tree Expansion + Offline Zero-Execution Replay Dreaming
        if (
            self.enable_dream_rsi
            and self.dream_engine is not None
            and (self._step_counter % self.dream_every == 0 or self._last_dream_out is None)
        ):
            dream_out = self.dream_engine.step_cycle(
                coe,
                tasks[0],
                max_seq_len=eff_seq_len,
            )
            self._last_dream_out = dream_out
            # Promote the Dream-RSI winning branch scaffold into ScaffoldCache and active task
            self.scaffold_cache.update_if_better(
                tasks[0].task_id,
                [dream_out.best_scaffold],
                [dream_out.best_node_score],
                step=self._step_counter,
            )
            tasks[0] = replace(tasks[0], scaffold_s=dream_out.best_scaffold)

        # Note: Solver context includes #U$#, #Q$#, and #S$# scaffold, NEVER the secret #H$# answer assertion!
        max_dec_len = max(eff_seq_len, 512)
        examples = [
            DecisionExample(
                question_id=t.task_id,
                kind="choice",
                context=f"{t.prompt_q}\n#S$# {t.scaffold_s}".encode("utf-8")[: max_dec_len - 4].decode("utf-8", errors="ignore"),
                descriptions=tuple(
                    d.encode("utf-8")[: max_dec_len - 4].decode("utf-8", errors="ignore")
                    for d in t.candidate_rollouts
                ),
                values=("opt_0", "opt_1", "opt_2", "opt_3"),
                target=t.gold_index,
            )
            for t in tasks
        ]
        batch = collate_decisions(
            examples,
            max_dec_len,
            device=self.device,
            tokenizer=self.tokenizer,
        )
        ctx_ids = batch["context_ids"]
        ctx_lens = batch["context_lengths"]
        opt_ids = batch["option_ids"]
        opt_lens = batch["option_lengths"]
        targets = batch["targets"]

        # 1. Stage 2 & 3: Evaluate all COE operators on the rollout group (single encoder pass)
        op_out = coe.forward_all_operators(
            ctx_ids,
            ctx_lens,
            opt_ids,
            opt_lens,
            targets,
        )

        # 2. Stage 1 (#Q$#): Compute <0.1ms Association Novelty N(q) & update NoveltyTaskBuffer
        # Reuses `op_out.decision.state_features` so `ctx_ids` is encoded only once per step.
        with torch.no_grad():
            s_raw = op_out.decision.state_features.detach().float()
            s_feat = s_raw.view(batch_size, -1, s_raw.shape[-1])[:, 0, :]
            q_embs = F.normalize(s_feat - s_feat.mean(dim=-1, keepdim=True), dim=-1, eps=1e-6)
            pos = torch.arange(ctx_ids.size(1), device=self.device, dtype=torch.float32).unsqueeze(0)
            freq = torch.arange(1, q_embs.size(-1) + 1, device=self.device, dtype=torch.float32).view(1, 1, -1)
            tok_f = (ctx_ids.float() / 256.0).unsqueeze(-1)
            sketch = (torch.sin(tok_f * freq * 6.283185 + pos.unsqueeze(-1) * 0.17)).mean(dim=1)
            q_hybrid = F.normalize(q_embs, dim=-1) + 0.65 * F.normalize(sketch, dim=-1)
            novelties = self.novelty_buffer.compute_novelty(q_hybrid)
            self.novelty_buffer.add(q_hybrid)

        probs = op_out.decision.probabilities  # [B, 4]
        log_probs = probs.clamp_min(1e-7).log()
        beh_log_probs = log_probs.detach()

        # Build verified harness score tensor [B, 4] via sandboxed Python execution + reference scores
        verified_rows: list[list[float]] = []
        for t in tasks:
            row_sc: list[float] = []
            for cand, ref_s in zip(t.candidate_rollouts, t.harness_scores):
                v_s = verify_python_rollout(cand, t.harness_h, t.gold_answer)
                row_sc.append(0.5 * v_s + 0.5 * float(ref_s) if v_s > 0.0 else 0.0)
            verified_rows.append(row_sc)

        h_scores = torch.tensor(
            verified_rows,
            device=self.device,
            dtype=torch.float32,
        )
        # Deterministic boundary monitor on the 0-score hack rollout in each row
        neg_scores = h_scores.min(dim=-1, keepdim=True).values

        # Task alignment C(q, h) from Deduction's `entailed` probability (index 0)
        task_alignment = op_out.deduction_probs[:, 0].detach().clamp(0.25, 1.0)
        # Reference utilities from Evaluation head reshaped to [B, 4]
        ref_utils = op_out.evaluation_utilities.detach().view(batch_size, -1)

        r_harness_vec = compute_harness_reward(
            task_alignment,
            h_scores,
            ref_utils,
            negative_harness_scores=neg_scores,
            hack_resistance=1.0,
        )

        # In-network frozen COE consistency check (Deduction contradicted > 0.55 or Metacognition correctness < 0.35)
        contradicted_p = op_out.deduction_probs[:, 1].detach()
        correctness_p = op_out.metacognition_probs[:, 0].detach()
        vetoed = (contradicted_p > 0.55) | (correctness_p < 0.35)
        veto_rate = float(vetoed.float().mean().item())

        # Empirical pass rate p (probability mass assigned to the permuted gold_index rollout)
        pass_rates = probs.gather(1, targets.unsqueeze(1)).squeeze(1).detach()
        d_frontier = frontier_difficulty(pass_rates, target_p=0.20)
        validity = (r_harness_vec > 0.05).float()
        r_task_vec = compute_task_reward(validity, pass_rates, novelties, target_p=0.20)

        # Update Ornith-2.0 ScaffoldCache & OrnithEnvArchitect capability frontier for each task
        for b_idx, t in enumerate(tasks):
            cand_scaffolds = [
                _SCAFFOLD_CANDIDATES[(b_idx + k) % len(_SCAFFOLD_CANDIDATES)]
                for k in range(len(t.candidate_rollouts))
            ]
            weighted_r = (h_scores[b_idx] * probs[b_idx].detach()).tolist()
            self.scaffold_cache.update_if_better(
                t.task_id,
                cand_scaffolds,
                weighted_r,
                step=self._step_counter,
            )
            if t.task_id in self._last_task_kinds:
                self.env_architect.record_outcome(
                    self._last_task_kinds[t.task_id],
                    float(pass_rates[b_idx].item()),
                    float(task_alignment[b_idx].item()),
                )

        # Group-normalized advantages A_i across the G=4 rollouts + staleness-weighted GRPO loss
        advantages = group_normalized_advantages(h_scores)
        grpo_loss = grpo_clipped_loss(
            log_probs,
            beh_log_probs,
            advantages,
            policy_age=0,
        )
        # Verified gold rollout negative log-likelihood + 0x8000 prompt-masked decoder self-distillation
        decision_distill_loss = F.nll_loss(log_probs, targets)
        max_lm_len = min(eff_seq_len, int(getattr(coe.backbone.config, "max_seq_len", 256)))
        inp_rows: list[list[int]] = []
        lbl_rows: list[list[int]] = []
        for t in tasks:
            u_clean = t.prompt_q.split("\n#Q$#")[0].replace("#U$#", "").strip()
            raw_turn = self.tokenizer.encode_masked_turn(u_clean, t.gold_thought, t.gold_answer)
            if len(raw_turn) > max_lm_len + 1:
                raw_turn = raw_turn[: max_lm_len + 1]
            inp_rows.append([tok & 0x7FFF for tok in raw_turn[:-1]])
            lbl_rows.append(
                [
                    -100
                    if (tok & self.tokenizer.prompt_mask_bit)
                    or (tok & 0x7FFF) == self.tokenizer.pad_id
                    else (tok & 0x7FFF)
                    for tok in raw_turn[1:]
                ]
            )
        pad_len = max(len(r) for r in inp_rows)
        inp_tensor = torch.tensor(
            [r + [self.tokenizer.pad_id] * (pad_len - len(r)) for r in inp_rows],
            dtype=torch.long,
            device=self.device,
        )
        lbl_tensor = torch.tensor(
            [r + [-100] * (pad_len - len(r)) for r in lbl_rows],
            dtype=torch.long,
            device=self.device,
        )
        lm_distill_out = coe.forward_inference(inp_tensor, targets=lbl_tensor)
        assert lm_distill_out.loss is not None
        self_distill_loss = 0.70 * decision_distill_loss + 0.30 * lm_distill_out.loss

        # LiquidAI Antidoom FTPO + Compiler-in-the-Loop Token Targeting (supports both ByteTokenizer & MorphCode-16K):
        # Strictly target either (a) an inner repetition trigger (`loop_hit is not None`), or
        # (b) the exact first token where the model's on-policy prediction diverges from the
        # verified rollout (`pred != gold`). Never penalize second-best tokens when greedy is 100% verified.
        ftpo_pos_logits: list[Tensor] = []
        ftpo_rej_ids: list[int] = []
        ftpo_cho_ids: list[list[int]] = []
        with torch.no_grad():
            if vocab_size > 1024:
                masked_l = lm_distill_out.logits.detach().clone()
                masked_l[:, :, :10] = -torch.inf
                masked_l[:, :, 11:32] = -torch.inf
                masked_l[:, :, 127:260] = -torch.inf
                pred_ids = masked_l.argmax(dim=-1)
            else:
                pred_ids = lm_distill_out.logits[:, :, :256].argmax(dim=-1)
        for b_i in range(inp_tensor.size(0)):
            valid_pos = (lbl_tensor[b_i] != -100).nonzero(as_tuple=False).squeeze(-1)
            if valid_pos.numel() < 4:
                continue
            seq_preds = pred_ids[b_i, valid_pos].tolist()
            loop_hit = detect_doom_loop(seq_preds, min_period=3, max_period=48, min_repeats=2, min_total_repeated=6)
            if loop_hit is not None:
                trig_local, rej_tok, _ = loop_hit
                pos_idx = int(valid_pos[min(trig_local, valid_pos.numel() - 1)].item())
            else:
                mismatches = (pred_ids[b_i, valid_pos] != lbl_tensor[b_i, valid_pos]).nonzero(as_tuple=False).squeeze(-1)
                if mismatches.numel() == 0:
                    continue
                first_err_local = int(mismatches[0].item())
                pos_idx = int(valid_pos[first_err_local].item())
                rej_tok = int(pred_ids[b_i, pos_idx].item())
            gold_tok = int(lbl_tensor[b_i, pos_idx].item())
            if rej_tok == gold_tok:
                continue
            if vocab_size > 1024:
                row_cand = lm_distill_out.logits[b_i, pos_idx].detach().clone()
                row_cand[:32] = -torch.inf
                row_cand[127:260] = -torch.inf
                row_cand[rej_tok] = -torch.inf
                top_alts = row_cand.topk(4).indices.tolist()
            else:
                row_ascii = lm_distill_out.logits[b_i, pos_idx, 32:127].detach().clone()
                if 32 <= rej_tok < 127:
                    row_ascii[rej_tok - 32] = -torch.inf
                top_alts = (row_ascii.topk(4).indices + 32).tolist()
            chosen_row = [gold_tok] + [t_id for t_id in top_alts if t_id != gold_tok][:3]
            while len(chosen_row) < 4:
                chosen_row.append(gold_tok)
            ftpo_pos_logits.append(lm_distill_out.logits[b_i, pos_idx])
            ftpo_rej_ids.append(rej_tok)
            ftpo_cho_ids.append(chosen_row[:4])

        if ftpo_pos_logits:
            ftpo_logits_t = torch.stack(ftpo_pos_logits, dim=0)
            ftpo_rej_t = torch.tensor(ftpo_rej_ids, dtype=torch.long, device=self.device)
            ftpo_cho_t = torch.tensor(ftpo_cho_ids, dtype=torch.long, device=self.device)
            antidoom_loss_t, antidoom_win = compute_antidoom_ftpo_loss(
                ftpo_logits_t,
                ftpo_logits_t.detach(),
                ftpo_rej_t,
                ftpo_cho_t,
            )
        else:
            antidoom_loss_t = torch.zeros((), device=self.device)
            antidoom_win = 1.0

        total_ornith_loss = (
            op_out.total_loss
            + 0.5 * grpo_loss
            + 0.25 * self_distill_loss
            + 0.15 * antidoom_loss_t
        )
        mean_rollout_r = float((probs.detach() * h_scores).sum(dim=-1).mean().item())
        d_out = self._last_dream_out

        return OrnithStepOutput(
            loss=total_ornith_loss,
            grpo_loss=grpo_loss.detach(),
            self_distill_loss=self_distill_loss.detach(),
            task_reward=round(float(r_task_vec.mean().item()), 4),
            harness_reward=round(float(r_harness_vec.mean().item()), 4),
            rollout_reward=round(mean_rollout_r, 4),
            novelty=round(float(novelties.mean().item()), 4),
            pass_rate=round(float(pass_rates.mean().item()), 4),
            frontier_difficulty=round(float(d_frontier.mean().item()), 4),
            veto_rate=round(veto_rate, 4),
            scaffolds_cached=len(self.scaffold_cache._entries),
            novelty_buffer_size=len(self.novelty_buffer),
            dream_cycle=0 if d_out is None else int(d_out.cycle),
            dream_replay_score=0.0 if d_out is None else float(d_out.final_replay_score),
            dream_initial_score=0.0 if d_out is None else float(d_out.initial_replay_score),
            dream_accepted_revisions=0 if d_out is None else int(d_out.total_accepted_revisions),
            dream_replays_dreamed=0 if d_out is None else int(d_out.total_replays_dreamed),
            dream_worlds_archived=0 if d_out is None else int(d_out.worlds_archived),
            dream_active_policy="" if d_out is None else str(d_out.active_policy_text),
            antidoom_loss=round(float(antidoom_loss_t.detach().item()), 4),
            antidoom_chosen_win=round(float(antidoom_win), 4),
        )


