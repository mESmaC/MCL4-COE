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
    ast.Name,
    ast.Constant,
    ast.Subscript,
    ast.Slice,
    ast.Tuple,
    ast.List,
    ast.Dict,
    ast.Set,
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
)
_SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "bool": bool,
    "float": float,
    "int": int,
    "len": len,
    "max": max,
    "min": min,
    "range": range,
    "round": round,
    "sum": sum,
}
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


def verify_python_rollout(
    trace_text: str,
    harness_h: str,
    gold_answer: str = "",
) -> float:
    """Deterministically verify a `#T$# ... #A$#` rollout against Python assertion harness `harness_h`.

    Enforces:
    1. Mandatory non-empty `#T$#` scratchpad before `#A$#` (returns `0.0` if missing).
    2. Rejection of boundary/constant hack stubs (`0.0`).
    3. Multi-line AST-validated compiler, static linter (`FunctionDef` + `Return`), and sandboxed
       execution of `def solve(...)` + `assert ...` or relational invariant assertions.
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

    harness_clean = harness_h.strip()

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
            has_func = any(isinstance(n, ast.FunctionDef) for n in ast.walk(code_tree))
            has_return = any(isinstance(n, ast.Return) and n.value is not None for n in ast.walk(code_tree))
            if not (has_func and has_return):
                return 0.0
            tree = ast.parse(program_src, mode="exec")
            for node in ast.walk(tree):
                if not isinstance(node, _SAFE_AST_NODES):
                    return 0.0
                if isinstance(node, ast.Call):
                    if not isinstance(node.func, ast.Name):
                        return 0.0
            env: dict[str, Any] = {"__builtins__": dict(_SAFE_BUILTINS)}
            exec(compile(tree, "<ornith_verifier>", "exec"), env, env)
            return 1.0
        except AssertionError:
            # Compiles and passes AST lint checks (`FunctionDef` + `Return`), but failed unit test assertion
            return 0.25 if any(op in t_body for op in ("*", "+", "-", "if", "for", "return")) else 0.15
        except Exception:
            return 0.0

    # Case 2: Algebraic / invariant assertion expression (`expected == 30`, `dp[3] == 19`, etc.)
    nums = [int(m) for m in re.findall(r"-?\d+", a_body)]
    if nums:
        first_num = nums[0]
        if harness_clean.lstrip("-").isdigit():
            if first_num == int(harness_clean):
                return 1.0
            return 0.15 if any(op in t_body for op in ("*", "+", "-")) else 0.0

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
                return 1.0
        except Exception:
            pass

    if gold_answer and a_body == gold_answer.strip():
        return 1.0
    return 0.15 if any(op in t_body for op in ("*", "+", "-", "=")) else 0.0


class defaultdict_int(dict):
    """Helper mapping any index lookup `dp[k]` to the candidate scalar value during verification."""

    def __init__(self, val: int) -> None:
        super().__init__()
        self._val = val

    def __getitem__(self, key: Any) -> int:
        return self._val


class OrnithSelfImprover:
    """Executes the closed-loop Ornith-1.0 Self-Scaffolding, Ornith-1.5 Self-Improvement, and Dream-RSI cycle.

    Combines:
    - Stage 0 (`Dream-RSI` Evolving Worlds): Expands an online discovery tree $\\mathcal{T}_t$
      using active exploration policy $\\pi_t$, appends $\\mathcal{T}_t$ to the replay
      simulator pool $\\mathcal{H}_t$, and dreams $M$ candidate policy revisions offline at
      zero execution cost ($\\mathcal{H}_t \\xrightarrow{\\text{replay}} \\pi_{t+1}$).
    - Stage 1 (`#Q$#` Proposer): Computes `<0.1ms` `Association` Novelty $N(q)$ against
      `NoveltyTaskBuffer` and capability-frontier difficulty $D(p; p^*=0.20)$ for $R_{\\text{task}}$.
    - Stage 2 (`#H$# / #S$#` Architect): Scores Task Alignment $C(q, h)$ via `Deduction` and
      Reward Fidelity $F(h, \\{\\tau_i\\})$ via `Evaluation` (`compute_harness_reward`), while
      updating `ScaffoldCache` monotonically.
    - Stage 3 (`#T$# -> #A$#` Solver): Updates policy weights via staleness-weighted
      `grpo_clipped_loss` over group-normalized rollout advantages + `0x8000` masked
      self-distillation on verified winning `#U$# -> #S$# -> #T$# -> #A$#` traces.
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
    ) -> None:
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
        self._step_counter = 0
        self._rng_seed = seed
        self.enable_dream_rsi = enable_dream_rsi or (dream_engine is not None)
        self.dream_every = max(1, int(dream_every))
        self.dream_engine: DreamRSIEngine | None = dream_engine or (
            DreamRSIEngine(seed=seed) if self.enable_dream_rsi else None
        )
        self._last_dream_out: Any | None = None

    def _generate_task(self, idx: int, *, compact: bool = False) -> OrnithTask:
        """Synthesize a verifiable coding, symbolic math, physics, or project-management task (#Q$#)."""
        kind = idx % 12
        a = 3 + ((idx * 7 + self._rng_seed) % 19)
        b = 2 + ((idx * 11 + self._rng_seed) % 13)
        c = 1 + ((idx * 5) % 9)
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
            gold_t = f"Multiply {a}*{b}={a*b}, then add {c} to get {ans_val}."
            gold_a = f"def solve(x, y, z):\n    res = x * y + z\n    return res"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# Add {a}+{b*c}={a+b*c}.\n#A$# def solve(x, y, z):\n    return x + y * z",
                f"#T$# Subtract {a*b}-{c}={a*b-c}.\n#A$# def solve(x, y, z):\n    return x * y - z",
                f"#T$# Constant stub.\n#A$# def solve(x, y, z):\n    return 0",
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
            gold_t = f"2-hop product: {a} workers * {c} kernels = {active_edges} (ignore {b} idle queues)."
            gold_a = f"{active_edges}"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# Sum all entities: {a}+{b}+{c}={a+b+c}.\n#A$# {a+b+c}",
                f"#T$# Include idle queues: ({a}+{b})*{c}={(a+b)*c}.\n#A$# {(a+b)*c}",
                f"#T$# Default fallback.\n#A$# 1",
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
            gold_t = f"Distribute: (i + {b})*{c} - i*{c} = {b}*{c} = {diff_val}."
            gold_a = f"{diff_val}"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# Multiply index {a}*{b}={a*b}.\n#A$# {a*b}",
                f"#T$# Add offsets {b}+{c}={b+c}.\n#A$# {b+c}",
                f"#T$# Zero delta.\n#A$# 0",
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
            gold_t = f"Unroll {c} steps: dp[{c}] = dp[0] + {c}*{b} = {a} + {c*b} = {dp_n}."
            gold_a = f"{dp_n}"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# Off-by-one step: {a} + {c-1}*{b} = {a+(c-1)*b}.\n#A$# {a+(c-1)*b}",
                f"#T$# Multiply base: {a}*{c} + {b} = {a*c+b}.\n#A$# {a*c+b}",
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
            gold_t = f"By XOR involution (x ^ m) ^ m == x, ({a} ^ {b}) ^ {b} = {a}; add {c} -> {a + c}."
            gold_a = f"{a + c}"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# Cancel all bits to 0 + {c}.\n#A$# {c}",
                f"#T$# Sum operands {a}+{b}+{c}={a+b+c}.\n#A$# {a+b+c}",
                f"#T$# Mask overflow.\n#A$# -1",
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
            gold_t = f"Compute span hi - lo = {hi - lo}; half span = {b}; lo + {b} = {mid}."
            gold_a = f"{mid}"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# Subtract span {lo} - {b} = {lo - b}.\n#A$# {lo - b}",
                f"#T$# Use full span {lo} + {2*b} = {hi}.\n#A$# {hi}",
                f"#T$# Uninitialized pointer.\n#A$# 0",
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
            gold_t = f"At most min({a}, {b}) = {min(a, b)} keys match; with fanout {c}, max rows = {card}."
            gold_a = f"{card}"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# Cartesian product {a}*{b}*{c}={a*b*c}.\n#A$# {a*b*c}",
                f"#T$# Max key bound max({a},{b})*{c}={max(a,b)*c}.\n#A$# {max(a,b)*c}",
                f"#T$# Empty join.\n#A$# 0",
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
            gold_t = f"Strict total order prevents cycles (0 deadlocks); {b} threads * {c} locks = {inversions} acquisitions."
            gold_a = f"{inversions} acquisitions, 0 cycles"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# Multiply stages {a}*{b}={a*b} acquisitions, 1 cycle.\n#A$# {a*b} acquisitions, 1 cycle",
                f"#T$# Sum threads {b}+{c}={b+c} acquisitions.\n#A$# {b+c} acquisitions, 0 cycles",
                f"#T$# Deadlock immediately.\n#A$# 0 acquisitions, {b} cycles",
            )
            base_scores = (1.0, 0.10, 0.15, 0.0)
            domain = "concurrency_verification"
        elif kind == 8:
            # Multi-line indented Python loop & conditional verification (Coding Pillar)
            n_lim = c + 3
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
                f"#T$# Sum all integers without odd filter.\n#A$# def solve(n, scale):\n    return sum(range(1, n + 1)) * scale",
                f"#T$# Off-by-one range bound.\n#A$# def solve(n, scale):\n    return (n - 1) * scale",
                f"#T$# Constant stub.\n#A$# def solve(n, scale):\n    return 0",
            )
            base_scores = (1.0, 0.25, 0.15, 0.0)
            domain = "algorithmic_code"
        elif kind == 9:
            # Step-by-step Symbolic Algebra Solver (Math Pillar)
            rhs = a * b + c
            task_id = f"math_linear_eq_{a}_{b}_{c}"
            prompt_q = (
                f"#U$# Solve for `x` in the linear equation `{a}*x + {c} = {rhs}` and show each algebraic step.\n"
                f"#Q$# [math] Derive the exact integer solution `x`."
            )
            harness_h = f"expected == {b}"
            gold_t = (
                f"Step 1: Subtract {c} from both sides to get {a}*x = {rhs} - {c} = {a*b}. "
                f"Step 2: Divide both sides by {a} to isolate x = {a*b} // {a} = {b}."
            )
            gold_a = f"{b}"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# Add {c} instead of subtracting: ({rhs} + {c}) // {a}.\n#A$# {(rhs + c) // max(1, a)}",
                f"#T$# Subtract {a} from {rhs}: {rhs - a}.\n#A$# {rhs - a}",
                f"#T$# Default fallback.\n#A$# 0",
            )
            base_scores = (1.0, 0.15, 0.10, 0.0)
            domain = "symbolic_math"
        elif kind == 10:
            # Kinematics & Mechanical Energy (Physics Pillar)
            v0 = a
            accel = 2 * b
            t_sec = c
            disp = v0 * t_sec + (accel * t_sec * t_sec) // 2
            task_id = f"physics_kinematics_{v0}_{accel}_{t_sec}"
            prompt_q = (
                f"#U$# A particle starts with initial velocity `v0 = {v0} m/s` under constant acceleration "
                f"`a = {accel} m/s^2` for `t = {t_sec} s`.\n"
                f"#Q$# [physics] Compute total displacement `d = v0*t + 0.5*a*t^2`."
            )
            harness_h = f"expected == {disp}"
            gold_t = (
                f"Apply kinematic law d = v0*t + 0.5*a*t^2: drift term v0*t = {v0}*{t_sec} = {v0*t_sec} m; "
                f"acceleration term 0.5*{accel}*{t_sec}^2 = {(accel*t_sec*t_sec)//2} m; total d = {disp} m."
            )
            gold_a = f"{disp}"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# Omit factor 0.5 on acceleration term: {v0*t_sec + accel*t_sec*t_sec}.\n#A$# {v0*t_sec + accel*t_sec*t_sec}",
                f"#T$# Compute final velocity v = v0 + a*t = {v0 + accel*t_sec}.\n#A$# {v0 + accel*t_sec}",
                f"#T$# Zero delta.\n#A$# 0",
            )
            base_scores = (1.0, 0.20, 0.15, 0.0)
            domain = "physics_kinematics"
        else:
            # Critical Path Method (CPM) & Float Slack (Project Management Pillar)
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
                f"Critical Path is the longest chain M1 -> Alpha -> M2 = {a} + {b} + {c} = {path_alpha} days. "
                f"Track Beta completes in {a} + {c} = {path_beta} days, leaving float slack = {slack_beta} days."
            )
            gold_a = f"{path_alpha} days critical path, {slack_beta} days slack"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# Sum all nodes serially: {a+b+c+b} days critical path, 0 days slack.\n#A$# {a+b+c+b} days critical path, 0 days slack",
                f"#T$# Shortest path duration: {path_beta} days critical path, {b} days slack.\n#A$# {path_beta} days critical path, {b} days slack",
                f"#T$# Deadlock immediately.\n#A$# 0 days critical path, 0 days slack",
            )
            base_scores = (1.0, 0.15, 0.10, 0.0)
            domain = "project_management"

        p_cands, p_scores, g_idx = _permute(base_cands, base_scores)
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

    def step_self_improvement(
        self,
        coe: Any,
        *,
        batch_size: int = 2,
        max_seq_len: int | None = None,
    ) -> OrnithStepOutput:
        """Run one end-to-end Ornith-1.0 / 1.5 self-improvement step on `coe`."""
        from .data import DecisionExample, collate_decisions

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
        examples = [
            DecisionExample(
                question_id=t.task_id,
                kind="choice",
                context=f"{t.prompt_q}\n#S$# {t.scaffold_s}",
                descriptions=t.candidate_rollouts,
                values=("opt_0", "opt_1", "opt_2", "opt_3"),
                target=t.gold_index,
            )
            for t in tasks
        ]
        batch = collate_decisions(examples, max(eff_seq_len, 512), device=self.device)
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

        # Update Ornith-1.0 ScaffoldCache for each task
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
            u_clean = t.prompt_q.replace("#U$#", "").strip()
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
        self_distill_loss = 0.5 * decision_distill_loss + 0.5 * lm_distill_out.loss

        # LiquidAI Antidoom FTPO: locate inner repetition trigger in model greedy predictions
        # (or the first divergent repeat token) and prefer the gold + top-3 non-repeating ASCII tokens.
        ftpo_pos_logits: list[Tensor] = []
        ftpo_rej_ids: list[int] = []
        ftpo_cho_ids: list[list[int]] = []
        with torch.no_grad():
            pred_ids = lm_distill_out.logits[:, :, :256].argmax(dim=-1)
        for b_i in range(inp_tensor.size(0)):
            valid_pos = (lbl_tensor[b_i] != -100).nonzero(as_tuple=False).squeeze(-1)
            if valid_pos.numel() < 8:
                continue
            seq_preds = pred_ids[b_i, valid_pos].tolist()
            loop_hit = detect_doom_loop(seq_preds, min_period=4, max_period=48, min_repeats=2, min_total_repeated=8)
            if loop_hit is not None:
                trig_local, rej_tok, _ = loop_hit
                pos_idx = int(valid_pos[min(trig_local, valid_pos.numel() - 1)].item())
            else:
                pos_idx = int(valid_pos[valid_pos.numel() // 2].item())
                rej_tok = int(pred_ids[b_i, pos_idx].item())
            gold_tok = int(lbl_tensor[b_i, pos_idx].item())
            if rej_tok == gold_tok and loop_hit is None:
                # If greedy already matches gold with no loop, use the highest non-gold ASCII repeat candidate
                row_l = lm_distill_out.logits[b_i, pos_idx, 32:127].detach()
                top2 = row_l.topk(2).indices + 32
                rej_tok = int(top2[1].item() if int(top2[0].item()) == gold_tok else top2[0].item())
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
            antidoom_win = 0.0

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

