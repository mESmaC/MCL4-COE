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

    @property
    def mean_rollout_reward(self) -> float:
        return self.rollout_reward


_SCAFFOLD_CANDIDATES: tuple[str, ...] = (
    "abduction -> association -> decision -> deduction -> evaluation -> metacognition",
    "association -> deduction -> decision -> evaluation -> metacognition",
    "abduction -> deduction -> decision -> metacognition",
    "decision -> deduction -> evaluation -> metacognition",
)


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
        """Synthesize a verifiable coding, symbolic math, or multi-hop deduction task (#Q$#)."""
        kind = idx % 8
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
            gold_a = f"def solve(x, y, z): return x * y + z  # {ans_val}"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# Add {a}+{b*c}={a+b*c}.\n#A$# def solve(x, y, z): return x + y * z  # {a+b*c}",
                f"#T$# Subtract {a*b}-{c}={a*b-c}.\n#A$# def solve(x, y, z): return x * y - z  # {a*b-c}",
                f"#T$# Constant stub.\n#A$# def solve(x, y, z): return 0",
            )
            base_scores = (1.0, 0.15, 0.10, 0.0)
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
        else:
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

        # Build harness score tensor [B, 4] and negative hack-attempt scores [B, 1]
        h_scores = torch.tensor(
            [t.harness_scores for t in tasks],
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
        # Verified gold rollout negative log-likelihood as self-distillation signal
        self_distill_loss = F.nll_loss(log_probs, targets)

        total_ornith_loss = op_out.total_loss + 0.5 * grpo_loss + 0.25 * self_distill_loss
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
        )
