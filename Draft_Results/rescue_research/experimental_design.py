from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence


VALID_POLICIES = {"adaptive", "strict"}


@dataclass(frozen=True)
class ThreeWaySplitPlan:
    n_icl: int
    n_selection: int
    n_eval: int
    policy: str
    total_available: int
    total_target: int
    scaling_factor: float
    rationale: str
    warnings: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class FourWaySplitPlan:
    n_icl_bank: int
    n_selection: int
    n_eval_open: int
    n_eval_blind: int
    policy: str
    total_available: int
    total_target: int
    scaling_factor: float
    rationale: str
    warnings: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class _AllocationPlan:
    counts: Dict[str, int]
    total_target: int
    scaling_factor: float
    warnings: List[str]
    policy: str


def _as_int_counts(names: Sequence[str], values: Iterable[int]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for name, value in zip(names, values):
        out[name] = int(value)
    return out


def _round_proportional(weights: Sequence[float], total: int) -> List[int]:
    n = len(weights)
    if total <= 0:
        return [0] * n
    if n == 0:
        return []

    safe_weights = [max(0.0, float(w)) for w in weights]
    denom = sum(safe_weights)
    if denom <= 0:
        out = [0] * n
        for i in range(total):
            out[i % n] += 1
        return out

    raw = [float(total) * (w / denom) for w in safe_weights]
    base = [int(math.floor(x)) for x in raw]
    rem = int(total - sum(base))
    if rem <= 0:
        return base

    fracs = [(raw[i] - base[i], i) for i in range(n)]
    fracs.sort(key=lambda x: (-x[0], x[1]))
    for _, idx in fracs[:rem]:
        base[idx] += 1
    return base


def _validate_policy(policy: str) -> str:
    p = str(policy).strip().lower()
    if p not in VALID_POLICIES:
        raise ValueError(f"Unknown split policy '{policy}'. Expected one of {sorted(VALID_POLICIES)}.")
    return p


def _allocate_counts(
    *,
    labels: Sequence[str],
    targets: Sequence[int],
    minima: Sequence[int],
    total_available: int,
    policy: str,
) -> _AllocationPlan:
    if len(labels) != len(targets) or len(labels) != len(minima):
        raise ValueError("labels/targets/minima must have equal length.")

    policy = _validate_policy(policy)
    total_available = int(total_available)
    target_vals = [max(0, int(x)) for x in targets]
    minima_vals: List[int] = []
    for t, m in zip(target_vals, minima):
        minima_vals.append(min(int(m), t) if t > 0 else 0)

    total_target = int(sum(target_vals))
    if total_target <= 0:
        raise ValueError("Target split counts sum to zero; nothing to allocate.")

    warnings: List[str] = []
    if policy == "strict":
        if total_available < total_target:
            raise ValueError(
                f"Strict split policy requires >= {total_target} items, but only {total_available} available."
            )
        return _AllocationPlan(
            counts=_as_int_counts(labels, target_vals),
            total_target=total_target,
            scaling_factor=1.0,
            warnings=warnings,
            policy=policy,
        )

    # Adaptive policy
    if total_available >= total_target:
        return _AllocationPlan(
            counts=_as_int_counts(labels, target_vals),
            total_target=total_target,
            scaling_factor=1.0,
            warnings=warnings,
            policy=policy,
        )

    min_total = int(sum(minima_vals))
    if total_available >= min_total:
        extras = [max(0, t - m) for t, m in zip(target_vals, minima_vals)]
        extra_budget = int(total_available - min_total)
        extra_alloc = _round_proportional(extras, extra_budget)
        counts = [m + e for m, e in zip(minima_vals, extra_alloc)]
        return _AllocationPlan(
            counts=_as_int_counts(labels, counts),
            total_target=total_target,
            scaling_factor=float(total_available) / float(total_target),
            warnings=warnings,
            policy=policy,
        )

    # Total is below minima: degrade to a minimum viable allocation while preserving all non-zero partitions.
    required_partitions = [i for i, t in enumerate(target_vals) if t > 0]
    if total_available < len(required_partitions):
        raise ValueError(
            f"Not enough items ({total_available}) to allocate one example to each required partition "
            f"({len(required_partitions)})."
        )
    warnings.append(
        "Total available is below configured minima; using minimum-viable allocation with one item per required partition."
    )
    base = [1 if t > 0 else 0 for t in target_vals]
    residual = int(total_available - sum(base))
    residual_weights = [max(0, m - b) for m, b in zip(minima_vals, base)]
    residual_alloc = _round_proportional(residual_weights, residual)
    counts = [b + r for b, r in zip(base, residual_alloc)]
    return _AllocationPlan(
        counts=_as_int_counts(labels, counts),
        total_target=total_target,
        scaling_factor=float(total_available) / float(total_target),
        warnings=warnings,
        policy=policy,
    )


def design_three_way_split(
    total_available: int,
    *,
    n_icl_target: int,
    n_selection_target: int,
    n_eval_target: int,
    policy: str = "adaptive",
    min_icl: int = 4,
    min_selection: int = 8,
    min_eval: int = 12,
) -> ThreeWaySplitPlan:
    alloc = _allocate_counts(
        labels=("icl", "selection", "eval"),
        targets=(n_icl_target, n_selection_target, n_eval_target),
        minima=(min_icl, min_selection, min_eval),
        total_available=total_available,
        policy=policy,
    )
    rationale = (
        "Three-way split is computed with the selected policy and preserves disjoint "
        "ICL/selection/eval partitions while scaling toward protocol targets."
    )
    return ThreeWaySplitPlan(
        n_icl=alloc.counts["icl"],
        n_selection=alloc.counts["selection"],
        n_eval=alloc.counts["eval"],
        policy=alloc.policy,
        total_available=int(total_available),
        total_target=alloc.total_target,
        scaling_factor=float(alloc.scaling_factor),
        rationale=rationale,
        warnings=list(alloc.warnings),
    )


def design_four_way_split(
    total_available: int,
    *,
    n_icl_bank_target: int,
    n_selection_target: int,
    n_eval_open_target: int,
    n_eval_blind_target: int,
    policy: str = "adaptive",
    min_icl_bank: int = 8,
    min_selection: int = 16,
    min_eval_open: int = 24,
    min_eval_blind: int = 8,
) -> FourWaySplitPlan:
    alloc = _allocate_counts(
        labels=("icl_bank", "selection", "eval_open", "eval_blind"),
        targets=(
            n_icl_bank_target,
            n_selection_target,
            n_eval_open_target,
            n_eval_blind_target,
        ),
        minima=(min_icl_bank, min_selection, min_eval_open, min_eval_blind),
        total_available=total_available,
        policy=policy,
    )
    rationale = (
        "Four-way protocol split uses adaptive/strict allocation with explicit minima "
        "and preserves disjoint ICL-bank/selection/open-eval/blind-eval partitions."
    )
    return FourWaySplitPlan(
        n_icl_bank=alloc.counts["icl_bank"],
        n_selection=alloc.counts["selection"],
        n_eval_open=alloc.counts["eval_open"],
        n_eval_blind=alloc.counts["eval_blind"],
        policy=alloc.policy,
        total_available=int(total_available),
        total_target=alloc.total_target,
        scaling_factor=float(alloc.scaling_factor),
        rationale=rationale,
        warnings=list(alloc.warnings),
    )
