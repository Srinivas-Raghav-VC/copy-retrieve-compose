from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class MediationEstimate:
    mediated_effect: float
    direct_effect: float
    ci_low: float
    ci_high: float


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def parse_mediation_result(shared_mediation_result: Dict[str, Any]) -> Tuple[MediationEstimate, Dict[str, Any]]:
    """
    Parse the causal_mediation result payload into a MediationEstimate.

    Expected schema (see causal_mediation.py):
    - aggregate_stats.mean_nie
    - aggregate_stats.bootstrap_ci_95 = [ci_low, ci_high]
    - causal_effects[].nde (direct effects per word; used for mean direct_effect)
    - triangulation.* (optional but usually present)
    """
    shared = shared_mediation_result or {}
    agg = (shared.get("aggregate_stats") or {}) if isinstance(shared, dict) else {}
    tri = (shared.get("triangulation") or {}) if isinstance(shared, dict) else {}
    causal = shared.get("causal_effects") if isinstance(shared, dict) else None

    mean_nie = agg.get("mean_nie")
    ci = agg.get("bootstrap_ci_95") or agg.get("bootstrap_ci_95".lower())
    ci_low = float("nan")
    ci_high = float("nan")
    if isinstance(ci, (list, tuple)) and len(ci) >= 2:
        ci_low = _coerce_float(ci[0])
        ci_high = _coerce_float(ci[1])
    mediated = _coerce_float(mean_nie) if mean_nie is not None else float("nan")

    # Direct effect: mean NDE across words if available.
    nde_vals = []
    if isinstance(causal, list):
        for eff in causal:
            if isinstance(eff, dict) and eff.get("nde") is not None:
                nde = _coerce_float(eff["nde"])
                if nde == nde:
                    nde_vals.append(nde)
    direct = sum(nde_vals) / len(nde_vals) if nde_vals else float("nan")

    estimate = MediationEstimate(
        mediated_effect=mediated,
        direct_effect=direct,
        ci_low=ci_low,
        ci_high=ci_high,
    )
    meta = {
        "p_value_nie_gt_0": (agg.get("p_value_nie_gt_0") if isinstance(agg, dict) else None),
        "positive_rate": (agg.get("positive_rate") if isinstance(agg, dict) else None),
        "triangulation": tri if isinstance(tri, dict) else {},
    }
    return estimate, meta


def mediation_direction_consistent(
    *,
    sufficiency_effect: float,
    necessity_effect: float,
    estimate: MediationEstimate,
) -> bool:
    """
    Simple plan-aligned consistency check:
    - sufficiency should be positive
    - necessity should be negative
    - mediated effect should be positive and CI should include positive support
    """
    return (
        sufficiency_effect > 0.0
        and necessity_effect < 0.0
        and estimate.mediated_effect > 0.0
        and estimate.ci_high > 0.0
    )


def h3_pass_strict(
    *,
    sufficiency_effect: float,
    necessity_effect: float,
    shared_mediation_result: Dict[str, Any],
    require_triangulation_accepted: bool = True,
) -> bool:
    """
    Stricter H3 check aligned to the protocol:
    - mediation direction consistent with sufficiency/necessity
    - NIE CI supports positive mediated effect
    - optionally require triangulation.accepted when available
    """
    est, meta = parse_mediation_result(shared_mediation_result)
    if not mediation_direction_consistent(
        sufficiency_effect=sufficiency_effect,
        necessity_effect=necessity_effect,
        estimate=est,
    ):
        return False
    tri = meta.get("triangulation") or {}
    if require_triangulation_accepted and isinstance(tri, dict) and tri:
        accepted = tri.get("accepted")
        if accepted is False:
            return False
    return True

