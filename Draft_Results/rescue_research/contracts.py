"""
Executable protocol contracts for the main-track rescue pipeline.

This module codifies plan-level decisions as data structures and pure functions
so pipeline code can enforce them consistently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping


CONFIRMATORY_CLAIM = (
    "A pre-registered set of internal features causally mediates ICL rescue in "
    "OOD cross-script transliteration under sufficiency and necessity tests."
)

SECONDARY_CLAIMS = [
    "The mechanism direction generalizes across script families.",
    "Mechanism-strength metrics align with transliteration quality metrics.",
]

EXPLORATORY_BOUNDARY = (
    "Any additional probes, prompt variants, pair additions, or unregistered "
    "metric families are exploratory and must not support the confirmatory claim."
)


LOCKED_LANGUAGE_PAIRS = (
    "hindi_telugu",
    "hindi_tamil",
    "english_arabic",
    "english_cyrillic",
)

# Pre-approved substitution candidates. These are only legal before Gate A.
PAIR_BACKUPS: Mapping[str, tuple[str, str]] = {
    "hindi_telugu": ("hindi_kannada", "hindi_malayalam"),
    "hindi_tamil": ("hindi_bengali", "hindi_gujarati"),
    "english_arabic": ("english_hebrew", "english_katakana"),
    "english_cyrillic": ("english_georgian", "english_devanagari"),
}

# Confirmatory split targets (per pair, per seed).
# Submission profile (depth-balanced): 100 / 400 / 1000 / 200
TARGET_SPLIT_COUNTS = {
    "icl_bank": 100,
    "selection": 400,
    "eval_open": 1000,
    "eval_blind": 200,
}

# Prompt policy
CONFIRMATORY_K_SHOT = 8
EXPLORATORY_K_OPTIONS = (4, 12)

# Confirmatory hypothesis family
HYPOTHESES = ("H1", "H2", "H3")
GLOBAL_ALPHA = 0.05


@dataclass(frozen=True)
class HypothesisResult:
    hypothesis_id: str
    passed: bool
    adjusted_p_value: float


@dataclass
class MainTrackRubricInput:
    """Minimal evidence required to evaluate the hard submission rubric."""

    hypothesis_results: List[HypothesisResult]
    practical_floor_passed: bool
    directional_pair_count: int
    controls_passed: bool
    reproducibility_passed: bool


@dataclass
class MainTrackRubricDecision:
    eligible: bool
    failed_rules: List[str] = field(default_factory=list)


def validate_locked_pair_matrix(pairs: Iterable[str]) -> None:
    provided = tuple(pairs)
    if provided != LOCKED_LANGUAGE_PAIRS:
        raise ValueError(
            f"Pair matrix mismatch. Expected {LOCKED_LANGUAGE_PAIRS}, got {provided}."
        )


def evaluate_main_track_rubric(payload: MainTrackRubricInput) -> MainTrackRubricDecision:
    """
    Enforce hard rubric R1-R5 from the plan.

    R1: H1/H2/H3 all pass Holm-adjusted confirmatory family.
    R2: practical significance floor passes.
    R3: directional consistency in >=3/4 locked pairs.
    R4: controls pass.
    R5: reproducibility pass.
    """
    failed: List[str] = []

    hyp_by_id = {h.hypothesis_id: h for h in payload.hypothesis_results}
    for h in HYPOTHESES:
        if h not in hyp_by_id or not hyp_by_id[h].passed:
            failed.append(f"R1:{h}")

    if not payload.practical_floor_passed:
        failed.append("R2")
    if payload.directional_pair_count < 3:
        failed.append("R3")
    if not payload.controls_passed:
        failed.append("R4")
    if not payload.reproducibility_passed:
        failed.append("R5")

    return MainTrackRubricDecision(eligible=not failed, failed_rules=failed)


def default_hypothesis_registry() -> Dict[str, str]:
    return {
        "H1": "DeltaPE > 0 for selected features vs corrupt control.",
        "H2": "Necessity ablation reduces rescue vs intact ICL.",
        "H3": (
            "Mediated component (NIE) is positive and directionally consistent "
            "with H1/H2. Note: nie_ratio is exploratory only due to nonlinear "
            "decomposition (Mueller et al. 2024)."
        ),
    }


@dataclass
class SubstitutionTrigger:
    data_audit_error_rate: float
    remediation_cycles: int
    unresolved_licensing_risk: bool
    effective_pool_below_minimum: bool
    gate_name: str
    substitutions_already_used: int


def substitution_allowed(trigger: SubstitutionTrigger) -> bool:
    """
    Enforce pre-approved substitution policy:
    - only before Gate A,
    - max one substitution total,
    - at least one objective trigger is active.
    """
    if trigger.gate_name.strip().upper() != "PRE_GATE_A":
        return False
    if trigger.substitutions_already_used >= 1:
        return False
    objective_trigger = (
        (trigger.data_audit_error_rate > 0.02 and trigger.remediation_cycles >= 1)
        or trigger.unresolved_licensing_risk
        or trigger.effective_pool_below_minimum
    )
    return bool(objective_trigger)


def validate_preapproved_substitution_matrix(pairs: Iterable[str]) -> List[dict]:
    """
    Validate pair substitutions against the pre-approved policy.

    Rules:
    - pair list must preserve locked matrix length/order semantics,
    - substitutions can only use pre-approved backups at each locked slot,
    - max one substitution total.
    """
    provided = tuple(str(p).strip() for p in pairs)
    if len(provided) != len(LOCKED_LANGUAGE_PAIRS):
        raise ValueError(
            "Custom pair matrix must keep the same length as locked matrix "
            f"({len(LOCKED_LANGUAGE_PAIRS)}). Got {len(provided)}."
        )
    if len(set(provided)) != len(provided):
        raise ValueError("Custom pair matrix contains duplicate pair IDs.")

    substitutions: List[dict] = []
    for expected, selected in zip(LOCKED_LANGUAGE_PAIRS, provided):
        if selected == expected:
            continue
        allowed = tuple(PAIR_BACKUPS.get(expected, ()))
        if selected not in allowed:
            raise ValueError(
                f"Invalid substitution for '{expected}': '{selected}'. "
                f"Allowed backups: {allowed}."
            )
        substitutions.append(
            {
                "from_locked_pair": expected,
                "to_substitute_pair": selected,
            }
        )

    if len(substitutions) > 1:
        raise ValueError(
            f"At most one substitution is allowed. Found {len(substitutions)} substitutions: {substitutions}."
        )
    return substitutions

