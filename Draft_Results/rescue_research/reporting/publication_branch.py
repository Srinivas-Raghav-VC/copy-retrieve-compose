from __future__ import annotations

from pathlib import Path
import json

from rescue_research.contracts import (
    HypothesisResult,
    MainTrackRubricInput,
    evaluate_main_track_rubric,
)


def decide_publication_branch(
    *,
    out_path: Path,
    h1_pass: bool,
    h2_pass: bool,
    h3_pass: bool,
    practical_floor_passed: bool,
    directional_pair_count: int,
    controls_passed: bool,
    reproducibility_passed: bool,
    protocol_compliance_passed: bool | None = None,
    protocol_notes: list[str] | None = None,
    fidelity_gate_passed: bool | None = None,
) -> str:
    decision = evaluate_main_track_rubric(
        MainTrackRubricInput(
            hypothesis_results=[
                HypothesisResult("H1", h1_pass, 1.0),
                HypothesisResult("H2", h2_pass, 1.0),
                HypothesisResult("H3", h3_pass, 1.0),
            ],
            practical_floor_passed=practical_floor_passed,
            directional_pair_count=directional_pair_count,
            controls_passed=controls_passed,
            reproducibility_passed=reproducibility_passed,
        )
    )
    branch = "main_track" if decision.eligible else "fallback"
    payload = {
        "branch": branch,
        "eligible": decision.eligible,
        "failed_rules": decision.failed_rules,
        "rubric_inputs": {
            "h1_pass": bool(h1_pass),
            "h2_pass": bool(h2_pass),
            "h3_pass": bool(h3_pass),
            "practical_floor_passed": bool(practical_floor_passed),
            "directional_pair_count": int(directional_pair_count),
            "controls_passed": bool(controls_passed),
            "reproducibility_passed": bool(reproducibility_passed),
        },
    }
    if protocol_compliance_passed is not None:
        payload["protocol_compliance_passed"] = bool(protocol_compliance_passed)
    if protocol_notes is not None:
        payload["protocol_notes"] = list(protocol_notes)
    if fidelity_gate_passed is not None:
        payload["fidelity_gate_passed"] = bool(fidelity_gate_passed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return branch

