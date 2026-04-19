from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def _read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _variant_evidence_available(out_dir: Path) -> bool:
    payload = _read_json(out_dir / "artifacts" / "stats" / "transcoder_variant_summary.json")
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    try:
        return int(summary.get("n_pair_model_deltas", 0)) > 0
    except Exception:
        return False


def _quality_evidence_available(out_dir: Path) -> bool:
    baseline_root = out_dir / "artifacts" / "baseline"
    if not baseline_root.exists():
        return False
    for p in sorted(baseline_root.glob("**/*.json")):
        payload = _read_json(p)
        stage = payload.get("stage_output", {}) if isinstance(payload, dict) else {}
        stats = stage.get("stats", {}) if isinstance(stage, dict) else {}
        if stats.get("mean_cer_icl") is not None or stats.get("exact_match_icl") is not None:
            return True
    return False


def build_claim_evidence_matrix(out_dir: Path | None = None) -> List[Dict[str, str]]:
    matrix: List[Dict[str, str]] = [
        {
            "claim_id": "C1_confirmatory",
            "claim": "Selected internal features causally mediate ICL rescue.",
            "evidence": (
                "table_2_confirmatory_tests.csv; figure_3_sufficiency_necessity.png; "
                "figure_5_mediation_decomposition.png; table_8_attention_controls.csv"
            ),
            "status": "confirmatory",
            "support_status": "unverified",
            "support_notes": "Requires publication rubric decision.",
        },
        {
            "claim_id": "C2_generalization",
            "claim": "Directionality is consistent across script families.",
            "evidence": "table_2_confirmatory_tests.csv; figure_2_baseline_rescue.png",
            "status": "secondary",
            "support_status": "unverified",
            "support_notes": "Requires confirmatory directional consistency outputs.",
        },
        {
            "claim_id": "C3_quality_alignment",
            "claim": "PE improvements align with CER/exact-match quality gains.",
            "evidence": "figure_6_pe_vs_quality.png",
            "status": "secondary",
            "support_status": "unverified",
            "support_notes": "Requires quality evaluation artifacts.",
        },
        {
            "claim_id": "C4_transcoder_architecture",
            "claim": "Rescue effect depends on transcoder architecture details (affine vs skipless).",
            "evidence": (
                "table_9_transcoder_variants.csv; figure_9_transcoder_variant_deltas.png; "
                "figure_4_affine_vs_skip.png"
            ),
            "status": "secondary",
            "support_status": "unverified",
            "support_notes": "Requires variant-comparison artifacts.",
        },
    ]
    if out_dir is None:
        return matrix

    decision = _read_json(out_dir / "artifacts" / "final" / "publication_decision.json")
    eligible = decision.get("eligible", None)
    branch = str(decision.get("branch", "")).strip() or "unknown"
    protocol_compliance = decision.get("protocol_compliance_passed", None)
    fidelity_gate = decision.get("fidelity_gate_passed", None)
    variant_ok = _variant_evidence_available(out_dir)
    quality_ok = _quality_evidence_available(out_dir)

    for row in matrix:
        row["publication_branch"] = branch
        if protocol_compliance is not None:
            row["protocol_compliance_passed"] = bool(protocol_compliance)
        if fidelity_gate is not None:
            row["fidelity_gate_passed"] = bool(fidelity_gate)
        if row["claim_id"] == "C1_confirmatory":
            if protocol_compliance is False:
                row["support_status"] = "unsupported_protocol"
                row["support_notes"] = "Protocol compliance failed; confirmatory claim is not admissible."
            elif fidelity_gate is False:
                row["support_status"] = "unsupported_protocol"
                row["support_notes"] = "Transcoder fidelity gate failed; confirmatory mediator claim is not admissible."
            elif eligible is True:
                row["support_status"] = "supported"
                row["support_notes"] = "Main-track rubric eligible."
            elif eligible is False:
                row["support_status"] = "unsupported_rubric"
                row["support_notes"] = "Main-track rubric failed; do not present as confirmatory."
            else:
                row["support_status"] = "unknown"
                row["support_notes"] = "Publication decision missing."
        elif row["claim_id"] == "C3_quality_alignment":
            if quality_ok:
                row["support_status"] = "supported"
                row["support_notes"] = "CER/exact-match quality artifacts found."
            else:
                row["support_status"] = "missing_evidence"
                row["support_notes"] = "Run with --run-quality-eval to support this claim."
        elif row["claim_id"] == "C4_transcoder_architecture":
            if variant_ok:
                row["support_status"] = "supported"
                row["support_notes"] = "Variant comparison deltas available."
            else:
                row["support_status"] = "missing_evidence"
                row["support_notes"] = "Run with --compare-variants; no variant deltas currently available."
        else:
            if protocol_compliance is False:
                row["support_status"] = "exploratory_only"
                row["support_notes"] = "Protocol compliance failed; treat only as exploratory evidence."
            elif eligible is False:
                row["support_status"] = "exploratory_only"
                row["support_notes"] = "Use as exploratory evidence only (fallback branch)."
            elif eligible is True:
                row["support_status"] = "supported"
                row["support_notes"] = "Compatible with eligible main-track branch."
            else:
                row["support_status"] = "unknown"
                row["support_notes"] = "Publication decision missing."
    return matrix


def write_claim_matrix(path: Path, *, out_dir: Path | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = build_claim_evidence_matrix(out_dir=out_dir)
    path.write_text(json.dumps(matrix, indent=2), encoding="utf-8")
