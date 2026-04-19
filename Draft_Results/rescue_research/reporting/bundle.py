from __future__ import annotations

import json
import shutil
from pathlib import Path

from rescue_research.reporting.claim_matrix import write_claim_matrix
from rescue_research.reporting.tables import MANDATORY_TABLES

try:
    from rescue_research.reporting.figures import MANDATORY_FIGURES
except ModuleNotFoundError:
    # Keep bundle creation available in minimal environments where figure deps
    # (e.g., matplotlib) are not installed.
    MANDATORY_FIGURES = []


BUNDLE_REQUIRED_ARTIFACTS = (
    Path("artifacts/stats/confirmatory_results.json"),
    Path("artifacts/stats/exploratory_results.json"),
    Path("artifacts/stats/attention_control_summary.json"),
    Path("artifacts/stats/transcoder_variant_summary.json"),
    Path("artifacts/stats/prompt_format_robustness.json"),
    Path("artifacts/audit/transcoder_fidelity_gate.json"),
    Path("artifacts/audit/skeptic_pass.json"),
    Path("artifacts/audit/protocol_compliance.json"),
    Path("artifacts/audit/reproducibility_check.json"),
    Path("artifacts/audit/submission_gates.json"),
    Path("artifacts/final/publication_decision.json"),
)


def _table_has_placeholder(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return False
    return "(no " in text


def _variant_evidence_available(out_dir: Path) -> bool:
    p = out_dir / "artifacts" / "stats" / "transcoder_variant_summary.json"
    if not p.exists():
        return False
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return False
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    try:
        return int(summary.get("n_pair_model_deltas", 0)) > 0
    except Exception:
        return False


def _missing_required_artifacts(out_dir: Path) -> list[str]:
    missing: list[str] = []
    for rel in BUNDLE_REQUIRED_ARTIFACTS:
        if not (out_dir / rel).exists():
            missing.append(str(rel))
    return missing


def build_submission_bundle(out_dir: Path, *, strict: bool = True) -> Path:
    """
    Build the final submission bundle directory from validated artifacts.

    In strict mode, missing mandatory artifacts/tables/figures raise RuntimeError
    so paper packaging fails loudly instead of silently shipping partial evidence.
    """
    final_dir = out_dir / "artifacts" / "final" / "submission_bundle"
    final_dir.mkdir(parents=True, exist_ok=True)

    # Create claim-evidence matrix.
    write_claim_matrix(final_dir / "claim_evidence_matrix.json", out_dir=out_dir)

    # Copy figures and tables if available.
    fig_src = out_dir / "artifacts" / "figures"
    tbl_src = out_dir / "artifacts" / "tables"
    copied_figures = []
    missing_figures = []
    for fname in MANDATORY_FIGURES:
        src = fig_src / fname
        if src.exists():
            shutil.copy2(src, final_dir / fname)
            copied_figures.append(fname)
        else:
            missing_figures.append(fname)
    copied_tables = []
    missing_tables = []
    for fname in MANDATORY_TABLES:
        src = tbl_src / fname
        if src.exists():
            shutil.copy2(src, final_dir / fname)
            copied_tables.append(fname)
        else:
            missing_tables.append(fname)

    publication_decision_path = out_dir / "artifacts" / "final" / "publication_decision.json"
    publication_decision = {}
    if publication_decision_path.exists():
        try:
            publication_decision = json.loads(
                publication_decision_path.read_text(encoding="utf-8")
            )
        except Exception:
            publication_decision = {}

    table_placeholders = [
        name for name in copied_tables if _table_has_placeholder(final_dir / name)
    ]
    missing_required = _missing_required_artifacts(out_dir)

    readiness = {
        "publication_decision": publication_decision,
        "bundle_completeness": {
            "copied_figures": copied_figures,
            "missing_figures": missing_figures,
            "copied_tables": copied_tables,
            "missing_tables": missing_tables,
            "table_placeholders": table_placeholders,
            "missing_required_artifacts": missing_required,
        },
        "evidence_flags": {
            "variant_evidence_available": _variant_evidence_available(out_dir),
        },
    }
    readiness["ready_for_submission"] = not (
        missing_required or missing_figures or missing_tables
    )

    (final_dir / "artifact_readiness.json").write_text(
        json.dumps(readiness, indent=2),
        encoding="utf-8",
    )

    if strict and not bool(readiness["ready_for_submission"]):
        failures = []
        if missing_required:
            failures.append("missing_required_artifacts=" + ", ".join(missing_required[:20]))
        if missing_tables:
            failures.append("missing_tables=" + ", ".join(missing_tables[:20]))
        if missing_figures:
            failures.append("missing_figures=" + ", ".join(missing_figures[:20]))
        raise RuntimeError(
            "Submission bundle readiness check failed: " + " | ".join(failures)
        )

    return final_dir
