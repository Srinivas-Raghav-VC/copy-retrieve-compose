#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PAIRS = ["aksharantar_hin_latin", "aksharantar_tel_latin"]
MODELS = ["1b", "4b"]


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Validate presence/basic integrity of all expected Paper-2 artifacts.")
    ap.add_argument("--out", type=str, default="")
    return ap.parse_args()


def _expectations() -> List[Path]:
    paths: List[Path] = []
    for model in MODELS:
        for pair in PAIRS:
            base = PROJECT_ROOT / "paper2_fidelity_calibrated" / "results"
            paths += [
                base / "logit_lens_rescue_trajectory" / pair / model / "logit_lens_rescue_trajectory.json",
                base / "causal_head_attention_patterns" / pair / model / "causal_head_attention_patterns.json",
                base / "language_script_feature_suppression" / pair / model / "language_script_feature_suppression.json",
                base / "head_to_mlp_edge_attribution" / pair / model / "head_to_mlp_edge_attribution.json",
                base / "head_to_mlp_edge_attribution" / pair / model / "g3_mediation_fraction_summary.json",
                base / "feature_knockout_panel" / pair / model / "feature_knockout_panel.json",
                base / "position_shift_sanity" / pair / model / "position_shift_sanity.json",
                base / "feature_stability_resamples" / pair / model / "feature_stability_resamples.json",
                base / "circuit_sufficiency" / pair / model / "circuit_sufficiency.json",
                base / "icl_contribution_curve" / pair / model / "icl_contribution_curve.json",
                base / "minimality_curve" / pair / model / "minimality_curve.json",
                base / "transcoder_family_consensus" / pair / model / "transcoder_family_consensus.json",
                base / "direct_icl_feature_necessity" / pair / model / "direct_icl_feature_necessity.json",
                base / "patch_geometry_robustness" / pair / model / "patch_geometry_robustness.json",
                base / "cross_artifact_feature_stability" / pair / model / "cross_artifact_feature_stability.json",
                base / "target_competitor_logit_gap" / pair / model / "target_competitor_logit_gap.json",
                base / "leave_k_out_icl_contribution" / pair / model / "leave_k_out_icl_contribution.json",
                base / "decoded_vs_latent_equivalence" / pair / model / "decoded_vs_latent_equivalence.json",
                base / "neighbor_layer_causality" / pair / model / "neighbor_layer_causality.json",
                base / "dense_mlp_sweep" / pair / model / "dense_layer_sweep_results.json",
                base / "activation_difference_baseline" / pair / model / "activation_difference_baseline.json",
                base / "head_attribution_stability" / pair / model / "head_attribution_stability.json",
                base / "neutral_filler_recency_controls" / pair / model / "neutral_filler_recency_controls.json",
                base / "induction_style_head_reanalysis" / pair / model / "induction_style_head_reanalysis.json",
                base / "selected_layer_fidelity_compare" / pair / model / "selected_layer_fidelity_compare.json",
            ]
        paths += [
            PROJECT_ROOT / "paper2_fidelity_calibrated" / "results" / "cfom_function_vector_tests" / model / "cfom_function_vector_tests.json",
            PROJECT_ROOT / "paper2_fidelity_calibrated" / "results" / "sparse_feature_circuit" / model / "sparse_feature_circuit.json",
        ]
    paths += [
        PROJECT_ROOT / "paper2_fidelity_calibrated" / "results" / "belief_dynamics_fit" / "belief_dynamics_fit.json",
    ]
    return paths


def _primary_count(obj: Any) -> int:
    if isinstance(obj, list):
        return len(obj)
    if not isinstance(obj, dict):
        return 0
    for key in ["item_rows", "summary", "summary_by_head", "summary_by_feature_index", "fits", "pairwise_jaccard", "resamples_detail", "rows"]:
        if isinstance(obj.get(key), list):
            return len(obj.get(key) or [])
        if isinstance(obj.get(key), dict):
            return len(obj.get(key) or {})
    return len(obj)


def main() -> int:
    args = parse_args()
    statuses: List[Dict[str, Any]] = []
    for path in _expectations():
        row: Dict[str, Any] = {"path": str(path.relative_to(PROJECT_ROOT))}
        if not path.exists():
            row.update({"status": "missing", "count": 0})
            statuses.append(row)
            continue
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            count = _primary_count(obj)
            row.update({"status": "ok" if count > 0 else "empty", "count": int(count)})
        except Exception as exc:
            row.update({"status": "parse_error", "error": str(exc), "count": 0})
        statuses.append(row)

    summary = {
        "expected": int(len(statuses)),
        "ok": int(sum(1 for r in statuses if r["status"] == "ok")),
        "missing": int(sum(1 for r in statuses if r["status"] == "missing")),
        "empty": int(sum(1 for r in statuses if r["status"] == "empty")),
        "parse_error": int(sum(1 for r in statuses if r["status"] == "parse_error")),
    }
    out_path = Path(args.out).resolve() if str(args.out).strip() else PROJECT_ROOT / "paper2_fidelity_calibrated" / "results" / "artifact_integrity_validator" / "artifact_integrity_validator.json"
    payload = {"experiment": "artifact_integrity_validator", "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "summary": summary, "artifacts": statuses}
    _write_json(out_path, payload)
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
