#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PAIRS = ["aksharantar_hin_latin", "aksharantar_tel_latin"]
MODELS = ["1b", "4b"]


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build a single cross-experiment synthesis table.")
    ap.add_argument("--uncertainty-json", type=str, default="")
    ap.add_argument("--out-dir", type=str, default="")
    return ap.parse_args()


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_mean(metrics: Dict[str, Any], key: str) -> float:
    obj = metrics.get(key) or {}
    try:
        return float(obj.get("mean", float("nan")))
    except Exception:
        return float("nan")


def _score_cell(metrics: Dict[str, Any]) -> Dict[str, Any]:
    # simple transparent heuristic, not a truth oracle
    positives = {
        "dense_positive_control_first_prob": _metric_mean(metrics, "dense_positive_control_first_prob"),
        "actdiff_layer_output_first_prob": _metric_mean(metrics, "actdiff_layer_output_first_prob"),
        "actdiff_mlp_output_first_prob": _metric_mean(metrics, "actdiff_mlp_output_first_prob"),
        "g3_head_effect_first_prob": _metric_mean(metrics, "g3_head_effect_first_prob"),
        "g3_mediation_specific_first_prob": _metric_mean(metrics, "g3_mediation_specific_first_prob"),
        "g6_feature_necessity_first_prob": _metric_mean(metrics, "g6_feature_necessity_first_prob"),
        "direct_icl_specific_necessity_first_prob": _metric_mean(metrics, "direct_icl_specific_necessity_first_prob"),
        "function_vector_specificity_first_prob": _metric_mean(metrics, "function_vector_specificity_first_prob"),
        "position_center_minus_neighbors": _metric_mean(metrics, "position_center_minus_neighbors"),
        "neighbor_layer_center_minus_adjacent": _metric_mean(metrics, "neighbor_layer_center_minus_adjacent"),
        "neutral_helpful_minus_corrupt_first_prob": _metric_mean(metrics, "neutral_helpful_minus_corrupt_first_prob"),
        "neutral_helpful_minus_null_first_prob": _metric_mean(metrics, "neutral_helpful_minus_null_first_prob"),
        "recency_similarity_desc_minus_asc_first_prob": _metric_mean(metrics, "recency_similarity_desc_minus_asc_first_prob"),
        "induction_paired_target_margin_icl": _metric_mean(metrics, "induction_paired_target_margin_icl"),
    }
    caution = {
        "geometry_raw_minus_sign_first_prob": abs(_metric_mean(metrics, "geometry_raw_minus_sign_first_prob")),
        "transcoder_variant_prob_delta": abs(_metric_mean(metrics, "transcoder_variant_prob_delta")),
        "decoded_minus_latent_first_prob": abs(_metric_mean(metrics, "decoded_minus_latent_first_prob")),
    }
    support_count = sum(1 for v in positives.values() if np.isfinite(v) and v > 0.0)
    caution_count = sum(1 for v in caution.values() if np.isfinite(v) and v > 0.05)
    verdict = "mixed_or_unknown"
    if support_count >= 5 and caution_count <= 1:
        verdict = "supports_main_claim"
    elif support_count >= 3:
        verdict = "supports_appendix_or_secondary_claim"
    elif caution_count >= 2:
        verdict = "negative_or_constraining"
    return {"support_count": int(support_count), "caution_count": int(caution_count), "verdict": verdict}


def main() -> int:
    args = parse_args()
    unc_path = Path(args.uncertainty_json).resolve() if str(args.uncertainty_json).strip() else PROJECT_ROOT / "paper2_fidelity_calibrated" / "results" / "uncertainty_estimates" / "uncertainty_estimates.json"
    obj = _read(unc_path)
    rows = list(obj.get("rows") or [])
    out_dir = Path(args.out_dir).resolve() if str(args.out_dir).strip() else PROJECT_ROOT / "paper2_fidelity_calibrated" / "results" / "cross_experiment_synthesis"
    out_dir.mkdir(parents=True, exist_ok=True)

    table_rows: List[Dict[str, Any]] = []
    md = ["# Cross-Experiment Synthesis Table", "", "| Model | Pair | Dense | ActDiff(L) | ActDiff(M) | G3 med. | G6 nec. | ICL nec. | G11 spec. | Pos shift | Neighbor | Consensus | Verdict |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        metrics = dict(row.get("metrics") or {})
        verdict = _score_cell(metrics)
        out = {
            "model": str(row.get("model")),
            "pair": str(row.get("pair")),
            "dense_positive_control_first_prob": _metric_mean(metrics, "dense_positive_control_first_prob"),
            "actdiff_layer_output_first_prob": _metric_mean(metrics, "actdiff_layer_output_first_prob"),
            "actdiff_mlp_output_first_prob": _metric_mean(metrics, "actdiff_mlp_output_first_prob"),
            "g3_mediation_specific_first_prob": _metric_mean(metrics, "g3_mediation_specific_first_prob"),
            "g6_feature_necessity_first_prob": _metric_mean(metrics, "g6_feature_necessity_first_prob"),
            "direct_icl_specific_necessity_first_prob": _metric_mean(metrics, "direct_icl_specific_necessity_first_prob"),
            "function_vector_specificity_first_prob": _metric_mean(metrics, "function_vector_specificity_first_prob"),
            "position_center_minus_neighbors": _metric_mean(metrics, "position_center_minus_neighbors"),
            "neighbor_layer_center_minus_adjacent": _metric_mean(metrics, "neighbor_layer_center_minus_adjacent"),
            "transcoder_consensus_jaccard": _metric_mean(metrics, "transcoder_consensus_jaccard"),
            **verdict,
        }
        table_rows.append(out)
        def fmt(v: float) -> str:
            return f"{float(v):.3f}" if np.isfinite(float(v)) else "nan"
        md.append(
            f"| {out['model']} | {out['pair']} | {fmt(out['dense_positive_control_first_prob'])} | {fmt(out['actdiff_layer_output_first_prob'])} | {fmt(out['actdiff_mlp_output_first_prob'])} | {fmt(out['g3_mediation_specific_first_prob'])} | {fmt(out['g6_feature_necessity_first_prob'])} | {fmt(out['direct_icl_specific_necessity_first_prob'])} | {fmt(out['function_vector_specificity_first_prob'])} | {fmt(out['position_center_minus_neighbors'])} | {fmt(out['neighbor_layer_center_minus_adjacent'])} | {fmt(out['transcoder_consensus_jaccard'])} | {out['verdict']} |"
        )

    _write_json(out_dir / "cross_experiment_synthesis.json", {"experiment": "cross_experiment_synthesis", "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rows": table_rows})
    _write_text(out_dir / "cross_experiment_synthesis.md", "\n".join(md) + "\n")
    print(f"Saved: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
