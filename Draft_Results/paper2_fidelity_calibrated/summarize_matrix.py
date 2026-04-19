#!/usr/bin/env python3
"""
Aggregate all Paper 2 JSON outputs into one summary table.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _safe_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _f(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _best_choices(payload: Dict[str, Any]) -> Tuple[str, float, str, float]:
    """
    Returns:
      (best_variant_mode, best_layer_mean, best_topk_mode, best_sel_score_mean)
    """
    seeds = payload.get("seeds") or {}
    variants: List[str] = []
    layers: List[int] = []
    topks: List[int] = []
    scores: List[float] = []
    for seed, info in seeds.items():
        b = (info or {}).get("best") or {}
        v = str(b.get("variant", "")).strip()
        if v:
            variants.append(v)
        try:
            layers.append(int(b.get("layer")))
        except Exception:
            pass
        try:
            topks.append(int(b.get("topk")))
        except Exception:
            pass
        scores.append(_f(b.get("score")))

    v_mode = Counter(variants).most_common(1)[0][0] if variants else ""
    k_mode = str(Counter(topks).most_common(1)[0][0]) if topks else ""
    layer_mean = float(sum(layers) / len(layers)) if layers else float("nan")
    score_mean = float(sum([s for s in scores if s == s]) / max(1, len([s for s in scores if s == s])))  # NaN-safe
    return v_mode, layer_mean, k_mode, score_mean


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize paper2 matrix outputs.")
    ap.add_argument(
        "--results-root",
        type=str,
        default="paper2_fidelity_calibrated/results",
        help="Root directory containing paper2_fidelity_calibrated_*.json outputs.",
    )
    ap.add_argument(
        "--output-csv",
        type=str,
        default="paper2_fidelity_calibrated/results/matrix_summary.csv",
    )
    ap.add_argument(
        "--output-json",
        type=str,
        default="paper2_fidelity_calibrated/results/matrix_summary.json",
    )
    args = ap.parse_args()

    root = Path(args.results_root).resolve()
    files = sorted(root.rglob("paper2_fidelity_calibrated_*.json"))

    rows: List[Dict[str, Any]] = []
    for path in files:
        payload = _safe_json(path)
        if not payload:
            continue
        pair = str(payload.get("pair", "unknown"))
        model = str(payload.get("model_key", "unknown"))
        agg = payload.get("aggregate") or {}
        v_mode, layer_mean, k_mode, score_mean = _best_choices(payload)
        rows.append(
            {
                "path": str(path),
                "pair": pair,
                "model": model,
                "best_variant_mode": v_mode,
                "best_layer_mean": layer_mean,
                "best_topk_mode": k_mode,
                "best_selection_score_mean_from_seeds": score_mean,
                "aggregate_best_selection_score_mean": _f(agg.get("best_selection_score_mean")),
                "aggregate_best_eval_mean_pe_mean": _f(agg.get("best_eval_mean_pe_mean")),
                "aggregate_best_eval_mean_pe_std": _f(agg.get("best_eval_mean_pe_std")),
            }
        )

    out_csv = Path(args.output_csv).resolve()
    out_json = Path(args.output_json).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    if rows:
        cols = sorted(rows[0].keys())
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)
    else:
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["empty"])

    out_json.write_text(json.dumps({"n_files": len(files), "rows": rows}, indent=2), encoding="utf-8")
    print(f"[summary] files found: {len(files)}")
    print(f"[summary] csv: {out_csv}")
    print(f"[summary] json: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

