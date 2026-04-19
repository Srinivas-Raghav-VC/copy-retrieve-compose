#!/usr/bin/env python3
"""
Make summary figures from paper2 matrix outputs.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _model_order(model: str) -> int:
    m = str(model).lower().strip()
    order = {"270m": 0, "1b": 1, "4b": 2, "12b": 3}
    return order.get(m, 99)


def main() -> int:
    ap = argparse.ArgumentParser(description="Make Paper 2 summary figures.")
    ap.add_argument(
        "--matrix-csv",
        type=str,
        default="paper2_fidelity_calibrated/results/matrix_summary.csv",
    )
    ap.add_argument(
        "--fig-dir",
        type=str,
        default="paper2_fidelity_calibrated/figures",
    )
    args = ap.parse_args()

    csv_path = Path(args.matrix_csv).resolve()
    fig_dir = Path(args.fig_dir).resolve()
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing matrix csv: {csv_path}")
    rows = _read_csv(csv_path)
    if not rows or (len(rows) == 1 and "empty" in rows[0]):
        raise RuntimeError("Matrix CSV is empty. Run summarize_matrix.py after running experiments.")

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(f"matplotlib unavailable: {e}")

    # Figure 1: scaling curve (best eval mean PE) per pair.
    per_pair = defaultdict(list)
    for r in rows:
        pair = r.get("pair", "unknown")
        model = r.get("model", "unknown")
        pe = _f(r.get("aggregate_best_eval_mean_pe_mean"))
        per_pair[pair].append((model, pe))

    plt.figure(figsize=(8, 4))
    for pair, pts in sorted(per_pair.items()):
        pts.sort(key=lambda t: _model_order(t[0]))
        xs = list(range(len(pts)))
        ys = [p[1] for p in pts]
        labels = [p[0] for p in pts]
        plt.plot(xs, ys, marker="o", linewidth=1.5, label=pair)
        for x, y, lab in zip(xs, ys, labels):
            if np.isfinite(y):
                plt.text(x, y, lab, fontsize=8, ha="center", va="bottom")
    plt.axhline(0.0, color="black", linewidth=1)
    plt.title("Paper 2: Best held-out rescue effect vs model size (mean PE)")
    plt.xlabel("Model size (ordered)")
    plt.ylabel("Mean PE on eval (best selected config)")
    plt.legend()
    out1 = fig_dir / "fig_paper2_scaling_curve.png"
    plt.tight_layout()
    plt.savefig(out1, dpi=150)
    plt.close()

    # Figure 2: variant mode counts.
    variant_counts = defaultdict(int)
    for r in rows:
        v = str(r.get("best_variant_mode", "")).strip() or "unknown"
        variant_counts[v] += 1
    labels = list(variant_counts.keys())
    values = [variant_counts[k] for k in labels]
    plt.figure(figsize=(7, 3))
    plt.bar(labels, values)
    plt.title("Paper 2: Most-selected variant (mode over seeds)")
    plt.xlabel("Variant")
    plt.ylabel("Count (pair x model jobs)")
    plt.xticks(rotation=20, ha="right")
    out2 = fig_dir / "fig_paper2_variant_mode_counts.png"
    plt.tight_layout()
    plt.savefig(out2, dpi=150)
    plt.close()

    # Figure 3: selection score vs eval PE.
    xs = [_f(r.get("aggregate_best_selection_score_mean")) for r in rows]
    ys = [_f(r.get("aggregate_best_eval_mean_pe_mean")) for r in rows]
    plt.figure(figsize=(5, 4))
    plt.scatter(xs, ys, s=25, alpha=0.8)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.title("Paper 2: Selection score vs held-out eval PE")
    plt.xlabel("Selection score (mean over seeds)")
    plt.ylabel("Eval mean PE (mean over seeds)")
    out3 = fig_dir / "fig_paper2_selection_vs_eval.png"
    plt.tight_layout()
    plt.savefig(out3, dpi=150)
    plt.close()

    print(f"[figures] wrote: {out1}")
    print(f"[figures] wrote: {out2}")
    print(f"[figures] wrote: {out3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

