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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Summarize mediation fractions from G3 outputs.")
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--out", type=str, default="")
    return ap.parse_args()


def _fraction(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or abs(float(den)) <= 1e-12:
        return float("nan")
    return float(num / den)


def main() -> int:
    args = parse_args()
    in_path = Path(str(args.input)).resolve()
    obj = json.loads(in_path.read_text(encoding="utf-8"))
    rows = list(obj.get("item_rows") or [])
    summary_rows: List[Dict[str, Any]] = []
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        key = f"L{int(row['head_layer']):02d}H{int(row['head']):02d}"
        grouped.setdefault(key, []).append(row)
    for key, bucket in sorted(grouped.items()):
        fracs = []
        rand_fracs = []
        for row in bucket:
            fracs.append(_fraction(float(row.get("feature_mediated_drop_first_prob", float("nan"))), float(row.get("head_only_delta_first_prob", float("nan")))))
            rand_fracs.append(_fraction(float(row.get("random_feature_mediated_drop_first_prob", float("nan"))), float(row.get("head_only_delta_first_prob", float("nan")))))
        summary_rows.append(
            {
                "head": key,
                "n_items": int(len(bucket)),
                "mean_mediation_fraction_first_prob": float(np.nanmean(fracs)),
                "mean_random_mediation_fraction_first_prob": float(np.nanmean(rand_fracs)),
            }
        )
    out_path = Path(args.out).resolve() if str(args.out).strip() else in_path.with_name("g3_mediation_fraction_summary.json")
    payload = {
        "experiment": "g3_mediation_fraction_summary",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input": str(in_path),
        "summary": summary_rows,
    }
    _write_json(out_path, payload)
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
