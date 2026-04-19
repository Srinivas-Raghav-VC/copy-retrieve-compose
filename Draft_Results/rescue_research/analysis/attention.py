from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


def _iter_json_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return sorted(root.glob("**/*.json"))


def _safe_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _read_json_dict(path: Path) -> Dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def compute_attention_control_summary(out_dir: Path) -> Dict:
    """
    Aggregate attention-control statistics from intervention artifacts.

    Expected schema:
      artifacts/interventions/<model>/<pair>/<seed>.json
      with topk_aggregate[topk].mean_pe_attention etc.
    """
    root = out_dir / "artifacts" / "interventions"
    rows: List[Dict] = []
    for p in _iter_json_files(root):
        payload = _read_json_dict(p)
        if payload is None:
            continue
        pair_id = str(payload.get("pair_id", ""))
        model = str(payload.get("model", ""))
        seed = payload.get("seed")
        topk_agg = payload.get("topk_aggregate", {})
        if not isinstance(topk_agg, dict):
            continue
        for topk, stats in topk_agg.items():
            if not isinstance(stats, dict):
                continue
            row = {
                "pair_id": pair_id,
                "model": model,
                "seed": seed,
                "topk": int(topk) if str(topk).isdigit() else topk,
                "mean_pe": _safe_float(stats.get("mean_pe")),
                "mean_pe_corrupt": _safe_float(stats.get("mean_pe_corrupt")),
                "mean_pe_attention": _safe_float(stats.get("mean_pe_attention")),
                "mean_pe_random": _safe_float(stats.get("mean_pe_random")),
                "mean_pe_shuffle": _safe_float(stats.get("mean_pe_shuffle")),
                "mean_pe_gauss": _safe_float(stats.get("mean_pe_gauss")),
                "mean_pe_basis": _safe_float(stats.get("mean_pe_basis")),
            }
            tm = stats.get("task_matched_control", {})
            if isinstance(tm, dict):
                row["mean_pe_minus_corrupt"] = _safe_float(
                    tm.get("mean_pe_minus_corrupt")
                )
            else:
                row["mean_pe_minus_corrupt"] = None
            rows.append(row)

    def mean(vals: List[float]) -> float | None:
        clean = [v for v in vals if v is not None]
        return (sum(clean) / len(clean)) if clean else None

    summary = {
        "n_rows": len(rows),
        "mean_pe": mean([r["mean_pe"] for r in rows]),
        "mean_pe_corrupt": mean([r["mean_pe_corrupt"] for r in rows]),
        "mean_pe_attention": mean([r["mean_pe_attention"] for r in rows]),
        "mean_pe_random": mean([r["mean_pe_random"] for r in rows]),
        "mean_pe_shuffle": mean([r["mean_pe_shuffle"] for r in rows]),
        "mean_pe_gauss": mean([r["mean_pe_gauss"] for r in rows]),
        "mean_pe_basis": mean([r["mean_pe_basis"] for r in rows]),
        "mean_pe_minus_corrupt": mean([r["mean_pe_minus_corrupt"] for r in rows]),
    }

    grouped: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in rows:
        key = f"{r['pair_id']}::{r['model']}"
        for metric in (
            "mean_pe",
            "mean_pe_corrupt",
            "mean_pe_attention",
            "mean_pe_random",
            "mean_pe_shuffle",
            "mean_pe_gauss",
            "mean_pe_basis",
            "mean_pe_minus_corrupt",
        ):
            v = r.get(metric)
            if v is not None:
                grouped[key][metric].append(v)

    by_pair_model: List[Dict] = []
    for key in sorted(grouped.keys()):
        pair_id, model = key.split("::", 1)
        metrics = grouped[key]
        by_pair_model.append(
            {
                "pair_id": pair_id,
                "model": model,
                **{
                    m: (sum(vals) / len(vals) if vals else None)
                    for m, vals in metrics.items()
                },
            }
        )

    return {"summary": summary, "rows": rows, "by_pair_model": by_pair_model}


def compute_and_save_attention_control_summary(out_dir: Path) -> Path:
    payload = compute_attention_control_summary(out_dir)
    out_path = out_dir / "artifacts" / "stats" / "attention_control_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path
