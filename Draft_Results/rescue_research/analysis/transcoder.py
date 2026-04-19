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


def compute_transcoder_variant_summary(out_dir: Path) -> Dict:
    """
    Aggregate affine-vs-skipless transcoder comparison artifacts.
    """
    root = out_dir / "artifacts" / "variants"
    rows: List[Dict] = []
    for p in _iter_json_files(root):
        payload = _read_json_dict(p)
        if payload is None:
            continue
        rows.append(
            {
                "pair_id": str(payload.get("pair_id", "")),
                "model": str(payload.get("model", "")),
                "variant": str(payload.get("variant", "")),
                "mean_pe": _safe_float(payload.get("mean_pe")),
                "n_samples": payload.get("n_samples"),
            }
        )

    keyed: Dict[str, Dict[str, float]] = defaultdict(dict)
    for r in rows:
        k = f"{r['pair_id']}::{r['model']}"
        if r["mean_pe"] is not None:
            keyed[k][r["variant"]] = r["mean_pe"]

    deltas: List[Dict] = []
    for k, variants in sorted(keyed.items()):
        pair_id, model = k.split("::", 1)
        affine = variants.get("affine_skip")
        skipless = variants.get("skipless_or_non_affine")
        delta = None
        if affine is not None and skipless is not None:
            delta = float(affine - skipless)
        deltas.append(
            {
                "pair_id": pair_id,
                "model": model,
                "affine_skip_mean_pe": affine,
                "skipless_mean_pe": skipless,
                "delta_affine_minus_skipless": delta,
            }
        )

    valid_deltas = [d["delta_affine_minus_skipless"] for d in deltas if d["delta_affine_minus_skipless"] is not None]
    summary = {
        "n_rows": len(rows),
        "n_pair_model_deltas": len(valid_deltas),
        "mean_delta_affine_minus_skipless": (
            sum(valid_deltas) / len(valid_deltas) if valid_deltas else None
        ),
    }
    return {"summary": summary, "rows": rows, "pair_model_deltas": deltas}


def compute_and_save_transcoder_variant_summary(out_dir: Path) -> Path:
    payload = compute_transcoder_variant_summary(out_dir)
    out_path = out_dir / "artifacts" / "stats" / "transcoder_variant_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path
