#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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
    ap = argparse.ArgumentParser(description="Fit a simple belief-dynamics curve to the density-crossover artifact.")
    ap.add_argument("--density-json", type=str, default="")
    ap.add_argument("--alpha-min", type=float, default=0.0)
    ap.add_argument("--alpha-max", type=float, default=1.0)
    ap.add_argument("--alpha-step", type=float, default=0.01)
    ap.add_argument("--out", type=str, default="")
    return ap.parse_args()


def _default_density_path() -> Path:
    candidates = [
        PROJECT_ROOT / "artifacts" / "phaseB_density_crossover" / "density_threshold_crossover_1b.json",
        PROJECT_ROOT / "artifacts" / "phase3_transition" / "transition_map_density_crossover_1b.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _clip_prob(p: float) -> float:
    p = float(p)
    if not np.isfinite(p):
        return float("nan")
    return float(min(1.0 - 1e-4, max(1e-4, p)))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _fit_curve(
    points: List[Tuple[float, float]],
    *,
    alpha_min: float,
    alpha_max: float,
    alpha_step: float,
) -> Dict[str, Any]:
    best: Dict[str, Any] | None = None
    xs_raw = np.array([float(x) for x, _ in points], dtype=np.float64)
    ys = np.array([_clip_prob(y) for _, y in points], dtype=np.float64)
    mask = np.isfinite(xs_raw) & np.isfinite(ys)
    xs_raw = xs_raw[mask]
    ys = ys[mask]
    if xs_raw.size < 3:
        return {"n_points": int(xs_raw.size), "error": "not_enough_points"}

    alpha_values = np.arange(float(alpha_min), float(alpha_max) + 0.5 * float(alpha_step), float(alpha_step))
    for alpha_value in alpha_values.tolist():
        alpha = min(float(alpha_max), max(float(alpha_min), float(alpha_value)))
        power = 1.0 - float(alpha)
        xs = np.array([0.0 if float(n) <= 0.0 else float(n) ** power for n in xs_raw.tolist()], dtype=np.float64)
        if not np.all(np.isfinite(xs)):
            continue
        design = np.column_stack([np.ones_like(xs), xs])
        target = np.log(ys / (1.0 - ys))
        try:
            coeffs, *_ = np.linalg.lstsq(design, target, rcond=None)
        except np.linalg.LinAlgError:
            continue
        pred_logit = design @ coeffs
        pred_prob = _sigmoid(pred_logit)
        if not np.all(np.isfinite(pred_prob)):
            continue
        rmse = float(np.sqrt(np.mean((pred_prob - ys) ** 2)))
        ss_res = float(np.sum((pred_prob - ys) ** 2))
        ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")
        row = {
            "alpha": float(alpha),
            "b": float(coeffs[0]),
            "gamma": float(coeffs[1]),
            "rmse": float(rmse),
            "r2": float(r2),
            "n_points": int(xs_raw.size),
            "points": [
                {"n": float(n), "observed": float(y), "predicted": float(p)}
                for n, y, p in zip(xs_raw.tolist(), ys.tolist(), pred_prob.tolist())
            ],
        }
        if best is None or row["rmse"] < best["rmse"]:
            best = row
    return best or {"n_points": int(xs_raw.size), "error": "fit_failed"}


def _extract_points(row: Dict[str, Any], condition: str, metric: str) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    helpful = row.get("helpful_only_summary") or {}
    if metric in helpful:
        points.append((0.0, float(helpful[metric])))
    trajectories = row.get("trajectories") or {}
    for item in list(trajectories.get(condition) or []):
        if metric not in item:
            continue
        n = float(item.get("filler_n", float("nan")))
        y = float(item.get(metric, float("nan")))
        if np.isfinite(n) and np.isfinite(y):
            points.append((n, y))
    points.sort(key=lambda xy: xy[0])
    return points


def main() -> int:
    args = parse_args()

    density_path = Path(str(args.density_json)).resolve() if str(args.density_json).strip() else _default_density_path()
    if not density_path.exists():
        raise FileNotFoundError(f"Missing density artifact: {density_path}")

    rows = json.loads(density_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise RuntimeError(f"Expected top-level list in density artifact: {density_path}")

    fit_rows: List[Dict[str, Any]] = []
    for row in rows:
        pair = str(row.get("pair", ""))
        model = str(row.get("model", ""))
        for condition in ("same_tail", "wrong_tail"):
            for metric in ("exact_match", "first_entry_correct"):
                points = _extract_points(row, condition, metric)
                fit = _fit_curve(
                    points,
                    alpha_min=float(args.alpha_min),
                    alpha_max=float(args.alpha_max),
                    alpha_step=float(args.alpha_step),
                )
                fit_rows.append(
                    {
                        "pair": pair,
                        "model": model,
                        "condition": condition,
                        "metric": metric,
                        "fit": fit,
                    }
                )

    out_path = (
        Path(args.out).resolve()
        if str(args.out).strip()
        else PROJECT_ROOT / "paper2_fidelity_calibrated" / "results" / "belief_dynamics_fit" / "belief_dynamics_fit.json"
    )
    payload = {
        "experiment": "belief_dynamics_fit",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "density_path": str(density_path),
        "alpha_min": float(args.alpha_min),
        "alpha_max": float(args.alpha_max),
        "alpha_step": float(args.alpha_step),
        "fits": fit_rows,
    }
    _write_json(out_path, payload)
    log(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
