"""
Stage 2: Layer sweep with held-out evaluation (no selection bias).

Runs the reference exp2_layer_sweep_cv with three-way split; ranks layers
on selection split only, optionally evaluates best layer on eval split.
Writes best_layer to config.out_dir for use by comprehensive stage.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from rescue_research.config import RunConfig


def run_layer_sweep_cv(config: RunConfig, top_layers: int = 5) -> None:
    """
    Run layer sweep with CV. top_layers: how many top layers to keep in summary
    (used for mediation band; should be >= mediation_band_size).
    """
    ref_root = Path(__file__).resolve().parent.parent.parent
    config.ensure_out_dir()
    out_path = config.out_dir / f"layer_sweep_cv_{config.model}.json"

    cmd = [
        sys.executable,
        str(ref_root / "experiments" / "exp2_layer_sweep_cv.py"),
        "--model", config.model,
        "--n-icl", str(config.n_icl),
        "--n-select", str(config.n_select),
        "--n-eval", str(config.n_eval),
        "--seeds", ",".join(map(str, config.seeds)),
        "--topk", "25",
        "--rank-metric", "pe_minus_corrupt",
        "--top-layers", str(max(1, int(top_layers))),
        "--eval-best",  # Evaluate best layer on held-out split
        "--resume",
        "--device", config.device,
        "--output", str(out_path),
    ]
    prepared_split_dir = str(getattr(config, "prepared_split_dir", "") or "").strip()
    if prepared_split_dir:
        split_dir_path = Path(prepared_split_dir)
        if not split_dir_path.exists():
            raise FileNotFoundError(
                f"Prepared split directory not found for layer_sweep_cv stage: {split_dir_path}"
            )
        cmd.extend(["--prepared-split-dir", str(split_dir_path)])
        if bool(getattr(config, "use_blind_eval", False)):
            cmd.append("--use-blind-eval")
    if config.pair:
        cmd.extend(["--pair", config.pair])

    print(f"[rescue_research] Running layer_sweep_cv: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=str(ref_root))
    if result.returncode != 0:
        raise RuntimeError(f"Layer sweep CV stage failed with exit code {result.returncode}")

    # Write best_layer for next stage
    with open(out_path, encoding="utf-8") as f:
        data = json.load(f)
    best = data.get("summary", {}).get("best_layer")
    if best is None:
        raise RuntimeError("Layer sweep CV did not produce best_layer")
    best_path = config.out_dir / "best_layer.txt"
    best_path.write_text(str(best), encoding="utf-8")
    print(f"[rescue_research] Best layer: {best} -> {best_path}", flush=True)
    print(f"[rescue_research] Layer sweep CV results: {out_path}", flush=True)
