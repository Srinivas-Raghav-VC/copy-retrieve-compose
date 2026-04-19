"""
Stage 3: Comprehensive validation at best layer (from layer_sweep_cv).

Runs the reference exp3_comprehensive with three-way split (n_select > 0)
so feature/layer selection and evaluation are disjoint. Reads best_layer
from config.out_dir / best_layer.txt (written by layer_sweep_cv).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from rescue_research.config import RunConfig


def run_comprehensive(config: RunConfig, run_quality_eval: bool = False) -> None:
    ref_root = Path(__file__).resolve().parent.parent.parent
    config.ensure_out_dir()

    best_path = config.out_dir / "best_layer.txt"
    if best_path.exists():
        layer = int(best_path.read_text(encoding="utf-8").strip())
    else:
        layer = config.layer
        print(f"[rescue_research] No best_layer.txt; using config layer {layer}", flush=True)

    out_path = config.out_dir / f"comprehensive_{config.model}_L{layer}.json"

    cmd = [
        sys.executable,
        str(ref_root / "experiments" / "exp3_comprehensive.py"),
        "--model", config.model,
        "--layer", str(layer),
        "--n-icl", str(config.n_icl),
        "--n-select", str(config.n_select),
        "--n-test", str(config.n_eval),
        "--seeds", ",".join(map(str, config.seeds)),
        "--topk-values", ",".join(map(str, config.topk_values)),
        "--output", str(out_path),
        "--device", config.device,
        "--patch-style", config.patch_style,
        "--control-mode", str(getattr(config, "control_mode", "default")),
    ]
    if getattr(config, "sweep_positions", False):
        cmd.append("--sweep-positions")
    if getattr(config, "decoupled_control", False):
        cmd.append("--decoupled-control")
    if run_quality_eval or getattr(config, "eval_generation", False):
        cmd.append("--eval-generation")

    prepared_split_dir = str(getattr(config, "prepared_split_dir", "") or "").strip()
    if prepared_split_dir:
        split_dir_path = Path(prepared_split_dir)
        if not split_dir_path.exists():
            raise FileNotFoundError(
                f"Prepared split directory not found for comprehensive stage: {split_dir_path}"
            )
        cmd.extend(["--prepared-split-dir", str(split_dir_path)])
        if bool(getattr(config, "use_blind_eval", False)):
            cmd.append("--use-blind-eval")
    if config.pair:
        cmd.extend(["--pair", config.pair])

    print(f"[rescue_research] Running comprehensive at layer {layer}: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=str(ref_root))
    if result.returncode != 0:
        raise RuntimeError(f"Comprehensive stage failed with exit code {result.returncode}")
    print(f"[rescue_research] Comprehensive results: {out_path}", flush=True)
