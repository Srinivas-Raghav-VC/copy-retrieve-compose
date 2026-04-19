"""
Stage 1: Baseline — verify ICL lift before any patching.

Runs the reference exp1_baseline with our config; writes to config.out_dir.
Uses two-way split (ICL + eval) only; no selection.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from rescue_research.config import RunConfig


def run_baseline(
    config: RunConfig,
    run_quality_eval: bool = False,
    eval_multi_token: bool = True,
) -> None:
    ref_root = Path(__file__).resolve().parent.parent.parent
    out_path = config.ensure_out_dir() / f"baseline_{config.model}.json"

    cmd = [
        sys.executable,
        str(ref_root / "experiments" / "exp1_baseline.py"),
        "--model", config.model,
        "--n-icl", str(config.n_icl),
        "--n-test", str(config.n_eval),
        "--seeds", ",".join(map(str, config.seeds)),
        "--output", str(out_path),
        "--device", config.device,
    ]
    prepared_split_dir = str(getattr(config, "prepared_split_dir", "") or "").strip()
    if prepared_split_dir:
        split_dir_path = Path(prepared_split_dir)
        if not split_dir_path.exists():
            raise FileNotFoundError(
                f"Prepared split directory not found for baseline stage: {split_dir_path}"
            )
        cmd.extend(["--prepared-split-dir", str(split_dir_path)])
        if bool(getattr(config, "use_blind_eval", False)):
            cmd.append("--use-blind-eval")
    if eval_multi_token:
        cmd.append("--eval-multi-token")
    if run_quality_eval:
        cmd.append("--eval-generation")
    if config.pair:
        cmd.extend(["--pair", config.pair])

    print(f"[rescue_research] Running baseline: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=str(ref_root))
    if result.returncode != 0:
        raise RuntimeError(f"Baseline stage failed with exit code {result.returncode}")
    print(f"[rescue_research] Baseline results: {out_path}", flush=True)
