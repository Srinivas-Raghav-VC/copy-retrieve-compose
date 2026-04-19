#!/usr/bin/env python3
"""
Run Paper 2 fidelity-calibrated experiment across a model/pair matrix.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def _parse_csv(raw: str) -> List[str]:
    vals = [x.strip() for x in str(raw).split(",") if x.strip()]
    if not vals:
        raise ValueError(f"Expected non-empty CSV string, got {raw!r}")
    return vals


def main() -> int:
    ap = argparse.ArgumentParser(description="Matrix runner for paper2 fidelity-calibrated interventions.")
    ap.add_argument(
        "--models",
        type=str,
        default="1b,4b",
        help="Comma-separated model keys. Defaults to the workshop-budget profile.",
    )
    ap.add_argument(
        "--pairs",
        type=str,
        default="aksharantar_hin_latin,aksharantar_tam_latin,aksharantar_tel_latin",
        help=(
            "Comma-separated pair ids. Override this for publication runs with "
            "externally sourced, powered pairs; the default matches the current "
            "registered workshop-oriented Aksharantar set."
        ),
    )
    ap.add_argument("--seeds", type=str, default="42,123,456")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--external-only", action="store_true")
    ap.add_argument("--require-external-sources", action="store_true")
    ap.add_argument("--min-pool-size", type=int, default=0)
    ap.add_argument("--continue-on-error", action="store_true")
    ap.add_argument("--extra-args", type=str, default="", help="Extra args appended verbatim.")
    args = ap.parse_args()

    this_dir = Path(__file__).resolve().parent
    runner = this_dir / "run.py"
    models = _parse_csv(args.models)
    pairs = _parse_csv(args.pairs)

    failures = []
    for pair in pairs:
        for model in models:
            cmd = [
                sys.executable,
                str(runner),
                "--model",
                model,
                "--pair",
                pair,
                "--seeds",
                args.seeds,
                "--device",
                args.device,
            ]
            if args.external_only:
                cmd.append("--external-only")
            if args.require_external_sources:
                cmd.append("--require-external-sources")
            if int(args.min_pool_size) > 0:
                cmd.extend(["--min-pool-size", str(int(args.min_pool_size))])
            if args.extra_args.strip():
                cmd.extend(args.extra_args.strip().split())
            print(f"\n[run-matrix] model={model} pair={pair}")
            print("[run-matrix] " + " ".join(cmd), flush=True)
            rc = subprocess.call(cmd)
            if rc != 0:
                failures.append((model, pair, rc))
                if not args.continue_on_error:
                    print(f"[run-matrix] failed: model={model} pair={pair} rc={rc}", flush=True)
                    return rc

    if failures:
        print("\n[run-matrix] completed with failures:", flush=True)
        for model, pair, rc in failures:
            print(f"  - model={model} pair={pair} rc={rc}", flush=True)
        return 1

    print("\n[run-matrix] all runs completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
