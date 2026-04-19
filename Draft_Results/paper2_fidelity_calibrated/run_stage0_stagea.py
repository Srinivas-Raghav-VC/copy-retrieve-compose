#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List


DEFAULT_STAGE0_PAIRS = ["aksharantar_hin_latin", "aksharantar_tel_latin"]
DEFAULT_MODELS = ["1b", "4b"]


def _parse_csv(raw: str | None, *, default: List[str] | None = None) -> List[str]:
    vals = [x.strip() for x in str(raw or "").split(",") if x.strip()]
    if vals:
        return vals
    if default:
        return list(default)
    raise ValueError(f"Expected non-empty CSV string, got {raw!r}")


def _run(cmd: List[str], *, dry_run: bool) -> None:
    print("[stage0-stagea] " + " ".join(cmd), flush=True)
    if dry_run:
        return
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(rc)


def _build_data_command(
    root: Path,
    *,
    pairs: List[str],
    splits: str,
    max_rows_per_pair: int,
) -> List[str]:
    return [
        sys.executable,
        str(root / "scripts" / "build_workshop_external_data.py"),
        "--pairs",
        ",".join(pairs),
        "--splits",
        str(splits),
        "--out-dir",
        str(root / "data" / "transliteration"),
        "--max-rows-per-pair",
        str(int(max_rows_per_pair)),
    ]


def _stage0_commands(
    root: Path,
    *,
    models: List[str],
    pairs: List[str],
    device: str,
    min_pool_size: int,
) -> List[List[str]]:
    out: List[List[str]] = []
    for pair in pairs:
        for model in models:
            if model not in {"1b", "4b"}:
                continue
            out.append(
                [
                    sys.executable,
                    str(root / "paper2_fidelity_calibrated" / "stage0_packet.py"),
                    "--model",
                    model,
                    "--pair",
                    pair,
                    "--device",
                    device,
                    "--external-only",
                    "--require-external-sources",
                    "--min-pool-size",
                    str(int(min_pool_size)),
                ]
            )
            out.append(
                [
                    sys.executable,
                    str(root / "paper2_fidelity_calibrated" / "run_premise_gate.py"),
                    "--model",
                    model,
                    "--pair",
                    pair,
                    "--device",
                    device,
                    "--external-only",
                    "--require-external-sources",
                    "--min-pool-size",
                    str(int(min_pool_size)),
                ]
            )
    return out


def _stagea_commands(
    root: Path,
    *,
    models: List[str],
    pairs: List[str],
    seeds: str,
    device: str,
    min_pool_size: int,
    layer_step: int,
    topk_options: str,
    extra_args: str,
) -> List[List[str]]:
    out: List[List[str]] = []
    extra = [x for x in str(extra_args).split() if x.strip()]
    for pair in pairs:
        for model in models:
            if model not in {"1b", "4b"}:
                continue
            cmd = [
                sys.executable,
                str(root / "paper2_fidelity_calibrated" / "run.py"),
                "--model",
                model,
                "--pair",
                pair,
                "--seeds",
                seeds,
                "--device",
                device,
                "--variants",
                "skipless_or_non_affine",
                "--layer-step",
                str(int(layer_step)),
                "--topk-options",
                topk_options,
                "--feature-selection",
                "topk_abs_delta",
                "--selector-reference",
                "corrupt_icl",
                "--prompt-variant",
                "canonical",
                "--require-query-span-match",
                "--norm-matching",
                "--external-only",
                "--require-external-sources",
                "--min-pool-size",
                str(int(min_pool_size)),
            ]
            if extra:
                cmd.extend(extra)
            out.append(cmd)
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the frozen Stage 0 / Stage A workshop protocol.")
    ap.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS))
    ap.add_argument("--stage0-pairs", type=str, default=",".join(DEFAULT_STAGE0_PAIRS))
    ap.add_argument("--stagea-pairs", type=str, default=",".join(DEFAULT_STAGE0_PAIRS))
    ap.add_argument("--seeds", type=str, default="42")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--min-pool-size", type=int, default=500)
    ap.add_argument("--layer-step", type=int, default=4)
    ap.add_argument("--topk-options", type=str, default="4,8,16,32")
    ap.add_argument("--extra-stagea-args", type=str, default="")
    ap.add_argument("--build-data", action="store_true")
    ap.add_argument("--build-splits", type=str, default="train,valid,test")
    ap.add_argument("--build-max-rows-per-pair", type=int, default=0)
    ap.add_argument("--stage0-only", action="store_true")
    ap.add_argument("--skip-stage0", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    models = _parse_csv(args.models, default=DEFAULT_MODELS)
    stage0_pairs = _parse_csv(args.stage0_pairs, default=DEFAULT_STAGE0_PAIRS)
    stagea_pairs = _parse_csv(args.stagea_pairs, default=DEFAULT_STAGE0_PAIRS)

    manifest = {
        "profile": "stage0_stagea_workshop",
        "models": models,
        "stage0_pairs": stage0_pairs,
        "stagea_pairs": stagea_pairs,
        "device": str(args.device),
        "seeds": _parse_csv(args.seeds, default=["42"]),
        "min_pool_size": int(args.min_pool_size),
        "layer_step": int(args.layer_step),
        "topk_options": _parse_csv(args.topk_options, default=["4", "8", "16", "32"]),
        "claim_level": "intervention_only",
        "prompt_template_frozen": True,
        "stage0_premise_gate_enabled": True,
        "build_data": bool(args.build_data),
        "stagea_local_stability_reporting_only": True,
        "dry_run": bool(args.dry_run),
    }
    manifest_path = root / "paper2_fidelity_calibrated" / "results" / "stage0_stagea_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[stage0-stagea] manifest: {manifest_path}", flush=True)

    if bool(args.build_data):
        build_pairs = sorted(set(stage0_pairs + stagea_pairs))
        _run(
            _build_data_command(
                root,
                pairs=build_pairs,
                splits=str(args.build_splits),
                max_rows_per_pair=int(args.build_max_rows_per_pair),
            ),
            dry_run=bool(args.dry_run),
        )

    if not args.skip_stage0:
        for cmd in _stage0_commands(
            root,
            models=models,
            pairs=stage0_pairs,
            device=str(args.device),
            min_pool_size=int(args.min_pool_size),
        ):
            _run(cmd, dry_run=bool(args.dry_run))

    if args.stage0_only:
        return 0

    for cmd in _stagea_commands(
        root,
        models=models,
        pairs=stagea_pairs,
        seeds=str(args.seeds),
        device=str(args.device),
        min_pool_size=int(args.min_pool_size),
        layer_step=int(args.layer_step),
        topk_options=str(args.topk_options),
        extra_args=str(args.extra_stagea_args),
    ):
        _run(cmd, dry_run=bool(args.dry_run))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
