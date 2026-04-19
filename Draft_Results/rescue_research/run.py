#!/usr/bin/env python3
"""
Single entry point for the rescue research pipeline.

Usage:
  python -m rescue_research.run --stage baseline [--out-dir DIR]
  python -m rescue_research.run --stage layer_sweep_cv [--model 1b]
  python -m rescue_research.run --stage comprehensive --layer 21
  python -m rescue_research.run --stage mediation --layer 21
  python -m rescue_research.run --stage full

Stages run in order: baseline → layer_sweep_cv → comprehensive → mediation.
Each stage reads/writes under --out-dir so the next stage can use prior results.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure package is importable
_ROOT = Path(__file__).resolve().parent
if str(_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_ROOT.parent))

from rescue_research.config import RunConfig, STAGES_IN_ORDER, DEFAULT_OUT_DIR
from rescue_research.contracts import (
    LOCKED_LANGUAGE_PAIRS,
    validate_locked_pair_matrix,
    validate_preapproved_substitution_matrix,
)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rescue research pipeline: baseline → layer_sweep_cv → comprehensive → mediation",
    )
    p.add_argument(
        "--stage",
        choices=STAGES_IN_ORDER + ["full"],
        default="full",
        help="Stage to run, or 'full' for all in order",
    )
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory")
    p.add_argument(
        "--pair",
        type=str,
        default="",
        help="Pair ID for single-stage runs (e.g. hindi_telugu). Ignored when --pipeline is set.",
    )
    p.add_argument(
        "--prepared-split-dir",
        type=Path,
        default=None,
        help="Optional path to deterministic split directory (data/processed/<pair>) for single-stage runs.",
    )
    p.add_argument(
        "--use-blind-eval",
        action="store_true",
        help="Allow eval_blind usage when --prepared-split-dir is provided.",
    )
    p.add_argument("--model", type=str, default="1b", help="Model size (270m, 1b, 4b, 12b)")
    p.add_argument("--layer", type=int, default=21, help="Layer (used after layer_sweep_cv)")
    p.add_argument("--n-icl", type=int, default=5)
    p.add_argument("--n-select", type=int, default=50)
    p.add_argument("--n-eval", type=int, default=50)
    p.add_argument("--seeds", type=str, default="42,123,456,789,1337", help="Comma-separated seeds")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--log-file", type=Path, default=None, help="Optional log file")
    p.add_argument(
        "--pipeline",
        choices=[
            "",
            "full_confirmatory",
            "prepare_data",
            "baseline_selection",
            "confirmatory",
            "robustness",
            "report_bundle",
        ],
        default="",
        help="New end-to-end pipeline mode. If set, --stage is ignored.",
    )
    p.add_argument(
        "--backend",
        choices=["local", "modal"],
        default="local",
        help="Execution backend for --pipeline mode.",
    )
    p.add_argument(
        "--pairs",
        type=str,
        default=",".join(LOCKED_LANGUAGE_PAIRS),
        help="Comma-separated pair IDs for --pipeline mode.",
    )
    p.add_argument(
        "--models",
        type=str,
        default="1b,12b",
        help="Comma-separated model keys for --pipeline mode.",
    )
    p.add_argument(
        "--no-execute",
        action="store_true",
        help="In --pipeline mode, generate contracts/manifests without running heavy experiments.",
    )
    p.add_argument(
        "--run-blind-eval",
        action="store_true",
        help="Enable blind holdout execution flag in --pipeline mode.",
    )
    p.add_argument("--mediation-band-size", type=int, default=3, help="Top-K layers for mediation band.")
    p.add_argument("--task", choices=["transliteration", "translation"], default="transliteration", help="Task mode for orchestration and audits.")
    p.add_argument("--control-mode", type=str, default="default", help="Primary patched-metric mode for skeptic controls (executed, not label-only). Supports default/null_icl/random_icl/corrupt_icl/auto_scale_zs/auto_shift_zs/attention_only/basis_random/shuffle_random/gaussian_noise/mean_pool_patch/attn_head_ablation; comma-separated aliases allowed with optional primary=<mode>.")
    p.add_argument("--n-seeds", type=int, default=0, help="Use only the first N seeds from --seeds (0 = all).")
    p.add_argument("--compare-variants", action="store_true", help="Run affine vs skipless transcoder comparison.")
    p.add_argument("--run-quality-eval", action="store_true", help="Compute CER/exact match in baseline (--eval-generation).")
    p.add_argument("--eval-generation", action="store_true", help="Alias for --run-quality-eval.")
    p.add_argument("--patch-style", choices=["sparse", "substitute"], default="sparse", help="Patch style for interventions.")
    p.add_argument(
        "--split-policy",
        choices=["adaptive", "strict"],
        default="adaptive",
        help="How to derive split counts from available pool size in --pipeline mode.",
    )
    p.add_argument("--runtime-min-icl", type=int, default=4, help="Adaptive minimum n_icl for stage runs.")
    p.add_argument("--runtime-min-selection", type=int, default=8, help="Adaptive minimum n_select for stage runs.")
    p.add_argument("--runtime-min-eval", type=int, default=12, help="Adaptive minimum n_eval for stage runs.")
    p.add_argument("--data-min-icl-bank", type=int, default=8, help="Adaptive minimum ICL bank count for data stage.")
    p.add_argument("--data-min-selection", type=int, default=16, help="Adaptive minimum selection count for data stage.")
    p.add_argument("--data-min-eval-open", type=int, default=24, help="Adaptive minimum eval_open count for data stage.")
    p.add_argument("--data-min-eval-blind", type=int, default=8, help="Adaptive minimum eval_blind count for data stage.")
    p.add_argument(
        "--allow-underpowered-pairs",
        action="store_true",
        help="Allow confirmatory pipeline to proceed even when pair-readiness thresholds fail (exploratory only).",
    )
    p.add_argument(
        "--disable-pair-readiness-check",
        action="store_true",
        help="Disable readiness guardrails entirely (not recommended for confirmatory claims).",
    )
    p.add_argument("--min-confirmatory-pool", type=int, default=40, help="Minimum available records required per confirmatory pair.")
    p.add_argument("--min-confirmatory-icl", type=int, default=4, help="Minimum runtime n_icl required per confirmatory pair.")
    p.add_argument("--min-confirmatory-selection", type=int, default=12, help="Minimum runtime n_selection required per confirmatory pair.")
    p.add_argument("--min-confirmatory-eval", type=int, default=24, help="Minimum runtime n_eval required per confirmatory pair.")
    p.add_argument(
        "--allow-custom-pairs",
        action="store_true",
        help="Allow non-locked pair matrix for --pipeline full_confirmatory (exploratory only).",
    )
    return p.parse_args()


def _to_config(args: argparse.Namespace) -> RunConfig:
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if int(getattr(args, "n_seeds", 0)) > 0:
        seeds = seeds[: int(args.n_seeds)]
    return RunConfig(
        out_dir=args.out_dir,
        log_file=args.log_file,
        n_icl=args.n_icl,
        n_select=args.n_select,
        n_eval=args.n_eval,
        seeds=seeds,
        model=args.model,
        layer=args.layer,
        device=args.device,
        pair=(args.pair or "").strip(),
        prepared_split_dir=str(args.prepared_split_dir) if args.prepared_split_dir else "",
        use_blind_eval=bool(getattr(args, "use_blind_eval", False)),
        patch_style=getattr(args, "patch_style", "sparse"),
        eval_generation=getattr(args, "run_quality_eval", False) or getattr(args, "eval_generation", False),
        task=getattr(args, "task", "transliteration"),
        control_mode=getattr(args, "control_mode", "default"),
    )


def main() -> int:
    args = _parse()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if int(getattr(args, "n_seeds", 0)) > 0:
        seeds = seeds[: int(args.n_seeds)]

    if args.pipeline:
        from rescue_research.pipeline.dag import run_pipeline
        from rescue_research.pipeline_config import PipelineConfig

        pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        substitution_plan = []
        if args.pipeline == "full_confirmatory":
            locked_len = len(LOCKED_LANGUAGE_PAIRS)
            if not bool(args.allow_custom_pairs):
                validate_locked_pair_matrix(pairs)
            else:
                if len(set(pairs)) != len(pairs):
                    raise ValueError("Custom pair list contains duplicate pair IDs.")
                if len(pairs) == locked_len:
                    if tuple(pairs) != tuple(LOCKED_LANGUAGE_PAIRS):
                        substitution_plan = validate_preapproved_substitution_matrix(pairs)
                        print(
                            "[rescue_research] WARNING: Running full_confirmatory with pre-approved pair substitution "
                            "(exploratory unless substitution policy is satisfied).",
                            flush=True,
                        )
                        if substitution_plan:
                            print(
                                "[rescue_research] Substitution plan: " + str(substitution_plan),
                                flush=True,
                            )
                else:
                    print(
                        "[rescue_research] WARNING: Running full_confirmatory with pair subset/custom matrix "
                        f"(n_pairs={len(pairs)} != locked_n={locked_len}). "
                        "Treat this run as exploratory only.",
                        flush=True,
                    )
        pipe_cfg = PipelineConfig(
            out_dir=args.out_dir,
            pairs=pairs,
            models=models,
            seeds=seeds,
            backend=args.backend,
            execute_experiments=not bool(args.no_execute),
            run_blind_eval=bool(args.run_blind_eval),
            mediation_band_size=getattr(args, "mediation_band_size", 3),
            compare_variants=args.compare_variants,
            run_quality_eval=args.run_quality_eval or args.eval_generation,
            eval_generation=args.run_quality_eval or args.eval_generation,
            patch_style=args.patch_style,
            task=args.task,
            control_mode=args.control_mode,
            n_seeds=int(args.n_seeds),
            split_policy=args.split_policy,
            runtime_min_icl=args.runtime_min_icl,
            runtime_min_selection=args.runtime_min_selection,
            runtime_min_eval=args.runtime_min_eval,
            data_min_icl_bank=args.data_min_icl_bank,
            data_min_selection=args.data_min_selection,
            data_min_eval_open=args.data_min_eval_open,
            data_min_eval_blind=args.data_min_eval_blind,
            enforce_pair_readiness=not bool(args.disable_pair_readiness_check),
            allow_underpowered_pairs=bool(args.allow_underpowered_pairs),
            min_confirmatory_pool=args.min_confirmatory_pool,
            min_confirmatory_icl=args.min_confirmatory_icl,
            min_confirmatory_selection=args.min_confirmatory_selection,
            min_confirmatory_eval=args.min_confirmatory_eval,
            allow_custom_pairs=bool(args.allow_custom_pairs),
            substitution_plan=list(substitution_plan),
        )
        run_pipeline(pipe_cfg, stage=args.pipeline)
        return 0

    config = _to_config(args)
    config.seeds = seeds
    config.ensure_out_dir()

    stages_to_run = STAGES_IN_ORDER if args.stage == "full" else [args.stage]

    for stage in stages_to_run:
        if stage == "baseline":
            from rescue_research.stages.baseline import run_baseline
            run_baseline(config)
        elif stage == "layer_sweep_cv":
            from rescue_research.stages.layer_sweep_cv import run_layer_sweep_cv
            run_layer_sweep_cv(config)
        elif stage == "comprehensive":
            from rescue_research.stages.comprehensive import run_comprehensive
            run_comprehensive(config, run_quality_eval=(args.run_quality_eval or args.eval_generation))
        elif stage == "mediation":
            from rescue_research.stages.mediation_run import run_mediation
            run_mediation(config)
        else:
            print(f"Unknown stage: {stage}", file=sys.stderr)
            return 1

    # After full run, compute primary outcome if we have comprehensive results
    if args.stage == "full" or "comprehensive" in stages_to_run:
        from rescue_research.analysis.primary_outcome import compute_and_save_primary_outcome
        compute_and_save_primary_outcome(config)

    return 0


if __name__ == "__main__":
    sys.exit(main())
