from __future__ import annotations

import json
import time
from typing import Callable, Dict, List

from rescue_research.pipeline.stages import (
    stage_baseline_selection,
    stage_confirmatory,
    stage_prepare_data,
    stage_report_bundle,
    stage_robustness,
)
from rescue_research.pipeline.validator import validate_artifacts
from rescue_research.pipeline_config import PipelineConfig


PIPELINE_ORDER = [
    "prepare_data",
    "baseline_selection",
    "confirmatory",
    "robustness",
    "report_bundle",
]


def _stage_map() -> Dict[str, Callable[[PipelineConfig], None]]:
    return {
        "prepare_data": stage_prepare_data,
        "baseline_selection": stage_baseline_selection,
        "confirmatory": stage_confirmatory,
        "robustness": stage_robustness,
        "report_bundle": stage_report_bundle,
    }


def run_pipeline(config: PipelineConfig, stage: str = "full_confirmatory") -> None:
    config.ensure_out_dir()
    stages = _stage_map()
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    stage_timings: List[Dict[str, float | str]] = []

    def _run_stage(name: str) -> None:
        t0 = time.time()
        stages[name](config)
        stage_timings.append(
            {
                "stage": name,
                "duration_sec": round(time.time() - t0, 3),
            }
        )

    if stage == "full_confirmatory":
        for name in PIPELINE_ORDER:
            _run_stage(name)
    else:
        if stage not in stages:
            raise ValueError(f"Unknown pipeline stage: {stage}")
        _run_stage(stage)

    # Validate at pipeline boundary. Only raise on failure after full pipeline;
    # single-stage runs may legitimately lack confirmatory/robustness artifacts.
    report = validate_artifacts(
        out_dir=config.out_dir,
        pairs=config.pairs,
        models=config.models,
        seeds=config.seeds,
    )
    if not report.ok:
        missing = "\n".join(report.missing_paths[:20])
        if stage == "full_confirmatory":
            raise RuntimeError(
                "Artifact contract validation failed. Missing paths (first 20):\n"
                + missing
            )
        for w in report.warnings:
            print(f"[rescue_research] Validation warning: {w}", flush=True)
        print(
            f"[rescue_research] Validation: {len(report.missing_paths)} missing path(s) "
            "(not raised for single-stage run). Run full_confirmatory to enforce contract.",
            flush=True,
        )

    timing_payload = {
        "started_at": started_at,
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "invocation_stage": stage,
        "backend": str(getattr(config, "backend", "local")),
        "execute_experiments": bool(getattr(config, "execute_experiments", True)),
        "stage_timings": stage_timings,
        "total_duration_sec": round(
            sum(float(row.get("duration_sec", 0.0)) for row in stage_timings), 3
        ),
    }
    timing_path = config.out_dir / "artifacts" / "manifests" / "pipeline_timing.json"
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    timing_path.write_text(json.dumps(timing_payload, indent=2), encoding="utf-8")

