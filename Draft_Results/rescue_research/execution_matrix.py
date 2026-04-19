from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class MatrixCell:
    cell_id: str
    pair_id: str
    model: str
    out_dir: str
    command: List[str]
    stage: str = "full_confirmatory"
    patch_style: str = "sparse"


def build_pipeline_matrix(
    *,
    python_executable: str,
    base_out_dir: Path,
    pairs: Iterable[str],
    models: Iterable[str],
    split_policy: str = "adaptive",
    backend: str = "local",
    no_execute: bool = False,
    run_quality_eval: bool = False,
    compare_variants: bool = False,
    run_blind_eval: bool = False,
    allow_underpowered_pairs: bool = False,
    disable_pair_readiness_check: bool = False,
    min_confirmatory_pool: int = 40,
    min_confirmatory_icl: int = 4,
    min_confirmatory_selection: int = 12,
    min_confirmatory_eval: int = 24,
    allow_custom_pairs: bool = True,
    patch_styles: Iterable[str] = ("sparse", "substitute"),
    seeds: Iterable[int] | None = None,
    task: str = "transliteration",
    control_mode: str = "default",
) -> List[MatrixCell]:
    cells: List[MatrixCell] = []
    for pair in pairs:
        pair_id = str(pair).strip()
        if not pair_id:
            continue

        for model in models:
            model_id = str(model).strip()
            if not model_id:
                continue

            for patch_style in patch_styles:
                style = str(patch_style).strip().lower()
                if style not in {"sparse", "substitute"}:
                    continue

                cell_id = f"{pair_id}__{model_id}__{style}"
                out_dir = base_out_dir / cell_id
                cmd = [
                    python_executable,
                    "-m",
                    "rescue_research.run",
                    "--pipeline",
                    "full_confirmatory",
                    "--backend",
                    backend,
                    "--split-policy",
                    split_policy,
                    "--pairs",
                    pair_id,
                    "--models",
                    model_id,
                    "--task",
                    str(task),
                    "--control-mode",
                    str(control_mode),
                    "--patch-style",
                    style,
                    "--out-dir",
                    str(out_dir),
                ]
                if seeds is not None:
                    seed_list = [str(int(s)) for s in seeds]
                    if seed_list:
                        cmd.extend(["--seeds", ",".join(seed_list)])
                if no_execute:
                    cmd.append("--no-execute")
                if run_quality_eval:
                    cmd.append("--run-quality-eval")
                if compare_variants:
                    cmd.append("--compare-variants")
                if run_blind_eval:
                    cmd.append("--run-blind-eval")
                if allow_underpowered_pairs:
                    cmd.append("--allow-underpowered-pairs")
                if disable_pair_readiness_check:
                    cmd.append("--disable-pair-readiness-check")
                if int(min_confirmatory_pool) != 40:
                    cmd.extend(["--min-confirmatory-pool", str(int(min_confirmatory_pool))])
                if int(min_confirmatory_icl) != 4:
                    cmd.extend(["--min-confirmatory-icl", str(int(min_confirmatory_icl))])
                if int(min_confirmatory_selection) != 12:
                    cmd.extend(["--min-confirmatory-selection", str(int(min_confirmatory_selection))])
                if int(min_confirmatory_eval) != 24:
                    cmd.extend(["--min-confirmatory-eval", str(int(min_confirmatory_eval))])
                if allow_custom_pairs:
                    cmd.append("--allow-custom-pairs")

                cells.append(
                    MatrixCell(
                        cell_id=cell_id,
                        pair_id=pair_id,
                        model=model_id,
                        out_dir=str(out_dir),
                        command=cmd,
                        patch_style=style,
                    )
                )
    return cells


def write_matrix_manifest(path: Path, cells: List[MatrixCell]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_cells": len(cells),
        "cells": [asdict(c) for c in cells],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
