from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

from rescue_research.modal_backend.resources import ModalResources


@dataclass
class ModalJobSpec:
    job_id: str
    stage_name: str
    pair_id: str
    model: str
    seed: int
    resources: ModalResources
    command: List[str]


def build_modal_job_specs(
    *,
    stage_name: str,
    pairs: List[str],
    models: List[str],
    seeds: List[int],
    command_prefix: List[str],
    resources: ModalResources | None = None,
    one_job_per_pair_model: bool = False,
) -> List[ModalJobSpec]:
    """
    Build job specs for Modal. By default: one job per (pair, model, seed).
    When one_job_per_pair_model=True (e.g. confirmatory): one job per (pair, model)
    with all seeds in the command so one comprehensive output per (pair, model)
    and no overwrite.
    """
    specs: List[ModalJobSpec] = []
    res = resources or ModalResources()
    ts = int(time.time())
    if one_job_per_pair_model and seeds:
        seeds_str = ",".join(str(s) for s in seeds)
        for pair in pairs:
            for model in models:
                job_id = f"{stage_name}-{pair}-{model}-{ts}"
                cmd = [
                    *command_prefix,
                    "--pair", pair,
                    "--model", model,
                    "--seeds", seeds_str,
                ]
                specs.append(
                    ModalJobSpec(
                        job_id=job_id,
                        stage_name=stage_name,
                        pair_id=pair,
                        model=model,
                        seed=seeds[0],
                        resources=res,
                        command=cmd,
                    )
                )
        return specs
    for pair in pairs:
        for model in models:
            for seed in seeds:
                job_id = f"{stage_name}-{pair}-{model}-{seed}-{ts}"
                cmd = [*command_prefix, "--pair", pair, "--model", model, "--seeds", str(seed)]
                specs.append(
                    ModalJobSpec(
                        job_id=job_id,
                        stage_name=stage_name,
                        pair_id=pair,
                        model=model,
                        seed=seed,
                        resources=res,
                        command=cmd,
                    )
                )
    return specs


def write_job_manifest(path: Path, specs: List[ModalJobSpec]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(s) for s in specs]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

