from __future__ import annotations

import json
from pathlib import Path

import modal

from ...multiscale_modal_suite.runner import (
    execute_task,
    resolve_tasks,
)
from ...multiscale_modal_suite.suite_spec import (
    SuiteTask,
)
from ..runners.common import (
    RESULTS_ROOT,
    stage_packet,
    write_json,
)


APP_NAME = "gemma-multiscale-suite"
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
REMOTE_WORKSPACE = "/workspace"
REMOTE_RESULTS = "/artifacts/multiscale_suite"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "modal>=0.72.0",
        "pyyaml",
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "sentencepiece",
        "pandas",
        "matplotlib",
        "marimo",
        "seaborn",
        "huggingface_hub",
    )
    .add_local_dir(str(WORKSPACE_ROOT), remote_path=REMOTE_WORKSPACE)
)

results_volume = modal.Volume.from_name(
    "gemma-multiscale-results", create_if_missing=True
)
hf_cache = modal.Volume.from_name("gemma-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=24 * 60 * 60,
    cpu=4,
    memory=32768,
    retries=modal.Retries(initial_delay=0.0, max_retries=4, backoff_coefficient=1.0),
    volumes={
        REMOTE_RESULTS: results_volume,
        "/cache/huggingface": hf_cache,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("google-generative-ai"),
    ],
)
def run_task(task_payload: dict, smoke: bool = False, force: bool = False) -> dict:
    import os
    import sys

    os.environ.setdefault("HF_HOME", "/cache/huggingface")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/cache/huggingface/transformers")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/cache/huggingface/hub")

    workspace_root = Path(REMOTE_WORKSPACE)
    sys.path.insert(0, str(workspace_root))
    sys.path.insert(0, str(workspace_root / "Draft_Results"))
    sys.path.insert(
        0, str(workspace_root / "Draft_Results" / "paper2_fidelity_calibrated")
    )

    task = SuiteTask(**task_payload)
    manifest = execute_task(
        task, smoke=smoke, force=force, workspace_root=workspace_root
    )
    return manifest.to_dict()


@app.local_entrypoint()
def main(
    task_ids: str = "", lanes: str = "", smoke: bool = False, force: bool = False
) -> None:
    lane_list = [x.strip() for x in str(lanes).split(",") if x.strip()]
    task_list = [x.strip() for x in str(task_ids).split(",") if x.strip()]

    packet = stage_packet(lane_list or None)
    write_json(RESULTS_ROOT / "packets" / "stage_packet.json", packet)
    tasks = resolve_tasks(task_ids=task_list or None, lanes=lane_list or None)

    print(
        json.dumps(
            {
                "task_count": len(tasks),
                "task_ids": [task.task_id for task in tasks],
                "stage_packet": str(RESULTS_ROOT / "packets" / "stage_packet.json"),
            },
            indent=2,
        )
    )
    for payload in [task.to_dict() for task in tasks]:
        run_task.spawn(payload, smoke=smoke, force=force)
