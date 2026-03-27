from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal

APP_NAME = "gemma-multiscale-modal-suite"
REMOTE_WORKSPACE = "/workspace"
REMOTE_RESULTS = "/artifacts/multiscale_modal_suite"


def _bootstrap_import_paths() -> None:
    candidates: list[Path] = []

    remote_root = Path(os.environ.get("MULTISCALE_WORKSPACE_ROOT", REMOTE_WORKSPACE))
    candidates.extend(
        [
            remote_root,
            remote_root / "Draft_Results",
            remote_root / "Draft_Results" / "paper2_fidelity_calibrated",
        ]
    )

    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "Draft_Results").exists() and (candidate / "research").exists():
            candidates.extend(
                [
                    candidate,
                    candidate / "Draft_Results",
                    candidate / "Draft_Results" / "paper2_fidelity_calibrated",
                ]
            )
            break

    for candidate in candidates:
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


_bootstrap_import_paths()

try:
    from .runner import execute_task, resolve_tasks, write_plan
    from .suite_spec import SuiteTask
    from .verify_suite import run_verification
except ImportError:  # pragma: no cover - support `modal run path/to/modal_app.py`
    from Draft_Results.paper2_fidelity_calibrated.multiscale_modal_suite.runner import (  # type: ignore
        execute_task,
        resolve_tasks,
        write_plan,
    )
    from Draft_Results.paper2_fidelity_calibrated.multiscale_modal_suite.suite_spec import (  # type: ignore
        SuiteTask,
    )
    from Draft_Results.paper2_fidelity_calibrated.multiscale_modal_suite.verify_suite import (  # type: ignore
        run_verification,
    )


def _resolve_workspace_root() -> Path:
    env_root = Path(os.environ.get("MULTISCALE_WORKSPACE_ROOT", REMOTE_WORKSPACE))
    if (env_root / "Draft_Results").exists() and (env_root / "research").exists():
        return env_root

    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "Draft_Results").exists() and (candidate / "research").exists():
            return candidate

    return env_root


WORKSPACE_ROOT = _resolve_workspace_root()

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "modal>=0.72.0",
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
    ],
)
def run_suite_task(
    task_payload: dict, smoke: bool = False, force: bool = False
) -> dict:
    import os
    import sys

    os.environ.setdefault("HF_HOME", "/cache/huggingface")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/cache/huggingface/transformers")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/cache/huggingface/hub")
    os.environ.setdefault("MULTISCALE_RESULTS_ROOT", REMOTE_RESULTS)

    workspace_root = Path(REMOTE_WORKSPACE)
    sys.path.insert(
        0, str(workspace_root / "Draft_Results" / "paper2_fidelity_calibrated")
    )
    sys.path.insert(0, str(workspace_root / "Draft_Results"))
    sys.path.insert(0, str(workspace_root))

    task = SuiteTask(**task_payload)
    manifest = execute_task(
        task, smoke=smoke, force=force, workspace_root=workspace_root
    )
    return manifest.to_dict()


@app.local_entrypoint()
def main(
    task_ids: str = "",
    lanes: str = "",
    smoke: bool = False,
    force: bool = False,
    wait: bool = False,
    emit_plan: str = "",
) -> None:
    os.environ.setdefault("MULTISCALE_RESULTS_ROOT", REMOTE_RESULTS)
    preflight = run_verification()
    if not preflight.get("preflight_ok", False):
        raise RuntimeError(
            "Refusing Modal launch because required research modules are missing: "
            + ", ".join(preflight.get("missing_research_modules", []))
        )

    lane_list = [x.strip() for x in str(lanes).split(",") if x.strip()]
    task_list = [x.strip() for x in str(task_ids).split(",") if x.strip()]
    tasks = resolve_tasks(task_ids=task_list or None, lanes=lane_list or None)

    if str(emit_plan).strip():
        write_plan(Path(str(emit_plan)).resolve(), tasks)

    print(
        json.dumps(
            {
                "task_count": len(tasks),
                "task_ids": [task.task_id for task in tasks],
                "smoke": bool(smoke),
                "wait": bool(wait),
                "results_root": os.environ.get("MULTISCALE_RESULTS_ROOT", str(REMOTE_RESULTS)),
            },
            indent=2,
        )
    )

    manifests: list[dict] = []
    for payload in [task.to_dict() for task in tasks]:
        if wait:
            manifests.append(run_suite_task.remote(payload, smoke=smoke, force=force))
        else:
            run_suite_task.spawn(payload, smoke=smoke, force=force)

    if wait:
        print(json.dumps({"manifests": manifests}, indent=2))
