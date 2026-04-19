from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from ...multiscale_modal_suite.runner import (
    resolve_tasks,
    write_plan,
)


SUITE_DIR = Path(__file__).resolve().parents[1]
PAPER2_DIR = SUITE_DIR.parent
RESULTS_ROOT = PAPER2_DIR / "results" / "multiscale_suite"
CONFIG_ROOT = SUITE_DIR / "configs"


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def stage_packet(selected_lanes: list[str] | None = None) -> dict[str, Any]:
    models = {
        path.stem: load_yaml(path)
        for path in sorted((CONFIG_ROOT / "models").glob("*.yaml"))
    }
    splits = {
        path.stem: load_json(path)
        for path in sorted((CONFIG_ROOT / "splits").glob("*.json"))
    }
    claim_registry = load_yaml(CONFIG_ROOT / "claim_registry.yaml")
    plan_path = RESULTS_ROOT / "packets" / "suite_plan.json"
    tasks = resolve_tasks(lanes=selected_lanes)
    write_plan(plan_path, tasks)
    return {
        "status": "plan_only",
        "packet_results_root": str(RESULTS_ROOT),
        "execution_results_root": str(
            (PAPER2_DIR / "results" / "multiscale_modal_suite").resolve()
        ),
        "models": models,
        "splits": splits,
        "claim_registry": claim_registry,
        "plan_path": str(plan_path),
        "task_ids": [task.task_id for task in tasks],
    }


def parse_common_args(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--emit", type=str, default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    return ap
