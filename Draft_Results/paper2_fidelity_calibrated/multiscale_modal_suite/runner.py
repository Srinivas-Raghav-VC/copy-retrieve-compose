from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path
from typing import Iterable

from .legacy_commands import build_command, default_results_root
from .result_schema import TaskManifest
from .suite_spec import SuiteTask, build_suite_plan, resolve_results_root


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_json(value):
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def resolve_tasks(
    task_ids: Iterable[str] | None = None, lanes: Iterable[str] | None = None
) -> list[SuiteTask]:
    selected = [
        task
        for task in build_suite_plan(lanes)
        if not task_ids or task.task_id in set(task_ids)
    ]
    if task_ids:
        missing = sorted(set(task_ids) - {task.task_id for task in selected})
        if missing:
            raise KeyError(f"Unknown task ids: {', '.join(missing)}")
    return selected


def write_plan(path: Path, tasks: list[SuiteTask]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "plan_only",
        "created_at_utc": _now_utc(),
        "execution_results_root": str(resolve_results_root()),
        "packet_location": str(path),
        "tasks": [task.to_dict() for task in tasks],
    }
    path.write_text(json.dumps(_safe_json(payload), indent=2), encoding="utf-8")


def execute_task(
    task: SuiteTask, *, smoke: bool, force: bool, workspace_root: Path | None = None
) -> TaskManifest:
    results_root = Path(task.out_dir or default_results_root())
    workspace = workspace_root or Path(__file__).resolve().parents[3]
    task_dir = results_root / "telemetry" / task.task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = task_dir / "stdout.log"
    stderr_path = task_dir / "stderr.log"
    manifest_path = task_dir / "manifest.json"
    command = build_command(task, smoke=smoke)

    manifest = TaskManifest(
        task_id=task.task_id,
        lane=task.lane,
        command=command,
        status="planned",
        created_at_utc=_now_utc(),
        model=task.model,
        pair=task.pair,
        smoke=bool(smoke),
        evidence_goal=task.evidence_goal,
        outputs=[str(results_root / rel) for rel in task.outputs],
        telemetry={
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "cwd": str(workspace),
        },
    )

    if not force and all(Path(p).exists() for p in manifest.outputs):
        manifest.status = "skipped_existing"
        manifest.finished_at_utc = _now_utc()
        manifest_path.write_text(
            json.dumps(manifest.to_dict(), indent=2), encoding="utf-8"
        )
        return manifest

    manifest.status = "running"
    manifest.started_at_utc = _now_utc()
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")

    with (
        open(stdout_path, "w", encoding="utf-8") as stdout_f,
        open(stderr_path, "w", encoding="utf-8") as stderr_f,
    ):
        proc = subprocess.run(
            command, cwd=str(workspace), stdout=stdout_f, stderr=stderr_f, check=False
        )

    manifest.exit_code = int(proc.returncode)
    manifest.finished_at_utc = _now_utc()
    manifest.status = "success" if proc.returncode == 0 else "failed"
    manifest.telemetry["command_text"] = " ".join(shlex.quote(part) for part in command)
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Plan or execute the multiscale Modal experiment suite."
    )
    ap.add_argument("--task-ids", type=str, default="")
    ap.add_argument("--lanes", type=str, default="")
    ap.add_argument("--emit-plan", type=str, default="")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    lanes = [x.strip() for x in str(args.lanes).split(",") if x.strip()]
    task_ids = [x.strip() for x in str(args.task_ids).split(",") if x.strip()]
    tasks = resolve_tasks(task_ids=task_ids or None, lanes=lanes or None)

    if str(args.emit_plan).strip():
        write_plan(Path(str(args.emit_plan)).resolve(), tasks)

    if not args.execute:
        print(
            json.dumps({"planned_tasks": [task.to_dict() for task in tasks]}, indent=2)
        )
        return 0

    manifests = [
        execute_task(task, smoke=bool(args.smoke), force=bool(args.force))
        for task in tasks
    ]
    failures = [m for m in manifests if m.status == "failed"]
    print(json.dumps({"executed": [m.to_dict() for m in manifests]}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
