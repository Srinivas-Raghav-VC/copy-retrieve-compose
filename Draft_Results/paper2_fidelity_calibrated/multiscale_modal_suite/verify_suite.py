from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import sys
from pathlib import Path

from .legacy_commands import expected_script_paths
from .suite_spec import SUITE_DIR, build_suite_plan


def _compile_python_files() -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    for path in sorted(SUITE_DIR.glob("*.py")):
        if path.name == "__init__.py":
            continue
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            issues.append({"path": str(path), "error": f"SyntaxError: {exc}"})
    return issues


def _import_health() -> dict[str, str]:
    report: dict[str, str] = {}
    extra_paths = [
        str(SUITE_DIR.parents[2]),
        str(SUITE_DIR.parents[1]),
    ]
    for extra in reversed(extra_paths):
        if extra not in sys.path:
            sys.path.insert(0, extra)
    for module_name in ["core", "config", "rescue_research", "modal", "marimo"]:
        spec = importlib.util.find_spec(module_name)
        report[module_name] = "present" if spec is not None else "missing"
    return report


def run_verification() -> dict:
    tasks = build_suite_plan()
    task_ids = [task.task_id for task in tasks]
    outputs = [output for task in tasks for output in task.outputs]
    script_paths = expected_script_paths()

    missing_scripts = [str(path) for path in script_paths if not path.exists()]
    syntax_issues = _compile_python_files()
    import_health = _import_health()
    missing_research_modules = [
        name
        for name in ["core", "config", "rescue_research"]
        if import_health.get(name) != "present"
    ]

    return {
        "task_count": len(tasks),
        "unique_task_ids": len(set(task_ids)) == len(task_ids),
        "duplicate_outputs": sorted(
            {path for path in outputs if outputs.count(path) > 1}
        ),
        "missing_legacy_scripts": missing_scripts,
        "syntax_issues": syntax_issues,
        "import_health": import_health,
        "missing_research_modules": missing_research_modules,
        "preflight_ok": not missing_research_modules,
        "task_lanes": sorted({task.lane for task in tasks}),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Verify the multiscale Modal suite wiring."
    )
    ap.add_argument("--out", type=str, default="")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    report = run_verification()
    blob = json.dumps(report, indent=2)
    if str(args.out).strip():
        out_path = Path(str(args.out)).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(blob, encoding="utf-8")
    print(blob)
    has_failures = bool(
        report["duplicate_outputs"]
        or report["missing_legacy_scripts"]
        or report["syntax_issues"]
        or report["missing_research_modules"]
    )
    return 1 if has_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
