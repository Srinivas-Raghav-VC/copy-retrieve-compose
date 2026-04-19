from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..runners.common import RESULTS_ROOT, write_json


EXECUTION_ROOT = RESULTS_ROOT.parent / "multiscale_modal_suite"
PACKET_ROOT = RESULTS_ROOT / "packets"
PAPER_TABLE_ROOT = RESULTS_ROOT / "paper_tables"
TELEMETRY_ROOT = EXECUTION_ROOT / "telemetry"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_manifests() -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    if not TELEMETRY_ROOT.exists():
        return manifests
    for path in sorted(TELEMETRY_ROOT.glob("*/manifest.json")):
        payload = _read_json(path)
        payload["manifest_path"] = str(path)
        manifests.append(payload)
    return manifests


def _load_plan() -> dict[str, Any]:
    plan_path = PACKET_ROOT / "suite_plan.json"
    return (
        _read_json(plan_path)
        if plan_path.exists()
        else {"status": "missing", "tasks": []}
    )


def _load_claim_templates() -> list[dict[str, Any]]:
    evidence_map = PAPER_TABLE_ROOT / "evidence_map_template.json"
    if not evidence_map.exists():
        return []
    return list(_read_json(evidence_map).get("claims", []))


def _status_counts(manifests: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for manifest in manifests:
        counts[str(manifest.get("status", "unknown"))] += 1
    return dict(sorted(counts.items()))


def _lane_counts(manifests: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    grouped: dict[str, Counter] = defaultdict(Counter)
    for manifest in manifests:
        lane = str(manifest.get("lane", "unknown"))
        status = str(manifest.get("status", "unknown"))
        grouped[lane][status] += 1
    return {
        lane: dict(sorted(counter.items())) for lane, counter in sorted(grouped.items())
    }


def _build_claim_status(
    claims: list[dict[str, Any]], manifests: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_lane: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for manifest in manifests:
        by_lane[str(manifest.get("lane", "unknown"))].append(manifest)

    rows: list[dict[str, Any]] = []
    for claim in claims:
        family = str(claim.get("evidence_family", ""))
        supporting = by_lane.get(family, [])
        completed = [m for m in supporting if str(m.get("status")) == "success"]
        rows.append(
            {
                "claim_id": claim.get("id", ""),
                "label": claim.get("label", ""),
                "family": family,
                "status": "ready_for_fill" if completed else "waiting_on_runs",
                "artifact_count": len(completed),
                "notes": "Populate from successful manifests."
                if completed
                else "No successful execution manifests yet.",
            }
        )
    return rows


def build_execution_summary() -> dict[str, Any]:
    manifests = _load_manifests()
    plan = _load_plan()
    claims = _load_claim_templates()
    summary = {
        "status": "summary_generated",
        "plan_status": plan.get("status", "missing"),
        "plan_task_count": len(plan.get("tasks", [])),
        "manifest_count": len(manifests),
        "status_counts": _status_counts(manifests),
        "lane_counts": _lane_counts(manifests),
        "missing_execution_root": not EXECUTION_ROOT.exists(),
        "missing_telemetry_root": not TELEMETRY_ROOT.exists(),
        "manifests": manifests,
    }
    claim_rows = _build_claim_status(claims, manifests)
    write_json(PAPER_TABLE_ROOT / "execution_summary.json", summary)
    _write_claim_status_csv(PAPER_TABLE_ROOT / "claim_status_live.csv", claim_rows)
    return summary


def _write_claim_status_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "claim_id",
                "label",
                "family",
                "status",
                "artifact_count",
                "notes",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    summary = build_execution_summary()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
