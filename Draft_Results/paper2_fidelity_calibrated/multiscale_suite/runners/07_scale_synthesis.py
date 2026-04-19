from __future__ import annotations

import csv
from pathlib import Path

from .common import (
    CONFIG_ROOT,
    RESULTS_ROOT,
    load_yaml,
    parse_common_args,
    stage_packet,
    write_json,
)


def main() -> int:
    ap = parse_common_args("Emit the final multiscale synthesis packet.")
    args = ap.parse_args()
    claim_registry = load_yaml(CONFIG_ROOT / "claim_registry.yaml")
    evidence_map_path = RESULTS_ROOT / "paper_tables" / "evidence_map_template.json"
    claim_status_path = RESULTS_ROOT / "paper_tables" / "claim_status_template.csv"

    evidence_map = {
        "claims": [
            {
                "id": claim["id"],
                "label": claim["label"],
                "evidence_family": claim["evidence_family"],
                "strength_target": claim["strength_target"],
                "applies_to": claim["applies_to"],
                "artifact_paths": [],
                "current_status": "unfilled",
                "reviewer_note": "Fill from actual task artifacts before manuscript claims.",
            }
            for claim in claim_registry.get("claims", [])
        ]
    }
    write_json(evidence_map_path, evidence_map)

    claim_status_path.parent.mkdir(parents=True, exist_ok=True)
    with open(claim_status_path, "w", encoding="utf-8", newline="") as handle:
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
        for claim in claim_registry.get("claims", []):
            writer.writerow(
                {
                    "claim_id": claim["id"],
                    "label": claim["label"],
                    "family": claim["evidence_family"],
                    "status": "unfilled",
                    "artifact_count": 0,
                    "notes": "Populate after reruns complete.",
                }
            )

    payload = {
        "lane": "scale_synthesis",
        "status": "packetized",
        "stage_packet": stage_packet(None),
        "required_outputs": [
            "evidence_map.json",
            "claim_status.csv",
            "paper_figures/",
            "paper_tables/",
        ],
        "skeleton_outputs": {
            "evidence_map_template": str(evidence_map_path),
            "claim_status_template": str(claim_status_path),
        },
        "conclusion_rule": "The final theory must separate hard facts, established intervention results, and interpretive unification.",
    }
    out = (
        Path(str(args.emit)).resolve()
        if str(args.emit).strip()
        else RESULTS_ROOT / "packets" / "scale_synthesis_packet.json"
    )
    write_json(out, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
