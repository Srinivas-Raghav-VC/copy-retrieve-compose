#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _top_rows(rows: List[Dict[str, Any]], key: str, n: int = 5, reverse: bool = True) -> List[Dict[str, Any]]:
    def score(row: Dict[str, Any]) -> float:
        try:
            return float(row.get(key, float("nan")))
        except Exception:
            return float("nan")

    filtered = [row for row in rows if isinstance(row, dict)]
    filtered.sort(key=score, reverse=reverse)
    return filtered[: int(n)]


def _collect_component_summaries(root: Path, stem: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for path in sorted(root.glob(f"results/{stem}/*/4b/{stem}_summary.json")):
        try:
            out[str(path.parent.parent.name)] = _load_json(path)
        except Exception as exc:
            out[str(path.parent.parent.name)] = {"error": str(exc), "path": str(path)}
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Summarize multilingual head-group appendix outputs.")
    ap.add_argument(
        "--out",
        type=str,
        default="paper2_fidelity_calibrated/results/head_group_appendix_summary.json",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    artifacts = {
        "shared_specific_head_ablation": PROJECT_ROOT / "artifacts/phase5_head_groups/shared_specific_head_ablation_4b_multilang.json",
        "shared_head_sufficiency": PROJECT_ROOT / "artifacts/phase5_head_groups/shared_head_sufficiency_panel_4b_multilang.json",
        "additive_synergy": PROJECT_ROOT / "artifacts/phase5_head_groups/additive_synergy_patch_panel_4b.json",
    }

    summary: Dict[str, Any] = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "artifacts": {},
        "component_localization": _collect_component_summaries(PROJECT_ROOT / "paper2_fidelity_calibrated", "component_localization_panel"),
        "layer_output_alignment": _collect_component_summaries(PROJECT_ROOT / "paper2_fidelity_calibrated", "layer_output_alignment_panel"),
    }

    for name, path in artifacts.items():
        if not path.exists():
            summary["artifacts"][name] = {"present": False, "path": str(path)}
            continue
        payload = _load_json(path)
        rows = payload.get("summary", []) if isinstance(payload, dict) else []
        block: Dict[str, Any] = {
            "present": True,
            "path": str(path),
            "n_summary_rows": len(rows),
        }
        if name == "shared_specific_head_ablation":
            block["largest_exact_match_drop"] = _top_rows(rows, "exact_match_drop")
            block["largest_first_entry_drop"] = _top_rows(rows, "first_entry_drop")
        elif name == "shared_head_sufficiency":
            block["largest_first_entry_gain"] = _top_rows(rows, "first_entry_gain")
            block["largest_nll_pos1_improvement"] = _top_rows(rows, "nll_pos1_improvement")
        elif name == "additive_synergy":
            block["largest_first_entry_gain"] = _top_rows(rows, "first_entry_gain")
            block["largest_continuation_gain"] = _top_rows(rows, "continuation_gain")
            block["largest_nll_pos1_improvement"] = _top_rows(rows, "nll_pos1_improvement")
        summary["artifacts"][name] = block

    out_path = (PROJECT_ROOT / str(args.out)).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(str(out_path), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
