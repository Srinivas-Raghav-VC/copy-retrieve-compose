#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PAIRS = ["aksharantar_hin_latin", "aksharantar_tel_latin"]
MODELS = ["1b", "4b"]


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate a human spot-check packet of strongest / weakest / weirdest cases.")
    ap.add_argument("--out", type=str, default="")
    return ap.parse_args()


def _top_bottom(rows: List[Dict[str, Any]], key: str, n: int = 3) -> List[Dict[str, Any]]:
    vals = [r for r in rows if np.isfinite(float(r.get(key, float("nan"))))]
    vals.sort(key=lambda r: float(r.get(key, float("nan"))))
    return vals[:n] + vals[-n:]


def main() -> int:
    lines = ["# Spotcheck Packet", "", f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}", ""]
    for model in MODELS:
        for pair in PAIRS:
            lines.append(f"## {model} / {pair}")
            # G3 strongest/weakest heads by mediation specificity
            p = PROJECT_ROOT / "paper2_fidelity_calibrated" / "results" / "head_to_mlp_edge_attribution" / pair / model / "head_to_mlp_edge_attribution.json"
            if p.exists():
                obj = _read(p)
                rows = list(obj.get("summary_by_head") or [])
                lines.append("### G3 head mediation extremes")
                for row in _top_bottom(rows, "mean_feature_mediated_drop_first_prob"):
                    lines.append(f"- {row.get('head')}: med_drop={row.get('mean_feature_mediated_drop_first_prob')} rand_med_drop={row.get('mean_random_feature_mediated_drop_first_prob')}")
                lines.append("")
            # G6 feature necessity extremes
            p = PROJECT_ROOT / "paper2_fidelity_calibrated" / "results" / "feature_knockout_panel" / pair / model / "feature_knockout_panel.json"
            if p.exists():
                obj = _read(p)
                rows = list(obj.get("summary_by_feature_index") or [])
                lines.append("### G6 feature knockout extremes")
                for row in _top_bottom(rows, "mean_drop_from_full_patch_first_prob"):
                    lines.append(f"- feature {row.get('feature_index')}: drop={row.get('mean_drop_from_full_patch_first_prob')} nll={row.get('mean_increase_from_full_patch_target_pos1_nll')}")
                lines.append("")
            # direct icl necessity weird cases
            p = PROJECT_ROOT / "paper2_fidelity_calibrated" / "results" / "direct_icl_feature_necessity" / pair / model / "direct_icl_feature_necessity.json"
            if p.exists():
                obj = _read(p)
                rows = list(obj.get("item_rows") or [])
                lines.append("### Direct ICL necessity strongest / weakest items")
                for row in _top_bottom(rows, "core_drop_first_prob"):
                    lines.append(f"- {row.get('word_ood')} -> {row.get('word_hindi')}: core_drop={row.get('core_drop_first_prob')} random_drop={row.get('random_drop_first_prob')}")
                lines.append("")
    out_path = Path(args.out).resolve() if str(args.out).strip() else PROJECT_ROOT / "paper2_fidelity_calibrated" / "results" / "spotcheck_packet" / "spotcheck_packet.md"
    _write_text(out_path, "\n".join(lines) + "\n")
    print(f"Saved: {out_path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
