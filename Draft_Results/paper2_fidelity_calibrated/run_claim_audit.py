#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Final claim-audit pass over synthesis outputs.")
    ap.add_argument("--synthesis-json", type=str, default="")
    ap.add_argument("--out", type=str, default="")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    syn_path = Path(args.synthesis_json).resolve() if str(args.synthesis_json).strip() else PROJECT_ROOT / "paper2_fidelity_calibrated" / "results" / "cross_experiment_synthesis" / "cross_experiment_synthesis.json"
    obj = _read(syn_path)
    rows = list(obj.get("rows") or [])
    lines: List[str] = ["# Final Claim Audit", "", f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}", ""]
    lines += [
        "## Falsification checklist",
        "",
        "- If the dense positive control is also weak, the selected site itself may be insufficient rather than merely the sparse basis.",
        "- If activation-difference patching is also weak, the negative result generalizes beyond the current sparse artifact family more strongly.",
        "- If selected-feature mediation is not above matched-random mediation, the G3 path-specific story weakens.",
        "- If direct ICL necessity is absent, the mechanism may be patch-specific rather than naturally used by ICL.",
        "- If transcoder-family consensus is poor, artifact dependence remains a concern.",
        "- If head-attribution stability is poor, keep head claims descriptive rather than mechanistic.",
        "- If helpful ICL does not beat neutral / random / corrupted controls cleanly, recency and generic context confounds remain live.",
        "- If the induction-style matched-example routing check is not stronger under helpful ICL than corrupted ICL, induction-head analogies should remain speculative.",
        "- If selected-layer fidelity is low across both transcoder families, sparse negative results may reflect a decomposition ceiling.",
        "- If center-position effects do not exceed neighbor positions, site-specificity is weak.",
        "- If decoded-vs-latent equivalence fails badly, latent-hook claims need caution.",
        "",
        "## Cell-by-cell audit",
        "",
    ]
    for row in rows:
        lines.append(f"### {row['model']} / {row['pair']}")
        lines.append(f"- Verdict: **{row['verdict']}**")
        lines.append(f"- Support count: {row['support_count']}")
        lines.append(f"- Caution count: {row['caution_count']}")
        if row['verdict'] == 'supports_main_claim':
            lines.append("- Classification: claim-bearing for the narrow intervention-level paper claim.")
        elif row['verdict'] == 'supports_appendix_or_secondary_claim':
            lines.append("- Classification: appendix / secondary-support evidence; keep wording cautious.")
        elif row['verdict'] == 'negative_or_constraining':
            lines.append("- Classification: negative or constraining evidence; do not promote to a positive mechanistic claim.")
        else:
            lines.append("- Classification: exploratory / mixed; not safe for a strong central claim without additional support.")
        lines.append("")
    out_path = Path(args.out).resolve() if str(args.out).strip() else PROJECT_ROOT / "paper2_fidelity_calibrated" / "results" / "claim_audit" / "claim_audit.md"
    _write_text(out_path, "\n".join(lines) + "\n")
    print(f"Saved: {out_path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
