#!/usr/bin/env python3
"""
Inspect pair data sources and pool size as seen by rescue_research ingestion.

Use this to verify that:
  - your external dataset file is auto-discovered (data/transliteration/*.jsonl)
  - provenance sidecar metadata is attached to the SourceDescriptor
  - pool size is large enough for your planned splits
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is importable when running from this subfolder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rescue_research.data_pipeline.ingest import load_pair_records_bundle


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect pair dataset bundle.")
    ap.add_argument("--pair", type=str, required=True)
    ap.add_argument("--external-only", action="store_true")
    ap.add_argument("--out", type=str, default="", help="Optional JSON output path.")
    args = ap.parse_args()

    pair_id = str(args.pair).strip()
    bundle = load_pair_records_bundle(pair_id, include_builtin=not bool(args.external_only))
    payload = {
        "pair_id": bundle.pair_id,
        "n_rows": len(bundle.rows),
        "source_counts": bundle.source_counts,
        "sources": [s.__dict__ for s in bundle.sources],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    if str(args.out).strip():
        out = Path(args.out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[inspect] wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
