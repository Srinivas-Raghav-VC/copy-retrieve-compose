#!/usr/bin/env python3
"""
Build an external transliteration dataset file as JSONL (+ sidecar metadata).

Why JSONL:
  - easy to stream and diff
  - stable per-row schema
  - works with rescue_research.data_pipeline.ingest auto-discovery

Sidecar metadata:
  For row-based formats (jsonl/csv), `ingest.py` will read
  `<datafile>.<suffix>.meta.json` and attach the provenance to the dataset
  source descriptor.

Output JSONL schema (one record per line):
  {"english": <id>, "source": <source_text>, "target": <target_text>}
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _pick_first_key(sample: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in sample:
            return k
    return None


def _infer_cols(sample: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    # Heuristics only; if wrong, user must override with explicit args.
    id_col = _pick_first_key(sample, ["english", "id", "key", "lemma"])
    src_col = _pick_first_key(sample, ["source", "src", "native", "input", "text", "word"])
    tgt_col = _pick_first_key(sample, ["target", "tgt", "transliteration", "romanized", "output"])
    return id_col, src_col, tgt_col


def main() -> int:
    ap = argparse.ArgumentParser(description="Build external transliteration JSONL from HF datasets.")
    ap.add_argument("--pair-id", type=str, required=True, help="Internal pair id for naming/provenance.")
    ap.add_argument("--hf-dataset", type=str, required=True, help="Hugging Face dataset name.")
    ap.add_argument("--hf-config", type=str, default="", help="Optional HF dataset config name.")
    ap.add_argument("--hf-split", type=str, default="train", help="HF split to export.")
    ap.add_argument("--out", type=str, required=True, help="Output JSONL path.")
    ap.add_argument("--max-rows", type=int, default=0, help="0 means all rows.")
    ap.add_argument("--id-col", type=str, default="", help="Column for record id (optional).")
    ap.add_argument("--source-col", type=str, default="", help="Column for source text.")
    ap.add_argument("--target-col", type=str, default="", help="Column for target text.")
    ap.add_argument("--min-len", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--meta-name", type=str, default="", help="Source name recorded in file metadata.")
    ap.add_argument("--meta-url", type=str, default="", help="Source URL recorded in file metadata.")
    ap.add_argument("--meta-license", type=str, default="unknown")
    ap.add_argument("--meta-version-date", type=str, default="")
    args = ap.parse_args()

    # Import at runtime so `--help` works without deps.
    from datasets import load_dataset  # type: ignore

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds_kwargs: Dict[str, Any] = {}
    if str(args.hf_config).strip():
        ds_kwargs["name"] = str(args.hf_config).strip()

    dataset = load_dataset(str(args.hf_dataset).strip(), **ds_kwargs)
    split_name = str(args.hf_split).strip()
    if split_name not in dataset:
        raise ValueError(f"Split {split_name!r} not found. Available: {list(dataset.keys())}")
    split = dataset[split_name]

    if len(split) == 0:
        raise ValueError("Dataset split is empty.")

    # Infer columns from first row if user didn't specify.
    first = dict(split[0])
    id_col = str(args.id_col).strip() or None
    src_col = str(args.source_col).strip() or None
    tgt_col = str(args.target_col).strip() or None
    if not (src_col and tgt_col):
        inf_id, inf_src, inf_tgt = _infer_cols(first)
        id_col = id_col or inf_id
        src_col = src_col or inf_src
        tgt_col = tgt_col or inf_tgt
    if not src_col or not tgt_col:
        raise ValueError(
            "Could not infer source/target columns. Pass --source-col and --target-col explicitly."
        )

    max_rows = int(args.max_rows)
    n_total = len(split)
    n_take = min(n_total, max_rows) if max_rows > 0 else n_total

    meta = {
        "schema_version": "v1",
        "built_at_utc": _now_utc(),
        "pair_id": str(args.pair_id),
        "hf_dataset": str(args.hf_dataset),
        "hf_config": str(args.hf_config),
        "hf_split": split_name,
        "columns": {"id": id_col or "", "source": src_col, "target": tgt_col},
        "name": str(args.meta_name or "").strip() or str(args.hf_dataset),
        "url": str(args.meta_url or "").strip(),
        "license": str(args.meta_license or "unknown").strip(),
        "version_date": str(args.meta_version_date or "").strip(),
    }

    kept = 0
    dropped = 0
    min_len = int(args.min_len)
    max_len = int(args.max_len)
    with out_path.open("w", encoding="utf-8") as f:
        for i in range(int(n_take)):
            row = dict(split[i])
            src = str(row.get(src_col, "")).strip()
            tgt = str(row.get(tgt_col, "")).strip()
            if not src or not tgt:
                dropped += 1
                continue
            if len(src) < min_len or len(src) > max_len or len(tgt) < min_len or len(tgt) > max_len:
                dropped += 1
                continue
            rec_id = ""
            if id_col:
                rec_id = str(row.get(id_col, "")).strip()
            if not rec_id:
                rec_id = f"{args.pair_id}::{split_name}::{i}"
            f.write(json.dumps({"english": rec_id, "source": src, "target": tgt}, ensure_ascii=False) + "\n")
            kept += 1

    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[build] out: {out_path}")
    print(f"[build] meta: {meta_path}")
    print(f"[build] rows: took={n_take} kept={kept} dropped={dropped}")
    print(f"[build] cols: id={id_col!r} source={src_col!r} target={tgt_col!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

