#!/usr/bin/env python3
"""
Build synthetic transliteration datasets for the default paper pairs.

This is designed for the "external-only" path: it writes JSONL files into
  fresh_experiments/data/transliteration/<pair_id>.jsonl
plus provenance sidecars:
  <pair_id>.jsonl.meta.json

Generation method:
  - Take a frequency-ranked word list (wordfreq) for the source language.
  - Transliterate words from source_script -> target_script using the same
    robust helper used by config_multiscript (`_safe_transliterate`), which
    tries aksharamukha and then indic_transliteration when available.

Important:
  This is *synthetic*, not a curated human-annotated dataset. It is still useful
  for scaling up pools and for debugging/benchmarking interventions, but you
  should disclose this clearly if used for paper claims.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Ensure project root is importable when running from this subfolder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _parse_csv(raw: str) -> List[str]:
    vals = [x.strip() for x in str(raw or "").split(",") if x.strip()]
    if not vals:
        raise ValueError(f"Expected non-empty CSV string, got: {raw!r}")
    return vals


def _normalize_word_list(words: List[str], *, min_len: int, max_len: int) -> List[str]:
    out: List[str] = []
    seen = set()
    for w in words:
        if not isinstance(w, str):
            continue
        w = w.strip()
        if not w:
            continue
        # Keep simple "word" items only.
        if any(ch.isspace() for ch in w) or "-" in w:
            continue
        if len(w) < int(min_len) or len(w) > int(max_len):
            continue
        # Filter digits/punctuation.
        if not w.isalpha():
            continue
        key = w.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(w)
    return out


def _wordfreq_lang_code(source_language: str) -> str:
    sl = str(source_language or "").strip().lower()
    if sl == "english":
        return "en"
    if sl == "hindi":
        return "hi"
    raise ValueError(f"Unsupported source_language for synthetic builder: {source_language!r}")


def _contains_script_char(text: str, *, script_name: str) -> bool:
    # Reuse canonical validation ranges when available.
    try:
        from rescue_research.data_pipeline.validate import SCRIPT_RANGES  # type: ignore
    except Exception:
        SCRIPT_RANGES = {}
    ranges = SCRIPT_RANGES.get(script_name)
    if not ranges:
        return True
    for ch in text:
        cp = ord(ch)
        for lo, hi in ranges:
            if int(lo) <= cp <= int(hi):
                return True
    return False


def _build_one_pair(
    pair_id: str,
    *,
    out_dir: Path,
    requested_n: int,
    max_rows: int,
    min_rows: int,
    min_len: int,
    max_len: int,
    source_url: str,
) -> Tuple[Path, int, int]:
    from rescue_research.data_pipeline.ingest import get_pair_prompt_metadata  # type: ignore

    meta = get_pair_prompt_metadata(pair_id)
    source_language = meta["source_language"]
    source_script = meta["source_script"]
    target_script = meta["target_script"]

    # wordfreq is an external dependency; import lazily with a clear error.
    try:
        from wordfreq import top_n_list  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "wordfreq is required for synthetic dataset building. Install via:\n"
            "  pip install -r requirements.txt\n"
            f"Import error: {e}"
        )

    # Use the repo's transliteration helper (tries aksharamukha, then indic_transliteration).
    try:
        import config_multiscript  # type: ignore

        safe_transliterate = getattr(config_multiscript, "_safe_transliterate")
    except Exception as e:
        raise RuntimeError(f"Failed importing config_multiscript transliterator: {e}")

    lang_code = _wordfreq_lang_code(source_language)
    raw = top_n_list(lang_code, n=int(max(1000, requested_n)))
    words = _normalize_word_list(list(raw), min_len=int(min_len), max_len=int(max_len))

    rows: List[Dict[str, str]] = []
    dropped_src_script = 0
    dropped_no_translit = 0
    for w in words:
        # Ensure the source word is actually in the claimed source script (especially for Hindi).
        if not _contains_script_char(w, script_name=source_script):
            dropped_src_script += 1
            continue
        out = safe_transliterate(w, source_script, target_script)
        if not out:
            dropped_no_translit += 1
            continue
        rows.append({"english": w, "source": w, "target": out})
        if len(rows) >= int(max_rows):
            break

    if len(rows) < int(min_rows):
        raise RuntimeError(
            f"Pair {pair_id!r}: only built {len(rows)} rows (min_rows={int(min_rows)}). "
            "This usually means transliteration libraries are missing or the script mapping is unsupported.\n"
            "Try: pip install -r requirements.txt\n"
            f"Stats: dropped_src_script={dropped_src_script} dropped_no_translit={dropped_no_translit}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = (out_dir / f"{pair_id}.jsonl").resolve()
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    sidecar = out_path.with_suffix(out_path.suffix + ".meta.json")
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "built_at_utc": _now_utc(),
                "pair_id": pair_id,
                "name": f"synthetic_wordfreq_transliteration::{pair_id}",
                "url": str(source_url).strip() or "https://github.com/rspeer/wordfreq",
                "license": "see upstream (wordfreq + transliteration libs)",
                "version_date": "",
                "generator": {
                    "type": "synthetic_wordfreq_transliteration",
                    "source_language": source_language,
                    "source_script": source_script,
                    "target_script": target_script,
                    "wordfreq_lang_code": lang_code,
                    "requested_n": int(requested_n),
                    "max_rows": int(max_rows),
                    "min_len": int(min_len),
                    "max_len": int(max_len),
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return out_path, len(rows), dropped_no_translit


def main() -> int:
    ap = argparse.ArgumentParser(description="Build synthetic transliteration datasets for default pairs.")
    ap.add_argument(
        "--pairs",
        type=str,
        default="hindi_telugu,hindi_tamil,english_arabic,english_cyrillic",
        help="Comma-separated pair ids.",
    )
    ap.add_argument("--out-dir", type=str, default="data/transliteration")
    ap.add_argument("--requested-n", type=int, default=50000, help="How many words to request from wordfreq.")
    ap.add_argument("--max-rows", type=int, default=5000, help="Max rows written per pair.")
    ap.add_argument("--min-rows", type=int, default=500, help="Fail if fewer rows built per pair.")
    ap.add_argument("--min-len", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=32)
    ap.add_argument(
        "--source-url",
        type=str,
        default="https://github.com/rspeer/wordfreq",
        help="Recorded in provenance sidecar.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    pairs = _parse_csv(args.pairs)

    print(f"[synthetic] out_dir: {out_dir}", flush=True)
    print(f"[synthetic] pairs: {pairs}", flush=True)
    print(
        f"[synthetic] requested_n={int(args.requested_n)} max_rows={int(args.max_rows)} min_rows={int(args.min_rows)}",
        flush=True,
    )

    for pair_id in pairs:
        print(f"\n[synthetic] building {pair_id} ...", flush=True)
        out_path, n_rows, dropped_no_translit = _build_one_pair(
            pair_id,
            out_dir=out_dir,
            requested_n=int(args.requested_n),
            max_rows=int(args.max_rows),
            min_rows=int(args.min_rows),
            min_len=int(args.min_len),
            max_len=int(args.max_len),
            source_url=str(args.source_url),
        )
        print(f"[synthetic] wrote: {out_path} ({n_rows} rows) dropped_no_translit={dropped_no_translit}", flush=True)

    print("\n[synthetic] done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
