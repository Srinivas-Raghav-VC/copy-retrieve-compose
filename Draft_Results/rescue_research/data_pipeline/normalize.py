from __future__ import annotations

import unicodedata
from typing import Dict, Iterable, List


def _norm(text: str) -> str:
    return unicodedata.normalize("NFC", str(text or "").strip())


def normalize_record(record: Dict[str, str]) -> Dict[str, str]:
    out = dict(record)
    out["source"] = _norm(record.get("source", ""))
    out["target"] = _norm(record.get("target", ""))
    out["english"] = _norm(record.get("english", ""))
    return out


def normalize_records(records: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Normalize records and remove exact duplicates by (source, target, english).
    """
    dedup: List[Dict[str, str]] = []
    seen = set()
    for raw in records:
        rec = normalize_record(raw)
        if not rec["source"] or not rec["target"] or not rec["english"]:
            continue
        key = (rec["source"], rec["target"], rec["english"])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(rec)
    return dedup

