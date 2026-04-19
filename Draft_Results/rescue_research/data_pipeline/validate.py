from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

try:
    # Reuse script coverage ranges from canonical pair config when available so
    # data quality checks stay aligned with the active dataset definitions.
    from config_multiscript import _SCRIPT_RANGES as _CANONICAL_SCRIPT_RANGES
except Exception:
    _CANONICAL_SCRIPT_RANGES = {}


SCRIPT_RANGES = {
    "Devanagari": ((0x0900, 0x097F),),
    "Telugu": ((0x0C00, 0x0C7F),),
    "Tamil": ((0x0B80, 0x0BFF),),
    "Arabic": (
        (0x0600, 0x06FF),
        (0x0750, 0x077F),
        (0x08A0, 0x08FF),
    ),
    "Cyrillic (Russian)": (
        (0x0400, 0x04FF),
        (0x0500, 0x052F),
    ),
    "Latin": (
        (0x0041, 0x007A),
        (0x00C0, 0x024F),
    ),
}
for _name, _ranges in _CANONICAL_SCRIPT_RANGES.items():
    if _name in SCRIPT_RANGES:
        continue
    SCRIPT_RANGES[_name] = tuple((int(lo), int(hi)) for lo, hi in _ranges)


@dataclass
class ValidationSummary:
    total: int
    kept: int
    dropped_empty: int
    dropped_script_mismatch: int
    dropped_length_bounds: int


def _contains_script_char(text: str, script_name: str) -> bool:
    ranges = SCRIPT_RANGES.get(script_name)
    if not ranges:
        return True
    for ch in text:
        cp = ord(ch)
        for lo, hi in ranges:
            if lo <= cp <= hi:
                return True
    return False


def validate_records(
    records: Iterable[Dict[str, str]],
    *,
    source_script: str,
    target_script: str,
    min_len: int = 2,
    max_len: int = 32,
) -> tuple[List[Dict[str, str]], ValidationSummary]:
    kept: List[Dict[str, str]] = []
    dropped_empty = 0
    dropped_script = 0
    dropped_length = 0
    total = 0

    for rec in records:
        total += 1
        src = str(rec.get("source", ""))
        tgt = str(rec.get("target", ""))
        eng = str(rec.get("english", ""))
        if not src or not tgt or not eng:
            dropped_empty += 1
            continue
        if len(src) < min_len or len(src) > max_len or len(tgt) < min_len or len(tgt) > max_len:
            dropped_length += 1
            continue
        if not _contains_script_char(src, source_script) or not _contains_script_char(
            tgt, target_script
        ):
            dropped_script += 1
            continue
        kept.append(dict(rec))

    summary = ValidationSummary(
        total=total,
        kept=len(kept),
        dropped_empty=dropped_empty,
        dropped_script_mismatch=dropped_script,
        dropped_length_bounds=dropped_length,
    )
    return kept, summary

