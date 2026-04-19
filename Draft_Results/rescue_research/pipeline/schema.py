from __future__ import annotations

from typing import Dict, Iterable


RUN_MANIFEST_REQUIRED_KEYS = (
    "schema_version",
    "created_at",
    "pipeline",
    "pairs",
    "models",
    "seeds",
    "backend",
    "blind_slice_sealed",
)


def missing_manifest_keys(payload: Dict, required: Iterable[str] = RUN_MANIFEST_REQUIRED_KEYS) -> list[str]:
    missing = []
    for key in required:
        if key not in payload:
            missing.append(str(key))
    return missing

