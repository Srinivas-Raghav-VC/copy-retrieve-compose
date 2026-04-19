from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List

from rescue_research.pipeline.artifact_contracts import (
    iter_required_manifest_files,
    required_paths_for_pair_seed,
)
from rescue_research.pipeline.schema import missing_manifest_keys


@dataclass
class ValidationReport:
    ok: bool
    missing_paths: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def _exists(out_dir: Path, rel: Path) -> bool:
    return (out_dir / rel).exists()


def validate_artifacts(
    *,
    out_dir: Path,
    pairs: Iterable[str],
    models: Iterable[str],
    seeds: Iterable[int],
) -> ValidationReport:
    missing: List[str] = []
    for rel in iter_required_manifest_files():
        if not _exists(out_dir, rel):
            missing.append(str(rel))

    for pair in pairs:
        for model in models:
            for seed in seeds:
                for rel in required_paths_for_pair_seed(pair_id=pair, model=model, seed=seed):
                    if not _exists(out_dir, rel):
                        missing.append(str(rel))

    warnings: List[str] = []
    manifest_path = out_dir / "artifacts" / "manifests" / "run_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            missing_keys = missing_manifest_keys(manifest)
            if missing_keys:
                warnings.append(
                    "run_manifest.json missing keys: " + ", ".join(sorted(missing_keys))
                )
        except Exception as exc:  # pragma: no cover - defensive
            warnings.append(f"Failed to parse run_manifest.json: {exc}")

    return ValidationReport(ok=not missing, missing_paths=missing, warnings=warnings)

