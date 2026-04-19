from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from rescue_research.reporting.tables import MANDATORY_TABLES

try:
    from rescue_research.reporting.figures import MANDATORY_FIGURES
except ModuleNotFoundError:
    MANDATORY_FIGURES = []


REQUIRED_DIRS = (
    "artifacts/manifests",
    "artifacts/data",
    "artifacts/baseline",
    "artifacts/selection",
    "artifacts/interventions",
    "artifacts/mediation",
    "artifacts/stats",
    "artifacts/figures",
    "artifacts/tables",
    "artifacts/audit",
)


def required_paths_for_pair_seed(
    *,
    pair_id: str,
    model: str,
    seed: int,
) -> List[Path]:
    return [
        Path(f"artifacts/baseline/{model}/{pair_id}/{seed}.json"),
        Path(f"artifacts/selection/{model}/{pair_id}/{seed}.json"),
        Path(f"artifacts/interventions/{model}/{pair_id}/{seed}.json"),
        Path(f"artifacts/mediation/{model}/{pair_id}/{seed}.json"),
    ]


def ensure_contract_dirs(out_dir: Path) -> None:
    for rel in REQUIRED_DIRS:
        (out_dir / rel).mkdir(parents=True, exist_ok=True)


def iter_required_manifest_files() -> Iterable[Path]:
    yield Path("artifacts/manifests/run_manifest.json")
    yield Path("artifacts/manifests/benchmark_registry.json")
    yield Path("artifacts/stats/confirmatory_results.json")
    yield Path("artifacts/stats/exploratory_results.json")
    yield Path("artifacts/stats/attention_control_summary.json")
    yield Path("artifacts/stats/transcoder_variant_summary.json")
    yield Path("artifacts/stats/prompt_format_robustness.json")
    yield Path("artifacts/audit/transcoder_fidelity_gate.json")
    yield Path("artifacts/audit/skeptic_pass.json")
    for name in MANDATORY_TABLES:
        yield Path("artifacts/tables") / name
    for name in MANDATORY_FIGURES:
        yield Path("artifacts/figures") / name

