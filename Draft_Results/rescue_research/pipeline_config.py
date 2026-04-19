"""
Structured configuration for the full confirmatory pipeline.

This supplements legacy `rescue_research.config.RunConfig` with main-track
protocol settings (pair matrix, split targets, hypotheses, and execution policy).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from rescue_research.contracts import (
    EXPLORATORY_K_OPTIONS,
    GLOBAL_ALPHA,
    LOCKED_LANGUAGE_PAIRS,
    PAIR_BACKUPS,
    TARGET_SPLIT_COUNTS,
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    role: str
    requires_confirmatory_pass: bool


@dataclass(frozen=True)
class PairSpec:
    pair_id: str
    family: str
    source_language: str
    source_script: str
    target_script: str
    backups: List[str]


@dataclass
class PipelineConfig:
    """
    Top-level config for the new end-to-end pipeline.

    Values mirror the locked protocol defaults and can be overridden for pilots.
    """

    out_dir: Path
    pairs: List[str] = field(default_factory=lambda: list(LOCKED_LANGUAGE_PAIRS))
    models: List[str] = field(default_factory=lambda: ["1b", "12b"])
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1337])
    k_confirmatory: int = 8
    k_exploratory: List[int] = field(default_factory=lambda: list(EXPLORATORY_K_OPTIONS))
    confirmatory_topk_values: List[int] = field(default_factory=lambda: [25])
    alpha: float = GLOBAL_ALPHA
    eval_open_count: int = int(TARGET_SPLIT_COUNTS["eval_open"])
    eval_blind_count: int = int(TARGET_SPLIT_COUNTS["eval_blind"])
    selection_count: int = int(TARGET_SPLIT_COUNTS["selection"])
    icl_bank_count: int = int(TARGET_SPLIT_COUNTS["icl_bank"])
    backend: str = "local"
    execute_experiments: bool = True
    run_blind_eval: bool = False
    mediation_band_size: int = 3  # Run mediation at top-K layers; load from stats.yaml if available
    compare_variants: bool = False  # Run affine vs skipless transcoder comparison (Figure 4)
    run_quality_eval: bool = False  # Compute CER/exact match in baseline (--eval-generation)
    eval_generation: bool = False # Same as run_quality_eval, mapped to --eval-generation
    patch_style: str = "sparse"     # 'sparse' or 'substitute'

    # Final protocol knobs (thorough+skeptic)
    cross_task_dataset: str = "flores200_en_te"
    cross_task_sample_size: int = 600
    unicode_normalization: str = "NFC"
    auto_scale_policy: str = "norm_match"  # confirmatory: norm_match; exploratory: multipliers
    auto_scale_multipliers: List[float] = field(default_factory=lambda: [3.0, 5.0, 10.0])
    seed_stability_jaccard_top5_min: float = 0.70
    seed_stability_jaccard_top25_min: float = 0.50
    random_layer_control_mode: str = "bracket"  # early=25%, late=85%
    contamination_filter_mode: str = "per_pair_per_model"
    contamination_exclude_percentile: float = 5.0
    compute_budget_gpu_hours: float = 100.0

    task: str = "transliteration"
    control_mode: str = "default"
    n_seeds: int = 0

    split_policy: str = "adaptive"  # adaptive|strict split derivation from available pool size
    allow_custom_pairs: bool = False  # If False, full_confirmatory requires locked pair matrix.
    substitution_plan: List[dict] = field(default_factory=list)  # Auditable locked->backup substitutions.

    # Minimum viable split counts used by adaptive policy when data is limited.
    runtime_min_icl: int = 4
    runtime_min_selection: int = 8
    runtime_min_eval: int = 12
    data_min_icl_bank: int = 8
    data_min_selection: int = 16
    data_min_eval_open: int = 24
    data_min_eval_blind: int = 8

    # Confirmatory readiness guardrails (fail fast on underpowered pair pools).
    enforce_pair_readiness: bool = True
    allow_underpowered_pairs: bool = False
    min_confirmatory_pool: int = 40
    min_confirmatory_icl: int = 4
    min_confirmatory_selection: int = 12
    min_confirmatory_eval: int = 24

    def ensure_out_dir(self) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        return self.out_dir


def default_model_specs() -> Dict[str, ModelSpec]:
    return {
        "270m": ModelSpec("270m", role="calibration_anchor", requires_confirmatory_pass=False),
        "1b": ModelSpec("1b", role="core_confirmatory", requires_confirmatory_pass=True),
        "4b": ModelSpec("4b", role="core_confirmatory", requires_confirmatory_pass=True),
        "12b": ModelSpec("12b", role="robustness_subset", requires_confirmatory_pass=False),
    }


def default_pair_specs() -> Dict[str, PairSpec]:
    return {
        "hindi_telugu": PairSpec(
            pair_id="hindi_telugu",
            family="indic",
            source_language="Hindi",
            source_script="Devanagari",
            target_script="Telugu",
            backups=list(PAIR_BACKUPS["hindi_telugu"]),
        ),
        "hindi_tamil": PairSpec(
            pair_id="hindi_tamil",
            family="indic",
            source_language="Hindi",
            source_script="Devanagari",
            target_script="Tamil",
            backups=list(PAIR_BACKUPS["hindi_tamil"]),
        ),
        "english_arabic": PairSpec(
            pair_id="english_arabic",
            family="non_indic",
            source_language="English",
            source_script="Latin",
            target_script="Arabic",
            backups=list(PAIR_BACKUPS["english_arabic"]),
        ),
        "english_cyrillic": PairSpec(
            pair_id="english_cyrillic",
            family="non_indic",
            source_language="English",
            source_script="Latin",
            target_script="Cyrillic (Russian)",
            backups=list(PAIR_BACKUPS["english_cyrillic"]),
        ),
    }

