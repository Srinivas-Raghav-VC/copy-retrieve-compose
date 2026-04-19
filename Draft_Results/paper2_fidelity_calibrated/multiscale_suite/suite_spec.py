"""
Suite Specification for Gemma 3 1B ICL Mechanistic Analysis

This module defines all tasks for the multiscale mechanistic interpretability suite.
Each task is grounded in the theoretical framework from the literature review.

Version: 2.0 (Literature-Grounded)
Date: 2026-03-21
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum


class Priority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TaskFamily(Enum):
    CAUSAL_INTERVENTION = "causal_intervention"
    PHASE_TRANSITION = "phase_transition"
    STRATEGY_ANALYSIS = "strategy_analysis"
    PROMPT_ENGINEERING = "prompt_engineering"
    SAMPLING_ANALYSIS = "sampling_analysis"
    ROBUSTNESS = "robustness"
    CAUSAL_TRACING = "causal_tracing"
    ATTENTION_VISUALIZATION = "attention_visualization"
    ARCHITECTURAL_ANALYSIS = "architectural_analysis"


@dataclass
class InterventionSpec:
    """Specification for a single intervention within an experiment."""

    intervention_id: str
    description: str
    layers: List[int]
    component: str  # "attention_output", "mlp_output", "both", "all"
    source: str  # "icl_run", "corrupt_run"
    target: str  # "zs_run", "corrupt_run"
    expected: str


@dataclass
class MetricSpec:
    """Specification for metrics to collect."""

    primary: str
    secondary: List[str] = field(default_factory=list)
    bootstrap_ci: bool = True
    n_bootstrap: int = 1000
    ci_level: float = 0.95


@dataclass
class TaskSpec:
    """Complete specification for a single experiment task."""

    task_id: str
    name: str
    family: TaskFamily
    priority: Priority
    languages: List[str]
    n_items: int
    icl_count: Optional[int]
    seeds: List[int]
    model: str
    precision: str
    theoretical_basis: List[str]
    hypothesis: Optional[str] = None
    interventions: List[InterventionSpec] = field(default_factory=list)
    metrics: Optional[MetricSpec] = None
    pass_criteria: List[str] = field(default_factory=list)
    failure_interpretation: Optional[str] = None
    depends_on: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)


# =============================================================================
# CRITICAL EXPERIMENTS (Tier 1)
# =============================================================================

CRITICAL_EXPERIMENTS: List[TaskSpec] = [
    TaskSpec(
        task_id="EXP-CRIT-01",
        name="Joint Attention+MLP Grouped Intervention",
        family=TaskFamily.CAUSAL_INTERVENTION,
        priority=Priority.CRITICAL,
        languages=["hindi", "telugu"],
        n_items=50,
        icl_count=16,
        seeds=[42],
        model="google/gemma-3-1b-it",
        precision="bf16",
        theoretical_basis=[
            "Coupled mechanism hypothesis (our finding)",
            "Phase transition theory (Park et al., ICLR 2025)",
            "Strategy framework (Wurgaft et al., NeurIPS 2025)",
        ],
        hypothesis=(
            "Joint attention+MLP intervention should show significantly more rescue than "
            "either component alone, proving the coupled mechanism."
        ),
        interventions=[
            InterventionSpec(
                intervention_id="attn_global_only",
                description="Replace attention output at global layers L05,L11,L17,L23",
                layers=[5, 11, 17, 23],
                component="attention_output",
                source="icl_run",
                target="zs_run",
                expected="~0 PE",
            ),
            InterventionSpec(
                intervention_id="mlp_global_only",
                description="Replace MLP output at global layers L05,L11,L17,L23",
                layers=[5, 11, 17, 23],
                component="mlp_output",
                source="icl_run",
                target="zs_run",
                expected="small positive PE",
            ),
            InterventionSpec(
                intervention_id="both_global",
                description="Replace BOTH attention and MLP at global layers",
                layers=[5, 11, 17, 23],
                component="both",
                source="icl_run",
                target="zs_run",
                expected="SIGNIFICANTLY > attn_global_only + mlp_global_only",
            ),
            InterventionSpec(
                intervention_id="both_L11_L17",
                description="Replace both at key pair L11+L17",
                layers=[11, 17],
                component="both",
                source="icl_run",
                target="zs_run",
                expected="Key pair test",
            ),
            InterventionSpec(
                intervention_id="both_L17_L23",
                description="Replace both at key pair L17+L23",
                layers=[17, 23],
                component="both",
                source="icl_run",
                target="zs_run",
                expected="Key pair test",
            ),
            InterventionSpec(
                intervention_id="both_all_layers",
                description="Replace both at all 26 layers (upper bound)",
                layers=list(range(26)),
                component="both",
                source="icl_run",
                target="zs_run",
                expected="Upper bound",
            ),
            InterventionSpec(
                intervention_id="mlp_all_layers",
                description="Replace MLP at all 26 layers (reference)",
                layers=list(range(26)),
                component="mlp_output",
                source="icl_run",
                target="zs_run",
                expected="Reference",
            ),
        ],
        metrics=MetricSpec(
            primary="first_token_probability_PE",
            secondary=["first_token_rank_change", "fraction_icl_gap_recovered"],
            bootstrap_ci=True,
            n_bootstrap=1000,
            ci_level=0.95,
        ),
        pass_criteria=[
            "both_global PE significantly > attn_global_only PE (p < 0.05)",
            "both_global PE significantly > mlp_global_only PE (p < 0.05)",
            "Bootstrap CI excludes zero for both_global",
        ],
        failure_interpretation=(
            "If both_global ≈ 0: Mechanism more distributed than hypothesized; "
            "requires circuit tracing, not component patching"
        ),
        outputs=[
            "results/1b_final/joint_intervention_{language}.json",
            "results/1b_final/figures/fig_m6_joint_intervention.pdf",
        ],
    ),
    TaskSpec(
        task_id="EXP-CRIT-02",
        name="Context-Length Geometry Sweep for Phase Transition",
        family=TaskFamily.PHASE_TRANSITION,
        priority=Priority.CRITICAL,
        languages=["hindi", "telugu"],
        n_items=30,
        icl_count=None,  # Variable
        seeds=[42],
        model="google/gemma-3-1b-it",
        precision="bf16",
        theoretical_basis=[
            "Representational phase transition (Park et al., ICLR 2025)",
            "Evidence accumulation model (Bigelow et al., ICLR 2024)",
        ],
        hypothesis=(
            "ICL behavior should show discontinuous jumps at critical context lengths, "
            "indicating phase transition in representation geometry."
        ),
        metrics=MetricSpec(
            primary="first_token_probability",
            secondary=[
                "exact_match",
                "judge_acceptable",
                "layerwise_pca_variance_explained",
                "dirichlet_energy_token_graph",
                "representation_alignment_to_task",
            ],
            bootstrap_ci=True,
        ),
        pass_criteria=[
            "Identify at least one phase transition per language",
            "Geometry transition precedes or coincides with behavioral transition",
        ],
        outputs=[
            "results/1b_final/context_length_geometry_{language}.json",
            "results/1b_final/figures/fig_phase_transition.pdf",
        ],
    ),
    TaskSpec(
        task_id="EXP-CRIT-03",
        name="Cross-Language Pretrained Prior Interaction",
        family=TaskFamily.STRATEGY_ANALYSIS,
        priority=Priority.CRITICAL,
        languages=[
            "hindi",
            "telugu",
            "bengali",
            "marathi",
            "kannada",
            "tamil",
            "gujarati",
            "malayalam",
        ],
        n_items=50,
        icl_count=16,
        seeds=[42],
        model="google/gemma-3-1b-it",
        precision="bf16",
        theoretical_basis=[
            "Strategy selection framework (Wurgaft et al., NeurIPS 2025)",
            "Pretrained prior interaction",
        ],
        hypothesis=(
            "High-resource languages use generalization strategy; "
            "low-resource languages use memorization or fall back to format signal."
        ),
        metrics=MetricSpec(
            primary="content_specificity_ratio",
            secondary=[
                "exact_match",
                "judge_acceptable",
                "head_attribution_top_heads",
                "logit_lens_separation_trajectory",
                "attention_mass_on_icl_examples",
            ],
            bootstrap_ci=True,
        ),
        pass_criteria=[
            "Content-specificity correlates with pretraining resource level",
            "Script sharing (Marathi/Hindi) shows transfer",
        ],
        outputs=[
            "results/1b_final/language_strategy_analysis.json",
            "results/1b_final/figures/fig_language_gradient.pdf",
        ],
    ),
]


# =============================================================================
# IMPORTANT EXPERIMENTS (Tier 2)
# =============================================================================

IMPORTANT_EXPERIMENTS: List[TaskSpec] = [
    TaskSpec(
        task_id="EXP-IMP-01",
        name="Prompt Structure Variations",
        family=TaskFamily.PROMPT_ENGINEERING,
        priority=Priority.HIGH,
        languages=["hindi", "telugu"],
        n_items=30,
        icl_count=16,
        seeds=[42],
        model="google/gemma-3-1b-it",
        precision="bf16",
        theoretical_basis=[
            "Strategy framework: prompts affect posterior P(task|context)"
        ],
        hypothesis=(
            "Instructional prompts shift posterior toward generalization; "
            "arrow-free prompts force model to infer task."
        ),
        outputs=["results/1b_final/prompt_variations.json"],
    ),
    TaskSpec(
        task_id="EXP-IMP-02",
        name="Temperature Sensitivity Sweep",
        family=TaskFamily.SAMPLING_ANALYSIS,
        priority=Priority.HIGH,
        languages=["hindi", "telugu"],
        n_items=50,
        icl_count=16,
        seeds=[42],
        model="google/gemma-3-1b-it",
        precision="bf16",
        theoretical_basis=[
            "Strategy mixture revealed by sampling diversity (Wurgaft et al.)"
        ],
        hypothesis=(
            "High temperature should reveal strategy diversity; "
            "low temperature should show dominant strategy."
        ),
        outputs=["results/1b_final/temperature_sensitivity.json"],
    ),
    TaskSpec(
        task_id="EXP-IMP-03",
        name="MLP Contribution (N=50)",
        family=TaskFamily.CAUSAL_INTERVENTION,
        priority=Priority.HIGH,
        languages=["hindi", "telugu"],
        n_items=50,
        icl_count=16,
        seeds=[42],
        model="google/gemma-3-1b-it",
        precision="bf16",
        theoretical_basis=["MLP transformation role in coupled mechanism"],
        hypothesis=(
            "Global layer MLPs show positive contribution; "
            "post-global local layers show negative or near-zero contribution."
        ),
        outputs=["results/1b_final/mlp_contribution_{language}_n50.json"],
    ),
    TaskSpec(
        task_id="EXP-IMP-04",
        name="Head Attribution (N=50)",
        family=TaskFamily.CAUSAL_INTERVENTION,
        priority=Priority.HIGH,
        languages=["hindi", "telugu"],
        n_items=50,
        icl_count=16,
        seeds=[42],
        model="google/gemma-3-1b-it",
        precision="bf16",
        theoretical_basis=["Induction/FV head identification"],
        hypothesis=(
            "L11H0 or L14H0 appears in top-5 across both languages; "
            "cross-language overlap ≥ 2 heads in top-5."
        ),
        outputs=["results/1b_final/head_attribution_{language}_n50.json"],
    ),
    TaskSpec(
        task_id="EXP-IMP-05",
        name="Head Attribution Seed Robustness",
        family=TaskFamily.ROBUSTNESS,
        priority=Priority.MEDIUM,
        languages=["hindi"],
        n_items=30,
        icl_count=16,
        seeds=[42, 123, 456],
        model="google/gemma-3-1b-it",
        precision="bf16",
        theoretical_basis=["Stability of head attribution across random seeds"],
        hypothesis="At least 2 of top-3 heads overlap across all 3 seeds.",
        outputs=["results/1b_final/head_attribution_seed_robustness.json"],
    ),
    TaskSpec(
        task_id="EXP-IMP-06",
        name="Density Degradation with Attention Mass",
        family=TaskFamily.ARCHITECTURAL_ANALYSIS,
        priority=Priority.HIGH,
        languages=["hindi", "telugu"],
        n_items=30,
        icl_count=None,  # Variable: 4, 8, 16, 32, 48, 64
        seeds=[42],
        model="google/gemma-3-1b-it",
        precision="bf16",
        theoretical_basis=["Evidence accumulation + architectural bottleneck"],
        hypothesis=(
            "Clear monotonic decline from 8→48 (dilution); "
            "sharp drop at 48→64 (window boundary)."
        ),
        outputs=["results/1b_final/density_degradation_{language}_n30.json"],
    ),
]


# =============================================================================
# EXPLORATORY EXPERIMENTS (Tier 3)
# =============================================================================

EXPLORATORY_EXPERIMENTS: List[TaskSpec] = [
    TaskSpec(
        task_id="EXP-EXP-01",
        name="Head-to-MLP Mediation Analysis",
        family=TaskFamily.CAUSAL_TRACING,
        priority=Priority.MEDIUM,
        languages=["hindi"],
        n_items=30,
        icl_count=16,
        seeds=[42],
        model="google/gemma-3-1b-it",
        precision="bf16",
        theoretical_basis=["Signal propagation path detection"],
        hypothesis="L11H0 →residual → L14H0 causal chain exists.",
        outputs=["results/1b_final/mediation_analysis.json"],
    ),
    TaskSpec(
        task_id="EXP-EXP-02",
        name="Key Head Attention Patterns",
        family=TaskFamily.ATTENTION_VISUALIZATION,
        priority=Priority.MEDIUM,
        languages=["hindi"],
        n_items=10,
        icl_count=16,
        seeds=[42],
        model="google/gemma-3-1b-it",
        precision="bf16",
        theoretical_basis=["Attention pattern analysis for interpreting head function"],
        outputs=["results/1b_final/attention_patterns.json"],
    ),
    TaskSpec(
        task_id="EXP-EXP-03",
        name="ICL Order Independence",
        family=TaskFamily.ROBUSTNESS,
        priority=Priority.LOW,
        languages=["hindi"],
        n_items=30,
        icl_count=16,
        seeds=[42],
        model="google/gemma-3-1b-it",
        precision="bf16",
        theoretical_basis=["Order independence test for ICL mechanism"],
        hypothesis="If ICL uses all examples equally, order should not matter.",
        outputs=["results/1b_final/icl_order_robustness.json"],
    ),
]


# =============================================================================
# SUITE ASSEMBLY
# =============================================================================


def get_all_tasks() -> List[TaskSpec]:
    """Return all tasks in priority order."""
    return CRITICAL_EXPERIMENTS + IMPORTANT_EXPERIMENTS + EXPLORATORY_EXPERIMENTS


def get_tasks_by_priority(priority: Priority) -> List[TaskSpec]:
    """Return tasks filtered by priority."""
    return [t for t in get_all_tasks() if t.priority == priority]


def get_tasks_by_family(family: TaskFamily) -> List[TaskSpec]:
    """Return tasks filtered by family."""
    return [t for t in get_all_tasks() if t.family == family]


def get_task_by_id(task_id: str) -> Optional[TaskSpec]:
    """Return a specific task by ID."""
    for task in get_all_tasks():
        if task.task_id == task_id:
            return task
    return None


def get_dependency_order() -> List[TaskSpec]:
    """Return tasks in dependency order (respecting depends_on field)."""
    all_tasks = get_all_tasks()
    task_map = {t.task_id: t for t in all_tasks}

    # Topological sort
    result = []
    visited = set()

    def visit(task_id: str):
        if task_id in visited:
            return
        task = task_map.get(task_id)
        if task is None:
            return
        visited.add(task_id)
        for dep_id in task.depends_on:
            visit(dep_id)
        result.append(task)

    for task in all_tasks:
        visit(task.task_id)

    return result


# =============================================================================
# VALIDATION
# =============================================================================


def validate_task_specs() -> Dict[str, Any]:
    """Validate all task specs for consistency."""
    results = {
        "valid": True,
        "task_count": len(get_all_tasks()),
        "unique_task_ids": True,
        "missing_research_modules": ["core", "config", "rescue_research"],
        "preflight_ok": False,  # Blocked until modules restored
        "errors": [],
    }

    # Check for duplicate task IDs
    task_ids = [t.task_id for t in get_all_tasks()]
    if len(set(task_ids)) != len(task_ids):
        results["unique_task_ids"] = False
        results["valid"] = False
        results["errors"].append("Duplicate task IDs found")

    # Count by priority
    results["critical_count"] = len(get_tasks_by_priority(Priority.CRITICAL))
    results["high_count"] = len(get_tasks_by_priority(Priority.HIGH))
    results["medium_count"] = len(get_tasks_by_priority(Priority.MEDIUM))
    results["low_count"] = len(get_tasks_by_priority(Priority.LOW))

    return results


if __name__ == "__main__":
    import json

    validation = validate_task_specs()
    print(json.dumps(validation, indent=2))
