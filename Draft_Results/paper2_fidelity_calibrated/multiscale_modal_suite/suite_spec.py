from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable


SUITE_DIR = Path(__file__).resolve().parent
PAPER2_DIR = SUITE_DIR.parent
WORKSPACE_ROOT = PAPER2_DIR.parents[2]


def resolve_results_root() -> Path:
    return Path(
        os.environ.get(
            "MULTISCALE_RESULTS_ROOT",
            str(PAPER2_DIR / "results" / "multiscale_modal_suite"),
        )
    )


RESULTS_ROOT = resolve_results_root()


@dataclass(frozen=True)
class ModelSpec:
    key: str
    family: str
    hf_hint: str
    context_tokens: int | None
    sliding_window: int | None
    notes: str


@dataclass(frozen=True)
class PairTrack:
    slug: str
    pair_id: str
    family: str
    tier: str
    source_status: str
    notes: str = ""


@dataclass(frozen=True)
class TaskTemplate:
    slug: str
    lane: str
    command_kind: str
    models: tuple[str, ...] = ()
    pairs: tuple[str, ...] = ()
    description: str = ""
    n_icl: int = 64
    n_select: int = 300
    n_eval: int = 200
    smoke_n_eval: int = 8
    extra_args: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    evidence_goal: str = ""


@dataclass(frozen=True)
class SuiteTask:
    task_id: str
    lane: str
    command_kind: str
    description: str
    model: str | None
    pair: str | None
    n_icl: int
    n_select: int
    n_eval: int
    smoke_n_eval: int
    out_dir: str
    outputs: tuple[str, ...]
    extra_args: tuple[str, ...] = ()
    evidence_goal: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


MODEL_SPECS: dict[str, ModelSpec] = {
    "270m": ModelSpec(
        key="270m",
        family="gemma",
        hf_hint="resolve via project config or HF override",
        context_tokens=None,
        sliding_window=None,
        notes="Treat as the small-scale contrast model used by existing paper2 runners.",
    ),
    "1b": ModelSpec(
        key="1b",
        family="gemma-3",
        hf_hint="google/gemma-3-1b-it",
        context_tokens=32768,
        sliding_window=512,
        notes="Architecture confirmed from authenticated HF config.",
    ),
    "4b": ModelSpec(
        key="4b",
        family="gemma-3",
        hf_hint="google/gemma-3-4b-it",
        context_tokens=131072,
        sliding_window=1024,
        notes="Use as the scale-up comparison against 1B.",
    ),
}


PAIR_TRACKS: dict[str, PairTrack] = {
    "hin": PairTrack(
        slug="hin",
        pair_id="aksharantar_hin_latin",
        family="Indic",
        tier="core_causal",
        source_status="verified",
        notes="Highest-confidence 1B content-specificity language.",
    ),
    "tel": PairTrack(
        slug="tel",
        pair_id="aksharantar_tel_latin",
        family="Indic",
        tier="core_causal",
        source_status="verified",
        notes="Second core language; important for proxy-vs-behavior divergence.",
    ),
    "ben": PairTrack(
        slug="ben",
        pair_id="aksharantar_ben_latin",
        family="Indic",
        tier="breadth",
        source_status="needs_dataset_registration",
        notes="Breadth-only until external source is confirmed in this repo.",
    ),
    "guj": PairTrack(
        slug="guj",
        pair_id="aksharantar_guj_latin",
        family="Indic",
        tier="breadth",
        source_status="needs_dataset_registration",
        notes="Breadth-only until external source is confirmed in this repo.",
    ),
    "kan": PairTrack(
        slug="kan",
        pair_id="aksharantar_kan_latin",
        family="Indic",
        tier="breadth",
        source_status="needs_dataset_registration",
        notes="Breadth-only until external source is confirmed in this repo.",
    ),
    "arb": PairTrack(
        slug="arb",
        pair_id="english_arabic",
        family="non_indic",
        tier="extension",
        source_status="synthetic_or_external",
        notes="Non-Indic extension lane; do not use as primary evidence without external data.",
    ),
    "cyr": PairTrack(
        slug="cyr",
        pair_id="english_cyrillic",
        family="non_indic",
        tier="extension",
        source_status="synthetic_or_external",
        notes="Non-Indic extension lane; do not use as primary evidence without external data.",
    ),
}


TASK_TEMPLATES: tuple[TaskTemplate, ...] = (
    TaskTemplate(
        slug="phase0_token_visibility",
        lane="architecture",
        command_kind="phase0_token_visibility",
        models=("270m", "1b", "4b"),
        pairs=(PAIR_TRACKS["hin"].pair_id, PAIR_TRACKS["tel"].pair_id),
        description="Measure local/global token visibility on the exact prompt rendering path.",
        n_icl=64,
        n_select=300,
        n_eval=200,
        smoke_n_eval=16,
        outputs=(
            "phase0/visibility_summary.json",
            "phase0/visibility_rows.csv",
        ),
        evidence_goal="Freeze architecture-conditioned visibility before interpretability claims.",
    ),
    TaskTemplate(
        slug="premise_gate",
        lane="behavioral",
        command_kind="premise_gate",
        models=("270m", "1b", "4b"),
        pairs=(PAIR_TRACKS["hin"].pair_id, PAIR_TRACKS["tel"].pair_id),
        description="Establish whether each scale exhibits a real ICL rescue gap before mechanistic work.",
        n_icl=64,
        n_select=300,
        n_eval=200,
        smoke_n_eval=24,
        outputs=("premise_gate/{model}/{pair}.json",),
        evidence_goal="Reject scales/pairs that do not show a reliable premise gap.",
    ),
    TaskTemplate(
        slug="fidelity_sanity",
        lane="preflight",
        command_kind="fidelity_sanity",
        models=("270m", "1b", "4b"),
        pairs=(PAIR_TRACKS["hin"].pair_id, PAIR_TRACKS["tel"].pair_id),
        description="Run fast fidelity checks before any large intervention sweep.",
        n_icl=64,
        n_select=300,
        n_eval=64,
        smoke_n_eval=16,
        outputs=(
            "fidelity/{model}/{pair}.csv",
            "fidelity/{model}/{pair}.png",
        ),
        evidence_goal="Fail fast on broken hookpoints or incompatible model adapters.",
    ),
    TaskTemplate(
        slug="script_space_map",
        lane="representation",
        command_kind="script_space_map",
        models=("270m", "1b", "4b"),
        pairs=(PAIR_TRACKS["hin"].pair_id, PAIR_TRACKS["tel"].pair_id),
        description="Map native-script mass across layers for explicit-ZS and ICL conditions.",
        n_icl=64,
        n_select=300,
        n_eval=200,
        smoke_n_eval=24,
        outputs=(
            "script_space/{model}/{pair}_summary.csv",
            "script_space/{model}/{pair}_mass.png",
        ),
        evidence_goal="Show where script-space preference appears and how it changes with scale.",
    ),
    TaskTemplate(
        slug="paper2_cfom_scaling",
        lane="intervention",
        command_kind="paper2_cfom",
        models=("270m", "1b", "4b"),
        pairs=(PAIR_TRACKS["hin"].pair_id, PAIR_TRACKS["tel"].pair_id),
        description="Run the fidelity-calibrated intervention pipeline on the three target scales.",
        n_icl=64,
        n_select=300,
        n_eval=200,
        smoke_n_eval=32,
        outputs=("paper2/{model}/{pair}.json",),
        evidence_goal="Provide a scaling lane consistent with the existing paper2 runner conventions.",
    ),
    TaskTemplate(
        slug="one_b_final_bundle",
        lane="1b_closure",
        command_kind="one_b_final_bundle",
        models=("1b",),
        pairs=(PAIR_TRACKS["hin"].pair_id, PAIR_TRACKS["tel"].pair_id),
        description="Run the full high-N 1B closure bundle on Modal A100 40GB.",
        n_icl=16,
        n_select=300,
        n_eval=50,
        smoke_n_eval=4,
        outputs=(
            "1b_final/hindi_mlp_contribution_n50.json",
            "1b_final/telugu_mlp_contribution_n50.json",
            "1b_final/density_degradation_hindi_telugu_n30.json",
            "1b_final/head_attribution_hindi_n50.json",
            "1b_final/head_attribution_telugu_n50.json",
            "1b_final/joint_attn_mlp_grouped_hindi.json",
            "1b_final/joint_attn_mlp_grouped_telugu.json",
            "1b_final/content_specificity_by_count.json",
            "1b_final/head_attribution_seed_robustness.json",
        ),
        evidence_goal="Close the 1B story with high-N causal evidence.",
    ),
)


def _task_id(
    template: TaskTemplate, model: str | None = None, pair: str | None = None
) -> str:
    pieces = [template.slug]
    if model:
        pieces.append(model)
    if pair:
        pieces.append(pair)
    return "__".join(pieces)


def _expand_outputs(
    template: TaskTemplate, model: str | None, pair: str | None
) -> tuple[str, ...]:
    out: list[str] = []
    for pattern in template.outputs:
        out.append(pattern.format(model=model or "all", pair=pair or "all"))
    return tuple(out)


def expand_template(template: TaskTemplate) -> list[SuiteTask]:
    if template.command_kind in {
        "phase0_token_visibility",
        "script_space_map",
        "one_b_final_bundle",
    }:
        model_label = ",".join(template.models) if template.models else None
        pair_label = ",".join(template.pairs) if template.pairs else None
        return [
            SuiteTask(
                task_id=_task_id(template),
                lane=template.lane,
                command_kind=template.command_kind,
                description=template.description,
                model=model_label,
                pair=pair_label,
                n_icl=template.n_icl,
                n_select=template.n_select,
                n_eval=template.n_eval,
                smoke_n_eval=template.smoke_n_eval,
                out_dir=str(resolve_results_root()),
                outputs=_expand_outputs(template, model_label, pair_label),
                extra_args=template.extra_args,
                evidence_goal=template.evidence_goal,
            )
        ]

    tasks: list[SuiteTask] = []
    for model in template.models:
        for pair in template.pairs:
            tasks.append(
                SuiteTask(
                    task_id=_task_id(template, model=model, pair=pair),
                    lane=template.lane,
                    command_kind=template.command_kind,
                    description=template.description,
                    model=model,
                    pair=pair,
                    n_icl=template.n_icl,
                    n_select=template.n_select,
                    n_eval=template.n_eval,
                    smoke_n_eval=template.smoke_n_eval,
                    out_dir=str(resolve_results_root()),
                    outputs=_expand_outputs(template, model, pair),
                    extra_args=template.extra_args,
                    evidence_goal=template.evidence_goal,
                )
            )
    return tasks


def build_suite_plan(selected_lanes: Iterable[str] | None = None) -> list[SuiteTask]:
    lane_filter = {str(x).strip() for x in selected_lanes or [] if str(x).strip()}
    tasks: list[SuiteTask] = []
    for template in TASK_TEMPLATES:
        if lane_filter and template.lane not in lane_filter:
            continue
        tasks.extend(expand_template(template))
    return tasks


def plan_as_dicts(selected_lanes: Iterable[str] | None = None) -> list[dict]:
    return [task.to_dict() for task in build_suite_plan(selected_lanes)]
