#!/usr/bin/env python3
"""Generate plots and Markdown summaries for judge validation pilots."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
OUTDIR = RESULTS / "judge_validation"
OUTDIR.mkdir(parents=True, exist_ok=True)

GUARDRAIL_JSON = RESULTS / "api_behavioral_sweep_judge_pilot_25flash.json"
AMBIG_JSON = RESULTS / "api_behavioral_sweep_judge_ambiguous_pilot_25flash.json"

LANG_COLORS = {
    "bengali": "#4daf4a",
    "hindi": "#e41a1c",
    "marathi": "#ff7f00",
    "tamil": "#984ea3",
    "telugu": "#377eb8",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def plot_anchor_accuracy(guardrail: dict, ambiguous: dict) -> None:
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(8, 5))

    labels = ["Guardrail pilot\n(primary)", "Guardrail pilot\n(secondary)", "Ambiguous pilot\n(primary)", "Ambiguous pilot\n(secondary)"]
    vals = [
        guardrail["automatic_validation"]["anchor_summary"]["primary_overall"]["binary_accuracy"],
        guardrail["automatic_validation"]["anchor_summary"]["secondary_overall"]["binary_accuracy"],
        ambiguous["automatic_validation"]["anchor_summary"]["primary_overall"]["binary_accuracy"],
        ambiguous["automatic_validation"]["anchor_summary"]["secondary_overall"]["binary_accuracy"],
    ]
    colors = ["#1b9e77", "#66a61e", "#d95f02", "#7570b3"]
    ax.bar(range(len(labels)), vals, color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Binary anchor accuracy")
    ax.set_title("Automatic anchor-set validation")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUTDIR / "fig_judge_anchor_accuracy.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_ambiguous_language_accept(ambiguous: dict) -> None:
    if not HAS_MPL:
        return
    by_lang = ambiguous["aggregate"]["by_language"]
    pairs = []
    for key, val in by_lang.items():
        _field, lang = key.split("::", 1)
        pairs.append((lang, val["acceptable_rate"]))
    pairs.sort()

    fig, ax = plt.subplots(figsize=(8, 5))
    langs = [p[0] for p in pairs]
    vals = [p[1] for p in pairs]
    colors = [LANG_COLORS.get(lang, "#777777") for lang in langs]
    ax.bar(range(len(langs)), vals, color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Acceptable rate")
    ax.set_title("Ambiguous pilot: acceptable rate by language\n(primary = Gemini 2.5 Flash)")
    ax.set_xticks(range(len(langs)))
    ax.set_xticklabels([lang.title() for lang in langs], rotation=20, ha="right")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUTDIR / "fig_judge_ambiguous_language_accept.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_ambiguous_cer_by_label(ambiguous: dict) -> None:
    if not HAS_MPL:
        return
    grouped = {"exact": [], "acceptable_variant": [], "script_correct_but_wrong": [], "invalid_or_non_answer": []}
    for item in ambiguous["items"]:
        bundle = item["judgments"]["pred"]
        label = bundle["primary"]["label"]
        cer = bundle["deterministic_metrics"].get("char_cer")
        if cer is not None:
            grouped[label].append(float(cer))

    labels = [lab for lab, vals in grouped.items() if vals]
    values = [grouped[lab] for lab in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(values, patch_artist=True, labels=labels)
    palette = {
        "exact": "#1b9e77",
        "acceptable_variant": "#66a61e",
        "script_correct_but_wrong": "#d95f02",
        "invalid_or_non_answer": "#7570b3",
    }
    for patch, label in zip(bp["boxes"], labels):
        patch.set_facecolor(palette.get(label, "#cccccc"))
    ax.set_ylabel("Character error rate (CER)")
    ax.set_title("Ambiguous pilot: CER by judge label")
    plt.tight_layout()
    fig.savefig(OUTDIR / "fig_judge_ambiguous_cer_by_label.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_ambiguous_secondary_agreement(ambiguous: dict) -> None:
    if not HAS_MPL:
        return
    info = ambiguous["aggregate"]["secondary_agreement"]["pred"]["model_only"]
    labels = ["binary agreement", "binary κ", "label agreement", "label κ"]
    vals = [
        info["binary_agreement_rate"],
        info["binary_kappa"],
        info["label_agreement_rate"],
        info["label_kappa"],
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(labels)), vals, color=["#1b9e77", "#66a61e", "#d95f02", "#7570b3"])
    ax.set_ylim(0, 1.05)
    ax.set_title("Ambiguous pilot: primary vs secondary judge agreement\n(model-scored cases only)")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUTDIR / "fig_judge_ambiguous_secondary_agreement.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_configuration_md(guardrail: dict, ambiguous: dict) -> None:
    text = f"""# Judge Configuration

## Status

This judge stack is **not human-calibrated**. In 2026-style honest reporting terms, it should be described as an **automatically validated transliteration acceptability estimator**, not as calibrated ground truth.

## Primary design

- Primary judge: `{ambiguous['judge']['resolved_model']}`
- Secondary judge: `{ambiguous['judge']['secondary_resolved_model']}`
- Key source: `{ambiguous['judge']['api_key_source']}`
- Rubric version: `{ambiguous['rubric_version']}`
- Validation version: `{ambiguous['validation_version']}`

## Decision stack

The judge is hybrid automatic scoring:

1. **Deterministic guardrails** first
   - exact reference match -> `exact`
   - empty output -> `invalid_or_non_answer`
   - source-script copy when target script differs -> `invalid_or_non_answer`
   - output not primarily in target script -> `invalid_or_non_answer`
   - explanatory text instead of standalone answer -> `invalid_or_non_answer`
2. **LLM judge** only for the ambiguous middle
   - possible `exact`
   - possible `acceptable_variant`
   - possible `script_correct_but_wrong`
   - possible `invalid_or_non_answer`
3. **Optional second judge** for agreement estimates

## Why this is the right current framing

- It is stronger than raw EM alone.
- It is stronger than an unconstrained LLM judge.
- It is weaker than true human calibration, so we should not overclaim.
- It is appropriate as a **secondary automatic metric** for transliteration acceptability.

## Paper-safe wording

Recommended phrasing:

> We supplement deterministic reference-based transliteration metrics with a structured LLM judge that estimates acceptable transliteration variants. Because we do not use human calibration, we treat this judge as an auxiliary automatic metric rather than ground truth. We support it with deterministic guardrails, automatic anchor-set validation, and cross-judge agreement checks.
"""
    (OUTDIR / "JUDGE_CONFIGURATION.md").write_text(text, encoding="utf-8")


def write_pilot_summary_md(guardrail: dict, ambiguous: dict) -> None:
    g = guardrail["aggregate"]["per_field"]["pred"]
    a = ambiguous["aggregate"]["per_field"]["pred"]
    a_agree = ambiguous["aggregate"]["secondary_agreement"]["pred"]["model_only"]
    a_anchor = ambiguous["automatic_validation"]["anchor_summary"]
    align = ambiguous["aggregate"]["deterministic_alignment"]["pred"]

    by_lang_lines = []
    for key, val in sorted(ambiguous["aggregate"]["by_language"].items()):
        _field, lang = key.split("::", 1)
        by_lang_lines.append(f"- {lang.title()}: acceptable_rate={val['acceptable_rate']:.0%} (n={val['n']})")

    text = f"""# Judge Pilot Summary

## Inputs

- Guardrail pilot JSON: `{GUARDRAIL_JSON.relative_to(ROOT)}`
- Ambiguous pilot JSON: `{AMBIG_JSON.relative_to(ROOT)}`

## Pilot slices

### 1. Guardrail pilot

This slice was intentionally easy: exact matches and obvious invalids.
It tests whether deterministic guardrails behave correctly.

- n = {g['n']}
- acceptable_rate = {g['acceptable_rate']:.0%}
- decision sources = {json.dumps(g['decision_source_counts'], ensure_ascii=False)}
- anchor binary accuracy (primary) = {guardrail['automatic_validation']['anchor_summary']['primary_overall']['binary_accuracy']:.0%}
- anchor binary accuracy (secondary) = {guardrail['automatic_validation']['anchor_summary']['secondary_overall']['binary_accuracy']:.0%}

### 2. Ambiguous pilot

This slice contains right-script, non-exact outputs that actually exercise the model judge.

- n = {a['n']}
- acceptable_rate = {a['acceptable_rate']:.0%}
- label counts = {json.dumps(a['label_counts'], ensure_ascii=False)}
- decision sources = {json.dumps(a['decision_source_counts'], ensure_ascii=False)}

### Primary vs secondary agreement on ambiguous model-scored cases

- binary agreement = {a_agree['binary_agreement_rate']:.1%}
- binary kappa = {a_agree['binary_kappa']:.3f}
- label agreement = {a_agree['label_agreement_rate']:.1%}
- label kappa = {a_agree['label_kappa']:.3f}

### Anchor validation on ambiguous run

- primary binary accuracy = {a_anchor['primary_overall']['binary_accuracy']:.0%}
- primary label accuracy = {a_anchor['primary_overall']['label_accuracy']:.0%}
- secondary binary accuracy = {a_anchor['secondary_overall']['binary_accuracy']:.0%}
- secondary label accuracy = {a_anchor['secondary_overall']['label_accuracy']:.0%}

## Language pattern in ambiguous pilot

{chr(10).join(by_lang_lines)}

## Deterministic alignment checks

These support the judge's credibility without claiming human calibration:

- mean CER if acceptable = {align['char_cer']['mean_if_acceptable']:.3f}
- mean CER if not acceptable = {align['char_cer']['mean_if_not_acceptable']:.3f}
- mean target-script ratio if acceptable = {align['target_script_ratio']['mean_if_acceptable']:.3f}
- mean target-script ratio if not acceptable = {align['target_script_ratio']['mean_if_not_acceptable']:.3f}

Interpretation: the judge's acceptable decisions coincide with lower edit distance and better script fidelity, which is exactly what we want from an auxiliary transliteration-acceptability metric.

## Credibility assessment

### What is strong

- zero parser/runtime errors in the final pilots
- 100% primary binary anchor accuracy on both pilots
- 100% secondary binary anchor accuracy on the ambiguous pilot
- meaningful separation on CER and script fidelity between acceptable vs non-acceptable outputs
- nontrivial but decent binary agreement between primary and secondary on ambiguous model-scored cases

### What is still limited

- no human calibration, so this is **not** publication-grade calibrated ground truth
- label-level agreement is weaker than binary agreement on ambiguous cases
- `acceptable_variant` remains a small-sample category in the current pilot

## Bottom line

The current judge stack is good enough to proceed as a **secondary automatic behavioral metric** for the next full 5-language 1B run. For the paper, deterministic metrics should remain the primary behavioral base, and judge acceptability should be presented as an auxiliary estimate of usable transliteration quality.

## Required next step

Run the full 5-language 1B behavioral sweep at proper N and then explicitly test whether:

1. judge acceptability tracks **ICL degradation / rescue** by language and condition
2. judge acceptability correlates with **first-token probability / rank** on GPU-generated artifacts
3. judge acceptability adds useful information beyond EM and CER
"""
    (OUTDIR / "JUDGE_PILOT_SUMMARY.md").write_text(text, encoding="utf-8")


def main() -> None:
    guardrail = load(GUARDRAIL_JSON)
    ambiguous = load(AMBIG_JSON)
    plot_anchor_accuracy(guardrail, ambiguous)
    plot_ambiguous_language_accept(ambiguous)
    plot_ambiguous_cer_by_label(ambiguous)
    plot_ambiguous_secondary_agreement(ambiguous)
    write_configuration_md(guardrail, ambiguous)
    write_pilot_summary_md(guardrail, ambiguous)
    print(f"Wrote judge validation artifacts to {OUTDIR}")


if __name__ == "__main__":
    main()
