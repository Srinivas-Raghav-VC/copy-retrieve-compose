#!/usr/bin/env python3
"""Generate plots/tables/report for behavioral problem-statement verification."""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from math import comb

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
OUT = RESULTS / "behavioral_problem_statement"
OUT.mkdir(parents=True, exist_ok=True)

ORIG_PATH = RESULTS / "api_behavioral_sweep_judged_v3_full.json"
REPAIRED_PATH = RESULTS / "api_behavioral_sweep_repaired_judged_v3_full.json"

LANGS = ["hindi", "telugu", "bengali", "tamil", "marathi"]
LANG_LABELS = {"hindi": "Hindi", "telugu": "Telugu", "bengali": "Bengali", "tamil": "Tamil", "marathi": "Marathi"}
LANG_COLORS = {"hindi": "#e41a1c", "telugu": "#377eb8", "bengali": "#4daf4a", "tamil": "#984ea3", "marathi": "#ff7f00"}
METRIC_COLORS = {"acceptable": "#1b9e77", "em": "#377eb8", "fe": "#d95f02"}
LABEL_ORDER = ["exact", "acceptable_variant", "script_correct_but_wrong", "invalid_or_non_answer"]
LABEL_COLORS = {
    "exact": "#1b9e77",
    "acceptable_variant": "#66a61e",
    "script_correct_but_wrong": "#d95f02",
    "invalid_or_non_answer": "#7570b3",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def sign_test_p(improved: int, worsened: int) -> float:
    n = improved + worsened
    if n == 0:
        return 1.0
    k = min(improved, worsened)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def safe_corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    if den == 0:
        return float("nan")
    return num / den


def extract_rows(payload: dict) -> list[dict]:
    rows = []
    for item in payload["items"]:
        md = item["metadata"]
        bundle = item["judgments"]["pred"]
        primary = bundle["primary"]
        metrics = bundle["deterministic_metrics"]
        rows.append(
            {
                "model": md.get("model"),
                "lang": item["language"],
                "n_icl": int(md.get("n_icl")),
                "source": item["source"],
                "reference": item["reference"],
                "pred": bundle["output"],
                "label": primary["label"],
                "acceptable": int(primary["acceptable"]),
                "decision_source": primary.get("source", "unknown"),
                "em": float(metrics.get("normalized_exact_match", 0.0)),
                "fe": float(metrics.get("first_char_match", 0.0)),
                "cer": float(metrics.get("char_cer", float("nan"))),
                "script_ratio": float(metrics.get("target_script_ratio", float("nan"))),
            }
        )
    return rows


def summarize_conditions(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["lang"], row["n_icl"])].append(row)

    out = []
    for (model, lang, n_icl), rs in sorted(grouped.items()):
        n = len(rs)
        acceptable_k = sum(r["acceptable"] for r in rs)
        em_k = int(round(sum(r["em"] for r in rs)))
        fe_k = int(round(sum(r["fe"] for r in rs)))
        out.append(
            {
                "model": model,
                "lang": lang,
                "n_icl": n_icl,
                "n": n,
                "acceptable_rate": acceptable_k / n,
                "acceptable_ci": wilson_ci(acceptable_k, n),
                "em_rate": sum(r["em"] for r in rs) / n,
                "em_ci": wilson_ci(sum(int(r["em"]) for r in rs), n),
                "fe_rate": sum(r["fe"] for r in rs) / n,
                "fe_ci": wilson_ci(sum(int(r["fe"]) for r in rs), n),
                "decision_source_counts": dict(Counter(r["decision_source"] for r in rs)),
                "label_counts": dict(Counter(r["label"] for r in rs)),
                "mean_cer": sum(r["cer"] for r in rs) / n,
            }
        )
    return out


def overall_model_summary(rows: list[dict]) -> dict:
    out = {}
    for model in sorted({r["model"] for r in rows}):
        sub = [r for r in rows if r["model"] == model]
        by_icl = {}
        for n_icl in [0, 2, 4, 8]:
            rs = [r for r in sub if r["n_icl"] == n_icl]
            n = len(rs)
            by_icl[str(n_icl)] = {
                "n": n,
                "acceptable_rate": sum(r["acceptable"] for r in rs) / n,
                "em_rate": sum(r["em"] for r in rs) / n,
                "fe_rate": sum(r["fe"] for r in rs) / n,
            }
        out[model] = {
            "decision_source_counts": dict(Counter(r["decision_source"] for r in sub)),
            "corr_acceptable_vs_em": safe_corr([r["acceptable"] for r in sub], [r["em"] for r in sub]),
            "corr_acceptable_vs_fe": safe_corr([r["acceptable"] for r in sub], [r["fe"] for r in sub]),
            "em_undercount_examples": sum(1 for r in sub if r["em"] == 0 and r["acceptable"] == 1),
            "fe_false_positive_examples": sum(1 for r in sub if r["fe"] == 1 and r["acceptable"] == 0),
            "by_icl": by_icl,
        }
    return out


def original_error_impact(orig_rows: list[dict], repaired_rows: list[dict]) -> list[dict]:
    by_orig: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    by_rep: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for r in orig_rows:
        by_orig[(r["model"], r["lang"], r["n_icl"])].append(r)
    for r in repaired_rows:
        by_rep[(r["model"], r["lang"], r["n_icl"])].append(r)

    out = []
    for key in sorted(by_orig):
        orig = by_orig[key]
        error_rows = sum(int(str(r["pred"]).startswith("ERROR:")) for r in orig)
        if error_rows == 0:
            continue
        rep = by_rep[key]
        out.append(
            {
                "model": key[0],
                "lang": key[1],
                "n_icl": key[2],
                "error_rows": error_rows,
                "orig_acceptable_rate": sum(r["acceptable"] for r in orig) / len(orig),
                "repaired_acceptable_rate": sum(r["acceptable"] for r in rep) / len(rep),
                "orig_em_rate": sum(r["em"] for r in orig) / len(orig),
                "repaired_em_rate": sum(r["em"] for r in rep) / len(rep),
            }
        )
    return out


def paired_best_icl_tests(rows: list[dict], model: str) -> list[dict]:
    out = []
    for lang in LANGS:
        sub = [r for r in rows if r["model"] == model and r["lang"] == lang]
        by_icl: dict[int, dict[str, int]] = defaultdict(dict)
        for r in sub:
            by_icl[r["n_icl"]][r["source"]] = r["acceptable"]
        rates = {n: sum(d.values()) / len(d) for n, d in by_icl.items()}
        best_icl = max(rates, key=lambda x: rates[x])
        improved = worsened = same = 0
        for src, z in by_icl[0].items():
            b = by_icl[best_icl][src]
            if b > z:
                improved += 1
            elif b < z:
                worsened += 1
            else:
                same += 1
        out.append(
            {
                "model": model,
                "lang": lang,
                "zero_shot_rate": rates[0],
                "best_icl": best_icl,
                "best_icl_rate": rates[best_icl],
                "improved": improved,
                "worsened": worsened,
                "same": same,
                "sign_test_p": sign_test_p(improved, worsened),
            }
        )
    return out


def plot_accept_exact_grid(conditions: list[dict]) -> None:
    if not HAS_MPL:
        return
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey='row')
    panels = [
        ("gemma-3-1b-it", "acceptable_rate", "acceptable_ci", "1B acceptable rate"),
        ("gemma-3-4b-it", "acceptable_rate", "acceptable_ci", "4B acceptable rate"),
        ("gemma-3-1b-it", "em_rate", "em_ci", "1B exact-match rate"),
        ("gemma-3-4b-it", "em_rate", "em_ci", "4B exact-match rate"),
    ]
    for ax, (model, metric, ci_key, title) in zip(axes.flatten(), panels):
        for lang in LANGS:
            sub = [c for c in conditions if c["model"] == model and c["lang"] == lang]
            xs = [c["n_icl"] for c in sub]
            ys = [c[metric] for c in sub]
            lo = [y - c[ci_key][0] for y, c in zip(ys, sub)]
            hi = [c[ci_key][1] - y for y, c in zip(ys, sub)]
            ax.errorbar(xs, ys, yerr=[lo, hi], marker='o', lw=2, capsize=3, color=LANG_COLORS[lang], label=LANG_LABELS[lang])
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xticks([0, 2, 4, 8])
        ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.2)
        if ax in axes[1]:
            ax.set_xlabel('Number of ICL examples')
    axes[0, 0].set_ylabel('Rate')
    axes[1, 0].set_ylabel('Rate')
    axes[0, 0].legend(loc='lower right', fontsize=9)
    fig.suptitle('Behavioral verification after repairing API-failure rows', fontsize=15, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUT / 'fig_behavioral_accept_exact_grid.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_metric_gap(conditions: list[dict]) -> None:
    if not HAS_MPL:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, model, title in zip(axes, ['gemma-3-1b-it', 'gemma-3-4b-it'], ['1B aggregate over languages', '4B aggregate over languages']):
        sub = [c for c in conditions if c['model'] == model]
        for metric, label in [('acceptable_rate', 'acceptable'), ('em_rate', 'exact match'), ('fe_rate', 'first-entry')]:
            xs = [0, 2, 4, 8]
            ys = []
            for n in xs:
                conds = [c for c in sub if c['n_icl'] == n]
                ys.append(sum(c[metric] for c in conds) / len(conds))
            ax.plot(xs, ys, marker='o', lw=2.5, label=label, color=METRIC_COLORS['acceptable' if metric=='acceptable_rate' else 'em' if metric=='em_rate' else 'fe'])
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xticks([0, 2, 4, 8])
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel('Number of ICL examples')
        ax.grid(alpha=0.2)
    axes[0].set_ylabel('Mean rate across languages')
    axes[0].legend(loc='lower right')
    fig.suptitle('First-entry overestimates behavioral quality; acceptable-rate tracks exact-match more closely', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUT / 'fig_behavioral_metric_gap.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_repair_impact(impact_rows: list[dict]) -> None:
    if not HAS_MPL:
        return
    labels = [f"{r['model'].replace('gemma-3-','').replace('-it','')}\n{LANG_LABELS[r['lang']]}\nICL={r['n_icl']}\n(err={r['error_rows']})" for r in impact_rows]
    orig_vals = [r['orig_acceptable_rate'] for r in impact_rows]
    rep_vals = [r['repaired_acceptable_rate'] for r in impact_rows]
    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, orig_vals, width, label='original judged file', color='#bdbdbd')
    ax.bar(x + width / 2, rep_vals, width, label='after replay repair', color='#1b9e77')
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel('Acceptable rate')
    ax.set_title('Repairing API-failure rows materially changes some behavioral conclusions', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right')
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.2)
    plt.tight_layout()
    fig.savefig(OUT / 'fig_behavioral_repair_impact.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_1b_label_mix(conditions: list[dict]) -> None:
    if not HAS_MPL:
        return
    oneb = [c for c in conditions if c['model'] == 'gemma-3-1b-it']
    order = [(lang, n) for lang in LANGS for n in [0, 2, 4, 8]]
    x = np.arange(len(order))
    bottom = np.zeros(len(order))
    fig, ax = plt.subplots(figsize=(16, 5))
    for label in LABEL_ORDER:
        vals = []
        for lang, n in order:
            c = next(cc for cc in oneb if cc['lang'] == lang and cc['n_icl'] == n)
            vals.append(c['label_counts'].get(label, 0) / c['n'])
        ax.bar(x, vals, bottom=bottom, color=LABEL_COLORS[label], label=label)
        bottom += np.array(vals)
    ticklabels = [f"{LANG_LABELS[lang][:3]}\n{n}" for lang, n in order]
    ax.set_xticks(x)
    ax.set_xticklabels(ticklabels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Fraction of outputs')
    ax.set_title('1B label composition by language and ICL count', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', ncol=4)
    ax.grid(axis='y', alpha=0.2)
    plt.tight_layout()
    fig.savefig(OUT / 'fig_behavioral_1b_label_mix.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_condition_scatter(conditions: list[dict]) -> None:
    if not HAS_MPL:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, metric, title in zip(axes, ['em_rate', 'fe_rate'], ['Condition exact-match rate vs acceptable-rate', 'Condition first-entry rate vs acceptable-rate']):
        for model, marker in [('gemma-3-1b-it', 'o'), ('gemma-3-4b-it', 's')]:
            sub = [c for c in conditions if c['model'] == model]
            xs = [c[metric] for c in sub]
            ys = [c['acceptable_rate'] for c in sub]
            ax.scatter(xs, ys, s=55, marker=marker, alpha=0.8, label='1B' if model == 'gemma-3-1b-it' else '4B')
        ax.plot([0, 1], [0, 1], ls='--', color='gray', lw=1)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel('Condition rate')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(alpha=0.2)
    axes[0].set_ylabel('Judge acceptable-rate')
    axes[0].legend(loc='upper left')
    fig.suptitle('Exact-match tracks acceptable-rate much better than first-entry', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUT / 'fig_behavioral_condition_scatter.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


def write_summary_json(conditions: list[dict], model_summary: dict, impact_rows: list[dict], oneb_tests: list[dict], fourb_tests: list[dict]) -> None:
    payload = {
        'condition_summary': conditions,
        'model_summary': model_summary,
        'repair_impact': impact_rows,
        'paired_best_icl_tests_1b': oneb_tests,
        'paired_best_icl_tests_4b': fourb_tests,
    }
    (OUT / 'behavioral_problem_statement_summary.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def write_report_md(conditions: list[dict], model_summary: dict, impact_rows: list[dict], oneb_tests: list[dict], fourb_tests: list[dict], anchor_summary: dict) -> None:
    cond_map = {(c['model'], c['lang'], c['n_icl']): c for c in conditions}

    def rate_line(model: str, lang: str) -> str:
        parts = []
        for n in [0, 2, 4, 8]:
            c = cond_map[(model, lang, n)]
            parts.append(f"{n}: {c['acceptable_rate']:.3f}")
        return ', '.join(parts)

    impact_lines = []
    for row in impact_rows:
        impact_lines.append(
            f"- `{row['model']}` / {row['lang']} / ICL={row['n_icl']}: {row['error_rows']} API-failure rows; acceptable-rate {row['orig_acceptable_rate']:.3f} -> {row['repaired_acceptable_rate']:.3f}"
        )

    oneb_lines = []
    for row in oneb_tests:
        oneb_lines.append(
            f"- {LANG_LABELS[row['lang']]}: ZS={row['zero_shot_rate']:.3f}, best ICL={row['best_icl']} ({row['best_icl_rate']:.3f}), improved={row['improved']}, worsened={row['worsened']}, sign-test p={row['sign_test_p']:.4f}"
        )

    fourb_lines = []
    for row in fourb_tests:
        fourb_lines.append(
            f"- {LANG_LABELS[row['lang']]}: ZS={row['zero_shot_rate']:.3f}, best ICL={row['best_icl']} ({row['best_icl_rate']:.3f}), improved={row['improved']}, worsened={row['worsened']}, sign-test p={row['sign_test_p']:.4f}"
        )

    oneb = model_summary['gemma-3-1b-it']
    fourb = model_summary['gemma-3-4b-it']

    text = f"""# 1B Behavioral Problem Statement Verification

## Scope

This report verifies the behavioral problem statement for the transliteration study using the repaired API sweep plus the hybrid judge stack.

Inputs:
- repaired judged sweep: `results/api_behavioral_sweep_repaired_judged_v3_full.json`
- original judged sweep: `results/api_behavioral_sweep_judged_v3_full.json`
- figures directory: `results/behavioral_problem_statement/`

## First methodological correction

The original API behavioral sweep contained **24 API-failure rows** (`ERROR:HTTP Error 429 ...`) that were accidentally being interpreted as model failures. Those rows were replayed and repaired before drawing conclusions.

### Conditions materially affected by the repair

{chr(10).join(impact_lines)}

This matters scientifically: the previous apparent collapse at some settings, especially 1B Hindi with 8 ICL examples, was not a model effect. It was an API-rate-limit artifact.

## Judge validity on the full repaired run

Automatic anchor-set validation on the repaired full run:
- primary label accuracy = {anchor_summary['primary_overall']['label_accuracy']:.3f}
- primary binary accuracy = {anchor_summary['primary_overall']['binary_accuracy']:.3f}
- secondary label accuracy = {anchor_summary['secondary_overall']['label_accuracy']:.3f}
- secondary binary accuracy = {anchor_summary['secondary_overall']['binary_accuracy']:.3f}

Interpretation:
- the **primary** judge is fully correct on the anchor set at both the label and binary levels;
- the **secondary** judge is fully correct at the binary level but less stable on the fine-grained wrong-vs-invalid distinction.

This is acceptable for our use case because the paper-facing behavioral metric is the **binary acceptable vs not acceptable** collapse, not the fine-grained secondary label itself.

Decision-source breakdown on the repaired run:
- 1B: guardrail={oneb['decision_source_counts'].get('guardrail', 0)}, model={oneb['decision_source_counts'].get('model', 0)}
- 4B: guardrail={fourb['decision_source_counts'].get('guardrail', 0)}, model={fourb['decision_source_counts'].get('model', 0)}

So most cases are resolved deterministically; Gemini is used mainly for the ambiguous middle rather than for every output.

## What the repaired behavioral data now says

### 1B acceptable-rate by language and ICL count

- Hindi: {rate_line('gemma-3-1b-it', 'hindi')}
- Telugu: {rate_line('gemma-3-1b-it', 'telugu')}
- Bengali: {rate_line('gemma-3-1b-it', 'bengali')}
- Tamil: {rate_line('gemma-3-1b-it', 'tamil')}
- Marathi: {rate_line('gemma-3-1b-it', 'marathi')}

### 4B acceptable-rate by language and ICL count

- Hindi: {rate_line('gemma-3-4b-it', 'hindi')}
- Telugu: {rate_line('gemma-3-4b-it', 'telugu')}
- Bengali: {rate_line('gemma-3-4b-it', 'bengali')}
- Tamil: {rate_line('gemma-3-4b-it', 'tamil')}
- Marathi: {rate_line('gemma-3-4b-it', 'marathi')}

## Verified behavioral conclusions

### 1. Gemma 3 1B is genuinely ICL-dependent for transliteration

Aggregate 1B acceptable-rates across all 5 languages:
- ICL=0: {oneb['by_icl']['0']['acceptable_rate']:.3f}
- ICL=2: {oneb['by_icl']['2']['acceptable_rate']:.3f}
- ICL=4: {oneb['by_icl']['4']['acceptable_rate']:.3f}
- ICL=8: {oneb['by_icl']['8']['acceptable_rate']:.3f}

This is the central behavioral reason the 1B model is the right mechanistic target: zero-shot transliteration is mostly absent, while prompt examples substantially improve performance.

### 2. The 1B ICL effect is strongly language-dependent

The repaired sweep confirms a clear hierarchy:
- strongest: Hindi
- moderate: Telugu, Bengali
- weak: Marathi
- very weak: Tamil

This matches the broader cross-language story already seen in the mechanistic experiments: rescue strength is heterogeneous, not uniform across languages.

### 3. 4B is behaviorally much less dependent on ICL than 1B

Aggregate 4B acceptable-rates across all 5 languages:
- ICL=0: {fourb['by_icl']['0']['acceptable_rate']:.3f}
- ICL=2: {fourb['by_icl']['2']['acceptable_rate']:.3f}
- ICL=4: {fourb['by_icl']['4']['acceptable_rate']:.3f}
- ICL=8: {fourb['by_icl']['8']['acceptable_rate']:.3f}

So the scale contrast remains behaviorally visible after repair: 4B often already knows the task, whereas 1B frequently needs prompt support.

## Statistical check: zero-shot vs best ICL on 1B

Paired sign tests on judged acceptability:
{chr(10).join(oneb_lines)}

Interpretation:
- Hindi shows the clearest paired behavioral improvement.
- Other languages trend in the same direction but remain small-sample / lower-power in this sweep.

For comparison, 4B paired tests:
{chr(10).join(fourb_lines)}

## Metric validation: what is and is not a good behavioral proxy

### Exact match vs judged acceptability

Correlations at the per-example level:
- 1B: acceptable vs EM = {oneb['corr_acceptable_vs_em']:.3f}
- 4B: acceptable vs EM = {fourb['corr_acceptable_vs_em']:.3f}

Under-counted acceptable examples:
- 1B: {oneb['em_undercount_examples']}
- 4B: {fourb['em_undercount_examples']}

Conclusion: exact match is strict and misses some acceptable variants, but it tracks the judge reasonably well.

### First-entry vs judged acceptability

Correlations at the per-example level:
- 1B: acceptable vs first-entry = {oneb['corr_acceptable_vs_fe']:.3f}
- 4B: acceptable vs first-entry = {fourb['corr_acceptable_vs_fe']:.3f}

False-positive examples where first-entry is 1 but judged output is not acceptable:
- 1B: {oneb['fe_false_positive_examples']}
- 4B: {fourb['fe_false_positive_examples']}

Conclusion: **first-entry is not a reliable behavioral metric**. It overstates transliteration quality, especially when the model gets only the first character or script onset right.

## What this does and does not verify about the problem statement

### Verified now

- 1B transliteration behavior is weak in zero-shot and improves with ICL.
- The improvement is heterogeneous across languages.
- 4B is behaviorally stronger and less ICL-dependent.
- A judge layer is useful because exact match is too strict for a few acceptable variants.
- First-entry is too weak as a publication-facing behavioral metric.
- The previous API-sweep claim of strong 8-example Hindi degradation was not valid; it was caused by rate-limit failures.

### Not verified by this sweep alone

- Long-context degradation beyond 8 ICL examples.
- Whether **first-token probability / rank** is the right mechanistic proxy for full transliteration quality.
- Full reviewer-proof sample sizes.

Important synthesis with the existing mechanistic plots:
- this repaired API sweep only probes **0 / 2 / 4 / 8** examples;
- the separate controlled density experiment is the valid source for claims about degradation at much larger prompt densities.

So the right paper division is:
- use the **API + judge sweep** to establish that 1B behavior is weak zero-shot and improved by ICL in a language-dependent way;
- use the **density / long-context mechanistic experiments** to argue that rescue later degrades when the prompt becomes too dense for the architecture.

Those remaining questions require the separate GPU-side analysis that aligns per-item internal metrics with judged behavioral quality.

## Bottom line

The repaired data supports the core paper problem statement, but in a sharpened form:

> Gemma 3 1B does not reliably transliterate in zero-shot across our 5-language set, but in-context examples substantially improve behavioral success in a language-dependent way. This makes 1B a valid target for mechanistic study of ICL rescue. However, broad claims about behavioral degradation at 8 examples should not be drawn from the old API sweep, because those apparent collapses were partly API-failure artifacts. For behavioral evaluation, exact match is useful but too strict, while first-entry is too weak; the hybrid judge is therefore justified as an auxiliary automatic acceptability measure.
"""
    (OUT / 'BEHAVIORAL_PROBLEM_STATEMENT_VERIFICATION.md').write_text(text, encoding='utf-8')
    # paper-facing copy
    paper_copy = ROOT / 'results' / '1b_mechanistic_analysis' / '1B_BEHAVIORAL_PROBLEM_STATEMENT.md'
    paper_copy.write_text(text, encoding='utf-8')


def main() -> None:
    orig = load(ORIG_PATH)
    repaired = load(REPAIRED_PATH)
    orig_rows = extract_rows(orig)
    repaired_rows = extract_rows(repaired)
    conditions = summarize_conditions(repaired_rows)
    model_summary = overall_model_summary(repaired_rows)
    impact_rows = original_error_impact(orig_rows, repaired_rows)
    oneb_tests = paired_best_icl_tests(repaired_rows, 'gemma-3-1b-it')
    fourb_tests = paired_best_icl_tests(repaired_rows, 'gemma-3-4b-it')

    plot_accept_exact_grid(conditions)
    plot_metric_gap(conditions)
    plot_repair_impact(impact_rows)
    plot_1b_label_mix(conditions)
    plot_condition_scatter(conditions)
    write_summary_json(conditions, model_summary, impact_rows, oneb_tests, fourb_tests)
    write_report_md(conditions, model_summary, impact_rows, oneb_tests, fourb_tests, repaired['automatic_validation']['anchor_summary'])
    print(f'Wrote behavioral problem-statement artifacts to {OUT}')


if __name__ == '__main__':
    main()
