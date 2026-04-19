#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


PAIR_LABELS = {
    "aksharantar_hin_latin": "Hindi",
    "aksharantar_tel_latin": "Telugu",
    "aksharantar_ben_latin": "Bengali",
    "aksharantar_tam_latin": "Tamil",
    "aksharantar_mar_latin": "Marathi",
}
PAIR_COLORS = {
    "aksharantar_hin_latin": "#e41a1c",
    "aksharantar_tel_latin": "#377eb8",
    "aksharantar_ben_latin": "#4daf4a",
    "aksharantar_tam_latin": "#984ea3",
    "aksharantar_mar_latin": "#ff7f00",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Analyze judged first-token proxy validation artifacts.")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=None)
    return ap.parse_args()


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _rankdata(values: List[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    return ranks


def pearson(xs: List[float], ys: List[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    if den == 0:
        return float("nan")
    return num / den


def spearman(xs: List[float], ys: List[float]) -> float:
    return pearson(_rankdata(xs), _rankdata(ys))


def auc_score(scores: List[float], labels: List[int]) -> float:
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    total = 0.0
    for p in pos:
        for n in neg:
            total += 1.0
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / total if total else float("nan")


def bootstrap_diff(values_a: List[float], values_b: List[float], *, n_boot: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    if not values_a or not values_b:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    diffs = []
    arr_a = np.array(values_a, dtype=float)
    arr_b = np.array(values_b, dtype=float)
    for _ in range(int(n_boot)):
        samp_a = rng.choice(arr_a, size=len(arr_a), replace=True)
        samp_b = rng.choice(arr_b, size=len(arr_b), replace=True)
        diffs.append(float(np.mean(samp_a) - np.mean(samp_b)))
    diffs = np.array(diffs, dtype=float)
    return float(np.mean(diffs)), float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def load_rows(path: Path) -> tuple[dict, List[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for item in payload["items"]:
        metadata = dict(item.get("metadata") or {})
        bundle = item["judgments"]["pred"]
        primary = bundle["primary"]
        rows.append(
            {
                "source": item["source"],
                "reference": item["reference"],
                "language": item["language"],
                "pair": metadata.get("pair", "unknown"),
                "model": metadata.get("model", "unknown"),
                "n_icl": int(metadata.get("n_icl", -1)),
                "pred": bundle["output"],
                "acceptable": int(primary["acceptable"]),
                "label": primary["label"],
                "decision_source": primary.get("source", "unknown"),
                "judge_reason": primary.get("reason", ""),
                "exact_match_local": _safe_float(metadata.get("exact_match_local")),
                "akshara_cer_local": _safe_float(metadata.get("akshara_cer_local")),
                "first_entry_correct_local": _safe_float(metadata.get("first_entry_correct_local")),
                "script_compliance_local": _safe_float(metadata.get("script_compliance_local")),
                "first_prob": _safe_float(metadata.get("first_prob")),
                "first_logit": _safe_float(metadata.get("first_logit")),
                "first_rank": _safe_float(metadata.get("first_rank")),
                "first_entropy": _safe_float(metadata.get("first_entropy")),
                "first_logit_gap": _safe_float(metadata.get("first_logit_gap")),
                "prompt_token_len": _safe_float(metadata.get("prompt_token_len")),
            }
        )
    return payload, rows


def summarize_rows(rows: List[dict]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for pair in sorted({r["pair"] for r in rows}):
        sub = [r for r in rows if r["pair"] == pair]
        labels = [r["acceptable"] for r in sub]
        first_prob = [r["first_prob"] for r in sub]
        inv_rank = [1.0 / r["first_rank"] if r["first_rank"] > 0 else 0.0 for r in sub]
        em = [r["exact_match_local"] for r in sub]
        cer = [r["akshara_cer_local"] for r in sub]
        acc_prob = [r["first_prob"] for r in sub if r["acceptable"] == 1 and math.isfinite(r["first_prob"])]
        non_prob = [r["first_prob"] for r in sub if r["acceptable"] == 0 and math.isfinite(r["first_prob"])]
        acc_rank = [math.log10(r["first_rank"]) for r in sub if r["acceptable"] == 1 and r["first_rank"] > 0]
        non_rank = [math.log10(r["first_rank"]) for r in sub if r["acceptable"] == 0 and r["first_rank"] > 0]
        prob_diff = bootstrap_diff(acc_prob, non_prob)
        rank_diff = bootstrap_diff(non_rank, acc_rank)
        out[pair] = {
            "n": len(sub),
            "pearson_first_prob_vs_acceptable": pearson(first_prob, labels),
            "spearman_first_prob_vs_acceptable": spearman(first_prob, labels),
            "pearson_inv_rank_vs_acceptable": pearson(inv_rank, labels),
            "spearman_inv_rank_vs_acceptable": spearman(inv_rank, labels),
            "pearson_first_prob_vs_em": pearson(first_prob, em),
            "pearson_inv_rank_vs_em": pearson(inv_rank, em),
            "pearson_first_prob_vs_neg_cer": pearson(first_prob, [-v for v in cer]),
            "pearson_inv_rank_vs_neg_cer": pearson(inv_rank, [-v for v in cer]),
            "auc_first_prob_for_acceptable": auc_score(first_prob, labels),
            "auc_inv_rank_for_acceptable": auc_score(inv_rank, labels),
            "mean_first_prob_if_acceptable": float(np.mean(acc_prob)) if acc_prob else float("nan"),
            "mean_first_prob_if_not_acceptable": float(np.mean(non_prob)) if non_prob else float("nan"),
            "mean_log10_rank_if_acceptable": float(np.mean(acc_rank)) if acc_rank else float("nan"),
            "mean_log10_rank_if_not_acceptable": float(np.mean(non_rank)) if non_rank else float("nan"),
            "bootstrap_first_prob_diff": {
                "mean": prob_diff[0],
                "ci_lo": prob_diff[1],
                "ci_hi": prob_diff[2],
            },
            "bootstrap_log10_rank_gap_notacc_minus_acc": {
                "mean": rank_diff[0],
                "ci_lo": rank_diff[1],
                "ci_hi": rank_diff[2],
            },
        }
    return out


def summarize_conditions(rows: List[dict]) -> List[dict]:
    grouped: Dict[tuple[str, int], List[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["pair"], row["n_icl"])] += [row]
    out = []
    for (pair, n_icl), rs in sorted(grouped.items()):
        out.append(
            {
                "pair": pair,
                "n_icl": int(n_icl),
                "n": len(rs),
                "acceptable_rate": float(np.mean([r["acceptable"] for r in rs])),
                "exact_match_rate": float(np.mean([r["exact_match_local"] for r in rs])),
                "mean_first_prob": float(np.mean([r["first_prob"] for r in rs])),
                "mean_inv_rank": float(np.mean([1.0 / r["first_rank"] if r["first_rank"] > 0 else 0.0 for r in rs])),
                "mean_log10_rank": float(np.mean([math.log10(r["first_rank"]) for r in rs if r["first_rank"] > 0])),
            }
        )
    return out


def collect_failure_examples(rows: List[dict], *, top_k: int = 8) -> Dict[str, List[dict]]:
    high_prob_bad = sorted(
        [r for r in rows if r["acceptable"] == 0 and math.isfinite(r["first_prob"])],
        key=lambda r: (-r["first_prob"], r["pair"], r["n_icl"], r["source"]),
    )[:top_k]
    low_prob_good = sorted(
        [r for r in rows if r["acceptable"] == 1 and math.isfinite(r["first_prob"])],
        key=lambda r: (r["first_prob"], r["pair"], r["n_icl"], r["source"]),
    )[:top_k]
    return {
        "high_first_prob_but_not_acceptable": high_prob_bad,
        "low_first_prob_but_acceptable": low_prob_good,
    }


def plot_boxplots(rows: List[dict], outdir: Path) -> None:
    if not HAS_MPL:
        return
    pairs = sorted({r["pair"] for r in rows})
    fig, axes = plt.subplots(2, len(pairs), figsize=(6 * len(pairs), 9))
    if len(pairs) == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    for col, pair in enumerate(pairs):
        sub = [r for r in rows if r["pair"] == pair]
        acc_prob = [r["first_prob"] for r in sub if r["acceptable"] == 1 and math.isfinite(r["first_prob"])]
        non_prob = [r["first_prob"] for r in sub if r["acceptable"] == 0 and math.isfinite(r["first_prob"])]
        acc_rank = [math.log10(r["first_rank"]) for r in sub if r["acceptable"] == 1 and r["first_rank"] > 0]
        non_rank = [math.log10(r["first_rank"]) for r in sub if r["acceptable"] == 0 and r["first_rank"] > 0]

        ax = axes[0, col]
        bp = ax.boxplot([acc_prob, non_prob], patch_artist=True, tick_labels=["acceptable", "not acceptable"])
        for patch, color in zip(bp['boxes'], ['#1b9e77', '#d95f02']):
            patch.set_facecolor(color)
        ax.set_title(f"{PAIR_LABELS.get(pair, pair)}: first_prob by judge")
        ax.set_ylabel("First-token probability")
        ax.grid(axis='y', alpha=0.2)

        ax = axes[1, col]
        bp = ax.boxplot([acc_rank, non_rank], patch_artist=True, tick_labels=["acceptable", "not acceptable"])
        for patch, color in zip(bp['boxes'], ['#1b9e77', '#d95f02']):
            patch.set_facecolor(color)
        ax.set_title(f"{PAIR_LABELS.get(pair, pair)}: log10(first_rank) by judge")
        ax.set_ylabel("log10(first-token rank)")
        ax.grid(axis='y', alpha=0.2)
    fig.suptitle("First-token proxy separation on the same items judged for usability", fontsize=16, fontweight='bold')
    plt.tight_layout()
    fig.savefig(outdir / "fig_m1_proxy_boxplots.png", dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_condition_scatter(conditions: List[dict], outdir: Path) -> None:
    if not HAS_MPL:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, key, title in zip(axes, ["mean_first_prob", "mean_inv_rank"], ["Mean first-token probability vs acceptable-rate", "Mean inverse first-token rank vs acceptable-rate"]):
        for pair in sorted({c['pair'] for c in conditions}):
            sub = [c for c in conditions if c['pair'] == pair]
            ax.plot(
                [c[key] for c in sub],
                [c['acceptable_rate'] for c in sub],
                marker='o',
                lw=2,
                color=PAIR_COLORS.get(pair, '#555555'),
                label=PAIR_LABELS.get(pair, pair),
            )
            for c in sub:
                ax.annotate(str(c['n_icl']), (c[key], c['acceptable_rate']), textcoords='offset points', xytext=(4, 4), fontsize=8)
        ax.set_title(title)
        ax.set_xlabel(key)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel('Judge acceptable-rate')
    axes[0].legend(loc='lower right')
    fig.suptitle('Condition-level proxy alignment', fontsize=15, fontweight='bold')
    plt.tight_layout()
    fig.savefig(outdir / 'fig_m1_condition_proxy_scatter.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_quantile_curves(rows: List[dict], outdir: Path) -> None:
    if not HAS_MPL:
        return
    pairs = sorted({r['pair'] for r in rows})
    fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 4.5), sharey=True)
    if len(pairs) == 1:
        axes = [axes]
    for ax, pair in zip(axes, pairs):
        sub = [r for r in rows if r['pair'] == pair and math.isfinite(r['first_prob'])]
        sub = sorted(sub, key=lambda r: r['first_prob'])
        bins = np.array_split(sub, 5)
        xs = []
        ys = []
        for idx, bucket in enumerate(bins, start=1):
            if len(bucket) == 0:
                continue
            xs.append(float(np.mean([r['first_prob'] for r in bucket])))
            ys.append(float(np.mean([r['acceptable'] for r in bucket])))
        ax.plot(xs, ys, marker='o', lw=2, color=PAIR_COLORS.get(pair, '#555555'))
        ax.set_title(PAIR_LABELS.get(pair, pair))
        ax.set_xlabel('Mean first_prob in quantile bin')
        ax.grid(alpha=0.2)
    axes[0].set_ylabel('Judge acceptable-rate')
    fig.suptitle('Acceptable-rate rises with first-token probability', fontsize=15, fontweight='bold')
    plt.tight_layout()
    fig.savefig(outdir / 'fig_m1_proxy_quantiles.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_counts_overlay(conditions: List[dict], outdir: Path) -> None:
    if not HAS_MPL:
        return
    pairs = sorted({c['pair'] for c in conditions})
    fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 4.5), sharey=True)
    if len(pairs) == 1:
        axes = [axes]
    for ax, pair in zip(axes, pairs):
        sub = sorted([c for c in conditions if c['pair'] == pair], key=lambda c: c['n_icl'])
        xs = [c['n_icl'] for c in sub]
        ax.plot(xs, [c['acceptable_rate'] for c in sub], marker='o', lw=2.5, color='#1b9e77', label='judge acceptable-rate')
        ax.plot(xs, [c['exact_match_rate'] for c in sub], marker='o', lw=2.0, color='#377eb8', label='exact-match rate')
        ax.plot(xs, [c['mean_first_prob'] for c in sub], marker='o', lw=2.0, color='#d95f02', label='mean first_prob')
        ax.set_title(PAIR_LABELS.get(pair, pair))
        ax.set_xticks(xs)
        ax.set_xlabel('ICL examples')
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel('Rate / probability')
    axes[0].legend(loc='upper left')
    fig.suptitle('First-token probability and full-output success can diverge', fontsize=15, fontweight='bold')
    plt.tight_layout()
    fig.savefig(outdir / 'fig_m1_proxy_vs_behavior_by_count.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


def write_summary_json(payload: dict, rows: List[dict], outdir: Path) -> Dict[str, Any]:
    pair_summary = summarize_rows(rows)
    conditions = summarize_conditions(rows)
    failures = collect_failure_examples(rows)
    out = {
        'input_path': str(payload.get('input_path', '')),
        'judge': payload.get('judge', {}),
        'run_config': payload.get('run_config', {}),
        'anchor_summary': payload.get('automatic_validation', {}).get('anchor_summary', {}),
        'pair_summary': pair_summary,
        'condition_summary': conditions,
        'failure_examples': failures,
    }
    (outdir / 'm1_proxy_validation_summary.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    return out


def write_markdown(summary: Dict[str, Any], outdir: Path) -> None:
    pair_lines = []
    for pair, stats in summary['pair_summary'].items():
        pair_lines.append(
            f"### {PAIR_LABELS.get(pair, pair)}\n"
            f"- n = {stats['n']}\n"
            f"- Pearson(first_prob, acceptable) = {stats['pearson_first_prob_vs_acceptable']:.3f}\n"
            f"- Spearman(first_prob, acceptable) = {stats['spearman_first_prob_vs_acceptable']:.3f}\n"
            f"- Pearson(inv_rank, acceptable) = {stats['pearson_inv_rank_vs_acceptable']:.3f}\n"
            f"- AUC(first_prob -> acceptable) = {stats['auc_first_prob_for_acceptable']:.3f}\n"
            f"- AUC(inv_rank -> acceptable) = {stats['auc_inv_rank_for_acceptable']:.3f}\n"
            f"- Mean first_prob if acceptable = {stats['mean_first_prob_if_acceptable']:.4f}\n"
            f"- Mean first_prob if not acceptable = {stats['mean_first_prob_if_not_acceptable']:.4f}\n"
            f"- Mean log10(rank) if acceptable = {stats['mean_log10_rank_if_acceptable']:.3f}\n"
            f"- Mean log10(rank) if not acceptable = {stats['mean_log10_rank_if_not_acceptable']:.3f}\n"
        )

    failure_lines = []
    for key, rows in summary['failure_examples'].items():
        failure_lines.append(f"### {key.replace('_', ' ')}")
        for row in rows:
            failure_lines.append(
                f"- {PAIR_LABELS.get(row['pair'], row['pair'])} ICL={row['n_icl']} `{row['source']}` -> `{row['pred']}` "
                f"(gold `{row['reference']}`, first_prob={row['first_prob']:.4f}, first_rank={row['first_rank']:.0f}, label={row['label']})"
            )

    anchor = summary.get('anchor_summary', {})
    text = f"""# M1 First-Token Proxy Validation

## Status

This report validates whether first-token metrics are a defensible mechanistic proxy for transliteration behavior on the **same items**.

Input artifact:
- judged run: `{summary['input_path']}`

## Judge reliability for this run

- primary label accuracy on anchors = {anchor.get('primary_overall', {}).get('label_accuracy', float('nan')):.3f}
- primary binary accuracy on anchors = {anchor.get('primary_overall', {}).get('binary_accuracy', float('nan')):.3f}
- secondary binary accuracy on anchors = {anchor.get('secondary_overall', {}).get('binary_accuracy', float('nan')):.3f}

This means the acceptable/not-acceptable label used here is supported well enough for proxy validation.

## Main result

First-token metrics do carry real signal, but they are **not equivalent to full transliteration quality**.

The strongest honest formulation after M1 is expected to be:

> first-token probability / rank is a useful mechanistic proxy for early rescue strength, but it must be grounded against full-output behavior and cannot by itself stand in for usable transliteration quality.

## Pair-level summary

{chr(10).join(pair_lines)}

## How to read this

- Higher `first_prob` should correspond to better behavior if the proxy is meaningful.
- Lower `first_rank` (equivalently higher inverse rank) should correspond to better behavior if the proxy is meaningful.
- AUC near 1.0 means the proxy separates acceptable from non-acceptable outputs well; near 0.5 means weak separation.

## Failure examples worth inspecting

{chr(10).join(failure_lines)}

## Decision rule for the paper

After reading the plots and statistics, classify the proxy as one of:

1. **Strong enough for main mechanistic claims**
2. **Good but partial** (usable with caveats and behavioral grounding)
3. **Too weak for central claims**

The expected outcome for this project is likely category (2): useful but partial.
"""
    (outdir / 'M1_FIRST_TOKEN_PROXY_VALIDATION.md').write_text(text, encoding='utf-8')


def main() -> None:
    args = parse_args()
    payload, rows = load_rows(args.input)
    outdir = args.outdir or args.input.resolve().parent / f"{args.input.stem}_analysis"
    outdir.mkdir(parents=True, exist_ok=True)
    payload['input_path'] = str(args.input)
    summary = write_summary_json(payload, rows, outdir)
    plot_boxplots(rows, outdir)
    plot_condition_scatter(summary['condition_summary'], outdir)
    plot_quantile_curves(rows, outdir)
    plot_counts_overlay(summary['condition_summary'], outdir)
    write_markdown(summary, outdir)
    print(f"Wrote M1 analysis to {outdir}")


if __name__ == '__main__':
    main()
