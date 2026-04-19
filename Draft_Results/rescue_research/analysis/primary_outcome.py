"""
Primary outcome computation for the rescue pipeline.

Decision rule (confirmatory):
- Use pre-registered top-k (DEFAULT_TOPK).
- Require mean PE > 0.
- Require Holm-adjusted p-value for PE vs corrupt control < alpha,
  where Holm correction is applied across tested top-k values.
"""

from __future__ import annotations

import json
from pathlib import Path

from rescue_research.config import (
    RunConfig,
    PRIMARY_OUTCOME_DESCRIPTION,
    PRIMARY_ALPHA,
    DEFAULT_TOPK,
)


def _to_float(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _holm_adjust(p_values: list[float]) -> list[float]:
    """Holm-adjust a list of p-values (returns adjusted p-values)."""
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda t: t[1])
    adjusted_sorted = [1.0] * m
    running_max = 0.0
    for rank, (_, p) in enumerate(indexed):
        factor = m - rank
        val = min(1.0, factor * p)
        running_max = max(running_max, val)
        adjusted_sorted[rank] = running_max
    adjusted = [1.0] * m
    for rank, (orig_idx, _) in enumerate(indexed):
        adjusted[orig_idx] = adjusted_sorted[rank]
    return adjusted


def compute_and_save_primary_outcome(config: RunConfig) -> None:
    config.ensure_out_dir()
    best_path = config.out_dir / "best_layer.txt"
    layer = int(best_path.read_text(encoding="utf-8").strip()) if best_path.exists() else config.layer
    comp_path = config.out_dir / f"comprehensive_{config.model}_L{layer}.json"

    if not comp_path.exists():
        print(f"[rescue_research] No comprehensive results at {comp_path}; skipping primary outcome", flush=True)
        return

    with open(comp_path, encoding="utf-8") as f:
        data = json.load(f)

    warnings: list[str] = []
    topk_agg = data.get("topk_aggregate", {})
    topk_stats: dict[int, dict] = {}

    # Current comprehensive schema.
    if isinstance(topk_agg, dict) and topk_agg:
        for k, v in topk_agg.items():
            try:
                k_int = int(k)
            except (TypeError, ValueError):
                continue
            if isinstance(v, dict):
                topk_stats[k_int] = v

    # Legacy fallback schema for old files.
    if not topk_stats:
        agg = data.get("aggregate", data.get("stats", {}))
        if isinstance(agg, dict):
            topk_stats[DEFAULT_TOPK] = agg
            warnings.append(
                "Comprehensive result used legacy aggregate schema; Holm across top-k not applied."
            )

    summary = []
    raw_ps: list[float] = []
    idx_with_p: list[int] = []

    for topk in sorted(topk_stats.keys()):
        st = topk_stats[topk]
        
        # Primary Metric: NLL improvement
        mean_nll_improvement = _to_float(st.get("mean_nll_improvement_patch", st.get("mean_pe")))
        
        # NLL Effect Size vs Corrupt Control
        effect_block = st.get("effect_size_nll_improvement_vs_corrupt", st.get("effect_size_pe_vs_corrupt", {})) or {}
        cohens_d = _to_float(effect_block.get("cohens_d"))
        interpretation = str(effect_block.get("interpretation", "")).strip()
        p_raw_ttest = _to_float(effect_block.get("paired_ttest_p"))
        p_raw_perm = _to_float(effect_block.get("paired_permutation_p"))
        p_raw_wilcox = _to_float(effect_block.get("wilcoxon_p"))
        
        # Priority: Wilcoxon (non-parametric, best for bounded data) > permutation > t-test
        if p_raw_wilcox == p_raw_wilcox:  # finite/not-NaN
            p_raw = p_raw_wilcox
            p_source = "wilcoxon"
        elif p_raw_perm == p_raw_perm:
            p_raw = p_raw_perm
            p_source = "paired_permutation"
        else:
            p_raw = p_raw_ttest
            p_source = "paired_ttest"
            
        row = {
            "topk": int(topk),
            "mean_nll_improvement_patch": mean_nll_improvement,
            "p_nll_vs_corrupt_raw": p_raw,
            "p_source": p_source,
            "p_nll_vs_corrupt_raw_ttest": p_raw_ttest,
            "p_nll_vs_corrupt_raw_permutation": p_raw_perm,
            "p_nll_vs_corrupt_raw_wilcoxon": p_raw_wilcox,
            "cohens_d_nll_vs_corrupt": cohens_d,
            "cohens_d_interpretation": interpretation,
        }
        summary.append(row)
        if p_raw == p_raw:  # finite/not-NaN check
            idx_with_p.append(len(summary) - 1)
            raw_ps.append(p_raw)

    # Holm-adjust across available top-k p-values.
    if raw_ps:
        adj = _holm_adjust(raw_ps)
        for i, p_adj in zip(idx_with_p, adj):
            summary[i]["p_nll_vs_corrupt_holm"] = p_adj
    else:
        warnings.append("No valid p-values found for NLL improvement vs corrupt comparison.")

    selected_topk = int(DEFAULT_TOPK)
    selected = next((r for r in summary if int(r["topk"]) == selected_topk), None)
    if selected is None:
        primary_passed = False
        nll_gt_0 = False
        nll_gt_corrupt_sig = False
        mean_nll_f = float("nan")
        p_f = float("nan")
        p_holm = float("nan")
        cohens_d = float("nan")
        cohens_d_interpretation = ""
        warnings.append(
            f"Pre-registered topk={DEFAULT_TOPK} missing in comprehensive results."
        )
    else:
        mean_nll_f = _to_float(selected.get("mean_nll_improvement_patch"))
        p_f = _to_float(selected.get("p_nll_vs_corrupt_raw"))
        p_holm = _to_float(selected.get("p_nll_vs_corrupt_holm"))
        p_source = str(selected.get("p_source", "paired_ttest"))
        p_ttest = _to_float(selected.get("p_nll_vs_corrupt_raw_ttest"))
        p_perm = _to_float(selected.get("p_nll_vs_corrupt_raw_permutation"))
        cohens_d = _to_float(selected.get("cohens_d_nll_vs_corrupt"))
        cohens_d_interpretation = str(selected.get("cohens_d_interpretation", "")).strip()
        nll_gt_0 = mean_nll_f > 0
        nll_gt_corrupt_sig = (p_holm < PRIMARY_ALPHA) and nll_gt_0
        primary_passed = nll_gt_0 and nll_gt_corrupt_sig

    outcome = {
        "description": PRIMARY_OUTCOME_DESCRIPTION,
        "alpha": PRIMARY_ALPHA,
        "decision_rule": (
            f"Use pre-registered topk={DEFAULT_TOPK}; require mean_nll_improvement_patch>0 and "
            "Holm-adjusted p(NLL improvement vs corrupt)<alpha across tested topk values "
            "(paired permutation preferred when available; paired t-test fallback)."
        ),
        "selected_topk": selected_topk,
        "mean_nll_improvement_patch": mean_nll_f,
        "p_nll_vs_corrupt_raw": p_f,
        "p_nll_vs_corrupt_raw_source": p_source if selected is not None else "paired_ttest",
        "p_nll_vs_corrupt_raw_ttest": p_ttest if selected is not None else float("nan"),
        "p_nll_vs_corrupt_raw_permutation": p_perm if selected is not None else float("nan"),
        "cohens_d_nll_vs_corrupt": cohens_d,
        "cohens_d_interpretation": cohens_d_interpretation,
        "p_nll_vs_corrupt_holm": p_holm,
        "NLL_improvement_gt_0": nll_gt_0,
        "NLL_improvement_gt_corrupt_significant": nll_gt_corrupt_sig,
        "primary_outcome_passed": primary_passed,
        "topk_summary": summary,
        "warnings": warnings,
        "source_file": str(comp_path),
    }

    out_path = config.out_dir / "primary_outcome.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(outcome, f, indent=2)
    print(f"[rescue_research] Primary outcome: passed={primary_passed} -> {out_path}", flush=True)
