#!/usr/bin/env python3
"""Generate all publication-quality figures for the 1B mechanistic analysis."""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not available, generating text summaries only")

D = Path(__file__).parent / "results" / "1b_definitive"
OUT = Path(__file__).parent / "results" / "1b_figures"
OUT.mkdir(parents=True, exist_ok=True)

LANG_COLORS = {"Hindi": "#e41a1c", "Telugu": "#377eb8", "Bengali": "#4daf4a",
               "Kannada": "#984ea3", "Gujarati": "#ff7f00"}
GLOBAL_LAYERS = {5, 11, 17, 23}


def load(name):
    return json.loads((D / name).read_text())


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 1: Logit Lens Trajectories — all 5 languages, helpful vs corrupt
# ═══════════════════════════════════════════════════════════════════════
def fig1_logit_lens():
    if not HAS_MPL: return
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, lang in enumerate(["hindi", "telugu", "bengali", "kannada", "gujarati"]):
        ax = axes[idx]
        d = load("logit_lens_%s.json" % lang)
        n_layers = d["n_layers"]
        ltypes = d["layer_types"]

        for cname, color, ls in [("helpful", "#2166ac", "-"), ("corrupt", "#b2182b", "--"), ("zs", "#999999", ":")]:
            ranks = [d["summary"][cname][li]["rank"]["mean"] for li in range(n_layers)]
            ax.plot(range(n_layers), ranks, color=color, ls=ls, lw=2, label=cname)

        # Mark global layers
        for gl in GLOBAL_LAYERS:
            if gl < n_layers:
                ax.axvline(gl, color="#aaaaaa", alpha=0.3, lw=1)

        ax.set_yscale("log")
        ax.set_ylim(1, 200000)
        ax.set_xlim(0, n_layers - 1)
        ax.set_title("%s (N=%d)" % (lang.title(), d["n_items"]), fontsize=13, fontweight="bold")
        ax.set_xlabel("Layer")
        if idx % 3 == 0:
            ax.set_ylabel("Target Token Rank (log)")
        if idx == 0:
            ax.legend(fontsize=10)

        # Annotate global layers
        for gl in GLOBAL_LAYERS:
            if gl < n_layers:
                ax.text(gl, ax.get_ylim()[1] * 0.7, "G", ha="center", fontsize=8, color="#666666")

    axes[5].axis("off")
    axes[5].text(0.5, 0.5, "Global layers marked with G\n(L5, L11, L17, L23)\n\n"
                 "Lower rank = model more confident\n"
                 "in correct target token",
                 ha="center", va="center", fontsize=12, transform=axes[5].transAxes)

    fig.suptitle("Logit Lens: Target Token Rank Trajectory Across Layers\n"
                 "Gemma 3 1B — Matched-Length Control Comparison", fontsize=15, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT / "fig1_logit_lens_trajectories.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "fig1_logit_lens_trajectories.pdf", bbox_inches="tight")
    plt.close()
    print("Saved fig1_logit_lens_trajectories")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 2: Content-Specificity Ratio (helpful/corrupt) at key layers
# ═══════════════════════════════════════════════════════════════════════
def fig2_content_specificity():
    if not HAS_MPL: return
    fig, ax = plt.subplots(figsize=(10, 6))

    langs = ["Hindi", "Telugu", "Bengali", "Kannada", "Gujarati"]
    key_layers = [11, 17, 23, 25]
    x = np.arange(len(key_layers))
    width = 0.15

    for i, lang in enumerate(langs):
        d = load("logit_lens_%s.json" % lang.lower())
        ratios = []
        for li in key_layers:
            h = d["summary"]["helpful"][li]["rank"]["mean"]
            c = d["summary"]["corrupt"][li]["rank"]["mean"]
            ratios.append(c / h if h > 0 else 1)
        ax.bar(x + i * width, ratios, width, label=lang, color=LANG_COLORS[lang], alpha=0.85)

    ax.axhline(1.0, color="black", ls="--", lw=1, alpha=0.5)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(["L%d\n(%s)" % (l, "GLOBAL" if l in GLOBAL_LAYERS else "local") for l in key_layers])
    ax.set_ylabel("Corrupt/Helpful Rank Ratio\n(>1 = content-specific)")
    ax.set_title("Content-Specificity of ICL Rescue by Language\n"
                 "Ratio > 1 means helpful ICL gives better rank than corrupt ICL at same prompt length",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper left")
    ax.set_ylim(0, max(4, ax.get_ylim()[1]))

    plt.tight_layout()
    fig.savefig(OUT / "fig2_content_specificity.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "fig2_content_specificity.pdf", bbox_inches="tight")
    plt.close()
    print("Saved fig2_content_specificity")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 3: Head Attribution Heatmap — cross-language
# ═══════════════════════════════════════════════════════════════════════
def fig3_head_attribution():
    if not HAS_MPL: return
    langs = ["Hindi", "Telugu", "Bengali", "Kannada", "Gujarati"]
    fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=True)

    for idx, lang in enumerate(langs):
        ax = axes[idx]
        d = load("head_attribution_%s.json" % lang.lower())
        matrix = np.array(d["full_matrix"])
        n_layers, n_heads = matrix.shape

        im = ax.imshow(matrix.T, aspect="auto", cmap="RdBu_r", vmin=-0.5, vmax=1.0,
                        origin="lower", interpolation="nearest")
        ax.set_title(lang, fontsize=12, fontweight="bold")
        ax.set_xlabel("Layer")
        if idx == 0:
            ax.set_ylabel("Head")
        ax.set_yticks(range(n_heads))

        # Mark global layers
        for gl in GLOBAL_LAYERS:
            if gl < n_layers:
                ax.axvline(gl, color="gold", lw=2, alpha=0.7)

        # Mark top-3 heads
        for h in d["top_heads"][:3]:
            ax.plot(h["layer"], h["head"], "k*", markersize=12)

    fig.colorbar(im, ax=axes, shrink=0.8, label="Attribution Effect (fraction of ICL-ZS gap)")
    fig.suptitle("Head Attribution: ICL→ZS Rescue Effect per Head\n"
                 "★ = top-3 heads, gold lines = global attention layers",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT / "fig3_head_attribution_heatmap.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "fig3_head_attribution_heatmap.pdf", bbox_inches="tight")
    plt.close()
    print("Saved fig3_head_attribution_heatmap")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 4: MLP Contribution per Layer (Hindi)
# ═══════════════════════════════════════════════════════════════════════
def fig4_mlp_contribution():
    if not HAS_MPL: return
    d = load("mlp_contribution_hindi.json")
    fig, ax = plt.subplots(figsize=(12, 5))

    layers_data = d["layers"]
    x = [lr["layer"] for lr in layers_data]
    means = [lr["pe"]["mean"] for lr in layers_data]
    ci_lo = [lr["pe"].get("ci_lo") or lr["pe"]["mean"] for lr in layers_data]
    ci_hi = [lr["pe"].get("ci_hi") or lr["pe"]["mean"] for lr in layers_data]
    errs_lo = [m - lo for m, lo in zip(means, ci_lo)]
    errs_hi = [hi - m for m, hi in zip(means, ci_hi)]

    colors = ["#2166ac" if i in GLOBAL_LAYERS else "#666666" for i in x]
    ax.bar(x, means, color=colors, alpha=0.8, edgecolor="white", lw=0.5)
    ax.errorbar(x, means, yerr=[errs_lo, errs_hi], fmt="none", ecolor="black",
                capsize=2, capthick=1, lw=1)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Patching Effect (PE)\n= p(patched) - p(ZS)", fontsize=11)
    ax.set_title("MLP Layer Contribution: ICL→ZS Patching Effect (Hindi, N=30)\n"
                 "Blue = global layers, gray = local layers",
                 fontsize=13, fontweight="bold")

    # Annotate key layers
    for lr in layers_data:
        li = lr["layer"]
        pe = lr["pe"]["mean"]
        if abs(pe) > 0.03:
            ax.annotate("L%d\n%.3f" % (li, pe), (li, pe),
                        textcoords="offset points", xytext=(0, 12 if pe > 0 else -18),
                        ha="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    fig.savefig(OUT / "fig4_mlp_contribution.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "fig4_mlp_contribution.pdf", bbox_inches="tight")
    plt.close()
    print("Saved fig4_mlp_contribution")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 5: Density Degradation + Attention
# ═══════════════════════════════════════════════════════════════════════
def fig5_density():
    if not HAS_MPL: return
    d = load("density_attention_dilution.json")
    gl = d["global_layers"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: target prob vs n_examples
    ns = [r["n_examples"] for r in d["results"]]
    probs = [r["mean_prob"] for r in d["results"]]
    ax1.plot(ns, probs, "ko-", lw=2, markersize=8)
    ax1.axvspan(0, 39, alpha=0.1, color="green", label="Within 512-token window")
    ax1.axvspan(39, 70, alpha=0.1, color="red", label="Exceeds 512-token window")
    ax1.set_xlabel("Number of ICL Examples", fontsize=12)
    ax1.set_ylabel("Target Token Probability", fontsize=12)
    ax1.set_title("Density Degradation\n(all examples are helpful)", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)

    # Right: per-example attention at global layers
    for li_str in ["L05", "L11", "L17", "L23"]:
        attn_per_ex = []
        for r in d["results"]:
            if r["attention_data"]:
                layer_data = r["attention_data"][0]["layers"].get(li_str, [])
                avg_mass = np.mean([h["icl_mass"] for h in layer_data]) if layer_data else 0
                attn_per_ex.append(avg_mass / r["n_examples"])
            else:
                attn_per_ex.append(0)
        ax2.plot(ns, attn_per_ex, "o-", lw=2, markersize=6, label=li_str)

    ax2.set_xlabel("Number of ICL Examples", fontsize=12)
    ax2.set_ylabel("Attention per Example\n(total ICL mass / N)", fontsize=12)
    ax2.set_title("Attention Dilution at Global Layers\n(each example gets less attention)",
                  fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(OUT / "fig5_density_degradation.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "fig5_density_degradation.pdf", bbox_inches="tight")
    plt.close()
    print("Saved fig5_density_degradation")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 6: Architecture diagram showing the mechanism
# ═══════════════════════════════════════════════════════════════════════
def fig6_architecture():
    if not HAS_MPL: return
    fig, ax = plt.subplots(figsize=(8, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 28)
    ax.axis("off")

    # Draw layers
    layer_y = {}
    for li in range(26):
        y = 27 - li
        layer_y[li] = y
        is_global = li in GLOBAL_LAYERS
        color = "#2166ac" if is_global else "#f0f0f0"
        edgecolor = "#2166ac" if is_global else "#cccccc"
        width = 8 if is_global else 6
        x_offset = 1 if is_global else 2

        rect = mpatches.FancyBboxPatch((x_offset, y - 0.35), width, 0.7,
                                        boxstyle="round,pad=0.1",
                                        facecolor=color, edgecolor=edgecolor, lw=2,
                                        alpha=0.3 if not is_global else 0.6)
        ax.add_patch(rect)
        label = "L%02d %s" % (li, "GLOBAL ★" if is_global else "local")
        ax.text(5, y, label, ha="center", va="center",
                fontsize=8 if not is_global else 10,
                fontweight="bold" if is_global else "normal",
                color="black" if is_global else "#888888")

    # Add annotations
    annotations = {
        5: "1st global: 91-99% attn on ICL\nTask format encoding",
        11: "2nd global: L11 H0 universal head\n92% attn on ICL examples",
        14: "L14 H0: strongest single head\nAmplifies L11's signal (local)",
        17: "3rd global: content-specific onset\nLogit lens h/c separation starts",
        23: "4th global: final consolidation\n~60% attn on ICL",
        25: "MLP readout: pe=+0.046\nConverts residual → output token",
    }
    for li, text in annotations.items():
        y = layer_y[li]
        ax.annotate(text, (9, y), (10.5, y), fontsize=7,
                    arrowprops=dict(arrowstyle="->", color="#333333", lw=1),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
                    ha="left", va="center")

    ax.set_title("Gemma 3 1B: ICL Rescue Mechanism\n"
                 "4 global layers (blue) carry the rescue signal",
                 fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    fig.savefig(OUT / "fig6_architecture_mechanism.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "fig6_architecture_mechanism.pdf", bbox_inches="tight")
    plt.close()
    print("Saved fig6_architecture_mechanism")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 7: Cross-language head universality table
# ═══════════════════════════════════════════════════════════════════════
def fig7_cross_language_heads():
    if not HAS_MPL: return
    langs = ["Hindi", "Telugu", "Bengali", "Kannada", "Gujarati"]

    # Collect all heads that appear in top-5 of any language
    head_ranks = defaultdict(dict)
    for lang in langs:
        d = load("head_attribution_%s.json" % lang.lower())
        for h in d["top_heads"][:5]:
            key = "L%02dH%d" % (h["layer"], h["head"])
            head_ranks[key][lang] = h["rank"]

    # Filter to heads appearing in 3+ languages
    shared = {k: v for k, v in head_ranks.items() if len(v) >= 3}
    shared = dict(sorted(shared.items(), key=lambda x: -len(x[1])))

    if not shared:
        print("No shared heads found")
        return

    heads = list(shared.keys())
    fig, ax = plt.subplots(figsize=(10, max(3, len(heads) * 0.8 + 2)))

    cell_text = []
    for head in heads:
        row = []
        for lang in langs:
            r = shared[head].get(lang)
            row.append("#%d" % r if r else "—")
        row.append("%d/5" % len(shared[head]))
        cell_text.append(row)

    table = ax.table(cellText=cell_text, rowLabels=heads,
                     colLabels=langs + ["Count"],
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)

    # Color cells based on rank
    for i, head in enumerate(heads):
        for j, lang in enumerate(langs):
            cell = table[i + 1, j]
            r = shared[head].get(lang)
            if r and r <= 2:
                cell.set_facecolor("#c7e9c0")
            elif r and r <= 5:
                cell.set_facecolor("#e5f5e0")

    # Color header row
    for j in range(len(langs) + 1):
        table[0, j].set_facecolor("#4292c6")
        table[0, j].set_text_props(color="white", fontweight="bold")

    ax.axis("off")
    ax.set_title("Cross-Language Head Universality\n"
                 "Heads appearing in top-5 attribution across 3+ languages\n"
                 "Green = top-2, light green = top-5",
                 fontsize=13, fontweight="bold", pad=20)

    plt.tight_layout()
    fig.savefig(OUT / "fig7_cross_language_heads.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "fig7_cross_language_heads.pdf", bbox_inches="tight")
    plt.close()
    print("Saved fig7_cross_language_heads")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 8: Telugu anomaly — top-5 overlap
# ═══════════════════════════════════════════════════════════════════════
def fig8_telugu_anomaly():
    if not HAS_MPL: return
    d = load("telugu_anomaly.json")

    h_items = [i for i in d["items"] if i["condition"] == "helpful"]
    c_items = [i for i in d["items"] if i["condition"] == "corrupt"]

    # Pair up by idx
    overlaps = []
    h_ranks, c_ranks = [], []
    for hi in h_items:
        ci = next((c for c in c_items if c["idx"] == hi["idx"]), None)
        if ci:
            h_top5 = set(t["id"] for t in hi["top5"])
            c_top5 = set(t["id"] for t in ci["top5"])
            overlaps.append(len(h_top5 & c_top5))
            h_ranks.append(hi["gold_rank"])
            c_ranks.append(ci["gold_rank"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: rank scatter
    ax1.scatter(h_ranks, c_ranks, alpha=0.6, s=40, c="#377eb8")
    ax1.plot([0, max(max(h_ranks), max(c_ranks))],
             [0, max(max(h_ranks), max(c_ranks))], "k--", alpha=0.3)
    ax1.set_xlabel("Helpful ICL — Target Rank")
    ax1.set_ylabel("Corrupt ICL — Target Rank")
    ax1.set_title("Telugu L25: Helpful vs Corrupt Rank\n(points on diagonal = same rank)",
                  fontweight="bold")
    ax1.set_xscale("symlog")
    ax1.set_yscale("symlog")

    # Right: overlap histogram
    ax2.hist(overlaps, bins=range(7), align="left", color="#377eb8", alpha=0.7, edgecolor="white")
    ax2.set_xlabel("Top-5 Token Overlap (out of 5)")
    ax2.set_ylabel("Count")
    ax2.set_title("Telugu: Helpful vs Corrupt Top-5 Token Overlap\n"
                  "High overlap = FORMAT signal dominates over CONTENT",
                  fontweight="bold")
    ax2.set_xticks(range(6))
    mean_overlap = np.mean(overlaps)
    ax2.axvline(mean_overlap, color="red", ls="--", lw=2, label="Mean=%.1f" % mean_overlap)
    ax2.legend()

    plt.tight_layout()
    fig.savefig(OUT / "fig8_telugu_anomaly.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "fig8_telugu_anomaly.pdf", bbox_inches="tight")
    plt.close()
    print("Saved fig8_telugu_anomaly")


# ═══════════════════════════════════════════════════════════════════════
# Run all
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating figures from: %s" % D)
    print("Output to: %s" % OUT)
    print()

    fig1_logit_lens()
    fig2_content_specificity()
    fig3_head_attribution()
    fig4_mlp_contribution()
    fig5_density()
    fig6_architecture()
    fig7_cross_language_heads()
    fig8_telugu_anomaly()

    print("\nAll figures saved to: %s" % OUT)
    for f in sorted(OUT.glob("*.png")):
        print("  %s (%.1f KB)" % (f.name, f.stat().st_size / 1024))
