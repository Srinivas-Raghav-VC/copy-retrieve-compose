from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt


MANDATORY_FIGURES = [
    "figure_1_task_pipeline.png",
    "figure_2_baseline_rescue.png",
    "figure_3_sufficiency_necessity.png",
    "figure_4_affine_vs_skip.png",
    "figure_5_mediation_decomposition.png",
    "figure_6_pe_vs_quality.png",
    "figure_7_scaling_curve.png",
    "figure_8_attention_controls.png",
    "figure_9_transcoder_variant_deltas.png",
]


def _save_figure(fig: plt.Figure, path: Path, dpi: int = 300) -> None:
    """
    Save publication-ready outputs.

    Always writes the requested path (typically PNG). If the requested path is
    a PNG, also emits a same-name PDF alongside it for paper submission.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if path.suffix.lower() == ".png":
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")


def _read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _iter_json_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return sorted(root.glob("**/*.json"))


def _draw_empty(path: Path, title: str, note: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis("off")
    ax.set_title(title)
    ax.text(
        0.5,
        0.5,
        note,
        ha="center",
        va="center",
        fontsize=11,
    )
    fig.tight_layout()
    _save_figure(fig, path)
    plt.close(fig)


def _figure_1_task_pipeline(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.axis("off")
    boxes = [
        "data",
        "baseline",
        "layer_sweep_cv",
        "comprehensive",
        "mediation",
        "primary_outcome",
    ]
    x0 = 0.05
    step = 0.15
    for i, name in enumerate(boxes):
        x = x0 + i * step
        ax.text(
            x,
            0.5,
            name,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.35", "fill": False},
            fontsize=9,
        )
        if i < len(boxes) - 1:
            ax.annotate(
                "",
                xy=(x + 0.06, 0.5),
                xytext=(x + step - 0.06, 0.5),
                arrowprops={"arrowstyle": "->"},
            )
    ax.set_title("Figure 1: Task pipeline")
    fig.tight_layout()
    _save_figure(fig, path)
    plt.close(fig)


def _figure_2_baseline_rescue(path: Path, out_dir: Path) -> None:
    baseline_root = out_dir / "artifacts" / "baseline"
    # Collect (zs, icl) per artifact file; then aggregate by pair_id so one bar per pair.
    pair_zs: Dict[str, List[float]] = {}
    pair_icl: Dict[str, List[float]] = {}
    for p in _iter_json_files(baseline_root):
        payload = _read_json(p)
        pair_id = str(payload.get("pair_id", "unknown"))
        stage = payload.get("stage_output", {}) if isinstance(payload, dict) else {}
        stats = stage.get("stats", {}) if isinstance(stage, dict) else {}
        zs = stats.get("top1_acc_zs")
        icl = stats.get("top1_acc_icl")
        if zs is None or icl is None:
            continue
        try:
            zs_f, icl_f = float(zs), float(icl)
            pair_zs.setdefault(pair_id, []).append(zs_f)
            pair_icl.setdefault(pair_id, []).append(icl_f)
        except Exception:
            continue
    if not pair_zs or not pair_icl:
        _draw_empty(
            path,
            "Figure 2: Baseline rescue",
            "No baseline top1_acc_zs/top1_acc_icl values found.",
        )
        return

    labels = sorted(pair_zs.keys())
    zs_vals = [
        sum(pair_zs[p]) / len(pair_zs[p]) if pair_zs[p] else 0.0
        for p in labels
    ]
    icl_vals = [
        sum(pair_icl[p]) / len(pair_icl[p]) if pair_icl.get(p) else 0.0
        for p in labels
    ]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar([i - 0.2 for i in x], zs_vals, width=0.4, label="ZS")
    ax.bar([i + 0.2 for i in x], icl_vals, width=0.4, label="ICL")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Top-1 accuracy")
    ax.set_title("Figure 2: Baseline rescue (ZS vs ICL, mean over seeds/models per pair)")
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, path)
    plt.close(fig)


def _figure_3_sufficiency_necessity(path: Path, out_dir: Path) -> None:
    interventions_root = out_dir / "artifacts" / "interventions"
    pe_vals: List[float] = []
    ae_vals: List[float] = []
    pe_corr_vals: List[float] = []
    pe_attn_vals: List[float] = []
    pe_basis_vals: List[float] = []
    for p in _iter_json_files(interventions_root):
        payload = _read_json(p)
        topk = payload.get("topk_aggregate", {}) if isinstance(payload, dict) else {}
        if not isinstance(topk, dict):
            continue
        for stats in topk.values():
            if not isinstance(stats, dict):
                continue
            for src, dst in [
                ("mean_pe", pe_vals),
                ("mean_ae", ae_vals),
                ("mean_pe_corrupt", pe_corr_vals),
                ("mean_pe_attention", pe_attn_vals),
                ("mean_pe_basis", pe_basis_vals),
            ]:
                v = stats.get(src)
                if v is None:
                    continue
                try:
                    dst.append(float(v))
                except Exception:
                    pass
    if not pe_vals and not ae_vals:
        _draw_empty(
            path,
            "Figure 3: Sufficiency and necessity",
            "No intervention aggregate metrics found.",
        )
        return

    metrics = ["PE", "AE", "PE_corrupt", "PE_attention", "PE_basis"]
    vals = [
        sum(pe_vals) / len(pe_vals) if pe_vals else float("nan"),
        sum(ae_vals) / len(ae_vals) if ae_vals else float("nan"),
        sum(pe_corr_vals) / len(pe_corr_vals) if pe_corr_vals else float("nan"),
        sum(pe_attn_vals) / len(pe_attn_vals) if pe_attn_vals else float("nan"),
        sum(pe_basis_vals) / len(pe_basis_vals) if pe_basis_vals else float("nan"),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(metrics, vals)
    ax.set_ylabel("Mean value")
    ax.set_title("Figure 3: Sufficiency/necessity/control summary")
    fig.tight_layout()
    _save_figure(fig, path)
    plt.close(fig)


def _figure_4_affine_vs_skip(path: Path, out_dir: Path) -> None:
    variants_root = out_dir / "artifacts" / "variants"
    if not variants_root.exists():
        _draw_empty(
            path,
            "Figure 4: Affine vs skip",
            "N/A — run with compare_variants=True to populate variant comparison.",
        )
        return

    affine_pe: List[float] = []
    skipless_pe: List[float] = []
    for var_dir in ["affine_skip", "skipless_or_non_affine"]:
        vroot = variants_root / var_dir
        if not vroot.exists():
            continue
        for p in vroot.glob("**/*.json"):
            payload = _read_json(p)
            if isinstance(payload, dict) and "mean_pe" in payload:
                try:
                    v = float(payload["mean_pe"])
                    if var_dir == "affine_skip":
                        affine_pe.append(v)
                    else:
                        skipless_pe.append(v)
                except (TypeError, ValueError):
                    pass

    if not affine_pe and not skipless_pe:
        _draw_empty(
            path,
            "Figure 4: Affine vs skip",
            "No variant comparison data found.",
        )
        return

    vals = [
        sum(affine_pe) / len(affine_pe) if affine_pe else float("nan"),
        sum(skipless_pe) / len(skipless_pe) if skipless_pe else float("nan"),
    ]
    labels = ["Affine (skip)", "Skipless"]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(labels, vals)
    ax.set_ylabel("Mean PE")
    ax.set_title("Figure 4: Affine vs skip transcoder (PE at best layer)")
    fig.tight_layout()
    _save_figure(fig, path)
    plt.close(fig)


def _figure_5_mediation(path: Path, out_dir: Path) -> None:
    med_root = out_dir / "artifacts" / "mediation"
    nie_vals: List[float] = []
    nde_vals: List[float] = []
    for p in _iter_json_files(med_root):
        payload = _read_json(p)
        shared = payload.get("shared_mediation_result", {}) if isinstance(payload, dict) else {}
        if not isinstance(shared, dict):
            continue
        agg = shared.get("aggregate_stats", {}) or {}
        mean_nie = agg.get("mean_nie")
        if mean_nie is not None:
            try:
                nie_vals.append(float(mean_nie))
            except (TypeError, ValueError):
                pass
        causal = shared.get("causal_effects", [])
        if isinstance(causal, list) and causal:
            for eff in causal:
                if isinstance(eff, dict) and "nde" in eff:
                    try:
                        nde_vals.append(float(eff["nde"]))
                    except (TypeError, ValueError):
                        pass
        if not nie_vals and (mean_nie is None):
            v_nie = shared.get("nie")
            v_nde = shared.get("nde")
            if v_nie is not None:
                try:
                    nie_vals.append(float(v_nie))
                except (TypeError, ValueError):
                    pass
            if v_nde is not None:
                try:
                    nde_vals.append(float(v_nde))
                except (TypeError, ValueError):
                    pass
    if not nie_vals and not nde_vals:
        _draw_empty(
            path,
            "Figure 5: Mediation decomposition",
            "No NIE/NDE values found in mediation artifacts (aggregate_stats.mean_nie or causal_effects[].nde).",
        )
        return

    vals = [
        sum(nie_vals) / len(nie_vals) if nie_vals else float("nan"),
        sum(nde_vals) / len(nde_vals) if nde_vals else float("nan"),
    ]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(["NIE", "NDE"], vals)
    ax.set_ylabel("Mean effect")
    ax.set_title("Figure 5: Mediation decomposition")
    fig.tight_layout()
    _save_figure(fig, path)
    plt.close(fig)


def _figure_6_pe_vs_quality(path: Path, out_dir: Path) -> None:
    # Try baseline artifacts first (CER/exact match when run_quality_eval=True)
    baseline_root = out_dir / "artifacts" / "baseline"
    xs: List[float] = []
    ys: List[float] = []
    for p in _iter_json_files(baseline_root):
        payload = _read_json(p)
        stage = (payload.get("stage_output") or {}) if isinstance(payload, dict) else {}
        stats = (stage.get("stats") or {}) if isinstance(stage, dict) else {}
        cer = stats.get("mean_cer_icl")
        acc = stats.get("top1_acc_icl")
        if cer is not None and acc is not None:
            try:
                xs.append(float(cer))
                ys.append(float(acc))
            except Exception:
                pass
    if not xs:
        # Fallback: interventions (no CER in schema)
        interventions_root = out_dir / "artifacts" / "interventions"
        for p in _iter_json_files(interventions_root):
            payload = _read_json(p)
            topk = payload.get("topk_aggregate", {}) if isinstance(payload, dict) else {}
            for stats in (topk.values() if isinstance(topk, dict) else []):
                if isinstance(stats, dict):
                    pe, cer = stats.get("mean_pe"), stats.get("cer")
                    if pe is not None and cer is not None:
                        try:
                            xs.append(float(cer))
                            ys.append(float(pe))
                        except Exception:
                            pass
    if not xs:
        _draw_empty(
            path,
            "Figure 6: PE vs quality",
            "CER not available. Run with run_quality_eval=True for baseline CER, "
            "or generation eval (CER) not in intervention schema.",
        )
        return

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(xs, ys, s=18)
    ax.set_xlabel("CER" if xs else "CER")
    ax.set_ylabel("Top1 acc / PE")
    ax.set_title("Figure 6: Quality (CER) vs rescue (top1 or PE)")
    fig.tight_layout()
    _save_figure(fig, path)
    plt.close(fig)


def _figure_7_scaling(path: Path, out_dir: Path) -> None:
    stats_path = out_dir / "artifacts" / "stats" / "confirmatory_results.json"
    payload = _read_json(stats_path)
    rows = payload.get("rows", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list) or not rows:
        _draw_empty(path, "Figure 7: Scaling curve", "No confirmatory rows found.")
        return

    by_model: Dict[str, List[float]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        model = str(row.get("model", ""))
        pe = row.get("mean_pe")
        if not model or pe is None:
            continue
        try:
            by_model.setdefault(model, []).append(float(pe))
        except Exception:
            continue
    if not by_model:
        _draw_empty(path, "Figure 7: Scaling curve", "No model-level PE values found.")
        return

    order = ["270m", "1b", "4b", "12b", "27b"]
    labels = [m for m in order if m in by_model] + [m for m in by_model if m not in order]
    means = [sum(by_model[m]) / len(by_model[m]) for m in labels]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(labels, means, marker="o")
    ax.set_xlabel("Model size")
    ax.set_ylabel("Mean PE")
    ax.set_title("Figure 7: Scaling curve")
    fig.tight_layout()
    _save_figure(fig, path)
    plt.close(fig)


def _figure_8_attention_controls(path: Path, out_dir: Path) -> None:
    payload = _read_json(out_dir / "artifacts" / "stats" / "attention_control_summary.json")
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    if not isinstance(summary, dict) or not summary:
        _draw_empty(path, "Figure 8: Attention controls", "No attention control summary found.")
        return
    labels = [
        "PE_real",
        "PE_corrupt",
        "PE_attention",
        "PE_basis",
        "PE_random",
        "PE_shuffle",
        "PE_gauss",
    ]
    vals = [
        summary.get("mean_pe"),
        summary.get("mean_pe_corrupt"),
        summary.get("mean_pe_attention"),
        summary.get("mean_pe_basis"),
        summary.get("mean_pe_random"),
        summary.get("mean_pe_shuffle"),
        summary.get("mean_pe_gauss"),
    ]
    clean = [v for v in vals if isinstance(v, (int, float))]
    if not clean:
        _draw_empty(path, "Figure 8: Attention controls", "Summary has no numeric values.")
        return
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(labels, vals)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_ylabel("Mean effect")
    ax.set_title("Figure 8: Attention/control intervention comparison")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    _save_figure(fig, path)
    plt.close(fig)


def _figure_9_transcoder_variant_deltas(path: Path, out_dir: Path) -> None:
    payload = _read_json(out_dir / "artifacts" / "stats" / "transcoder_variant_summary.json")
    rows = payload.get("pair_model_deltas", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list) or not rows:
        _draw_empty(path, "Figure 9: Transcoder variant deltas", "No transcoder variant summary found.")
        return
    labels: List[str] = []
    vals: List[float] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        d = r.get("delta_affine_minus_skipless")
        if d is None:
            continue
        try:
            vals.append(float(d))
            labels.append(f"{r.get('pair_id', 'pair')}::{r.get('model', 'model')}")
        except Exception:
            continue
    if not vals:
        _draw_empty(path, "Figure 9: Transcoder variant deltas", "No numeric deltas available.")
        return
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(labels, vals)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_ylabel("Delta PE (affine - skipless)")
    ax.set_title("Figure 9: Transcoder variant effect deltas")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    _save_figure(fig, path)
    plt.close(fig)


def generate_mandatory_figures(out_dir: Path) -> List[Path]:
    """
    Generate the nine mandatory figures into artifacts/figures.

    The generator is intentionally resilient: each figure is produced even if
    partial or missing data requires an explicit "no data" panel.
    """
    fig_dir = out_dir / "artifacts" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig_paths = [fig_dir / name for name in MANDATORY_FIGURES]
    _figure_1_task_pipeline(fig_paths[0])
    _figure_2_baseline_rescue(fig_paths[1], out_dir)
    _figure_3_sufficiency_necessity(fig_paths[2], out_dir)
    _figure_4_affine_vs_skip(fig_paths[3], out_dir)
    _figure_5_mediation(fig_paths[4], out_dir)
    _figure_6_pe_vs_quality(fig_paths[5], out_dir)
    _figure_7_scaling(fig_paths[6], out_dir)
    _figure_8_attention_controls(fig_paths[7], out_dir)
    _figure_9_transcoder_variant_deltas(fig_paths[8], out_dir)
    return fig_paths

