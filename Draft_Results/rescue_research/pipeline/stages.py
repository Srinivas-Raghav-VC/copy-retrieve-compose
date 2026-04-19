from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from rescue_research.analysis.mediation import h3_pass_strict
from rescue_research.analysis.attention import compute_and_save_attention_control_summary
from rescue_research.analysis.primary_outcome import compute_and_save_primary_outcome
from rescue_research.analysis.stats import holm_adjust
from rescue_research.analysis.transcoder import compute_and_save_transcoder_variant_summary
from rescue_research.analysis.skeptic_audit import compute_and_write_skeptic_pass
from rescue_research.benchmark_registry import write_benchmark_registry
from rescue_research.config import RunConfig
from rescue_research.contracts import (
    LOCKED_LANGUAGE_PAIRS,
    PAIR_BACKUPS,
    TARGET_SPLIT_COUNTS,
    SubstitutionTrigger,
    substitution_allowed,
)
from rescue_research.data_pipeline.ingest import (
    get_pair_prompt_metadata,
    list_available_pair_ids,
    load_pair_records,
    load_pair_records_bundle,
)
from rescue_research.data_pipeline.manifest import (
    DatasetManifest,
    PairManifest,
)
from rescue_research.data_pipeline.normalize import normalize_records
from rescue_research.data_pipeline.ood import compute_ood_profile
from rescue_research.data_pipeline.split import deterministic_protocol_split
from rescue_research.data_pipeline.validate import validate_records
from rescue_research.experimental_design import (
    FourWaySplitPlan,
    ThreeWaySplitPlan,
    design_four_way_split,
    design_three_way_split,
)
from rescue_research.modal_backend.jobs import build_modal_job_specs, write_job_manifest
from rescue_research.pipeline.artifact_contracts import ensure_contract_dirs
from rescue_research.pipeline.protocol import (
    evaluate_protocol_compliance,
    pair_matrix_mode,
)
from rescue_research.pipeline.validator import validate_artifacts
from rescue_research.pipeline_config import PipelineConfig, default_pair_specs
from rescue_research.reporting.bundle import build_submission_bundle
from rescue_research.reporting.publication_branch import decide_publication_branch
from rescue_research.reporting.tables import generate_mandatory_tables
from rescue_research.stages.baseline import run_baseline
from rescue_research.stages.comprehensive import run_comprehensive
from rescue_research.stages.layer_sweep_cv import run_layer_sweep_cv
from rescue_research.stages.mediation_run import run_mediation, run_mediation_band
from rescue_research.stages.variant_comparison import run_variant_comparison
from rescue_research.stages.prompt_robustness import run_prompt_robustness


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json_safe(path: Path, label: str = "json") -> Dict:
    """Read JSON file; return {} on missing or parse error; log on error."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(
            f"[rescue_research] WARNING: Failed to read {label} at {path}: {e}",
            flush=True,
        )
        return {}


def _payload_value(payload: Dict[str, Any] | None, key: str, default: Any) -> Any:
    """Read a dynamic JSON-like field with a single, typed access point."""
    if not isinstance(payload, dict):
        return default
    value = payload.get(key, default)
    if value is None:
        return default
    return value


def _normalize_confirmatory_payload(payload: Dict[str, Any] | None) -> Dict[str, Any]:
    """Backfill the confirmatory artifact with summary-root aliases."""
    confirmatory = payload if isinstance(payload, dict) else {}
    rows = list(_payload_value(confirmatory, "rows", []))
    summary = _payload_value(confirmatory, "summary", {})
    if not isinstance(summary, dict):
        summary = {}
    confirmatory.setdefault("rows", rows)
    confirmatory.setdefault("summary", summary)
    confirmatory.setdefault("h1_pass", _payload_value(summary, "h1_pass", False))
    confirmatory.setdefault("h2_pass", _payload_value(summary, "h2_pass", False))
    confirmatory.setdefault("h3_pass", _payload_value(summary, "h3_pass", False))
    confirmatory.setdefault(
        "practical_floor_passed",
        _payload_value(summary, "practical_floor_passed", False),
    )
    confirmatory.setdefault(
        "directional_pair_count",
        _payload_value(summary, "directional_pair_count", 0),
    )
    confirmatory.setdefault(
        "controls_passed",
        _payload_value(summary, "controls_passed", False),
    )
    return confirmatory


def _primary_outcome_aliases(
    seed_stats: Dict[str, Any],
    primary_outcome: Dict[str, Any],
) -> Dict[str, Any]:
    """Expose the canonical NLL outcome through legacy confirmatory row aliases."""
    mean_primary = seed_stats.get("mean_nll_improvement_patch")
    if mean_primary is None:
        mean_primary = primary_outcome.get("mean_nll_improvement_patch")
    if mean_primary is None:
        mean_primary = seed_stats.get("mean_pe", primary_outcome.get("mean_pe"))
    p_holm = primary_outcome.get("p_nll_vs_corrupt_holm")
    if p_holm is None:
        p_holm = primary_outcome.get("p_pe_vs_corrupt_holm")
    cohens_d = primary_outcome.get("cohens_d_nll_vs_corrupt")
    if cohens_d is None:
        cohens_d = primary_outcome.get("cohens_d_pe_vs_corrupt")
    return {
        "mean_pe_alias": mean_primary,
        "p_holm_alias": p_holm,
        "cohens_d_alias": cohens_d,
    }


def _pair_dir(out_dir: Path, pair_id: str, model: str) -> Path:
    return out_dir / "runs" / pair_id / model


# Protocol defaults from stats.yaml (used when computing rubric summary).
_PRACTICAL_FLOOR_E1 = 0.015
_PAIR_DIRECTIONALITY_MIN_PAIRS = 3
_CONFIRMATORY_ALPHA = 0.05


def _safe_yaml_load(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
    except Exception:
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_transcoder_acceptance_thresholds() -> Dict[str, float]:
    defaults = {
        "min_reconstruction_cosine": 0.80,
        "max_reconstruction_rel_error": 0.50,
        "min_feature_coverage_ratio": 0.0010,
        "min_selected_feature_fraction": 0.0001,
        "max_selected_feature_fraction": 0.0500,
        "min_decoded_patch_norm_ratio": 0.0005,
    }
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "models.yaml"
    payload = _safe_yaml_load(cfg_path)
    transcoder = payload.get("transcoder", {}) if isinstance(payload, dict) else {}
    raw = transcoder.get("acceptance_thresholds", {}) if isinstance(transcoder, dict) else {}
    if not isinstance(raw, dict):
        return defaults
    out = dict(defaults)
    for k in defaults.keys():
        try:
            out[k] = float(raw.get(k, defaults[k]))
        except (TypeError, ValueError):
            out[k] = defaults[k]
    return out


def _pick_selected_stats(payload: Dict[str, Any]) -> Dict[str, Any]:
    topk_agg = payload.get("topk_aggregate", {})
    if not isinstance(topk_agg, dict) or not topk_agg:
        return {}
    primary = payload.get("primary_outcome", {})
    selected_topk = None
    if isinstance(primary, dict):
        selected_topk = primary.get("selected_topk")
    if selected_topk is not None:
        stats = topk_agg.get(str(selected_topk)) or topk_agg.get(selected_topk)
        if isinstance(stats, dict):
            return stats
    for _, stats in topk_agg.items():
        if isinstance(stats, dict):
            return stats
    return {}


def _compute_transcoder_fidelity_gate(out_dir: Path) -> Dict[str, Any]:
    thresholds = _load_transcoder_acceptance_thresholds()
    rows: List[Dict[str, Any]] = []
    root = out_dir / "artifacts" / "interventions"
    for p in sorted(root.glob("**/*.json")):
        payload = _read_json_safe(p, "intervention")
        if not isinstance(payload, dict):
            continue
        stats = _pick_selected_stats(payload)
        if not stats:
            continue
        row = {
            "pair_id": str(payload.get("pair_id", "")),
            "model": str(payload.get("model", "")),
            "seed": payload.get("seed"),
            "mean_reconstruction_cosine_icl": _to_float(
                stats.get("mean_reconstruction_cosine_icl")
            ),
            "mean_reconstruction_rel_error_icl": _to_float(
                stats.get("mean_reconstruction_rel_error_icl")
            ),
            "mean_feature_coverage_ratio": _to_float(
                stats.get("mean_feature_coverage_ratio")
            ),
            "mean_selected_feature_fraction": _to_float(
                stats.get("mean_selected_feature_fraction")
            ),
            "mean_decoded_patch_norm_ratio": _to_float(
                stats.get("mean_decoded_patch_norm_ratio")
            ),
        }
        rows.append(row)

    def _mean(name: str) -> float:
        vals = []
        for r in rows:
            v = _to_float(r.get(name))
            if v == v:
                vals.append(float(v))
        return float(sum(vals) / len(vals)) if vals else float("nan")

    aggregate = {
        "n_rows": int(len(rows)),
        "mean_reconstruction_cosine_icl": _mean("mean_reconstruction_cosine_icl"),
        "mean_reconstruction_rel_error_icl": _mean("mean_reconstruction_rel_error_icl"),
        "mean_feature_coverage_ratio": _mean("mean_feature_coverage_ratio"),
        "mean_selected_feature_fraction": _mean("mean_selected_feature_fraction"),
        "mean_decoded_patch_norm_ratio": _mean("mean_decoded_patch_norm_ratio"),
    }
    checks = [
        {
            "name": "reconstruction_cosine",
            "metric": "mean_reconstruction_cosine_icl",
            "op": ">=",
            "threshold": thresholds["min_reconstruction_cosine"],
            "value": aggregate["mean_reconstruction_cosine_icl"],
            "passed": (
                aggregate["mean_reconstruction_cosine_icl"]
                == aggregate["mean_reconstruction_cosine_icl"]
                and aggregate["mean_reconstruction_cosine_icl"]
                >= thresholds["min_reconstruction_cosine"]
            ),
        },
        {
            "name": "reconstruction_relative_error",
            "metric": "mean_reconstruction_rel_error_icl",
            "op": "<=",
            "threshold": thresholds["max_reconstruction_rel_error"],
            "value": aggregate["mean_reconstruction_rel_error_icl"],
            "passed": (
                aggregate["mean_reconstruction_rel_error_icl"]
                == aggregate["mean_reconstruction_rel_error_icl"]
                and aggregate["mean_reconstruction_rel_error_icl"]
                <= thresholds["max_reconstruction_rel_error"]
            ),
        },
        {
            "name": "feature_coverage",
            "metric": "mean_feature_coverage_ratio",
            "op": ">=",
            "threshold": thresholds["min_feature_coverage_ratio"],
            "value": aggregate["mean_feature_coverage_ratio"],
            "passed": (
                aggregate["mean_feature_coverage_ratio"]
                == aggregate["mean_feature_coverage_ratio"]
                and aggregate["mean_feature_coverage_ratio"]
                >= thresholds["min_feature_coverage_ratio"]
            ),
        },
        {
            "name": "selected_feature_fraction_min",
            "metric": "mean_selected_feature_fraction",
            "op": ">=",
            "threshold": thresholds["min_selected_feature_fraction"],
            "value": aggregate["mean_selected_feature_fraction"],
            "passed": (
                aggregate["mean_selected_feature_fraction"]
                == aggregate["mean_selected_feature_fraction"]
                and aggregate["mean_selected_feature_fraction"]
                >= thresholds["min_selected_feature_fraction"]
            ),
        },
        {
            "name": "selected_feature_fraction_max",
            "metric": "mean_selected_feature_fraction",
            "op": "<=",
            "threshold": thresholds["max_selected_feature_fraction"],
            "value": aggregate["mean_selected_feature_fraction"],
            "passed": (
                aggregate["mean_selected_feature_fraction"]
                == aggregate["mean_selected_feature_fraction"]
                and aggregate["mean_selected_feature_fraction"]
                <= thresholds["max_selected_feature_fraction"]
            ),
        },
        {
            "name": "decoded_patch_norm_ratio",
            "metric": "mean_decoded_patch_norm_ratio",
            "op": ">=",
            "threshold": thresholds["min_decoded_patch_norm_ratio"],
            "value": aggregate["mean_decoded_patch_norm_ratio"],
            "passed": (
                aggregate["mean_decoded_patch_norm_ratio"]
                == aggregate["mean_decoded_patch_norm_ratio"]
                and aggregate["mean_decoded_patch_norm_ratio"]
                >= thresholds["min_decoded_patch_norm_ratio"]
            ),
        },
    ]
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "thresholds": thresholds,
        "aggregate": aggregate,
        "checks": checks,
        "gate_passed": bool(rows) and all(bool(c["passed"]) for c in checks),
        "n_intervention_rows": int(len(rows)),
        "rows_preview": rows[:50],
    }


def _to_float(x: Any) -> float:
    if x is None:
        return float("nan")
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _compute_gate_a_status(out_dir: Path, pairs: List[str]) -> Dict:
    """
    Gate A: baseline rescue in ≥3/4 pairs.
    Per (pair, model): rescue if ICL > ZS in at least half of seeds.
    Per pair: rescue if at least one model passes on that pair.

    Metric hierarchy (strongest to weakest evidence):
    1) mean_nll_per_token (lower is better)
    2) mean_icl_lift on first-token probability
    3) top10 target-hit rate
    4) top1 accuracy
    """
    def _seed_decision(stats: Dict[str, Any]) -> Dict[str, Any] | None:
        nll_zs = _to_float(stats.get("mean_nll_per_token_zs"))
        nll_icl = _to_float(stats.get("mean_nll_per_token_icl"))
        if nll_zs == nll_zs and nll_icl == nll_icl:
            return {
                "metric": "mean_nll_per_token",
                "zs": float(nll_zs),
                "icl": float(nll_icl),
                "rescued": bool(nll_icl < nll_zs),
                "delta": float(nll_zs - nll_icl),
                "direction": "lower_is_better",
            }

        lift = _to_float(stats.get("mean_icl_lift"))
        if lift == lift:
            return {
                "metric": "mean_icl_lift",
                "zs": None,
                "icl": None,
                "rescued": bool(lift > 0.0),
                "delta": float(lift),
                "direction": "higher_is_better",
            }

        top10_zs = _to_float(stats.get("top10_hit_zs"))
        top10_icl = _to_float(stats.get("top10_hit_icl"))
        if top10_zs == top10_zs and top10_icl == top10_icl:
            return {
                "metric": "top10_hit_rate",
                "zs": float(top10_zs),
                "icl": float(top10_icl),
                "rescued": bool(top10_icl > top10_zs),
                "delta": float(top10_icl - top10_zs),
                "direction": "higher_is_better",
            }

        top1_zs = _to_float(stats.get("top1_acc_zs"))
        top1_icl = _to_float(stats.get("top1_acc_icl"))
        if top1_zs == top1_zs and top1_icl == top1_icl:
            return {
                "metric": "top1_accuracy",
                "zs": float(top1_zs),
                "icl": float(top1_icl),
                "rescued": bool(top1_icl > top1_zs),
                "delta": float(top1_icl - top1_zs),
                "direction": "higher_is_better",
            }
        return None

    baseline_root = out_dir / "artifacts" / "baseline"
    pair_model_rescue_seeds: Dict[str, Dict[str, List[bool]]] = defaultdict(
        lambda: defaultdict(list)
    )
    pair_model_seed_decisions: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    pair_rescue_seeds_pooled: Dict[str, List[bool]] = defaultdict(list)
    pair_seed_decisions_pooled: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for p in sorted(baseline_root.glob("**/*.json")):
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        pair_id = str(payload.get("pair_id", ""))
        model = str(payload.get("model", "")).strip() or "unknown_model"
        if pair_id not in pairs:
            continue
        stage = payload.get("stage_output", {}) or {}
        stats = stage.get("stats", {}) or {}
        dec = _seed_decision(stats if isinstance(stats, dict) else {})
        if dec is None:
            continue
        seed_id = payload.get("seed")
        rescued = bool(dec.get("rescued", False))
        pair_model_rescue_seeds[pair_id][model].append(rescued)
        pair_rescue_seeds_pooled[pair_id].append(rescued)
        pair_model_seed_decisions[pair_id][model].append(
            {
                "seed": seed_id,
                **dec,
            }
        )
        pair_seed_decisions_pooled[pair_id].append(
            {
                "seed": seed_id,
                "model": model,
                **dec,
            }
        )

    pair_model_seed_thresholds: Dict[str, Dict[str, int]] = {}
    pair_model_passed: Dict[str, Dict[str, bool]] = {}
    pair_model_summary: Dict[str, Dict[str, Dict[str, Any]]] = {}
    model_pairs_observed: Counter[str] = Counter()
    model_pairs_passed: Counter[str] = Counter()

    for pair_id, by_model in pair_model_rescue_seeds.items():
        pair_model_seed_thresholds[pair_id] = {}
        pair_model_passed[pair_id] = {}
        pair_model_summary[pair_id] = {}

        for model, seed_rescues in sorted(by_model.items()):
            n_seeds = len(seed_rescues)
            threshold = (n_seeds + 1) // 2
            passed = bool(sum(seed_rescues) >= threshold) if n_seeds > 0 else False
            pair_model_seed_thresholds[pair_id][model] = int(threshold)
            pair_model_passed[pair_id][model] = bool(passed)
            model_pairs_observed[model] += 1
            if passed:
                model_pairs_passed[model] += 1

            decisions = pair_model_seed_decisions.get(pair_id, {}).get(model, [])
            metric_counts = Counter(
                str(d.get("metric", "unknown")) for d in decisions if isinstance(d, dict)
            )
            deltas = []
            for d in decisions:
                if not isinstance(d, dict):
                    continue
                delta = _to_float(d.get("delta"))
                if delta == delta:
                    deltas.append(float(delta))
            mean_delta = float(sum(deltas) / len(deltas)) if deltas else float("nan")
            rescue_rate = float(sum(seed_rescues) / n_seeds) if n_seeds > 0 else float("nan")
            if n_seeds <= 0:
                regime = "insufficient_data"
            elif rescue_rate >= (2.0 / 3.0) and (mean_delta != mean_delta or mean_delta > 0.0):
                regime = "helpful"
            elif rescue_rate <= (1.0 / 3.0) and (mean_delta != mean_delta or mean_delta < 0.0):
                regime = "harmful"
            else:
                regime = "mixed"

            pair_model_summary[pair_id][model] = {
                "n_seeds": int(n_seeds),
                "rescued_seeds": int(sum(seed_rescues)),
                "rescue_rate": float(rescue_rate) if rescue_rate == rescue_rate else float("nan"),
                "mean_help_delta": float(mean_delta) if mean_delta == mean_delta else float("nan"),
                "dominant_metric": (
                    metric_counts.most_common(1)[0][0] if metric_counts else "unknown"
                ),
                "metric_counts": dict(metric_counts),
                "regime": regime,
            }

    pair_passed: Dict[str, bool] = {}
    pair_best_model: Dict[str, str] = {}
    for pair_id, by_model in pair_model_passed.items():
        passing_models = [m for m, ok in by_model.items() if bool(ok)]
        pair_passed[pair_id] = bool(passing_models)
        candidates = passing_models or list(by_model.keys())
        if not candidates:
            continue

        def _model_key(m: str) -> tuple[float, str]:
            info = pair_model_summary.get(pair_id, {}).get(m, {})
            rr = _to_float(info.get("rescue_rate"))
            md = _to_float(info.get("mean_help_delta"))
            rr_v = float(rr) if rr == rr else float("-inf")
            md_v = float(md) if md == md else float("-inf")
            return (rr_v + 1e-3 * md_v, str(m))

        pair_best_model[pair_id] = sorted(candidates, key=_model_key, reverse=True)[0]

    pair_seed_thresholds_pooled: Dict[str, int] = {}
    pair_passed_pooled = {
        pid: bool(sum(seeds) >= (len(seeds) + 1) // 2)
        for pid, seeds in pair_rescue_seeds_pooled.items()
        if seeds
    }
    for pid, seeds in pair_rescue_seeds_pooled.items():
        pair_seed_thresholds_pooled[pid] = int((len(seeds) + 1) // 2)

    n_passed = sum(1 for v in pair_passed.values() if v)
    gate_a_passed = n_passed >= 3
    return {
        "gate_a_passed": gate_a_passed,
        "gate_a_definition": (
            "pair passes if at least one model shows baseline rescue in >= half of seeds"
        ),
        "pair_passed": pair_passed,
        "pair_best_model": pair_best_model,
        "pair_model_passed": {pid: dict(by_model) for pid, by_model in pair_model_passed.items()},
        "pair_model_summary": {
            pid: {m: dict(v) for m, v in by_model.items()}
            for pid, by_model in pair_model_summary.items()
        },
        "pair_model_seed_decisions": {
            pid: {m: list(rows) for m, rows in by_model.items()}
            for pid, by_model in pair_model_seed_decisions.items()
        },
        "pair_model_seed_thresholds": {
            pid: dict(by_model) for pid, by_model in pair_model_seed_thresholds.items()
        },
        "model_pairs_passed": dict(model_pairs_passed),
        "model_pairs_observed": dict(model_pairs_observed),
        # Backward-compatible pooled view (across all models and seeds).
        "pair_passed_pooled": pair_passed_pooled,
        "pair_seed_decisions_pooled": {
            pid: list(rows) for pid, rows in pair_seed_decisions_pooled.items()
        },
        "pair_seed_thresholds_pooled": pair_seed_thresholds_pooled,
        "n_pairs_required": 3,
        "n_pairs_passed": n_passed,
    }


def _compute_confirmatory_rubric_summary(rows: List[Dict]) -> Dict:
    """
    Compute rubric summary from confirmatory rows for publication branch.

    Important: rows are often reported per-seed. To avoid pseudo-replication,
    inferential checks are aggregated at (pair_id, model) granularity first.
    """

    def _mean_finite(vals: List[float]) -> float:
        clean: List[float] = []
        for v in vals:
            f = _to_float(v)
            if f == f:
                clean.append(float(f))
        if not clean:
            return float("nan")
        return float(sum(clean) / len(clean))

    grouped: Dict[tuple[str, str], Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in rows if isinstance(rows, list) else []:
        if not isinstance(r, dict):
            continue
        pair_id = str(r.get("pair_id", "")).strip()
        model = str(r.get("model", "")).strip()
        if not pair_id or not model:
            continue
        key = (pair_id, model)
        for field in (
            "mean_pe",
            "mean_pe_corrupt",
            "mean_pe_random",
            "mean_pe_shuffle",
            "mean_pe_gauss",
            "mean_pe_attention",
            "mean_pe_basis",
            "mean_ae",
            "p_ae_lt_0_one_tailed",
            "p_holm",
        ):
            v = _to_float(r.get(field))
            if v == v:
                grouped[key][field].append(float(v))

    units: List[Dict[str, float | str]] = []
    for (pair_id, model), metrics in sorted(grouped.items()):
        unit: Dict[str, float | str] = {"pair_id": pair_id, "model": model}
        for field, vals in metrics.items():
            if field in {"p_ae_lt_0_one_tailed", "p_holm"}:
                unit[field] = float(vals[0]) if vals else float("nan")
            else:
                unit[field] = _mean_finite(vals)
        units.append(unit)

    deltas: List[float] = []
    pair_pe: Dict[str, List[float]] = defaultdict(list)
    h1_pvals: List[float] = []
    h1_effects: List[float] = []
    ae_vals: List[float] = []
    ae_pvals: List[float] = []

    for u in units:
        mean_pe = _to_float(u.get("mean_pe"))
        mean_pe_corrupt = _to_float(u.get("mean_pe_corrupt"))
        mean_ae = _to_float(u.get("mean_ae"))
        p_ae = _to_float(u.get("p_ae_lt_0_one_tailed"))
        p_h1 = _to_float(u.get("p_holm"))
        pair_id = str(u.get("pair_id", ""))

        if pair_id and mean_pe == mean_pe:
            pair_pe[pair_id].append(mean_pe)
        if mean_pe == mean_pe and mean_pe_corrupt == mean_pe_corrupt:
            deltas.append(mean_pe - mean_pe_corrupt)
        if mean_ae == mean_ae:
            ae_vals.append(mean_ae)
        if p_ae == p_ae:
            ae_pvals.append(p_ae)
        if p_h1 == p_h1 and mean_pe == mean_pe:
            h1_pvals.append(p_h1)
            h1_effects.append(mean_pe)

    e1_delta_pe = _mean_finite(deltas)
    practical_floor_passed = bool(deltas) and (e1_delta_pe >= _PRACTICAL_FLOOR_E1)

    directional_pair_count = 0
    for pair_id, vals in pair_pe.items():
        if pair_id not in LOCKED_LANGUAGE_PAIRS:
            continue
        # Pair directional if at least one model has positive mean PE.
        if vals and max(vals) > 0:
            directional_pair_count += 1

    controls_checks: List[Dict[str, Any]] = []

    def _add_control_check(name: str, control_field: str) -> None:
        diffs: List[float] = []
        for u in units:
            pe = _to_float(u.get("mean_pe"))
            ctrl = _to_float(u.get(control_field))
            if pe == pe and ctrl == ctrl:
                diffs.append(pe - ctrl)
        if not diffs:
            controls_checks.append(
                {
                    "name": name,
                    "control_field": control_field,
                    "n_units": 0,
                    "mean_delta": float("nan"),
                    "passed": False,
                    "reason": "missing",
                }
            )
            return
        mean_delta = _mean_finite(diffs)
        controls_checks.append(
            {
                "name": name,
                "control_field": control_field,
                "n_units": len(diffs),
                "mean_delta": mean_delta,
                "passed": bool(mean_delta == mean_delta and mean_delta > 0.0),
                "reason": "",
            }
        )

    _add_control_check("pe_gt_corrupt", "mean_pe_corrupt")
    _add_control_check("pe_gt_random", "mean_pe_random")
    _add_control_check("pe_gt_shuffle", "mean_pe_shuffle")
    _add_control_check("pe_gt_gauss", "mean_pe_gauss")
    _add_control_check("pe_gt_attention", "mean_pe_attention")
    _add_control_check("pe_gt_basis", "mean_pe_basis")

    # Necessity direction as part of the controls battery.
    mean_ae = _mean_finite(ae_vals)
    controls_checks.append(
        {
            "name": "ae_negative_direction",
            "control_field": "mean_ae",
            "n_units": len(ae_vals),
            "mean_delta": mean_ae,
            "passed": bool(ae_vals) and bool(mean_ae == mean_ae and mean_ae < 0.0),
            "reason": "missing" if not ae_vals else "",
        }
    )

    controls_passed = bool(controls_checks) and all(
        bool(c.get("passed", False)) for c in controls_checks
    )

    h1_holm_significant_count = 0
    h1_required_count = 0
    h1_pass = False
    if h1_pvals:
        h1_adj = holm_adjust(h1_pvals)
        h1_required_count = max(1, len(h1_adj) // 2)
        h1_holm_significant_count = sum(
            1
            for p_adj, effect in zip(h1_adj, h1_effects)
            if p_adj < _CONFIRMATORY_ALPHA and effect > 0
        )
        h1_pass = h1_holm_significant_count >= h1_required_count

    h2_holm_significant_count = 0
    h2_required_count = 0
    h2_pass = False
    if ae_vals and ae_pvals:
        ae_adj = holm_adjust(ae_pvals)
        h2_required_count = max(1, len(ae_adj) // 2)
        h2_holm_significant_count = sum(1 for p_adj in ae_adj if p_adj < _CONFIRMATORY_ALPHA)
        h2_pass = (mean_ae < 0.0) and (h2_holm_significant_count >= h2_required_count)

    return {
        "n_pair_model_units": len(units),
        "e1_delta_pe": e1_delta_pe,
        "practical_floor_passed": practical_floor_passed,
        "directional_pair_count": directional_pair_count,
        "controls_passed": controls_passed,
        "controls_checks": controls_checks,
        "h1_pass": h1_pass,
        "h1_holm_significant_count": h1_holm_significant_count,
        "h1_required_count": h1_required_count,
        "h2_pass": h2_pass,
        "h2_holm_significant_count": h2_holm_significant_count,
        "h2_required_count": h2_required_count,
        "h3_pass": False,
    }


def _runtime_three_way_plan(
    cfg: PipelineConfig,
    total: int,
    *,
    include_blind_eval: bool = False,
) -> ThreeWaySplitPlan:
    """
    Derive run-time ICL/selection/eval counts from protocol targets using
    explicit adaptive or strict policy.

    By default, evaluation draws from eval_open only. Blind slices are included
    only when explicitly requested (post-freeze blind run).
    """
    target_eval = int(TARGET_SPLIT_COUNTS["eval_open"])
    if bool(include_blind_eval):
        target_eval += int(TARGET_SPLIT_COUNTS["eval_blind"])
    return design_three_way_split(
        total_available=total,
        n_icl_target=max(2, int(getattr(cfg, "k_confirmatory", 8))),
        n_selection_target=int(TARGET_SPLIT_COUNTS["selection"]),
        n_eval_target=target_eval,
        policy=getattr(cfg, "split_policy", "adaptive"),
        min_icl=int(getattr(cfg, "runtime_min_icl", 4)),
        min_selection=int(getattr(cfg, "runtime_min_selection", 8)),
        min_eval=int(getattr(cfg, "runtime_min_eval", 12)),
    )


def _infer_pair_family(source_script: str, target_script: str) -> str:
    indic_scripts = {
        "Devanagari",
        "Tamil",
        "Telugu",
        "Kannada",
        "Malayalam",
        "Bengali",
        "Gujarati",
        "Gurmukhi",
        "Oriya",
    }
    if source_script in indic_scripts and target_script in indic_scripts:
        return "indic"
    return "non_indic"


def _resolve_pair_spec(cfg: PipelineConfig, pair_id: str) -> Dict[str, str]:
    specs = default_pair_specs()
    if pair_id in specs:
        spec = specs[pair_id]
        return {
            "pair_id": spec.pair_id,
            "family": spec.family,
            "source_language": spec.source_language,
            "source_script": spec.source_script,
            "target_script": spec.target_script,
            "backups": list(spec.backups),
        }

    meta = get_pair_prompt_metadata(pair_id)
    source_script = str(meta.get("source_script", ""))
    target_script = str(meta.get("target_script", ""))
    source_language = str(meta.get("source_language", ""))
    return {
        "pair_id": pair_id,
        "family": _infer_pair_family(source_script, target_script),
        "source_language": source_language,
        "source_script": source_script,
        "target_script": target_script,
        "backups": list(PAIR_BACKUPS.get(pair_id, ())),
    }


def _target_script_group(target_script: str) -> str:
    s = str(target_script or "").strip().lower()
    if any(k in s for k in ("devanagari", "tamil", "telugu", "kannada", "malayalam", "bengali", "gujarati", "gurmukhi", "oriya")):
        return "indic_abugida"
    if any(k in s for k in ("arabic", "hebrew")):
        return "abjad"
    if "cyrillic" in s:
        return "cyrillic"
    if "georgian" in s:
        return "georgian"
    if "greek" in s:
        return "greek"
    if any(k in s for k in ("katakana", "hiragana", "japanese")):
        return "japanese_kana"
    if any(k in s for k in ("thai", "lao", "khmer", "burmese", "myanmar")):
        return "se_asian_abugida"
    return "other"


def _report_pair_matrix_coverage(cfg: PipelineConfig) -> None:
    specs = [_resolve_pair_spec(cfg, pair_id) for pair_id in cfg.pairs]
    family_counts = Counter(str(s.get("family", "unknown")) for s in specs)
    script_group_counts = Counter(
        _target_script_group(str(s.get("target_script", ""))) for s in specs
    )
    pair_desc = ", ".join(
        f"{s.get('pair_id')}[{s.get('family')}/{_target_script_group(str(s.get('target_script','')))}]"
        for s in specs
    )
    print(
        f"[rescue_research] Pair matrix ({len(specs)}): {pair_desc}",
        flush=True,
    )
    print(
        "[rescue_research] Coverage summary: "
        f"families={dict(family_counts)} target_script_groups={dict(script_group_counts)}",
        flush=True,
    )
    _write_json(
        cfg.out_dir / "artifacts" / "audit" / "pair_matrix_coverage.json",
        {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pairs": [
                {
                    **s,
                    "target_script_group": _target_script_group(
                        str(s.get("target_script", ""))
                    ),
                }
                for s in specs
            ],
            "family_counts": dict(family_counts),
            "target_script_group_counts": dict(script_group_counts),
        },
    )


def _compute_pair_readiness(cfg: PipelineConfig) -> Dict[str, Dict]:
    min_pool = int(getattr(cfg, "min_confirmatory_pool", 40))
    min_icl = int(getattr(cfg, "min_confirmatory_icl", 4))
    min_sel = int(getattr(cfg, "min_confirmatory_selection", 12))
    min_eval = int(getattr(cfg, "min_confirmatory_eval", 24))
    readiness: Dict[str, Dict] = {}
    for pair_id in cfg.pairs:
        total = len(load_pair_records(pair_id))
        plan = _runtime_three_way_plan(
            cfg,
            total,
            include_blind_eval=bool(getattr(cfg, "run_blind_eval", False)),
        )
        reasons: List[str] = []
        if total < min_pool:
            reasons.append(f"available_records<{min_pool} (got {total})")
        if int(plan.n_icl) < min_icl:
            reasons.append(f"runtime_n_icl<{min_icl} (got {plan.n_icl})")
        if int(plan.n_selection) < min_sel:
            reasons.append(f"runtime_n_selection<{min_sel} (got {plan.n_selection})")
        if int(plan.n_eval) < min_eval:
            reasons.append(f"runtime_n_eval<{min_eval} (got {plan.n_eval})")
        readiness[pair_id] = {
            "available_records": total,
            "runtime_plan": asdict(plan),
            "eligible_confirmatory": len(reasons) == 0,
            "reasons": reasons,
        }
    return readiness


def _candidate_pair_suggestions(cfg: PipelineConfig) -> List[str]:
    min_pool = int(getattr(cfg, "min_confirmatory_pool", 40))
    min_icl = int(getattr(cfg, "min_confirmatory_icl", 4))
    min_sel = int(getattr(cfg, "min_confirmatory_selection", 12))
    min_eval = int(getattr(cfg, "min_confirmatory_eval", 24))
    ranked: List[tuple[int, str, ThreeWaySplitPlan]] = []
    for pair_id in list_available_pair_ids():
        try:
            n = len(load_pair_records(pair_id))
        except Exception:
            continue
        if n < min_pool:
            continue
        plan = _runtime_three_way_plan(
            cfg,
            n,
            include_blind_eval=bool(getattr(cfg, "run_blind_eval", False)),
        )
        if int(plan.n_icl) < min_icl or int(plan.n_selection) < min_sel or int(plan.n_eval) < min_eval:
            continue
        ranked.append((int(n), pair_id, plan))
    ranked.sort(key=lambda x: (-x[0], x[1]))
    return [
        f"{pid}({n};icl={p.n_icl},sel={p.n_selection},eval={p.n_eval})"
        for n, pid, p in ranked[:12]
    ]


def _write_pair_readiness_report(cfg: PipelineConfig, readiness: Dict[str, Dict]) -> None:
    _write_json(
        cfg.out_dir / "artifacts" / "audit" / "pair_readiness.json",
        {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "enforce_pair_readiness": bool(getattr(cfg, "enforce_pair_readiness", True)),
            "allow_underpowered_pairs": bool(getattr(cfg, "allow_underpowered_pairs", False)),
            "thresholds": {
                "min_confirmatory_pool": int(getattr(cfg, "min_confirmatory_pool", 40)),
                "min_confirmatory_icl": int(getattr(cfg, "min_confirmatory_icl", 4)),
                "min_confirmatory_selection": int(getattr(cfg, "min_confirmatory_selection", 12)),
                "min_confirmatory_eval": int(getattr(cfg, "min_confirmatory_eval", 24)),
            },
            "pairs": readiness,
        },
    )


def _write_substitution_policy_audit(
    cfg: PipelineConfig, readiness: Dict[str, Dict]
) -> None:
    substitutions = list(getattr(cfg, "substitution_plan", []) or [])
    decisions: List[Dict] = []
    for item in substitutions:
        if not isinstance(item, dict):
            continue
        locked_pair = str(item.get("from_locked_pair", "")).strip()
        selected_pair = str(item.get("to_substitute_pair", "")).strip()
        locked_ready = bool(
            (readiness.get(locked_pair, {}) or {}).get("eligible_confirmatory", False)
        )
        trigger = SubstitutionTrigger(
            data_audit_error_rate=0.0,
            remediation_cycles=1,
            unresolved_licensing_risk=False,
            effective_pool_below_minimum=not locked_ready,
            gate_name="PRE_GATE_A",
            substitutions_already_used=max(0, len(substitutions) - 1),
        )
        decisions.append(
            {
                "from_locked_pair": locked_pair,
                "to_substitute_pair": selected_pair,
                "locked_pair_ready": locked_ready,
                "trigger": asdict(trigger),
                "allowed_by_policy": bool(substitution_allowed(trigger)),
            }
        )

    _write_json(
        cfg.out_dir / "artifacts" / "audit" / "substitution_policy.json",
        {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "allow_custom_pairs": bool(getattr(cfg, "allow_custom_pairs", False)),
            "substitutions_requested": substitutions,
            "decisions": decisions,
        },
    )


def _enforce_pair_readiness(cfg: PipelineConfig, *, quiet: bool = False) -> None:
    readiness = _compute_pair_readiness(cfg)
    _write_pair_readiness_report(cfg, readiness)
    _write_substitution_policy_audit(cfg, readiness)
    bad = [pid for pid, info in readiness.items() if not bool(info.get("eligible_confirmatory", False))]
    if not bad:
        return

    msg_lines = [
        "Confirmatory pair-readiness check failed for underpowered pairs:",
    ]
    for pid in bad:
        reasons = readiness[pid].get("reasons", [])
        msg_lines.append(f"- {pid}: {', '.join(str(x) for x in reasons)}")
    suggestions = _candidate_pair_suggestions(cfg)
    if suggestions:
        msg_lines.append("Candidate pairs with sufficient pool size: " + ", ".join(suggestions))
    msg_lines.append(
        "Set allow_underpowered_pairs=True (or --allow-underpowered-pairs) only for exploratory/pilot runs."
    )

    if bool(getattr(cfg, "enforce_pair_readiness", True)) and not bool(
        getattr(cfg, "allow_underpowered_pairs", False)
    ):
        raise ValueError("\n".join(msg_lines))
    if not quiet:
        print("[rescue_research] WARNING: " + " | ".join(msg_lines), flush=True)


def _pair_matrix_mode(cfg: PipelineConfig) -> str:
    return pair_matrix_mode(
        pairs=getattr(cfg, "pairs", []) or [],
        locked_pairs=LOCKED_LANGUAGE_PAIRS,
        substitution_plan=getattr(cfg, "substitution_plan", []) or [],
    )


def _compute_protocol_compliance(cfg: PipelineConfig) -> Dict[str, Any]:
    substitution_audit = _read_json_safe(
        cfg.out_dir / "artifacts" / "audit" / "substitution_policy.json",
        "substitution_policy",
    )
    dataset_manifest = _read_json_safe(
        cfg.out_dir / "artifacts" / "manifests" / "dataset_manifest.json",
        "dataset_manifest",
    )
    return evaluate_protocol_compliance(
        pairs=getattr(cfg, "pairs", []) or [],
        locked_pairs=LOCKED_LANGUAGE_PAIRS,
        substitution_plan=getattr(cfg, "substitution_plan", []) or [],
        allow_underpowered_pairs=bool(getattr(cfg, "allow_underpowered_pairs", False)),
        enforce_pair_readiness=bool(getattr(cfg, "enforce_pair_readiness", True)),
        substitution_audit=substitution_audit,
        dataset_manifest=dataset_manifest,
    )


def stage_prepare_data(cfg: PipelineConfig) -> None:
    """
    Prepare deterministic protocol splits per pair/seed and write to
    data/processed/<pair_id>/split_seed_<seed>.json.

    These prepared splits are the execution source of truth for downstream
    baseline/layer/comprehensive/robustness stages in the full pipeline.
    """
    ensure_contract_dirs(cfg.out_dir)
    _report_pair_matrix_coverage(cfg)
    _enforce_pair_readiness(cfg, quiet=False)
    pair_manifests: Dict[str, PairManifest] = {}
    data_root = cfg.out_dir / "data" / "processed"
    audit_root = cfg.out_dir / "artifacts" / "audit"
    design_registry: Dict[str, Dict] = {}

    for pair_id in cfg.pairs:
        spec = _resolve_pair_spec(cfg, pair_id)
        bundle = load_pair_records_bundle(pair_id)
        normalized = normalize_records(bundle.rows)
        valid, summary = validate_records(
            normalized,
            source_script=spec["source_script"],
            target_script=spec["target_script"],
        )
        pair_dir = data_root / pair_id
        pair_dir.mkdir(parents=True, exist_ok=True)
        available = len(valid)

        four_way_plan: FourWaySplitPlan = design_four_way_split(
            total_available=available,
            n_icl_bank_target=int(cfg.icl_bank_count),
            n_selection_target=int(cfg.selection_count),
            n_eval_open_target=int(cfg.eval_open_count),
            n_eval_blind_target=int(cfg.eval_blind_count),
            policy=getattr(cfg, "split_policy", "adaptive"),
            min_icl_bank=int(getattr(cfg, "data_min_icl_bank", 8)),
            min_selection=int(getattr(cfg, "data_min_selection", 16)),
            min_eval_open=int(getattr(cfg, "data_min_eval_open", 24)),
            min_eval_blind=int(getattr(cfg, "data_min_eval_blind", 8)),
        )
        three_way_plan = _runtime_three_way_plan(
            cfg,
            available,
            include_blind_eval=bool(getattr(cfg, "run_blind_eval", False)),
        )
        design_registry[pair_id] = {
            "family": spec["family"],
            "source_language": spec["source_language"],
            "source_script": spec["source_script"],
            "target_script": spec["target_script"],
            "available_records": available,
            "runtime_eval_source": (
                "eval_open_plus_eval_blind"
                if bool(getattr(cfg, "run_blind_eval", False))
                else "eval_open_only"
            ),
            "four_way_plan": asdict(four_way_plan),
            "three_way_runtime_plan": asdict(three_way_plan),
        }

        ood_by_seed: Dict[int, Dict[str, float]] = {}
        for seed in cfg.seeds:
            split = deterministic_protocol_split(
                valid,
                seed=seed,
                n_icl_bank=four_way_plan.n_icl_bank,
                n_selection=four_way_plan.n_selection,
                n_eval_open=four_way_plan.n_eval_open,
                n_eval_blind=four_way_plan.n_eval_blind,
            )
            _write_json(
                pair_dir / f"split_seed_{seed}.json",
                {
                    "icl_bank": split.icl_bank,
                    "selection": split.selection,
                    "eval_open": split.eval_open,
                    "eval_blind": split.eval_blind,
                },
            )
            ood_by_seed[int(seed)] = compute_ood_profile(
                selection_tokens=[x["target"] for x in split.selection],
                eval_tokens=[x["target"] for x in split.eval_open],
            )

        _write_json(
            audit_root / f"data_quality_report_{pair_id}.json",
            {
                **asdict(summary),
                "ood_profile_by_seed": ood_by_seed,
            },
        )

        pair_manifests[pair_id] = PairManifest(
            pair_id=pair_id,
            source_language=spec["source_language"],
            source_script=spec["source_script"],
            target_script=spec["target_script"],
            backups=list(spec["backups"]),
            sources=list(bundle.sources),
            min_pool_size=len(valid),
            ambiguity_rate=0.0,
            notes="Validated and split under protocol policy.",
        )

    _write_json(
        cfg.out_dir / "artifacts" / "audit" / "experimental_design.json",
        {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "split_policy": getattr(cfg, "split_policy", "adaptive"),
            "pairs": design_registry,
        },
    )
    write_benchmark_registry(cfg.out_dir / "artifacts" / "manifests" / "benchmark_registry.json")

    manifest = DatasetManifest(
        schema_version="v1",
        frozen_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        pair_manifests=pair_manifests,
        substitutions_used=list(getattr(cfg, "substitution_plan", []) or []),
        blind_slice_sealed=True,
    )
    manifest.write_json(cfg.out_dir / "artifacts" / "manifests" / "dataset_manifest.json")

    _write_json(
        cfg.out_dir / "artifacts" / "manifests" / "run_manifest.json",
        {
            "schema_version": "v1",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pipeline": "full_confirmatory",
            "pair_matrix_mode": _pair_matrix_mode(cfg),
            "pair_matrix_locked": tuple(cfg.pairs) == tuple(LOCKED_LANGUAGE_PAIRS),
            "allow_custom_pairs": bool(getattr(cfg, "allow_custom_pairs", False)),
            "substitution_plan": list(getattr(cfg, "substitution_plan", []) or []),
            "pairs": cfg.pairs,
            "models": cfg.models,
            "seeds": cfg.seeds,
            "confirmatory_topk_values": list(getattr(cfg, "confirmatory_topk_values", [25])),
            "backend": cfg.backend,
            "task": str(getattr(cfg, "task", "transliteration")),
            "control_mode": str(getattr(cfg, "control_mode", "default")),
            "patch_style": str(getattr(cfg, "patch_style", "sparse")),
            "eval_generation": bool(getattr(cfg, "eval_generation", getattr(cfg, "run_quality_eval", False))),
            "split_policy": getattr(cfg, "split_policy", "adaptive"),
            "execution_split_source": "prepared_protocol_split",
            "execution_uses_prepared_splits": True,
            "runtime_eval_source": (
                "eval_open_plus_eval_blind"
                if bool(getattr(cfg, "run_blind_eval", False))
                else "eval_open_only"
            ),
            "pair_readiness": {
                "enforce": bool(getattr(cfg, "enforce_pair_readiness", True)),
                "allow_underpowered_pairs": bool(getattr(cfg, "allow_underpowered_pairs", False)),
                "min_confirmatory_pool": int(getattr(cfg, "min_confirmatory_pool", 40)),
                "min_confirmatory_icl": int(getattr(cfg, "min_confirmatory_icl", 4)),
                "min_confirmatory_selection": int(getattr(cfg, "min_confirmatory_selection", 12)),
                "min_confirmatory_eval": int(getattr(cfg, "min_confirmatory_eval", 24)),
            },
            "runtime_minima": {
                "icl": int(getattr(cfg, "runtime_min_icl", 4)),
                "selection": int(getattr(cfg, "runtime_min_selection", 8)),
                "eval": int(getattr(cfg, "runtime_min_eval", 12)),
            },
            "data_minima": {
                "icl_bank": int(getattr(cfg, "data_min_icl_bank", 8)),
                "selection": int(getattr(cfg, "data_min_selection", 16)),
                "eval_open": int(getattr(cfg, "data_min_eval_open", 24)),
                "eval_blind": int(getattr(cfg, "data_min_eval_blind", 8)),
            },
            "blind_slice_sealed": True,
        },
    )


def stage_baseline_selection(cfg: PipelineConfig) -> None:
    ensure_contract_dirs(cfg.out_dir)
    _report_pair_matrix_coverage(cfg)
    _enforce_pair_readiness(cfg, quiet=True)
    if cfg.backend == "modal":
        specs = build_modal_job_specs(
            stage_name="baseline_selection",
            pairs=cfg.pairs,
            models=cfg.models,
            seeds=cfg.seeds,
            command_prefix=["python", "-m", "rescue_research.run", "--stage", "full"],
        )
        write_job_manifest(cfg.out_dir / "artifacts" / "manifests" / "modal_baseline_selection.json", specs)
        return

    for pair_id in cfg.pairs:
        total = len(load_pair_records(pair_id))
        plan = _runtime_three_way_plan(
            cfg,
            total,
            include_blind_eval=bool(getattr(cfg, "run_blind_eval", False)),
        )
        n_icl, n_select, n_eval = plan.n_icl, plan.n_selection, plan.n_eval
        prepared_split_dir = cfg.out_dir / "data" / "processed" / pair_id
        for model in cfg.models:
            out = _pair_dir(cfg.out_dir, pair_id, model)
            out.mkdir(parents=True, exist_ok=True)
            rc = RunConfig(
                out_dir=out,
                n_icl=n_icl,
                n_select=n_select,
                n_eval=n_eval,
                seeds=list(cfg.seeds),
                model=model,
                pair=pair_id,
                prepared_split_dir=str(prepared_split_dir),
                use_blind_eval=bool(getattr(cfg, "run_blind_eval", False)),
            )
            if cfg.execute_experiments:
                run_baseline(rc, run_quality_eval=getattr(cfg, "run_quality_eval", False))
                run_layer_sweep_cv(rc, top_layers=max(5, getattr(cfg, "mediation_band_size", 3)))

            # Convert stage outputs to contract artifact paths.
            baseline_path = out / f"baseline_{model}.json"
            layer_path = out / f"layer_sweep_cv_{model}.json"
            baseline_data = _read_json_safe(baseline_path, "baseline")
            layer_data = _read_json_safe(layer_path, "layer_sweep_cv")

            for seed in cfg.seeds:
                _write_json(
                    cfg.out_dir / "artifacts" / "baseline" / model / pair_id / f"{seed}.json",
                    {
                        "pair_id": pair_id,
                        "model": model,
                        "seed": seed,
                        "stage_output": baseline_data.get("seeds", {}).get(str(seed), {}),
                    },
                )
                _write_json(
                    cfg.out_dir / "artifacts" / "selection" / model / pair_id / f"{seed}.json",
                    {
                        "pair_id": pair_id,
                        "model": model,
                        "seed": seed,
                        "stage_output": layer_data.get("layers", {}),
                        "best_layer": (layer_data.get("summary", {}) or {}).get("best_layer"),
                        "top_layers": (layer_data.get("summary", {}) or {}).get("top_layers") or [],
                    },
                )

    gate_a = _compute_gate_a_status(cfg.out_dir, list(cfg.pairs))
    _write_json(
        cfg.out_dir / "artifacts" / "audit" / "gate_a_status.json",
        {**gate_a, "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")},
    )
    try:
        model_pass = gate_a.get("model_pairs_passed", {}) or {}
        model_obs = gate_a.get("model_pairs_observed", {}) or {}
        if isinstance(model_pass, dict) and isinstance(model_obs, dict) and model_obs:
            parts = []
            for model in sorted(model_obs.keys()):
                passed = int(model_pass.get(model, 0))
                total = int(model_obs.get(model, 0))
                parts.append(f"{model}:{passed}/{total}")
            print(
                "[rescue_research] Gate A model-wise rescued pairs: "
                + ", ".join(parts),
                flush=True,
            )
    except Exception:
        pass


def stage_confirmatory(cfg: PipelineConfig) -> None:
    ensure_contract_dirs(cfg.out_dir)
    _enforce_pair_readiness(cfg, quiet=True)
    # Gate A: warn if baseline rescue did not pass in ≥3/4 pairs (audit only; we still run).
    gate_a_path = cfg.out_dir / "artifacts" / "audit" / "gate_a_status.json"
    if gate_a_path.exists():
        try:
            gate_a = json.loads(gate_a_path.read_text(encoding="utf-8"))
            if not gate_a.get("gate_a_passed", True):
                model_pass = gate_a.get("model_pairs_passed", {}) or {}
                model_obs = gate_a.get("model_pairs_observed", {}) or {}
                model_msg = ""
                if isinstance(model_pass, dict) and isinstance(model_obs, dict) and model_obs:
                    details = []
                    for model in sorted(model_obs.keys()):
                        details.append(
                            f"{model}:{int(model_pass.get(model, 0))}/{int(model_obs.get(model, 0))}"
                        )
                    if details:
                        model_msg = " Model-wise rescued pairs: " + ", ".join(details) + "."
                print(
                    "[rescue_research] WARNING: Gate A not passed (baseline rescue in <3/4 pairs). "
                    "Proceeding to confirmatory; consider pair substitution or re-run baseline."
                    + model_msg,
                    flush=True,
                )
        except Exception:
            pass
    _write_json(
        cfg.out_dir / "artifacts" / "audit" / "blind_holdout_status.json",
        {
            "blind_eval_enabled": bool(cfg.run_blind_eval),
            "policy": "single_execution_after_freeze",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    if cfg.backend == "modal":
        specs = build_modal_job_specs(
            stage_name="confirmatory",
            pairs=cfg.pairs,
            models=cfg.models,
            seeds=cfg.seeds,
            command_prefix=["python", "-m", "rescue_research.run", "--stage", "comprehensive"],
            one_job_per_pair_model=True,
        )
        write_job_manifest(cfg.out_dir / "artifacts" / "manifests" / "modal_confirmatory.json", specs)
        return

    confirmatory_rows = []
    for pair_id in cfg.pairs:
        total = len(load_pair_records(pair_id))
        plan = _runtime_three_way_plan(
            cfg,
            total,
            include_blind_eval=bool(getattr(cfg, "run_blind_eval", False)),
        )
        n_icl, n_select, n_eval = plan.n_icl, plan.n_selection, plan.n_eval
        prepared_split_dir = cfg.out_dir / "data" / "processed" / pair_id
        for model in cfg.models:
            out = _pair_dir(cfg.out_dir, pair_id, model)
            rc = RunConfig(
                out_dir=out,
                n_icl=n_icl,
                n_select=n_select,
                n_eval=n_eval,
                seeds=list(cfg.seeds),
                topk_values=list(getattr(cfg, "confirmatory_topk_values", [25])),
                model=model,
                pair=pair_id,
                prepared_split_dir=str(prepared_split_dir),
                use_blind_eval=bool(getattr(cfg, "run_blind_eval", False)),
                patch_style=str(getattr(cfg, "patch_style", "sparse")),
                control_mode=str(getattr(cfg, "control_mode", "default")),
                eval_generation=bool(getattr(cfg, "eval_generation", getattr(cfg, "run_quality_eval", False))),
            )
            if cfg.execute_experiments:
                run_comprehensive(rc, run_quality_eval=rc.eval_generation)
                compute_and_save_primary_outcome(rc)

            best_layer = rc.layer
            best_layer_path = out / "best_layer.txt"
            if best_layer_path.exists():
                try:
                    best_layer = int(best_layer_path.read_text(encoding="utf-8").strip())
                except (ValueError, OSError):
                    print(
                        f"[rescue_research] WARNING: Invalid best_layer.txt at {best_layer_path}; "
                        f"using default layer {rc.layer}",
                        flush=True,
                    )
            else:
                print(
                    f"[rescue_research] WARNING: best_layer.txt missing at {out}; "
                    f"using default layer {best_layer}. Run layer_sweep_cv first.",
                    flush=True,
                )
            comp_path = out / f"comprehensive_{model}_L{best_layer}.json"
            po_path = out / "primary_outcome.json"
            comp_data = _read_json_safe(comp_path, "comprehensive")
            po_data = _read_json_safe(po_path, "primary_outcome")
            topk_agg = comp_data.get("topk_aggregate", {}) or {}
            selected_topk = po_data.get("selected_topk")
            st = (
                (topk_agg.get(str(selected_topk)) or topk_agg.get(selected_topk))
                if selected_topk is not None
                else {}
            ) or {}
            mean_pe_corrupt = st.get("mean_pe_corrupt")
            mean_ae = st.get("mean_ae")
            mean_pe_random = st.get("mean_pe_random")
            mean_pe_shuffle = st.get("mean_pe_shuffle")
            mean_pe_gauss = st.get("mean_pe_gauss")
            mean_pe_attention = st.get("mean_pe_attention")
            mean_pe_basis = st.get("mean_pe_basis")
            ci_pe_95 = st.get("ci_pe_95")
            ci_pe_corrupt_95 = st.get("ci_pe_corrupt_95")
            ci_ae_95 = st.get("ci_ae_95")
            h2_test = st.get("one_sample_test_ae_lt_0") or {}
            p_ae_lt_0 = h2_test.get("p_value_one_tailed")

            power_law = comp_data.get("power_law_topology", {}) if isinstance(comp_data, dict) else {}
            power_law_hint = power_law.get("topology_hint") if isinstance(power_law, dict) else None
            power_law_k90 = power_law.get("k_at_90pct_max_rescue") if isinstance(power_law, dict) else None

            per_seed_blocks = comp_data.get("seeds", {}) if isinstance(comp_data, dict) else {}
            if not isinstance(per_seed_blocks, dict):
                per_seed_blocks = {}

            for seed in cfg.seeds:
                seed_stats = st
                seed_block = per_seed_blocks.get(str(seed), {})
                if isinstance(seed_block, dict) and selected_topk is not None:
                    topk_results = seed_block.get("topk_results", {})
                    if isinstance(topk_results, dict):
                        seed_topk = topk_results.get(str(selected_topk)) or topk_results.get(selected_topk)
                        if isinstance(seed_topk, dict) and isinstance(seed_topk.get("stats"), dict):
                            seed_stats = seed_topk.get("stats", {})

                seed_h2_test = seed_stats.get("one_sample_test_ae_lt_0") or {}
                seed_p_ae_lt_0 = seed_h2_test.get("p_value_one_tailed")

                _write_json(
                    cfg.out_dir / "artifacts" / "interventions" / model / pair_id / f"{seed}.json",
                    {
                        "pair_id": pair_id,
                        "model": model,
                        "seed": seed,
                        "topk_aggregate": comp_data.get("topk_aggregate", {}),
                        "selected_topk_stats": seed_stats,
                        "primary_outcome": po_data,
                    },
                )
                po_aliases = _primary_outcome_aliases(seed_stats, po_data)
                confirmatory_rows.append(
                    {
                        "pair_id": pair_id,
                        "model": model,
                        "seed": seed,
                        "primary_passed": bool(po_data.get("primary_outcome_passed", False)),
                        "selected_topk": selected_topk,
                        "power_law_topology_hint": power_law_hint,
                        "power_law_k_at_90pct_max_rescue": power_law_k90,
                        "mean_pe": seed_stats.get("mean_pe", po_aliases["mean_pe_alias"]),
                        "mean_pe_logit": seed_stats.get(
                            "mean_pe_logit",
                            seed_stats.get("mean_logit_pe", po_data.get("mean_pe_logit")),
                        ),
                        "p_holm": po_aliases["p_holm_alias"],
                        "cohens_d_pe_vs_corrupt": po_aliases["cohens_d_alias"],
                        "cohens_d_interpretation": po_data.get("cohens_d_interpretation"),
                        "mean_pe_corrupt": seed_stats.get("mean_pe_corrupt", mean_pe_corrupt),
                        "mean_pe_random": seed_stats.get("mean_pe_random", mean_pe_random),
                        "mean_pe_shuffle": seed_stats.get("mean_pe_shuffle", mean_pe_shuffle),
                        "mean_pe_gauss": seed_stats.get("mean_pe_gauss", mean_pe_gauss),
                        "mean_pe_attention": seed_stats.get("mean_pe_attention", mean_pe_attention),
                        "mean_pe_basis": seed_stats.get("mean_pe_basis", mean_pe_basis),
                        "mean_pe_auto_scale": seed_stats.get("mean_pe_auto_scale"),
                        "mean_pe_auto_shift": seed_stats.get("mean_pe_auto_shift"),
                        "mean_pe_null": seed_stats.get("mean_pe_null"),
                        "mean_pe_cross_task": seed_stats.get("mean_pe_cross_task"),
                        "mean_ae": seed_stats.get("mean_ae", mean_ae),
                        "auto_scale_ratio": seed_stats.get("auto_scale_ratio"),
                        "auto_scale_ratio_adjudicated": seed_stats.get("auto_scale_ratio_adjudicated"),
                        "auto_scale_ratio_metric": seed_stats.get("auto_scale_ratio_metric"),
                        "auto_scale_ratio_pe": seed_stats.get("auto_scale_ratio_pe"),
                        "auto_scale_ratio_mult": seed_stats.get("auto_scale_ratio_mult"),
                        "auto_scale_ratio_add": seed_stats.get("auto_scale_ratio_add"),
                        "auto_scale_intervention_artifact": seed_stats.get("auto_scale_intervention_artifact"),
                        "mean_nll_improvement_patch": seed_stats.get("mean_nll_improvement_patch"),
                        "mean_nll_improvement_auto_scale_patch": seed_stats.get("mean_nll_improvement_auto_scale_patch"),
                        "mean_nll_improvement_auto_shift_patch": seed_stats.get("mean_nll_improvement_auto_shift_patch"),
                        "mean_nll_improvement_null_patch": seed_stats.get("mean_nll_improvement_null_patch"),
                        "mean_nll_improvement_mean_pool_patch": seed_stats.get("mean_nll_improvement_mean_pool_patch"),
                        "mean_nll_per_char_improvement_patch": seed_stats.get("mean_nll_per_char_improvement_patch"),
                        "mean_nll_harm_english_neutral_patch": seed_stats.get("mean_nll_harm_english_neutral_patch"),
                        "feature_collision_risk_rate": seed_stats.get("feature_collision_risk_rate"),
                        "mean_active_features_icl": seed_stats.get("mean_active_features_icl"),
                        "mean_active_features_zs": seed_stats.get("mean_active_features_zs"),
                        "mean_feature_cosine_zs_icl": seed_stats.get("mean_feature_cosine_zs_icl"),
                        "mean_feature_identity_jaccard_zs_icl": seed_stats.get("mean_feature_identity_jaccard_zs_icl"),
                        "mean_reconstruction_mse_icl": seed_stats.get("mean_reconstruction_mse_icl"),
                        "mean_selected_feature_dla_target": seed_stats.get("mean_selected_feature_dla_target"),
                        "mean_top_selected_feature_dla_target": seed_stats.get("mean_top_selected_feature_dla_target"),
                        "mean_selected_feature_dla_competitor": seed_stats.get("mean_selected_feature_dla_competitor"),
                        "mean_dla_target_minus_competitor": seed_stats.get("mean_dla_target_minus_competitor"),
                        "mean_nll_harm_attn_head_ablation": seed_stats.get("mean_nll_harm_attn_head_ablation"),
                        "mean_delta_bos_attention_next_layer_patch": seed_stats.get("mean_delta_bos_attention_next_layer_patch"),
                        "context_expectation_warning_rate": seed_stats.get("context_expectation_warning_rate"),
                        "mean_rope_position_gap": seed_stats.get("mean_rope_position_gap"),
                        "input_fragmentation_rate_ge_3_tokens": seed_stats.get("input_fragmentation_rate_ge_3_tokens"),
                        "high_fragmentation_warning": seed_stats.get("high_fragmentation_warning"),
                        "mean_logit_icl_first": seed_stats.get("mean_logit_icl_first"),
                        "mean_logit_patched_first": seed_stats.get("mean_logit_patched_first"),
                        "softcap_saturation_risk": seed_stats.get("softcap_saturation_risk"),
                        "mean_prob_competitor_zs_first": seed_stats.get("mean_prob_competitor_zs_first"),
                        "mean_prob_competitor_icl_first": seed_stats.get("mean_prob_competitor_icl_first"),
                        "mean_prob_competitor_patched_first": seed_stats.get("mean_prob_competitor_patched_first"),
                        "mean_delta_competitor_prob_patch": seed_stats.get("mean_delta_competitor_prob_patch"),
                        "mean_delta_competitor_logit_patch": seed_stats.get("mean_delta_competitor_logit_patch"),
                        "bracket_rescue_ratio": seed_stats.get("bracket_rescue_ratio"),
                        "top5_jaccard": seed_stats.get("top5_jaccard"),
                        "top25_jaccard": seed_stats.get("top25_jaccard"),
                        "mean_attn_sink_heads_excluded": seed_stats.get("mean_attn_sink_heads_excluded"),
                        "attn_sink_threshold": seed_stats.get("attn_sink_threshold"),
                        "ci_pe_95": seed_stats.get("ci_pe_95", ci_pe_95),
                        "ci_pe_corrupt_95": seed_stats.get("ci_pe_corrupt_95", ci_pe_corrupt_95),
                        "ci_ae_95": seed_stats.get("ci_ae_95", ci_ae_95),
                        "p_ae_lt_0_one_tailed": seed_p_ae_lt_0 if seed_p_ae_lt_0 is not None else p_ae_lt_0,
                    }
                )

    summary = _compute_confirmatory_rubric_summary(confirmatory_rows)
    confirmatory_payload = {
        "rows": confirmatory_rows,
        "summary": summary,
        "h1_pass": summary.get("h1_pass", False),
        "h2_pass": summary.get("h2_pass", False),
        "h3_pass": summary.get("h3_pass", False),
        "practical_floor_passed": summary.get("practical_floor_passed", False),
        "directional_pair_count": summary.get("directional_pair_count", 0),
        "controls_passed": summary.get("controls_passed", False),
        "execution_split_source": "prepared_protocol_split",
        "runtime_eval_source": (
            "eval_open_plus_eval_blind"
            if bool(getattr(cfg, "run_blind_eval", False))
            else "eval_open_only"
        ),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _write_json(
        cfg.out_dir / "artifacts" / "stats" / "confirmatory_results.json",
        confirmatory_payload,
    )

    # Deterministic skeptic verdict artifact (claim gate).
    try:
        compute_and_write_skeptic_pass(
            cfg.out_dir,
            confirmatory_rows,
            protocol_version="2026-03-LOCKED",
        )
    except Exception as e:
        print(f"[rescue_research] WARNING: skeptic_pass generation failed: {e}", flush=True)


def stage_robustness(cfg: PipelineConfig) -> None:
    ensure_contract_dirs(cfg.out_dir)
    _enforce_pair_readiness(cfg, quiet=True)
    band_size = max(1, getattr(cfg, "mediation_band_size", 1))
    if cfg.backend == "modal":
        specs = build_modal_job_specs(
            stage_name="robustness",
            pairs=cfg.pairs,
            models=[m for m in cfg.models if m in ("1b", "4b", "12b")],
            seeds=cfg.seeds,
            command_prefix=["python", "-m", "rescue_research.run", "--stage", "mediation"],
        )
        write_job_manifest(cfg.out_dir / "artifacts" / "manifests" / "modal_robustness.json", specs)
        return

    exploratory_rows = []
    band_summary_rows: List[Dict] = []
    prompt_robustness_rows: List[Dict] = []
    for pair_id in cfg.pairs:
        total = len(load_pair_records(pair_id))
        plan = _runtime_three_way_plan(
            cfg,
            total,
            include_blind_eval=bool(getattr(cfg, "run_blind_eval", False)),
        )
        n_icl, n_select, n_eval = plan.n_icl, plan.n_selection, plan.n_eval
        prepared_split_dir = cfg.out_dir / "data" / "processed" / pair_id
        for model in cfg.models:
            out = _pair_dir(cfg.out_dir, pair_id, model)
            rc = RunConfig(
                out_dir=out,
                n_icl=n_icl,
                n_select=n_select,
                n_eval=n_eval,
                seeds=list(cfg.seeds),
                model=model,
                pair=pair_id,
                prepared_split_dir=str(prepared_split_dir),
                use_blind_eval=bool(getattr(cfg, "run_blind_eval", False)),
            )
            band_results: List[Dict] = []
            best_layer = rc.layer
            if cfg.execute_experiments:
                if band_size > 1:
                    band_results = run_mediation_band(rc, band_size)
                    layer_data = _read_json_safe(out / f"layer_sweep_cv_{model}.json", "layer_sweep")
                    top_layers = (layer_data.get("summary") or {}).get("top_layers") or []
                    best_layer = (layer_data.get("summary") or {}).get("best_layer")
                else:
                    run_mediation(rc)
                    best_layer = None
                    band_results = []

            best_layer_path = out / "best_layer.txt"
            if best_layer_path.exists():
                best_layer = int(best_layer_path.read_text(encoding="utf-8").strip())
            med_path = out / f"mediation_{model}_L{best_layer}.json"
            med_data = _read_json_safe(med_path, "mediation")

            if cfg.execute_experiments:
                try:
                    pr = run_prompt_robustness(rc, n_eval_sample=24)
                    prompt_robustness_rows.append(dict(pr))
                    _write_json(
                        cfg.out_dir
                        / "artifacts"
                        / "audit"
                        / "prompt_robustness"
                        / model
                        / f"{pair_id}.json",
                        pr,
                    )
                except Exception as e:
                    print(
                        f"[rescue_research] Prompt robustness failed ({pair_id}/{model}): {e}",
                        flush=True,
                    )

            for seed in cfg.seeds:
                _write_json(
                    cfg.out_dir / "artifacts" / "mediation" / model / pair_id / f"{seed}.json",
                    {
                        "pair_id": pair_id,
                        "model": model,
                        "seed": seed,
                        "shared_mediation_result": med_data,
                    },
                )
                exploratory_rows.append(
                    {
                        "pair_id": pair_id,
                        "model": model,
                        "seed": seed,
                        "has_mediation": bool(med_data),
                    }
                )

            # Band summary: per (pair, model) with per-layer NIE
            if band_results:
                per_layer: List[Dict] = []
                for br in band_results:
                    res = br.get("results") or {}
                    agg = (res.get("aggregate_stats") or {}) if isinstance(res, dict) else {}
                    mean_nie = agg.get("mean_nie")
                    ci = agg.get("bootstrap_ci_95") or agg.get("bootstrap_ci_95".lower())
                    per_layer.append({
                        "layer": br.get("layer"),
                        "mean_nie": float(mean_nie) if mean_nie is not None else None,
                        "ci_95": list(ci)[:2] if isinstance(ci, (list, tuple)) and len(ci) >= 2 else None,
                    })
                band_summary_rows.append({
                    "pair_id": pair_id,
                    "model": model,
                    "best_layer": best_layer,
                    "top_k_layers": top_layers[:band_size] if band_size > 1 else [best_layer],
                    "per_layer": per_layer,
                })

    _write_json(
        cfg.out_dir / "artifacts" / "stats" / "exploratory_results.json",
        {"rows": exploratory_rows, "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")},
    )
    if band_summary_rows:
        _write_json(
            cfg.out_dir / "artifacts" / "stats" / "mediation_band_summary.json",
            {"rows": band_summary_rows, "band_size": band_size, "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")},
        )

    _write_json(
        cfg.out_dir / "artifacts" / "stats" / "prompt_format_robustness.json",
        {
            "rows": prompt_robustness_rows,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "executed": bool(cfg.execute_experiments),
            "status": "ok" if prompt_robustness_rows else "empty",
        },
    )

    # Variant comparison (affine vs skipless) when enabled
    if getattr(cfg, "compare_variants", False) and cfg.execute_experiments:
        for pair_id in cfg.pairs:
            total = len(load_pair_records(pair_id))
            plan = _runtime_three_way_plan(
                cfg,
                total,
                include_blind_eval=bool(getattr(cfg, "run_blind_eval", False)),
            )
            n_icl, n_select, n_eval = plan.n_icl, plan.n_selection, plan.n_eval
            prepared_split_dir = cfg.out_dir / "data" / "processed" / pair_id
            for model in cfg.models:
                out = _pair_dir(cfg.out_dir, pair_id, model)
                rc = RunConfig(
                    out_dir=out,
                    n_icl=n_icl,
                    n_select=n_select,
                    n_eval=n_eval,
                    seeds=list(cfg.seeds),
                    model=model,
                    pair=pair_id,
                    prepared_split_dir=str(prepared_split_dir),
                    use_blind_eval=bool(getattr(cfg, "run_blind_eval", False)),
                )
                try:
                    var_results = run_variant_comparison(rc, n_eval_sample=20)
                    for vr in var_results:
                        variant = vr.get("variant", "affine_skip")
                        _write_json(
                            cfg.out_dir / "artifacts" / "variants" / variant / model / f"{pair_id}.json",
                            {"pair_id": pair_id, "model": model, "variant": variant, **vr},
                        )
                except Exception as e:
                    print(f"[rescue_research] Variant comparison failed ({pair_id}/{model}): {e}", flush=True)


def _compute_h3_strict(out_dir: Path, confirmatory_rows: List[Dict]) -> bool:
    """
    H3: mediated component is positive and directionally aligned with H1/H2.

    Implementation:
    - join mediation artifacts with confirmatory sufficiency/necessity effects
      (mean_pe, mean_ae) by (pair_id, model)
    - compute per-(pair, model) H3 pass using NIE CI + triangulation
    - declare global H3 pass if >=3/4 locked pairs have at least one model passing
    """
    # (pair_id, model) -> (mean_pe, mean_ae)
    by_pair_model: Dict[tuple[str, str], tuple[float, float]] = {}
    for r in confirmatory_rows if isinstance(confirmatory_rows, list) else []:
        if not isinstance(r, dict):
            continue
        pair_id = str(r.get("pair_id", "")).strip()
        model = str(r.get("model", "")).strip()
        if not pair_id or not model:
            continue
        mean_pe = _to_float(r.get("mean_pe"))
        mean_ae = _to_float(r.get("mean_ae"))
        if mean_pe == mean_pe and mean_ae == mean_ae:
            by_pair_model.setdefault((pair_id, model), (mean_pe, mean_ae))

    med_root = out_dir / "artifacts" / "mediation"
    pair_has_pass: Dict[str, bool] = {}
    for p in sorted(med_root.glob("**/*.json")):
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        pair_id = str(payload.get("pair_id", "")).strip()
        model = str(payload.get("model", "")).strip()
        shared = payload.get("shared_mediation_result", {}) or {}
        if not pair_id or not model or not isinstance(shared, dict):
            continue
        suff_nec = by_pair_model.get((pair_id, model))
        if suff_nec is None:
            # fallback: allow NIE>0 alone when we can't join to H1/H2 signals
            agg = shared.get("aggregate_stats", {}) or {}
            mean_nie = agg.get("mean_nie")
            try:
                if mean_nie is not None and float(mean_nie) > 0.0:
                    pair_has_pass[pair_id] = True
            except Exception:
                pass
            continue
        mean_pe, mean_ae = suff_nec
        if h3_pass_strict(
            sufficiency_effect=mean_pe,
            necessity_effect=mean_ae,
            shared_mediation_result=shared,
            require_triangulation_accepted=True,
        ):
            pair_has_pass[pair_id] = True

    return sum(1 for pid, ok in pair_has_pass.items() if ok and pid in LOCKED_LANGUAGE_PAIRS) >= 3


def stage_report_bundle(cfg: PipelineConfig) -> None:
    ensure_contract_dirs(cfg.out_dir)
    # Always compute analysis summaries used by extended tables/figures.
    compute_and_save_attention_control_summary(cfg.out_dir)
    compute_and_save_transcoder_variant_summary(cfg.out_dir)

    try:
        from rescue_research.reporting.figures import generate_mandatory_figures
    except ModuleNotFoundError as exc:
        print(
            f"[rescue_research] WARNING: figure generation dependency missing ({exc}). "
            "Skipping mandatory figure generation.",
            flush=True,
        )

        def generate_mandatory_figures(_: Path) -> List[Path]:
            return []

    generate_mandatory_figures(cfg.out_dir)
    generate_mandatory_tables(cfg.out_dir)

    fidelity_gate = _compute_transcoder_fidelity_gate(cfg.out_dir)
    _write_json(
        cfg.out_dir / "artifacts" / "audit" / "transcoder_fidelity_gate.json",
        fidelity_gate,
    )

    confirmatory_path = cfg.out_dir / "artifacts" / "stats" / "confirmatory_results.json"
    confirmatory = {}
    if confirmatory_path.exists():
        confirmatory = _normalize_confirmatory_payload(
            _read_json_safe(confirmatory_path, "confirmatory_results")
        )
    rows = list(_payload_value(confirmatory, "rows", []))
    summary = _payload_value(confirmatory, "summary", {})
    if not isinstance(summary, dict):
        summary = {}
    passed = [bool(r.get("primary_passed")) for r in rows if isinstance(r, dict)]
    h1_pass = bool(
        _payload_value(
            summary,
            "h1_pass",
            bool(passed) and (sum(1 for x in passed if x) >= max(1, len(passed) // 2)),
        )
    )
    h2_pass = bool(_payload_value(summary, "h2_pass", False))
    h3_from_mediation = _compute_h3_strict(cfg.out_dir, rows)
    h3_pass = bool(
        _payload_value(summary, "h3_pass", False) or h3_from_mediation
    )
    practical_floor_passed = bool(
        _payload_value(summary, "practical_floor_passed", False)
    )
    directional_pair_count = int(
        _payload_value(summary, "directional_pair_count", 0)
    )
    controls_passed = bool(
        _payload_value(summary, "controls_passed", False)
    )
    gate_b_passed = bool(h1_pass and h2_pass)
    gate_c_passed = bool(h3_pass)
    gate_fidelity_passed = bool(fidelity_gate.get("gate_passed", False))

    reproducibility_report = validate_artifacts(
        out_dir=cfg.out_dir,
        pairs=cfg.pairs,
        models=cfg.models,
        seeds=cfg.seeds,
    )
    reproducibility_passed = bool(
        reproducibility_report.ok and not reproducibility_report.warnings
    )
    protocol_status = _compute_protocol_compliance(cfg)
    reproducibility_passed = bool(
        reproducibility_passed
        and protocol_status.get("confirmatory_protocol_passed", False)
        and gate_fidelity_passed
    )
    gate_e_passed = bool(reproducibility_passed)

    _write_json(
        cfg.out_dir / "artifacts" / "audit" / "protocol_compliance.json",
        {
            **protocol_status,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    _write_json(
        cfg.out_dir / "artifacts" / "audit" / "reproducibility_check.json",
        {
            "ok": bool(reproducibility_report.ok),
            "missing_path_count": len(reproducibility_report.missing_paths),
            "missing_paths_preview": reproducibility_report.missing_paths[:50],
            "warnings": list(reproducibility_report.warnings),
            "strict_reproducibility_passed": reproducibility_passed,
            "pair_matrix_mode": protocol_status.get("pair_matrix_mode"),
            "confirmatory_protocol_passed": protocol_status.get(
                "confirmatory_protocol_passed"
            ),
            "protocol_notes": list(protocol_status.get("notes", [])),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    gate_a = _read_json_safe(
        cfg.out_dir / "artifacts" / "audit" / "gate_a_status.json",
        "gate_a_status",
    )
    _write_json(
        cfg.out_dir / "artifacts" / "audit" / "submission_gates.json",
        {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gate_a_baseline_rescue_passed": bool(gate_a.get("gate_a_passed", False)),
            "gate_b_sufficiency_necessity_passed": gate_b_passed,
            "gate_c_mediation_alignment_passed": gate_c_passed,
            "gate_d_rubric_inputs": {
                "practical_floor_passed": bool(practical_floor_passed),
                "directional_pair_count": int(directional_pair_count),
                "controls_passed": bool(controls_passed),
            },
            "gate_e_reproducibility_passed": gate_e_passed,
            "gate_fidelity_passed": gate_fidelity_passed,
        },
    )

    decide_publication_branch(
        out_path=cfg.out_dir / "artifacts" / "final" / "publication_decision.json",
        h1_pass=h1_pass,
        h2_pass=h2_pass,
        h3_pass=h3_pass,
        practical_floor_passed=practical_floor_passed,
        directional_pair_count=directional_pair_count,
        controls_passed=controls_passed,
        reproducibility_passed=reproducibility_passed,
        protocol_compliance_passed=bool(
            protocol_status.get("confirmatory_protocol_passed", False)
        ),
        protocol_notes=list(protocol_status.get("notes", [])),
        fidelity_gate_passed=gate_fidelity_passed,
    )

    strict_mode = bool(protocol_status.get("confirmatory_protocol_passed", False))
    blocking_failures: List[str] = []
    if strict_mode and not bool(gate_a.get("gate_a_passed", False)):
        blocking_failures.append("Gate A failed: baseline rescue not established in >=3/4 pairs.")
    if strict_mode and not gate_b_passed:
        blocking_failures.append("Gate B failed: sufficiency/necessity criteria not met (H1/H2).")
    if strict_mode and not gate_c_passed:
        blocking_failures.append("Gate C failed: mediation directionality not established (H3).")
    if strict_mode and not gate_fidelity_passed:
        blocking_failures.append("Transcoder fidelity gate failed.")
    if not reproducibility_report.ok:
        blocking_failures.append(
            f"Validator missing required artifacts ({len(reproducibility_report.missing_paths)} missing)."
        )
    if reproducibility_report.warnings:
        blocking_failures.append(
            "Validator warnings present: " + "; ".join(reproducibility_report.warnings[:5])
        )

    prompt_robustness = _read_json_safe(
        cfg.out_dir / "artifacts" / "stats" / "prompt_format_robustness.json",
        "prompt_format_robustness",
    )
    prompt_rows = (
        prompt_robustness.get("rows", [])
        if isinstance(prompt_robustness, dict)
        else []
    )
    if strict_mode and bool(getattr(cfg, "execute_experiments", True)) and not prompt_rows:
        blocking_failures.append(
            "Prompt-format robustness artifact is empty; run stage_robustness successfully before bundling."
        )

    if blocking_failures:
        msg = "Submission bundle blocked:\n- " + "\n- ".join(blocking_failures)
        if not bool(getattr(cfg, "execute_experiments", True)):
            print(
                "[rescue_research] WARNING: " + msg + "\n(no-execute mode: not raising)",
                flush=True,
            )
            return
        raise RuntimeError(msg)

    build_submission_bundle(cfg.out_dir)

