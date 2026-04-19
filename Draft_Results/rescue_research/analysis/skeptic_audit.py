from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _finite(values: Iterable[Any]) -> List[float]:
    out: List[float] = []
    for v in values:
        f = _to_float(v)
        if f == f:
            out.append(float(f))
    return out


def _mean(values: Iterable[Any]) -> float:
    vals = _finite(values)
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _verdict_from_threshold(value: float, threshold: float, op: str) -> str:
    if value != value:
        return "UNAVAILABLE"
    if op == "<":
        return "PASS" if value < threshold else "FAIL"
    if op == "<=":
        return "PASS" if value <= threshold else "FAIL"
    if op == ">":
        return "PASS" if value > threshold else "FAIL"
    if op == ">=":
        return "PASS" if value >= threshold else "FAIL"
    return "UNAVAILABLE"


def _claim_from_auto_scale_ratio(ratio: float, threshold: float = 0.80) -> str:
    if ratio != ratio:
        return "UNDECIDED"
    return "Calibration/Gating" if ratio >= threshold else "Algorithmic Induction"


def build_skeptic_pass_payload(
    rows: List[Dict[str, Any]],
    *,
    protocol_version: str = "2026-03-LOCKED",
    auto_scale_threshold: float = 0.80,
    cross_task_threshold: float = 0.20,
    bracket_threshold: float = 0.15,
    jaccard_top5_threshold: float = 0.70,
    jaccard_top25_threshold: float = 0.50,
) -> Dict[str, Any]:
    by_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows if isinstance(rows, list) else []:
        if not isinstance(r, dict):
            continue
        model = str(r.get("model", "")).strip()
        if not model:
            continue
        by_model[model].append(r)

    model_reports: Dict[str, Dict[str, Any]] = {}

    for model, mrows in sorted(by_model.items()):
        auto_ratio_nll = _mean(r.get("auto_scale_ratio") for r in mrows)
        auto_ratio = _mean(r.get("auto_scale_ratio_adjudicated") for r in mrows)
        if auto_ratio != auto_ratio:
            auto_ratio = auto_ratio_nll
        auto_ratio_pe = _mean(r.get("auto_scale_ratio_pe") for r in mrows)
        auto_ratio_metric = str(
            next(
                (
                    r.get("auto_scale_ratio_metric")
                    for r in mrows
                    if isinstance(r, dict) and r.get("auto_scale_ratio_metric")
                ),
                "nll",
            )
        )
        auto_artifact_rate = _mean(r.get("auto_scale_intervention_artifact") for r in mrows)
        mean_feature_cosine = _mean(r.get("mean_feature_cosine_zs_icl") for r in mrows)
        mean_feature_jaccard = _mean(r.get("mean_feature_identity_jaccard_zs_icl") for r in mrows)

        mean_pe = _mean(r.get("mean_pe") for r in mrows)
        mean_pe_cross_task = _mean(r.get("mean_pe_cross_task") for r in mrows)
        mean_pe_shuffle = _mean(r.get("mean_pe_shuffle") for r in mrows)
        mean_active_features_icl = _mean(r.get("mean_active_features_icl") for r in mrows)
        mean_active_features_zs = _mean(r.get("mean_active_features_zs") for r in mrows)
        if mean_pe == mean_pe and abs(mean_pe) > 1e-8 and mean_pe_cross_task == mean_pe_cross_task:
            cross_task_ratio = float(abs(mean_pe_cross_task) / max(1e-8, abs(mean_pe)))
        else:
            cross_task_ratio = float("nan")
        if mean_pe_cross_task == mean_pe_cross_task and mean_pe_shuffle == mean_pe_shuffle:
            cross_task_vs_random_feature_ratio = float(
                abs(mean_pe_cross_task) / max(1e-8, abs(mean_pe_shuffle))
            )
        else:
            cross_task_vs_random_feature_ratio = float("nan")

        bracket_rescue_ratio = _mean(r.get("bracket_rescue_ratio") for r in mrows)
        top5_jaccard = _mean(r.get("top5_jaccard") for r in mrows)
        top25_jaccard = _mean(r.get("top25_jaccard") for r in mrows)
        attn_ablation_harm = _mean(r.get("mean_nll_harm_attn_head_ablation") for r in mrows)
        reconstruction_mse = _mean(r.get("mean_reconstruction_mse_icl") for r in mrows)
        mean_rope_position_gap = _mean(r.get("mean_rope_position_gap") for r in mrows)
        mean_fragmentation_rate = _mean(r.get("input_fragmentation_rate_ge_3_tokens") for r in mrows)
        mean_logit_icl_first = _mean(r.get("mean_logit_icl_first") for r in mrows)
        mean_logit_patched_first = _mean(r.get("mean_logit_patched_first") for r in mrows)
        softcap_saturation_risk = bool(
            _mean(r.get("softcap_saturation_risk") for r in mrows) >= 0.5
        )
        mean_delta_competitor_prob_patch = _mean(
            r.get("mean_delta_competitor_prob_patch") for r in mrows
        )
        mean_delta_competitor_logit_patch = _mean(
            r.get("mean_delta_competitor_logit_patch") for r in mrows
        )
        mean_selected_feature_dla_target = _mean(
            r.get("mean_selected_feature_dla_target") for r in mrows
        )
        mean_selected_feature_dla_competitor = _mean(
            r.get("mean_selected_feature_dla_competitor") for r in mrows
        )
        mean_dla_target_minus_competitor = _mean(
            r.get("mean_dla_target_minus_competitor") for r in mrows
        )
        mean_nll_harm_english_neutral_patch = _mean(
            r.get("mean_nll_harm_english_neutral_patch") for r in mrows
        )
        feature_collision_risk_rate = _mean(
            r.get("feature_collision_risk_rate") for r in mrows
        )
        mean_delta_bos_attention_next_layer_patch = _mean(
            r.get("mean_delta_bos_attention_next_layer_patch") for r in mrows
        )
        context_expectation_warning_rate = _mean(
            r.get("context_expectation_warning_rate") for r in mrows
        )
        power_law_k90 = _mean(
            r.get("power_law_k_at_90pct_max_rescue") for r in mrows
        )
        power_law_hint = str(
            next(
                (
                    r.get("power_law_topology_hint")
                    for r in mrows
                    if isinstance(r, dict) and r.get("power_law_topology_hint")
                ),
                "",
            )
        )

        # Tokenization sanity: improvement sign should agree between token- and char-normalized NLL.
        nll_tok = _mean(r.get("mean_nll_improvement_patch") for r in mrows)
        nll_char = _mean(r.get("mean_nll_per_char_improvement_patch") for r in mrows)
        nll_null = _mean(r.get("mean_nll_improvement_null_patch") for r in mrows)
        if nll_tok == nll_tok and nll_char == nll_char:
            tokenization_consistent = bool((nll_tok >= 0 and nll_char >= 0) or (nll_tok < 0 and nll_char < 0))
            nll_alignment_status = "PASS" if tokenization_consistent else "FAIL"
        else:
            tokenization_consistent = None
            nll_alignment_status = "UNAVAILABLE"

        auto_validity = (
            "VALID"
            if (
                mean_feature_cosine == mean_feature_cosine
                and mean_feature_jaccard == mean_feature_jaccard
                and mean_feature_cosine >= 0.70
                and mean_feature_jaccard >= 0.60
            )
            else (
                "DISJOINT_FEATURES"
                if (
                    mean_feature_cosine == mean_feature_cosine
                    and mean_feature_cosine < 0.30
                )
                else "UNCERTAIN"
            )
        )
        if mean_active_features_zs == mean_active_features_zs and mean_active_features_zs < 2.0:
            auto_validity = "ZS_L0_TOO_LOW"

        auto_verdict = _verdict_from_threshold(auto_ratio, auto_scale_threshold, "<")
        if auto_validity == "ZS_L0_TOO_LOW":
            auto_verdict = "UNAVAILABLE"
        if cross_task_vs_random_feature_ratio == cross_task_vs_random_feature_ratio:
            if cross_task_vs_random_feature_ratio < 0.5:
                diss_verdict = "PASS"
            elif cross_task_vs_random_feature_ratio > 0.9:
                diss_verdict = "FAIL"
            else:
                diss_verdict = "UNAVAILABLE"
        else:
            diss_verdict = _verdict_from_threshold(cross_task_ratio, cross_task_threshold, "<")
        bracket_verdict = _verdict_from_threshold(bracket_rescue_ratio, bracket_threshold, "<")
        top5_verdict = _verdict_from_threshold(top5_jaccard, jaccard_top5_threshold, ">=")
        top25_verdict = _verdict_from_threshold(top25_jaccard, jaccard_top25_threshold, ">=")
        attn_verdict = _verdict_from_threshold(attn_ablation_harm, 0.10, ">")
        recon_verdict = _verdict_from_threshold(reconstruction_mse, 0.05, "<=")
        if nll_tok == nll_tok and nll_tok > 0 and nll_null == nll_null:
            null_ratio = float(nll_null / max(1e-8, nll_tok))
            null_verdict = "PASS" if null_ratio < 0.25 else "FAIL"
        elif nll_null == nll_null and nll_tok != nll_tok:
            null_ratio = float("nan")
            null_verdict = "UNAVAILABLE"
        else:
            null_ratio = float("nan")
            null_verdict = "UNAVAILABLE"

        skeptic_checks = [recon_verdict, auto_verdict, diss_verdict, bracket_verdict, top5_verdict, top25_verdict, attn_verdict, null_verdict]
        passed_checks = sum(1 for v in skeptic_checks if v == "PASS")
        known_checks = sum(1 for v in skeptic_checks if v in {"PASS", "FAIL"})
        confidence = float(passed_checks / known_checks) if known_checks else float("nan")

        model_reports[model] = {
            "validity_gates": {
                "reconstruction_fidelity": {
                    "mean_reconstruction_mse_icl": reconstruction_mse,
                    "threshold": 0.05,
                    "verdict": _verdict_from_threshold(reconstruction_mse, 0.05, "<="),
                }
            },
            "skeptic_tests": {
                "auto_scale_test": {
                    "ratio": auto_ratio,
                    "ratio_nll": auto_ratio_nll,
                    "ratio_pe": auto_ratio_pe,
                    "ratio_metric": auto_ratio_metric,
                    "ratio_mult": _mean(r.get("auto_scale_ratio_mult") for r in mrows),
                    "ratio_add": _mean(r.get("auto_scale_ratio_add") for r in mrows),
                    "threshold": auto_scale_threshold,
                    "intervention_artifact_rate": auto_artifact_rate,
                    "mean_feature_cosine_zs_icl": mean_feature_cosine,
                    "mean_feature_identity_jaccard_zs_icl": mean_feature_jaccard,
                    "mean_active_features_icl": mean_active_features_icl,
                    "mean_active_features_zs": mean_active_features_zs,
                    "validity": auto_validity,
                    "verdict": auto_verdict,
                    "claim_if_isolated": _claim_from_auto_scale_ratio(auto_ratio, auto_scale_threshold),
                },
                "double_dissociation": {
                    "task_specificity_ratio": cross_task_ratio,
                    "threshold": cross_task_threshold,
                    "cross_task_vs_random_feature_ratio": cross_task_vs_random_feature_ratio,
                    "ratio_bands": {
                        "task_specific_pass": "<0.5",
                        "layer_general_fail": ">0.9",
                    },
                    "verdict": diss_verdict,
                },
                "localization_test": {
                    "bracket_rescue_ratio": bracket_rescue_ratio,
                    "threshold": bracket_threshold,
                    "verdict": bracket_verdict,
                },
                "stability_test": {
                    "top_5_jaccard": top5_jaccard,
                    "top_25_jaccard": top25_jaccard,
                    "thresholds": {
                        "top5": jaccard_top5_threshold,
                        "top25": jaccard_top25_threshold,
                    },
                    "top5_verdict": top5_verdict,
                    "top25_verdict": top25_verdict,
                },
                "tokenization_sanity": {
                    "tokenization_consistent": tokenization_consistent,
                    "verdict": nll_alignment_status,
                    "mean_nll_improvement_patch": nll_tok,
                    "mean_nll_per_char_improvement_patch": nll_char,
                },
                "null_icl_control": {
                    "mean_nll_improvement_null_patch": nll_null,
                    "relative_to_main_patch": null_ratio,
                    "threshold": 0.25,
                    "verdict": null_verdict,
                    "notes": "PASS means context-length-only (Null-ICL) rescue stays far below main patch rescue.",
                },
                "attention_ablation_test": {
                    "mean_nll_harm": attn_ablation_harm,
                    "mean_sink_heads_excluded": _mean(r.get("mean_attn_sink_heads_excluded") for r in mrows),
                    "sink_threshold": _mean(r.get("attn_sink_threshold") for r in mrows),
                    "threshold": 0.10,
                    "verdict": attn_verdict,
                    "notes": "PASS means ablating selected heads harms ICL (attention route materially contributes).",
                },
            },
            "contextual_diagnostics": {
                "rope_position_drift": {
                    "mean_rope_position_gap": mean_rope_position_gap,
                    "drift_warning": bool(
                        mean_rope_position_gap == mean_rope_position_gap
                        and abs(mean_rope_position_gap) >= 64.0
                    ),
                    "notes": "Large ICL-vs-ZS position gaps can induce RoPE/GQA positional dissonance, especially in deeper models.",
                },
                "softcap_saturation_guard": {
                    "mean_logit_icl_first": mean_logit_icl_first,
                    "mean_logit_patched_first": mean_logit_patched_first,
                    "softcap_saturation_risk": softcap_saturation_risk,
                    "recommendation": (
                        "Prefer PE-based adjudication when mean correct logit exceeds ~25."
                        if softcap_saturation_risk
                        else "NLL and PE should both be reliable."
                    ),
                },
                "agglutinative_fragmentation_guard": {
                    "input_fragmentation_rate_ge_3_tokens": mean_fragmentation_rate,
                    "high_fragmentation_warning": bool(
                        mean_fragmentation_rate == mean_fragmentation_rate
                        and mean_fragmentation_rate > 0.50
                    ),
                    "notes": "If high, single-token patching reflects terminal-state rescue more than full word-level algorithm execution.",
                },
                "suppression_vs_promotion": {
                    "mean_delta_competitor_prob_patch": mean_delta_competitor_prob_patch,
                    "mean_delta_competitor_logit_patch": mean_delta_competitor_logit_patch,
                    "mechanism_hint": (
                        "suppression/gating"
                        if (
                            mean_delta_competitor_logit_patch == mean_delta_competitor_logit_patch
                            and mean_delta_competitor_logit_patch < 0.0
                        )
                        else "promotion/mapping"
                    ),
                    "notes": "Negative competitor-logit deltas indicate Hindi-off-switch behavior; positive deltas suggest direct target promotion.",
                },
                "dla_alignment": {
                    "mean_selected_feature_dla_target": mean_selected_feature_dla_target,
                    "mean_selected_feature_dla_competitor": mean_selected_feature_dla_competitor,
                    "mean_dla_target_minus_competitor": mean_dla_target_minus_competitor,
                    "interpretation_hint": (
                        "calibration-compatible (direct logit channel)"
                        if (
                            mean_selected_feature_dla_target
                            == mean_selected_feature_dla_target
                            and mean_selected_feature_dla_target > 0.0
                        )
                        else "operator-like / indirect routing"
                    ),
                },
                "feature_superposition_guard": {
                    "mean_nll_harm_english_neutral_patch": mean_nll_harm_english_neutral_patch,
                    "feature_collision_risk_rate": feature_collision_risk_rate,
                    "collision_warning": bool(
                        feature_collision_risk_rate == feature_collision_risk_rate
                        and feature_collision_risk_rate > 0.25
                    ),
                    "notes": "High harm on an unrelated English copy task suggests feature collision/superposition leakage.",
                },
                "kv_context_expectation_guard": {
                    "mean_delta_bos_attention_next_layer_patch": mean_delta_bos_attention_next_layer_patch,
                    "context_expectation_warning_rate": context_expectation_warning_rate,
                    "phantom_context_warning": bool(
                        context_expectation_warning_rate
                        == context_expectation_warning_rate
                        and context_expectation_warning_rate > 0.25
                    ),
                    "notes": "Large positive BOS-attention shifts after patching may indicate context-expectation (KV-ghost) features.",
                },
                "circuit_topology": {
                    "k_at_90pct_max_rescue": power_law_k90,
                    "topology_hint": power_law_hint,
                    "interpretation": (
                        "few-feature bottleneck"
                        if (power_law_k90 == power_law_k90 and power_law_k90 <= 2.0)
                        else (
                            "distributed circuit"
                            if (power_law_k90 == power_law_k90 and power_law_k90 > 8.0)
                            else "mixed"
                        )
                    ),
                },
            },
            "final_claim_gate": {
                "primary_claim": _claim_from_auto_scale_ratio(auto_ratio, auto_scale_threshold),
                "confidence_score": confidence,
                "skeptic_consensus": (
                    "ACCEPTED" if known_checks and all(v != "FAIL" for v in skeptic_checks if v != "UNAVAILABLE") else "REVIEW_REQUIRED"
                ),
            },
        }

    cross_model_diagnostics: Dict[str, Any] = {}
    if "1b" in model_reports and "12b" in model_reports:
        m1 = model_reports.get("1b", {})
        m12 = model_reports.get("12b", {})
        c1 = (
            m1.get("final_claim_gate", {}).get("skeptic_consensus")
            if isinstance(m1, dict)
            else None
        )
        c12 = (
            m12.get("final_claim_gate", {}).get("skeptic_consensus")
            if isinstance(m12, dict)
            else None
        )
        rope12 = (
            m12.get("contextual_diagnostics", {})
            .get("rope_position_drift", {})
            .get("drift_warning")
            if isinstance(m12, dict)
            else None
        )
        cross_model_diagnostics["positional_dissonance_pattern"] = bool(
            c1 == "ACCEPTED" and c12 != "ACCEPTED" and bool(rope12)
        )
        cross_model_diagnostics["notes"] = (
            "If true, interpret 12B failure with 1B success as potential depth/GQA positional sensitivity."
        )

    return {
        "metadata": {
            "protocol_version": protocol_version,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_rows": len(rows),
        },
        "models": model_reports,
        "cross_model_diagnostics": cross_model_diagnostics,
    }


def _sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, float):
        return obj if obj == obj and obj not in (float("inf"), float("-inf")) else None
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def compute_and_write_skeptic_pass(
    out_dir: Path,
    rows: List[Dict[str, Any]],
    *,
    protocol_version: str = "2026-03-LOCKED",
) -> Dict[str, Any]:
    payload = build_skeptic_pass_payload(rows, protocol_version=protocol_version)
    payload = _sanitize_for_json(payload)
    out_path = Path(out_dir) / "artifacts" / "audit" / "skeptic_pass.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return payload
