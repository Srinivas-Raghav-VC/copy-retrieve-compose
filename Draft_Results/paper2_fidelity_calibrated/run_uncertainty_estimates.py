#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PAIRS = ["aksharantar_hin_latin", "aksharantar_tel_latin"]
MODELS = ["1b", "4b"]


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _bootstrap_mean(values: List[float], *, rng: np.random.Generator, n_boot: int) -> Dict[str, float]:
    arr = np.array([float(v) for v in values if np.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0.0, "mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    boots = []
    for _ in range(max(1, int(n_boot))):
        sample = rng.choice(arr, size=arr.size, replace=True)
        boots.append(float(np.mean(sample)))
    return {
        "n": float(arr.size),
        "mean": float(np.mean(arr)),
        "ci_low": float(np.quantile(boots, 0.025)),
        "ci_high": float(np.quantile(boots, 0.975)),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Bootstrap uncertainty estimates over key mechanistic deltas.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--out", type=str, default="")
    return ap.parse_args()


def _pair_short(pair: str) -> str:
    return pair.split("_")[1]


def _path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(int(args.seed))
    rows: List[Dict[str, Any]] = []

    for model in MODELS:
        fv_path = _path("paper2_fidelity_calibrated", "results", "cfom_function_vector_tests", model, "cfom_function_vector_tests.json")
        fv_obj = _load_json(fv_path) if fv_path.exists() else None
        for pair in PAIRS:
            cell = {"model": model, "pair": pair, "metrics": {}}
            # G3
            p = _path("paper2_fidelity_calibrated", "results", "head_to_mlp_edge_attribution", pair, model, "head_to_mlp_edge_attribution.json")
            if p.exists():
                obj = _load_json(p)
                item_rows = list(obj.get("item_rows") or [])
                cell["metrics"]["g3_head_effect_first_prob"] = _bootstrap_mean([r.get("head_only_delta_first_prob", float("nan")) for r in item_rows], rng=rng, n_boot=int(args.n_boot))
                cell["metrics"]["g3_mediation_specific_first_prob"] = _bootstrap_mean([
                    float(r.get("feature_mediated_drop_first_prob", float("nan"))) - float(r.get("random_feature_mediated_drop_first_prob", float("nan")))
                    for r in item_rows
                ], rng=rng, n_boot=int(args.n_boot))
            # G6
            p = _path("paper2_fidelity_calibrated", "results", "feature_knockout_panel", pair, model, "feature_knockout_panel.json")
            if p.exists():
                obj = _load_json(p)
                item_rows = list(obj.get("item_rows") or [])
                cell["metrics"]["g6_feature_necessity_first_prob"] = _bootstrap_mean([r.get("drop_from_full_patch_first_prob", float("nan")) for r in item_rows], rng=rng, n_boot=int(args.n_boot))
            # G4
            p = _path("paper2_fidelity_calibrated", "results", "circuit_sufficiency", pair, model, "circuit_sufficiency.json")
            if p.exists():
                obj = _load_json(p)
                item_rows = list(obj.get("item_rows") or [])
                cell["metrics"]["g4_circuit_only_minus_zs_first_prob"] = _bootstrap_mean([
                    float(r.get("circuit_only_first_prob", float("nan"))) - float(r.get("zs_first_prob", float("nan"))) for r in item_rows
                ], rng=rng, n_boot=int(args.n_boot))
                cell["metrics"]["g4_circuit_only_minus_icl_first_prob"] = _bootstrap_mean([
                    float(r.get("circuit_only_first_prob", float("nan"))) - float(r.get("icl_first_prob", float("nan"))) for r in item_rows
                ], rng=rng, n_boot=int(args.n_boot))
            # direct icl necessity
            p = _path("paper2_fidelity_calibrated", "results", "direct_icl_feature_necessity", pair, model, "direct_icl_feature_necessity.json")
            if p.exists():
                obj = _load_json(p)
                item_rows = list(obj.get("item_rows") or [])
                cell["metrics"]["direct_icl_specific_necessity_first_prob"] = _bootstrap_mean([
                    float(r.get("core_drop_first_prob", float("nan"))) - float(r.get("random_drop_first_prob", float("nan"))) for r in item_rows
                ], rng=rng, n_boot=int(args.n_boot))
            # geometry robustness
            p = _path("paper2_fidelity_calibrated", "results", "patch_geometry_robustness", pair, model, "patch_geometry_robustness.json")
            if p.exists():
                obj = _load_json(p)
                item_rows = list(obj.get("item_rows") or [])
                by_item = {}
                for r in item_rows:
                    by_item.setdefault(int(r["item_index"]), {})[str(r["geometry"])] = r
                raw_minus_sign = []
                raw_minus_clip = []
                for bucket in by_item.values():
                    if "raw" in bucket and "sign_normalized" in bucket:
                        raw_minus_sign.append(float(bucket["raw"].get("prob_patched_first", float("nan"))) - float(bucket["sign_normalized"].get("prob_patched_first", float("nan"))))
                    if "raw" in bucket and "clipped" in bucket:
                        raw_minus_clip.append(float(bucket["raw"].get("prob_patched_first", float("nan"))) - float(bucket["clipped"].get("prob_patched_first", float("nan"))))
                cell["metrics"]["geometry_raw_minus_sign_first_prob"] = _bootstrap_mean(raw_minus_sign, rng=rng, n_boot=int(args.n_boot))
                cell["metrics"]["geometry_raw_minus_clipped_first_prob"] = _bootstrap_mean(raw_minus_clip, rng=rng, n_boot=int(args.n_boot))
            # transcoder-family consensus
            p = _path("paper2_fidelity_calibrated", "results", "transcoder_family_consensus", pair, model, "transcoder_family_consensus.json")
            if p.exists():
                obj = _load_json(p)
                item_rows = list(obj.get("item_rows") or [])
                key_j = "jaccard__skipless_or_non_affine__affine_skip"
                key_d = "delta_prob_patched_first__skipless_or_non_affine__affine_skip"
                cell["metrics"]["transcoder_consensus_jaccard"] = _bootstrap_mean([r.get(key_j, float("nan")) for r in item_rows], rng=rng, n_boot=int(args.n_boot))
                cell["metrics"]["transcoder_variant_prob_delta"] = _bootstrap_mean([r.get(key_d, float("nan")) for r in item_rows], rng=rng, n_boot=int(args.n_boot))
            # feature stability
            p = _path("paper2_fidelity_calibrated", "results", "feature_stability_resamples", pair, model, "feature_stability_resamples.json")
            if p.exists():
                obj = _load_json(p)
                item_rows = list(obj.get("pairwise_jaccard") or [])
                cell["metrics"]["feature_resample_jaccard"] = _bootstrap_mean([r.get("jaccard", float("nan")) for r in item_rows], rng=rng, n_boot=int(args.n_boot))
            # cross-artifact feature stability
            p = _path("paper2_fidelity_calibrated", "results", "cross_artifact_feature_stability", pair, model, "cross_artifact_feature_stability.json")
            if p.exists():
                obj = _load_json(p)
                summ = dict(obj.get("summary") or {})
                cell["metrics"]["cross_artifact_jaccard"] = {"n": 1.0, "mean": float(summ.get("jaccard__skipless_or_non_affine__affine_skip", float("nan"))), "ci_low": float("nan"), "ci_high": float("nan")}
            # position shift
            p = _path("paper2_fidelity_calibrated", "results", "position_shift_sanity", pair, model, "position_shift_sanity.json")
            if p.exists():
                obj = _load_json(p)
                item_rows = list(obj.get("item_rows") or [])
                by_item = {}
                for r in item_rows:
                    by_item.setdefault(int(r["item_index"]), {})[int(r["offset"])] = r
                center_minus_neighbors = []
                for bucket in by_item.values():
                    if 0 in bucket and -1 in bucket and 1 in bucket:
                        neigh = np.nanmean([
                            float(bucket[-1].get("prob_patched_first", float("nan"))),
                            float(bucket[1].get("prob_patched_first", float("nan"))),
                        ])
                        center_minus_neighbors.append(float(bucket[0].get("prob_patched_first", float("nan"))) - float(neigh))
                cell["metrics"]["position_center_minus_neighbors"] = _bootstrap_mean(center_minus_neighbors, rng=rng, n_boot=int(args.n_boot))
            # gap analysis
            p = _path("paper2_fidelity_calibrated", "results", "target_competitor_logit_gap", pair, model, "target_competitor_logit_gap.json")
            if p.exists():
                obj = _load_json(p)
                item_rows = list(obj.get("item_rows") or [])
                cell["metrics"]["patched_gap_minus_zs_gap"] = _bootstrap_mean([
                    float(r.get("patched_target_minus_competitor_logit", float("nan"))) - float(r.get("zs_target_minus_competitor_logit", float("nan"))) for r in item_rows
                ], rng=rng, n_boot=int(args.n_boot))
            # leave-k-out contribution
            p = _path("paper2_fidelity_calibrated", "results", "leave_k_out_icl_contribution", pair, model, "leave_k_out_icl_contribution.json")
            if p.exists():
                obj = _load_json(p)
                item_rows = list(obj.get("item_rows") or [])
                for k in sorted(set(int(r["k"]) for r in item_rows)):
                    rows_k = [r for r in item_rows if int(r["k"]) == int(k)]
                    cell["metrics"][f"leavek_drop_first_prob_k{k}"] = _bootstrap_mean([r.get("drop_first_prob", float("nan")) for r in rows_k], rng=rng, n_boot=int(args.n_boot))
            # decoded vs latent
            p = _path("paper2_fidelity_calibrated", "results", "decoded_vs_latent_equivalence", pair, model, "decoded_vs_latent_equivalence.json")
            if p.exists():
                obj = _load_json(p)
                item_rows = list(obj.get("item_rows") or [])
                cell["metrics"]["decoded_minus_latent_first_prob"] = _bootstrap_mean([
                    float(r.get("decoded_sparse_first_prob", float("nan"))) - float(r.get("latent_first_prob", float("nan"))) for r in item_rows
                ], rng=rng, n_boot=int(args.n_boot))
            # neighbor layer
            p = _path("paper2_fidelity_calibrated", "results", "neighbor_layer_causality", pair, model, "neighbor_layer_causality.json")
            if p.exists():
                obj = _load_json(p)
                item_rows = list(obj.get("item_rows") or [])
                by_item = {}
                for r in item_rows:
                    by_item.setdefault(int(r["item_index"]), {})[int(r["layer_offset"])] = r
                center_minus_adj = []
                for bucket in by_item.values():
                    if 0 in bucket and -1 in bucket and 1 in bucket:
                        adj = np.nanmean([
                            float(bucket[-1].get("prob_patched_first", float("nan"))),
                            float(bucket[1].get("prob_patched_first", float("nan"))),
                        ])
                        center_minus_adj.append(float(bucket[0].get("prob_patched_first", float("nan"))) - float(adj))
                cell["metrics"]["neighbor_layer_center_minus_adjacent"] = _bootstrap_mean(center_minus_adj, rng=rng, n_boot=int(args.n_boot))
            # dense positive control
            p = _path("paper2_fidelity_calibrated", "results", "dense_mlp_sweep", pair, model, "dense_layer_sweep_item_level.json")
            q = _path("paper2_fidelity_calibrated", "results", "dense_mlp_sweep", pair, model, "dense_layer_sweep_decision_note.json")
            if p.exists():
                obj = _load_json(p)
                item_rows = list(obj or [])
                best_layer = None
                if q.exists():
                    try:
                        best_layer = int((_load_json(q) or {}).get("best_layer"))
                    except Exception:
                        best_layer = None
                if best_layer is not None:
                    item_rows = [r for r in item_rows if int(r.get("layer", -999)) == int(best_layer)]
                cell["metrics"]["dense_positive_control_first_prob"] = _bootstrap_mean([r.get("pe_first", float("nan")) for r in item_rows], rng=rng, n_boot=int(args.n_boot))
                cell["metrics"]["dense_positive_control_exact_match_delta"] = _bootstrap_mean([
                    float(r.get("exact_match_patched", float("nan"))) - float(r.get("exact_match_zs", float("nan"))) for r in item_rows
                ], rng=rng, n_boot=int(args.n_boot))
            # activation-difference baseline
            p = _path("paper2_fidelity_calibrated", "results", "activation_difference_baseline", pair, model, "activation_difference_baseline.json")
            if p.exists():
                obj = _load_json(p)
                item_rows = list(obj.get("item_rows") or [])
                for space in ("layer_output", "mlp_output"):
                    rows_space = [r for r in item_rows if str(r.get("space")) == space]
                    cell["metrics"][f"actdiff_{space}_first_prob"] = _bootstrap_mean([r.get("pe_first", float("nan")) for r in rows_space], rng=rng, n_boot=int(args.n_boot))
                    cell["metrics"][f"actdiff_{space}_rescue_frac"] = _bootstrap_mean([r.get("rescue_frac_first", float("nan")) for r in rows_space], rng=rng, n_boot=int(args.n_boot))
                    cell["metrics"][f"actdiff_{space}_specificity_margin"] = _bootstrap_mean([
                        float(r.get("pe_first", float("nan"))) - float(r.get("pe_shuffled_first", float("nan"))) for r in rows_space
                    ], rng=rng, n_boot=int(args.n_boot))
            # neutral filler / recency controls
            p = _path("paper2_fidelity_calibrated", "results", "neutral_filler_recency_controls", pair, model, "neutral_filler_recency_controls.json")
            if p.exists():
                obj = _load_json(p)
                item_rows = list(obj.get("item_rows") or [])
                by_item = {}
                for r in item_rows:
                    by_item.setdefault(int(r["item_index"]), {})[str(r["condition"])] = r
                helpful_minus_null = []
                helpful_minus_random = []
                helpful_minus_corrupt = []
                desc_minus_asc = []
                original_minus_reversed = []
                for bucket in by_item.values():
                    if "icl_helpful" in bucket and "icl_null_filler" in bucket:
                        helpful_minus_null.append(float(bucket["icl_helpful"].get("first_prob", float("nan"))) - float(bucket["icl_null_filler"].get("first_prob", float("nan"))))
                    if "icl_helpful" in bucket and "icl_random_indic" in bucket:
                        helpful_minus_random.append(float(bucket["icl_helpful"].get("first_prob", float("nan"))) - float(bucket["icl_random_indic"].get("first_prob", float("nan"))))
                    if "icl_helpful" in bucket and "icl_corrupt" in bucket:
                        helpful_minus_corrupt.append(float(bucket["icl_helpful"].get("first_prob", float("nan"))) - float(bucket["icl_corrupt"].get("first_prob", float("nan"))))
                    if "icl_helpful_similarity_desc" in bucket and "icl_helpful_similarity_asc" in bucket:
                        desc_minus_asc.append(float(bucket["icl_helpful_similarity_desc"].get("first_prob", float("nan"))) - float(bucket["icl_helpful_similarity_asc"].get("first_prob", float("nan"))))
                    if "icl_helpful" in bucket and "icl_helpful_reversed" in bucket:
                        original_minus_reversed.append(float(bucket["icl_helpful"].get("first_prob", float("nan"))) - float(bucket["icl_helpful_reversed"].get("first_prob", float("nan"))))
                cell["metrics"]["neutral_helpful_minus_null_first_prob"] = _bootstrap_mean(helpful_minus_null, rng=rng, n_boot=int(args.n_boot))
                cell["metrics"]["neutral_helpful_minus_random_first_prob"] = _bootstrap_mean(helpful_minus_random, rng=rng, n_boot=int(args.n_boot))
                cell["metrics"]["neutral_helpful_minus_corrupt_first_prob"] = _bootstrap_mean(helpful_minus_corrupt, rng=rng, n_boot=int(args.n_boot))
                cell["metrics"]["recency_similarity_desc_minus_asc_first_prob"] = _bootstrap_mean(desc_minus_asc, rng=rng, n_boot=int(args.n_boot))
                cell["metrics"]["recency_original_minus_reversed_first_prob"] = _bootstrap_mean(original_minus_reversed, rng=rng, n_boot=int(args.n_boot))
            # induction-style head reanalysis
            p = _path("paper2_fidelity_calibrated", "results", "induction_style_head_reanalysis", pair, model, "induction_style_head_reanalysis.json")
            if p.exists():
                obj = _load_json(p)
                item_rows = list(obj.get("item_rows") or [])
                helpful_rows = [r for r in item_rows if str(r.get("condition")) == "icl64"]
                corrupt_rows = [r for r in item_rows if str(r.get("condition")) == "corrupt_icl"]
                cell["metrics"]["induction_matched_pair_mass_icl"] = _bootstrap_mean([r.get("matched_pair_mass", float("nan")) for r in helpful_rows], rng=rng, n_boot=int(args.n_boot))
                cell["metrics"]["induction_paired_target_margin_icl"] = _bootstrap_mean([r.get("paired_target_margin_at_top_source", float("nan")) for r in helpful_rows], rng=rng, n_boot=int(args.n_boot))
                cell["metrics"]["induction_matched_pair_mass_corrupt"] = _bootstrap_mean([r.get("matched_pair_mass", float("nan")) for r in corrupt_rows], rng=rng, n_boot=int(args.n_boot))
                cell["metrics"]["induction_paired_target_margin_corrupt"] = _bootstrap_mean([r.get("paired_target_margin_at_top_source", float("nan")) for r in corrupt_rows], rng=rng, n_boot=int(args.n_boot))
            # head-attribution stability
            p = _path("paper2_fidelity_calibrated", "results", "head_attribution_stability", pair, model, "head_attribution_stability.json")
            if p.exists():
                obj = _load_json(p)
                pairwise = list(obj.get("pairwise_jaccard") or [])
                cell["metrics"]["head_attr_stability_jaccard"] = _bootstrap_mean([r.get("jaccard", float("nan")) for r in pairwise], rng=rng, n_boot=int(args.n_boot))
                cell["metrics"]["head_attr_reference_overlap_jaccard"] = {
                    "n": 1.0,
                    "mean": float(obj.get("mean_reference_overlap_jaccard", float("nan"))),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                }
            # selected-layer fidelity / ceiling compare
            p = _path("paper2_fidelity_calibrated", "results", "selected_layer_fidelity_compare", pair, model, "selected_layer_fidelity_compare.json")
            if p.exists():
                obj = _load_json(p)
                item_rows = list(obj.get("item_rows") or [])
                for variant in ("skipless_or_non_affine", "affine_skip"):
                    rows_v = [r for r in item_rows if str(r.get("variant")) == variant]
                    if not rows_v:
                        continue
                    key_suffix = variant.replace("/", "_")
                    cell["metrics"][f"selected_fidelity_icl_cosine__{key_suffix}"] = _bootstrap_mean([
                        r.get("icl_reconstruction_cosine", float("nan")) for r in rows_v
                    ], rng=rng, n_boot=int(args.n_boot))
                    cell["metrics"][f"selected_fidelity_delta_share__{key_suffix}"] = _bootstrap_mean([
                        r.get("delta_topk_mass_share", float("nan")) for r in rows_v
                    ], rng=rng, n_boot=int(args.n_boot))
            # G11 function vectors
            if fv_obj is not None:
                item_rows = [r for r in list(fv_obj.get("item_rows") or []) if str(r.get("test_pair")) == pair and str(r.get("donor_pair")) == pair]
                if item_rows:
                    cell["metrics"]["function_vector_samepair_first_prob"] = _bootstrap_mean([
                        float(r.get("fv_first_prob", float("nan"))) - float(r.get("zs_first_prob", float("nan"))) for r in item_rows
                    ], rng=rng, n_boot=int(args.n_boot))
                    cell["metrics"]["function_vector_specificity_first_prob"] = _bootstrap_mean([
                        float(r.get("fv_first_prob", float("nan"))) - float(r.get("random_fv_first_prob", float("nan"))) for r in item_rows
                    ], rng=rng, n_boot=int(args.n_boot))
            rows.append(cell)

    out_path = Path(args.out).resolve() if str(args.out).strip() else _path("paper2_fidelity_calibrated", "results", "uncertainty_estimates", "uncertainty_estimates.json")
    payload = {"experiment": "uncertainty_estimates", "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rows": rows}
    _write_json(out_path, payload)
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
