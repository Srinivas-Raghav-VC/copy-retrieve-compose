#!/usr/bin/env python3
"""
Stage A.5 diagnostic tranche for the frozen CFOM workshop plan.

Scope:
  - Gemma 3 4B only
  - Hindi + Telugu anchor pairs only
  - same selector family and frozen prompts as Stage A
  - tests only:
      1) source_last_subtoken vs target_pos1_teacher_forced
      2) raw vs clipped vs normalized_sign geometry
      3) zero-ablation necessity at the same loci
  - keeps one tiny structured attn_out fairness check only
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _parse_csv(raw: str) -> List[str]:
    return [x.strip() for x in str(raw or "").split(",") if x.strip()]


def _canonical_position_mode(name: str) -> Tuple[str, str]:
    raw = str(name or "source_last_subtoken").strip().lower()
    if raw in {"target_pos1", "target_pos1_teacher_forced"}:
        return "target_pos1_teacher_forced", "target_pos1_teacher_forced"
    return "source_last_subtoken", "source_last_subtoken"


def _canonical_geometry_mode(name: str) -> Tuple[str, str]:
    raw = str(name or "raw").strip().lower()
    if raw in {"normalized_sign", "sign_normalized"}:
        return "normalized_sign", "normalized_sign"
    if raw == "clipped":
        return "clipped", "clipped"
    return "raw", "raw"


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


def _write_rows(base: Path, rows: List[Dict[str, Any]]) -> None:
    _write_json(base.with_suffix(".json"), rows)
    keys: List[str] = sorted({str(k) for row in rows for k in row.keys()}) if rows else []
    with base.with_suffix(".csv").open("w", encoding="utf-8", newline="") as f:
        if not keys:
            f.write("")
            return
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out


def _safe_ratio(numer: Any, denom: Any) -> float:
    n = _safe_float(numer)
    d = _safe_float(denom)
    if not np.isfinite(n) or not np.isfinite(d) or abs(d) < 1e-12:
        return float("nan")
    return float(n / d)


def _git_commit_hash() -> str:
    env_hash = str(os.environ.get("PROJECT_GIT_COMMIT_HASH", "") or "").strip()
    if env_hash:
        return env_hash
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(PROJECT_ROOT),
                text=True,
                stderr=subprocess.DEVNULL,
            )
            .strip()
        )
    except Exception:
        return ""


def _fragmentation_bucket(n_tokens: Any) -> str:
    n = _safe_float(n_tokens)
    if not np.isfinite(n):
        return "unknown"
    if n <= 2:
        return "low"
    if n <= 4:
        return "medium"
    return "high"


def _load_dense_reference_row(pair_id: str, model_key: str, seed: int, layer: int) -> Dict[str, Any] | None:
    dense_path = (
        PROJECT_ROOT
        / "paper2_fidelity_calibrated"
        / "results"
        / "dense_mlp_sweep"
        / pair_id
        / model_key
        / "dense_layer_sweep_results.json"
    )
    if not dense_path.exists():
        return None
    try:
        payload = json.loads(dense_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, list):
        return None
    for row in payload:
        if (
            str(row.get("pair")) == str(pair_id)
            and int(row.get("seed", -1)) == int(seed)
            and int(row.get("layer", -1)) == int(layer)
        ):
            out = dict(row)
            out["dense_reference_path"] = str(dense_path)
            return out
    return None


def _dense_recovery_fields(condition_row: Dict[str, Any], dense_ref: Dict[str, Any] | None) -> Dict[str, Any]:
    if not dense_ref:
        return {
            "dense_reference_available": False,
            "dense_reference_path": "",
            "dense_reference_formula": (
                "dense_recovery_ratio_pe = sparse_mean_pe / dense_mean_pe; "
                "dense_recovery_ratio_exact_match = sparse_exact_match_delta / dense_exact_match_delta; "
                "dense_recovery_ratio_first_entry = sparse_first_entry_delta / dense_first_entry_delta"
            ),
        }

    sparse_em_delta = _safe_float(condition_row.get("exact_match_patched")) - _safe_float(condition_row.get("exact_match_zs"))
    dense_em_delta = _safe_float(dense_ref.get("exact_match_delta"))
    sparse_entry_delta = _safe_float(condition_row.get("first_entry_correct_patched")) - _safe_float(condition_row.get("first_entry_correct_zs"))
    dense_entry_delta = _safe_float(dense_ref.get("first_entry_delta"))

    return {
        "dense_reference_available": True,
        "dense_reference_path": str(dense_ref.get("dense_reference_path", "")),
        "dense_reference_layer": int(dense_ref.get("layer", -1)),
        "dense_reference_mean_pe": _safe_float(dense_ref.get("mean_pe")),
        "dense_reference_exact_match_delta": dense_em_delta,
        "dense_reference_first_entry_delta": dense_entry_delta,
        "dense_reference_specificity_margin_vs_controls": _safe_float(dense_ref.get("specificity_margin_vs_controls")),
        "dense_reference_formula": (
            "dense_recovery_ratio_pe = sparse_mean_pe / dense_mean_pe; "
            "dense_recovery_ratio_exact_match = sparse_exact_match_delta / dense_exact_match_delta; "
            "dense_recovery_ratio_first_entry = sparse_first_entry_delta / dense_first_entry_delta"
        ),
        "dense_recovery_ratio_pe": _safe_ratio(condition_row.get("mean_pe"), dense_ref.get("mean_pe")),
        "dense_recovery_ratio_exact_match": _safe_ratio(sparse_em_delta, dense_em_delta),
        "dense_recovery_ratio_first_entry": _safe_ratio(sparse_entry_delta, dense_entry_delta),
    }


def _mean_raw(results: Iterable[Any], field: str) -> float:
    vals = np.array([_safe_float(getattr(r, field, float("nan"))) for r in results], dtype=np.float64)
    mask = np.isfinite(vals)
    return float(np.mean(vals[mask])) if bool(np.any(mask)) else float("nan")


def _load_stagea_best(stagea_path: Path, seed: int) -> Dict[str, Any]:
    payload = json.loads(stagea_path.read_text(encoding="utf-8"))
    seed_key = str(seed)
    if seed_key not in payload.get("seeds", {}):
        raise KeyError(f"Seed {seed_key} missing from Stage A artifact {stagea_path}")
    best = payload["seeds"][seed_key].get("best") or {}
    if not best or best.get("layer") is None or best.get("topk") is None or not best.get("variant"):
        raise RuntimeError(f"Stage A artifact {stagea_path} has no valid best config for seed {seed_key}")
    return {
        "variant": str(best["variant"]),
        "layer": int(best["layer"]),
        "topk": int(best["topk"]),
        "selection_score": float(best.get("score", float("nan"))),
    }


def _prompt_naming(prompt_meta: Dict[str, str]) -> Tuple[str, str, str]:
    source_language = str(prompt_meta.get("source_language", "")).strip() or "Hindi"
    input_script_name = str(prompt_meta.get("source_script", "")).strip() or "Latin"
    output_script_name = str(prompt_meta.get("target_script", "")).strip() or "Devanagari"
    return source_language, input_script_name, output_script_name


def _load_words(
    pair_id: str,
    *,
    external_only: bool,
    require_external_sources: bool,
    min_pool_size: int,
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    data_path = PROJECT_ROOT / f"data/transliteration/{pair_id}.jsonl"
    meta_path = PROJECT_ROOT / f"data/transliteration/{pair_id}.jsonl.meta.json"
    if not data_path.exists():
        raise RuntimeError(f"Missing workshop data file: {data_path}")
    if not meta_path.exists():
        raise RuntimeError(f"Missing workshop metadata file: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    words: List[Dict[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            row = json.loads(raw)
            target = str(row["target"]).strip()
            source = str(row["source"]).strip()
            pair_key = (target, source)
            if not target or not source or pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            words.append(
                {
                    "english": str(row.get("english", source)).strip(),
                    "hindi": target,
                    "ood": source,
                }
            )

    total = len(words)
    source_names = [str(meta.get("dataset", {}).get("repo_id", "")).strip() or "external_jsonl"]
    external_sources = [n for n in source_names if n and n != "config_multiscript"]
    if bool(require_external_sources) and not external_sources:
        raise RuntimeError(
            f"Pair {pair_id!r} has no external sources (only builtin). "
            "Provide external data under data/transliteration/ or disable --require-external-sources."
        )
    if bool(external_only) and not external_sources:
        raise RuntimeError(f"Pair {pair_id!r} does not satisfy --external-only")
    if int(min_pool_size) > 0 and total < int(min_pool_size):
        raise RuntimeError(f"Pair {pair_id!r} pool too small: total={total} < {int(min_pool_size)}")

    provenance = {
        "pair_id": pair_id,
        "total_rows": total,
        "sources": source_names,
        "meta": meta,
    }
    return words, provenance


def _get_pair_prompt_metadata(pair_id: str) -> Dict[str, str]:
    meta_path = PROJECT_ROOT / f"data/transliteration/{pair_id}.jsonl.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    dataset = dict(meta.get("dataset", {}))
    return {
        "source_language": str(dataset.get("source_language", "Hindi")),
        "source_script": str(dataset.get("source_script", "Latin")),
        "target_script": str(dataset.get("target_script", "Devanagari")),
    }


def _summarize_generation(results: List[Any], *, target_script: str) -> Dict[str, Any]:
    from paper2_fidelity_calibrated.eval_utils import (
        akshara_cer,
        continuation_akshara_cer,
        first_entry_correct,
        normalize_text,
        script_compliance,
    )

    out: Dict[str, Any] = {}
    for prefix, pred_attr in (("zs", "gen_zs"), ("icl", "gen_icl"), ("patched", "gen_patched")):
        ems: List[float] = []
        cers: List[float] = []
        scripts: List[float] = []
        entries: List[float] = []
        conts: List[float] = []
        for r in results:
            gold = normalize_text(getattr(r, "word_hindi", ""))
            pred = normalize_text(getattr(r, pred_attr, ""))
            ems.append(float(pred == gold))
            cers.append(float(akshara_cer(pred, gold)))
            scripts.append(float(script_compliance(pred, target_script)))
            entries.append(float(first_entry_correct(pred, gold)))
            cont = continuation_akshara_cer(pred, gold)
            if np.isfinite(cont):
                conts.append(float(cont))
        out[prefix] = {
            "exact_match": float(np.mean(ems)) if ems else float("nan"),
            "akshara_cer": float(np.mean(cers)) if cers else float("nan"),
            "script_compliance": float(np.mean(scripts)) if scripts else float("nan"),
            "first_entry_correct": float(np.mean(entries)) if entries else float("nan"),
            "continuation_fidelity": float(np.mean(conts)) if conts else float("nan"),
        }
    return out


def _item_generation_metrics(gold_text: str, pred_text: str, *, target_script: str) -> Dict[str, float]:
    from paper2_fidelity_calibrated.eval_utils import (
        akshara_cer,
        continuation_akshara_cer,
        first_entry_correct,
        normalize_text,
        script_compliance,
    )

    gold = normalize_text(gold_text)
    pred = normalize_text(pred_text)
    cont = continuation_akshara_cer(pred, gold)
    return {
        "exact_match": float(pred == gold),
        "akshara_cer": float(akshara_cer(pred, gold)),
        "script_compliance": float(script_compliance(pred, target_script)),
        "first_entry_correct": float(first_entry_correct(pred, gold)),
        "continuation_fidelity": float(cont) if np.isfinite(cont) else float("nan"),
    }


def _item_row(
    *,
    pair_id: str,
    seed: int,
    best: Dict[str, Any],
    locus_label: str,
    geometry_label: str,
    result: Any,
    target_script: str,
) -> Dict[str, Any]:
    zs_metrics = _item_generation_metrics(getattr(result, "word_hindi", ""), getattr(result, "gen_zs", ""), target_script=target_script)
    icl_metrics = _item_generation_metrics(getattr(result, "word_hindi", ""), getattr(result, "gen_icl", ""), target_script=target_script)
    patched_metrics = _item_generation_metrics(getattr(result, "word_hindi", ""), getattr(result, "gen_patched", ""), target_script=target_script)
    return {
        "pair": pair_id,
        "seed": int(seed),
        "language": pair_id.replace("aksharantar_", "").replace("_latin", ""),
        "variant": str(best["variant"]),
        "layer": int(best["layer"]),
        "topk": int(best["topk"]),
        "locus": locus_label,
        "geometry": geometry_label,
        "word_english": str(getattr(result, "word_english", "")),
        "word_target": str(getattr(result, "word_hindi", "")),
        "word_source_romanized": str(getattr(result, "word_telugu", "")),
        "n_input_tokens_ood": int(getattr(result, "n_input_tokens_ood", 0) or 0),
        "fragmentation_bucket": _fragmentation_bucket(getattr(result, "n_input_tokens_ood", float("nan"))),
        "pe_logit": _safe_float(getattr(result, "pe_logit_first", float("nan"))),
        "exact_match_zs": zs_metrics["exact_match"],
        "exact_match_icl": icl_metrics["exact_match"],
        "exact_match_patched": patched_metrics["exact_match"],
        "akshara_cer_zs": zs_metrics["akshara_cer"],
        "akshara_cer_icl": icl_metrics["akshara_cer"],
        "akshara_cer_patched": patched_metrics["akshara_cer"],
        "first_entry_correct_zs": zs_metrics["first_entry_correct"],
        "first_entry_correct_icl": icl_metrics["first_entry_correct"],
        "first_entry_correct_patched": patched_metrics["first_entry_correct"],
        "continuation_fidelity_zs": zs_metrics["continuation_fidelity"],
        "continuation_fidelity_icl": icl_metrics["continuation_fidelity"],
        "continuation_fidelity_patched": patched_metrics["continuation_fidelity"],
        "script_compliance_zs": zs_metrics["script_compliance"],
        "script_compliance_icl": icl_metrics["script_compliance"],
        "script_compliance_patched": patched_metrics["script_compliance"],
        "pe_corrupt_first": _safe_float(getattr(result, "pe_corrupt_first", float("nan"))),
        "pe_shuffle_first": _safe_float(getattr(result, "pe_shuffle_first", float("nan"))),
        "pe_random_first": _safe_float(getattr(result, "pe_random_first", float("nan"))),
        "pe_gauss_first": _safe_float(getattr(result, "pe_gauss_first", float("nan"))),
        "pe_basis_first": _safe_float(getattr(result, "pe_basis_first", float("nan"))),
        "pe_decoupled_first": _safe_float(getattr(result, "pe_decoupled_first", float("nan"))),
        "pe_attention_structured_first": _safe_float(getattr(result, "pe_attention_structured_first", float("nan"))),
        "ae_first": _safe_float(getattr(result, "ae_first", float("nan"))),
        "pe_necessity_first": _safe_float(getattr(result, "pe_necessity_first", float("nan"))),
        "nll_per_token_zs": _safe_float(getattr(result, "nll_per_token_zs", float("nan"))),
        "nll_per_token_icl": _safe_float(getattr(result, "nll_per_token_icl", float("nan"))),
        "nll_per_token_patched": _safe_float(getattr(result, "nll_per_token_patched", float("nan"))),
        "nll_pos1_zs": _safe_float(getattr(result, "nll_pos1_zs", float("nan"))),
        "nll_pos2_zs": _safe_float(getattr(result, "nll_pos2_zs", float("nan"))),
        "nll_pos3_zs": _safe_float(getattr(result, "nll_pos3_zs", float("nan"))),
        "nll_pos1_icl": _safe_float(getattr(result, "nll_pos1_icl", float("nan"))),
        "nll_pos2_icl": _safe_float(getattr(result, "nll_pos2_icl", float("nan"))),
        "nll_pos3_icl": _safe_float(getattr(result, "nll_pos3_icl", float("nan"))),
        "nll_pos1_patched": _safe_float(getattr(result, "nll_pos1_patched", float("nan"))),
        "nll_pos2_patched": _safe_float(getattr(result, "nll_pos2_patched", float("nan"))),
        "nll_pos3_patched": _safe_float(getattr(result, "nll_pos3_patched", float("nan"))),
        "hookpoint_fidelity_cosine": _safe_float(getattr(result, "reconstruction_cosine_icl", float("nan"))),
        "hookpoint_fidelity_rel_error": _safe_float(getattr(result, "reconstruction_rel_error_icl", float("nan"))),
        "indic_span_fidelity_cosine": _safe_float(getattr(result, "reconstruction_cosine_icl", float("nan"))),
        "indic_span_fidelity_rel_error": _safe_float(getattr(result, "reconstruction_rel_error_icl", float("nan"))),
        "selected_feature_indices": str(getattr(result, "selected_feature_indices", "")),
        "selected_feature_magnitudes_raw": list(getattr(result, "selected_feature_magnitudes_raw", []) or []),
        "patch_geometry_clip_cap_used": _safe_float(getattr(result, "patch_geometry_clip_cap_used", float("nan"))),
        "patch_geometry_sign_scale_used": _safe_float(getattr(result, "patch_geometry_sign_scale_used", float("nan"))),
        "patch_geometry_fraction_clipped": _safe_float(getattr(result, "patch_geometry_fraction_clipped", float("nan"))),
        "latent_patch_norm_pre_geometry": _safe_float(getattr(result, "latent_patch_norm_pre_geometry", float("nan"))),
        "latent_patch_norm_post_geometry": _safe_float(getattr(result, "latent_patch_norm_post_geometry", float("nan"))),
        "decoded_patch_norm_pre_geometry": _safe_float(getattr(result, "decoded_patch_norm_pre_geometry", float("nan"))),
        "decoded_patch_norm_post_geometry": _safe_float(getattr(result, "decoded_patch_norm_post_geometry", float("nan"))),
    }


def _fragmentation_summary_rows(item_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    grouped: Dict[Tuple[str, int, str, str, str], List[Dict[str, Any]]] = {}
    for row in item_rows:
        key = (row["pair"], int(row["seed"]), str(row["locus"]), str(row["geometry"]), str(row["fragmentation_bucket"]))
        grouped.setdefault(key, []).append(row)
    for (pair, seed, locus, geometry, bucket), subset in sorted(grouped.items()):
        rows.append(
            {
                "pair": pair,
                "seed": seed,
                "locus": locus,
                "geometry": geometry,
                "fragmentation_bucket": bucket,
                "n_items": len(subset),
                "mean_pe_logit": float(np.mean([_safe_float(r["pe_logit"]) for r in subset if np.isfinite(_safe_float(r["pe_logit"]))])),
                "exact_match_patched": float(np.mean([_safe_float(r["exact_match_patched"]) for r in subset])),
                "akshara_cer_patched": float(np.mean([_safe_float(r["akshara_cer_patched"]) for r in subset])),
                "first_entry_correct_patched": float(np.mean([_safe_float(r["first_entry_correct_patched"]) for r in subset])),
                "continuation_fidelity_patched": float(np.nanmean(np.array([_safe_float(r["continuation_fidelity_patched"]) for r in subset], dtype=np.float64))),
            }
        )
    return rows


def _geometry_diagnostic_rows(item_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    grouped: Dict[Tuple[str, int, str, str], List[Dict[str, Any]]] = {}
    for row in item_rows:
        key = (row["pair"], int(row["seed"]), str(row["locus"]), str(row["geometry"]))
        grouped.setdefault(key, []).append(row)
    for (pair, seed, locus, geometry), subset in sorted(grouped.items()):
        raw_vals = [
            abs(float(v))
            for row in subset
            for v in (row.get("selected_feature_magnitudes_raw") or [])
            if np.isfinite(_safe_float(v))
        ]
        rows.append(
            {
                "pair": pair,
                "seed": seed,
                "locus": locus,
                "geometry": geometry,
                "n_items": len(subset),
                "clip_cap_used": _safe_float(subset[0].get("patch_geometry_clip_cap_used")),
                "normalized_sign_scale_used": _safe_float(subset[0].get("patch_geometry_sign_scale_used")),
                "mean_fraction_clipped": float(np.nanmean(np.array([_safe_float(r.get("patch_geometry_fraction_clipped")) for r in subset], dtype=np.float64))),
                "mean_raw_selected_feature_magnitude": float(np.mean(raw_vals)) if raw_vals else float("nan"),
                "max_raw_selected_feature_magnitude": float(np.max(raw_vals)) if raw_vals else float("nan"),
                "mean_latent_patch_norm_pre_geometry": float(np.nanmean(np.array([_safe_float(r.get("latent_patch_norm_pre_geometry")) for r in subset], dtype=np.float64))),
                "mean_latent_patch_norm_post_geometry": float(np.nanmean(np.array([_safe_float(r.get("latent_patch_norm_post_geometry")) for r in subset], dtype=np.float64))),
                "mean_decoded_patch_norm_pre_geometry": float(np.nanmean(np.array([_safe_float(r.get("decoded_patch_norm_pre_geometry")) for r in subset], dtype=np.float64))),
                "mean_decoded_patch_norm_post_geometry": float(np.nanmean(np.array([_safe_float(r.get("decoded_patch_norm_post_geometry")) for r in subset], dtype=np.float64))),
            }
        )
    return rows


def _condition_row(
    *,
    pair_id: str,
    seed: int,
    best: Dict[str, Any],
    locus_label: str,
    geometry_label: str,
    stats: Dict[str, Any],
    generation: Dict[str, Any],
    geometry_params: Dict[str, Any],
    dense_ref: Dict[str, Any] | None,
) -> Dict[str, Any]:
    controls = [
        _safe_float(stats.get("mean_pe_corrupt")),
        _safe_float(stats.get("mean_pe_shuffle")),
        _safe_float(stats.get("mean_pe_gauss")),
        _safe_float(stats.get("mean_pe_basis")),
    ]
    finite_controls = [x for x in controls if np.isfinite(x)]
    max_control = max(finite_controls) if finite_controls else float("nan")
    mean_pe = _safe_float(stats.get("mean_pe"))
    mismatch = _safe_float(stats.get("mean_pe_decoupled"))
    row = {
        "pair": pair_id,
        "seed": int(seed),
        "language": pair_id.replace("aksharantar_", "").replace("_latin", ""),
        "variant": str(best["variant"]),
        "layer": int(best["layer"]),
        "topk": int(best["topk"]),
        "selection_score": _safe_float(best.get("selection_score")),
        "locus": locus_label,
        "geometry": geometry_label,
        "clip_cap": _safe_float(geometry_params.get("clip_cap")),
        "sign_scale": _safe_float(geometry_params.get("sign_scale")),
        "geometry_selection_count": int(geometry_params.get("n_selected_values", 0)),
        "mean_pe": mean_pe,
        "ci_pe_low": _safe_float((stats.get("ci_pe_95") or [float("nan"), float("nan")])[0]),
        "ci_pe_high": _safe_float((stats.get("ci_pe_95") or [float("nan"), float("nan")])[1]),
        "mean_pe_logit": _safe_float(stats.get("mean_pe_logit")),
        "mean_pe_corrupt": _safe_float(stats.get("mean_pe_corrupt")),
        "mean_pe_shuffle": _safe_float(stats.get("mean_pe_shuffle")),
        "mean_pe_gauss": _safe_float(stats.get("mean_pe_gauss")),
        "mean_pe_basis": _safe_float(stats.get("mean_pe_basis")),
        "mean_pe_decoupled": mismatch,
        "specificity_margin_vs_controls": mean_pe - max_control if np.isfinite(mean_pe) and np.isfinite(max_control) else float("nan"),
        "specificity_margin_vs_mismatch": mean_pe - mismatch if np.isfinite(mean_pe) and np.isfinite(mismatch) else float("nan"),
        "mean_nll_per_token_zs": _safe_float(stats.get("mean_nll_per_token_zs")),
        "mean_nll_per_token_icl": _safe_float(stats.get("mean_nll_per_token_icl")),
        "mean_nll_per_token_patched": _safe_float(stats.get("mean_nll_per_token_patched")),
        "mean_nll_improvement_patch": _safe_float(stats.get("mean_nll_improvement_patch")),
        "mean_nll_target_pos1_zs": _safe_float(stats.get("mean_nll_target_pos1_zs")),
        "mean_nll_target_pos1_icl": _safe_float(stats.get("mean_nll_target_pos1_icl")),
        "mean_nll_target_pos1_patched": _safe_float(stats.get("mean_nll_target_pos1_patched")),
        "mean_nll_target_pos2_zs": _safe_float(stats.get("mean_nll_target_pos2_zs")),
        "mean_nll_target_pos2_icl": _safe_float(stats.get("mean_nll_target_pos2_icl")),
        "mean_nll_target_pos2_patched": _safe_float(stats.get("mean_nll_target_pos2_patched")),
        "mean_nll_target_pos3_zs": _safe_float(stats.get("mean_nll_target_pos3_zs")),
        "mean_nll_target_pos3_icl": _safe_float(stats.get("mean_nll_target_pos3_icl")),
        "mean_nll_target_pos3_patched": _safe_float(stats.get("mean_nll_target_pos3_patched")),
        "hookpoint_fidelity_cosine": _safe_float(stats.get("mean_reconstruction_cosine_icl")),
        "hookpoint_fidelity_rel_error": _safe_float(stats.get("mean_reconstruction_rel_error_icl")),
        "indic_span_fidelity_cosine": _safe_float(stats.get("mean_reconstruction_cosine_icl")),
        "indic_span_fidelity_rel_error": _safe_float(stats.get("mean_reconstruction_rel_error_icl")),
        "softcap_saturation_risk": bool(stats.get("softcap_saturation_risk", False)),
        "selected_outlier_gt5sigma_rate": _safe_float(stats.get("selected_outlier_gt5sigma_rate")),
        "query_span_success_rate_zs": _safe_float(stats.get("query_span_success_rate_zs")),
        "query_span_success_rate_icl": _safe_float(stats.get("query_span_success_rate_icl")),
        "query_span_success_rate_selector_reference": _safe_float(stats.get("query_span_success_rate_selector_reference")),
        "exact_match_zs": _safe_float(generation["zs"].get("exact_match")),
        "exact_match_icl": _safe_float(generation["icl"].get("exact_match")),
        "exact_match_patched": _safe_float(generation["patched"].get("exact_match")),
        "akshara_cer_zs": _safe_float(generation["zs"].get("akshara_cer")),
        "akshara_cer_icl": _safe_float(generation["icl"].get("akshara_cer")),
        "akshara_cer_patched": _safe_float(generation["patched"].get("akshara_cer")),
        "script_compliance_zs": _safe_float(generation["zs"].get("script_compliance")),
        "script_compliance_icl": _safe_float(generation["icl"].get("script_compliance")),
        "script_compliance_patched": _safe_float(generation["patched"].get("script_compliance")),
        "first_entry_correct_zs": _safe_float(generation["zs"].get("first_entry_correct")),
        "first_entry_correct_icl": _safe_float(generation["icl"].get("first_entry_correct")),
        "first_entry_correct_patched": _safe_float(generation["patched"].get("first_entry_correct")),
        "continuation_fidelity_zs": _safe_float(generation["zs"].get("continuation_fidelity")),
        "continuation_fidelity_icl": _safe_float(generation["icl"].get("continuation_fidelity")),
        "continuation_fidelity_patched": _safe_float(generation["patched"].get("continuation_fidelity")),
    }
    row.update(_dense_recovery_fields(row, dense_ref))
    return row


def _necessity_row(base_row: Dict[str, Any], stats: Dict[str, Any], results: List[Any]) -> Dict[str, Any]:
    row = {k: base_row[k] for k in ("pair", "seed", "language", "variant", "layer", "topk", "locus", "geometry")}
    row.update(
        {
            "mean_ae": _safe_float(stats.get("mean_ae")),
            "ci_ae_low": _safe_float((stats.get("ci_ae_95") or [float("nan"), float("nan")])[0]),
            "ci_ae_high": _safe_float((stats.get("ci_ae_95") or [float("nan"), float("nan")])[1]),
            "negative_ae_rate": _safe_float(stats.get("negative_ae_rate")),
            "mean_nll_per_token_ablated": _mean_raw(results, "nll_per_token_ablated"),
            "mean_pe_necessity_first": _mean_raw(results, "pe_necessity_first"),
            "mean_pe_necessity_multi": _mean_raw(results, "pe_necessity_multi"),
            "mean_prob_patch_then_ablate_first": _mean_raw(results, "prob_patch_then_ablate_first"),
            "mean_prob_patch_then_ablate_multi": _mean_raw(results, "prob_patch_then_ablate_multi"),
        }
    )
    return row


def _entry_continuation_row(base_row: Dict[str, Any]) -> Dict[str, Any]:
    keep = [
        "pair", "seed", "language", "variant", "layer", "topk", "locus", "geometry",
        "first_entry_correct_zs", "first_entry_correct_icl", "first_entry_correct_patched",
        "continuation_fidelity_zs", "continuation_fidelity_icl", "continuation_fidelity_patched",
        "mean_nll_target_pos1_zs", "mean_nll_target_pos1_icl", "mean_nll_target_pos1_patched",
        "mean_nll_target_pos2_zs", "mean_nll_target_pos2_icl", "mean_nll_target_pos2_patched",
        "mean_nll_target_pos3_zs", "mean_nll_target_pos3_icl", "mean_nll_target_pos3_patched",
    ]
    return {k: base_row.get(k) for k in keep}


def _supports_positive_narrow(stats: Dict[str, Any], fairness: Dict[str, Any] | None) -> bool:
    mean_pe = _safe_float(stats.get("mean_pe"))
    ci = stats.get("ci_pe_95", [float("nan"), float("nan")])
    ci_low = _safe_float(ci[0]) if isinstance(ci, list) and len(ci) == 2 else float("nan")
    controls = [
        _safe_float(stats.get("mean_pe_corrupt")),
        _safe_float(stats.get("mean_pe_shuffle")),
        _safe_float(stats.get("mean_pe_gauss")),
        _safe_float(stats.get("mean_pe_basis")),
    ]
    control_max = max(v for v in controls if np.isfinite(v)) if any(np.isfinite(v) for v in controls) else float("nan")
    fair = _safe_float((fairness or {}).get("mean_pe"))
    return bool(
        np.isfinite(mean_pe)
        and np.isfinite(ci_low)
        and ci_low > 0.0
        and (not np.isfinite(control_max) or mean_pe > control_max)
        and (not np.isfinite(fair) or mean_pe > fair)
    )


def _supports_fidelity_limited(stats: Dict[str, Any]) -> bool:
    mean_pe = _safe_float(stats.get("mean_pe"))
    pos_rate = _safe_float(stats.get("positive_pe_rate"))
    entry_patch = _safe_float(stats.get("mean_nll_target_pos1_patched"))
    entry_zs = _safe_float(stats.get("mean_nll_target_pos1_zs"))
    return bool(
        np.isfinite(mean_pe)
        and mean_pe > 0.0
        and np.isfinite(pos_rate)
        and pos_rate >= 0.60
        and np.isfinite(entry_patch)
        and np.isfinite(entry_zs)
        and entry_patch < entry_zs
    )


def _fairness_row(*, pair_id: str, seed: int, best: Dict[str, Any], locus_label: str, geometry_label: str, stats: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "pair": pair_id,
        "seed": int(seed),
        "language": pair_id.replace("aksharantar_", "").replace("_latin", ""),
        "variant": str(best["variant"]),
        "layer": int(best["layer"]),
        "topk": int(best["topk"]),
        "locus": locus_label,
        "geometry": geometry_label,
        "mean_pe_attention_structured": _safe_float(stats.get("mean_pe_attention_structured")),
        "mean_nll_per_token_attention_structured": _safe_float(stats.get("mean_nll_per_token_attention_structured")),
    }


def _condition_supports_progress(row: Dict[str, Any], necessity_row: Dict[str, Any]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    geometry_ok = str(row.get("geometry")) in {"clipped", "normalized_sign"}
    if not geometry_ok:
        reasons.append("geometry not bounded")

    mean_pe = _safe_float(row.get("mean_pe"))
    control_candidates = [
        _safe_float(row.get("mean_pe_corrupt")),
        _safe_float(row.get("mean_pe_shuffle")),
        _safe_float(row.get("mean_pe_gauss")),
        _safe_float(row.get("mean_pe_basis")),
    ]
    finite_controls = [x for x in control_candidates if np.isfinite(x)]
    max_control = max(finite_controls) if finite_controls else float("nan")
    mismatch = _safe_float(row.get("mean_pe_decoupled"))
    patch_specific = bool(
        np.isfinite(mean_pe)
        and mean_pe > 0.0
        and (not np.isfinite(max_control) or mean_pe > max_control)
        and (not np.isfinite(mismatch) or mean_pe > mismatch)
    )
    if not patch_specific:
        reasons.append("patch not more specific than controls/mismatch")

    ae = _safe_float(necessity_row.get("mean_ae"))
    neg_rate = _safe_float(necessity_row.get("negative_ae_rate"))
    necessity_ok = bool(np.isfinite(ae) and ae < 0.0 and np.isfinite(neg_rate) and neg_rate >= 0.50)
    if not necessity_ok:
        reasons.append("ablation does not show clear necessity")

    entry_delta = _safe_float(row.get("first_entry_correct_patched")) - _safe_float(row.get("first_entry_correct_zs"))
    cont_delta = _safe_float(row.get("continuation_fidelity_zs")) - _safe_float(row.get("continuation_fidelity_patched"))
    nll_pos1_delta = _safe_float(row.get("mean_nll_target_pos1_zs")) - _safe_float(row.get("mean_nll_target_pos1_patched"))
    nll_pos2_delta = _safe_float(row.get("mean_nll_target_pos2_zs")) - _safe_float(row.get("mean_nll_target_pos2_patched"))
    nll_pos3_delta = _safe_float(row.get("mean_nll_target_pos3_zs")) - _safe_float(row.get("mean_nll_target_pos3_patched"))
    entry_or_cont = bool(
        (np.isfinite(entry_delta) and entry_delta > 0.0)
        or (np.isfinite(nll_pos1_delta) and nll_pos1_delta > 0.0)
        or (np.isfinite(cont_delta) and cont_delta > 0.0)
        or (np.isfinite(nll_pos2_delta) and nll_pos2_delta > 0.0)
        or (np.isfinite(nll_pos3_delta) and nll_pos3_delta > 0.0)
    )
    if not entry_or_cont:
        reasons.append("no entry/continuation rescue signal")

    fidelity_ok = bool(
        np.isfinite(_safe_float(row.get("hookpoint_fidelity_cosine")))
        and np.isfinite(_safe_float(row.get("hookpoint_fidelity_rel_error")))
    )
    if not fidelity_ok:
        reasons.append("missing fidelity diagnostics")

    ok = geometry_ok and patch_specific and necessity_ok and entry_or_cont and fidelity_ok
    return ok, reasons


def _signal_interpretation(row: Dict[str, Any]) -> str:
    mean_pe = _safe_float(row.get("mean_pe"))
    controls = [
        _safe_float(row.get("mean_pe_corrupt")),
        _safe_float(row.get("mean_pe_shuffle")),
        _safe_float(row.get("mean_pe_gauss")),
        _safe_float(row.get("mean_pe_basis")),
    ]
    finite_controls = [x for x in controls if np.isfinite(x)]
    max_control = max(finite_controls) if finite_controls else float("nan")
    mismatch = _safe_float(row.get("mean_pe_decoupled"))
    if not (
        np.isfinite(mean_pe)
        and mean_pe > 0.0
        and (not np.isfinite(max_control) or mean_pe > max_control)
        and (not np.isfinite(mismatch) or mean_pe > mismatch)
    ):
        return "non-specific / control-matched"

    entry_delta = _safe_float(row.get("first_entry_correct_patched")) - _safe_float(row.get("first_entry_correct_zs"))
    nll_pos1_delta = _safe_float(row.get("mean_nll_target_pos1_zs")) - _safe_float(row.get("mean_nll_target_pos1_patched"))
    cont_delta = _safe_float(row.get("continuation_fidelity_zs")) - _safe_float(row.get("continuation_fidelity_patched"))
    nll_pos2_delta = _safe_float(row.get("mean_nll_target_pos2_zs")) - _safe_float(row.get("mean_nll_target_pos2_patched"))
    nll_pos3_delta = _safe_float(row.get("mean_nll_target_pos3_zs")) - _safe_float(row.get("mean_nll_target_pos3_patched"))

    entry_ok = bool((np.isfinite(entry_delta) and entry_delta > 0.0) or (np.isfinite(nll_pos1_delta) and nll_pos1_delta > 0.0))
    cont_ok = bool(
        (np.isfinite(cont_delta) and cont_delta > 0.0)
        or (np.isfinite(nll_pos2_delta) and nll_pos2_delta > 0.0)
        or (np.isfinite(nll_pos3_delta) and nll_pos3_delta > 0.0)
    )
    if entry_ok and cont_ok:
        return "both"
    if entry_ok:
        return "entry-only"
    if cont_ok:
        return "early-continuation"
    return "non-specific / control-matched"


def _decision_note(
    condition_rows: List[Dict[str, Any]],
    necessity_rows: List[Dict[str, Any]],
    *,
    incomplete: bool = False,
    missing_artifacts: List[str] | None = None,
    force_layer: int | None = None,
) -> Dict[str, Any]:
    if incomplete:
        return {
            "status": "incomplete",
            "run_complete": False,
            "missing_artifacts": list(missing_artifacts or []),
            "proceed_to_4b_stageb": "no",
            "next_step": "fix_incomplete_run",
            "rationale": "Stage A.5 run is incomplete; required artifacts are missing, so no normal decision note is emitted.",
            "supporting_conditions": [],
            "rejected_conditions": [],
        }

    necessity_index = {
        (row["pair"], row["seed"], row["locus"], row["geometry"]): row for row in necessity_rows
    }
    supporting: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for row in condition_rows:
        key = (row["pair"], row["seed"], row["locus"], row["geometry"])
        nec = necessity_index[key]
        ok, reasons = _condition_supports_progress(row, nec)
        record = {
            "pair": row["pair"],
            "seed": row["seed"],
            "locus": row["locus"],
            "geometry": row["geometry"],
            "mean_pe": row["mean_pe"],
            "mean_ae": nec["mean_ae"],
            "dense_recovery_ratio_pe": _safe_float(row.get("dense_recovery_ratio_pe")),
            "dense_recovery_ratio_exact_match": _safe_float(row.get("dense_recovery_ratio_exact_match")),
            "dense_recovery_ratio_first_entry": _safe_float(row.get("dense_recovery_ratio_first_entry")),
            "signal_interpretation": _signal_interpretation(row),
            "reasons": reasons,
        }
        if ok:
            supporting.append(record)
        else:
            rejected.append(record)
    proceed = bool(supporting)
    best_sparse = max(condition_rows, key=lambda r: _safe_float(r.get("mean_pe"))) if condition_rows else None
    rationale = (
        "At least one bounded Stage A.5 condition on the anchor set shows coherent patch/ablation behavior with better specificity than corrupt/shuffle/random-feature controls and mismatch."
        if proceed
        else "No bounded Stage A.5 condition on the anchor set produced target-specific patch/ablation evidence stronger than skeptical controls; do not force a positive mechanistic paper."
    )
    if not proceed and force_layer is not None:
        dense_clause = ""
        if best_sparse is not None and np.isfinite(_safe_float(best_sparse.get("dense_recovery_ratio_pe"))):
            dense_clause = (
                f" Dense recovery was limited "
                f"(PE={_safe_float(best_sparse.get('dense_recovery_ratio_pe')):.3f}, "
                f"EM={_safe_float(best_sparse.get('dense_recovery_ratio_exact_match')):.3f}, "
                f"entry={_safe_float(best_sparse.get('dense_recovery_ratio_first_entry')):.3f})."
            )
        rationale = (
            f"Final sparse retry at forced layer {int(force_layer)} remained weak or control-matched. "
            "Close the local MLP-positive path and pivot to a skeptical mechanism-validity framing, "
            "with bounded attention analysis as the only next mechanistic step."
            + dense_clause
        )
    return {
        "status": "complete",
        "run_complete": True,
        "proceed_to_4b_stageb": "yes" if proceed else "no",
        "close_local_mlp_positive_path": "no" if proceed else ("yes" if force_layer is not None else "no"),
        "next_step": "proceed_to_4b_stageb" if proceed else ("bounded_attention_analysis" if force_layer is not None else "do_not_force_positive_mechanistic_paper"),
        "rationale": rationale,
        "supporting_conditions": supporting,
        "rejected_conditions": rejected,
        "dense_recovery_interpretation": (
            "Dense-recovery ratios are diagnostic only. They are interpreted jointly with specificity vs controls, "
            "zero-ablation necessity, entry/continuation structure, and confidence interval quality."
        ),
    }


def _required_artifact_paths(out_root: Path, pairs: List[str]) -> List[Path]:
    paths = [
        out_root / "stagea5_config.json",
        out_root / "stagea5_summary.json",
        out_root / "locus_comparison_table.json",
        out_root / "locus_comparison_table.csv",
        out_root / "geometry_comparison_table.json",
        out_root / "geometry_comparison_table.csv",
        out_root / "geometry_diagnostics_table.json",
        out_root / "geometry_diagnostics_table.csv",
        out_root / "necessity_ablation_table.json",
        out_root / "necessity_ablation_table.csv",
        out_root / "entry_vs_continuation_table.json",
        out_root / "entry_vs_continuation_table.csv",
        out_root / "tiny_attn_out_fairness_table.json",
        out_root / "tiny_attn_out_fairness_table.csv",
        out_root / "item_level_results.json",
        out_root / "item_level_results.csv",
        out_root / "fragmentation_bucket_table.json",
        out_root / "fragmentation_bucket_table.csv",
    ]
    for pair in pairs:
        paths.extend(
            [
                out_root / f"{pair}_results_table.json",
                out_root / f"{pair}_results_table.csv",
            ]
        )
    return paths


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run Stage A.5 diagnostics on anchor languages only.")
    ap.add_argument("--model", type=str, default="4b", choices=["4b"])
    ap.add_argument("--pairs", type=str, default="aksharantar_hin_latin,aksharantar_tel_latin")
    ap.add_argument("--seeds", type=str, default="42")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--patch-position-modes", type=str, default="source_last_subtoken,target_pos1_teacher_forced")
    ap.add_argument("--patch-geometries", type=str, default="raw,clipped,normalized_sign")
    ap.add_argument("--external-only", action="store_true")
    ap.add_argument("--require-external-sources", action="store_true")
    ap.add_argument("--min-pool-size", type=int, default=500)
    ap.add_argument("--eval-generation", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--force-layer", type=int, default=-1, help="Override Stage A best layer with a fixed layer for bounded sparse retry.")
    ap.add_argument("--out", type=str, default="")
    return ap.parse_args()


def main() -> int:
    from config import get_model_config
    from core import (
        compute_statistics,
        estimate_patch_geometry_params,
        get_layer_device,
        load_model,
        load_transcoder,
        run_patching_experiment,
        save_json,
        split_data_three_way,
    )
    from paper2_fidelity_calibrated.protocol_utils import prompt_template_fingerprint, runtime_identity

    args = parse_args()
    model_key = str(args.model)
    pairs = _parse_csv(args.pairs)
    seeds = [int(x) for x in _parse_csv(args.seeds)]
    position_modes = [_canonical_position_mode(x) for x in _parse_csv(args.patch_position_modes)]
    geometries = [_canonical_geometry_mode(x) for x in _parse_csv(args.patch_geometries)]

    cfg = get_model_config(model_key)
    model, tokenizer = load_model(model_key, device=str(args.device))

    results_root = Path(__file__).resolve().parent / "results"
    out_root = Path(args.out).resolve() if str(args.out).strip() else results_root / "stagea5"
    out_root.mkdir(parents=True, exist_ok=True)

    config_payload: Dict[str, Any] = {
        "paper": "paper2_stagea5_diagnostic",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_key": model_key,
        "runtime_identity": runtime_identity(model_key=model_key, hf_id=cfg.hf_id, tokenizer=tokenizer, model=model),
        "prompt_template_fingerprint": prompt_template_fingerprint(tokenizer),
        "git_commit_hash": _git_commit_hash(),
        "config": {
            "pairs": pairs,
            "seeds": seeds,
            "force_layer": (int(args.force_layer) if int(args.force_layer) >= 0 else None),
            "patch_position_modes": [label for label, _ in position_modes],
            "patch_geometries": [label for label, _ in geometries],
            "eval_generation": bool(args.eval_generation),
            "max_new_tokens": int(args.max_new_tokens),
            "selector_reference": "corrupt_icl",
            "feature_selection": "topk_abs_delta",
            "claim_level": "intervention_only",
            "fairness_pair": pairs[0] if pairs else None,
            "dense_recovery_formula": (
                "dense_recovery_ratio_pe = sparse_mean_pe / dense_mean_pe; "
                "dense_recovery_ratio_exact_match = sparse_exact_match_delta / dense_exact_match_delta; "
                "dense_recovery_ratio_first_entry = sparse_first_entry_delta / dense_first_entry_delta"
            ),
            "reproducibility": {
                "run_seeds": seeds,
                "control_seed_policy": "deterministic local RNG derived from run seed, layer, topk, and word identity",
                "mismatch_sampling_policy": "cyclic_next_eval_item",
            },
        },
        "artifacts": {},
        "dense_reference": {},
    }
    _write_json(out_root / "stagea5_config.json", config_payload)

    condition_rows: List[Dict[str, Any]] = []
    necessity_rows: List[Dict[str, Any]] = []
    entry_rows: List[Dict[str, Any]] = []
    fairness_rows: List[Dict[str, Any]] = []
    item_rows: List[Dict[str, Any]] = []
    pair_tables: Dict[str, List[Dict[str, Any]]] = {pair: [] for pair in pairs}
    summary_pairs: Dict[str, Any] = {}

    for pair_id in pairs:
        log(f"Stage A.5 pair={pair_id}")
        words, provenance = _load_words(
            pair_id,
            external_only=bool(args.external_only),
            require_external_sources=bool(args.require_external_sources),
            min_pool_size=int(args.min_pool_size),
        )
        prompt_meta = _get_pair_prompt_metadata(pair_id)
        source_language, input_script_name, output_script_name = _prompt_naming(prompt_meta)
        pair_summary: Dict[str, Any] = {
            "pair_meta": dict(prompt_meta),
            "provenance": provenance,
            "seeds": {},
        }

        for seed in seeds:
            stagea_path = results_root / f"{pair_id}/{model_key}/paper2_fidelity_calibrated_{model_key}.json"
            best = _load_stagea_best(stagea_path, int(seed))
            if int(args.force_layer) >= 0:
                best = dict(best)
                best["layer"] = int(args.force_layer)
            dense_ref = _load_dense_reference_row(pair_id, model_key, int(seed), int(best["layer"]))
            icl, sel, ev = split_data_three_way(words=words, n_icl=64, n_select=300, n_eval=200, seed=int(seed))
            tc = load_transcoder(
                model,
                cfg.scope_repo,
                int(best["layer"]),
                get_layer_device(model, int(best["layer"])),
                variant=str(best["variant"]),
            )

            pair_summary["seeds"][str(seed)] = {
                "stagea_best": best,
                "artifact": dict(getattr(tc, "load_info", {}) or {}),
                "geometry_params": {},
                "conditions": {},
                "fairness": {},
            }
            config_payload["artifacts"].setdefault(pair_id, {})[str(seed)] = dict(getattr(tc, "load_info", {}) or {})
            config_payload["dense_reference"].setdefault(pair_id, {})[str(seed)] = {
                "available": bool(dense_ref),
                "layer": int(best["layer"]),
                "reference": _json_safe(dense_ref) if dense_ref else None,
            }

            decoupled_words = [ev[(i + 1) % len(ev)] for i in range(len(ev))] if ev else []
            fairness_enabled = pair_id == pairs[0]

            for locus_label, locus_internal in position_modes:
                geom_params = estimate_patch_geometry_params(
                    model,
                    tokenizer,
                    tc,
                    int(best["layer"]),
                    sel,
                    icl_examples=icl,
                    topk=int(best["topk"]),
                    device=str(args.device),
                    input_script_name=input_script_name,
                    source_language=source_language,
                    output_script_name=output_script_name,
                    patch_style="sparse",
                    feature_selection="topk_abs_delta",
                    selector_reference_mode="corrupt_icl",
                    prompt_variant="canonical",
                    patch_position_mode="target_pos1" if locus_internal == "target_pos1_teacher_forced" else locus_internal,
                )
                pair_summary["seeds"][str(seed)]["geometry_params"][locus_label] = geom_params

                for geometry_label, geometry_internal in geometries:
                    key = f"{locus_label}__{geometry_label}"
                    log(f"Seed {seed}: {pair_id} {key}")
                    results = []
                    for word, mismatch_word in zip(ev, decoupled_words):
                        results.append(
                            run_patching_experiment(
                                model,
                                tokenizer,
                                tc,
                                int(best["layer"]),
                                word,
                                icl_examples=icl,
                                topk=int(best["topk"]),
                                device=str(args.device),
                                seed=int(seed),
                                input_script_name=input_script_name,
                                source_language=source_language,
                                output_script_name=output_script_name,
                                patch_style="sparse",
                                feature_selection="topk_abs_delta",
                                selector_reference_mode="corrupt_icl",
                                require_query_span_match=True,
                                use_norm_matching=True,
                                prompt_variant="canonical",
                                eval_generation=bool(args.eval_generation),
                                max_new_tokens=int(args.max_new_tokens),
                                patch_position_mode=locus_internal,
                                patch_geometry=geometry_internal,
                                patch_geometry_clip_cap=_safe_float(geom_params.get("clip_cap")),
                                patch_geometry_sign_scale=_safe_float(geom_params.get("sign_scale")),
                                decoupled_word=mismatch_word,
                            )
                        )
                    stats = compute_statistics(results)
                    generation = _summarize_generation(results, target_script=output_script_name)
                    row = _condition_row(
                        pair_id=pair_id,
                        seed=int(seed),
                        best=best,
                        locus_label=locus_label,
                        geometry_label=geometry_label,
                        stats=stats,
                        generation=generation,
                        geometry_params=geom_params,
                        dense_ref=dense_ref,
                    )
                    pair_tables[pair_id].append(row)
                    condition_rows.append(row)
                    necessity_rows.append(_necessity_row(row, stats, results))
                    entry_rows.append(_entry_continuation_row(row))
                    item_rows.extend(
                        _item_row(
                            pair_id=pair_id,
                            seed=int(seed),
                            best=best,
                            locus_label=locus_label,
                            geometry_label=geometry_label,
                            result=r,
                            target_script=output_script_name,
                        )
                        for r in results
                    )
                    pair_summary["seeds"][str(seed)]["conditions"][key] = {
                        "stats": stats,
                        "generation": generation,
                        "row": row,
                    }

                    if fairness_enabled:
                        fairness_results = []
                        for word in ev:
                            fairness_results.append(
                                run_patching_experiment(
                                    model,
                                    tokenizer,
                                    tc,
                                    int(best["layer"]),
                                    word,
                                    icl_examples=icl,
                                    topk=int(best["topk"]),
                                    device=str(args.device),
                                    seed=int(seed),
                                    input_script_name=input_script_name,
                                    source_language=source_language,
                                    output_script_name=output_script_name,
                                    patch_style="sparse",
                                    feature_selection="topk_abs_delta",
                                    selector_reference_mode="corrupt_icl",
                                    require_query_span_match=True,
                                    use_norm_matching=True,
                                    prompt_variant="canonical",
                                    eval_generation=False,
                                    patch_position_mode=locus_internal,
                                    patch_geometry=geometry_internal,
                                    patch_geometry_clip_cap=_safe_float(geom_params.get("clip_cap")),
                                    patch_geometry_sign_scale=_safe_float(geom_params.get("sign_scale")),
                                    control_mode="attention_structured",
                                )
                            )
                        fairness_stats = compute_statistics(fairness_results)
                        fairness_row = _fairness_row(
                            pair_id=pair_id,
                            seed=int(seed),
                            best=best,
                            locus_label=locus_label,
                            geometry_label=geometry_label,
                            stats=fairness_stats,
                        )
                        fairness_rows.append(fairness_row)
                        pair_summary["seeds"][str(seed)]["fairness"][key] = fairness_row

        summary_pairs[pair_id] = pair_summary
        _write_rows(out_root / f"{pair_id}_results_table", pair_tables[pair_id])

    locus_rows: List[Dict[str, Any]] = []
    for pair in pairs:
        for locus_label, _ in position_modes:
            subset = [r for r in condition_rows if r["pair"] == pair and r["locus"] == locus_label]
            if not subset:
                continue
            best_row = max(subset, key=lambda r: _safe_float(r.get("mean_pe")))
            locus_rows.append(
                {
                    "pair": pair,
                    "locus": locus_label,
                    "n_conditions": len(subset),
                    "mean_of_mean_pe": float(np.mean([_safe_float(r.get("mean_pe")) for r in subset if np.isfinite(_safe_float(r.get("mean_pe")))])),
                    "best_geometry": best_row["geometry"],
                    "best_geometry_mean_pe": best_row["mean_pe"],
                    "best_geometry_entry_delta": _safe_float(best_row.get("first_entry_correct_patched")) - _safe_float(best_row.get("first_entry_correct_zs")),
                    "best_geometry_continuation_delta": _safe_float(best_row.get("continuation_fidelity_zs")) - _safe_float(best_row.get("continuation_fidelity_patched")),
                }
            )

    geometry_rows: List[Dict[str, Any]] = []
    for pair in pairs:
        for geometry_label, _ in geometries:
            subset = [r for r in condition_rows if r["pair"] == pair and r["geometry"] == geometry_label]
            if not subset:
                continue
            best_row = max(subset, key=lambda r: _safe_float(r.get("mean_pe")))
            geometry_rows.append(
                {
                    "pair": pair,
                    "geometry": geometry_label,
                    "n_conditions": len(subset),
                    "mean_of_mean_pe": float(np.mean([_safe_float(r.get("mean_pe")) for r in subset if np.isfinite(_safe_float(r.get("mean_pe")))])),
                    "best_locus": best_row["locus"],
                    "best_locus_mean_pe": best_row["mean_pe"],
                    "clip_cap": best_row.get("clip_cap"),
                    "sign_scale": best_row.get("sign_scale"),
                }
            )

    _write_rows(out_root / "locus_comparison_table", locus_rows)
    _write_rows(out_root / "geometry_comparison_table", geometry_rows)
    _write_rows(out_root / "necessity_ablation_table", necessity_rows)
    _write_rows(out_root / "entry_vs_continuation_table", entry_rows)
    _write_rows(out_root / "tiny_attn_out_fairness_table", fairness_rows)
    _write_rows(out_root / "item_level_results", item_rows)
    fragmentation_rows = _fragmentation_summary_rows(item_rows)
    geometry_diag_rows = _geometry_diagnostic_rows(item_rows)
    _write_rows(out_root / "fragmentation_bucket_table", fragmentation_rows)
    _write_rows(out_root / "geometry_diagnostics_table", geometry_diag_rows)

    summary_payload = {
        "paper": "paper2_stagea5_diagnostic",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_key": model_key,
        "dense_recovery": {
            "formula": config_payload["config"]["dense_recovery_formula"],
            "references_used": config_payload["dense_reference"],
        },
        "pairs": summary_pairs,
        "aggregate": {
            "n_condition_rows": len(condition_rows),
            "n_fairness_rows": len(fairness_rows),
            "n_item_rows": len(item_rows),
            "n_fragmentation_rows": len(fragmentation_rows),
            "n_geometry_diagnostic_rows": len(geometry_diag_rows),
        },
        "geometry_diagnostics": geometry_diag_rows,
    }
    _write_json(out_root / "stagea5_summary.json", summary_payload)
    missing = [str(p.relative_to(out_root)) for p in _required_artifact_paths(out_root, pairs) if not p.exists()]
    decision = _decision_note(
        condition_rows,
        necessity_rows,
        incomplete=bool(missing),
        missing_artifacts=missing,
        force_layer=(int(args.force_layer) if int(args.force_layer) >= 0 else None),
    )
    _write_json(out_root / "stagea5_decision_note.json", decision)
    summary_payload["aggregate"]["decision"] = decision
    summary_payload["aggregate"]["run_complete"] = not bool(missing)
    summary_payload["aggregate"]["missing_artifacts"] = missing
    _write_json(out_root / "stagea5_summary.json", summary_payload)
    _write_json(out_root / "stagea5_config.json", config_payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
