#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_model_config  # noqa: E402
from core import run_patching_experiment, load_model, set_all_seeds  # noqa: E402
from paper2_fidelity_calibrated.eval_utils import (  # noqa: E402
    akshara_cer,
    first_entry_correct,
    normalize_text,
    script_compliance,
)
from paper2_fidelity_calibrated.phase1_common import (  # noqa: E402
    extract_feature_delta_vector,
    load_pair_split,
    load_stagea_best,
    load_transcoder_for_stagea,
    log,
    parse_selected_feature_indices,
    resolve_alt_pair,
    resolve_stagea_path,
)


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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Disentangle script-identity vs shared rescue features.")
    ap.add_argument("--model", type=str, default="4b", choices=["1b", "4b"])
    ap.add_argument("--pair", type=str, default="aksharantar_hin_latin")
    ap.add_argument("--alt-pair", type=str, default="")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-icl", type=int, default=64)
    ap.add_argument("--n-select", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--max-items", type=int, default=30)
    ap.add_argument("--n-stats-items", type=int, default=50)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--stagea", type=str, default="")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--external-only", action="store_true")
    ap.add_argument("--require-external-sources", action="store_true")
    ap.add_argument("--min-pool-size", type=int, default=500)
    ap.add_argument("--out", type=str, default="")
    return ap.parse_args()


def _generation_metrics(gold_text: str, pred_text: str, *, target_script: str) -> Dict[str, float]:
    gold = normalize_text(gold_text)
    pred = normalize_text(pred_text)
    return {
        "exact_match": float(pred == gold),
        "akshara_cer": float(akshara_cer(pred, gold)),
        "script_compliance": float(script_compliance(pred, target_script)),
        "first_entry_correct": float(first_entry_correct(pred, gold)),
    }


def _mean_abs_delta(
    *,
    model: Any,
    tokenizer: Any,
    transcoder: Any,
    layer: int,
    pair_bundle: Dict[str, Any],
    stagea_best: Dict[str, Any],
    device: str,
    n_items: int,
    seed: int,
) -> torch.Tensor:
    rows = list(pair_bundle["eval_rows"][: max(1, int(n_items))])
    values: List[torch.Tensor] = []
    patch_position_mode = str(stagea_best.get("patch_position_mode", "source_last_subtoken")).strip().lower()
    if patch_position_mode == "target_pos1_teacher_forced":
        patch_position_mode = "target_pos1"
    for word in rows:
        delta = extract_feature_delta_vector(
            model=model,
            tokenizer=tokenizer,
            transcoder=transcoder,
            layer=int(layer),
            word=word,
            icl_examples=pair_bundle["icl_examples"],
            input_script_name=pair_bundle["input_script_name"],
            source_language=pair_bundle["source_language"],
            output_script_name=pair_bundle["output_script_name"],
            prompt_variant=str(stagea_best.get("prompt_variant", "canonical")),
            selector_reference=str(stagea_best.get("selector_reference", "zs")),
            patch_position_mode=str(patch_position_mode),
            device=device,
            seed=int(seed),
        )
        values.append(torch.abs(delta.detach().float().cpu()))
    if not values:
        raise RuntimeError("Could not compute any feature deltas for suppression analysis.")
    return torch.stack(values, dim=0).mean(dim=0)


def main() -> int:
    args = parse_args()
    set_all_seeds(int(args.seed))

    pair_id = str(args.pair)
    alt_pair_id = resolve_alt_pair(pair_id, str(args.alt_pair))

    pair_bundle = load_pair_split(
        pair_id,
        seed=int(args.seed),
        n_icl=int(args.n_icl),
        n_select=int(args.n_select),
        n_eval=int(args.n_eval),
        external_only=bool(args.external_only),
        require_external_sources=bool(args.require_external_sources),
        min_pool_size=int(args.min_pool_size),
    )
    alt_bundle = load_pair_split(
        alt_pair_id,
        seed=int(args.seed),
        n_icl=int(args.n_icl),
        n_select=int(args.n_select),
        n_eval=max(int(args.n_eval), int(args.n_stats_items)),
        external_only=bool(args.external_only),
        require_external_sources=bool(args.require_external_sources),
        min_pool_size=int(args.min_pool_size),
    )

    cfg = get_model_config(str(args.model))
    model, tokenizer = load_model(str(args.model), device=str(args.device))
    device = str(next(model.parameters()).device)

    stagea_path = resolve_stagea_path(pair_id, str(args.model), str(args.stagea))
    if not stagea_path.exists():
        raise FileNotFoundError(f"Missing Stage A artifact: {stagea_path}")
    stagea_best = load_stagea_best(stagea_path, seed=int(args.seed))
    stagea_best["scope_repo"] = str(cfg.scope_repo)

    transcoder = load_transcoder_for_stagea(
        model,
        {**stagea_best, "scope_repo": str(cfg.scope_repo)},
        device,
    )
    layer = int(stagea_best["layer"])

    log(
        f"Running G9 script-vs-concept suppression: pair={pair_id} alt_pair={alt_pair_id} model={args.model} "
        f"stageA=(variant={stagea_best['variant']}, layer={layer}, topk={stagea_best['topk']})"
    )

    main_abs = _mean_abs_delta(
        model=model,
        tokenizer=tokenizer,
        transcoder=transcoder,
        layer=layer,
        pair_bundle=pair_bundle,
        stagea_best=stagea_best,
        device=device,
        n_items=int(args.n_stats_items),
        seed=int(args.seed),
    )
    alt_abs = _mean_abs_delta(
        model=model,
        tokenizer=tokenizer,
        transcoder=transcoder,
        layer=layer,
        pair_bundle=alt_bundle,
        stagea_best=stagea_best,
        device=device,
        n_items=int(args.n_stats_items),
        seed=int(args.seed),
    )

    script_score = torch.abs(main_abs - alt_abs)
    shared_score = torch.minimum(main_abs, alt_abs)

    group_size = max(1, int(args.group_size))
    script_idx = torch.topk(script_score, k=min(group_size, int(script_score.numel()))).indices.tolist()
    script_set = {int(i) for i in script_idx}

    shared_order = torch.topk(shared_score, k=min(max(group_size * 4, group_size), int(shared_score.numel()))).indices.tolist()
    shared_idx: List[int] = []
    for idx in shared_order:
        idx_int = int(idx)
        if idx_int in script_set:
            continue
        shared_idx.append(idx_int)
        if len(shared_idx) >= group_size:
            break
    shared_set = {int(i) for i in shared_idx}

    eval_rows = list(pair_bundle["eval_rows"][: max(1, int(args.max_items))])
    item_rows: List[Dict[str, Any]] = []

    def _run_condition(word: Dict[str, str], idx_list: List[int] | None) -> Dict[str, Any]:
        idx_override = None
        if idx_list is not None:
            idx_override = torch.tensor(sorted(set(int(i) for i in idx_list)), device=device, dtype=torch.long)
        result = run_patching_experiment(
            model,
            tokenizer,
            transcoder,
            int(layer),
            word,
            pair_bundle["icl_examples"],
            topk=max(1, len(idx_list)) if idx_list is not None else int(stagea_best["topk"]),
            device=device,
            seed=int(args.seed),
            input_script_name=pair_bundle["input_script_name"],
            source_language=pair_bundle["source_language"],
            output_script_name=pair_bundle["output_script_name"],
            patch_style=str(stagea_best.get("patch_style", "sparse")),
            feature_selection=str(stagea_best.get("feature_selection", "topk_abs_delta")),
            idx_override=idx_override,
            prompt_variant=str(stagea_best.get("prompt_variant", "canonical")),
            selector_reference_mode=str(stagea_best.get("selector_reference", "zs")),
            require_query_span_match=bool(stagea_best.get("require_query_span_match", False)),
            use_norm_matching=bool(stagea_best.get("norm_matching", True)),
            eval_generation=True,
            max_new_tokens=int(args.max_new_tokens),
        )
        row = result.to_dict()
        row.update(_generation_metrics(str(word["hindi"]), str(row.get("gen_patched", "")), target_script=pair_bundle["output_script_name"]))
        row["selected_feature_indices_list"] = parse_selected_feature_indices(row.get("selected_feature_indices", ""))
        return row

    for item_idx, word in enumerate(eval_rows, start=1):
        log(f"[{item_idx}/{len(eval_rows)}] {word['ood']} -> {word['hindi']}")
        full = _run_condition(word, None)
        full_selected = set(int(i) for i in full["selected_feature_indices_list"])
        script_intersection = sorted(full_selected.intersection(script_set))
        shared_intersection = sorted(full_selected.intersection(shared_set))
        minus_script = sorted(full_selected.difference(script_set))
        minus_shared = sorted(full_selected.difference(shared_set))

        conditions = {
            "full_selected": full,
            "script_only": _run_condition(word, script_intersection),
            "shared_only": _run_condition(word, shared_intersection),
            "full_minus_script": _run_condition(word, minus_script),
            "full_minus_shared": _run_condition(word, minus_shared),
        }

        for condition_name, row in conditions.items():
            item_rows.append(
                {
                    "pair": pair_id,
                    "alt_pair": alt_pair_id,
                    "model": str(args.model),
                    "seed": int(args.seed),
                    "item_index": int(item_idx - 1),
                    "word_ood": str(word["ood"]),
                    "word_hindi": str(word["hindi"]),
                    "condition": str(condition_name),
                    "patch_layer": int(layer),
                    "patch_topk": int(stagea_best["topk"]),
                    "patch_variant": str(stagea_best["variant"]),
                    "script_group_overlap": list(script_intersection),
                    "shared_group_overlap": list(shared_intersection),
                    "n_selected_features": int(len(row.get("selected_feature_indices_list", []))),
                    "pe_first": float(row.get("pe_first", float("nan"))),
                    "pe_multi": float(row.get("pe_multi", float("nan"))),
                    "exact_match": float(row.get("exact_match", float("nan"))),
                    "script_compliance": float(row.get("script_compliance", float("nan"))),
                    "first_entry_correct": float(row.get("first_entry_correct", float("nan"))),
                    "akshara_cer": float(row.get("akshara_cer", float("nan"))),
                    "gen_patched": str(row.get("gen_patched", "")),
                    "selected_feature_indices": list(row.get("selected_feature_indices_list", [])),
                }
            )

    summary_rows: List[Dict[str, Any]] = []
    if item_rows:
        for condition in sorted({str(r["condition"]) for r in item_rows}):
            rows = [r for r in item_rows if str(r["condition"]) == condition]
            def _m(key: str) -> float:
                vals = np.array([float(r[key]) for r in rows], dtype=np.float64)
                return float(np.nanmean(vals))
            summary_rows.append(
                {
                    "condition": str(condition),
                    "n_items": int(len(rows)),
                    "mean_pe_first": _m("pe_first"),
                    "mean_pe_multi": _m("pe_multi"),
                    "mean_exact_match": _m("exact_match"),
                    "mean_script_compliance": _m("script_compliance"),
                    "mean_first_entry_correct": _m("first_entry_correct"),
                    "mean_akshara_cer": _m("akshara_cer"),
                    "mean_n_selected_features": _m("n_selected_features"),
                }
            )

    payload = {
        "experiment": "language_script_feature_suppression",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pair": pair_id,
        "alt_pair": alt_pair_id,
        "model": str(args.model),
        "seed": int(args.seed),
        "stagea_best": stagea_best,
        "layer": int(layer),
        "feature_grouping": {
            "group_size": int(group_size),
            "script_group_indices": sorted(script_set),
            "shared_group_indices": sorted(shared_set),
            "script_group_main_minus_alt": [float(main_abs[int(i)].item() - alt_abs[int(i)].item()) for i in sorted(script_set)],
            "shared_group_main_abs": [float(main_abs[int(i)].item()) for i in sorted(shared_set)],
            "shared_group_alt_abs": [float(alt_abs[int(i)].item()) for i in sorted(shared_set)],
            "definition": {
                "script_score": "abs(mean_abs_delta_main - mean_abs_delta_alt)",
                "shared_score": "min(mean_abs_delta_main, mean_abs_delta_alt)",
                "comparison_layer": "main-pair Stage-A best layer applied to both pair prompts",
            },
        },
        "notes": {
            "conditions": [
                "full_selected",
                "script_only",
                "shared_only",
                "full_minus_script",
                "full_minus_shared",
            ],
            "interpretation": (
                "script-like features should disproportionately affect script_compliance; "
                "shared/concept-like features should disproportionately affect first-entry/exact-match rescue."
            ),
        },
        "summary": summary_rows,
        "items": item_rows,
    }

    out_root = (
        Path(args.out).resolve()
        if str(args.out).strip()
        else PROJECT_ROOT / "paper2_fidelity_calibrated" / "results" / "language_script_feature_suppression" / pair_id / str(args.model)
    )
    out_root.mkdir(parents=True, exist_ok=True)
    _write_json(out_root / "language_script_feature_suppression.json", payload)
    log(f"Saved: {out_root / 'language_script_feature_suppression.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
