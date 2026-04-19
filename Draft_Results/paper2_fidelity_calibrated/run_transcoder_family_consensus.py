#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_model_config  # noqa: E402
from core import load_model, run_patching_experiment, set_all_seeds  # noqa: E402
from paper2_fidelity_calibrated.phase1_common import (  # noqa: E402
    load_pair_split,
    load_stagea_best,
    load_transcoder_for_stagea,
    log,
    parse_selected_feature_indices,
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
    ap = argparse.ArgumentParser(description="Compare patch effects and selected features across transcoder families.")
    ap.add_argument("--model", type=str, default="4b", choices=["1b", "4b"])
    ap.add_argument("--pair", type=str, default="aksharantar_hin_latin")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-icl", type=int, default=64)
    ap.add_argument("--n-select", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--max-items", type=int, default=20)
    ap.add_argument("--variants", type=str, default="skipless_or_non_affine,affine_skip")
    ap.add_argument("--stagea", type=str, default="")
    ap.add_argument("--external-only", action="store_true")
    ap.add_argument("--require-external-sources", action="store_true")
    ap.add_argument("--min-pool-size", type=int, default=500)
    ap.add_argument("--out", type=str, default="")
    return ap.parse_args()


def _parse_variants(raw: str) -> List[str]:
    vals = [str(x).strip() for x in str(raw or "").split(",") if str(x).strip()]
    return vals or ["skipless_or_non_affine", "affine_skip"]


def _jaccard(a: List[int], b: List[int]) -> float:
    sa = set(int(x) for x in a)
    sb = set(int(x) for x in b)
    union = len(sa | sb)
    return float(len(sa & sb) / union) if union > 0 else float("nan")


def main() -> int:
    args = parse_args()
    set_all_seeds(int(args.seed))

    pair_bundle = load_pair_split(
        str(args.pair),
        seed=int(args.seed),
        n_icl=int(args.n_icl),
        n_select=int(args.n_select),
        n_eval=int(args.n_eval),
        external_only=bool(args.external_only),
        require_external_sources=bool(args.require_external_sources),
        min_pool_size=int(args.min_pool_size),
    )
    cfg = get_model_config(str(args.model))
    model, tokenizer = load_model(str(args.model), device=str(args.device))
    device = str(next(model.parameters()).device)

    stagea_path = resolve_stagea_path(str(args.pair), str(args.model), str(args.stagea))
    stagea_best = load_stagea_best(stagea_path, seed=int(args.seed))
    stagea_best["scope_repo"] = str(cfg.scope_repo)
    variants = _parse_variants(str(args.variants))

    transcoders: Dict[str, Any] = {}
    for variant in variants:
        best_variant = dict(stagea_best)
        best_variant["variant"] = str(variant)
        transcoders[str(variant)] = load_transcoder_for_stagea(model, best_variant, device)

    out_root = (
        Path(args.out).resolve()
        if str(args.out).strip()
        else PROJECT_ROOT / "paper2_fidelity_calibrated" / "results" / "transcoder_family_consensus" / str(args.pair) / str(args.model)
    )
    out_root.mkdir(parents=True, exist_ok=True)

    eval_rows = list(pair_bundle["eval_rows"][: max(1, int(args.max_items))])
    item_rows: List[Dict[str, Any]] = []

    log(f"Running transcoder-family consensus: pair={args.pair} model={args.model} variants={variants} items={len(eval_rows)}")

    for item_idx, word in enumerate(eval_rows, start=1):
        log(f"[{item_idx}/{len(eval_rows)}] {word['ood']} -> {word['hindi']}")
        per_variant: Dict[str, Dict[str, Any]] = {}
        for variant in variants:
            result = run_patching_experiment(
                model,
                tokenizer,
                transcoders[str(variant)],
                int(stagea_best["layer"]),
                word,
                pair_bundle["icl_examples"],
                topk=int(stagea_best["topk"]),
                device=device,
                seed=int(args.seed),
                input_script_name=pair_bundle["input_script_name"],
                source_language=pair_bundle["source_language"],
                output_script_name=pair_bundle["output_script_name"],
                patch_style=str(stagea_best.get("patch_style", "sparse")),
                feature_selection=str(stagea_best.get("feature_selection", "topk_abs_delta")),
                prompt_variant=str(stagea_best.get("prompt_variant", "canonical")),
                selector_reference_mode=str(stagea_best.get("selector_reference", "zs")),
                require_query_span_match=bool(stagea_best.get("require_query_span_match", False)),
                use_norm_matching=bool(stagea_best.get("norm_matching", True)),
                eval_generation=True,
                max_new_tokens=16,
            )
            row = result.to_dict()
            row["selected_feature_indices_list"] = parse_selected_feature_indices(row.get("selected_feature_indices", ""))
            per_variant[str(variant)] = row

        base_variant = str(variants[0])
        base_row = per_variant[base_variant]
        merged = {
            "pair": str(args.pair),
            "model": str(args.model),
            "seed": int(args.seed),
            "item_index": int(item_idx - 1),
            "word_ood": str(word["ood"]),
            "word_hindi": str(word["hindi"]),
            "stagea_layer": int(stagea_best["layer"]),
            "stagea_topk": int(stagea_best["topk"]),
        }
        for variant, row in per_variant.items():
            key = str(variant)
            merged[f"{key}__prob_patched_first"] = float(row.get("prob_patched_first", float("nan")))
            merged[f"{key}__pe_first"] = float(row.get("pe_first", float("nan")))
            merged[f"{key}__nll_pos1_patched"] = float(row.get("nll_pos1_patched", float("nan")))
            merged[f"{key}__gen_patched"] = str(row.get("gen_patched", ""))
            merged[f"{key}__selected_feature_indices"] = list(row["selected_feature_indices_list"])
        if len(variants) >= 2:
            for i in range(len(variants)):
                for j in range(i + 1, len(variants)):
                    va, vb = str(variants[i]), str(variants[j])
                    merged[f"jaccard__{va}__{vb}"] = float(
                        _jaccard(
                            per_variant[va]["selected_feature_indices_list"],
                            per_variant[vb]["selected_feature_indices_list"],
                        )
                    )
                    merged[f"delta_prob_patched_first__{va}__{vb}"] = float(
                        float(per_variant[va].get("prob_patched_first", float("nan")))
                        - float(per_variant[vb].get("prob_patched_first", float("nan")))
                    )
        item_rows.append(merged)

    summary: Dict[str, float] = {"n_items": float(len(item_rows))}
    if len(variants) >= 2:
        for i in range(len(variants)):
            for j in range(i + 1, len(variants)):
                va, vb = str(variants[i]), str(variants[j])
                summary[f"mean_jaccard__{va}__{vb}"] = float(np.nanmean([row.get(f"jaccard__{va}__{vb}", float("nan")) for row in item_rows]))
                summary[f"mean_delta_prob_patched_first__{va}__{vb}"] = float(np.nanmean([row.get(f"delta_prob_patched_first__{va}__{vb}", float("nan")) for row in item_rows]))
    for variant in variants:
        key = str(variant)
        summary[f"mean_{key}__prob_patched_first"] = float(np.nanmean([row.get(f"{key}__prob_patched_first", float("nan")) for row in item_rows]))
        summary[f"mean_{key}__pe_first"] = float(np.nanmean([row.get(f"{key}__pe_first", float("nan")) for row in item_rows]))
        summary[f"mean_{key}__nll_pos1_patched"] = float(np.nanmean([row.get(f"{key}__nll_pos1_patched", float("nan")) for row in item_rows]))

    payload = {
        "experiment": "transcoder_family_consensus",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": str(args.model),
        "pair": str(args.pair),
        "seed": int(args.seed),
        "variants": variants,
        "stagea_best": stagea_best,
        "summary": summary,
        "item_rows": item_rows,
    }
    _write_json(out_root / "transcoder_family_consensus.json", payload)
    log(f"Saved: {out_root / 'transcoder_family_consensus.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
