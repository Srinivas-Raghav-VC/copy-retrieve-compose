#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
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
from core import load_model, run_patching_experiment, set_all_seeds  # noqa: E402
from paper2_fidelity_calibrated.phase1_common import (  # noqa: E402
    build_patch_packet,
    load_pair_split,
    load_stagea_best,
    load_transcoder_for_stagea,
    log,
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
    ap = argparse.ArgumentParser(description="Sparse feature circuit panel using G6-ranked core features.")
    ap.add_argument("--model", type=str, default="4b", choices=["1b", "4b"])
    ap.add_argument("--pairs", type=str, default="aksharantar_hin_latin,aksharantar_tel_latin")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-icl", type=int, default=64)
    ap.add_argument("--n-select", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--max-items", type=int, default=20)
    ap.add_argument("--top-features", type=int, default=8)
    ap.add_argument("--stagea-hin", type=str, default="")
    ap.add_argument("--stagea-tel", type=str, default="")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--external-only", action="store_true")
    ap.add_argument("--require-external-sources", action="store_true")
    ap.add_argument("--min-pool-size", type=int, default=500)
    ap.add_argument("--out", type=str, default="")
    return ap.parse_args()


def _resolve_feature_knockout_path(pair_id: str, model_key: str) -> Path:
    return PROJECT_ROOT / "paper2_fidelity_calibrated" / "results" / "feature_knockout_panel" / str(pair_id) / str(model_key) / "feature_knockout_panel.json"


def _load_ranked_features(path: Path, limit: int) -> List[int]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    rows = list(obj.get("summary_by_feature_index") or [])
    rows.sort(key=lambda row: float(row.get("mean_drop_from_full_patch_first_prob", float("-inf"))), reverse=True)
    out = [int(row["feature_index"]) for row in rows[: max(1, int(limit))]]
    if not out:
        raise RuntimeError(f"No feature ranking found in {path}")
    return out


def _random_matched(reference: List[int], dim: int, *, seed: int) -> List[int]:
    pool = [i for i in range(int(dim)) if int(i) not in set(int(x) for x in reference)]
    rng = random.Random(int(seed))
    rng.shuffle(pool)
    return sorted(pool[: len(reference)])


def _run_condition(
    *,
    model: Any,
    tokenizer: Any,
    transcoder: Any,
    layer: int,
    word: Dict[str, str],
    pair_bundle: Dict[str, Any],
    stagea_best: Dict[str, Any],
    device: str,
    max_new_tokens: int,
    idx_override: List[int] | None,
    label: str,
) -> Dict[str, Any]:
    idx_tensor = None
    topk = int(stagea_best["topk"])
    if idx_override is not None:
        idx_tensor = torch.tensor(sorted(set(int(i) for i in idx_override)), device=device, dtype=torch.long)
        topk = max(1, int(idx_tensor.numel()))
    result = run_patching_experiment(
        model,
        tokenizer,
        transcoder,
        int(layer),
        word,
        pair_bundle["icl_examples"],
        topk=int(topk),
        device=device,
        seed=int(stagea_best["seed"]),
        input_script_name=pair_bundle["input_script_name"],
        source_language=pair_bundle["source_language"],
        output_script_name=pair_bundle["output_script_name"],
        patch_style=str(stagea_best.get("patch_style", "sparse")),
        feature_selection=str(stagea_best.get("feature_selection", "topk_abs_delta")),
        idx_override=idx_tensor,
        prompt_variant=str(stagea_best.get("prompt_variant", "canonical")),
        selector_reference_mode=str(stagea_best.get("selector_reference", "zs")),
        require_query_span_match=bool(stagea_best.get("require_query_span_match", False)),
        use_norm_matching=bool(stagea_best.get("norm_matching", True)),
        eval_generation=True,
        max_new_tokens=int(max_new_tokens),
    )
    row = result.to_dict()
    row["condition"] = str(label)
    row["idx_override"] = list(idx_override or [])
    return row


def main() -> int:
    args = parse_args()
    set_all_seeds(int(args.seed))
    pair_ids = [x.strip() for x in str(args.pairs).split(",") if x.strip()]
    if len(pair_ids) < 2:
        raise RuntimeError("Sparse feature circuit expects at least two pairs")

    cfg = get_model_config(str(args.model))
    model, tokenizer = load_model(str(args.model), device=str(args.device))
    device = str(next(model.parameters()).device)

    pair_bundles: Dict[str, Dict[str, Any]] = {}
    stagea_by_pair: Dict[str, Dict[str, Any]] = {}
    transcoders: Dict[str, Any] = {}
    ranked_features: Dict[str, List[int]] = {}

    stagea_overrides = {
        "aksharantar_hin_latin": str(args.stagea_hin),
        "aksharantar_tel_latin": str(args.stagea_tel),
    }

    for pair_id in pair_ids:
        pair_bundles[pair_id] = load_pair_split(
            pair_id,
            seed=int(args.seed),
            n_icl=int(args.n_icl),
            n_select=int(args.n_select),
            n_eval=int(args.n_eval),
            external_only=bool(args.external_only),
            require_external_sources=bool(args.require_external_sources),
            min_pool_size=int(args.min_pool_size),
        )
        stagea_path = resolve_stagea_path(pair_id, str(args.model), stagea_overrides.get(pair_id, ""))
        if not stagea_path.exists():
            raise FileNotFoundError(f"Missing Stage A artifact: {stagea_path}")
        stagea_by_pair[pair_id] = load_stagea_best(stagea_path, seed=int(args.seed))
        stagea_by_pair[pair_id]["scope_repo"] = str(cfg.scope_repo)
        transcoders[pair_id] = load_transcoder_for_stagea(model, stagea_by_pair[pair_id], device)
        feature_path = _resolve_feature_knockout_path(pair_id, str(args.model))
        if not feature_path.exists():
            raise FileNotFoundError(f"Missing feature-knockout artifact: {feature_path}")
        ranked_features[pair_id] = _load_ranked_features(feature_path, int(args.top_features))

    shared_features = sorted(set(ranked_features[pair_ids[0]]).intersection(*[set(ranked_features[p]) for p in pair_ids[1:]]))
    union_features: List[int] = []
    for pair_id in pair_ids:
        for idx in ranked_features[pair_id]:
            if int(idx) not in union_features:
                union_features.append(int(idx))
    union_features = union_features[: max(1, int(args.top_features) * len(pair_ids))]

    out_root = (
        Path(args.out).resolve()
        if str(args.out).strip()
        else PROJECT_ROOT / "paper2_fidelity_calibrated" / "results" / "sparse_feature_circuit" / str(args.model)
    )
    out_root.mkdir(parents=True, exist_ok=True)

    item_rows: List[Dict[str, Any]] = []
    for pair_id in pair_ids:
        pair_bundle = pair_bundles[pair_id]
        stagea_best = stagea_by_pair[pair_id]
        transcoder = transcoders[pair_id]
        eval_rows = list(pair_bundle["eval_rows"][: max(1, int(args.max_items))])
        pair_core = list(ranked_features[pair_id])
        log(
            f"Running G12 sparse feature circuit: model={args.model} pair={pair_id} items={len(eval_rows)} pair_core={len(pair_core)} shared={len(shared_features)} union={len(union_features)}"
        )
        latent_dim = None
        if eval_rows:
            packet = build_patch_packet(
                model=model,
                tokenizer=tokenizer,
                transcoder=transcoder,
                word=eval_rows[0],
                icl_examples=pair_bundle["icl_examples"],
                stagea_best=stagea_best,
                input_script_name=pair_bundle["input_script_name"],
                source_language=pair_bundle["source_language"],
                output_script_name=pair_bundle["output_script_name"],
                device=device,
            )
            latent_dim = int(packet["icl_feats"].numel())
        if latent_dim is None or latent_dim <= 0:
            raise RuntimeError(f"Could not determine latent dimension for {pair_id}")
        random_union = _random_matched(union_features, latent_dim, seed=int(args.seed) + (11 if "hin" in pair_id else 29))

        for item_idx, word in enumerate(eval_rows, start=1):
            log(f"[{pair_id} {item_idx}/{len(eval_rows)}] {word['ood']} -> {word['hindi']}")
            conditions = [
                ("full_stagea", None),
                ("shared_core", list(shared_features) if shared_features else []),
                ("pair_core", list(pair_core)),
                ("union_core", list(union_features)),
                ("random_union", list(random_union)),
            ]
            for label, idx_override in conditions:
                row = _run_condition(
                    model=model,
                    tokenizer=tokenizer,
                    transcoder=transcoder,
                    layer=int(stagea_best["layer"]),
                    word=word,
                    pair_bundle=pair_bundle,
                    stagea_best=stagea_best,
                    device=device,
                    max_new_tokens=int(args.max_new_tokens),
                    idx_override=idx_override,
                    label=label,
                )
                row.update(
                    {
                        "pair": pair_id,
                        "model": str(args.model),
                        "seed": int(args.seed),
                        "item_index": int(item_idx - 1),
                        "word_ood": str(word["ood"]),
                        "word_hindi": str(word["hindi"]),
                    }
                )
                item_rows.append(row)

    summary: Dict[str, Dict[str, float]] = {}
    for pair_id in pair_ids:
        for label in ["full_stagea", "shared_core", "pair_core", "union_core", "random_union"]:
            rows = [row for row in item_rows if str(row.get("pair")) == pair_id and str(row.get("condition")) == label]
            if not rows:
                continue
            summary[f"{pair_id}::{label}"] = {
                "n_items": float(len(rows)),
                "mean_prob_patched_first": float(np.nanmean([float(row.get("prob_patched_first", float("nan"))) for row in rows])),
                "mean_pe_first": float(np.nanmean([float(row.get("pe_first", float("nan"))) for row in rows])),
                "mean_target_pos1_nll": float(np.nanmean([float(row.get("nll_pos1_patched", float("nan"))) for row in rows])),
                "mean_exact_match": float(np.nanmean([float(row.get("gen_patched", "") == row.get("target", "")) for row in rows])),
            }

    payload = {
        "experiment": "sparse_feature_circuit",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": str(args.model),
        "pairs": pair_ids,
        "seed": int(args.seed),
        "ranked_features": ranked_features,
        "shared_features": shared_features,
        "union_features": union_features,
        "summary": summary,
        "item_rows": item_rows,
    }
    _write_json(out_root / "sparse_feature_circuit.json", payload)
    log(f"Saved: {out_root / 'sparse_feature_circuit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
