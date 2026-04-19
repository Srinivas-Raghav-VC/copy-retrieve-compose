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
from core import (  # noqa: E402
    _get_unembedding_weight,
    load_model,
    register_transcoder_feature_patch_hook,
    set_all_seeds,
)
from paper2_fidelity_calibrated.phase1_common import (  # noqa: E402
    build_patch_packet,
    get_final_norm,
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
    ap = argparse.ArgumentParser(description="Layerwise logit-lens rescue trajectory for CFOM rescue.")
    ap.add_argument("--model", type=str, default="4b", choices=["1b", "4b"])
    ap.add_argument("--pair", type=str, default="aksharantar_hin_latin")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-icl", type=int, default=64)
    ap.add_argument("--n-select", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--max-items", type=int, default=50)
    ap.add_argument("--stagea", type=str, default="")
    ap.add_argument("--external-only", action="store_true")
    ap.add_argument("--require-external-sources", action="store_true")
    ap.add_argument("--min-pool-size", type=int, default=500)
    ap.add_argument("--out", type=str, default="")
    return ap.parse_args()


def _rank_of_target(logits: torch.Tensor, target_id: int) -> int:
    target_logit = logits[int(target_id)]
    return int(torch.sum(logits > target_logit).item()) + 1


def _layerwise_token_metrics(
    *,
    hidden_states: Any,
    final_norm: torch.nn.Module,
    unembed: torch.Tensor,
    target_id: int,
    tokenizer: Any,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if target_id < 0:
        return rows
    for layer_idx, h in enumerate(hidden_states[1:]):
        vec = h[0, -1, :].detach()
        normed = final_norm(vec.unsqueeze(0).to(dtype=unembed.dtype))
        logits = torch.nn.functional.linear(normed, unembed).float()[0]
        probs = torch.softmax(logits, dim=-1)
        rank = _rank_of_target(logits, target_id)
        top5 = torch.topk(logits, k=min(5, int(logits.numel())))
        rows.append(
            {
                "layer": int(layer_idx),
                "target_prob": float(probs[int(target_id)].item()),
                "target_logit": float(logits[int(target_id)].item()),
                "target_rank": int(rank),
                "top5_tokens": [tokenizer.decode([int(i)]).strip() for i in top5.indices.tolist()],
                "top5_logits": [float(x) for x in top5.values.tolist()],
            }
        )
    return rows


def _run_condition(
    *,
    model: Any,
    input_ids: torch.Tensor,
    target_id: int,
    tokenizer: Any,
    final_norm: torch.nn.Module,
    unembed: torch.Tensor,
    patch_hook: Any = None,
) -> List[Dict[str, Any]]:
    try:
        with torch.inference_mode():
            out = model(input_ids=input_ids, use_cache=False, output_hidden_states=True)
        hidden_states = getattr(out, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError("Model did not return hidden states.")
        return _layerwise_token_metrics(
            hidden_states=hidden_states,
            final_norm=final_norm,
            unembed=unembed,
            target_id=int(target_id),
            tokenizer=tokenizer,
        )
    finally:
        if patch_hook is not None:
            patch_hook.remove()


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
    if not stagea_path.exists():
        raise FileNotFoundError(f"Missing Stage A artifact: {stagea_path}")
    stagea_best = load_stagea_best(stagea_path, seed=int(args.seed))
    stagea_best["scope_repo"] = str(cfg.scope_repo)

    transcoder = load_transcoder_for_stagea(
        model,
        {**stagea_best, "scope_repo": str(cfg.scope_repo)},
        device,
    )

    final_norm = get_final_norm(model).to(device)
    unembed = _get_unembedding_weight(model)
    if unembed is None:
        raise RuntimeError("Could not locate output embedding / lm_head weight.")
    unembed = unembed.detach().to(device)

    out_root = (
        Path(args.out).resolve()
        if str(args.out).strip()
        else PROJECT_ROOT / "paper2_fidelity_calibrated" / "results" / "logit_lens_rescue_trajectory" / str(args.pair) / str(args.model)
    )
    out_root.mkdir(parents=True, exist_ok=True)

    eval_rows = list(pair_bundle["eval_rows"][: max(1, int(args.max_items))])
    item_rows: List[Dict[str, Any]] = []

    log(
        f"Running G1 logit-lens trajectory: pair={args.pair} model={args.model} items={len(eval_rows)} "
        f"stageA=(variant={stagea_best['variant']}, layer={stagea_best['layer']}, topk={stagea_best['topk']})"
    )

    for item_idx, word in enumerate(eval_rows, start=1):
        log(f"[{item_idx}/{len(eval_rows)}] {word['ood']} -> {word['hindi']}")
        packet = build_patch_packet(
            model=model,
            tokenizer=tokenizer,
            transcoder=transcoder,
            word=word,
            icl_examples=pair_bundle["icl_examples"],
            stagea_best=stagea_best,
            input_script_name=pair_bundle["input_script_name"],
            source_language=pair_bundle["source_language"],
            output_script_name=pair_bundle["output_script_name"],
            device=device,
        )
        target_id = int(packet["target_id"])
        if target_id < 0:
            continue

        zs_rows = _run_condition(
            model=model,
            input_ids=packet["zs_input_ids"],
            target_id=target_id,
            tokenizer=tokenizer,
            final_norm=final_norm,
            unembed=unembed,
        )
        icl_rows = _run_condition(
            model=model,
            input_ids=packet["icl_input_ids"],
            target_id=target_id,
            tokenizer=tokenizer,
            final_norm=final_norm,
            unembed=unembed,
        )
        patch_hook = register_transcoder_feature_patch_hook(
            model,
            transcoder,
            int(stagea_best["layer"]),
            packet["patch_feats"],
            patch_position=int(packet["zs_patch_position"]),
            target_output_norm=packet["target_output_norm"],
        )
        patched_rows = _run_condition(
            model=model,
            input_ids=packet["zs_input_ids"],
            target_id=target_id,
            tokenizer=tokenizer,
            final_norm=final_norm,
            unembed=unembed,
            patch_hook=patch_hook,
        )

        for condition, rows in (("zs", zs_rows), ("icl64", icl_rows), ("patched", patched_rows)):
            for row in rows:
                item_rows.append(
                    {
                        "pair": str(args.pair),
                        "model": str(args.model),
                        "seed": int(args.seed),
                        "item_index": int(item_idx - 1),
                        "word_ood": str(word["ood"]),
                        "word_hindi": str(word["hindi"]),
                        "condition": str(condition),
                        "target_id": int(target_id),
                        "patch_layer": int(stagea_best["layer"]),
                        "patch_topk": int(stagea_best["topk"]),
                        "patch_variant": str(stagea_best["variant"]),
                        "selected_feature_indices": list(packet["selected_idx"]),
                        **row,
                    }
                )

    summary_rows: List[Dict[str, Any]] = []
    if item_rows:
        conditions = sorted({str(r["condition"]) for r in item_rows})
        layers = sorted({int(r["layer"]) for r in item_rows})
        for condition in conditions:
            for layer in layers:
                rows = [r for r in item_rows if str(r["condition"]) == condition and int(r["layer"]) == layer]
                if not rows:
                    continue
                probs = np.array([float(r["target_prob"]) for r in rows], dtype=np.float64)
                ranks = np.array([float(r["target_rank"]) for r in rows], dtype=np.float64)
                logits = np.array([float(r["target_logit"]) for r in rows], dtype=np.float64)
                summary_rows.append(
                    {
                        "condition": str(condition),
                        "layer": int(layer),
                        "n_items": int(len(rows)),
                        "mean_target_prob": float(np.nanmean(probs)),
                        "mean_target_rank": float(np.nanmean(ranks)),
                        "median_target_rank": float(np.nanmedian(ranks)),
                        "mean_target_logit": float(np.nanmean(logits)),
                    }
                )

    payload = {
        "experiment": "logit_lens_rescue_trajectory",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pair": str(args.pair),
        "model": str(args.model),
        "seed": int(args.seed),
        "stagea_best": stagea_best,
        "n_items": int(len(eval_rows)),
        "notes": {
            "readout_position": "final prompt position (next-token prediction site)",
            "target_definition": "first target token id from gold transliteration",
            "conditions": ["zs", "icl64", "patched"],
            "patch_condition": "Stage-A best sparse/hybrid feature patch applied to ZS prompt at source locus",
        },
        "summary": summary_rows,
        "items": item_rows,
    }
    _write_json(out_root / "logit_lens_rescue_trajectory.json", payload)
    log(f"Saved: {out_root / 'logit_lens_rescue_trajectory.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
