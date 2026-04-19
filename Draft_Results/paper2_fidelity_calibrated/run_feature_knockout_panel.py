#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_model_config  # noqa: E402
from core import (  # noqa: E402
    _teacher_forced_metrics_from_input_ids,
    load_model,
    register_transcoder_feature_patch_hook,
    set_all_seeds,
)
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
    ap = argparse.ArgumentParser(description="Per-feature necessity panel over the selected Stage-A patch features.")
    ap.add_argument("--model", type=str, default="4b", choices=["1b", "4b"])
    ap.add_argument("--pair", type=str, default="aksharantar_hin_latin")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-icl", type=int, default=64)
    ap.add_argument("--n-select", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--max-items", type=int, default=30)
    ap.add_argument("--max-features", type=int, default=8)
    ap.add_argument("--stagea", type=str, default="")
    ap.add_argument("--external-only", action="store_true")
    ap.add_argument("--require-external-sources", action="store_true")
    ap.add_argument("--min-pool-size", type=int, default=500)
    ap.add_argument("--out", type=str, default="")
    return ap.parse_args()


def _run_teacher_forced(
    *,
    model: Any,
    input_ids: torch.Tensor,
    target_ids: List[int],
    target_id: int,
    device: str,
    hook: Any = None,
) -> Dict[str, float]:
    try:
        return _teacher_forced_metrics_from_input_ids(
            model=model,
            input_ids=input_ids,
            target_ids=list(target_ids),
            target_id=int(target_id),
            device=str(device),
        )
    finally:
        if hook is not None:
            try:
                hook.remove()
            except Exception:
                pass


def _drop_one_feature(packet: Dict[str, Any], feature_index: int) -> torch.Tensor:
    patch_feats = packet["patch_feats"].detach().clone()
    idx = int(feature_index)
    if idx < 0 or idx >= int(patch_feats.numel()):
        return patch_feats
    if str(packet.get("patch_style", "sparse")).strip().lower() == "hybrid":
        patch_feats[idx] = packet["selector_ref_feats"][idx].detach().to(device=patch_feats.device, dtype=patch_feats.dtype)
    else:
        patch_feats[idx] = 0.0
    return patch_feats


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
    transcoder = load_transcoder_for_stagea(model, stagea_best, device)

    out_root = (
        Path(args.out).resolve()
        if str(args.out).strip()
        else PROJECT_ROOT / "paper2_fidelity_calibrated" / "results" / "feature_knockout_panel" / str(args.pair) / str(args.model)
    )
    out_root.mkdir(parents=True, exist_ok=True)

    eval_rows = list(pair_bundle["eval_rows"][: max(1, int(args.max_items))])
    item_rows: List[Dict[str, Any]] = []

    log(
        f"Running G6 feature-knockout panel: pair={args.pair} model={args.model} items={len(eval_rows)} "
        f"stageA=(layer={stagea_best['layer']}, topk={stagea_best['topk']}, variant={stagea_best['variant']})"
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
        target_ids = list(packet["target_ids"])
        target_id = int(packet["target_id"])
        if target_id < 0 or not target_ids:
            continue

        selected_idx = list(packet["selected_idx"][: max(1, int(args.max_features))])
        if not selected_idx:
            continue

        zs_metrics = _run_teacher_forced(
            model=model,
            input_ids=packet["zs_input_ids"],
            target_ids=target_ids,
            target_id=target_id,
            device=device,
        )
        full_hook = register_transcoder_feature_patch_hook(
            model,
            transcoder,
            int(stagea_best["layer"]),
            packet["patch_feats"],
            patch_position=int(packet["zs_patch_position"]),
            target_output_norm=packet["target_output_norm"],
        )
        full_patch_metrics = _run_teacher_forced(
            model=model,
            input_ids=packet["zs_input_ids"],
            target_ids=target_ids,
            target_id=target_id,
            device=device,
            hook=full_hook,
        )

        for feature_rank, feature_index in enumerate(selected_idx, start=1):
            knockout_feats = _drop_one_feature(packet, int(feature_index))
            knockout_hook = register_transcoder_feature_patch_hook(
                model,
                transcoder,
                int(stagea_best["layer"]),
                knockout_feats,
                patch_position=int(packet["zs_patch_position"]),
                target_output_norm=packet["target_output_norm"],
            )
            knockout_metrics = _run_teacher_forced(
                model=model,
                input_ids=packet["zs_input_ids"],
                target_ids=target_ids,
                target_id=target_id,
                device=device,
                hook=knockout_hook,
            )

            item_rows.append(
                {
                    "pair": str(args.pair),
                    "model": str(args.model),
                    "seed": int(args.seed),
                    "item_index": int(item_idx - 1),
                    "word_ood": str(word["ood"]),
                    "word_hindi": str(word["hindi"]),
                    "patch_layer": int(stagea_best["layer"]),
                    "patch_topk": int(stagea_best["topk"]),
                    "patch_variant": str(stagea_best["variant"]),
                    "feature_rank": int(feature_rank),
                    "feature_index": int(feature_index),
                    "full_selected_feature_indices": list(packet["selected_idx"]),
                    "zs_first_prob": float(zs_metrics["first_prob"]),
                    "full_patch_first_prob": float(full_patch_metrics["first_prob"]),
                    "knockout_first_prob": float(knockout_metrics["first_prob"]),
                    "zs_first_logit": float(zs_metrics["first_logit"]),
                    "full_patch_first_logit": float(full_patch_metrics["first_logit"]),
                    "knockout_first_logit": float(knockout_metrics["first_logit"]),
                    "zs_target_pos1_nll": float(zs_metrics["target_pos1_nll"]),
                    "full_patch_target_pos1_nll": float(full_patch_metrics["target_pos1_nll"]),
                    "knockout_target_pos1_nll": float(knockout_metrics["target_pos1_nll"]),
                    "drop_from_full_patch_first_prob": float(full_patch_metrics["first_prob"] - knockout_metrics["first_prob"]),
                    "drop_from_full_patch_first_logit": float(full_patch_metrics["first_logit"] - knockout_metrics["first_logit"]),
                    "increase_from_full_patch_target_pos1_nll": float(knockout_metrics["target_pos1_nll"] - full_patch_metrics["target_pos1_nll"]),
                }
            )

    by_rank: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    by_feature: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in item_rows:
        by_rank[int(row["feature_rank"])].append(row)
        by_feature[int(row["feature_index"])].append(row)

    summary_by_rank: List[Dict[str, Any]] = []
    for rank in sorted(by_rank):
        rows = by_rank[rank]
        summary_by_rank.append(
            {
                "feature_rank": int(rank),
                "n_items": int(len(rows)),
                "mean_drop_from_full_patch_first_prob": float(np.nanmean([row["drop_from_full_patch_first_prob"] for row in rows])),
                "mean_drop_from_full_patch_first_logit": float(np.nanmean([row["drop_from_full_patch_first_logit"] for row in rows])),
                "mean_increase_from_full_patch_target_pos1_nll": float(np.nanmean([row["increase_from_full_patch_target_pos1_nll"] for row in rows])),
            }
        )

    summary_by_feature: List[Dict[str, Any]] = []
    for feature_index in sorted(by_feature):
        rows = by_feature[feature_index]
        summary_by_feature.append(
            {
                "feature_index": int(feature_index),
                "n_items": int(len(rows)),
                "mean_feature_rank": float(np.nanmean([row["feature_rank"] for row in rows])),
                "mean_drop_from_full_patch_first_prob": float(np.nanmean([row["drop_from_full_patch_first_prob"] for row in rows])),
                "mean_drop_from_full_patch_first_logit": float(np.nanmean([row["drop_from_full_patch_first_logit"] for row in rows])),
                "mean_increase_from_full_patch_target_pos1_nll": float(np.nanmean([row["increase_from_full_patch_target_pos1_nll"] for row in rows])),
            }
        )

    payload = {
        "experiment": "feature_knockout_panel",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": str(args.model),
        "pair": str(args.pair),
        "seed": int(args.seed),
        "stagea_path": str(stagea_path),
        "stagea_best": stagea_best,
        "max_features": int(args.max_features),
        "summary_by_rank": summary_by_rank,
        "summary_by_feature_index": summary_by_feature,
        "item_rows": item_rows,
    }
    _write_json(out_root / "feature_knockout_panel.json", payload)
    log(f"Saved: {out_root / 'feature_knockout_panel.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
