#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_model_config  # noqa: E402
from core import (  # noqa: E402
    _extract_mlp_io_at_position_from_input_ids,
    _teacher_forced_metrics_from_input_ids,
    load_model,
    register_transcoder_feature_ablation_hook,
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
from paper2_fidelity_calibrated.phase23_common import (  # noqa: E402
    capture_o_proj_inputs_at_position,
    register_attention_head_replace_hooks,
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
    ap = argparse.ArgumentParser(description="Head→MLP edge attribution with feature mediation diagnostics.")
    ap.add_argument("--model", type=str, default="4b", choices=["1b", "4b"])
    ap.add_argument("--pair", type=str, default="aksharantar_hin_latin")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-icl", type=int, default=64)
    ap.add_argument("--n-select", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--max-items", type=int, default=30)
    ap.add_argument("--stagea", type=str, default="")
    ap.add_argument("--top-heads-json", type=str, default="")
    ap.add_argument("--top-n-heads", type=int, default=8)
    ap.add_argument("--external-only", action="store_true")
    ap.add_argument("--require-external-sources", action="store_true")
    ap.add_argument("--min-pool-size", type=int, default=500)
    ap.add_argument("--out", type=str, default="")
    return ap.parse_args()


def _lang_from_pair(pair_id: str) -> str:
    parts = str(pair_id).split("_")
    return parts[1] if len(parts) >= 2 else str(pair_id)


def _resolve_top_heads_path(pair_id: str, model_key: str, raw: str) -> Path:
    if str(raw).strip():
        return Path(str(raw)).resolve()
    lang = _lang_from_pair(pair_id)
    candidates = [
        PROJECT_ROOT / "artifacts" / "phase5_attribution" / f"top_heads_{model_key}_{lang}.json",
        PROJECT_ROOT / "artifacts" / "phase5_attribution" / f"top_heads_{model_key}_{lang}_multilang.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _load_top_heads(path: Path, *, top_n: int, max_layer: int) -> List[Dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    out: List[Dict[str, Any]] = []
    for row in rows:
        layer = int(row["layer"])
        if layer > int(max_layer):
            continue
        out.append(
            {
                "rank": int(row.get("rank", len(out) + 1)),
                "layer": layer,
                "head": int(row["head"]),
                "effect": float(row.get("effect", float("nan"))),
            }
        )
        if len(out) >= max(1, int(top_n)):
            break
    return out


def _run_teacher_forced(
    *,
    model: Any,
    input_ids: torch.Tensor,
    target_ids: List[int],
    target_id: int,
    device: str,
    hooks: Optional[List[Any]] = None,
) -> Dict[str, float]:
    active = list(hooks or [])
    try:
        return _teacher_forced_metrics_from_input_ids(
            model=model,
            input_ids=input_ids,
            target_ids=list(target_ids),
            target_id=int(target_id),
            device=str(device),
        )
    finally:
        for hook in reversed(active):
            try:
                hook.remove()
            except Exception:
                pass


def _capture_stagea_feats(
    *,
    model: Any,
    transcoder: Any,
    input_ids: torch.Tensor,
    layer: int,
    position: int,
    hooks: Optional[List[Any]] = None,
) -> torch.Tensor:
    active = list(hooks or [])
    try:
        mlp_in, _ = _extract_mlp_io_at_position_from_input_ids(
            model=model,
            input_ids=input_ids,
            layer=int(layer),
            position=int(position),
        )
        return transcoder.encode(mlp_in.unsqueeze(0)).squeeze(0).detach().float().cpu()
    finally:
        for hook in reversed(active):
            try:
                hook.remove()
            except Exception:
                pass


def _safe_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    na = float(torch.norm(a).item())
    nb = float(torch.norm(b).item())
    if not np.isfinite(na) or not np.isfinite(nb) or na <= 0.0 or nb <= 0.0:
        return float("nan")
    return float(torch.dot(a, b).item() / max(1e-12, na * nb))


def _signed_recovery(observed_delta: torch.Tensor, target_delta: torch.Tensor) -> float:
    denom = float(torch.dot(target_delta, target_delta).item())
    if not np.isfinite(denom) or denom <= 0.0:
        return float("nan")
    return float(torch.dot(observed_delta, target_delta).item() / max(1e-12, denom))


def _head_key(layer: int, head: int) -> str:
    return f"L{int(layer):02d}H{int(head):02d}"


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

    top_heads_path = _resolve_top_heads_path(str(args.pair), str(args.model), str(args.top_heads_json))
    if not top_heads_path.exists():
        raise FileNotFoundError(f"Missing top-head artifact: {top_heads_path}")
    candidate_heads = _load_top_heads(
        top_heads_path,
        top_n=int(args.top_n_heads),
        max_layer=int(stagea_best["layer"]),
    )
    if not candidate_heads:
        raise RuntimeError(
            f"No candidate heads at or before Stage-A layer {stagea_best['layer']} in {top_heads_path}"
        )

    out_root = (
        Path(args.out).resolve()
        if str(args.out).strip()
        else PROJECT_ROOT / "paper2_fidelity_calibrated" / "results" / "head_to_mlp_edge_attribution" / str(args.pair) / str(args.model)
    )
    out_root.mkdir(parents=True, exist_ok=True)

    eval_rows = list(pair_bundle["eval_rows"][: max(1, int(args.max_items))])
    item_rows: List[Dict[str, Any]] = []

    log(
        f"Running maximal G3 edge-attribution: pair={args.pair} model={args.model} items={len(eval_rows)} "
        f"stageA=(layer={stagea_best['layer']}, topk={stagea_best['topk']}, variant={stagea_best['variant']}) "
        f"heads={len(candidate_heads)}"
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

        selected_idx = torch.tensor(list(packet["selected_idx"]), dtype=torch.long)
        if selected_idx.numel() == 0:
            continue
        latent_dim = int(packet["icl_feats"].numel())
        selected_set = {int(i) for i in selected_idx.tolist()}
        random_pool = [i for i in range(int(latent_dim)) if int(i) not in selected_set]
        rng = np.random.default_rng(int(args.seed) + int(item_idx) * 1009)
        random_idx_list = sorted(rng.choice(np.array(random_pool, dtype=np.int64), size=int(selected_idx.numel()), replace=False).tolist()) if len(random_pool) >= int(selected_idx.numel()) else list(selected_set)
        random_idx = torch.tensor(random_idx_list, dtype=torch.long)

        stagea_layer = int(stagea_best["layer"])
        zs_pos = int(packet["zs_patch_position"])
        icl_pos = int(packet["icl_feature_position"])

        zs_metrics = _run_teacher_forced(
            model=model,
            input_ids=packet["zs_input_ids"],
            target_ids=target_ids,
            target_id=target_id,
            device=device,
        )
        icl_metrics = _run_teacher_forced(
            model=model,
            input_ids=packet["icl_input_ids"],
            target_ids=target_ids,
            target_id=target_id,
            device=device,
        )
        stagea_patch_hook = register_transcoder_feature_patch_hook(
            model,
            transcoder,
            stagea_layer,
            packet["patch_feats"],
            patch_position=zs_pos,
            target_output_norm=packet["target_output_norm"],
        )
        stagea_patch_metrics = _run_teacher_forced(
            model=model,
            input_ids=packet["zs_input_ids"],
            target_ids=target_ids,
            target_id=target_id,
            device=device,
            hooks=[stagea_patch_hook],
        )

        zs_feats = packet["selector_ref_feats"].detach().float().cpu()
        icl_feats = packet["icl_feats"].detach().float().cpu()
        target_delta = (icl_feats[selected_idx] - zs_feats[selected_idx]).detach().float()

        layers_needed = sorted({int(row["layer"]) for row in candidate_heads})
        donor_icl = capture_o_proj_inputs_at_position(model, packet["icl_input_ids"], layers_needed, icl_pos)

        for head_row in candidate_heads:
            head_layer = int(head_row["layer"])
            head_idx = int(head_row["head"])
            head_group = {int(head_layer): [int(head_idx)]}

            head_patch_hook = register_attention_head_replace_hooks(
                model,
                donor_icl,
                head_group,
                replace_position=zs_pos,
            )
            head_patch_metrics = _run_teacher_forced(
                model=model,
                input_ids=packet["zs_input_ids"],
                target_ids=target_ids,
                target_id=target_id,
                device=device,
                hooks=[head_patch_hook],
            )

            head_patch_hook = register_attention_head_replace_hooks(
                model,
                donor_icl,
                head_group,
                replace_position=zs_pos,
            )
            head_patch_feats = _capture_stagea_feats(
                model=model,
                transcoder=transcoder,
                input_ids=packet["zs_input_ids"],
                layer=stagea_layer,
                position=zs_pos,
                hooks=[head_patch_hook],
            )
            observed_delta = (head_patch_feats[selected_idx] - zs_feats[selected_idx]).detach().float()

            head_patch_hook = register_attention_head_replace_hooks(
                model,
                donor_icl,
                head_group,
                replace_position=zs_pos,
            )
            feature_ablation_hook = register_transcoder_feature_ablation_hook(
                model,
                transcoder,
                stagea_layer,
                selected_idx.to(device=device),
                ablate_position=zs_pos,
            )
            mediated_metrics = _run_teacher_forced(
                model=model,
                input_ids=packet["zs_input_ids"],
                target_ids=target_ids,
                target_id=target_id,
                device=device,
                hooks=[head_patch_hook, feature_ablation_hook],
            )

            head_patch_hook = register_attention_head_replace_hooks(
                model,
                donor_icl,
                head_group,
                replace_position=zs_pos,
            )
            random_feature_ablation_hook = register_transcoder_feature_ablation_hook(
                model,
                transcoder,
                stagea_layer,
                random_idx.to(device=device),
                ablate_position=zs_pos,
            )
            random_mediated_metrics = _run_teacher_forced(
                model=model,
                input_ids=packet["zs_input_ids"],
                target_ids=target_ids,
                target_id=target_id,
                device=device,
                hooks=[head_patch_hook, random_feature_ablation_hook],
            )

            item_rows.append(
                {
                    "pair": str(args.pair),
                    "model": str(args.model),
                    "seed": int(args.seed),
                    "item_index": int(item_idx - 1),
                    "word_ood": str(word["ood"]),
                    "word_hindi": str(word["hindi"]),
                    "patch_layer": int(stagea_layer),
                    "patch_topk": int(stagea_best["topk"]),
                    "patch_variant": str(stagea_best["variant"]),
                    "selected_feature_indices": list(packet["selected_idx"]),
                    "random_feature_indices": list(random_idx_list),
                    "head_rank": int(head_row["rank"]),
                    "head_layer": int(head_layer),
                    "head": int(head_idx),
                    "head_effect": float(head_row.get("effect", float("nan"))),
                    "zs_first_prob": float(zs_metrics["first_prob"]),
                    "icl_first_prob": float(icl_metrics["first_prob"]),
                    "stagea_patch_first_prob": float(stagea_patch_metrics["first_prob"]),
                    "head_patch_first_prob": float(head_patch_metrics["first_prob"]),
                    "head_patch_selected_feature_ablation_first_prob": float(mediated_metrics["first_prob"]),
                    "head_patch_random_feature_ablation_first_prob": float(random_mediated_metrics["first_prob"]),
                    "zs_first_logit": float(zs_metrics["first_logit"]),
                    "icl_first_logit": float(icl_metrics["first_logit"]),
                    "stagea_patch_first_logit": float(stagea_patch_metrics["first_logit"]),
                    "head_patch_first_logit": float(head_patch_metrics["first_logit"]),
                    "head_patch_selected_feature_ablation_first_logit": float(mediated_metrics["first_logit"]),
                    "head_patch_random_feature_ablation_first_logit": float(random_mediated_metrics["first_logit"]),
                    "zs_target_pos1_nll": float(zs_metrics["target_pos1_nll"]),
                    "icl_target_pos1_nll": float(icl_metrics["target_pos1_nll"]),
                    "stagea_patch_target_pos1_nll": float(stagea_patch_metrics["target_pos1_nll"]),
                    "head_patch_target_pos1_nll": float(head_patch_metrics["target_pos1_nll"]),
                    "head_patch_selected_feature_ablation_target_pos1_nll": float(mediated_metrics["target_pos1_nll"]),
                    "head_patch_random_feature_ablation_target_pos1_nll": float(random_mediated_metrics["target_pos1_nll"]),
                    "head_only_delta_first_prob": float(head_patch_metrics["first_prob"] - zs_metrics["first_prob"]),
                    "head_only_delta_first_logit": float(head_patch_metrics["first_logit"] - zs_metrics["first_logit"]),
                    "head_only_delta_target_pos1_nll": float(zs_metrics["target_pos1_nll"] - head_patch_metrics["target_pos1_nll"]),
                    "feature_mediated_drop_first_prob": float(head_patch_metrics["first_prob"] - mediated_metrics["first_prob"]),
                    "feature_mediated_drop_first_logit": float(head_patch_metrics["first_logit"] - mediated_metrics["first_logit"]),
                    "feature_mediated_drop_target_pos1_nll": float(mediated_metrics["target_pos1_nll"] - head_patch_metrics["target_pos1_nll"]),
                    "random_feature_mediated_drop_first_prob": float(head_patch_metrics["first_prob"] - random_mediated_metrics["first_prob"]),
                    "random_feature_mediated_drop_first_logit": float(head_patch_metrics["first_logit"] - random_mediated_metrics["first_logit"]),
                    "random_feature_mediated_drop_target_pos1_nll": float(random_mediated_metrics["target_pos1_nll"] - head_patch_metrics["target_pos1_nll"]),
                    "selected_feature_target_delta_l1": float(torch.sum(torch.abs(target_delta)).item()),
                    "selected_feature_observed_delta_l1": float(torch.sum(torch.abs(observed_delta)).item()),
                    "selected_feature_recovery_l1_ratio": float(
                        torch.sum(torch.abs(observed_delta)).item() / max(1e-12, torch.sum(torch.abs(target_delta)).item())
                    ),
                    "selected_feature_delta_cosine": float(_safe_cos(observed_delta, target_delta)),
                    "selected_feature_signed_recovery": float(_signed_recovery(observed_delta, target_delta)),
                }
            )

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in item_rows:
        grouped[_head_key(int(row["head_layer"]), int(row["head"]))].append(row)

    summary_rows: List[Dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        summary_rows.append(
            {
                "head": key,
                "n_items": int(len(rows)),
                "mean_head_rank": float(np.nanmean([row["head_rank"] for row in rows])),
                "mean_head_effect": float(np.nanmean([row["head_effect"] for row in rows])),
                "mean_head_only_delta_first_prob": float(np.nanmean([row["head_only_delta_first_prob"] for row in rows])),
                "mean_head_only_delta_first_logit": float(np.nanmean([row["head_only_delta_first_logit"] for row in rows])),
                "mean_head_only_delta_target_pos1_nll": float(np.nanmean([row["head_only_delta_target_pos1_nll"] for row in rows])),
                "mean_feature_mediated_drop_first_prob": float(np.nanmean([row["feature_mediated_drop_first_prob"] for row in rows])),
                "mean_feature_mediated_drop_first_logit": float(np.nanmean([row["feature_mediated_drop_first_logit"] for row in rows])),
                "mean_feature_mediated_drop_target_pos1_nll": float(np.nanmean([row["feature_mediated_drop_target_pos1_nll"] for row in rows])),
                "mean_random_feature_mediated_drop_first_prob": float(np.nanmean([row["random_feature_mediated_drop_first_prob"] for row in rows])),
                "mean_random_feature_mediated_drop_first_logit": float(np.nanmean([row["random_feature_mediated_drop_first_logit"] for row in rows])),
                "mean_random_feature_mediated_drop_target_pos1_nll": float(np.nanmean([row["random_feature_mediated_drop_target_pos1_nll"] for row in rows])),
                "mean_selected_feature_recovery_l1_ratio": float(np.nanmean([row["selected_feature_recovery_l1_ratio"] for row in rows])),
                "mean_selected_feature_delta_cosine": float(np.nanmean([row["selected_feature_delta_cosine"] for row in rows])),
                "mean_selected_feature_signed_recovery": float(np.nanmean([row["selected_feature_signed_recovery"] for row in rows])),
            }
        )

    payload = {
        "experiment": "head_to_mlp_edge_attribution",
        "variant": "maximal_head_patch_feature_mediation",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": str(args.model),
        "pair": str(args.pair),
        "seed": int(args.seed),
        "stagea_path": str(stagea_path),
        "top_heads_path": str(top_heads_path),
        "stagea_best": stagea_best,
        "top_heads_used": candidate_heads,
        "summary_by_head": summary_rows,
        "item_rows": item_rows,
    }
    _write_json(out_root / "head_to_mlp_edge_attribution.json", payload)
    log(f"Saved: {out_root / 'head_to_mlp_edge_attribution.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
