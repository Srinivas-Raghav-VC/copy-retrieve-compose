#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
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
    register_dense_mlp_output_patch_hook,
    register_transcoder_feature_patch_hook,
    set_all_seeds,
)
from paper2_fidelity_calibrated.eval_utils import (  # noqa: E402
    akshara_cer,
    continuation_akshara_cer,
    first_entry_correct,
    normalize_text,
    script_compliance,
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
    ap = argparse.ArgumentParser(description="Compare latent sparse patching to decoded sparse / dense output replacement.")
    ap.add_argument("--model", type=str, default="4b", choices=["1b", "4b"])
    ap.add_argument("--pair", type=str, default="aksharantar_hin_latin")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-icl", type=int, default=64)
    ap.add_argument("--n-select", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--max-items", type=int, default=20)
    ap.add_argument("--stagea", type=str, default="")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--external-only", action="store_true")
    ap.add_argument("--require-external-sources", action="store_true")
    ap.add_argument("--min-pool-size", type=int, default=500)
    ap.add_argument("--out", type=str, default="")
    return ap.parse_args()


def _run_condition(
    *,
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    target_text: str,
    target_script: str,
    device: str,
    max_new_tokens: int,
    hooks: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    active = list(hooks or [])
    try:
        target_ids = tokenizer.encode(str(target_text), add_special_tokens=False)
        target_id = int(target_ids[0]) if target_ids else -1
        tf = _teacher_forced_metrics_from_input_ids(
            model=model,
            input_ids=input_ids,
            target_ids=target_ids,
            target_id=target_id,
            device=str(device),
        )
        attention_mask = torch.ones_like(input_ids)
        pad_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", 0)
        with torch.inference_mode():
            out = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                use_cache=False,
                pad_token_id=int(pad_id),
            )
        pred = normalize_text(tokenizer.decode(out[0, input_ids.shape[1] :], skip_special_tokens=True).strip())
        gold = normalize_text(target_text)
        cont = continuation_akshara_cer(pred, gold)
        return {
            "prediction": pred,
            "exact_match": float(pred == gold),
            "akshara_cer": float(akshara_cer(pred, gold)),
            "script_compliance": float(script_compliance(pred, target_script)),
            "first_entry_correct": float(first_entry_correct(pred, gold)),
            "continuation_akshara_cer": float(cont),
            "joint_logprob": float(tf.get("joint_logprob", float("nan"))),
            "target_pos1_nll": float(tf.get("target_pos1_nll", float("nan"))),
            "first_prob": float(tf.get("first_prob", float("nan"))),
            "first_logit": float(tf.get("first_logit", float("nan"))),
        }
    finally:
        for hook in reversed(active):
            try:
                hook.remove()
            except Exception:
                pass


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
    transcoder = load_transcoder_for_stagea(model, stagea_best, device)
    layer = int(stagea_best["layer"])

    out_root = (
        Path(args.out).resolve()
        if str(args.out).strip()
        else PROJECT_ROOT / "paper2_fidelity_calibrated" / "results" / "decoded_vs_latent_equivalence" / str(args.pair) / str(args.model)
    )
    out_root.mkdir(parents=True, exist_ok=True)

    eval_rows = list(pair_bundle["eval_rows"][: max(1, int(args.max_items))])
    item_rows: List[Dict[str, Any]] = []
    log(f"Running decoded-vs-latent equivalence: pair={args.pair} model={args.model} items={len(eval_rows)}")

    for item_idx, word in enumerate(eval_rows, start=1):
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
        _, donor_mlp_out = _extract_mlp_io_at_position_from_input_ids(
            model=model,
            input_ids=packet["icl_input_ids"],
            layer=layer,
            position=int(packet["icl_feature_position"]),
        )
        decoded_sparse = transcoder.decode(packet["patch_feats"].unsqueeze(0)).squeeze(0).detach()

        zs_metrics = _run_condition(
            model=model,
            tokenizer=tokenizer,
            input_ids=packet["zs_input_ids"],
            target_text=str(word["hindi"]),
            target_script=pair_bundle["output_script_name"],
            device=device,
            max_new_tokens=int(args.max_new_tokens),
        )
        latent_hook = register_transcoder_feature_patch_hook(
            model,
            transcoder,
            layer,
            packet["patch_feats"],
            patch_position=int(packet["zs_patch_position"]),
            target_output_norm=packet["target_output_norm"],
        )
        latent_metrics = _run_condition(
            model=model,
            tokenizer=tokenizer,
            input_ids=packet["zs_input_ids"],
            target_text=str(word["hindi"]),
            target_script=pair_bundle["output_script_name"],
            device=device,
            max_new_tokens=int(args.max_new_tokens),
            hooks=[latent_hook],
        )
        decoded_hook = register_dense_mlp_output_patch_hook(
            model,
            layer,
            decoded_sparse.to(device),
            patch_position=int(packet["zs_patch_position"]),
        )
        decoded_metrics = _run_condition(
            model=model,
            tokenizer=tokenizer,
            input_ids=packet["zs_input_ids"],
            target_text=str(word["hindi"]),
            target_script=pair_bundle["output_script_name"],
            device=device,
            max_new_tokens=int(args.max_new_tokens),
            hooks=[decoded_hook],
        )
        donor_hook = register_dense_mlp_output_patch_hook(
            model,
            layer,
            donor_mlp_out.to(device),
            patch_position=int(packet["zs_patch_position"]),
        )
        donor_metrics = _run_condition(
            model=model,
            tokenizer=tokenizer,
            input_ids=packet["zs_input_ids"],
            target_text=str(word["hindi"]),
            target_script=pair_bundle["output_script_name"],
            device=device,
            max_new_tokens=int(args.max_new_tokens),
            hooks=[donor_hook],
        )
        item_rows.append(
            {
                "pair": str(args.pair),
                "model": str(args.model),
                "seed": int(args.seed),
                "item_index": int(item_idx - 1),
                "word_ood": str(word["ood"]),
                "word_hindi": str(word["hindi"]),
                "zs_first_prob": float(zs_metrics["first_prob"]),
                "latent_first_prob": float(latent_metrics["first_prob"]),
                "decoded_sparse_first_prob": float(decoded_metrics["first_prob"]),
                "dense_donor_first_prob": float(donor_metrics["first_prob"]),
                "zs_first_logit": float(zs_metrics["first_logit"]),
                "latent_first_logit": float(latent_metrics["first_logit"]),
                "decoded_sparse_first_logit": float(decoded_metrics["first_logit"]),
                "dense_donor_first_logit": float(donor_metrics["first_logit"]),
                "zs_target_pos1_nll": float(zs_metrics["target_pos1_nll"]),
                "latent_target_pos1_nll": float(latent_metrics["target_pos1_nll"]),
                "decoded_sparse_target_pos1_nll": float(decoded_metrics["target_pos1_nll"]),
                "dense_donor_target_pos1_nll": float(donor_metrics["target_pos1_nll"]),
            }
        )

    summary = {
        "n_items": float(len(item_rows)),
        "mean_latent_first_prob": float(np.nanmean([row["latent_first_prob"] for row in item_rows])),
        "mean_decoded_sparse_first_prob": float(np.nanmean([row["decoded_sparse_first_prob"] for row in item_rows])),
        "mean_dense_donor_first_prob": float(np.nanmean([row["dense_donor_first_prob"] for row in item_rows])),
        "mean_latent_target_pos1_nll": float(np.nanmean([row["latent_target_pos1_nll"] for row in item_rows])),
        "mean_decoded_sparse_target_pos1_nll": float(np.nanmean([row["decoded_sparse_target_pos1_nll"] for row in item_rows])),
        "mean_dense_donor_target_pos1_nll": float(np.nanmean([row["dense_donor_target_pos1_nll"] for row in item_rows])),
    }

    payload = {
        "experiment": "decoded_vs_latent_equivalence",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": str(args.model),
        "pair": str(args.pair),
        "seed": int(args.seed),
        "stagea_best": stagea_best,
        "summary": summary,
        "item_rows": item_rows,
    }
    _write_json(out_root / "decoded_vs_latent_equivalence.json", payload)
    log(f"Saved: {out_root / 'decoded_vs_latent_equivalence.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
