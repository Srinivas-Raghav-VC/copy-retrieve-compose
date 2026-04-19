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
    _teacher_forced_metrics_from_input_ids,
    load_model,
    register_attention_head_ablation_hook,
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
from paper2_fidelity_calibrated.phase23_common import (  # noqa: E402
    get_attn_module,
    infer_num_heads,
    register_keep_only_transcoder_features_hook,
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
    ap = argparse.ArgumentParser(description="Circuit sufficiency panel using G6-ranked core features.")
    ap.add_argument("--model", type=str, default="4b", choices=["1b", "4b"])
    ap.add_argument("--pair", type=str, default="aksharantar_hin_latin")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-icl", type=int, default=64)
    ap.add_argument("--n-select", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--max-items", type=int, default=20)
    ap.add_argument("--core-features", type=int, default=8)
    ap.add_argument("--top-n-heads", type=int, default=8)
    ap.add_argument("--stagea", type=str, default="")
    ap.add_argument("--feature-knockout-json", type=str, default="")
    ap.add_argument("--top-heads-json", type=str, default="")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--external-only", action="store_true")
    ap.add_argument("--require-external-sources", action="store_true")
    ap.add_argument("--min-pool-size", type=int, default=500)
    ap.add_argument("--out", type=str, default="")
    return ap.parse_args()


def _lang_from_pair(pair_id: str) -> str:
    parts = str(pair_id).split("_")
    return parts[1] if len(parts) >= 2 else str(pair_id)


def _resolve_feature_knockout_path(pair_id: str, model_key: str, raw: str) -> Path:
    if str(raw).strip():
        return Path(str(raw)).resolve()
    return PROJECT_ROOT / "paper2_fidelity_calibrated" / "results" / "feature_knockout_panel" / str(pair_id) / str(model_key) / "feature_knockout_panel.json"


def _resolve_top_heads_path(pair_id: str, model_key: str, raw: str) -> Path:
    if str(raw).strip():
        return Path(str(raw)).resolve()
    lang = _lang_from_pair(pair_id)
    candidates = [
        PROJECT_ROOT / "artifacts" / "phase5_attribution" / f"top_heads_{model_key}_{lang}_multilang.json",
        PROJECT_ROOT / "artifacts" / "phase5_attribution" / f"top_heads_{model_key}_{lang}.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _load_core_feature_indices(path: Path, limit: int) -> List[int]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    rows = list(obj.get("summary_by_feature_index") or [])
    rows.sort(key=lambda row: float(row.get("mean_drop_from_full_patch_first_prob", float("-inf"))), reverse=True)
    out = [int(row["feature_index"]) for row in rows[: max(1, int(limit))]]
    if not out:
        raise RuntimeError(f"No core features found in {path}")
    return out


def _load_top_heads(path: Path, top_n: int) -> List[Dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [
        {
            "rank": int(row.get("rank", idx + 1)),
            "layer": int(row["layer"]),
            "head": int(row["head"]),
            "effect": float(row.get("effect", float("nan"))),
        }
        for idx, row in enumerate(rows[: max(1, int(top_n))])
    ]


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


def _build_noncircuit_ablation_hooks(model: Any, top_heads: List[Dict[str, Any]], position: int) -> List[Any]:
    grouped: Dict[int, List[int]] = defaultdict(list)
    for row in top_heads:
        grouped[int(row["layer"])] .append(int(row["head"]))
    hooks: List[Any] = []
    for layer, heads in sorted(grouped.items()):
        attn_module = get_attn_module(model, int(layer))
        num_heads = infer_num_heads(model, attn_module)
        keep = {int(h) for h in heads}
        ablate = [int(h) for h in range(int(num_heads)) if int(h) not in keep]
        hooks.append(
            register_attention_head_ablation_hook(
                model,
                int(layer),
                ablate,
                ablate_position=int(position),
            )
        )
    return hooks


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

    feature_knockout_path = _resolve_feature_knockout_path(str(args.pair), str(args.model), str(args.feature_knockout_json))
    if not feature_knockout_path.exists():
        raise FileNotFoundError(f"Missing feature-knockout artifact: {feature_knockout_path}")
    core_feature_indices = _load_core_feature_indices(feature_knockout_path, int(args.core_features))
    core_feature_tensor = torch.tensor(core_feature_indices, device=device, dtype=torch.long)

    top_heads_path = _resolve_top_heads_path(str(args.pair), str(args.model), str(args.top_heads_json))
    if not top_heads_path.exists():
        raise FileNotFoundError(f"Missing top-head artifact: {top_heads_path}")
    top_heads = _load_top_heads(top_heads_path, int(args.top_n_heads))

    out_root = (
        Path(args.out).resolve()
        if str(args.out).strip()
        else PROJECT_ROOT / "paper2_fidelity_calibrated" / "results" / "circuit_sufficiency" / str(args.pair) / str(args.model)
    )
    out_root.mkdir(parents=True, exist_ok=True)

    eval_rows = list(pair_bundle["eval_rows"][: max(1, int(args.max_items))])
    item_rows: List[Dict[str, Any]] = []

    log(
        f"Running G4 circuit sufficiency: pair={args.pair} model={args.model} items={len(eval_rows)} core_features={len(core_feature_indices)} heads={len(top_heads)}"
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

        zs_metrics = _run_condition(
            model=model,
            tokenizer=tokenizer,
            input_ids=packet["zs_input_ids"],
            target_text=str(word["hindi"]),
            target_script=pair_bundle["output_script_name"],
            device=device,
            max_new_tokens=int(args.max_new_tokens),
        )
        icl_metrics = _run_condition(
            model=model,
            tokenizer=tokenizer,
            input_ids=packet["icl_input_ids"],
            target_text=str(word["hindi"]),
            target_script=pair_bundle["output_script_name"],
            device=device,
            max_new_tokens=int(args.max_new_tokens),
        )
        stagea_patch_hook = register_transcoder_feature_patch_hook(
            model,
            transcoder,
            int(stagea_best["layer"]),
            packet["patch_feats"],
            patch_position=int(packet["zs_patch_position"]),
            target_output_norm=packet["target_output_norm"],
        )
        stagea_patch_metrics = _run_condition(
            model=model,
            tokenizer=tokenizer,
            input_ids=packet["zs_input_ids"],
            target_text=str(word["hindi"]),
            target_script=pair_bundle["output_script_name"],
            device=device,
            max_new_tokens=int(args.max_new_tokens),
            hooks=[stagea_patch_hook],
        )

        keep_only_hook = register_keep_only_transcoder_features_hook(
            model,
            transcoder,
            int(stagea_best["layer"]),
            core_feature_tensor,
            keep_position=int(packet["icl_feature_position"]),
        )
        keep_only_metrics = _run_condition(
            model=model,
            tokenizer=tokenizer,
            input_ids=packet["icl_input_ids"],
            target_text=str(word["hindi"]),
            target_script=pair_bundle["output_script_name"],
            device=device,
            max_new_tokens=int(args.max_new_tokens),
            hooks=[keep_only_hook],
        )

        keep_only_hook = register_keep_only_transcoder_features_hook(
            model,
            transcoder,
            int(stagea_best["layer"]),
            core_feature_tensor,
            keep_position=int(packet["icl_feature_position"]),
        )
        noncircuit_head_hooks = _build_noncircuit_ablation_hooks(
            model,
            top_heads,
            position=int(packet["icl_feature_position"]),
        )
        circuit_only_metrics = _run_condition(
            model=model,
            tokenizer=tokenizer,
            input_ids=packet["icl_input_ids"],
            target_text=str(word["hindi"]),
            target_script=pair_bundle["output_script_name"],
            device=device,
            max_new_tokens=int(args.max_new_tokens),
            hooks=[keep_only_hook] + noncircuit_head_hooks,
        )

        item_rows.append(
            {
                "pair": str(args.pair),
                "model": str(args.model),
                "seed": int(args.seed),
                "item_index": int(item_idx - 1),
                "word_ood": str(word["ood"]),
                "word_hindi": str(word["hindi"]),
                "core_feature_indices": core_feature_indices,
                "top_heads": top_heads,
                "zs_exact_match": float(zs_metrics["exact_match"]),
                "icl_exact_match": float(icl_metrics["exact_match"]),
                "stagea_patch_exact_match": float(stagea_patch_metrics["exact_match"]),
                "keep_only_exact_match": float(keep_only_metrics["exact_match"]),
                "circuit_only_exact_match": float(circuit_only_metrics["exact_match"]),
                "zs_first_entry_correct": float(zs_metrics["first_entry_correct"]),
                "icl_first_entry_correct": float(icl_metrics["first_entry_correct"]),
                "stagea_patch_first_entry_correct": float(stagea_patch_metrics["first_entry_correct"]),
                "keep_only_first_entry_correct": float(keep_only_metrics["first_entry_correct"]),
                "circuit_only_first_entry_correct": float(circuit_only_metrics["first_entry_correct"]),
                "zs_first_prob": float(zs_metrics["first_prob"]),
                "icl_first_prob": float(icl_metrics["first_prob"]),
                "stagea_patch_first_prob": float(stagea_patch_metrics["first_prob"]),
                "keep_only_first_prob": float(keep_only_metrics["first_prob"]),
                "circuit_only_first_prob": float(circuit_only_metrics["first_prob"]),
                "zs_target_pos1_nll": float(zs_metrics["target_pos1_nll"]),
                "icl_target_pos1_nll": float(icl_metrics["target_pos1_nll"]),
                "stagea_patch_target_pos1_nll": float(stagea_patch_metrics["target_pos1_nll"]),
                "keep_only_target_pos1_nll": float(keep_only_metrics["target_pos1_nll"]),
                "circuit_only_target_pos1_nll": float(circuit_only_metrics["target_pos1_nll"]),
            }
        )

    summary = {
        "n_items": int(len(item_rows)),
        "zs_exact_match": float(np.nanmean([row["zs_exact_match"] for row in item_rows])),
        "icl_exact_match": float(np.nanmean([row["icl_exact_match"] for row in item_rows])),
        "stagea_patch_exact_match": float(np.nanmean([row["stagea_patch_exact_match"] for row in item_rows])),
        "keep_only_exact_match": float(np.nanmean([row["keep_only_exact_match"] for row in item_rows])),
        "circuit_only_exact_match": float(np.nanmean([row["circuit_only_exact_match"] for row in item_rows])),
        "zs_first_entry_correct": float(np.nanmean([row["zs_first_entry_correct"] for row in item_rows])),
        "icl_first_entry_correct": float(np.nanmean([row["icl_first_entry_correct"] for row in item_rows])),
        "stagea_patch_first_entry_correct": float(np.nanmean([row["stagea_patch_first_entry_correct"] for row in item_rows])),
        "keep_only_first_entry_correct": float(np.nanmean([row["keep_only_first_entry_correct"] for row in item_rows])),
        "circuit_only_first_entry_correct": float(np.nanmean([row["circuit_only_first_entry_correct"] for row in item_rows])),
        "zs_first_prob": float(np.nanmean([row["zs_first_prob"] for row in item_rows])),
        "icl_first_prob": float(np.nanmean([row["icl_first_prob"] for row in item_rows])),
        "stagea_patch_first_prob": float(np.nanmean([row["stagea_patch_first_prob"] for row in item_rows])),
        "keep_only_first_prob": float(np.nanmean([row["keep_only_first_prob"] for row in item_rows])),
        "circuit_only_first_prob": float(np.nanmean([row["circuit_only_first_prob"] for row in item_rows])),
        "zs_target_pos1_nll": float(np.nanmean([row["zs_target_pos1_nll"] for row in item_rows])),
        "icl_target_pos1_nll": float(np.nanmean([row["icl_target_pos1_nll"] for row in item_rows])),
        "stagea_patch_target_pos1_nll": float(np.nanmean([row["stagea_patch_target_pos1_nll"] for row in item_rows])),
        "keep_only_target_pos1_nll": float(np.nanmean([row["keep_only_target_pos1_nll"] for row in item_rows])),
        "circuit_only_target_pos1_nll": float(np.nanmean([row["circuit_only_target_pos1_nll"] for row in item_rows])),
    }

    payload = {
        "experiment": "circuit_sufficiency",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": str(args.model),
        "pair": str(args.pair),
        "seed": int(args.seed),
        "stagea_path": str(stagea_path),
        "feature_knockout_path": str(feature_knockout_path),
        "top_heads_path": str(top_heads_path),
        "stagea_best": stagea_best,
        "core_feature_indices": core_feature_indices,
        "top_heads": top_heads,
        "summary": summary,
        "item_rows": item_rows,
    }
    _write_json(out_root / "circuit_sufficiency.json", payload)
    log(f"Saved: {out_root / 'circuit_sufficiency.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
