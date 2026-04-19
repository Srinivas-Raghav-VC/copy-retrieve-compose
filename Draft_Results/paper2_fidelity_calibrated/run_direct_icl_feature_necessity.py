#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
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
    _teacher_forced_metrics_from_input_ids,
    load_model,
    register_transcoder_feature_ablation_hook,
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
    ap = argparse.ArgumentParser(description="Directly ablate top-ranked features inside full ICL to test natural-state necessity.")
    ap.add_argument("--model", type=str, default="4b", choices=["1b", "4b"])
    ap.add_argument("--pair", type=str, default="aksharantar_hin_latin")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-icl", type=int, default=64)
    ap.add_argument("--n-select", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--max-items", type=int, default=20)
    ap.add_argument("--core-features", type=int, default=8)
    ap.add_argument("--stagea", type=str, default="")
    ap.add_argument("--feature-knockout-json", type=str, default="")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--external-only", action="store_true")
    ap.add_argument("--require-external-sources", action="store_true")
    ap.add_argument("--min-pool-size", type=int, default=500)
    ap.add_argument("--out", type=str, default="")
    return ap.parse_args()


def _resolve_feature_knockout_path(pair_id: str, model_key: str, raw: str) -> Path:
    if str(raw).strip():
        return Path(str(raw)).resolve()
    return PROJECT_ROOT / "paper2_fidelity_calibrated" / "results" / "feature_knockout_panel" / str(pair_id) / str(model_key) / "feature_knockout_panel.json"


def _load_core_features(path: Path, limit: int) -> List[int]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    rows = list(obj.get("summary_by_feature_index") or [])
    rows.sort(key=lambda row: float(row.get("mean_drop_from_full_patch_first_prob", float("-inf"))), reverse=True)
    out = [int(row["feature_index"]) for row in rows[: max(1, int(limit))]]
    if not out:
        raise RuntimeError(f"No core features found in {path}")
    return out


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
        mask = torch.ones_like(input_ids)
        pad_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", 0)
        with torch.inference_mode():
            out = model.generate(
                input_ids,
                attention_mask=mask,
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

    feature_path = _resolve_feature_knockout_path(str(args.pair), str(args.model), str(args.feature_knockout_json))
    core_features = _load_core_features(feature_path, int(args.core_features))
    core_tensor = torch.tensor(core_features, device=device, dtype=torch.long)

    out_root = (
        Path(args.out).resolve()
        if str(args.out).strip()
        else PROJECT_ROOT / "paper2_fidelity_calibrated" / "results" / "direct_icl_feature_necessity" / str(args.pair) / str(args.model)
    )
    out_root.mkdir(parents=True, exist_ok=True)

    eval_rows = list(pair_bundle["eval_rows"][: max(1, int(args.max_items))])
    item_rows: List[Dict[str, Any]] = []

    log(f"Running direct ICL feature necessity: pair={args.pair} model={args.model} items={len(eval_rows)} core_features={len(core_features)}")

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
        latent_dim = int(packet["icl_feats"].numel())
        pool = [i for i in range(int(latent_dim)) if int(i) not in set(core_features)]
        rng = random.Random(int(args.seed) + int(item_idx) * 17)
        rng.shuffle(pool)
        random_features = sorted(pool[: len(core_features)]) if len(pool) >= len(core_features) else list(core_features)
        random_tensor = torch.tensor(random_features, device=device, dtype=torch.long)

        icl_metrics = _run_condition(
            model=model,
            tokenizer=tokenizer,
            input_ids=packet["icl_input_ids"],
            target_text=str(word["hindi"]),
            target_script=pair_bundle["output_script_name"],
            device=device,
            max_new_tokens=int(args.max_new_tokens),
        )
        core_ablate = register_transcoder_feature_ablation_hook(
            model,
            transcoder,
            int(stagea_best["layer"]),
            core_tensor,
            ablate_position=int(packet["icl_feature_position"]),
        )
        core_metrics = _run_condition(
            model=model,
            tokenizer=tokenizer,
            input_ids=packet["icl_input_ids"],
            target_text=str(word["hindi"]),
            target_script=pair_bundle["output_script_name"],
            device=device,
            max_new_tokens=int(args.max_new_tokens),
            hooks=[core_ablate],
        )
        rand_ablate = register_transcoder_feature_ablation_hook(
            model,
            transcoder,
            int(stagea_best["layer"]),
            random_tensor,
            ablate_position=int(packet["icl_feature_position"]),
        )
        rand_metrics = _run_condition(
            model=model,
            tokenizer=tokenizer,
            input_ids=packet["icl_input_ids"],
            target_text=str(word["hindi"]),
            target_script=pair_bundle["output_script_name"],
            device=device,
            max_new_tokens=int(args.max_new_tokens),
            hooks=[rand_ablate],
        )
        item_rows.append(
            {
                "pair": str(args.pair),
                "model": str(args.model),
                "seed": int(args.seed),
                "item_index": int(item_idx - 1),
                "word_ood": str(word["ood"]),
                "word_hindi": str(word["hindi"]),
                "core_features": core_features,
                "random_features": random_features,
                "icl_first_prob": float(icl_metrics["first_prob"]),
                "core_ablate_first_prob": float(core_metrics["first_prob"]),
                "random_ablate_first_prob": float(rand_metrics["first_prob"]),
                "icl_first_logit": float(icl_metrics["first_logit"]),
                "core_ablate_first_logit": float(core_metrics["first_logit"]),
                "random_ablate_first_logit": float(rand_metrics["first_logit"]),
                "icl_target_pos1_nll": float(icl_metrics["target_pos1_nll"]),
                "core_ablate_target_pos1_nll": float(core_metrics["target_pos1_nll"]),
                "random_ablate_target_pos1_nll": float(rand_metrics["target_pos1_nll"]),
                "icl_exact_match": float(icl_metrics["exact_match"]),
                "core_ablate_exact_match": float(core_metrics["exact_match"]),
                "random_ablate_exact_match": float(rand_metrics["exact_match"]),
                "core_drop_first_prob": float(icl_metrics["first_prob"] - core_metrics["first_prob"]),
                "random_drop_first_prob": float(icl_metrics["first_prob"] - rand_metrics["first_prob"]),
                "core_drop_first_logit": float(icl_metrics["first_logit"] - core_metrics["first_logit"]),
                "random_drop_first_logit": float(icl_metrics["first_logit"] - rand_metrics["first_logit"]),
                "core_increase_target_pos1_nll": float(core_metrics["target_pos1_nll"] - icl_metrics["target_pos1_nll"]),
                "random_increase_target_pos1_nll": float(rand_metrics["target_pos1_nll"] - icl_metrics["target_pos1_nll"]),
            }
        )

    summary = {
        "n_items": float(len(item_rows)),
        "mean_core_drop_first_prob": float(np.nanmean([row["core_drop_first_prob"] for row in item_rows])),
        "mean_random_drop_first_prob": float(np.nanmean([row["random_drop_first_prob"] for row in item_rows])),
        "mean_core_drop_first_logit": float(np.nanmean([row["core_drop_first_logit"] for row in item_rows])),
        "mean_random_drop_first_logit": float(np.nanmean([row["random_drop_first_logit"] for row in item_rows])),
        "mean_core_increase_target_pos1_nll": float(np.nanmean([row["core_increase_target_pos1_nll"] for row in item_rows])),
        "mean_random_increase_target_pos1_nll": float(np.nanmean([row["random_increase_target_pos1_nll"] for row in item_rows])),
    }

    payload = {
        "experiment": "direct_icl_feature_necessity",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": str(args.model),
        "pair": str(args.pair),
        "seed": int(args.seed),
        "stagea_best": stagea_best,
        "core_features": core_features,
        "summary": summary,
        "item_rows": item_rows,
    }
    _write_json(out_root / "direct_icl_feature_necessity.json", payload)
    log(f"Saved: {out_root / 'direct_icl_feature_necessity.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
