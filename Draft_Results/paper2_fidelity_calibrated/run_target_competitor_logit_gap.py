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
from core import load_model, register_transcoder_feature_patch_hook, set_all_seeds  # noqa: E402
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
    ap = argparse.ArgumentParser(description="Track target-vs-top-competitor logit gap under ZS / ICL / patched.")
    ap.add_argument("--model", type=str, default="4b", choices=["1b", "4b"])
    ap.add_argument("--pair", type=str, default="aksharantar_hin_latin")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-icl", type=int, default=64)
    ap.add_argument("--n-select", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--max-items", type=int, default=30)
    ap.add_argument("--stagea", type=str, default="")
    ap.add_argument("--external-only", action="store_true")
    ap.add_argument("--require-external-sources", action="store_true")
    ap.add_argument("--min-pool-size", type=int, default=500)
    ap.add_argument("--out", type=str, default="")
    return ap.parse_args()


def _first_step_gap(model: Any, input_ids: torch.Tensor, target_id: int, *, hooks: Optional[List[Any]] = None) -> Dict[str, float]:
    active = list(hooks or [])
    try:
        full_input_ids = torch.cat(
            [input_ids, torch.tensor([[int(target_id)]], device=input_ids.device, dtype=input_ids.dtype)],
            dim=1,
        )
        start = int(input_ids.shape[1] - 1)
        with torch.inference_mode():
            outputs = model(input_ids=full_input_ids, use_cache=False)
        logits = outputs.logits[0, start, :].float()
        target_logit = float(logits[int(target_id)].item())
        logits_masked = logits.clone()
        logits_masked[int(target_id)] = -float("inf")
        competitor_id = int(torch.argmax(logits_masked).item())
        competitor_logit = float(logits[int(competitor_id)].item())
        return {
            "target_id": int(target_id),
            "competitor_id": int(competitor_id),
            "target_logit": float(target_logit),
            "competitor_logit": float(competitor_logit),
            "target_minus_competitor_logit": float(target_logit - competitor_logit),
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

    out_root = (
        Path(args.out).resolve()
        if str(args.out).strip()
        else PROJECT_ROOT / "paper2_fidelity_calibrated" / "results" / "target_competitor_logit_gap" / str(args.pair) / str(args.model)
    )
    out_root.mkdir(parents=True, exist_ok=True)

    eval_rows = list(pair_bundle["eval_rows"][: max(1, int(args.max_items))])
    item_rows: List[Dict[str, Any]] = []
    log(f"Running target-competitor gap analysis: pair={args.pair} model={args.model} items={len(eval_rows)}")

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
        target_id = int(packet["target_id"])
        if target_id < 0:
            continue
        zs_gap = _first_step_gap(model, packet["zs_input_ids"], target_id)
        icl_gap = _first_step_gap(model, packet["icl_input_ids"], target_id)
        patch_hook = register_transcoder_feature_patch_hook(
            model,
            transcoder,
            int(stagea_best["layer"]),
            packet["patch_feats"],
            patch_position=int(packet["zs_patch_position"]),
            target_output_norm=packet["target_output_norm"],
        )
        patched_gap = _first_step_gap(model, packet["zs_input_ids"], target_id, hooks=[patch_hook])
        item_rows.append(
            {
                "pair": str(args.pair),
                "model": str(args.model),
                "seed": int(args.seed),
                "item_index": int(item_idx - 1),
                "word_ood": str(word["ood"]),
                "word_hindi": str(word["hindi"]),
                **{f"zs_{k}": v for k, v in zs_gap.items()},
                **{f"icl_{k}": v for k, v in icl_gap.items()},
                **{f"patched_{k}": v for k, v in patched_gap.items()},
            }
        )

    summary = {
        "n_items": float(len(item_rows)),
        "mean_zs_gap": float(np.nanmean([row["zs_target_minus_competitor_logit"] for row in item_rows])),
        "mean_icl_gap": float(np.nanmean([row["icl_target_minus_competitor_logit"] for row in item_rows])),
        "mean_patched_gap": float(np.nanmean([row["patched_target_minus_competitor_logit"] for row in item_rows])),
    }

    payload = {
        "experiment": "target_competitor_logit_gap",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": str(args.model),
        "pair": str(args.pair),
        "seed": int(args.seed),
        "stagea_best": stagea_best,
        "summary": summary,
        "item_rows": item_rows,
    }
    _write_json(out_root / "target_competitor_logit_gap.json", payload)
    log(f"Saved: {out_root / 'target_competitor_logit_gap.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
