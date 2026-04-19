#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import build_task_prompt, load_model, set_all_seeds  # noqa: E402
from paper2_fidelity_calibrated.eval_utils import evaluate_prompt_condition  # noqa: E402
from paper2_fidelity_calibrated.phase1_common import load_pair_split, log  # noqa: E402


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
    ap = argparse.ArgumentParser(description="Sampled leave-k-out contribution analysis over the frozen 64-example ICL bank.")
    ap.add_argument("--model", type=str, default="4b", choices=["1b", "4b"])
    ap.add_argument("--pair", type=str, default="aksharantar_hin_latin")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-icl", type=int, default=64)
    ap.add_argument("--n-select", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--max-items", type=int, default=10)
    ap.add_argument("--ks", type=str, default="1,4,8")
    ap.add_argument("--subsets-per-k", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--external-only", action="store_true")
    ap.add_argument("--require-external-sources", action="store_true")
    ap.add_argument("--min-pool-size", type=int, default=500)
    ap.add_argument("--out", type=str, default="")
    return ap.parse_args()


def _parse_ks(raw: str) -> List[int]:
    vals = sorted(set(int(x.strip()) for x in str(raw or "").split(",") if x.strip()))
    return vals or [1, 4, 8]


def main() -> int:
    args = parse_args()
    set_all_seeds(int(args.seed))
    rng = random.Random(int(args.seed))

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
    model, tokenizer = load_model(str(args.model), device=str(args.device))
    device = str(next(model.parameters()).device)
    ks = _parse_ks(str(args.ks))

    out_root = (
        Path(args.out).resolve()
        if str(args.out).strip()
        else PROJECT_ROOT / "paper2_fidelity_calibrated" / "results" / "leave_k_out_icl_contribution" / str(args.pair) / str(args.model)
    )
    out_root.mkdir(parents=True, exist_ok=True)

    eval_rows = list(pair_bundle["eval_rows"][: max(1, int(args.max_items))])
    icl_examples = list(pair_bundle["icl_examples"])
    item_rows: List[Dict[str, Any]] = []

    log(f"Running sampled leave-k-out contribution: pair={args.pair} model={args.model} items={len(eval_rows)} ks={ks}")

    for item_idx, word in enumerate(eval_rows, start=1):
        full_prompt = build_task_prompt(
            str(word["ood"]),
            icl_examples,
            input_script_name=pair_bundle["input_script_name"],
            source_language=pair_bundle["source_language"],
            output_script_name=pair_bundle["output_script_name"],
            prompt_variant="canonical",
        )
        full_metrics = evaluate_prompt_condition(
            model,
            tokenizer,
            prompt_text=full_prompt,
            target_text=str(word["hindi"]),
            target_script=pair_bundle["output_script_name"],
            device=device,
            max_new_tokens=int(args.max_new_tokens),
        )
        for k in ks:
            k = min(int(k), len(icl_examples))
            for subset_idx in range(max(1, int(args.subsets_per_k))):
                drop = set(rng.sample(range(len(icl_examples)), k))
                kept = [ex for i, ex in enumerate(icl_examples) if i not in drop]
                prompt = build_task_prompt(
                    str(word["ood"]),
                    kept,
                    input_script_name=pair_bundle["input_script_name"],
                    source_language=pair_bundle["source_language"],
                    output_script_name=pair_bundle["output_script_name"],
                    prompt_variant="canonical",
                )
                metrics = evaluate_prompt_condition(
                    model,
                    tokenizer,
                    prompt_text=prompt,
                    target_text=str(word["hindi"]),
                    target_script=pair_bundle["output_script_name"],
                    device=device,
                    max_new_tokens=int(args.max_new_tokens),
                )
                item_rows.append(
                    {
                        "pair": str(args.pair),
                        "model": str(args.model),
                        "seed": int(args.seed),
                        "item_index": int(item_idx - 1),
                        "word_ood": str(word["ood"]),
                        "word_hindi": str(word["hindi"]),
                        "k": int(k),
                        "subset_index": int(subset_idx),
                        "drop_indices": sorted(int(x) for x in drop),
                        "full_first_prob": float(full_metrics["first_prob"]),
                        "leavek_first_prob": float(metrics["first_prob"]),
                        "full_first_logit": float(full_metrics["first_logit"]),
                        "leavek_first_logit": float(metrics["first_logit"]),
                        "full_target_pos1_nll": float(full_metrics["target_pos1_nll"]),
                        "leavek_target_pos1_nll": float(metrics["target_pos1_nll"]),
                        "full_exact_match": float(full_metrics["exact_match"]),
                        "leavek_exact_match": float(metrics["exact_match"]),
                        "drop_first_prob": float(full_metrics["first_prob"] - metrics["first_prob"]),
                        "drop_first_logit": float(full_metrics["first_logit"] - metrics["first_logit"]),
                        "increase_target_pos1_nll": float(metrics["target_pos1_nll"] - full_metrics["target_pos1_nll"]),
                    }
                )

    summary: Dict[int, Dict[str, float]] = {}
    for k in ks:
        rows = [row for row in item_rows if int(row["k"]) == int(k)]
        summary[int(k)] = {
            "n_rows": float(len(rows)),
            "mean_drop_first_prob": float(np.nanmean([row["drop_first_prob"] for row in rows])),
            "mean_drop_first_logit": float(np.nanmean([row["drop_first_logit"] for row in rows])),
            "mean_increase_target_pos1_nll": float(np.nanmean([row["increase_target_pos1_nll"] for row in rows])),
        }

    payload = {
        "experiment": "leave_k_out_icl_contribution",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": str(args.model),
        "pair": str(args.pair),
        "seed": int(args.seed),
        "ks": ks,
        "subsets_per_k": int(args.subsets_per_k),
        "summary": summary,
        "item_rows": item_rows,
    }
    _write_json(out_root / "leave_k_out_icl_contribution.json", payload)
    log(f"Saved: {out_root / 'leave_k_out_icl_contribution.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
