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
    ap = argparse.ArgumentParser(description="Frozen ICL contribution curve over a fixed prefix ladder.")
    ap.add_argument("--model", type=str, default="4b", choices=["1b", "4b"])
    ap.add_argument("--pair", type=str, default="aksharantar_hin_latin")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-icl", type=int, default=64)
    ap.add_argument("--n-select", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--max-items", type=int, default=50)
    ap.add_argument("--counts", type=str, default="0,1,2,4,8,16,32,64")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--external-only", action="store_true")
    ap.add_argument("--require-external-sources", action="store_true")
    ap.add_argument("--min-pool-size", type=int, default=500)
    ap.add_argument("--out", type=str, default="")
    return ap.parse_args()


def _parse_counts(raw: str, n_icl: int) -> List[int]:
    vals = []
    for part in str(raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(max(0, min(int(part), int(n_icl))))
    vals = sorted(set(vals))
    if not vals:
        vals = [0, int(n_icl)]
    return vals


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

    model, tokenizer = load_model(str(args.model), device=str(args.device))
    device = str(next(model.parameters()).device)
    counts = _parse_counts(str(args.counts), int(args.n_icl))
    eval_rows = list(pair_bundle["eval_rows"][: max(1, int(args.max_items))])

    out_root = (
        Path(args.out).resolve()
        if str(args.out).strip()
        else PROJECT_ROOT / "paper2_fidelity_calibrated" / "results" / "icl_contribution_curve" / str(args.pair) / str(args.model)
    )
    out_root.mkdir(parents=True, exist_ok=True)

    log(
        f"Running G5 ICL contribution curve: pair={args.pair} model={args.model} items={len(eval_rows)} counts={counts}"
    )

    item_rows: List[Dict[str, Any]] = []
    for item_idx, word in enumerate(eval_rows, start=1):
        log(f"[{item_idx}/{len(eval_rows)}] {word['ood']} -> {word['hindi']}")
        for count in counts:
            icl_examples = list(pair_bundle["icl_examples"][: int(count)]) if int(count) > 0 else None
            prompt_text = build_task_prompt(
                str(word["ood"]),
                icl_examples,
                input_script_name=pair_bundle["input_script_name"],
                source_language=pair_bundle["source_language"],
                output_script_name=pair_bundle["output_script_name"],
                prompt_variant="canonical",
            )
            metrics = evaluate_prompt_condition(
                model,
                tokenizer,
                prompt_text=prompt_text,
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
                    "icl_count": int(count),
                    **metrics,
                }
            )

    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in item_rows:
        grouped[int(row["icl_count"])].append(row)

    summary_rows: List[Dict[str, Any]] = []
    for count in sorted(grouped):
        rows = grouped[count]
        summary_rows.append(
            {
                "icl_count": int(count),
                "n_items": int(len(rows)),
                "exact_match": float(np.nanmean([row["exact_match"] for row in rows])),
                "akshara_cer": float(np.nanmean([row["akshara_cer"] for row in rows])),
                "script_compliance": float(np.nanmean([row["script_compliance"] for row in rows])),
                "first_entry_correct": float(np.nanmean([row["first_entry_correct"] for row in rows])),
                "continuation_akshara_cer": float(np.nanmean([row["continuation_akshara_cer"] for row in rows])),
                "joint_logprob": float(np.nanmean([row["joint_logprob"] for row in rows])),
                "target_pos1_nll": float(np.nanmean([row["target_pos1_nll"] for row in rows])),
                "first_prob": float(np.nanmean([row["first_prob"] for row in rows])),
                "first_logit": float(np.nanmean([row["first_logit"] for row in rows])),
            }
        )

    payload = {
        "experiment": "icl_contribution_curve",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": str(args.model),
        "pair": str(args.pair),
        "seed": int(args.seed),
        "counts": counts,
        "summary": summary_rows,
        "item_rows": item_rows,
    }
    _write_json(out_root / "icl_contribution_curve.json", payload)
    log(f"Saved: {out_root / 'icl_contribution_curve.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
