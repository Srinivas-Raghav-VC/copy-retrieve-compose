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

from core import build_task_prompt, get_first_token_stats, load_model, set_all_seeds, split_data_three_way  # noqa: E402
from paper2_fidelity_calibrated.eval_utils import evaluate_prompt_condition  # noqa: E402
from paper2_fidelity_calibrated.phase1_common import load_pair_split, log  # noqa: E402


PAIR_LANGUAGE_OVERRIDES = {
    "aksharantar_hin_latin": "hindi",
    "aksharantar_tel_latin": "telugu",
    "aksharantar_ben_latin": "bengali",
    "aksharantar_tam_latin": "tamil",
    "aksharantar_mar_latin": "marathi",
}

AKSHARANTAR_FALLBACKS = {
    "aksharantar_hin_latin": {"hf_config": "hin", "source_language": "Hindi", "output_script_name": "Devanagari"},
    "aksharantar_tel_latin": {"hf_config": "tel", "source_language": "Telugu", "output_script_name": "Telugu"},
    "aksharantar_tam_latin": {"hf_config": "tam", "source_language": "Tamil", "output_script_name": "Tamil"},
}
SPLIT_SNAPSHOT_DIR = PROJECT_ROOT / "paper2_fidelity_calibrated" / "split_snapshots"


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
    ap = argparse.ArgumentParser(description="GPU-side first-token proxy validation for Gemma transliteration.")
    ap.add_argument("--model", type=str, default="1b", choices=["1b", "4b"])
    ap.add_argument("--pairs", type=str, default="aksharantar_hin_latin,aksharantar_tel_latin")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-icl", type=int, default=16)
    ap.add_argument("--n-select", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--max-items", type=int, default=50)
    ap.add_argument("--counts", type=str, default="0,2,4,8,16")
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--external-only", action="store_true")
    ap.add_argument("--require-external-sources", action="store_true")
    ap.add_argument("--min-pool-size", type=int, default=500)
    ap.add_argument("--prompt-variant", type=str, default="canonical")
    ap.add_argument("--out", type=str, default="")
    return ap.parse_args()


def _parse_pairs(raw: str) -> List[str]:
    pairs = [part.strip() for part in str(raw or "").split(",") if part.strip()]
    if not pairs:
        raise ValueError("No pairs provided")
    return pairs


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


def _slug(parts: List[str]) -> str:
    return "__".join(part.replace("/", "_") for part in parts)


def _language_for_pair(pair_id: str, source_language: str) -> str:
    override = PAIR_LANGUAGE_OVERRIDES.get(str(pair_id))
    if override:
        return override
    return str(source_language or "unknown").strip().lower()


def _top1_token_text(tokenizer: Any, token_id: int) -> str:
    if int(token_id) < 0:
        return ""
    try:
        return str(tokenizer.decode([int(token_id)], skip_special_tokens=True)).strip()
    except Exception:
        return ""


def _snapshot_path(pair_id: str, *, seed: int, n_icl: int, n_select: int, n_eval: int) -> Path:
    return SPLIT_SNAPSHOT_DIR / f"{pair_id}_split_seed{int(seed)}_nicl{int(n_icl)}_nselect{int(n_select)}_neval{int(n_eval)}.json"


def _load_pair_split_with_fallback(
    pair_id: str,
    *,
    seed: int,
    n_icl: int,
    n_select: int,
    n_eval: int,
    external_only: bool,
    require_external_sources: bool,
    min_pool_size: int,
) -> Dict[str, Any]:
    snap = _snapshot_path(str(pair_id), seed=int(seed), n_icl=int(n_icl), n_select=int(n_select), n_eval=int(n_eval))
    if snap.exists():
        payload = json.loads(snap.read_text(encoding="utf-8"))
        return {
            "pair": str(payload["pair"]),
            "words": list(payload.get("icl_examples", [])) + list(payload.get("eval_rows", [])),
            "provenance": {"loader": "local_split_snapshot", "path": str(snap)},
            "prompt_meta": {},
            "source_language": str(payload["source_language"]),
            "input_script_name": str(payload["input_script_name"]),
            "output_script_name": str(payload["output_script_name"]),
            "icl_examples": list(payload["icl_examples"]),
            "select_rows": [],
            "eval_rows": list(payload["eval_rows"]),
        }

    try:
        return load_pair_split(
            str(pair_id),
            seed=int(seed),
            n_icl=int(n_icl),
            n_select=int(n_select),
            n_eval=int(n_eval),
            external_only=bool(external_only),
            require_external_sources=bool(require_external_sources),
            min_pool_size=int(min_pool_size),
        )
    except Exception as e:
        if str(pair_id) not in AKSHARANTAR_FALLBACKS:
            raise
        log(f"Falling back to direct HF Aksharantar loader for {pair_id}: {e}")
        from datasets import load_dataset

        cfg = AKSHARANTAR_FALLBACKS[str(pair_id)]
        ds = load_dataset("ai4bharat/Aksharantar", split="train")
        words = []
        for row in ds:
            if str(row.get("lang", "")).strip().lower() not in {cfg["hf_config"], str(pair_id).split("_")[1]}:
                continue
            src = str(row.get("english word", row.get("english", row.get("source", "")))).strip()
            tgt = str(row.get("native word", row.get("native", row.get("target", "")))).strip()
            if src and tgt:
                words.append({"ood": src, "hindi": tgt})
        if len(words) < int(n_icl) + int(n_select) + int(n_eval):
            raise RuntimeError(
                f"Not enough fallback records for {pair_id}: {len(words)} < {int(n_icl) + int(n_select) + int(n_eval)}"
            )
        icl_examples, select_rows, eval_rows = split_data_three_way(
            words,
            n_icl=int(n_icl),
            n_select=int(n_select),
            n_eval=int(n_eval),
            seed=int(seed),
        )
        return {
            "pair": str(pair_id),
            "words": words,
            "provenance": {"loader": "hf_direct_aksharantar", "hf_config": cfg["hf_config"]},
            "prompt_meta": {},
            "source_language": cfg["source_language"],
            "input_script_name": "Latin",
            "output_script_name": cfg["output_script_name"],
            "icl_examples": icl_examples,
            "select_rows": select_rows,
            "eval_rows": eval_rows,
        }


def main() -> int:
    args = parse_args()
    set_all_seeds(int(args.seed))

    pairs = _parse_pairs(str(args.pairs))
    counts = _parse_counts(str(args.counts), int(args.n_icl))

    model, tokenizer = load_model(str(args.model), device=str(args.device))
    device = str(next(model.parameters()).device)

    out_path = (
        Path(args.out).resolve()
        if str(args.out).strip()
        else PROJECT_ROOT
        / "paper2_fidelity_calibrated"
        / "results"
        / "first_token_proxy_validation"
        / str(args.model)
        / f"first_token_proxy_validation_{_slug(pairs)}.json"
    )

    items: List[Dict[str, Any]] = []
    started_at = time.time()

    for pair_id in pairs:
        bundle = _load_pair_split_with_fallback(
            str(pair_id),
            seed=int(args.seed),
            n_icl=int(args.n_icl),
            n_select=int(args.n_select),
            n_eval=int(args.n_eval),
            external_only=bool(args.external_only),
            require_external_sources=bool(args.require_external_sources),
            min_pool_size=int(args.min_pool_size),
        )
        eval_rows = list(bundle["eval_rows"][: max(1, int(args.max_items))])
        language = _language_for_pair(str(pair_id), str(bundle["source_language"]))
        log(
            f"Running M1 proxy validation: pair={pair_id} language={language} model={args.model} "
            f"items={len(eval_rows)} counts={counts}"
        )

        for item_idx, word in enumerate(eval_rows, start=1):
            source_text = str(word["ood"])
            target_text = str(word["hindi"])
            for count in counts:
                icl_examples = list(bundle["icl_examples"][: int(count)]) if int(count) > 0 else None
                prompt_text = build_task_prompt(
                    source_text,
                    icl_examples,
                    input_script_name=bundle["input_script_name"],
                    source_language=bundle["source_language"],
                    output_script_name=bundle["output_script_name"],
                    prompt_variant=str(args.prompt_variant),
                )
                prompt_token_len = int(len(tokenizer.encode(prompt_text, add_special_tokens=False)))
                cond = evaluate_prompt_condition(
                    model,
                    tokenizer,
                    prompt_text=prompt_text,
                    target_text=target_text,
                    target_script=bundle["output_script_name"],
                    device=device,
                    max_new_tokens=int(args.max_new_tokens),
                )
                ft = get_first_token_stats(
                    model,
                    tokenizer,
                    prompt=prompt_text,
                    target_text=target_text,
                    device=device,
                )
                items.append(
                    {
                        "source": source_text,
                        "reference": target_text,
                        "language": language,
                        "pair": str(pair_id),
                        "model": str(args.model),
                        "seed": int(args.seed),
                        "n_icl": int(count),
                        "item_index": int(item_idx - 1),
                        "prompt_variant": str(args.prompt_variant),
                        "prompt_token_len": int(prompt_token_len),
                        "target_script": str(bundle["output_script_name"]),
                        "pred": str(cond["prediction"]),
                        "exact_match_local": float(cond["exact_match"]),
                        "akshara_cer_local": float(cond["akshara_cer"]),
                        "script_compliance_local": float(cond["script_compliance"]),
                        "first_entry_correct_local": float(cond["first_entry_correct"]),
                        "continuation_akshara_cer_local": float(cond["continuation_akshara_cer"]),
                        "joint_logprob": float(cond["joint_logprob"]),
                        "target_pos1_nll": float(cond["target_pos1_nll"]),
                        "target_pos2_nll": float(cond["target_pos2_nll"]),
                        "target_pos3_nll": float(cond["target_pos3_nll"]),
                        "first_prob": float(cond["first_prob"]),
                        "first_logit": float(cond["first_logit"]),
                        "first_rank": float(ft.get("target_rank", float("nan"))),
                        "first_entropy": float(ft.get("entropy", float("nan"))),
                        "first_logit_gap": float(ft.get("logit_gap", float("nan"))),
                        "target_token_id": int(ft.get("target_id", -1)),
                        "top1_token_id": int(ft.get("top1_id", -1)),
                        "top1_prob": float(ft.get("top1_prob", float("nan"))),
                        "target_token_text": _top1_token_text(tokenizer, int(ft.get("target_id", -1))),
                        "top1_token_text": _top1_token_text(tokenizer, int(ft.get("top1_id", -1))),
                        "topk_hit_1": bool(ft.get("topk_hits", {}).get("1", False)),
                        "topk_hit_10": bool(ft.get("topk_hits", {}).get("10", False)),
                        "topk_hit_100": bool(ft.get("topk_hits", {}).get("100", False)),
                    }
                )
            if item_idx % 10 == 0 or item_idx == len(eval_rows):
                log(f"  pair={pair_id} progress {item_idx}/{len(eval_rows)}")

    grouped: Dict[tuple[str, str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in items:
        grouped[(str(row["pair"]), str(row["language"]), int(row["n_icl"]))].append(row)

    summary_rows: List[Dict[str, Any]] = []
    for (pair_id, language, count), rows in sorted(grouped.items()):
        summary_rows.append(
            {
                "pair": pair_id,
                "language": language,
                "n_icl": int(count),
                "n_items": int(len(rows)),
                "exact_match_local": float(np.nanmean([row["exact_match_local"] for row in rows])),
                "akshara_cer_local": float(np.nanmean([row["akshara_cer_local"] for row in rows])),
                "first_entry_correct_local": float(np.nanmean([row["first_entry_correct_local"] for row in rows])),
                "first_prob": float(np.nanmean([row["first_prob"] for row in rows])),
                "first_rank": float(np.nanmean([row["first_rank"] for row in rows])),
                "first_logit": float(np.nanmean([row["first_logit"] for row in rows])),
                "joint_logprob": float(np.nanmean([row["joint_logprob"] for row in rows])),
            }
        )

    payload = {
        "experiment": "first_token_proxy_validation",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": str(args.model),
        "pairs": pairs,
        "counts": counts,
        "seed": int(args.seed),
        "n_eval": int(args.n_eval),
        "max_items": int(args.max_items),
        "prompt_variant": str(args.prompt_variant),
        "duration_sec": float(time.time() - started_at),
        "summary": summary_rows,
        "items": items,
    }
    _write_json(out_path, payload)
    log(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
