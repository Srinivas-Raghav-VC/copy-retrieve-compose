#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import (  # noqa: E402
    _teacher_forced_metrics_from_input_ids,
    apply_chat_template,
    build_task_prompt,
    get_model_config,
    load_model,
    register_attention_head_ablation_hook,
    set_all_seeds,
    split_data_three_way,
)
from paper2_fidelity_calibrated.eval_utils import normalize_text  # noqa: E402
from paper2_fidelity_calibrated.run import _load_words, _prompt_naming  # noqa: E402
from paper2_fidelity_calibrated.run_bounded_attention_analysis import (  # noqa: E402
    _extract_prompt_regions,
    _generation_metrics,
)
from rescue_research.data_pipeline.ingest import get_pair_prompt_metadata  # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Shared-vs-specific grouped head ablation across Hindi/Telugu.")
    ap.add_argument("--model", type=str, default="4b", choices=["4b"])
    ap.add_argument("--pairs", type=str, default="aksharantar_hin_latin,aksharantar_tel_latin")
    ap.add_argument("--n-icl", type=int, default=64)
    ap.add_argument("--n-select", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--topk-intersection", type=int, default=12)
    ap.add_argument("--artifact-dir", type=str, default="artifacts/phase5_attribution")
    ap.add_argument("--out", type=str, default="artifacts/phase5_head_groups/shared_specific_head_ablation_4b_multilang.json")
    return ap.parse_args()


def _get_num_heads(model) -> int:
    cfg = getattr(model.config, "text_config", model.config)
    return int(cfg.num_attention_heads)


def _load_top_heads(path: Path, topk: int) -> List[Tuple[int, int]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [(int(row["layer"]), int(row["head"])) for row in rows[: int(topk)]]


def _group_by_layer(pairs: List[Tuple[int, int]]) -> Dict[int, List[int]]:
    grouped: Dict[int, List[int]] = defaultdict(list)
    for layer, head in pairs:
        if int(head) not in grouped[int(layer)]:
            grouped[int(layer)].append(int(head))
    return {int(k): list(v) for k, v in sorted(grouped.items())}


def _make_random_matched_group(reference_group: Dict[int, List[int]], num_heads: int, seed: int) -> Dict[int, List[int]]:
    rng = random.Random(seed)
    out: Dict[int, List[int]] = {}
    for layer, heads in sorted(reference_group.items()):
        need = len(heads)
        pool = [h for h in range(int(num_heads)) if h not in set(int(x) for x in heads)]
        rng.shuffle(pool)
        out[int(layer)] = sorted(pool[:need])
    return out


def _derive_groups(artifact_dir: Path, num_heads: int, topk_intersection: int, seed: int) -> Dict[str, Dict[int, List[int]]]:
    hin = _load_top_heads(artifact_dir / "top_heads_4b_hin_multilang.json", topk_intersection)
    tel = _load_top_heads(artifact_dir / "top_heads_4b_tel_multilang.json", topk_intersection)

    hin_set = set(hin)
    tel_set = set(tel)
    intersection = [x for x in hin if x in tel_set]
    hin_unique = [x for x in hin if x not in tel_set]
    tel_unique = [x for x in tel if x not in hin_set]
    shared_l17 = [x for x in intersection if int(x[0]) == 17]

    shared_union = _group_by_layer(intersection)
    return {
        "shared_l17": _group_by_layer(shared_l17),
        "shared_union": shared_union,
        "hindi_unique": _group_by_layer(hin_unique[:5]),
        "telugu_unique": _group_by_layer(tel_unique[:5]),
        "random_matched_shared": _make_random_matched_group(shared_union, num_heads=num_heads, seed=seed + 99),
    }


def _generate_text(model, tokenizer, input_ids: torch.Tensor, pad_id: int, max_new_tokens: int) -> str:
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        out = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            pad_token_id=int(pad_id),
        )
    return normalize_text(tokenizer.decode(out[0, input_ids.shape[1] :], skip_special_tokens=True).strip())


def _collect_item_payload(tokenizer, word, prompt_text: str, prompt_ids: torch.Tensor, target_ids: List[int]):
    rendered = apply_chat_template(tokenizer, prompt_text)
    regions = _extract_prompt_regions(
        tokenizer=tokenizer,
        raw_prompt=prompt_text,
        rendered_prompt=rendered,
        query_token=word["ood"],
    )
    query_span = regions.get("query_span")
    if query_span is None:
        raise RuntimeError(f"Failed query-span localization for word={word['ood']}")
    source_pos = int(query_span[1] - 1)
    target_id = int(target_ids[0]) if target_ids else -1
    return {
        "prompt_ids": prompt_ids,
        "source_pos": source_pos,
        "target_id": target_id,
        "target_ids": target_ids,
    }


def _mean(xs: List[float]) -> float:
    vals = [float(x) for x in xs if np.isfinite(float(x))]
    return float(np.mean(vals)) if vals else float("nan")


def main() -> int:
    args = parse_args()
    set_all_seeds(int(args.seed))
    pair_ids = [x.strip() for x in str(args.pairs).split(",") if x.strip()]
    if not pair_ids:
        raise RuntimeError("No pairs provided")

    cfg = get_model_config(str(args.model))
    model, tokenizer = load_model(str(args.model), device=str(args.device))
    device = str(next(model.parameters()).device)
    pad_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", 0)
    num_heads = _get_num_heads(model)

    artifact_dir = (PROJECT_ROOT / str(args.artifact_dir)).resolve()
    groups = _derive_groups(
        artifact_dir,
        num_heads=num_heads,
        topk_intersection=int(args.topk_intersection),
        seed=int(args.seed),
    )
    log(f"Derived groups: {groups}")

    summary_rows = []
    item_rows = []

    for pair_id in pair_ids:
        meta = dict(get_pair_prompt_metadata(pair_id))
        source_lang, input_script, output_script = _prompt_naming(meta)
        words, _ = _load_words(pair_id, external_only=True, require_external_sources=True, min_pool_size=500)
        icl_examples, _, eval_rows = split_data_three_way(
            words,
            n_icl=int(args.n_icl),
            n_select=int(args.n_select),
            n_eval=int(args.n_eval),
            seed=int(args.seed),
        )

        cached = []
        log(f"Preparing baseline cache for pair={pair_id} n_eval={len(eval_rows)}")
        for idx, word in enumerate(eval_rows):
            if idx > 0 and idx % 20 == 0:
                log(f"  cached {idx}/{len(eval_rows)}")
            prompt = build_task_prompt(
                word["ood"],
                list(icl_examples),
                input_script_name=input_script,
                source_language=source_lang,
                output_script_name=output_script,
                prompt_variant="canonical",
            )
            prompt_ids = tokenizer(apply_chat_template(tokenizer, prompt), return_tensors="pt").input_ids.to(device)
            target_ids = tokenizer.encode(str(word["hindi"]), add_special_tokens=False)
            payload = _collect_item_payload(tokenizer, word, prompt, prompt_ids, target_ids)
            baseline_tf = _teacher_forced_metrics_from_input_ids(
                model=model,
                input_ids=prompt_ids,
                target_ids=target_ids,
                target_id=payload["target_id"],
                device=device,
                competitor_id=-1,
            )
            baseline_pred = _generate_text(model, tokenizer, prompt_ids, pad_id=pad_id, max_new_tokens=int(args.max_new_tokens))
            baseline_gen = _generation_metrics(word["hindi"], baseline_pred, target_script=output_script)
            cached.append(
                {
                    "idx": idx,
                    "word": word,
                    **payload,
                    "baseline_tf": baseline_tf,
                    "baseline_gen": baseline_gen,
                }
            )

        for group_name, group in groups.items():
            log(f"Running group={group_name} pair={pair_id} group={group}")
            group_item_rows = []
            for item in cached:
                handles = []
                try:
                    for layer, heads in sorted(group.items()):
                        handles.append(
                            register_attention_head_ablation_hook(
                                model,
                                int(layer),
                                list(heads),
                                ablate_position=int(item["source_pos"]),
                            )
                        )
                    ablated_tf = _teacher_forced_metrics_from_input_ids(
                        model=model,
                        input_ids=item["prompt_ids"],
                        target_ids=item["target_ids"],
                        target_id=item["target_id"],
                        device=device,
                        competitor_id=-1,
                    )
                    ablated_pred = _generate_text(
                        model,
                        tokenizer,
                        item["prompt_ids"],
                        pad_id=pad_id,
                        max_new_tokens=int(args.max_new_tokens),
                    )
                finally:
                    for handle in handles:
                        handle.remove()
                ablated_gen = _generation_metrics(item["word"]["hindi"], ablated_pred, target_script=output_script)
                row = {
                    "pair": pair_id,
                    "group": group_name,
                    "group_layers": {str(k): list(v) for k, v in group.items()},
                    "item_index": int(item["idx"]),
                    "word_source": str(item["word"]["ood"]),
                    "word_target": str(item["word"]["hindi"]),
                    "exact_match_base": float(item["baseline_gen"]["exact_match"]),
                    "exact_match_ablated": float(ablated_gen["exact_match"]),
                    "cer_base": float(item["baseline_gen"]["akshara_cer"]),
                    "cer_ablated": float(ablated_gen["akshara_cer"]),
                    "first_entry_base": float(item["baseline_gen"]["first_entry_correct"]),
                    "first_entry_ablated": float(ablated_gen["first_entry_correct"]),
                    "continuation_base": float(item["baseline_gen"]["continuation_fidelity"]),
                    "continuation_ablated": float(ablated_gen["continuation_fidelity"]),
                    "nll_pos1_base": float(item["baseline_tf"].get("target_pos1_nll", float("nan"))),
                    "nll_pos1_ablated": float(ablated_tf.get("target_pos1_nll", float("nan"))),
                    "nll_pos2_base": float(item["baseline_tf"].get("target_pos2_nll", float("nan"))),
                    "nll_pos2_ablated": float(ablated_tf.get("target_pos2_nll", float("nan"))),
                    "nll_pos3_base": float(item["baseline_tf"].get("target_pos3_nll", float("nan"))),
                    "nll_pos3_ablated": float(ablated_tf.get("target_pos3_nll", float("nan"))),
                }
                item_rows.append(row)
                group_item_rows.append(row)

            summary_rows.append(
                {
                    "pair": pair_id,
                    "group": group_name,
                    "group_layers": {str(k): list(v) for k, v in group.items()},
                    "exact_match_drop": _mean([r["exact_match_base"] - r["exact_match_ablated"] for r in group_item_rows]),
                    "first_entry_drop": _mean([r["first_entry_base"] - r["first_entry_ablated"] for r in group_item_rows]),
                    "continuation_drop": _mean([r["continuation_base"] - r["continuation_ablated"] for r in group_item_rows]),
                    "cer_delta": _mean([r["cer_ablated"] - r["cer_base"] for r in group_item_rows]),
                    "nll_pos1_delta": _mean([r["nll_pos1_ablated"] - r["nll_pos1_base"] for r in group_item_rows]),
                    "nll_pos2_delta": _mean([r["nll_pos2_ablated"] - r["nll_pos2_base"] for r in group_item_rows]),
                    "nll_pos3_delta": _mean([r["nll_pos3_ablated"] - r["nll_pos3_base"] for r in group_item_rows]),
                }
            )
            log(f"Completed group={group_name} pair={pair_id}")

    out_path = (PROJECT_ROOT / str(args.out)).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "shared_specific_head_ablation",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": str(args.model),
        "pairs": pair_ids,
        "topk_intersection": int(args.topk_intersection),
        "runtime_model": {
            "hf_id": str(cfg.hf_id),
            "num_heads": int(num_heads),
        },
        "groups": {k: {str(layer): list(heads) for layer, heads in v.items()} for k, v in groups.items()},
        "summary": summary_rows,
        "items": item_rows,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log(f"Saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
