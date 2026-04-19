#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import (  # noqa: E402
    _find_last_subsequence,
    _teacher_forced_metrics_from_input_ids,
    apply_chat_template,
    build_task_prompt,
    load_model,
    set_all_seeds,
)
from paper2_fidelity_calibrated.eval_utils import (  # noqa: E402
    akshara_cer,
    continuation_akshara_cer,
    first_entry_correct,
    normalize_text,
    script_compliance,
)
from paper2_fidelity_calibrated.phase1_common import load_pair_split, log  # noqa: E402
from paper2_fidelity_calibrated.phase23_common import (  # noqa: E402
    capture_o_proj_inputs_at_position,
    get_attn_module,
    infer_num_heads,
    register_attention_head_addition_hooks,
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
    ap = argparse.ArgumentParser(description="Function-vector tests from averaged top-head deltas.")
    ap.add_argument("--model", type=str, default="4b", choices=["1b", "4b"])
    ap.add_argument("--pairs", type=str, default="aksharantar_hin_latin,aksharantar_tel_latin")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-icl", type=int, default=64)
    ap.add_argument("--n-select", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--max-items", type=int, default=30)
    ap.add_argument("--donor-items", type=int, default=50)
    ap.add_argument("--top-n-heads", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--external-only", action="store_true")
    ap.add_argument("--require-external-sources", action="store_true")
    ap.add_argument("--min-pool-size", type=int, default=500)
    ap.add_argument("--out", type=str, default="")
    return ap.parse_args()


def _lang_from_pair(pair_id: str) -> str:
    parts = str(pair_id).split("_")
    return parts[1] if len(parts) >= 2 else str(pair_id)


def _top_heads_path(model_key: str, pair_id: str) -> Path:
    lang = _lang_from_pair(pair_id)
    candidates = [
        PROJECT_ROOT / "artifacts" / "phase5_attribution" / f"top_heads_{model_key}_{lang}_multilang.json",
        PROJECT_ROOT / "artifacts" / "phase5_attribution" / f"top_heads_{model_key}_{lang}.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _load_top_heads(model_key: str, pair_id: str, top_n: int) -> List[Dict[str, Any]]:
    path = _top_heads_path(model_key, pair_id)
    if not path.exists():
        raise FileNotFoundError(f"Missing top-head artifact: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    out: List[Dict[str, Any]] = []
    for row in rows[: max(1, int(top_n))]:
        out.append(
            {
                "rank": int(row.get("rank", len(out) + 1)),
                "layer": int(row["layer"]),
                "head": int(row["head"]),
                "effect": float(row.get("effect", float("nan"))),
            }
        )
    return out


def _prepare_prompt_ids(
    *,
    tokenizer: Any,
    query_text: str,
    icl_examples: Optional[List[Dict[str, str]]],
    input_script_name: str,
    source_language: str,
    output_script_name: str,
) -> Tuple[torch.Tensor, int]:
    prompt = build_task_prompt(
        str(query_text),
        icl_examples,
        input_script_name=input_script_name,
        source_language=source_language,
        output_script_name=output_script_name,
        prompt_variant="canonical",
    )
    rendered = apply_chat_template(tokenizer, prompt)
    input_ids = tokenizer(rendered, return_tensors="pt").input_ids
    query_ids = tokenizer.encode(str(query_text), add_special_tokens=False)
    span = _find_last_subsequence(
        input_ids[0].detach().to("cpu", dtype=torch.long).tolist(),
        [int(x) for x in query_ids],
    )
    if span is None:
        raise RuntimeError(f"Failed to localize query span for {query_text!r}")
    return input_ids, int(span[1] - 1)


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
            input_ids=input_ids.to(device),
            target_ids=target_ids,
            target_id=target_id,
            device=str(device),
        )
        attention_mask = torch.ones_like(input_ids, device=device)
        pad_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", 0)
        with torch.inference_mode():
            out = model.generate(
                input_ids.to(device),
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


def _build_function_vector(
    *,
    model: Any,
    tokenizer: Any,
    pair_bundle: Dict[str, Any],
    top_heads: List[Dict[str, Any]],
    donor_items: int,
) -> Dict[int, torch.Tensor]:
    layers_needed = sorted({int(row["layer"]) for row in top_heads})
    head_groups: Dict[int, List[int]] = defaultdict(list)
    for row in top_heads:
        head_groups[int(row["layer"])].append(int(row["head"]))

    sums: Dict[int, torch.Tensor] = {}
    n = 0
    for word in list(pair_bundle["select_rows"][: max(1, int(donor_items))]):
        zs_input_ids, zs_pos = _prepare_prompt_ids(
            tokenizer=tokenizer,
            query_text=str(word["ood"]),
            icl_examples=None,
            input_script_name=pair_bundle["input_script_name"],
            source_language=pair_bundle["source_language"],
            output_script_name=pair_bundle["output_script_name"],
        )
        icl_input_ids, icl_pos = _prepare_prompt_ids(
            tokenizer=tokenizer,
            query_text=str(word["ood"]),
            icl_examples=pair_bundle["icl_examples"],
            input_script_name=pair_bundle["input_script_name"],
            source_language=pair_bundle["source_language"],
            output_script_name=pair_bundle["output_script_name"],
        )
        donor_zs = capture_o_proj_inputs_at_position(model, zs_input_ids.to(next(model.parameters()).device), layers_needed, zs_pos)
        donor_icl = capture_o_proj_inputs_at_position(model, icl_input_ids.to(next(model.parameters()).device), layers_needed, icl_pos)
        for layer in layers_needed:
            full_delta = donor_icl[int(layer)] - donor_zs[int(layer)]
            masked_delta = torch.zeros_like(full_delta)
            num_heads = len(head_groups[int(layer)])
            d_model = int(full_delta.numel())
            # infer head slices from actual selected heads count at runtime via max head index not needed here
            # use o_proj-input shape and chosen head indices only
            attn_heads = max(int(row["head"]) for row in top_heads if int(row["layer"]) == int(layer)) + 1
            # head count from model capture shape is not inferable here; use equal chunks from config at apply time
            # so keep full delta and rely on addition hook to slice chosen heads only.
            masked_delta = full_delta.detach().clone()
            if int(layer) not in sums:
                sums[int(layer)] = torch.zeros_like(masked_delta)
            sums[int(layer)] += masked_delta.detach().cpu()
        n += 1
    if n <= 0:
        raise RuntimeError("No donor items available for function-vector estimation")
    return {int(layer): (vec / float(n)).detach() for layer, vec in sums.items()}


def _random_head_groups(model: Any, top_heads: List[Dict[str, Any]], *, seed: int) -> Dict[int, List[int]]:
    grouped: Dict[int, List[int]] = defaultdict(list)
    for row in top_heads:
        grouped[int(row["layer"])].append(int(row["head"]))
    rng = random.Random(int(seed))
    out: Dict[int, List[int]] = {}
    for layer, heads in sorted(grouped.items()):
        attn_module = get_attn_module(model, int(layer))
        num_heads = infer_num_heads(model, attn_module)
        pool = [h for h in range(int(num_heads)) if int(h) not in set(int(x) for x in heads)]
        rng.shuffle(pool)
        out[int(layer)] = sorted(pool[: len(heads)])
    return out


def main() -> int:
    args = parse_args()
    set_all_seeds(int(args.seed))

    pair_ids = [x.strip() for x in str(args.pairs).split(",") if x.strip()]
    if not pair_ids:
        raise RuntimeError("No pairs provided")

    model, tokenizer = load_model(str(args.model), device=str(args.device))
    device = str(next(model.parameters()).device)

    bundles: Dict[str, Dict[str, Any]] = {}
    top_heads_by_pair: Dict[str, List[Dict[str, Any]]] = {}
    random_head_groups_by_pair: Dict[str, Dict[int, List[int]]] = {}
    vectors_by_pair: Dict[str, Dict[int, torch.Tensor]] = {}
    top_head_paths: Dict[str, str] = {}

    for pair_id in pair_ids:
        bundles[pair_id] = load_pair_split(
            pair_id,
            seed=int(args.seed),
            n_icl=int(args.n_icl),
            n_select=int(args.n_select),
            n_eval=int(args.n_eval),
            external_only=bool(args.external_only),
            require_external_sources=bool(args.require_external_sources),
            min_pool_size=int(args.min_pool_size),
        )
        top_head_paths[pair_id] = str(_top_heads_path(str(args.model), pair_id))
        top_heads_by_pair[pair_id] = _load_top_heads(str(args.model), pair_id, int(args.top_n_heads))
        random_head_groups_by_pair[pair_id] = _random_head_groups(
            model,
            top_heads_by_pair[pair_id],
            seed=int(args.seed) + (11 if "hin" in pair_id else 29),
        )
        vectors_by_pair[pair_id] = _build_function_vector(
            model=model,
            tokenizer=tokenizer,
            pair_bundle=bundles[pair_id],
            top_heads=top_heads_by_pair[pair_id],
            donor_items=int(args.donor_items),
        )

    out_root = (
        Path(args.out).resolve()
        if str(args.out).strip()
        else PROJECT_ROOT / "paper2_fidelity_calibrated" / "results" / "cfom_function_vector_tests" / str(args.model)
    )
    out_root.mkdir(parents=True, exist_ok=True)

    item_rows: List[Dict[str, Any]] = []
    for donor_pair in pair_ids:
        donor_head_groups: Dict[int, List[int]] = defaultdict(list)
        for row in top_heads_by_pair[donor_pair]:
            donor_head_groups[int(row["layer"])].append(int(row["head"]))
        random_donor_head_groups = random_head_groups_by_pair[donor_pair]

        for test_pair in pair_ids:
            pair_bundle = bundles[test_pair]
            eval_rows = list(pair_bundle["eval_rows"][: max(1, int(args.max_items))])
            log(
                f"Running G11 function-vector tests: model={args.model} donor={donor_pair} test={test_pair} items={len(eval_rows)}"
            )
            for item_idx, word in enumerate(eval_rows, start=1):
                zs_input_ids, zs_pos = _prepare_prompt_ids(
                    tokenizer=tokenizer,
                    query_text=str(word["ood"]),
                    icl_examples=None,
                    input_script_name=pair_bundle["input_script_name"],
                    source_language=pair_bundle["source_language"],
                    output_script_name=pair_bundle["output_script_name"],
                )
                icl_input_ids, _ = _prepare_prompt_ids(
                    tokenizer=tokenizer,
                    query_text=str(word["ood"]),
                    icl_examples=pair_bundle["icl_examples"],
                    input_script_name=pair_bundle["input_script_name"],
                    source_language=pair_bundle["source_language"],
                    output_script_name=pair_bundle["output_script_name"],
                )
                zs_metrics = _run_condition(
                    model=model,
                    tokenizer=tokenizer,
                    input_ids=zs_input_ids,
                    target_text=str(word["hindi"]),
                    target_script=pair_bundle["output_script_name"],
                    device=device,
                    max_new_tokens=int(args.max_new_tokens),
                )
                icl_metrics = _run_condition(
                    model=model,
                    tokenizer=tokenizer,
                    input_ids=icl_input_ids,
                    target_text=str(word["hindi"]),
                    target_script=pair_bundle["output_script_name"],
                    device=device,
                    max_new_tokens=int(args.max_new_tokens),
                )
                patch_vectors = {int(k): v.to(device) for k, v in vectors_by_pair[donor_pair].items()}
                patch_hook = register_attention_head_addition_hooks(
                    model,
                    patch_vectors,
                    donor_head_groups,
                    replace_position=int(zs_pos),
                )
                fv_metrics = _run_condition(
                    model=model,
                    tokenizer=tokenizer,
                    input_ids=zs_input_ids,
                    target_text=str(word["hindi"]),
                    target_script=pair_bundle["output_script_name"],
                    device=device,
                    max_new_tokens=int(args.max_new_tokens),
                    hooks=[patch_hook],
                )
                random_patch_hook = register_attention_head_addition_hooks(
                    model,
                    patch_vectors,
                    random_donor_head_groups,
                    replace_position=int(zs_pos),
                )
                random_fv_metrics = _run_condition(
                    model=model,
                    tokenizer=tokenizer,
                    input_ids=zs_input_ids,
                    target_text=str(word["hindi"]),
                    target_script=pair_bundle["output_script_name"],
                    device=device,
                    max_new_tokens=int(args.max_new_tokens),
                    hooks=[random_patch_hook],
                )
                item_rows.append(
                    {
                        "model": str(args.model),
                        "seed": int(args.seed),
                        "donor_pair": donor_pair,
                        "test_pair": test_pair,
                        "same_pair": bool(donor_pair == test_pair),
                        "item_index": int(item_idx - 1),
                        "word_ood": str(word["ood"]),
                        "word_hindi": str(word["hindi"]),
                        "zs_exact_match": float(zs_metrics["exact_match"]),
                        "icl_exact_match": float(icl_metrics["exact_match"]),
                        "fv_exact_match": float(fv_metrics["exact_match"]),
                        "random_fv_exact_match": float(random_fv_metrics["exact_match"]),
                        "zs_first_entry_correct": float(zs_metrics["first_entry_correct"]),
                        "icl_first_entry_correct": float(icl_metrics["first_entry_correct"]),
                        "fv_first_entry_correct": float(fv_metrics["first_entry_correct"]),
                        "random_fv_first_entry_correct": float(random_fv_metrics["first_entry_correct"]),
                        "zs_first_prob": float(zs_metrics["first_prob"]),
                        "icl_first_prob": float(icl_metrics["first_prob"]),
                        "fv_first_prob": float(fv_metrics["first_prob"]),
                        "random_fv_first_prob": float(random_fv_metrics["first_prob"]),
                        "zs_first_logit": float(zs_metrics["first_logit"]),
                        "icl_first_logit": float(icl_metrics["first_logit"]),
                        "fv_first_logit": float(fv_metrics["first_logit"]),
                        "random_fv_first_logit": float(random_fv_metrics["first_logit"]),
                        "zs_target_pos1_nll": float(zs_metrics["target_pos1_nll"]),
                        "icl_target_pos1_nll": float(icl_metrics["target_pos1_nll"]),
                        "fv_target_pos1_nll": float(fv_metrics["target_pos1_nll"]),
                        "random_fv_target_pos1_nll": float(random_fv_metrics["target_pos1_nll"]),
                    }
                )

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in item_rows:
        grouped[(str(row["donor_pair"]), str(row["test_pair"]))].append(row)

    summary_rows: List[Dict[str, Any]] = []
    for (donor_pair, test_pair), rows in sorted(grouped.items()):
        summary_rows.append(
            {
                "donor_pair": donor_pair,
                "test_pair": test_pair,
                "same_pair": bool(donor_pair == test_pair),
                "n_items": int(len(rows)),
                "zs_exact_match": float(np.nanmean([row["zs_exact_match"] for row in rows])),
                "icl_exact_match": float(np.nanmean([row["icl_exact_match"] for row in rows])),
                "fv_exact_match": float(np.nanmean([row["fv_exact_match"] for row in rows])),
                "random_fv_exact_match": float(np.nanmean([row["random_fv_exact_match"] for row in rows])),
                "zs_first_entry_correct": float(np.nanmean([row["zs_first_entry_correct"] for row in rows])),
                "icl_first_entry_correct": float(np.nanmean([row["icl_first_entry_correct"] for row in rows])),
                "fv_first_entry_correct": float(np.nanmean([row["fv_first_entry_correct"] for row in rows])),
                "random_fv_first_entry_correct": float(np.nanmean([row["random_fv_first_entry_correct"] for row in rows])),
                "zs_first_prob": float(np.nanmean([row["zs_first_prob"] for row in rows])),
                "icl_first_prob": float(np.nanmean([row["icl_first_prob"] for row in rows])),
                "fv_first_prob": float(np.nanmean([row["fv_first_prob"] for row in rows])),
                "random_fv_first_prob": float(np.nanmean([row["random_fv_first_prob"] for row in rows])),
                "zs_target_pos1_nll": float(np.nanmean([row["zs_target_pos1_nll"] for row in rows])),
                "icl_target_pos1_nll": float(np.nanmean([row["icl_target_pos1_nll"] for row in rows])),
                "fv_target_pos1_nll": float(np.nanmean([row["fv_target_pos1_nll"] for row in rows])),
                "random_fv_target_pos1_nll": float(np.nanmean([row["random_fv_target_pos1_nll"] for row in rows])),
            }
        )

    payload = {
        "experiment": "cfom_function_vector_tests",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": str(args.model),
        "pairs": pair_ids,
        "seed": int(args.seed),
        "top_head_paths": top_head_paths,
        "top_heads_by_pair": top_heads_by_pair,
        "random_head_groups_by_pair": random_head_groups_by_pair,
        "summary": summary_rows,
        "item_rows": item_rows,
    }
    _write_json(out_root / "cfom_function_vector_tests.json", payload)
    log(f"Saved: {out_root / 'cfom_function_vector_tests.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
