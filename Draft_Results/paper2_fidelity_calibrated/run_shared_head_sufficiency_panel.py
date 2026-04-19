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
    get_model_layers,
    load_model,
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
    ap = argparse.ArgumentParser(description="Shared-head sufficiency / rescue patching panel for 4B.")
    ap.add_argument("--pairs", type=str, default="aksharantar_hin_latin,aksharantar_tel_latin")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--n-icl", type=int, default=64)
    ap.add_argument("--n-select", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=100)
    ap.add_argument("--filler-n", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--topk-intersection", type=int, default=12)
    ap.add_argument("--artifact-dir", type=str, default="artifacts/phase5_attribution")
    ap.add_argument("--out", type=str, default="artifacts/phase5_head_groups/shared_head_sufficiency_panel_4b_multilang.json")
    return ap.parse_args()


def _infer_num_heads(model, attn_module) -> int:
    candidates = [
        getattr(attn_module, "num_heads", None),
        getattr(attn_module, "num_attention_heads", None),
        getattr(getattr(attn_module, "config", None), "num_attention_heads", None),
        getattr(getattr(model, "config", None), "num_attention_heads", None),
        getattr(getattr(getattr(model, "config", None), "text_config", None), "num_attention_heads", None),
    ]
    for candidate in candidates:
        try:
            value = int(candidate)
        except Exception:
            value = 0
        if value > 0:
            return value
    raise RuntimeError("Could not infer num_attention_heads")


def _get_attn_module(model, layer: int):
    layers = get_model_layers(model)
    attn_module = getattr(layers[int(layer)], "self_attn", None)
    if attn_module is None:
        attn_module = getattr(layers[int(layer)], "self_attention", None)
    if attn_module is None or not hasattr(attn_module, "o_proj"):
        raise AttributeError(f"Layer {layer} attention has no o_proj")
    return attn_module


def _capture_o_proj_inputs_at_position(model, input_ids: torch.Tensor, layers_to_capture: List[int], pos: int) -> Dict[int, torch.Tensor]:
    captured: Dict[int, torch.Tensor] = {}
    handles = []

    for layer in layers_to_capture:
        attn_module = _get_attn_module(model, int(layer))

        def make_hook(layer_idx: int):
            def pre_hook(module, inputs_tuple):
                if not inputs_tuple:
                    return None
                x = inputs_tuple[0]
                if not torch.is_tensor(x) or x.ndim != 3:
                    return None
                if int(pos) >= int(x.shape[1]):
                    return None
                captured[int(layer_idx)] = x[0, int(pos), :].detach().clone()
                return None
            return pre_hook

        handles.append(attn_module.o_proj.register_forward_pre_hook(make_hook(int(layer))))

    try:
        with torch.inference_mode():
            model(input_ids=input_ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    missing = [int(layer) for layer in layers_to_capture if int(layer) not in captured]
    if missing:
        raise RuntimeError(f"Failed to capture o_proj inputs for layers={missing}")
    return captured


class HookGroup:
    def __init__(self, handles):
        self.handles = list(handles)

    def remove(self) -> None:
        for handle in self.handles:
            try:
                handle.remove()
            except Exception:
                pass


def register_attention_head_replace_hooks(
    model,
    donor_vectors_by_layer: Dict[int, torch.Tensor],
    group_layers: Dict[int, List[int]],
    *,
    replace_position: int,
):
    handles = []
    for layer, heads in sorted(group_layers.items()):
        attn_module = _get_attn_module(model, int(layer))
        num_heads = _infer_num_heads(model, attn_module)
        donor_vec = donor_vectors_by_layer[int(layer)].detach().clone()
        if int(donor_vec.numel()) % int(num_heads) != 0:
            raise RuntimeError(f"Layer {layer}: donor vector shape incompatible with num_heads")
        head_dim = int(donor_vec.numel()) // int(num_heads)
        donor_heads = donor_vec.view(int(num_heads), int(head_dim))
        chosen_heads = [int(head) for head in heads]

        def make_hook(clean_heads: torch.Tensor, chosen: List[int], n_heads: int, pos: int):
            def pre_hook(module, inputs_tuple):
                if not inputs_tuple:
                    return None
                x = inputs_tuple[0]
                if not torch.is_tensor(x) or x.ndim != 3:
                    return None
                if int(pos) >= int(x.shape[1]):
                    return None
                d_model = int(x.shape[2])
                if d_model % int(n_heads) != 0:
                    return None
                head_dim_local = d_model // int(n_heads)
                x_new = x.clone()
                row = x_new[0, int(pos), :].view(int(n_heads), head_dim_local)
                src = clean_heads.to(device=row.device, dtype=row.dtype)
                for head_idx in chosen:
                    if 0 <= int(head_idx) < int(n_heads):
                        row[int(head_idx), :] = src[int(head_idx), :]
                x_new[0, int(pos), :] = row.view(-1)
                if len(inputs_tuple) == 1:
                    return (x_new,)
                return (x_new,) + tuple(inputs_tuple[1:])
            return pre_hook

        handles.append(
            attn_module.o_proj.register_forward_pre_hook(
                make_hook(donor_heads, chosen_heads, num_heads, int(replace_position))
            )
        )
    return HookGroup(handles)


def _load_top_heads(path: Path, topk: int) -> List[Tuple[int, int]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [(int(row["layer"]), int(row["head"])) for row in rows[: int(topk)]]


def _group_by_layer(items: List[Tuple[int, int]]) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = defaultdict(list)
    for layer, head in items:
        if int(head) not in out[int(layer)]:
            out[int(layer)].append(int(head))
    return {int(k): list(v) for k, v in sorted(out.items())}


def _make_random_matched_group(reference_group: Dict[int, List[int]], num_heads: int, seed: int) -> Dict[int, List[int]]:
    rng = random.Random(seed)
    out: Dict[int, List[int]] = {}
    for layer, heads in sorted(reference_group.items()):
        pool = [h for h in range(int(num_heads)) if h not in set(int(x) for x in heads)]
        rng.shuffle(pool)
        out[int(layer)] = sorted(pool[: len(heads)])
    return out


def _derive_groups(artifact_dir: Path, num_heads: int, topk_intersection: int, seed: int) -> Dict[str, Dict[int, List[int]]]:
    hin = _load_top_heads(artifact_dir / "top_heads_4b_hin_multilang.json", topk_intersection)
    tel = _load_top_heads(artifact_dir / "top_heads_4b_tel_multilang.json", topk_intersection)
    tel_set = set(tel)
    intersection = [x for x in hin if x in tel_set]
    shared_l17 = [x for x in intersection if int(x[0]) == 17]
    l24_h0 = [x for x in intersection if int(x[0]) == 24 and int(x[1]) == 0]
    l17_plus_l24 = list(shared_l17) + [x for x in l24_h0 if x not in shared_l17]
    shared_union = list(intersection)
    shared_union_group = _group_by_layer(shared_union)
    return {
        "shared_l17": _group_by_layer(shared_l17),
        "l24_h0": _group_by_layer(l24_h0),
        "l17_plus_l24h0": _group_by_layer(l17_plus_l24),
        "shared_union": shared_union_group,
        "random_matched_shared": _make_random_matched_group(shared_union_group, num_heads=num_heads, seed=seed + 123),
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


def _prepare_prompt_payload(tokenizer, prompt_text: str, query_text: str) -> Tuple[torch.Tensor, int]:
    rendered = apply_chat_template(tokenizer, prompt_text)
    input_ids = tokenizer(rendered, return_tensors="pt").input_ids
    regions = _extract_prompt_regions(
        tokenizer=tokenizer,
        raw_prompt=prompt_text,
        rendered_prompt=rendered,
        query_token=query_text,
    )
    query_span = regions.get("query_span")
    if query_span is None:
        raise RuntimeError(f"Query span localization failed for query={query_text}")
    return input_ids, int(query_span[1] - 1)


def _mean(vals: List[float]) -> float:
    clean = [float(v) for v in vals if np.isfinite(float(v))]
    return float(np.mean(clean)) if clean else float("nan")


def _recovery_ratio_higher_better(base: float, patched: float, donor: float) -> float:
    if not (np.isfinite(base) and np.isfinite(patched) and np.isfinite(donor)):
        return float("nan")
    denom = float(donor - base)
    if denom <= 0.0:
        return float("nan")
    return float((patched - base) / denom)


def _recovery_ratio_lower_better(base: float, patched: float, donor: float) -> float:
    if not (np.isfinite(base) and np.isfinite(patched) and np.isfinite(donor)):
        return float("nan")
    denom = float(base - donor)
    if denom <= 0.0:
        return float("nan")
    return float((base - patched) / denom)


def main() -> int:
    args = parse_args()
    set_all_seeds(int(args.seed))

    pair_ids = [x.strip() for x in str(args.pairs).split(",") if x.strip()]
    pair_specs = {
        "aksharantar_hin_latin": ("aksharantar_tel_latin", 16),
        "aksharantar_tel_latin": ("aksharantar_hin_latin", 24),
    }
    if any(pair_id not in pair_specs for pair_id in pair_ids):
        raise RuntimeError(f"Unexpected pair ids: {pair_ids}")

    model, tokenizer = load_model("4b", device=str(args.device))
    device = str(next(model.parameters()).device)
    pad_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", 0)
    num_heads = _infer_num_heads(model, _get_attn_module(model, 0))

    artifact_dir = (PROJECT_ROOT / str(args.artifact_dir)).resolve()
    groups = _derive_groups(
        artifact_dir=artifact_dir,
        num_heads=num_heads,
        topk_intersection=int(args.topk_intersection),
        seed=int(args.seed),
    )
    all_layers = sorted({int(layer) for group in groups.values() for layer in group.keys()})
    log(f"Derived groups: {groups}")

    summary_rows = []
    item_rows = []

    for pair_id in pair_ids:
        alt_pair_id, helpful_n = pair_specs[pair_id]
        meta = dict(get_pair_prompt_metadata(pair_id))
        source_lang, input_script, output_script = _prompt_naming(meta)

        words, _ = _load_words(pair_id, external_only=True, require_external_sources=True, min_pool_size=500)
        icl64, _, eval_rows = split_data_three_way(
            words,
            n_icl=int(args.n_icl),
            n_select=int(args.n_select),
            n_eval=200,
            seed=int(args.seed),
        )
        eval_rows = eval_rows[: int(args.n_eval)]
        helpful = list(icl64[-int(helpful_n) :])
        same_fillers = list(icl64[-int(helpful_n) - int(args.filler_n) : -int(helpful_n)])

        alt_words, _ = _load_words(alt_pair_id, external_only=True, require_external_sources=True, min_pool_size=500)
        alt_icl64, _, _ = split_data_three_way(
            alt_words,
            n_icl=int(args.n_icl),
            n_select=int(args.n_select),
            n_eval=200,
            seed=int(args.seed),
        )
        wrong_fillers = list(alt_icl64[-int(args.filler_n) :])

        condition_banks = {
            "explicit_zs": None,
            "same_fillers_helpful_tail": list(same_fillers) + list(helpful),
            "wrong_fillers_helpful_tail": list(wrong_fillers) + list(helpful),
        }

        cached = []
        log(f"Preparing donor/recipient cache for pair={pair_id} n_eval={len(eval_rows)}")
        for idx, word in enumerate(eval_rows):
            if idx > 0 and idx % 20 == 0:
                log(f"  cached {idx}/{len(eval_rows)}")
            query = str(word["ood"])
            target = str(word["hindi"])
            target_ids = tokenizer.encode(target, add_special_tokens=False)
            target_id = int(target_ids[0]) if target_ids else -1

            donor_prompt = build_task_prompt(
                query,
                list(icl64),
                input_script_name=input_script,
                source_language=source_lang,
                output_script_name=output_script,
                prompt_variant="canonical",
            )
            donor_input_ids_cpu, donor_pos = _prepare_prompt_payload(tokenizer, donor_prompt, query)
            donor_input_ids = donor_input_ids_cpu.to(device)
            donor_vectors = _capture_o_proj_inputs_at_position(model, donor_input_ids, all_layers, donor_pos)
            donor_tf = _teacher_forced_metrics_from_input_ids(
                model=model,
                input_ids=donor_input_ids,
                target_ids=target_ids,
                target_id=target_id,
                device=device,
                competitor_id=-1,
            )
            donor_pred = _generate_text(model, tokenizer, donor_input_ids, pad_id=pad_id, max_new_tokens=int(args.max_new_tokens))
            donor_gen = _generation_metrics(target, donor_pred, target_script=output_script)

            recipients = {}
            for condition_name, bank in condition_banks.items():
                prompt = build_task_prompt(
                    query,
                    None if bank is None else list(bank),
                    input_script_name=input_script,
                    source_language=source_lang,
                    output_script_name=output_script,
                    prompt_variant="canonical",
                )
                recip_input_ids_cpu, recip_pos = _prepare_prompt_payload(tokenizer, prompt, query)
                recip_input_ids = recip_input_ids_cpu.to(device)
                recip_tf = _teacher_forced_metrics_from_input_ids(
                    model=model,
                    input_ids=recip_input_ids,
                    target_ids=target_ids,
                    target_id=target_id,
                    device=device,
                    competitor_id=-1,
                )
                recip_pred = _generate_text(model, tokenizer, recip_input_ids, pad_id=pad_id, max_new_tokens=int(args.max_new_tokens))
                recip_gen = _generation_metrics(target, recip_pred, target_script=output_script)
                recipients[condition_name] = {
                    "prompt_ids": recip_input_ids,
                    "query_pos": recip_pos,
                    "tf": recip_tf,
                    "gen": recip_gen,
                }

            cached.append(
                {
                    "idx": idx,
                    "word": word,
                    "target_ids": target_ids,
                    "target_id": target_id,
                    "donor_vectors": donor_vectors,
                    "donor_tf": donor_tf,
                    "donor_gen": donor_gen,
                    "recips": recipients,
                }
            )

        for recipient_name in condition_banks.keys():
            for group_name, group_layers in groups.items():
                log(f"Running pair={pair_id} recipient={recipient_name} group={group_name} layers={group_layers}")
                group_item_rows = []
                for item in cached:
                    recip = item["recips"][recipient_name]
                    hook_group = register_attention_head_replace_hooks(
                        model,
                        donor_vectors_by_layer=item["donor_vectors"],
                        group_layers=group_layers,
                        replace_position=int(recip["query_pos"]),
                    )
                    try:
                        patched_tf = _teacher_forced_metrics_from_input_ids(
                            model=model,
                            input_ids=recip["prompt_ids"],
                            target_ids=item["target_ids"],
                            target_id=item["target_id"],
                            device=device,
                            competitor_id=-1,
                        )
                        patched_pred = _generate_text(
                            model,
                            tokenizer,
                            recip["prompt_ids"],
                            pad_id=pad_id,
                            max_new_tokens=int(args.max_new_tokens),
                        )
                    finally:
                        hook_group.remove()
                    patched_gen = _generation_metrics(str(item["word"]["hindi"]), patched_pred, target_script=output_script)

                    row = {
                        "pair": pair_id,
                        "recipient_condition": recipient_name,
                        "group": group_name,
                        "group_layers": {str(k): list(v) for k, v in group_layers.items()},
                        "item_index": int(item["idx"]),
                        "word_source": str(item["word"]["ood"]),
                        "word_target": str(item["word"]["hindi"]),
                        "donor_exact_match": float(item["donor_gen"]["exact_match"]),
                        "donor_first_entry": float(item["donor_gen"]["first_entry_correct"]),
                        "donor_cer": float(item["donor_gen"]["akshara_cer"]),
                        "donor_nll_pos1": float(item["donor_tf"].get("target_pos1_nll", float("nan"))),
                        "base_exact_match": float(recip["gen"]["exact_match"]),
                        "base_first_entry": float(recip["gen"]["first_entry_correct"]),
                        "base_cer": float(recip["gen"]["akshara_cer"]),
                        "base_continuation": float(recip["gen"]["continuation_fidelity"]),
                        "base_nll_pos1": float(recip["tf"].get("target_pos1_nll", float("nan"))),
                        "base_nll_pos2": float(recip["tf"].get("target_pos2_nll", float("nan"))),
                        "base_nll_pos3": float(recip["tf"].get("target_pos3_nll", float("nan"))),
                        "patched_exact_match": float(patched_gen["exact_match"]),
                        "patched_first_entry": float(patched_gen["first_entry_correct"]),
                        "patched_cer": float(patched_gen["akshara_cer"]),
                        "patched_continuation": float(patched_gen["continuation_fidelity"]),
                        "patched_nll_pos1": float(patched_tf.get("target_pos1_nll", float("nan"))),
                        "patched_nll_pos2": float(patched_tf.get("target_pos2_nll", float("nan"))),
                        "patched_nll_pos3": float(patched_tf.get("target_pos3_nll", float("nan"))),
                    }
                    item_rows.append(row)
                    group_item_rows.append(row)

                summary_rows.append(
                    {
                        "pair": pair_id,
                        "recipient_condition": recipient_name,
                        "group": group_name,
                        "group_layers": {str(k): list(v) for k, v in group_layers.items()},
                        "exact_match_gain": _mean([r["patched_exact_match"] - r["base_exact_match"] for r in group_item_rows]),
                        "first_entry_gain": _mean([r["patched_first_entry"] - r["base_first_entry"] for r in group_item_rows]),
                        "continuation_gain": _mean([r["patched_continuation"] - r["base_continuation"] for r in group_item_rows]),
                        "cer_improvement": _mean([r["base_cer"] - r["patched_cer"] for r in group_item_rows]),
                        "nll_pos1_improvement": _mean([r["base_nll_pos1"] - r["patched_nll_pos1"] for r in group_item_rows]),
                        "nll_pos2_improvement": _mean([r["base_nll_pos2"] - r["patched_nll_pos2"] for r in group_item_rows]),
                        "nll_pos3_improvement": _mean([r["base_nll_pos3"] - r["patched_nll_pos3"] for r in group_item_rows]),
                        "first_entry_recovery_ratio": _mean([
                            _recovery_ratio_higher_better(r["base_first_entry"], r["patched_first_entry"], r["donor_first_entry"])
                            for r in group_item_rows
                        ]),
                        "nll_pos1_recovery_ratio": _mean([
                            _recovery_ratio_lower_better(r["base_nll_pos1"], r["patched_nll_pos1"], r["donor_nll_pos1"])
                            for r in group_item_rows
                        ]),
                    }
                )
                log(f"Completed pair={pair_id} recipient={recipient_name} group={group_name}")

    out_path = (PROJECT_ROOT / str(args.out)).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "shared_head_sufficiency_panel",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": "4b",
        "pairs": pair_ids,
        "recipient_conditions": ["explicit_zs", "same_fillers_helpful_tail", "wrong_fillers_helpful_tail"],
        "groups": {k: {str(layer): list(heads) for layer, heads in v.items()} for k, v in groups.items()},
        "summary": summary_rows,
        "items": item_rows,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log(f"Saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
