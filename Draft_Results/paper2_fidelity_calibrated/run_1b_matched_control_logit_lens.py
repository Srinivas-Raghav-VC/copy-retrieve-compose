#!/usr/bin/env python3
"""
CRITICAL VERIFICATION EXPERIMENT:
Logit lens with MATCHED-LENGTH controls for 1B.

Compares target token rank/probability at every layer for:
  1. helpful_icl  — N correct transliteration examples + query
  2. corrupt_icl  — N examples with shuffled outputs (same length)
  3. random_icl   — N random script-matched pairs (same length)
  4. null_filler   — filler tokens matching ICL length + query
  5. zs            — zero-shot (SHORT — the unmatched baseline)

If helpful_icl has better target rank than corrupt/random/null at late layers,
the ICL rescue trajectory is REAL and content-specific, not a length artifact.

If all long-prompt conditions show similar trajectories (and differ only from ZS),
then the logit lens is just measuring prompt length.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_model_config
from core import (
    apply_chat_template,
    build_corrupted_icl_prompt,
    build_null_icl_prompt,
    build_random_icl_prompt,
    build_task_prompt,
    get_model_layers,
    load_model,
    set_all_seeds,
)
from paper2_fidelity_calibrated.phase1_common import log


def load_pair_split_flexible(pair_id, seed, n_icl, n_select, n_eval, external_only, require_external_sources, min_pool_size):
    """Load pair split, falling back to direct JSONL loading for unregistered pairs."""
    try:
        from paper2_fidelity_calibrated.phase1_common import load_pair_split
        return load_pair_split(
            pair_id, seed=seed, n_icl=n_icl, n_select=n_select, n_eval=n_eval,
            external_only=external_only, require_external_sources=require_external_sources,
            min_pool_size=min_pool_size,
        )
    except (ValueError, KeyError):
        pass

    # Direct loading for unregistered pairs
    import random as _random
    data_path = PROJECT_ROOT / "data" / "transliteration" / (pair_id + ".jsonl")
    meta_path = PROJECT_ROOT / "data" / "transliteration" / (pair_id + ".jsonl.meta.json")
    if not data_path.exists():
        raise FileNotFoundError("Missing data file: %s" % data_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    ds_meta = meta.get("dataset", {})

    words = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line.strip())
            src = str(row.get("source", row.get("english word", ""))).strip()
            tgt = str(row.get("target", row.get("native word", ""))).strip()
            if src and tgt:
                words.append({"ood": src, "hindi": tgt, "english": src})

    rng = _random.Random(seed)
    rng.shuffle(words)
    total_needed = n_icl + n_select + n_eval
    if len(words) < total_needed:
        n_eval = min(n_eval, max(1, len(words) - n_icl - n_select))
    icl_examples = words[:n_icl]
    select_rows = words[n_icl:n_icl + n_select]
    eval_rows = words[n_icl + n_select:n_icl + n_select + n_eval]

    return {
        "pair": pair_id,
        "words": words,
        "provenance": {"pair_id": pair_id, "total_rows": len(words)},
        "prompt_meta": ds_meta,
        "source_language": str(ds_meta.get("source_language", "Hindi")),
        "input_script_name": str(ds_meta.get("source_script", "Latin")),
        "output_script_name": str(ds_meta.get("target_script", "Devanagari")),
        "icl_examples": icl_examples,
        "select_rows": select_rows,
        "eval_rows": eval_rows,
    }


def _json_safe(v: Any) -> Any:
    if isinstance(v, (str, int, bool)) or v is None: return v
    if isinstance(v, float): return v if np.isfinite(v) else None
    if isinstance(v, dict): return {str(k): _json_safe(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)): return [_json_safe(x) for x in v]
    return v


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(data), indent=2, ensure_ascii=False), encoding="utf-8")


def _get_final_norm(model):
    for chain in [("model","norm"),("language_model","norm"),("model","model","norm")]:
        cur = model
        ok = True
        for name in chain:
            if not hasattr(cur, name):
                ok = False; break
            cur = getattr(cur, name)
        if ok and isinstance(cur, torch.nn.Module):
            return cur
    raise AttributeError("Could not find final norm")


def _logit_lens_all_layers(model, tokenizer, input_ids: torch.Tensor, target_id: int) -> List[Dict[str, float]]:
    """Run logit lens at every layer for a single input."""
    layers = get_model_layers(model)
    n_layers = len(layers)
    lm_head = model.lm_head if hasattr(model, "lm_head") else model.get_output_embeddings()
    final_norm = _get_final_norm(model)
    last_pos = int(input_ids.shape[1] - 1)

    # Capture hidden states at every layer
    hidden_states = {}
    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            hidden_states[layer_idx] = h[0, last_pos, :].detach().float()
        return hook_fn

    handles = []
    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(i)))

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, use_cache=False)

    for h in handles:
        h.remove()

    # Also get the final logits directly
    final_logits = outputs.logits[0, last_pos, :].detach().float()
    final_probs = torch.softmax(final_logits, dim=-1)

    results = []
    for layer_idx in range(n_layers):
        h = hidden_states.get(layer_idx)
        if h is None:
            continue
        # Apply final norm + lm_head to get logit lens readout
        # Must match model dtype (bfloat16) for lm_head
        h_model_dtype = h.to(dtype=next(lm_head.parameters()).dtype)
        normed = final_norm(h_model_dtype.unsqueeze(0).unsqueeze(0)).squeeze()
        logits = lm_head(normed.unsqueeze(0)).squeeze()
        probs = torch.softmax(logits.float(), dim=-1)

        target_prob = float(probs[target_id].item())
        target_logit = float(logits[target_id].item())
        sorted_indices = torch.argsort(logits, descending=True)
        target_rank = int((sorted_indices == target_id).nonzero(as_tuple=True)[0].item()) + 1

        results.append({
            "layer": layer_idx,
            "target_prob": target_prob,
            "target_logit": target_logit,
            "target_rank": target_rank,
        })

    # Final layer (actual model output)
    results.append({
        "layer": n_layers,
        "target_prob": float(final_probs[target_id].item()),
        "target_logit": float(final_logits[target_id].item()),
        "target_rank": int((torch.argsort(final_logits, descending=True) == target_id).nonzero(as_tuple=True)[0].item()) + 1,
    })

    return results


def parse_args():
    ap = argparse.ArgumentParser(description="Matched-control logit lens for 1B")
    ap.add_argument("--model", default="1b", choices=["1b", "4b"])
    ap.add_argument("--pair", default="aksharantar_hin_latin")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-icl", type=int, default=16)
    ap.add_argument("--n-select", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--max-items", type=int, default=20)
    ap.add_argument("--external-only", action="store_true")
    ap.add_argument("--require-external-sources", action="store_true")
    ap.add_argument("--min-pool-size", type=int, default=500)
    ap.add_argument("--out", default="")
    return ap.parse_args()


def main():
    args = parse_args()
    set_all_seeds(args.seed)

    pair_bundle = load_pair_split_flexible(
        args.pair, seed=args.seed, n_icl=args.n_icl,
        n_select=args.n_select, n_eval=args.n_eval,
        external_only=args.external_only,
        require_external_sources=args.require_external_sources,
        min_pool_size=args.min_pool_size,
    )

    model, tokenizer = load_model(args.model, device=args.device)
    device = str(next(model.parameters()).device)

    out_root = (
        Path(args.out).resolve() if args.out.strip()
        else PROJECT_ROOT / "paper2_fidelity_calibrated" / "results" / "matched_control_logit_lens" / args.pair / args.model
    )
    out_root.mkdir(parents=True, exist_ok=True)

    eval_rows = list(pair_bundle["eval_rows"][:args.max_items])
    icl_examples = list(pair_bundle["icl_examples"])
    src_script = pair_bundle["input_script_name"]
    src_lang = pair_bundle["source_language"]
    tgt_script = pair_bundle["output_script_name"]

    all_items = []
    log("Running matched-control logit lens: model=%s pair=%s items=%d n_icl=%d" % (args.model, args.pair, len(eval_rows), args.n_icl))

    for item_idx, word in enumerate(eval_rows):
        query = str(word["ood"])
        target_text = str(word["hindi"])
        target_ids = tokenizer.encode(target_text, add_special_tokens=False)
        if not target_ids:
            continue
        target_id = int(target_ids[0])

        # Build all 5 conditions
        conditions = {}

        # 1. Helpful ICL
        helpful_raw = build_task_prompt(query, icl_examples, input_script_name=src_script,
                                        source_language=src_lang, output_script_name=tgt_script)
        conditions["helpful_icl"] = apply_chat_template(tokenizer, helpful_raw)

        # 2. Corrupted ICL (same length, shuffled outputs)
        corrupt_raw = build_corrupted_icl_prompt(query, icl_examples, input_script_name=src_script,
                                                  source_language=src_lang, output_script_name=tgt_script,
                                                  seed=args.seed)
        conditions["corrupt_icl"] = apply_chat_template(tokenizer, corrupt_raw)

        # 3. Random ICL (same length, random script-matched pairs)
        random_raw = build_random_icl_prompt(query, len(icl_examples), input_script_name=src_script,
                                              source_language=src_lang, output_script_name=tgt_script,
                                              use_indic_control=True, length_reference_examples=icl_examples,
                                              seed=args.seed)
        conditions["random_icl"] = apply_chat_template(tokenizer, random_raw)

        # 4. Null filler (same token budget, no examples)
        helpful_toks = len(tokenizer(conditions["helpful_icl"], return_tensors="pt")["input_ids"][0])
        zs_raw = build_task_prompt(query, None, input_script_name=src_script,
                                    source_language=src_lang, output_script_name=tgt_script)
        zs_toks = len(tokenizer(apply_chat_template(tokenizer, zs_raw), return_tensors="pt")["input_ids"][0])
        filler_budget = max(32, helpful_toks - zs_toks)
        null_raw = build_null_icl_prompt(query, input_script_name=src_script,
                                          source_language=src_lang, output_script_name=tgt_script,
                                          seed=args.seed, target_token_budget=filler_budget)
        conditions["null_filler"] = apply_chat_template(tokenizer, null_raw)

        # 5. Zero-shot (SHORT — unmatched baseline)
        conditions["zs"] = apply_chat_template(tokenizer, zs_raw)

        # Run logit lens for each condition
        item_result = {
            "item_index": item_idx,
            "word_ood": query,
            "word_hindi": target_text,
            "target_id": target_id,
            "target_token": tokenizer.decode([target_id]),
            "prompt_lengths": {},
            "trajectories": {},
        }

        for cond_name, rendered in conditions.items():
            input_ids = tokenizer(rendered, return_tensors="pt").to(device).input_ids
            item_result["prompt_lengths"][cond_name] = int(input_ids.shape[1])
            trajectory = _logit_lens_all_layers(model, tokenizer, input_ids, target_id)
            item_result["trajectories"][cond_name] = trajectory

        all_items.append(item_result)
        if (item_idx + 1) % 5 == 0 or item_idx == 0:
            log("[%d/%d] %s -> %s  helpful_len=%d zs_len=%d" % (
                item_idx+1, len(eval_rows), query, target_text,
                item_result["prompt_lengths"].get("helpful_icl", 0),
                item_result["prompt_lengths"].get("zs", 0)))

    # Aggregate summary
    conditions_list = ["helpful_icl", "corrupt_icl", "random_icl", "null_filler", "zs"]
    n_layers_max = max(len(item["trajectories"].get("zs", [])) for item in all_items) if all_items else 0
    summary = []
    for cond in conditions_list:
        for layer in range(n_layers_max):
            ranks = []
            probs = []
            for item in all_items:
                traj = item["trajectories"].get(cond, [])
                if layer < len(traj):
                    ranks.append(traj[layer]["target_rank"])
                    probs.append(traj[layer]["target_prob"])
            if ranks:
                summary.append({
                    "condition": cond,
                    "layer": layer,
                    "n_items": len(ranks),
                    "mean_target_rank": float(np.mean(ranks)),
                    "median_target_rank": float(np.median(ranks)),
                    "mean_target_prob": float(np.mean(probs)),
                })

    payload = {
        "experiment": "matched_control_logit_lens",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": args.model,
        "pair": args.pair,
        "seed": args.seed,
        "n_icl": args.n_icl,
        "n_items": len(all_items),
        "conditions": conditions_list,
        "notes": {
            "purpose": "Verify whether ICL internal trajectory is content-specific or a prompt-length artifact",
            "key_comparison": "helpful_icl vs corrupt_icl at same prompt length",
            "if_helpful_better_than_corrupt": "ICL rescue trajectory is REAL and content-specific",
            "if_all_long_prompts_similar": "logit lens measures prompt length, not ICL content",
        },
        "summary": summary,
        "items": all_items,
    }
    _write_json(out_root / "matched_control_logit_lens.json", payload)
    log("Saved: %s" % (out_root / "matched_control_logit_lens.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
