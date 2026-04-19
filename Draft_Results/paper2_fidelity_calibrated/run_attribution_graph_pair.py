#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import (  # noqa: E402
    apply_chat_template,
    build_task_prompt,
    get_model_layers,
    load_model,
    set_all_seeds,
    split_data_three_way,
)
from paper2_fidelity_calibrated.run import _load_words, _prompt_naming  # noqa: E402
from rescue_research.data_pipeline.ingest import get_pair_prompt_metadata  # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Pair-configurable attention-head attribution ranking.")
    ap.add_argument("--pair", required=True)
    ap.add_argument("--model", default="4b", choices=["1b", "4b"])
    ap.add_argument("--batch-words", type=int, default=15)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--n-icl", type=int, default=64)
    ap.add_argument("--n-select", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--external-only", action="store_true")
    ap.add_argument("--require-external-sources", action="store_true")
    ap.add_argument("--min-pool-size", type=int, default=500)
    ap.add_argument("--out-tag", type=str, default="")
    ap.add_argument("--out", type=str, default="")
    return ap.parse_args()


def _get_head_dimensions(model) -> tuple[int, int]:
    config = getattr(model.config, "text_config", model.config)
    return int(config.num_attention_heads), int(config.head_dim)


def _run_attention_attribution(
    model,
    tokenizer,
    eval_rows,
    icl_examples,
    source_lang: str,
    input_script: str,
    output_script: str,
    device: str,
    *,
    batch_words: int,
) -> torch.Tensor:
    layers = get_model_layers(model)
    n_layers = len(layers)
    num_heads, head_dim = _get_head_dimensions(model)
    scores = torch.zeros((n_layers, num_heads), dtype=torch.float32, device="cpu")
    batch_rows = list(eval_rows[: int(batch_words)])

    for word_idx, word in enumerate(batch_rows):
        log(f"Processing {word_idx + 1}/{len(batch_rows)}: {word['ood']} -> {word['hindi']}")
        query = str(word["ood"])
        prompt_icl = build_task_prompt(
            query,
            list(icl_examples),
            input_script_name=input_script,
            source_language=source_lang,
            output_script_name=output_script,
            prompt_variant="canonical",
        )
        prompt_zs = build_task_prompt(
            query,
            None,
            input_script_name=input_script,
            source_language=source_lang,
            output_script_name=output_script,
            prompt_variant="canonical",
        )

        target_ids = tokenizer.encode(str(word["hindi"]).strip(), add_special_tokens=False)
        if not target_ids:
            continue
        gold_id = int(target_ids[0])

        ids_icl = tokenizer(apply_chat_template(tokenizer, prompt_icl), return_tensors="pt").input_ids.to(device)
        ids_zs = tokenizer(apply_chat_template(tokenizer, prompt_zs), return_tensors="pt").input_ids.to(device)

        pos_icl = int(ids_icl.shape[1] - 1)
        pos_zs = int(ids_zs.shape[1] - 1)
        clean_head_outputs: Dict[int, torch.Tensor] = {}

        def make_clean_capture_hook(layer_idx: int):
            def hook(module, args, kwargs):
                h_concat = args[0][0, pos_icl, :].detach().clone()
                clean_head_outputs[int(layer_idx)] = h_concat.view(num_heads, head_dim)
                return args, kwargs
            return hook

        clean_handles = []
        for layer_idx in range(n_layers):
            handle = layers[layer_idx].self_attn.o_proj.register_forward_pre_hook(
                make_clean_capture_hook(layer_idx),
                with_kwargs=True,
            )
            clean_handles.append(handle)
        try:
            with torch.inference_mode():
                out_clean = model(input_ids=ids_icl, use_cache=False)
                clean_logit = float(out_clean.logits[0, pos_icl, gold_id].item())
        finally:
            for handle in clean_handles:
                handle.remove()

        with torch.inference_mode():
            out_zs = model(input_ids=ids_zs, use_cache=False)
            zs_logit = float(out_zs.logits[0, pos_zs, gold_id].item())

        total_effect = clean_logit - zs_logit
        if total_effect <= 0.0:
            continue

        for layer_idx in range(n_layers):
            for head_idx in range(num_heads):
                def make_patch_hook(clean_val: torch.Tensor):
                    def hook(module, args, kwargs):
                        h_concat = args[0].clone()
                        h_reshaped = h_concat[0, pos_zs, :].view(num_heads, head_dim)
                        h_reshaped[head_idx, :] = clean_val.to(device=h_reshaped.device, dtype=h_reshaped.dtype)
                        h_concat[0, pos_zs, :] = h_reshaped.view(-1)
                        return (h_concat,) + args[1:], kwargs
                    return hook

                patch_handle = layers[layer_idx].self_attn.o_proj.register_forward_pre_hook(
                    make_patch_hook(clean_head_outputs[layer_idx][head_idx]),
                    with_kwargs=True,
                )
                try:
                    with torch.inference_mode():
                        patched_out = model(input_ids=ids_zs, use_cache=False)
                        patched_logit = float(patched_out.logits[0, pos_zs, gold_id].item())
                finally:
                    patch_handle.remove()

                recovered_effect = patched_logit - zs_logit
                scores[layer_idx, head_idx] += float(recovered_effect / total_effect)

    if batch_rows:
        scores /= float(len(batch_rows))
    return scores


def main() -> int:
    args = parse_args()
    set_all_seeds(int(args.seed))

    model, tokenizer = load_model(str(args.model), device=str(args.device))
    device = str(next(model.parameters()).device)

    prompt_meta = dict(get_pair_prompt_metadata(str(args.pair)))
    source_lang, input_script, output_script = _prompt_naming(prompt_meta)
    words, _ = _load_words(
        str(args.pair),
        external_only=bool(args.external_only),
        require_external_sources=bool(args.require_external_sources),
        min_pool_size=int(args.min_pool_size),
    )
    icl_examples, _, eval_rows = split_data_three_way(
        words,
        n_icl=int(args.n_icl),
        n_select=int(args.n_select),
        n_eval=int(args.n_eval),
        seed=int(args.seed),
    )

    scores = _run_attention_attribution(
        model,
        tokenizer,
        eval_rows,
        icl_examples,
        source_lang,
        input_script,
        output_script,
        device,
        batch_words=int(args.batch_words),
    )

    flat_scores = scores.flatten()
    top_indices = torch.topk(flat_scores, int(args.topk)).indices
    num_heads, _ = _get_head_dimensions(model)

    results = []
    log(f"=== TOP {int(args.topk)} HEADS: {args.pair} ===")
    for rank, flat_idx in enumerate(top_indices.tolist(), start=1):
        layer = int(flat_idx) // int(num_heads)
        head = int(flat_idx) % int(num_heads)
        value = float(flat_scores[int(flat_idx)].item())
        log(f"{rank:2d} | L{layer:02d} H{head:02d} | {value:>7.2%}")
        results.append(
            {
                "rank": int(rank),
                "layer": int(layer),
                "head": int(head),
                "effect": value,
                "pair": str(args.pair),
                "model": str(args.model),
            }
        )

    if str(args.out).strip():
        out_path = Path(str(args.out)).resolve()
    else:
        lang = str(args.pair).split("_")[1]
        suffix = f"_{args.out_tag}" if str(args.out_tag).strip() else ""
        out_path = PROJECT_ROOT / "artifacts" / "phase5_attribution" / f"top_heads_{args.model}_{lang}{suffix}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log(f"Saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
