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

from paper2_fidelity_calibrated.run_dense_mlp_sweep import (  # noqa: E402
    _deterministic_shuffle_vector,
    _get_pair_prompt_metadata,
    _git_commit_hash,
    _load_words,
    _prompt_fingerprint,
    _runtime_identity,
    _write_json,
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _parse_csv(raw: str) -> List[int]:
    return [int(x.strip()) for x in str(raw or "").split(",") if x.strip()]


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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Component-localization panel at selected layers.")
    ap.add_argument("--model", type=str, default="4b", choices=["4b"])
    ap.add_argument("--pair", type=str, default="aksharantar_hin_latin")
    ap.add_argument("--alt-pair", type=str, default="aksharantar_tel_latin")
    ap.add_argument("--layers", type=str, default="24,25")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-icl", type=int, default=64)
    ap.add_argument("--n-select", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--external-only", action="store_true")
    ap.add_argument("--require-external-sources", action="store_true")
    ap.add_argument("--min-pool-size", type=int, default=500)
    ap.add_argument("--out", type=str, default="")
    return ap.parse_args()


def main() -> int:
    from core import (
        _extract_attention_output_at_position_from_input_ids,
        _extract_mlp_io_at_position_from_input_ids,
        _find_last_subsequence,
        apply_chat_template,
        build_task_prompt,
        get_model_config,
        load_model,
        register_attention_output_replace_hook,
        register_dense_mlp_output_patch_hook,
        set_all_seeds,
        split_data_three_way,
    )
    from paper2_fidelity_calibrated.eval_utils import (
        akshara_cer,
        continuation_akshara_cer,
        first_entry_correct,
        normalize_text,
        script_compliance,
    )
    from paper2_fidelity_calibrated.run import _prompt_naming
    from rescue_research.data_pipeline.ingest import get_pair_prompt_metadata

    args = parse_args()
    set_all_seeds(int(args.seed))
    layers = _parse_csv(args.layers)
    if not layers:
        raise RuntimeError("No layers provided.")

    words, provenance = _load_words(
        args.pair,
        external_only=bool(args.external_only),
        require_external_sources=bool(args.require_external_sources),
        min_pool_size=int(args.min_pool_size),
    )
    alt_words, alt_provenance = _load_words(
        args.alt_pair,
        external_only=bool(args.external_only),
        require_external_sources=bool(args.require_external_sources),
        min_pool_size=int(args.min_pool_size),
    )

    prompt_meta = dict(get_pair_prompt_metadata(args.pair))
    source_language, input_script_name, target_script = _prompt_naming(prompt_meta)

    icl_examples, _, eval_samples = split_data_three_way(
        words,
        n_icl=int(args.n_icl),
        n_select=int(args.n_select),
        n_eval=int(args.n_eval),
        seed=int(args.seed),
    )
    alt_icl_examples, _, _ = split_data_three_way(
        alt_words,
        n_icl=int(args.n_icl),
        n_select=int(args.n_select),
        n_eval=max(10, int(args.n_eval)),
        seed=int(args.seed),
    )

    cfg = get_model_config(args.model)
    model, tokenizer = load_model(args.model, device=str(args.device))

    out_root = (
        Path(args.out).resolve()
        if str(args.out).strip()
        else PROJECT_ROOT / "paper2_fidelity_calibrated" / "results" / "component_localization_panel" / args.pair / args.model
    )
    out_root.mkdir(parents=True, exist_ok=True)

    example_prompt = build_task_prompt(
        eval_samples[0]["ood"],
        icl_examples,
        input_script_name=input_script_name,
        source_language=source_language,
        output_script_name=target_script,
        prompt_variant="canonical",
    )
    rendered_example = apply_chat_template(tokenizer, example_prompt)
    config = {
        "experiment": "component_localization_panel",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": args.model,
        "pair": args.pair,
        "alt_pair": args.alt_pair,
        "seed": int(args.seed),
        "runtime_identity": _runtime_identity(model_key=args.model, hf_id=cfg.hf_id, tokenizer=tokenizer, model=model),
        "git_commit_hash": _git_commit_hash(),
        "prompt_fingerprint": _prompt_fingerprint(raw_prompt=example_prompt, rendered_prompt=rendered_example),
        "frozen_prompt_variant": "canonical",
        "claim_level": "intervention_only",
        "layers": layers,
        "components": ["attention_output", "mlp_output"],
        "controls": ["correct", "wrong_language", "shuffled"],
        "hook_semantics": {
            "attention_output": "replace self-attention output vector at source_last_subtoken",
            "mlp_output": "replace MLP output vector at source_last_subtoken",
            "generation_mode": "one-shot prefill patch via absolute prompt position",
        },
        "split_sizes": {"n_icl": int(args.n_icl), "n_select": int(args.n_select), "n_eval": int(args.n_eval)},
        "provenance": {"pair": provenance, "alt_pair": alt_provenance},
    }
    _write_json(out_root / "component_localization_panel_config.json", config)

    pad_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", 0)

    def generation_metrics(gold_text: str, pred_text: str) -> Dict[str, float]:
        gold = normalize_text(gold_text)
        pred = normalize_text(pred_text)
        cont = continuation_akshara_cer(pred, gold)
        return {
            "exact_match": float(pred == gold),
            "akshara_cer": float(akshara_cer(pred, gold)),
            "script_compliance": float(script_compliance(pred, target_script)),
            "first_entry_correct": float(first_entry_correct(pred, gold)),
            "continuation_fidelity": float(cont) if np.isfinite(cont) else float("nan"),
        }

    def generate_with_hook(input_ids: torch.Tensor, *, component: Optional[str], patch_vector: Optional[torch.Tensor], layer: Optional[int], patch_pos: Optional[int]) -> str:
        handle = None
        try:
            if component == "mlp_output" and patch_vector is not None and layer is not None:
                handle = register_dense_mlp_output_patch_hook(model, int(layer), patch_vector, patch_position=patch_pos)
            elif component == "attention_output" and patch_vector is not None and layer is not None:
                handle = register_attention_output_replace_hook(model, int(layer), patch_vector, patch_position=patch_pos)
            attention_mask = torch.ones_like(input_ids)
            with torch.inference_mode():
                out = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=int(args.max_new_tokens),
                    do_sample=False,
                    use_cache=False,
                    pad_token_id=int(pad_id),
                )
            new_tokens = out[0, input_ids.shape[1] :]
            return normalize_text(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
        finally:
            if handle is not None:
                handle.remove()

    cached_items: List[Dict[str, Any]] = []
    for idx, word in enumerate(eval_samples):
        zs_prompt = build_task_prompt(
            word["ood"],
            None,
            input_script_name=input_script_name,
            source_language=source_language,
            output_script_name=target_script,
            prompt_variant="canonical",
        )
        icl_prompt = build_task_prompt(
            word["ood"],
            icl_examples,
            input_script_name=input_script_name,
            source_language=source_language,
            output_script_name=target_script,
            prompt_variant="canonical",
        )
        wrong_lang_prompt = build_task_prompt(
            word["ood"],
            alt_icl_examples,
            input_script_name=input_script_name,
            source_language=source_language,
            output_script_name=target_script,
            prompt_variant="canonical",
        )

        zs_rendered = apply_chat_template(tokenizer, zs_prompt)
        icl_rendered = apply_chat_template(tokenizer, icl_prompt)
        wrong_lang_rendered = apply_chat_template(tokenizer, wrong_lang_prompt)
        zs_inputs = tokenizer(zs_rendered, return_tensors="pt").to(args.device)
        icl_inputs = tokenizer(icl_rendered, return_tensors="pt").to(args.device)
        wrong_lang_inputs = tokenizer(wrong_lang_rendered, return_tensors="pt").to(args.device)

        query_ids = tokenizer.encode(word["ood"], add_special_tokens=False)
        span_zs = _find_last_subsequence(zs_inputs.input_ids[0].detach().cpu().tolist(), [int(x) for x in query_ids])
        span_icl = _find_last_subsequence(icl_inputs.input_ids[0].detach().cpu().tolist(), [int(x) for x in query_ids])
        span_wrong = _find_last_subsequence(wrong_lang_inputs.input_ids[0].detach().cpu().tolist(), [int(x) for x in query_ids])
        if span_zs is None or span_icl is None or span_wrong is None:
            raise ValueError(
                f"Fail-closed query span localization failed for item={idx}: zs={span_zs is not None} icl={span_icl is not None} wrong={span_wrong is not None}"
            )

        patch_pos = int(span_zs[1] - 1)
        icl_pos = int(span_icl[1] - 1)
        wrong_pos = int(span_wrong[1] - 1)

        zs_pred = generate_with_hook(zs_inputs.input_ids, component=None, patch_vector=None, layer=None, patch_pos=None)
        icl_pred = generate_with_hook(icl_inputs.input_ids, component=None, patch_vector=None, layer=None, patch_pos=None)
        cached_items.append(
            {
                "index": int(idx),
                "word": word,
                "zs_ids": zs_inputs.input_ids,
                "icl_ids": icl_inputs.input_ids,
                "wrong_lang_ids": wrong_lang_inputs.input_ids,
                "patch_pos": patch_pos,
                "icl_pos": icl_pos,
                "wrong_pos": wrong_pos,
                "zs_gen": generation_metrics(word["hindi"], zs_pred),
                "icl_gen": generation_metrics(word["hindi"], icl_pred),
            }
        )

    def extract_component_vector(component: str, input_ids: torch.Tensor, layer: int, position: int) -> torch.Tensor:
        if component == "mlp_output":
            _, vec = _extract_mlp_io_at_position_from_input_ids(model=model, input_ids=input_ids, layer=layer, position=position)
            return vec.detach().float()
        if component == "attention_output":
            vec = _extract_attention_output_at_position_from_input_ids(model=model, input_ids=input_ids, layer=layer, position=position)
            return vec.detach().float()
        raise ValueError(f"Unknown component: {component}")

    summary_rows: List[Dict[str, Any]] = []
    item_rows: List[Dict[str, Any]] = []

    baseline_zs = {
        "component": "baseline",
        "condition": "zs",
        "layer": None,
        "exact_match": float(np.mean([item["zs_gen"]["exact_match"] for item in cached_items])),
        "akshara_cer": float(np.mean([item["zs_gen"]["akshara_cer"] for item in cached_items])),
        "first_entry_correct": float(np.mean([item["zs_gen"]["first_entry_correct"] for item in cached_items])),
        "script_compliance": float(np.mean([item["zs_gen"]["script_compliance"] for item in cached_items])),
        "continuation_fidelity": float(np.mean([item["zs_gen"]["continuation_fidelity"] for item in cached_items])),
        "n": len(cached_items),
    }
    baseline_icl = {
        "component": "baseline",
        "condition": "icl64",
        "layer": None,
        "exact_match": float(np.mean([item["icl_gen"]["exact_match"] for item in cached_items])),
        "akshara_cer": float(np.mean([item["icl_gen"]["akshara_cer"] for item in cached_items])),
        "first_entry_correct": float(np.mean([item["icl_gen"]["first_entry_correct"] for item in cached_items])),
        "script_compliance": float(np.mean([item["icl_gen"]["script_compliance"] for item in cached_items])),
        "continuation_fidelity": float(np.mean([item["icl_gen"]["continuation_fidelity"] for item in cached_items])),
        "n": len(cached_items),
    }
    summary_rows.extend([baseline_zs, baseline_icl])
    log(
        f"zs                 | EM={baseline_zs['exact_match']*100:>5.1f}% CER={baseline_zs['akshara_cer']:.3f} "
        f"1stEnt={baseline_zs['first_entry_correct']*100:>5.1f}% Comp={baseline_zs['script_compliance']*100:>5.1f}%"
    )
    log(
        f"icl64              | EM={baseline_icl['exact_match']*100:>5.1f}% CER={baseline_icl['akshara_cer']:.3f} "
        f"1stEnt={baseline_icl['first_entry_correct']*100:>5.1f}% Comp={baseline_icl['script_compliance']*100:>5.1f}%"
    )

    for component in ["mlp_output", "attention_output"]:
        for layer in layers:
            for condition in ["correct", "wrong_language", "shuffled"]:
                metrics: List[Dict[str, float]] = []
                for item in cached_items:
                    if condition == "correct":
                        donor_vec = extract_component_vector(component, item["icl_ids"], int(layer), int(item["icl_pos"]))
                    elif condition == "wrong_language":
                        donor_vec = extract_component_vector(component, item["wrong_lang_ids"], int(layer), int(item["wrong_pos"]))
                    else:
                        donor_vec = extract_component_vector(component, item["icl_ids"], int(layer), int(item["icl_pos"]))
                        donor_vec = _deterministic_shuffle_vector(
                            donor_vec,
                            seed=int(args.seed),
                            layer=int(layer),
                            word_key=f"{component}::{item['word']['english']}",
                        ).float()

                    pred = generate_with_hook(
                        item["zs_ids"],
                        component=component,
                        patch_vector=donor_vec,
                        layer=int(layer),
                        patch_pos=int(item["patch_pos"]),
                    )
                    gen = generation_metrics(item["word"]["hindi"], pred)
                    metrics.append(gen)
                    item_rows.append(
                        {
                            "component": component,
                            "layer": int(layer),
                            "condition": condition,
                            "word_english": str(item["word"]["english"]),
                            "word_target": str(item["word"]["hindi"]),
                            "word_source_romanized": str(item["word"]["ood"]),
                            **{k: float(v) for k, v in gen.items()},
                        }
                    )

                row = {
                    "component": component,
                    "layer": int(layer),
                    "condition": condition,
                    "exact_match": float(np.mean([m["exact_match"] for m in metrics])),
                    "akshara_cer": float(np.mean([m["akshara_cer"] for m in metrics])),
                    "first_entry_correct": float(np.mean([m["first_entry_correct"] for m in metrics])),
                    "script_compliance": float(np.mean([m["script_compliance"] for m in metrics])),
                    "continuation_fidelity": float(np.mean([m["continuation_fidelity"] for m in metrics])),
                    "n": len(metrics),
                }
                summary_rows.append(row)
                log(
                    f"{component[:9]:<9} L{int(layer):2d} {condition:<14} | EM={row['exact_match']*100:>5.1f}% "
                    f"CER={row['akshara_cer']:.3f} 1stEnt={row['first_entry_correct']*100:>5.1f}% Comp={row['script_compliance']*100:>5.1f}%"
                )

    _write_json(out_root / "component_localization_panel_summary.json", _json_safe(summary_rows))
    _write_json(out_root / "component_localization_panel_items.json", _json_safe(item_rows))
    log(f"Saved summary to {out_root / 'component_localization_panel_summary.json'}")
    log(f"Saved items to {out_root / 'component_localization_panel_items.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
