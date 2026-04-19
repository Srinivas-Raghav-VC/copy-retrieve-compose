#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out


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


def _write_rows(base: Path, rows: List[Dict[str, Any]]) -> None:
    _write_json(base.with_suffix(".json"), rows)
    keys: List[str] = sorted({str(k) for row in rows for k in row.keys()}) if rows else []
    with base.with_suffix(".csv").open("w", encoding="utf-8", newline="") as f:
        if not keys:
            f.write("")
            return
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def _parse_csv(raw: str) -> List[str]:
    return [x.strip() for x in str(raw or "").split(",") if x.strip()]


def _git_commit_hash() -> str:
    env_hash = str(os.environ.get("PROJECT_GIT_COMMIT_HASH", "") or "").strip()
    if env_hash:
        return env_hash
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(PROJECT_ROOT),
                text=True,
                stderr=subprocess.DEVNULL,
            )
            .strip()
        )
    except Exception:
        return ""


def _load_words(
    pair_id: str,
    *,
    external_only: bool,
    require_external_sources: bool,
    min_pool_size: int,
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    data_path = PROJECT_ROOT / f"data/transliteration/{pair_id}.jsonl"
    meta_path = PROJECT_ROOT / f"data/transliteration/{pair_id}.jsonl.meta.json"
    if not data_path.exists():
        raise RuntimeError(f"Missing workshop data file: {data_path}")
    if not meta_path.exists():
        raise RuntimeError(f"Missing workshop metadata file: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    words: List[Dict[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            row = json.loads(raw)
            target = str(row["target"]).strip()
            source = str(row["source"]).strip()
            pair_key = (target, source)
            if not target or not source or pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            words.append(
                {
                    "english": str(row.get("english", source)).strip(),
                    "hindi": target,
                    "ood": source,
                }
            )

    total = len(words)
    source_names = [str(meta.get("dataset", {}).get("repo_id", "")).strip() or "external_jsonl"]
    external_sources = [n for n in source_names if n and n != "config_multiscript"]
    if bool(require_external_sources) and not external_sources:
        raise RuntimeError(
            f"Pair {pair_id!r} has no external sources (only builtin). "
            "Provide external data under data/transliteration/ or disable --require-external-sources."
        )
    if bool(external_only) and not external_sources:
        raise RuntimeError(f"Pair {pair_id!r} does not satisfy --external-only")
    if int(min_pool_size) > 0 and total < int(min_pool_size):
        raise RuntimeError(f"Pair {pair_id!r} pool too small: total={total} < {int(min_pool_size)}")

    provenance = {
        "pair_id": pair_id,
        "total_rows": total,
        "sources": source_names,
        "meta": meta,
    }
    return words, provenance


def _get_pair_prompt_metadata(pair_id: str) -> Dict[str, str]:
    meta_path = PROJECT_ROOT / f"data/transliteration/{pair_id}.jsonl.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    dataset = dict(meta.get("dataset", {}))
    return {
        "source_language": str(dataset.get("source_language", "Hindi")),
        "source_script": str(dataset.get("source_script", "Latin")),
        "target_script": str(dataset.get("target_script", "Devanagari")),
    }


def _deterministic_shuffle_vector(vec: torch.Tensor, *, seed: int, layer: int, word_key: str) -> torch.Tensor:
    msg = f"dense_shuffle::{seed}::{layer}::{word_key}".encode("utf-8")
    seed32 = int.from_bytes(hashlib.sha256(msg).digest()[:4], "little", signed=False)
    rng = np.random.default_rng(seed32)
    perm = rng.permutation(int(vec.numel()))
    perm_t = torch.tensor(perm, device=vec.device, dtype=torch.long)
    return torch.index_select(vec, 0, perm_t)


def _prompt_fingerprint(*, raw_prompt: str, rendered_prompt: str) -> Dict[str, Any]:
    def _sha(text: str) -> str:
        return hashlib.sha256(str(text).encode("utf-8")).hexdigest()

    return {
        "raw_length_chars": len(str(raw_prompt)),
        "rendered_length_chars": len(str(rendered_prompt)),
        "raw_sha256": _sha(raw_prompt),
        "rendered_sha256": _sha(rendered_prompt),
    }


def _runtime_identity(*, model_key: str, hf_id: str, tokenizer: Any, model: Any) -> Dict[str, Any]:
    tokenizer_name = str(getattr(tokenizer, "name_or_path", "") or "")
    tokenizer_revision = ""
    if hasattr(tokenizer, "init_kwargs") and isinstance(tokenizer.init_kwargs, dict):
        tokenizer_revision = str(tokenizer.init_kwargs.get("revision", "") or "")
    model_name = ""
    model_revision = ""
    config = getattr(model, "config", None)
    if config is not None:
        model_name = str(getattr(config, "_name_or_path", "") or "")
        model_revision = str(getattr(config, "_commit_hash", "") or "")
    return {
        "model_key": str(model_key),
        "hf_id": str(hf_id),
        "tokenizer_name_or_path": tokenizer_name,
        "tokenizer_revision": tokenizer_revision,
        "model_name_or_path": model_name,
        "model_revision": model_revision,
        "torch_version": str(getattr(torch, "__version__", "") or ""),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Dense whole-MLP-output control sweep at the selected site family.")
    ap.add_argument("--model", type=str, default="4b", choices=["1b", "4b"])
    ap.add_argument("--pair", type=str, default="aksharantar_hin_latin")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--n-icl", type=int, default=64)
    ap.add_argument("--n-select", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--external-only", action="store_true")
    ap.add_argument("--require-external-sources", action="store_true")
    ap.add_argument("--min-pool-size", type=int, default=500)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--layer-start", type=int, default=0)
    ap.add_argument("--layer-end", type=int, default=-1, help="Inclusive end; -1 means final layer")
    ap.add_argument("--out", type=str, default="")
    return ap.parse_args()


def main() -> int:
    from core import (
        _extract_mlp_io_at_position_from_input_ids,
        _find_last_subsequence,
        _teacher_forced_metrics_from_input_ids,
        apply_chat_template,
        build_corrupted_icl_prompt,
        build_task_prompt,
        get_model_config,
        get_model_layers,
        load_model,
        register_dense_mlp_output_patch_hook,
        save_json,
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

    args = parse_args()
    set_all_seeds(int(args.seed))

    words, provenance = _load_words(
        args.pair,
        external_only=bool(args.external_only),
        require_external_sources=bool(args.require_external_sources),
        min_pool_size=int(args.min_pool_size),
    )
    prompt_meta = _get_pair_prompt_metadata(args.pair)
    source_language = str(prompt_meta["source_language"])
    input_script_name = str(prompt_meta["source_script"])
    target_script = str(prompt_meta["target_script"])

    icl_examples, _, eval_samples = split_data_three_way(
        words,
        n_icl=int(args.n_icl),
        n_select=int(args.n_select),
        n_eval=int(args.n_eval),
        seed=int(args.seed),
    )
    mismatch_samples = eval_samples[1:] + eval_samples[:1]

    cfg = get_model_config(args.model)
    model, tokenizer = load_model(args.model, device=str(args.device))
    layers = get_model_layers(model)
    n_layers = int(len(layers))
    layer_start = max(0, int(args.layer_start))
    layer_end = n_layers - 1 if int(args.layer_end) < 0 else min(int(args.layer_end), n_layers - 1)
    sweep_layers = list(range(layer_start, layer_end + 1))

    out_root = (
        Path(args.out).resolve()
        if str(args.out).strip()
        else PROJECT_ROOT / "paper2_fidelity_calibrated" / "results" / "dense_mlp_sweep" / args.pair / args.model
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
    example_rendered = apply_chat_template(tokenizer, example_prompt)

    config_payload = {
        "experiment": "dense_mlp_out_truth_serum",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": args.model,
        "pair": args.pair,
        "seed": int(args.seed),
        "runtime_identity": _runtime_identity(model_key=args.model, hf_id=cfg.hf_id, tokenizer=tokenizer, model=model),
        "git_commit_hash": _git_commit_hash(),
        "provenance": provenance,
        "prompt_fingerprint": _prompt_fingerprint(raw_prompt=example_prompt, rendered_prompt=example_rendered),
        "frozen_prompt_variant": "canonical",
        "claim_level": "intervention_only",
        "site_family": "mlp_out_dense",
        "locus": "source_last_subtoken",
        "source_condition": "icl64",
        "base_condition": "explicit_zs",
        "controls": ["corrupt_dense", "mismatch_dense", "shuffled_vector_dense"],
        "layers": sweep_layers,
        "split_sizes": {
            "n_icl": int(args.n_icl),
            "n_select": int(args.n_select),
            "n_eval": int(args.n_eval),
        },
    }
    _write_json(out_root / "dense_layer_sweep_config.json", config_payload)

    pad_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", 0)

    def generate_with_hook(input_ids: torch.Tensor, patch_layer: Optional[int], patch_vec: Optional[torch.Tensor], patch_pos: Optional[int]) -> str:
        handle = None
        try:
            if patch_layer is not None and patch_vec is not None:
                handle = register_dense_mlp_output_patch_hook(model, patch_layer, patch_vec, patch_position=patch_pos)
            attention_mask = torch.ones_like(input_ids)
            with torch.inference_mode():
                out = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=int(args.max_new_tokens),
                    do_sample=False,
                    pad_token_id=int(pad_id),
                )
            new_tokens = out[0, input_ids.shape[1] :]
            return normalize_text(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
        finally:
            if handle is not None:
                handle.remove()

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

    item_rows: List[Dict[str, Any]] = []
    layer_rows: List[Dict[str, Any]] = []

    cached_items: List[Dict[str, Any]] = []
    for i, word in enumerate(eval_samples):
        mismatch_word = mismatch_samples[i]
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
        corrupt_prompt = build_corrupted_icl_prompt(
            word["ood"],
            icl_examples,
            input_script_name=input_script_name,
            source_language=source_language,
            output_script_name=target_script,
            seed=int(args.seed),
        )
        mismatch_icl_prompt = build_task_prompt(
            mismatch_word["ood"],
            icl_examples,
            input_script_name=input_script_name,
            source_language=source_language,
            output_script_name=target_script,
            prompt_variant="canonical",
        )
        zs_rendered = apply_chat_template(tokenizer, zs_prompt)
        icl_rendered = apply_chat_template(tokenizer, icl_prompt)
        corrupt_rendered = apply_chat_template(tokenizer, corrupt_prompt)
        mismatch_rendered = apply_chat_template(tokenizer, mismatch_icl_prompt)
        zs_inputs = tokenizer(zs_rendered, return_tensors="pt").to(args.device)
        icl_inputs = tokenizer(icl_rendered, return_tensors="pt").to(args.device)
        corrupt_inputs = tokenizer(corrupt_rendered, return_tensors="pt").to(args.device)
        mismatch_inputs = tokenizer(mismatch_rendered, return_tensors="pt").to(args.device)
        query_ids = tokenizer.encode(word["ood"], add_special_tokens=False)
        span_zs = _find_last_subsequence(zs_inputs.input_ids[0].detach().cpu().tolist(), [int(x) for x in query_ids])
        span_icl = _find_last_subsequence(icl_inputs.input_ids[0].detach().cpu().tolist(), [int(x) for x in query_ids])
        span_corrupt = _find_last_subsequence(corrupt_inputs.input_ids[0].detach().cpu().tolist(), [int(x) for x in query_ids])
        mismatch_ids = tokenizer.encode(mismatch_word["ood"], add_special_tokens=False)
        span_mismatch = _find_last_subsequence(
            mismatch_inputs.input_ids[0].detach().cpu().tolist(),
            [int(x) for x in mismatch_ids],
        )
        if span_zs is None or span_icl is None or span_corrupt is None or span_mismatch is None:
            raise ValueError(
                f"Fail-closed query span localization failed for {args.pair} item={i}: "
                f"zs={span_zs is not None} icl={span_icl is not None} corrupt={span_corrupt is not None} mismatch={span_mismatch is not None}"
            )
        patch_pos = int(span_zs[1] - 1)
        icl_pos = int(span_icl[1] - 1)
        corrupt_pos = int(span_corrupt[1] - 1)
        mismatch_pos = int(span_mismatch[1] - 1)
        target_ids = tokenizer.encode(str(word["hindi"]), add_special_tokens=False)
        target_id = int(target_ids[0]) if target_ids else -1
        zs_tf = _teacher_forced_metrics_from_input_ids(
            model=model,
            input_ids=zs_inputs.input_ids,
            target_ids=target_ids,
            target_id=target_id,
            device=str(args.device),
            competitor_id=-1,
        )
        icl_tf = _teacher_forced_metrics_from_input_ids(
            model=model,
            input_ids=icl_inputs.input_ids,
            target_ids=target_ids,
            target_id=target_id,
            device=str(args.device),
            competitor_id=-1,
        )
        zs_pred = generate_with_hook(zs_inputs.input_ids, None, None, None)
        icl_pred = generate_with_hook(icl_inputs.input_ids, None, None, None)
        cached_items.append(
            {
                "index": i,
                "word": word,
                "mismatch_word": mismatch_word,
                "zs_prompt": zs_prompt,
                "icl_prompt": icl_prompt,
                "corrupt_prompt": corrupt_prompt,
                "mismatch_icl_prompt": mismatch_icl_prompt,
                "zs_ids": zs_inputs.input_ids,
                "icl_ids": icl_inputs.input_ids,
                "corrupt_ids": corrupt_inputs.input_ids,
                "mismatch_ids": mismatch_inputs.input_ids,
                "patch_pos": patch_pos,
                "icl_pos": icl_pos,
                "corrupt_pos": corrupt_pos,
                "mismatch_pos": mismatch_pos,
                "target_ids": target_ids,
                "target_id": target_id,
                "zs_tf": zs_tf,
                "icl_tf": icl_tf,
                "zs_gen": generation_metrics(word["hindi"], zs_pred),
                "icl_gen": generation_metrics(word["hindi"], icl_pred),
                "n_input_tokens_ood": int(len(query_ids)),
            }
        )

    for layer in sweep_layers:
        log(f"Dense sweep layer {layer}")
        layer_item_rows: List[Dict[str, Any]] = []
        for item in cached_items:
            mlp_in_icl, mlp_out_icl = _extract_mlp_io_at_position_from_input_ids(
                model=model,
                input_ids=item["icl_ids"],
                layer=layer,
                position=item["icl_pos"],
            )
            _, mlp_out_corrupt = _extract_mlp_io_at_position_from_input_ids(
                model=model,
                input_ids=item["corrupt_ids"],
                layer=layer,
                position=item["corrupt_pos"],
            )
            _, mlp_out_mismatch = _extract_mlp_io_at_position_from_input_ids(
                model=model,
                input_ids=item["mismatch_ids"],
                layer=layer,
                position=item["mismatch_pos"],
            )
            shuffled_vec = _deterministic_shuffle_vector(
                mlp_out_icl.detach().float(),
                seed=int(args.seed),
                layer=int(layer),
                word_key=str(item["word"]["english"]),
            )

            def tf_metrics_for(vec: torch.Tensor) -> Dict[str, float]:
                handle = register_dense_mlp_output_patch_hook(
                    model,
                    layer,
                    vec,
                    patch_position=item["patch_pos"],
                )
                try:
                    return _teacher_forced_metrics_from_input_ids(
                        model=model,
                        input_ids=item["zs_ids"],
                        target_ids=item["target_ids"],
                        target_id=item["target_id"],
                        device=str(args.device),
                        competitor_id=-1,
                    )
                finally:
                    handle.remove()

            patched_tf = tf_metrics_for(mlp_out_icl)
            corrupt_tf = tf_metrics_for(mlp_out_corrupt)
            mismatch_tf = tf_metrics_for(mlp_out_mismatch)
            shuffled_tf = tf_metrics_for(shuffled_vec)
            patched_pred = generate_with_hook(item["zs_ids"], layer, mlp_out_icl, item["patch_pos"])
            patched_gen = generation_metrics(item["word"]["hindi"], patched_pred)

            zs_tf = item["zs_tf"]
            icl_tf = item["icl_tf"]
            row = {
                "pair": args.pair,
                "seed": int(args.seed),
                "layer": int(layer),
                "word_english": str(item["word"]["english"]),
                "word_target": str(item["word"]["hindi"]),
                "word_source_romanized": str(item["word"]["ood"]),
                "n_input_tokens_ood": int(item["n_input_tokens_ood"]),
                "pe_logit": _safe_float(patched_tf.get("first_logit")) - _safe_float(zs_tf.get("first_logit")),
                "pe_first": _safe_float(patched_tf.get("first_prob")) - _safe_float(zs_tf.get("first_prob")),
                "pe_corrupt": _safe_float(corrupt_tf.get("first_prob")) - _safe_float(zs_tf.get("first_prob")),
                "pe_mismatch": _safe_float(mismatch_tf.get("first_prob")) - _safe_float(zs_tf.get("first_prob")),
                "pe_shuffled_vector": _safe_float(shuffled_tf.get("first_prob")) - _safe_float(zs_tf.get("first_prob")),
                "exact_match_zs": item["zs_gen"]["exact_match"],
                "exact_match_icl": item["icl_gen"]["exact_match"],
                "exact_match_patched": patched_gen["exact_match"],
                "akshara_cer_zs": item["zs_gen"]["akshara_cer"],
                "akshara_cer_icl": item["icl_gen"]["akshara_cer"],
                "akshara_cer_patched": patched_gen["akshara_cer"],
                "script_compliance_zs": item["zs_gen"]["script_compliance"],
                "script_compliance_icl": item["icl_gen"]["script_compliance"],
                "script_compliance_patched": patched_gen["script_compliance"],
                "first_entry_correct_zs": item["zs_gen"]["first_entry_correct"],
                "first_entry_correct_icl": item["icl_gen"]["first_entry_correct"],
                "first_entry_correct_patched": patched_gen["first_entry_correct"],
                "continuation_fidelity_zs": item["zs_gen"]["continuation_fidelity"],
                "continuation_fidelity_icl": item["icl_gen"]["continuation_fidelity"],
                "continuation_fidelity_patched": patched_gen["continuation_fidelity"],
                "nll_per_token_zs": -_safe_float(zs_tf.get("joint_logprob")) / max(1, len(item["target_ids"])),
                "nll_per_token_icl": -_safe_float(icl_tf.get("joint_logprob")) / max(1, len(item["target_ids"])),
                "nll_per_token_patched": -_safe_float(patched_tf.get("joint_logprob")) / max(1, len(item["target_ids"])),
                "nll_pos1_zs": _safe_float(zs_tf.get("target_pos1_nll")),
                "nll_pos2_zs": _safe_float(zs_tf.get("target_pos2_nll")),
                "nll_pos3_zs": _safe_float(zs_tf.get("target_pos3_nll")),
                "nll_pos1_patched": _safe_float(patched_tf.get("target_pos1_nll")),
                "nll_pos2_patched": _safe_float(patched_tf.get("target_pos2_nll")),
                "nll_pos3_patched": _safe_float(patched_tf.get("target_pos3_nll")),
            }
            layer_item_rows.append(row)
            item_rows.append(row)

        pe_vals = np.array([_safe_float(r["pe_first"]) for r in layer_item_rows], dtype=np.float64)
        pe_corrupt = np.array([_safe_float(r["pe_corrupt"]) for r in layer_item_rows], dtype=np.float64)
        pe_mismatch = np.array([_safe_float(r["pe_mismatch"]) for r in layer_item_rows], dtype=np.float64)
        pe_shuffled = np.array([_safe_float(r["pe_shuffled_vector"]) for r in layer_item_rows], dtype=np.float64)
        exact_patch = np.array([_safe_float(r["exact_match_patched"]) for r in layer_item_rows], dtype=np.float64)
        exact_zs = np.array([_safe_float(r["exact_match_zs"]) for r in layer_item_rows], dtype=np.float64)
        entry_patch = np.array([_safe_float(r["first_entry_correct_patched"]) for r in layer_item_rows], dtype=np.float64)
        entry_zs = np.array([_safe_float(r["first_entry_correct_zs"]) for r in layer_item_rows], dtype=np.float64)
        cont_patch = np.array([_safe_float(r["continuation_fidelity_patched"]) for r in layer_item_rows], dtype=np.float64)
        cont_zs = np.array([_safe_float(r["continuation_fidelity_zs"]) for r in layer_item_rows], dtype=np.float64)
        nll1_patch = np.array([_safe_float(r["nll_pos1_patched"]) for r in layer_item_rows], dtype=np.float64)
        nll1_zs = np.array([_safe_float(r["nll_pos1_zs"]) for r in layer_item_rows], dtype=np.float64)
        nll2_patch = np.array([_safe_float(r["nll_pos2_patched"]) for r in layer_item_rows], dtype=np.float64)
        nll2_zs = np.array([_safe_float(r["nll_pos2_zs"]) for r in layer_item_rows], dtype=np.float64)
        nll3_patch = np.array([_safe_float(r["nll_pos3_patched"]) for r in layer_item_rows], dtype=np.float64)
        nll3_zs = np.array([_safe_float(r["nll_pos3_zs"]) for r in layer_item_rows], dtype=np.float64)
        finite_pe = pe_vals[np.isfinite(pe_vals)]
        if finite_pe.size:
            boot = finite_pe
            rng = np.random.default_rng(int(args.seed) + int(layer))
            boots = []
            for _ in range(2000):
                samp = rng.choice(boot, size=boot.size, replace=True)
                boots.append(float(np.mean(samp)))
            ci_low, ci_high = float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
        else:
            ci_low = ci_high = float("nan")

        layer_rows.append(
            {
                "pair": args.pair,
                "seed": int(args.seed),
                "layer": int(layer),
                "mean_pe": float(np.nanmean(pe_vals)),
                "ci_pe_low": ci_low,
                "ci_pe_high": ci_high,
                "mean_pe_corrupt": float(np.nanmean(pe_corrupt)),
                "mean_pe_mismatch": float(np.nanmean(pe_mismatch)),
                "mean_pe_shuffled_vector": float(np.nanmean(pe_shuffled)),
                "specificity_margin_vs_controls": float(np.nanmean(pe_vals) - max(np.nanmean(pe_corrupt), np.nanmean(pe_mismatch), np.nanmean(pe_shuffled))),
                "exact_match_zs": float(np.nanmean(exact_zs)),
                "exact_match_patched": float(np.nanmean(exact_patch)),
                "exact_match_delta": float(np.nanmean(exact_patch) - np.nanmean(exact_zs)),
                "first_entry_correct_zs": float(np.nanmean(entry_zs)),
                "first_entry_correct_patched": float(np.nanmean(entry_patch)),
                "first_entry_delta": float(np.nanmean(entry_patch) - np.nanmean(entry_zs)),
                "continuation_fidelity_zs": float(np.nanmean(cont_zs)),
                "continuation_fidelity_patched": float(np.nanmean(cont_patch)),
                "continuation_delta": float(np.nanmean(cont_zs) - np.nanmean(cont_patch)),
                "mean_nll_pos1_delta": float(np.nanmean(nll1_zs) - np.nanmean(nll1_patch)),
                "mean_nll_pos2_delta": float(np.nanmean(nll2_zs) - np.nanmean(nll2_patch)),
                "mean_nll_pos3_delta": float(np.nanmean(nll3_zs) - np.nanmean(nll3_patch)),
            }
        )

    _write_rows(out_root / "dense_layer_sweep_item_level", item_rows)
    _write_rows(out_root / "dense_layer_sweep_results", layer_rows)

    stagea_path = PROJECT_ROOT / "paper2_fidelity_calibrated" / "results" / args.pair / args.model / f"paper2_fidelity_calibrated_{args.model}.json"
    seeded_layer = None
    if stagea_path.exists():
        try:
            payload = json.loads(stagea_path.read_text(encoding="utf-8"))
            seeded_layer = int(payload["seeds"][str(args.seed)]["best"]["layer"])
        except Exception:
            seeded_layer = None

    best_row = None
    if layer_rows:
        def _score(row: Dict[str, Any]) -> float:
            pe = _safe_float(row.get("mean_pe"))
            ctrl = _safe_float(row.get("specificity_margin_vs_controls"))
            if not np.isfinite(pe):
                return -1e9
            if not np.isfinite(ctrl):
                ctrl = -1e9
            return min(pe, ctrl)
        best_row = max(layer_rows, key=_score)

    decision: Dict[str, Any]
    if best_row is None:
        decision = {
            "status": "incomplete",
            "proceed_sparse_retry": "no",
            "next_step": "bounded_attention_analysis",
            "rationale": "Dense sweep produced no valid layer rows.",
        }
    else:
        best_layer = int(best_row["layer"])
        mean_pe = _safe_float(best_row["mean_pe"])
        control_margin = _safe_float(best_row["specificity_margin_vs_controls"])
        if not np.isfinite(mean_pe) or mean_pe <= 0.0 or not np.isfinite(control_margin) or control_margin <= 0.0:
            decision = {
                "status": "complete",
                "proceed_sparse_retry": "no",
                "next_step": "bounded_attention_analysis",
                "rationale": "Dense whole-MLP patching at all layers failed to produce task-specific rescue stronger than controls, falsifying the hypothesis that the CFOM rescue is localized in MLP sublayers at this site family and locus.",
                "best_layer": best_layer,
                "best_row": best_row,
                "seeded_layer": seeded_layer,
            }
        elif seeded_layer is not None and best_layer != int(seeded_layer):
            decision = {
                "status": "complete",
                "proceed_sparse_retry": "yes",
                "retry_scope": "peak_layer_only",
                "next_step": "rerun_sparse_at_peak_layer",
                "rationale": "Dense sweep found a task-specific peak at a different layer than Stage A seeded, indicating seeded localization failure rather than an absence of an MLP-side signal.",
                "best_layer": best_layer,
                "best_row": best_row,
                "seeded_layer": seeded_layer,
            }
        else:
            decision = {
                "status": "complete",
                "proceed_sparse_retry": "yes",
                "retry_scope": "seeded_layer_only",
                "next_step": "refine_sparse_selection_at_peak_layer",
                "rationale": "Dense sweep produced a task-specific peak at the seeded layer/site family, implying the sparse selector or sparsity bottleneck is the main failure mode rather than the MLP site itself.",
                "best_layer": best_layer,
                "best_row": best_row,
                "seeded_layer": seeded_layer,
            }

    _write_json(out_root / "dense_layer_sweep_decision_note.json", decision)
    save_json(str(out_root / "dense_layer_sweep_decision_note_legacy.json"), decision)
    log(f"Dense sweep complete. Decision: {decision.get('next_step')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
