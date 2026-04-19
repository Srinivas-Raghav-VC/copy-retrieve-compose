#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import (  # noqa: E402
    _extract_mlp_io_at_position_from_input_ids,
    apply_chat_template,
    build_task_prompt,
    get_model_layers,
    load_model,
    register_dense_mlp_output_patch_hook,
    set_all_seeds,
)
from rescue_research.prompts.templates import confirmatory_user_prompt  # noqa: E402

SPLIT_SNAPSHOT_DIR = PROJECT_ROOT / "paper2_fidelity_calibrated" / "split_snapshots"
DEFAULT_OUT = PROJECT_ROOT / "paper2_fidelity_calibrated" / "results" / "1b_final"

ATTN_SUBSETS: Dict[str, List[int]] = {
    "single_L05": [5],
    "single_L11": [11],
    "single_L14": [14],
    "single_L17": [17],
    "single_L23": [23],
    "first_two_global": [5, 11],
    "last_two_global": [17, 23],
    "all_global_attn": [5, 11, 17, 23],
    "all_local_attn": [i for i in range(26) if i not in [5, 11, 17, 23]],
    "all_attn": list(range(26)),
}

PAIR_META: Dict[str, Dict[str, str]] = {
    "aksharantar_hin_latin": {
        "source_language": "Hindi",
        "input_script_name": "Latin",
        "output_script_name": "Devanagari",
        "lang_slug": "hindi",
    },
    "aksharantar_tel_latin": {
        "source_language": "Telugu",
        "input_script_name": "Latin",
        "output_script_name": "Telugu",
        "lang_slug": "telugu",
    },
}

AKSHARANTAR_FALLBACKS: Dict[str, Dict[str, str]] = {
    "aksharantar_hin_latin": {
        "hf_config": "hin",
        "source_language": "Hindi",
        "output_script_name": "Devanagari",
    },
    "aksharantar_tel_latin": {
        "hf_config": "tel",
        "source_language": "Telugu",
        "output_script_name": "Telugu",
    },
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _json_safe(v: Any) -> Any:
    if isinstance(v, (str, int, bool)) or v is None:
        return v
    if isinstance(v, float):
        return v if np.isfinite(v) else None
    if isinstance(v, np.floating):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, dict):
        return {str(k): _json_safe(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if isinstance(v, np.ndarray):
        return _json_safe(v.tolist())
    return str(v)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def bootstrap_ci(vals: List[float], n_boot: int = 2000, ci: float = 0.95) -> Dict[str, Any]:
    arr = np.array(vals, dtype=float)
    if len(arr) == 0:
        return {"mean": None, "ci_lo": None, "ci_hi": None, "std": None, "n": 0}
    if len(arr) < 3:
        return {
            "mean": float(np.mean(arr)),
            "ci_lo": float(np.min(arr)),
            "ci_hi": float(np.max(arr)),
            "std": float(np.std(arr)),
            "n": int(len(arr)),
        }
    boots = np.array([np.mean(np.random.choice(arr, size=len(arr), replace=True)) for _ in range(int(n_boot))])
    lo = float(np.percentile(boots, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boots, (1 + ci) / 2 * 100))
    return {
        "mean": float(np.mean(arr)),
        "ci_lo": lo,
        "ci_hi": hi,
        "std": float(np.std(arr)),
        "n": int(len(arr)),
    }


def _snapshot_path(pair_id: str, *, seed: int, n_icl: int, n_select: int, n_eval: int) -> Path:
    return SPLIT_SNAPSHOT_DIR / f"{pair_id}_split_seed{int(seed)}_nicl{int(n_icl)}_nselect{int(n_select)}_neval{int(n_eval)}.json"


def load_pair_bundle(
    pair_id: str,
    *,
    seed: int,
    n_icl: int,
    n_select: int,
    n_eval: int,
    prefer_snapshot: bool = True,
) -> Dict[str, Any]:
    if prefer_snapshot:
        snap = _snapshot_path(pair_id, seed=seed, n_icl=n_icl, n_select=n_select, n_eval=n_eval)
        candidate_snaps: List[Path] = []
        if snap.exists():
            candidate_snaps.append(snap)
        else:
            pattern = f"{pair_id}_split_seed{int(seed)}_nicl{int(n_icl)}_nselect{int(n_select)}_neval*.json"
            candidate_snaps.extend(sorted(SPLIT_SNAPSHOT_DIR.glob(pattern)))
        for candidate in candidate_snaps:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            eval_rows = list(payload.get("eval_rows", []))
            if len(eval_rows) < int(n_eval):
                continue
            return {
                "pair": str(payload["pair"]),
                "provenance": {"loader": "snapshot", "path": str(candidate)},
                "source_language": str(payload["source_language"]),
                "input_script_name": str(payload["input_script_name"]),
                "output_script_name": str(payload["output_script_name"]),
                "icl_examples": list(payload["icl_examples"][: int(n_icl)]),
                "select_rows": [],
                "eval_rows": eval_rows[: int(n_eval)],
            }

    try:
        from paper2_fidelity_calibrated.phase1_common import load_pair_split

        pb = load_pair_split(
            pair_id,
            seed=seed,
            n_icl=n_icl,
            n_select=n_select,
            n_eval=n_eval,
            external_only=True,
            require_external_sources=True,
            min_pool_size=max(500, n_icl + n_select + n_eval),
        )
        return pb
    except Exception:
        pass

    data_path = PROJECT_ROOT / "data" / "transliteration" / f"{pair_id}.jsonl"
    meta_path = data_path.with_suffix(".jsonl.meta.json")
    if data_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        ds = meta.get("dataset", {})
        words: List[Dict[str, str]] = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line.strip())
                src = str(row.get("source", row.get("english word", ""))).strip()
                tgt = str(row.get("target", row.get("native word", ""))).strip()
                if src and tgt:
                    words.append({"ood": src, "hindi": tgt, "english": src})
        rng = random.Random(seed)
        rng.shuffle(words)
        total = n_icl + n_select + n_eval
        if len(words) < total:
            raise RuntimeError(f"Not enough rows for {pair_id}: {len(words)} < {total}")
        return {
            "pair": pair_id,
            "provenance": {"loader": "jsonl_fallback", "path": str(data_path)},
            "source_language": str(ds.get("source_language", PAIR_META[pair_id]["source_language"])),
            "input_script_name": str(ds.get("source_script", PAIR_META[pair_id]["input_script_name"])),
            "output_script_name": str(ds.get("target_script", PAIR_META[pair_id]["output_script_name"])),
            "icl_examples": words[:n_icl],
            "select_rows": words[n_icl : n_icl + n_select],
            "eval_rows": words[n_icl + n_select : n_icl + n_select + n_eval],
        }

    if pair_id in AKSHARANTAR_FALLBACKS:
        from datasets import load_dataset
        from core import split_data_three_way

        cfg = AKSHARANTAR_FALLBACKS[pair_id]
        ds = load_dataset("ai4bharat/Aksharantar", split="train")
        words: List[Dict[str, str]] = []
        for row in ds:
            if str(row.get("lang", "")).strip().lower() not in {cfg["hf_config"], pair_id.split("_")[1]}:
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
            "pair": pair_id,
            "provenance": {"loader": "hf_direct_aksharantar", "hf_config": cfg["hf_config"]},
            "source_language": cfg["source_language"],
            "input_script_name": "Latin",
            "output_script_name": cfg["output_script_name"],
            "icl_examples": icl_examples,
            "select_rows": select_rows,
            "eval_rows": eval_rows,
        }

    raise FileNotFoundError(f"Could not load data for {pair_id}: no snapshot, external loader, JSONL, or HF fallback available")


def _first_token_prob_rank(logits: torch.Tensor, gold_id: int) -> Tuple[float, int]:
    probs = torch.softmax(logits.float(), dim=-1)
    prob = float(probs[int(gold_id)].item())
    rank = int((torch.argsort(logits.float(), descending=True) == int(gold_id)).nonzero(as_tuple=True)[0].item()) + 1
    return prob, rank


def _capture_attention_o_proj_inputs(
    model,
    input_ids: torch.Tensor,
    layers_to_capture: List[int],
    position: int,
) -> Dict[int, torch.Tensor]:
    layers = get_model_layers(model)
    captured: Dict[int, torch.Tensor] = {}
    handles = []

    def mk_hook(layer_idx: int):
        def hook(module, args, kwargs):
            y = args[0]
            if torch.is_tensor(y):
                seq_len = int(y.shape[1])
                pos = int(max(0, min(int(position), seq_len - 1)))
                captured[layer_idx] = y[0, pos, :].detach().clone()
            return args, kwargs
        return hook

    for li in layers_to_capture:
        attn_module = getattr(layers[li], "self_attn", None) or getattr(layers[li], "self_attention", None)
        if attn_module is None or not hasattr(attn_module, "o_proj"):
            raise AttributeError(f"Layer {li} has no self-attention o_proj module")
        handles.append(attn_module.o_proj.register_forward_pre_hook(mk_hook(li), with_kwargs=True))

    with torch.inference_mode():
        model(input_ids=input_ids, use_cache=False)

    for h in handles:
        h.remove()
    return captured


def _register_attention_o_proj_replace_hook(
    model,
    layer: int,
    patch_vector: torch.Tensor,
    *,
    patch_position: int,
):
    layers = get_model_layers(model)
    attn_module = getattr(layers[layer], "self_attn", None) or getattr(layers[layer], "self_attention", None)
    if attn_module is None or not hasattr(attn_module, "o_proj"):
        raise AttributeError(f"Layer {layer} has no self-attention o_proj module")
    patch_vector = patch_vector.detach()

    def hook(module, args, kwargs):
        patched = args[0].clone()
        seq_len = int(patched.shape[1])
        pos = int(max(0, min(int(patch_position), seq_len - 1)))
        patched[0, pos, :] = patch_vector.to(device=patched.device, dtype=patched.dtype)
        return (patched,) + args[1:], kwargs

    return attn_module.o_proj.register_forward_pre_hook(hook, with_kwargs=True)


def _capture_mlp_outputs(
    model,
    input_ids: torch.Tensor,
    layers_to_capture: List[int],
    position: int,
) -> Dict[int, torch.Tensor]:
    captured: Dict[int, torch.Tensor] = {}
    layers = get_model_layers(model)
    handles = []

    def mk_hook(layer_idx: int):
        def hook(module, inputs_tuple, output):
            y = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(y):
                seq_len = int(y.shape[1])
                pos = int(max(0, min(int(position), seq_len - 1)))
                captured[layer_idx] = y[0, pos, :].detach().clone()
        return hook

    for li in layers_to_capture:
        handles.append(layers[li].mlp.register_forward_hook(mk_hook(li)))

    with torch.inference_mode():
        model(input_ids=input_ids, use_cache=False)

    for h in handles:
        h.remove()
    return captured


def _run_head_attribution_pair(
    model,
    tokenizer,
    device: str,
    pair_id: str,
    *,
    seed: int,
    n_icl: int,
    n_select: int,
    n_eval: int,
) -> Dict[str, Any]:
    pb = load_pair_bundle(pair_id, seed=seed, n_icl=n_icl, n_select=n_select, n_eval=n_eval, prefer_snapshot=True)
    icl_examples = list(pb["icl_examples"])
    eval_rows = list(pb["eval_rows"][:n_eval])
    source_language = str(pb["source_language"])
    input_script_name = str(pb["input_script_name"])
    output_script_name = str(pb["output_script_name"])

    layers = get_model_layers(model)
    cfg = getattr(model.config, "text_config", model.config)
    n_layers = int(cfg.num_hidden_layers)
    n_heads = int(cfg.num_attention_heads)
    head_dim = int(cfg.head_dim)
    layer_types = [str(x) for x in getattr(cfg, "layer_types", [])]

    effects = torch.zeros(n_layers, n_heads)
    per_item_effects: List[torch.Tensor] = []
    n_valid = 0

    for widx, word in enumerate(eval_rows):
        query = str(word["ood"])
        target = str(word["hindi"])
        tids = tokenizer.encode(target, add_special_tokens=False)
        if not tids:
            continue
        gold = int(tids[0])

        icl_raw = build_task_prompt(
            query,
            icl_examples,
            input_script_name=input_script_name,
            source_language=source_language,
            output_script_name=output_script_name,
        )
        zs_raw = build_task_prompt(
            query,
            None,
            input_script_name=input_script_name,
            source_language=source_language,
            output_script_name=output_script_name,
        )
        ids_i = tokenizer(apply_chat_template(tokenizer, icl_raw), return_tensors="pt").to(device).input_ids
        ids_z = tokenizer(apply_chat_template(tokenizer, zs_raw), return_tensors="pt").to(device).input_ids
        pi = int(ids_i.shape[1] - 1)
        pz = int(ids_z.shape[1] - 1)

        captured: Dict[int, torch.Tensor] = {}

        def mk_cap(li: int):
            def hook(module, args, kwargs):
                captured[li] = args[0][0, pi, :].detach().clone().view(n_heads, head_dim)
                return args, kwargs
            return hook

        handles = [layers[i].self_attn.o_proj.register_forward_pre_hook(mk_cap(i), with_kwargs=True) for i in range(n_layers)]
        with torch.inference_mode():
            oi = model(input_ids=ids_i, use_cache=False)
        for h in handles:
            h.remove()
        icl_prob, _ = _first_token_prob_rank(oi.logits[0, pi], gold)

        with torch.inference_mode():
            oz = model(input_ids=ids_z, use_cache=False)
        zs_logits = oz.logits[0, pz]
        zs_prob, _ = _first_token_prob_rank(zs_logits, gold)
        total_gap = float(icl_prob - zs_prob)
        if total_gap <= 0:
            continue

        item_scores = torch.zeros(n_layers, n_heads)
        for li in range(n_layers):
            donor_heads = captured[li]
            for hi in range(n_heads):
                donor_cat = torch.zeros(n_heads * head_dim, device=device, dtype=donor_heads.dtype)
                start = hi * head_dim
                donor_cat[start : start + head_dim] = donor_heads[hi]

                def patch_head(module, args, kwargs, donor=donor_cat):
                    patched = args[0].clone()
                    patched[0, pz, :] = donor.to(device=patched.device, dtype=patched.dtype)
                    return (patched,) + args[1:], kwargs

                h = layers[li].self_attn.o_proj.register_forward_pre_hook(patch_head, with_kwargs=True)
                with torch.inference_mode():
                    po = model(input_ids=ids_z, use_cache=False)
                h.remove()
                patched_prob, _ = _first_token_prob_rank(po.logits[0, pz], gold)
                item_scores[li, hi] = float((patched_prob - zs_prob) / total_gap)

        effects += item_scores
        per_item_effects.append(item_scores.clone())
        n_valid += 1
        if (widx + 1) % 10 == 0:
            log(f"  head-attribution {pair_id} {widx + 1}/{len(eval_rows)}")

    if n_valid <= 0:
        raise RuntimeError(f"No valid head-attribution items for {pair_id}")

    mean_scores = effects / float(n_valid)
    per_item_stack = torch.stack(per_item_effects, dim=0)

    rows: List[Dict[str, Any]] = []
    for li in range(n_layers):
        for hi in range(n_heads):
            vals = per_item_stack[:, li, hi].cpu().numpy().tolist()
            rows.append(
                {
                    "layer": li,
                    "head": hi,
                    "type": layer_types[li] if li < len(layer_types) else "?",
                    "effect": float(mean_scores[li, hi].item()),
                    "effect_ci": bootstrap_ci(vals),
                }
            )
    rows.sort(key=lambda row: float(row["effect"]), reverse=True)
    return {
        "experiment": "head_attribution",
        "pair": pair_id,
        "seed": seed,
        "n_eval": n_valid,
        "layer_types": layer_types,
        "rows": rows,
        "top_heads": rows[:10],
    }


def _run_attention_only_pair(
    model,
    tokenizer,
    device: str,
    pair_id: str,
    *,
    seed: int,
    n_icl: int,
    n_select: int,
    n_eval: int,
    prefer_snapshot: bool = True,
) -> Dict[str, Any]:
    pb = load_pair_bundle(pair_id, seed=seed, n_icl=n_icl, n_select=n_select, n_eval=n_eval, prefer_snapshot=prefer_snapshot)
    icl_examples = list(pb["icl_examples"])
    eval_rows = list(pb["eval_rows"][:n_eval])
    source_language = str(pb["source_language"])
    input_script_name = str(pb["input_script_name"])
    output_script_name = str(pb["output_script_name"])

    layers = get_model_layers(model)
    out: Dict[str, Any] = {
        "experiment": "attention_only_contribution",
        "pair": pair_id,
        "seed": seed,
        "n_eval": len(eval_rows),
        "subsets": [],
    }

    for sname, slayers in ATTN_SUBSETS.items():
        pe_vals: List[float] = []
        rank_delta_vals: List[float] = []
        frac_vals: List[float] = []
        for word in eval_rows:
            query = str(word["ood"])
            target = str(word["hindi"])
            tids = tokenizer.encode(target, add_special_tokens=False)
            if not tids:
                continue
            gold = int(tids[0])

            icl_raw = build_task_prompt(query, icl_examples, input_script_name=input_script_name, source_language=source_language, output_script_name=output_script_name)
            zs_raw = build_task_prompt(query, None, input_script_name=input_script_name, source_language=source_language, output_script_name=output_script_name)
            ids_i = tokenizer(apply_chat_template(tokenizer, icl_raw), return_tensors="pt").to(device).input_ids
            ids_z = tokenizer(apply_chat_template(tokenizer, zs_raw), return_tensors="pt").to(device).input_ids
            pi = int(ids_i.shape[1] - 1)
            pz = int(ids_z.shape[1] - 1)

            donor_attn = _capture_attention_o_proj_inputs(model, ids_i, slayers, pi)
            with torch.inference_mode():
                oi = model(input_ids=ids_i, use_cache=False)
                oz = model(input_ids=ids_z, use_cache=False)
            icl_prob, _ = _first_token_prob_rank(oi.logits[0, pi], gold)
            zs_prob, zs_rank = _first_token_prob_rank(oz.logits[0, pz], gold)
            total_gap = float(icl_prob - zs_prob)

            handles = [_register_attention_o_proj_replace_hook(model, li, donor_attn[li], patch_position=pz) for li in slayers]
            with torch.inference_mode():
                po = model(input_ids=ids_z, use_cache=False)
            for h in handles:
                h.remove()
            patched_prob, patched_rank = _first_token_prob_rank(po.logits[0, pz], gold)
            pe_vals.append(float(patched_prob - zs_prob))
            rank_delta_vals.append(float(zs_rank - patched_rank))
            if total_gap > 0:
                frac_vals.append(float((patched_prob - zs_prob) / total_gap))

        out["subsets"].append(
            {
                "subset": sname,
                "layers": slayers,
                "n_layers": len(slayers),
                "prob_effect": bootstrap_ci(pe_vals),
                "rank_effect": bootstrap_ci(rank_delta_vals),
                "frac_recovered": bootstrap_ci(frac_vals),
            }
        )
        log(f"  attn-only {pair_id} {sname} mean_pe={float(np.mean(pe_vals)) if pe_vals else 0.0:.6f}")

    return out


def _run_attention_only_robustness(
    model,
    tokenizer,
    device: str,
    *,
    seeds: List[int],
    n_icl: int,
    n_select: int,
    n_eval: int,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "experiment": "attention_only_robustness",
        "settings": {
            "snapshot_seed": int(seeds[0]),
            "fresh_seeds": [int(s) for s in seeds],
            "n_icl": int(n_icl),
            "n_select": int(n_select),
            "n_eval": int(n_eval),
        },
        "pairs": {},
    }
    for pair_id in ["aksharantar_hin_latin", "aksharantar_tel_latin"]:
        rows = []
        rows.append(
            {
                "split_family": "snapshot",
                "seed": int(seeds[0]),
                "result": _run_attention_only_pair(
                    model,
                    tokenizer,
                    device,
                    pair_id,
                    seed=int(seeds[0]),
                    n_icl=n_icl,
                    n_select=n_select,
                    n_eval=n_eval,
                    prefer_snapshot=True,
                ),
            }
        )
        for seed in seeds:
            rows.append(
                {
                    "split_family": "fresh_loader",
                    "seed": int(seed),
                    "result": _run_attention_only_pair(
                        model,
                        tokenizer,
                        device,
                        pair_id,
                        seed=int(seed),
                        n_icl=n_icl,
                        n_select=n_select,
                        n_eval=n_eval,
                        prefer_snapshot=False,
                    ),
                }
            )
        payload["pairs"][pair_id] = rows
    return payload


def _run_mlp_contribution_pair(
    model,
    tokenizer,
    device: str,
    pair_id: str,
    *,
    seed: int,
    n_icl: int,
    n_select: int,
    n_eval: int,
) -> Dict[str, Any]:
    pb = load_pair_bundle(pair_id, seed=seed, n_icl=n_icl, n_select=n_select, n_eval=n_eval, prefer_snapshot=True)
    icl_examples = list(pb["icl_examples"])
    eval_rows = list(pb["eval_rows"][:n_eval])
    source_language = str(pb["source_language"])
    input_script_name = str(pb["input_script_name"])
    output_script_name = str(pb["output_script_name"])

    cfg = getattr(model.config, "text_config", model.config)
    n_layers = int(cfg.num_hidden_layers)
    layer_types = [str(x) for x in getattr(cfg, "layer_types", [])]

    layers_out = []
    for layer in range(n_layers):
        pe_vals: List[float] = []
        rank_delta_vals: List[float] = []
        for word in eval_rows:
            query = str(word["ood"])
            target = str(word["hindi"])
            tids = tokenizer.encode(target, add_special_tokens=False)
            if not tids:
                continue
            gold = int(tids[0])

            icl_raw = build_task_prompt(query, icl_examples, input_script_name=input_script_name, source_language=source_language, output_script_name=output_script_name)
            zs_raw = build_task_prompt(query, None, input_script_name=input_script_name, source_language=source_language, output_script_name=output_script_name)
            ids_i = tokenizer(apply_chat_template(tokenizer, icl_raw), return_tensors="pt").to(device).input_ids
            ids_z = tokenizer(apply_chat_template(tokenizer, zs_raw), return_tensors="pt").to(device).input_ids
            pi = int(ids_i.shape[1] - 1)
            pz = int(ids_z.shape[1] - 1)

            _, donor_mlp = _extract_mlp_io_at_position_from_input_ids(model, ids_i, layer, pi)
            with torch.inference_mode():
                zo = model(input_ids=ids_z, use_cache=False)
            zs_prob, zs_rank = _first_token_prob_rank(zo.logits[0, pz], gold)
            h = register_dense_mlp_output_patch_hook(model, layer, donor_mlp.detach(), patch_position=pz)
            with torch.inference_mode():
                po = model(input_ids=ids_z, use_cache=False)
            h.remove()
            patched_prob, patched_rank = _first_token_prob_rank(po.logits[0, pz], gold)
            pe_vals.append(float(patched_prob - zs_prob))
            rank_delta_vals.append(float(zs_rank - patched_rank))

        layers_out.append(
            {
                "layer": layer,
                "type": layer_types[layer] if layer < len(layer_types) else "?",
                "pe": bootstrap_ci(pe_vals),
                "rank_delta": bootstrap_ci(rank_delta_vals),
            }
        )
        if layer % 5 == 0:
            log(f"  mlp {pair_id} L{layer:02d} mean_pe={float(np.mean(pe_vals)) if pe_vals else 0.0:.6f}")

    return {
        "experiment": "mlp_contribution",
        "pair": pair_id,
        "seed": seed,
        "n_eval": len(eval_rows),
        "layers": layers_out,
    }


def _build_header_only_prompt(*, source_language: str, input_script_name: str, output_script_name: str) -> str:
    return "\n".join(
        [
            f"Task: Transliterate {source_language} written in {input_script_name} into {output_script_name}.",
            "Output only the transliterated token.",
        ]
    )


def _build_prefix_prompt(examples: List[Dict[str, str]], *, source_language: str, input_script_name: str, output_script_name: str) -> str:
    return confirmatory_user_prompt(
        query_token="",
        input_script_name=input_script_name,
        source_language=source_language,
        output_script_name=output_script_name,
        icl_examples=[{"input": str(ex["ood"]), "output": str(ex["hindi"])} for ex in examples],
        variant="canonical",
    ).rsplit("Now transliterate:", 1)[0].rstrip()


def _run_density_pair(
    model,
    tokenizer,
    device: str,
    pair_id: str,
    *,
    seed: int,
    n_select: int,
    n_eval: int,
    densities: List[int],
) -> Dict[str, Any]:
    max_icl = max(densities)
    pb = load_pair_bundle(pair_id, seed=seed, n_icl=max_icl, n_select=n_select, n_eval=n_eval, prefer_snapshot=False)
    all_examples = list(pb["icl_examples"][:max_icl])
    eval_rows = list(pb["eval_rows"][:n_eval])
    source_language = str(pb["source_language"])
    input_script_name = str(pb["input_script_name"])
    output_script_name = str(pb["output_script_name"])

    cfg = getattr(model.config, "text_config", model.config)
    layer_types = [str(x) for x in getattr(cfg, "layer_types", [])]
    global_layers = [i for i, lt in enumerate(layer_types) if "full" in lt]
    if hasattr(model, "set_attn_implementation"):
        try:
            model.set_attn_implementation("eager")
        except Exception:
            pass

    results = []
    header_raw = _build_header_only_prompt(
        source_language=source_language,
        input_script_name=input_script_name,
        output_script_name=output_script_name,
    )
    header_len = int(tokenizer(apply_chat_template(tokenizer, header_raw), return_tensors="pt").input_ids.shape[1])

    for n_ex in densities:
        examples = all_examples[:n_ex]
        prefix_raw = _build_prefix_prompt(examples, source_language=source_language, input_script_name=input_script_name, output_script_name=output_script_name)
        prefix_len = int(tokenizer(apply_chat_template(tokenizer, prefix_raw), return_tensors="pt").input_ids.shape[1])
        per_example_attn: List[Dict[str, Any]] = []
        prob_vals: List[float] = []
        seq_lens: List[int] = []

        for word in eval_rows:
            query = str(word["ood"])
            target = str(word["hindi"])
            tids = tokenizer.encode(target, add_special_tokens=False)
            if not tids:
                continue
            gold = int(tids[0])
            raw = build_task_prompt(query, examples, input_script_name=input_script_name, source_language=source_language, output_script_name=output_script_name)
            rendered = apply_chat_template(tokenizer, raw)
            ids = tokenizer(rendered, return_tensors="pt").to(device).input_ids
            seq_len = int(ids.shape[1])
            seq_lens.append(seq_len)
            last_pos = seq_len - 1
            with torch.inference_mode():
                out = model(input_ids=ids, use_cache=False, output_attentions=True)
            prob, _ = _first_token_prob_rank(out.logits[0, last_pos], gold)
            prob_vals.append(prob)
            attns = out.attentions
            if attns is None:
                continue
            layer_rows = []
            for li in global_layers:
                a = attns[li]
                head_rows = []
                for hi in range(int(a.shape[1])):
                    dist = a[0, hi, last_pos, :].detach().float().cpu().numpy()
                    header_mass = float(dist[: min(header_len, len(dist))].sum())
                    prefix_mass = float(dist[: min(prefix_len, len(dist))].sum())
                    ex_mass = max(0.0, prefix_mass - header_mass)
                    query_mass = float(dist[min(prefix_len, len(dist)) :].sum())
                    head_rows.append(
                        {
                            "head": hi,
                            "example_region_mass": ex_mass,
                            "per_example_mass": float(ex_mass / max(1, n_ex)),
                            "query_region_mass": query_mass,
                            "header_region_mass": header_mass,
                        }
                    )
                layer_rows.append({"layer": li, "heads": head_rows})
            per_example_attn.append({"seq_len": seq_len, "layers": layer_rows})

        summary_layers = []
        for li in global_layers:
            layer_heads: Dict[int, List[float]] = {}
            for item in per_example_attn:
                for layer_entry in item["layers"]:
                    if int(layer_entry["layer"]) != int(li):
                        continue
                    for head_entry in layer_entry["heads"]:
                        layer_heads.setdefault(int(head_entry["head"]), []).append(float(head_entry["per_example_mass"]))
            summary_layers.append(
                {
                    "layer": li,
                    "per_example_mass_by_head": [
                        {"head": h, "per_example_mass": bootstrap_ci(vals)}
                        for h, vals in sorted(layer_heads.items())
                    ],
                }
            )

        results.append(
            {
                "n_examples": int(n_ex),
                "mean_target_prob": bootstrap_ci(prob_vals),
                "mean_seq_len": float(np.mean(seq_lens)) if seq_lens else None,
                "header_len": int(header_len),
                "prefix_len": int(prefix_len),
                "global_attention_summary": summary_layers,
            }
        )
        log(f"  density {pair_id} n_ex={n_ex} mean_prob={float(np.mean(prob_vals)) if prob_vals else 0.0:.6f} mean_len={float(np.mean(seq_lens)) if seq_lens else 0.0:.1f}")

    return {
        "experiment": "density_degradation",
        "pair": pair_id,
        "seed": seed,
        "n_eval": len(eval_rows),
        "densities": densities,
        "global_layers": global_layers,
        "results": results,
    }


JOINT_GROUPS: Dict[str, Dict[str, List[int]]] = {
    "attn_global_only": {"attn": [5, 11, 17, 23], "mlp": []},
    "mlp_global_only": {"attn": [], "mlp": [5, 11, 17, 23]},
    "both_global": {"attn": [5, 11, 17, 23], "mlp": [5, 11, 17, 23]},
    "both_L11_L17": {"attn": [11, 17], "mlp": [11, 17]},
    "both_L17_L23": {"attn": [17, 23], "mlp": [17, 23]},
    "attn_all_layers": {"attn": list(range(26)), "mlp": []},
    "mlp_all_layers": {"attn": [], "mlp": list(range(26))},
    "both_all_layers": {"attn": list(range(26)), "mlp": list(range(26))},
}


def _run_joint_attn_mlp_pair(
    model,
    tokenizer,
    device: str,
    pair_id: str,
    *,
    seed: int,
    n_icl: int,
    n_select: int,
    n_eval: int,
) -> Dict[str, Any]:
    pb = load_pair_bundle(pair_id, seed=seed, n_icl=n_icl, n_select=n_select, n_eval=n_eval, prefer_snapshot=True)
    icl_examples = list(pb["icl_examples"])
    eval_rows = list(pb["eval_rows"][:n_eval])
    source_language = str(pb["source_language"])
    input_script_name = str(pb["input_script_name"])
    output_script_name = str(pb["output_script_name"])

    needed_attn_layers = sorted({li for spec in JOINT_GROUPS.values() for li in spec["attn"]})
    needed_mlp_layers = sorted({li for spec in JOINT_GROUPS.values() for li in spec["mlp"]})

    group_prob_effects: Dict[str, List[float]] = {name: [] for name in JOINT_GROUPS}
    group_rank_effects: Dict[str, List[float]] = {name: [] for name in JOINT_GROUPS}
    group_frac_recovered: Dict[str, List[float]] = {name: [] for name in JOINT_GROUPS}

    for widx, word in enumerate(eval_rows):
        query = str(word["ood"])
        target = str(word["hindi"])
        tids = tokenizer.encode(target, add_special_tokens=False)
        if not tids:
            continue
        gold = int(tids[0])

        icl_raw = build_task_prompt(query, icl_examples, input_script_name=input_script_name, source_language=source_language, output_script_name=output_script_name)
        zs_raw = build_task_prompt(query, None, input_script_name=input_script_name, source_language=source_language, output_script_name=output_script_name)
        ids_i = tokenizer(apply_chat_template(tokenizer, icl_raw), return_tensors="pt").to(device).input_ids
        ids_z = tokenizer(apply_chat_template(tokenizer, zs_raw), return_tensors="pt").to(device).input_ids
        pi = int(ids_i.shape[1] - 1)
        pz = int(ids_z.shape[1] - 1)

        donor_attn = _capture_attention_o_proj_inputs(model, ids_i, needed_attn_layers, pi)
        donor_mlp = _capture_mlp_outputs(model, ids_i, needed_mlp_layers, pi)

        with torch.inference_mode():
            oi = model(input_ids=ids_i, use_cache=False)
            oz = model(input_ids=ids_z, use_cache=False)
        icl_prob, icl_rank = _first_token_prob_rank(oi.logits[0, pi], gold)
        zs_prob, zs_rank = _first_token_prob_rank(oz.logits[0, pz], gold)
        total_gap = float(icl_prob - zs_prob)

        for gname, spec in JOINT_GROUPS.items():
            handles = []
            for li in spec["attn"]:
                handles.append(_register_attention_o_proj_replace_hook(model, li, donor_attn[li], patch_position=pz))
            for li in spec["mlp"]:
                handles.append(register_dense_mlp_output_patch_hook(model, li, donor_mlp[li], patch_position=pz))
            with torch.inference_mode():
                po = model(input_ids=ids_z, use_cache=False)
            for h in handles:
                h.remove()
            patched_prob, patched_rank = _first_token_prob_rank(po.logits[0, pz], gold)
            group_prob_effects[gname].append(float(patched_prob - zs_prob))
            group_rank_effects[gname].append(float(zs_rank - patched_rank))
            if total_gap > 0:
                group_frac_recovered[gname].append(float((patched_prob - zs_prob) / total_gap))

        if (widx + 1) % 10 == 0:
            log(f"  joint {pair_id} {widx + 1}/{len(eval_rows)}")

    groups_out = []
    for gname in JOINT_GROUPS:
        groups_out.append(
            {
                "group": gname,
                "attn_layers": JOINT_GROUPS[gname]["attn"],
                "mlp_layers": JOINT_GROUPS[gname]["mlp"],
                "prob_effect": bootstrap_ci(group_prob_effects[gname]),
                "rank_effect": bootstrap_ci(group_rank_effects[gname]),
                "frac_recovered": bootstrap_ci(group_frac_recovered[gname]),
            }
        )
    return {
        "experiment": "joint_attention_mlp_grouped",
        "pair": pair_id,
        "seed": seed,
        "n_eval": len(eval_rows),
        "groups": groups_out,
    }


def _run_content_specificity_by_count(
    *,
    out_root: Path,
    counts: List[int],
    max_items: int = 30,
    seed: int = 42,
) -> Dict[str, Any]:
    script = PROJECT_ROOT / "paper2_fidelity_calibrated" / "run_1b_matched_control_logit_lens.py"
    summary: Dict[str, Any] = {"pairs": {}}
    for pair_id in ["aksharantar_hin_latin", "aksharantar_tel_latin"]:
        pair_rows = []
        for n_icl in counts:
            out_dir = out_root / "matched_control_by_count" / pair_id / f"n{int(n_icl)}"
            cmd = [
                sys.executable,
                str(script),
                "--model",
                "1b",
                "--pair",
                str(pair_id),
                "--seed",
                str(int(seed)),
                "--n-icl",
                str(int(n_icl)),
                "--n-select",
                "300",
                "--n-eval",
                str(int(max_items)),
                "--max-items",
                str(int(max_items)),
                "--out",
                str(out_dir),
            ]
            log(f"  content-specificity {pair_id} n_icl={n_icl}")
            subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
            out_file = out_dir / "matched_control_logit_lens_summary.json"
            if not out_file.exists():
                # Fallback to most likely script output name.
                out_file = out_dir / "matched_control_logit_lens.json"
            if not out_file.exists():
                candidates = sorted(out_dir.glob("*.json"))
                if not candidates:
                    raise FileNotFoundError(f"No JSON output found in {out_dir}")
                out_file = candidates[-1]
            payload = json.loads(out_file.read_text(encoding="utf-8"))
            by_name = {}
            if isinstance(payload.get("summary"), dict):
                for cname, rows in payload["summary"].items():
                    for layer_row in rows:
                        by_name.setdefault(cname, {})[int(layer_row["layer"])] = layer_row
            def _ratio_at(layer_idx: int) -> Optional[float]:
                try:
                    helpful_rank = float(by_name["helpful"][layer_idx]["rank"]["mean"])
                    corrupt_rank = float(by_name["corrupt"][layer_idx]["rank"]["mean"])
                    if helpful_rank <= 0:
                        return None
                    return corrupt_rank / helpful_rank
                except Exception:
                    return None
            pair_rows.append(
                {
                    "n_icl": int(n_icl),
                    "l17_corrupt_over_helpful_rank": _ratio_at(17),
                    "l25_corrupt_over_helpful_rank": _ratio_at(25),
                    "source_json": str(out_file),
                }
            )
        summary["pairs"][pair_id] = pair_rows
    return summary


def _run_head_seed_robustness(
    model,
    tokenizer,
    device: str,
    *,
    pair_id: str,
    seeds: List[int],
    n_icl: int,
    n_select: int,
    n_eval: int,
) -> Dict[str, Any]:
    per_seed = []
    top_sets = []
    for seed in seeds:
        result = _run_head_attribution_pair(
            model,
            tokenizer,
            device,
            pair_id,
            seed=seed,
            n_icl=n_icl,
            n_select=n_select,
            n_eval=n_eval,
        )
        top_heads = [(int(row["layer"]), int(row["head"])) for row in result["top_heads"][:3]]
        top_sets.append(set(top_heads))
        per_seed.append({"seed": seed, "top3": [{"layer": l, "head": h} for l, h in top_heads]})

    pairwise = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            inter = len(top_sets[i] & top_sets[j])
            union = len(top_sets[i] | top_sets[j])
            pairwise.append({"seed_i": seeds[i], "seed_j": seeds[j], "jaccard": float(inter / union) if union else None, "overlap": int(inter)})

    freq = Counter()
    for s in top_sets:
        for item in s:
            freq[item] += 1
    stable_heads = [
        {"layer": int(layer), "head": int(head), "count": int(count)}
        for (layer, head), count in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
    ]
    return {
        "experiment": "head_attribution_seed_robustness",
        "pair": pair_id,
        "seeds": seeds,
        "per_seed": per_seed,
        "pairwise": pairwise,
        "stable_heads": stable_heads,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Final 1B GPU bundle for Modal runs.")
    ap.add_argument("--tasks", type=str, default="mlp,head,attn-only,density,joint,content-count,seed-robust")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-root", type=str, default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    set_all_seeds(int(args.seed))
    out_root = Path(args.out_root).resolve() if str(args.out_root).strip() else DEFAULT_OUT
    out_root.mkdir(parents=True, exist_ok=True)

    tasks = [t.strip() for t in str(args.tasks).split(",") if t.strip()]
    log(f"Running final 1B GPU bundle tasks={tasks} out_root={out_root}")
    _write_json(out_root / "run_manifest.json", {
        "status": "running",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tasks": tasks,
        "smoke": bool(args.smoke),
        "seed": int(args.seed),
    })

    model, tokenizer = load_model("1b", device=str(args.device))
    device = str(next(model.parameters()).device)
    log(f"Model loaded on {device}")

    # Shared config
    head_n_eval = 4 if args.smoke else 50
    mlp_n_eval = 4 if args.smoke else 50
    density_n_eval = 4 if args.smoke else 30
    joint_n_eval = 4 if args.smoke else 50
    content_counts = [4, 8] if args.smoke else [4, 8, 16, 32]
    density_counts = [4, 8] if args.smoke else [4, 8, 16, 32, 48, 64]
    seeds_for_robust = [42, 123] if args.smoke else [42, 123, 456]

    if "head" in tasks:
        for pair_id in ["aksharantar_hin_latin", "aksharantar_tel_latin"]:
            lang = PAIR_META[pair_id]["lang_slug"]
            out = out_root / f"head_attribution_{lang}_n{head_n_eval}.json"
            if args.skip_existing and out.exists():
                log(f"Skipping existing {out.name}")
            else:
                payload = _run_head_attribution_pair(
                    model,
                    tokenizer,
                    device,
                    pair_id,
                    seed=int(args.seed),
                    n_icl=16,
                    n_select=300,
                    n_eval=head_n_eval,
                )
                _write_json(out, payload)
                log(f"Saved {out}")

    if "attn-only" in tasks:
        for pair_id in ["aksharantar_hin_latin", "aksharantar_tel_latin"]:
            lang = PAIR_META[pair_id]["lang_slug"]
            out = out_root / f"attention_only_contribution_{lang}.json"
            if args.skip_existing and out.exists():
                log(f"Skipping existing {out.name}")
            else:
                payload = _run_attention_only_pair(
                    model,
                    tokenizer,
                    device,
                    pair_id,
                    seed=int(args.seed),
                    n_icl=16,
                    n_select=300,
                    n_eval=(4 if args.smoke else 50),
                    prefer_snapshot=True,
                )
                _write_json(out, payload)
                log(f"Saved {out}")

    if "attn-robust" in tasks:
        out = out_root / "attention_only_robustness.json"
        if args.skip_existing and out.exists():
            log(f"Skipping existing {out.name}")
        else:
            payload = _run_attention_only_robustness(
                model,
                tokenizer,
                device,
                seeds=([42, 123] if args.smoke else [42, 123, 456]),
                n_icl=16,
                n_select=300,
                n_eval=(4 if args.smoke else 30),
            )
            _write_json(out, payload)
            log(f"Saved {out}")

    if "mlp" in tasks:
        for pair_id in ["aksharantar_hin_latin", "aksharantar_tel_latin"]:
            lang = PAIR_META[pair_id]["lang_slug"]
            out = out_root / f"{lang}_mlp_contribution_n{mlp_n_eval}.json"
            if args.skip_existing and out.exists():
                log(f"Skipping existing {out.name}")
            else:
                payload = _run_mlp_contribution_pair(
                    model,
                    tokenizer,
                    device,
                    pair_id,
                    seed=int(args.seed),
                    n_icl=16,
                    n_select=300,
                    n_eval=mlp_n_eval,
                )
                _write_json(out, payload)
                log(f"Saved {out}")

    if "density" in tasks:
        density_out = out_root / f"density_degradation_hindi_telugu_n{density_n_eval}.json"
        if args.skip_existing and density_out.exists():
            log(f"Skipping existing {density_out.name}")
        else:
            payload = {
                "experiment": "density_degradation_hindi_telugu",
                "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "pairs": {},
            }
            for pair_id in ["aksharantar_hin_latin", "aksharantar_tel_latin"]:
                payload["pairs"][pair_id] = _run_density_pair(
                    model,
                    tokenizer,
                    device,
                    pair_id,
                    seed=int(args.seed),
                    n_select=300,
                    n_eval=density_n_eval,
                    densities=density_counts,
                )
            _write_json(density_out, payload)
            log(f"Saved {density_out}")

    if "joint" in tasks:
        for pair_id in ["aksharantar_hin_latin", "aksharantar_tel_latin"]:
            lang = PAIR_META[pair_id]["lang_slug"]
            out = out_root / f"joint_attn_mlp_grouped_{lang}.json"
            if args.skip_existing and out.exists():
                log(f"Skipping existing {out.name}")
            else:
                payload = _run_joint_attn_mlp_pair(
                    model,
                    tokenizer,
                    device,
                    pair_id,
                    seed=int(args.seed),
                    n_icl=16,
                    n_select=300,
                    n_eval=joint_n_eval,
                )
                _write_json(out, payload)
                log(f"Saved {out}")

    if "content-count" in tasks:
        out = out_root / "content_specificity_by_count.json"
        if args.skip_existing and out.exists():
            log(f"Skipping existing {out.name}")
        else:
            payload = _run_content_specificity_by_count(out_root=out_root, counts=content_counts, max_items=(4 if args.smoke else 30), seed=int(args.seed))
            _write_json(out, payload)
            log(f"Saved {out}")

    if "seed-robust" in tasks:
        out = out_root / "head_attribution_seed_robustness.json"
        if args.skip_existing and out.exists():
            log(f"Skipping existing {out.name}")
        else:
            payload = _run_head_seed_robustness(
                model,
                tokenizer,
                device,
                pair_id="aksharantar_hin_latin",
                seeds=seeds_for_robust,
                n_icl=16,
                n_select=300,
                n_eval=(4 if args.smoke else 30),
            )
            _write_json(out, payload)
            log(f"Saved {out}")

    _write_json(out_root / "run_complete.json", {
        "status": "complete",
        "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tasks": tasks,
        "smoke": bool(args.smoke),
        "seed": int(args.seed),
    })
    log("Final 1B GPU bundle complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
