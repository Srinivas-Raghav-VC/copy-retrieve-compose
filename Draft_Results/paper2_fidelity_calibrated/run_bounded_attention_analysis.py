#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
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


def _prompt_fingerprint(*, raw_prompt: str, rendered_prompt: str) -> Dict[str, Any]:
    def _sha(text: str) -> str:
        return hashlib.sha256(str(text).encode("utf-8")).hexdigest()

    return {
        "raw_length_chars": len(str(raw_prompt)),
        "rendered_length_chars": len(str(rendered_prompt)),
        "raw_sha256": _sha(raw_prompt),
        "rendered_sha256": _sha(rendered_prompt),
    }


def _find_first_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> Optional[Tuple[int, int]]:
    if not needle or len(needle) > len(haystack):
        return None
    n = len(needle)
    for start in range(0, len(haystack) - n + 1):
        if list(haystack[start : start + n]) == list(needle):
            return (start, start + n)
    return None


def _find_region_span(
    tokenizer: Any,
    rendered_prompt: str,
    region_text: str,
    *,
    prefer_last: bool = False,
) -> Optional[Tuple[int, int]]:
    from core import _find_last_subsequence

    region_ids = tokenizer.encode(region_text, add_special_tokens=False)
    if not region_ids:
        return None
    prompt_ids = tokenizer(rendered_prompt, return_tensors="pt")["input_ids"][0].tolist()
    if prefer_last:
        return _find_last_subsequence(prompt_ids, region_ids)
    return _find_first_subsequence(prompt_ids, region_ids)


def _extract_prompt_regions(
    *,
    tokenizer: Any,
    raw_prompt: str,
    rendered_prompt: str,
    query_token: str,
) -> Dict[str, Optional[Tuple[int, int]]]:
    from core import _find_query_span_in_rendered_prompt

    lines = raw_prompt.splitlines()
    instruction_text = "\n".join(lines[:2])
    instruction_span = _find_region_span(tokenizer, rendered_prompt, instruction_text, prefer_last=False)

    examples_span = None
    if "Examples:\n" in raw_prompt and "\nNow transliterate:" in raw_prompt:
        start = raw_prompt.index("Examples:")
        end = raw_prompt.index("\nNow transliterate:")
        examples_text = raw_prompt[start:end]
        examples_span = _find_region_span(tokenizer, rendered_prompt, examples_text, prefer_last=False)

    query_span = _find_query_span_in_rendered_prompt(tokenizer, rendered_prompt, query_token)
    return {
        "instruction_span": instruction_span,
        "examples_span": examples_span,
        "query_span": query_span,
    }


def _slice_mass(dist: torch.Tensor, span: Optional[Tuple[int, int]]) -> float:
    if span is None:
        return 0.0
    s, e = int(span[0]), int(span[1])
    if s < 0 or e <= s or s >= int(dist.shape[-1]):
        return 0.0
    e = min(e, int(dist.shape[-1]))
    return float(dist[s:e].sum().item())


def _entropy(dist: torch.Tensor) -> float:
    p = dist.float().clamp_min(1e-12)
    return float((-(p * torch.log(p))).sum().item())


def _extract_attention_audit(
    *,
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    query_position: int,
    prompt_regions: Dict[str, Optional[Tuple[int, int]]],
    layer_start: int,
    layer_end: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=False, output_attentions=True)
    attentions = getattr(out, "attentions", None)
    if attentions is None:
        raise RuntimeError("Model did not return attentions for bounded attention analysis.")

    item_rows: List[Dict[str, Any]] = []
    head_rows: List[Dict[str, Any]] = []
    n_layers = len(attentions)
    for layer in range(max(0, layer_start), min(layer_end, n_layers - 1) + 1):
        attn = attentions[layer]
        if not torch.is_tensor(attn) or attn.ndim != 4:
            continue
        qpos = int(query_position)
        if qpos < 0 or qpos >= int(attn.shape[2]):
            continue
        head_dists = attn[0, :, qpos, :].detach().float().cpu()
        instr = []
        ex = []
        query = []
        ents = []
        conc = []
        for head_idx in range(int(head_dists.shape[0])):
            dist = head_dists[head_idx]
            instr_mass = _slice_mass(dist, prompt_regions.get("instruction_span"))
            ex_mass = _slice_mass(dist, prompt_regions.get("examples_span"))
            query_mass = _slice_mass(dist, prompt_regions.get("query_span"))
            ent = _entropy(dist)
            mx = float(dist.max().item()) if int(dist.numel()) > 0 else float("nan")
            instr.append(instr_mass)
            ex.append(ex_mass)
            query.append(query_mass)
            ents.append(ent)
            conc.append(mx)
            head_rows.append(
                {
                    "layer": int(layer),
                    "head": int(head_idx),
                    "instruction_mass": instr_mass,
                    "examples_mass": ex_mass,
                    "query_source_mass": query_mass,
                    "entropy": ent,
                    "concentration": mx,
                }
            )
        item_rows.append(
            {
                "layer": int(layer),
                "instruction_mass": float(np.mean(instr)) if instr else float("nan"),
                "examples_mass": float(np.mean(ex)) if ex else float("nan"),
                "query_source_mass": float(np.mean(query)) if query else float("nan"),
                "entropy": float(np.mean(ents)) if ents else float("nan"),
                "concentration": float(np.mean(conc)) if conc else float("nan"),
            }
        )
    return item_rows, head_rows


def _generation_metrics(gold_text: str, pred_text: str, *, target_script: str) -> Dict[str, float]:
    from paper2_fidelity_calibrated.eval_utils import (
        akshara_cer,
        continuation_akshara_cer,
        first_entry_correct,
        normalize_text,
        script_compliance,
    )

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


def _choose_implicated_layers(
    pair_rows: List[Dict[str, Any]],
    *,
    top_layers: int,
) -> List[int]:
    scores: Dict[int, float] = defaultdict(float)
    for row in pair_rows:
        if str(row.get("condition")) != "icl64":
            continue
        layer = int(row["layer"])
        locus = str(row["locus"])
        examples_mass = _safe_float(row.get("examples_mass"))
        if not np.isfinite(examples_mass):
            continue
        if locus == "source_last_subtoken":
            scores[layer] += examples_mass
        elif locus == "target_pos1_teacher_forced":
            scores[layer] += examples_mass
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [int(layer) for layer, _ in ordered[: max(1, int(top_layers))]]


def _choose_top_heads(
    pair_rows: List[Dict[str, Any]],
    *,
    layer: int,
    top_heads: int,
) -> List[int]:
    head_scores: Dict[int, float] = defaultdict(float)
    for row in pair_rows:
        if int(row.get("layer", -1)) != int(layer):
            continue
        if str(row.get("condition")) != "icl64":
            continue
        if str(row.get("locus")) not in {"source_last_subtoken", "target_pos1_teacher_forced"}:
            continue
        head = int(row.get("head", -1))
        score = _safe_float(row.get("examples_mass"))
        if head >= 0 and np.isfinite(score):
            head_scores[head] += score
    ordered = sorted(head_scores.items(), key=lambda kv: kv[1], reverse=True)
    return [int(head) for head, _ in ordered[: max(1, int(top_heads))]]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Bounded attention analysis for 4B Hindi/Telugu CFOM rescue.")
    ap.add_argument("--model", type=str, default="4b", choices=["4b"])
    ap.add_argument("--pairs", type=str, default="aksharantar_hin_latin,aksharantar_tel_latin")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-icl", type=int, default=64)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--n-select", type=int, default=300)
    ap.add_argument("--external-only", action="store_true")
    ap.add_argument("--require-external-sources", action="store_true")
    ap.add_argument("--min-pool-size", type=int, default=500)
    ap.add_argument("--layer-start", type=int, default=7)
    ap.add_argument("--layer-end", type=int, default=20)
    ap.add_argument("--top-layers", type=int, default=2)
    ap.add_argument("--top-heads", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--out", type=str, default="")
    return ap.parse_args()


def main() -> int:
    from core import (
        _find_last_subsequence,
        _teacher_forced_metrics_from_input_ids,
        apply_chat_template,
        build_task_prompt,
        get_model_config,
        load_model,
        register_attention_head_ablation_hook,
        set_all_seeds,
        split_data_three_way,
    )
    from paper2_fidelity_calibrated.eval_utils import normalize_text

    args = parse_args()
    set_all_seeds(int(args.seed))

    pair_ids = _parse_csv(args.pairs)
    if not pair_ids:
        raise RuntimeError("No pairs provided.")

    cfg = get_model_config(args.model)
    model, tokenizer = load_model(args.model, device=str(args.device))
    attn_impl = ""
    if hasattr(model, "set_attn_implementation"):
        try:
            model.set_attn_implementation("eager")
            attn_impl = "eager"
            log("Forced attention implementation to eager for attention capture.")
        except Exception as exc:
            attn_impl = f"eager_failed:{type(exc).__name__}"
            log(f"Could not force eager attention implementation: {exc}")

    out_root = (
        Path(args.out).resolve()
        if str(args.out).strip()
        else PROJECT_ROOT / "paper2_fidelity_calibrated" / "results" / "bounded_attention_analysis"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    pair_prompt_meta = _get_pair_prompt_metadata(pair_ids[0])
    words0, provenance0 = _load_words(
        pair_ids[0],
        external_only=bool(args.external_only),
        require_external_sources=bool(args.require_external_sources),
        min_pool_size=int(args.min_pool_size),
    )
    icl0, _, ev0 = split_data_three_way(words0, n_icl=int(args.n_icl), n_select=int(args.n_select), n_eval=int(args.n_eval), seed=int(args.seed))
    prompt0 = build_task_prompt(
        ev0[0]["ood"],
        icl0,
        input_script_name=str(pair_prompt_meta["source_script"]),
        source_language=str(pair_prompt_meta["source_language"]),
        output_script_name=str(pair_prompt_meta["target_script"]),
        prompt_variant="canonical",
    )
    rendered0 = apply_chat_template(tokenizer, prompt0)

    config_payload = {
        "experiment": "bounded_attention_analysis",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": args.model,
        "pairs": pair_ids,
        "seed": int(args.seed),
        "runtime_identity": _runtime_identity(model_key=args.model, hf_id=cfg.hf_id, tokenizer=tokenizer, model=model),
        "attention_implementation": attn_impl or "unknown",
        "git_commit_hash": _git_commit_hash(),
        "prompt_fingerprint": _prompt_fingerprint(raw_prompt=prompt0, rendered_prompt=rendered0),
        "frozen_prompt_variant": "canonical",
        "claim_level": "intervention_only",
        "analysis_scope": "bounded_attention_routing",
        "conditions": ["explicit_zs", "icl8", "icl64"],
        "loci": ["source_last_subtoken", "target_pos1_teacher_forced"],
        "layers": {"start": int(args.layer_start), "end": int(args.layer_end)},
        "top_layers": int(args.top_layers),
        "top_heads": int(args.top_heads),
        "ablation_kind": "grouped_attention_head_ablation",
        "ablation_position": "source_last_subtoken",
        "split_sizes": {"n_icl": int(args.n_icl), "n_select": int(args.n_select), "n_eval": int(args.n_eval)},
    }
    _write_json(out_root / "attention_analysis_config.json", config_payload)

    item_rows: List[Dict[str, Any]] = []
    head_rows: List[Dict[str, Any]] = []
    ablation_rows: List[Dict[str, Any]] = []
    entry_rows: List[Dict[str, Any]] = []
    pair_decisions: List[Dict[str, Any]] = []

    pad_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", 0)

    def _generate_with_ablation(input_ids: torch.Tensor, *, layer: int, heads: List[int], pos: int) -> str:
        handle = register_attention_head_ablation_hook(model, layer, heads, ablate_position=pos)
        try:
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
            handle.remove()

    for pair_id in pair_ids:
        log(f"Pair {pair_id}")
        words, provenance = _load_words(
            pair_id,
            external_only=bool(args.external_only),
            require_external_sources=bool(args.require_external_sources),
            min_pool_size=int(args.min_pool_size),
        )
        prompt_meta = _get_pair_prompt_metadata(pair_id)
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
        icl8_examples = list(icl_examples[:8])

        pair_item_rows: List[Dict[str, Any]] = []
        pair_head_rows: List[Dict[str, Any]] = []
        cached_icl64: List[Dict[str, Any]] = []

        for idx, word in enumerate(eval_samples):
            prompts = {
                "explicit_zs": build_task_prompt(
                    word["ood"],
                    None,
                    input_script_name=input_script_name,
                    source_language=source_language,
                    output_script_name=target_script,
                    prompt_variant="canonical",
                ),
                "icl8": build_task_prompt(
                    word["ood"],
                    icl8_examples,
                    input_script_name=input_script_name,
                    source_language=source_language,
                    output_script_name=target_script,
                    prompt_variant="canonical",
                ),
                "icl64": build_task_prompt(
                    word["ood"],
                    icl_examples,
                    input_script_name=input_script_name,
                    source_language=source_language,
                    output_script_name=target_script,
                    prompt_variant="canonical",
                ),
            }
            target_ids = tokenizer.encode(str(word["hindi"]), add_special_tokens=False)
            target_id = int(target_ids[0]) if target_ids else -1
            target_tensor = (
                torch.tensor(target_ids, device=args.device, dtype=torch.long).unsqueeze(0)
                if target_ids
                else None
            )
            query_ids = tokenizer.encode(word["ood"], add_special_tokens=False)

            for condition_name, raw_prompt in prompts.items():
                log(f"{pair_id}: item={idx} condition={condition_name}")
                rendered = apply_chat_template(tokenizer, raw_prompt)
                prompt_inputs = tokenizer(rendered, return_tensors="pt").to(args.device)
                prompt_ids = prompt_inputs.input_ids
                prompt_len = int(prompt_ids.shape[1])
                regions = _extract_prompt_regions(
                    tokenizer=tokenizer,
                    raw_prompt=raw_prompt,
                    rendered_prompt=rendered,
                    query_token=word["ood"],
                )
                query_span = regions.get("query_span")
                if query_span is None:
                    raise ValueError(f"Fail-closed query span localization failed for {pair_id} {condition_name} item={idx}")
                source_pos = int(query_span[1] - 1)
                full_ids = prompt_ids if target_tensor is None else torch.cat([prompt_ids, target_tensor.to(prompt_ids.dtype)], dim=1)
                target_pos1 = int(prompt_len) if target_tensor is not None and len(target_ids) > 0 else -1

                for locus_name, query_pos, audit_ids in (
                    ("source_last_subtoken", source_pos, prompt_ids),
                    ("target_pos1_teacher_forced", target_pos1, full_ids),
                ):
                    if query_pos < 0:
                        continue
                    log(f"{pair_id}: item={idx} condition={condition_name} locus={locus_name}")
                    audit_item_rows, audit_head_rows = _extract_attention_audit(
                        model=model,
                        tokenizer=tokenizer,
                        input_ids=audit_ids,
                        query_position=query_pos,
                        prompt_regions=regions,
                        layer_start=int(args.layer_start),
                        layer_end=int(args.layer_end),
                    )
                    for row in audit_item_rows:
                        full_row = {
                            "pair": pair_id,
                            "seed": int(args.seed),
                            "item_index": int(idx),
                            "condition": condition_name,
                            "locus": locus_name,
                            "word_english": str(word["english"]),
                            "word_target": str(word["hindi"]),
                            "word_source_romanized": str(word["ood"]),
                            "query_position": int(query_pos),
                            "source_fragmentation_tokens": int(len(query_ids)),
                            **row,
                        }
                        item_rows.append(full_row)
                        pair_item_rows.append(full_row)
                    for row in audit_head_rows:
                        full_row = {
                            "pair": pair_id,
                            "seed": int(args.seed),
                            "item_index": int(idx),
                            "condition": condition_name,
                            "locus": locus_name,
                            **row,
                        }
                        head_rows.append(full_row)
                        pair_head_rows.append(full_row)

                if condition_name == "icl64":
                    zs_prompt = prompts["explicit_zs"]
                    icl64_metrics = _teacher_forced_metrics_from_input_ids(
                        model=model,
                        input_ids=prompt_ids,
                        target_ids=target_ids,
                        target_id=target_id,
                        device=str(args.device),
                        competitor_id=-1,
                    )
                    attention_mask = torch.ones_like(prompt_ids)
                    with torch.inference_mode():
                        out = model.generate(
                            prompt_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=int(args.max_new_tokens),
                            do_sample=False,
                            pad_token_id=int(pad_id),
                        )
                    pred = normalize_text(tokenizer.decode(out[0, prompt_ids.shape[1] :], skip_special_tokens=True).strip())
                    icl64_gen = _generation_metrics(word["hindi"], pred, target_script=target_script)
                    cached_icl64.append(
                        {
                            "word": word,
                            "prompt_ids": prompt_ids,
                            "prompt_text": raw_prompt,
                            "target_ids": target_ids,
                            "target_id": target_id,
                            "query_pos": source_pos,
                            "icl64_tf": icl64_metrics,
                            "icl64_gen": icl64_gen,
                            "zs_prompt": zs_prompt,
                        }
                    )

        # Pair-level summaries for attention mass.
        pair_layer_rows: List[Dict[str, Any]] = []
        grouped: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = defaultdict(list)
        for row in pair_item_rows:
            grouped[(str(row["condition"]), str(row["locus"]), int(row["layer"]))].append(row)
        for (condition_name, locus_name, layer), rows in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
            pair_layer_rows.append(
                {
                    "pair": pair_id,
                    "condition": condition_name,
                    "locus": locus_name,
                    "layer": int(layer),
                    "instruction_mass": float(np.nanmean([_safe_float(r["instruction_mass"]) for r in rows])),
                    "examples_mass": float(np.nanmean([_safe_float(r["examples_mass"]) for r in rows])),
                    "query_source_mass": float(np.nanmean([_safe_float(r["query_source_mass"]) for r in rows])),
                    "entropy": float(np.nanmean([_safe_float(r["entropy"]) for r in rows])),
                    "concentration": float(np.nanmean([_safe_float(r["concentration"]) for r in rows])),
                }
            )

        implicated_layers = _choose_implicated_layers(pair_layer_rows, top_layers=int(args.top_layers))
        log(f"{pair_id}: implicated attention layers={implicated_layers}")

        pair_head_summary_rows: List[Dict[str, Any]] = []
        head_grouped: Dict[Tuple[str, str, int, int], List[Dict[str, Any]]] = defaultdict(list)
        for row in pair_head_rows:
            head_grouped[(str(row["condition"]), str(row["locus"]), int(row["layer"]), int(row["head"]))].append(row)
        for (condition_name, locus_name, layer, head), rows in sorted(head_grouped.items(), key=lambda x: (x[0][0], x[0][1], x[0][2], x[0][3])):
            pair_head_summary_rows.append(
                {
                    "pair": pair_id,
                    "condition": condition_name,
                    "locus": locus_name,
                    "layer": int(layer),
                    "head": int(head),
                    "instruction_mass": float(np.nanmean([_safe_float(r["instruction_mass"]) for r in rows])),
                    "examples_mass": float(np.nanmean([_safe_float(r["examples_mass"]) for r in rows])),
                    "query_source_mass": float(np.nanmean([_safe_float(r["query_source_mass"]) for r in rows])),
                    "entropy": float(np.nanmean([_safe_float(r["entropy"]) for r in rows])),
                    "concentration": float(np.nanmean([_safe_float(r["concentration"]) for r in rows])),
                }
            )

        # Tiny grouped-head ablation on implicated layers only.
        pair_ablation_rows: List[Dict[str, Any]] = []
        for layer in implicated_layers:
            heads = _choose_top_heads(pair_head_summary_rows, layer=int(layer), top_heads=int(args.top_heads))
            if not heads:
                continue
            log(f"{pair_id}: grouped head ablation layer={layer} heads={heads}")
            ablated_rows = []
            for item in cached_icl64:
                handle = register_attention_head_ablation_hook(
                    model,
                    int(layer),
                    heads,
                    ablate_position=int(item["query_pos"]),
                )
                try:
                    ablated_tf = _teacher_forced_metrics_from_input_ids(
                        model=model,
                        input_ids=item["prompt_ids"],
                        target_ids=item["target_ids"],
                        target_id=item["target_id"],
                        device=str(args.device),
                        competitor_id=-1,
                    )
                    pred = _generate_with_ablation(
                        item["prompt_ids"],
                        layer=int(layer),
                        heads=heads,
                        pos=int(item["query_pos"]),
                    )
                finally:
                    handle.remove()
                ablated_gen = _generation_metrics(item["word"]["hindi"], pred, target_script=target_script)
                ablated_rows.append(
                    {
                        "pair": pair_id,
                        "layer": int(layer),
                        "heads": ",".join(str(h) for h in heads),
                        "exact_match_icl64": _safe_float(item["icl64_gen"]["exact_match"]),
                        "exact_match_ablated": _safe_float(ablated_gen["exact_match"]),
                        "first_entry_icl64": _safe_float(item["icl64_gen"]["first_entry_correct"]),
                        "first_entry_ablated": _safe_float(ablated_gen["first_entry_correct"]),
                        "continuation_icl64": _safe_float(item["icl64_gen"]["continuation_fidelity"]),
                        "continuation_ablated": _safe_float(ablated_gen["continuation_fidelity"]),
                        "pe_logit_icl64": _safe_float(item["icl64_tf"]["first_logit"]),
                        "pe_logit_ablated": _safe_float(ablated_tf["first_logit"]),
                        "nll_pos1_icl64": _safe_float(item["icl64_tf"].get("target_pos1_nll")),
                        "nll_pos1_ablated": _safe_float(ablated_tf.get("target_pos1_nll")),
                        "nll_pos2_icl64": _safe_float(item["icl64_tf"].get("target_pos2_nll")),
                        "nll_pos2_ablated": _safe_float(ablated_tf.get("target_pos2_nll")),
                        "nll_pos3_icl64": _safe_float(item["icl64_tf"].get("target_pos3_nll")),
                        "nll_pos3_ablated": _safe_float(ablated_tf.get("target_pos3_nll")),
                    }
                )
            if ablated_rows:
                pair_ablation_rows.append(
                    {
                        "pair": pair_id,
                        "layer": int(layer),
                        "heads": heads,
                        "exact_match_drop": float(np.nanmean([r["exact_match_icl64"] - r["exact_match_ablated"] for r in ablated_rows])),
                        "first_entry_drop": float(np.nanmean([r["first_entry_icl64"] - r["first_entry_ablated"] for r in ablated_rows])),
                        "continuation_drop": float(np.nanmean([r["continuation_icl64"] - r["continuation_ablated"] for r in ablated_rows if np.isfinite(r["continuation_icl64"]) and np.isfinite(r["continuation_ablated"])])) if any(np.isfinite(r["continuation_icl64"]) and np.isfinite(r["continuation_ablated"]) for r in ablated_rows) else float("nan"),
                        "nll_pos1_delta": float(np.nanmean([r["nll_pos1_ablated"] - r["nll_pos1_icl64"] for r in ablated_rows])),
                        "nll_pos2_delta": float(np.nanmean([r["nll_pos2_ablated"] - r["nll_pos2_icl64"] for r in ablated_rows])),
                        "nll_pos3_delta": float(np.nanmean([r["nll_pos3_ablated"] - r["nll_pos3_icl64"] for r in ablated_rows])),
                    }
                )
                entry_rows.extend(ablated_rows)
        ablation_rows.extend(pair_ablation_rows)

        strongest_ablation = max(pair_ablation_rows, key=lambda r: _safe_float(r.get("first_entry_drop")), default=None)
        pair_decisions.append(
            {
                "pair": pair_id,
                "implicated_layers": implicated_layers,
                "top_attention_layer": int(implicated_layers[0]) if implicated_layers else None,
                "top_attention_head_group": strongest_ablation.get("heads") if strongest_ablation else [],
                "attention_hypothesis_supported": bool(
                    strongest_ablation
                    and np.isfinite(_safe_float(strongest_ablation.get("first_entry_drop")))
                    and _safe_float(strongest_ablation.get("first_entry_drop")) > 0.0
                ),
                "rationale": (
                    "Grouped head ablation reduced ICL64 entry/continuation metrics in implicated mid-to-late layers."
                    if strongest_ablation and _safe_float(strongest_ablation.get("first_entry_drop")) > 0.0
                    else "Bounded attention audit found implicated layers, but grouped head ablation did not yet show a clear positive necessity signal."
                ),
            }
        )

        _write_rows(out_root / f"{pair_id}_attention_layer_summary", pair_layer_rows)
        _write_rows(out_root / f"{pair_id}_attention_head_summary", pair_head_summary_rows)
        _write_rows(out_root / f"{pair_id}_grouped_head_ablation", pair_ablation_rows)

    summary = {
        "paper": "bounded_attention_routing_diagnostic",
        "status": "complete",
        "pairs": pair_ids,
        "pair_decisions": pair_decisions,
        "recommendation": (
            "attention_routing_supported"
            if any(bool(x.get("attention_hypothesis_supported")) for x in pair_decisions)
            else "attention_routing_inconclusive"
        ),
    }
    decision = {
        "status": "complete",
        "next_step": (
            "paper_framing_skeptical_with_attention_support"
            if any(bool(x.get("attention_hypothesis_supported")) for x in pair_decisions)
            else "paper_framing_skeptical_attention_inconclusive"
        ),
        "pair_decisions": pair_decisions,
        "bounded_window": [int(args.layer_start), int(args.layer_end)],
        "note": "This is a bounded attention-routing diagnostic over layers 7–20; null results here do not falsify full-stack attention involvement.",
    }

    _write_rows(out_root / "attention_item_table", item_rows)
    _write_rows(out_root / "attention_head_delta_table", head_rows)
    _write_rows(out_root / "grouped_head_ablation_table", ablation_rows)
    _write_rows(out_root / "entry_vs_continuation_table", entry_rows)
    _write_json(out_root / "attention_analysis_summary.json", summary)
    _write_json(out_root / "attention_analysis_decision_note.json", decision)
    log(f"Bounded attention analysis complete. Recommendation={summary['recommendation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
