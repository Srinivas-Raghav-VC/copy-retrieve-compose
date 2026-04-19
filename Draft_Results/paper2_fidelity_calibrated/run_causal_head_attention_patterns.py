#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_model_config  # noqa: E402
from core import (  # noqa: E402
    _find_query_span_in_rendered_prompt,
    apply_chat_template,
    build_task_prompt,
    load_model,
    register_transcoder_feature_patch_hook,
    set_all_seeds,
)
from paper2_fidelity_calibrated.phase1_common import (  # noqa: E402
    build_patch_packet,
    load_pair_split,
    load_stagea_best,
    load_transcoder_for_stagea,
    log,
    resolve_stagea_path,
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
    ap = argparse.ArgumentParser(description="Extract attention patterns of top causal heads by prompt region.")
    ap.add_argument("--model", type=str, default="4b", choices=["1b", "4b"])
    ap.add_argument("--pair", type=str, default="aksharantar_hin_latin")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-icl", type=int, default=64)
    ap.add_argument("--n-select", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--max-items", type=int, default=50)
    ap.add_argument("--stagea", type=str, default="")
    ap.add_argument("--top-heads-json", type=str, default="")
    ap.add_argument("--top-n-heads", type=int, default=8)
    ap.add_argument("--external-only", action="store_true")
    ap.add_argument("--require-external-sources", action="store_true")
    ap.add_argument("--min-pool-size", type=int, default=500)
    ap.add_argument("--out", type=str, default="")
    return ap.parse_args()


def _lang_from_pair(pair_id: str) -> str:
    parts = str(pair_id).split("_")
    return parts[1] if len(parts) >= 2 else str(pair_id)


def _default_top_heads_path(pair_id: str, model_key: str) -> Path:
    return PROJECT_ROOT / "artifacts" / "phase5_attribution" / f"top_heads_{model_key}_{_lang_from_pair(pair_id)}.json"


def _find_first_subsequence_after(haystack: Sequence[int], needle: Sequence[int], start: int) -> Optional[Tuple[int, int]]:
    if not needle or len(needle) > len(haystack):
        return None
    n = len(needle)
    for idx in range(max(0, int(start)), len(haystack) - n + 1):
        if list(haystack[idx : idx + n]) == list(needle):
            return (idx, idx + n)
    return None


def _region_mass(dist: torch.Tensor, spans: List[Tuple[int, int]]) -> float:
    if not spans:
        return 0.0
    total = 0.0
    seq_len = int(dist.shape[0])
    for s, e in spans:
        s = max(0, int(s))
        e = min(int(e), seq_len)
        if e > s:
            total += float(dist[s:e].sum().item())
    return total


def _entropy(dist: torch.Tensor) -> float:
    p = dist.float().clamp_min(1e-12)
    return float((-(p * torch.log(p))).sum().item())


def _extract_prompt_regions(
    *,
    tokenizer: Any,
    raw_prompt: str,
    rendered_prompt: str,
    query_token: str,
    icl_examples: List[Dict[str, str]],
) -> Dict[str, List[Tuple[int, int]]]:
    prompt_ids = tokenizer(rendered_prompt, return_tensors="pt")["input_ids"][0].tolist()

    instruction_spans: List[Tuple[int, int]] = []
    prefix_end = raw_prompt.find("Examples:")
    instruction_text = raw_prompt[:prefix_end].strip() if prefix_end >= 0 else raw_prompt.strip()
    instruction_ids = tokenizer.encode(instruction_text, add_special_tokens=False)
    if instruction_ids:
        span = _find_first_subsequence_after(prompt_ids, instruction_ids, 0)
        if span is not None:
            instruction_spans.append(span)

    icl_source_spans: List[Tuple[int, int]] = []
    icl_target_spans: List[Tuple[int, int]] = []
    cursor = 0
    for ex in icl_examples:
        src_text = str(ex.get("ood", "")).strip()
        tgt_text = str(ex.get("hindi", "")).strip()
        if src_text:
            src_ids = tokenizer.encode(src_text, add_special_tokens=False)
            span = _find_first_subsequence_after(prompt_ids, src_ids, cursor)
            if span is not None:
                icl_source_spans.append(span)
                cursor = span[1]
        if tgt_text:
            tgt_ids = tokenizer.encode(tgt_text, add_special_tokens=False)
            span = _find_first_subsequence_after(prompt_ids, tgt_ids, cursor)
            if span is not None:
                icl_target_spans.append(span)
                cursor = span[1]

    query_span = _find_query_span_in_rendered_prompt(tokenizer, rendered_prompt, query_token)
    query_spans = [query_span] if query_span is not None else []

    return {
        "instruction": instruction_spans,
        "icl_source": icl_source_spans,
        "icl_target": icl_target_spans,
        "query": query_spans,
    }


def _load_top_heads(path: Path, *, top_n: int) -> List[Dict[str, Any]]:
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


def _extract_attention_rows(
    *,
    model: Any,
    input_ids: torch.Tensor,
    heads: List[Dict[str, Any]],
    qpos: int,
    prompt_regions: Dict[str, List[Tuple[int, int]]],
    condition: str,
    item_index: int,
    word: Dict[str, str],
) -> List[Dict[str, Any]]:
    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=False, output_attentions=True)
    attentions = getattr(out, "attentions", None)
    if attentions is None:
        raise RuntimeError("Model did not return attentions.")

    rows: List[Dict[str, Any]] = []
    for head_info in heads:
        layer = int(head_info["layer"])
        head = int(head_info["head"])
        if layer >= len(attentions):
            continue
        attn = attentions[layer]
        if not torch.is_tensor(attn) or attn.ndim != 4:
            continue
        if qpos < 0 or qpos >= int(attn.shape[2]) or head < 0 or head >= int(attn.shape[1]):
            continue
        dist = attn[0, head, qpos, :].detach().float().cpu()
        instruction_mass = _region_mass(dist, prompt_regions["instruction"])
        icl_source_mass = _region_mass(dist, prompt_regions["icl_source"])
        icl_target_mass = _region_mass(dist, prompt_regions["icl_target"])
        query_mass = _region_mass(dist, prompt_regions["query"])
        tracked = instruction_mass + icl_source_mass + icl_target_mass + query_mass
        rows.append(
            {
                "item_index": int(item_index),
                "word_ood": str(word["ood"]),
                "word_hindi": str(word["hindi"]),
                "condition": str(condition),
                "layer": int(layer),
                "head": int(head),
                "rank": int(head_info.get("rank", -1)),
                "effect": float(head_info.get("effect", float("nan"))),
                "instruction_mass": float(instruction_mass),
                "icl_source_mass": float(icl_source_mass),
                "icl_target_mass": float(icl_target_mass),
                "query_mass": float(query_mass),
                "other_mass": float(max(0.0, 1.0 - tracked)),
                "entropy": float(_entropy(dist)),
                "concentration": float(dist.max().item()) if int(dist.numel()) > 0 else float("nan"),
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    set_all_seeds(int(args.seed))

    pair_bundle = load_pair_split(
        str(args.pair),
        seed=int(args.seed),
        n_icl=int(args.n_icl),
        n_select=int(args.n_select),
        n_eval=int(args.n_eval),
        external_only=bool(args.external_only),
        require_external_sources=bool(args.require_external_sources),
        min_pool_size=int(args.min_pool_size),
    )

    cfg = get_model_config(str(args.model))
    model, tokenizer = load_model(str(args.model), device=str(args.device))
    device = str(next(model.parameters()).device)

    if hasattr(model, "set_attn_implementation"):
        try:
            model.set_attn_implementation("eager")
            log("Forced attention implementation to eager.")
        except Exception as exc:
            log(f"Could not force eager attention implementation: {exc}")

    stagea_path = resolve_stagea_path(str(args.pair), str(args.model), str(args.stagea))
    if not stagea_path.exists():
        raise FileNotFoundError(f"Missing Stage A artifact: {stagea_path}")
    stagea_best = load_stagea_best(stagea_path, seed=int(args.seed))
    stagea_best["scope_repo"] = str(cfg.scope_repo)

    top_heads_path = Path(str(args.top_heads_json)).resolve() if str(args.top_heads_json).strip() else _default_top_heads_path(str(args.pair), str(args.model))
    if not top_heads_path.exists():
        raise FileNotFoundError(f"Missing top-heads artifact: {top_heads_path}")
    top_heads = _load_top_heads(top_heads_path, top_n=int(args.top_n_heads))

    transcoder = load_transcoder_for_stagea(
        model,
        {**stagea_best, "scope_repo": str(cfg.scope_repo)},
        device,
    )

    out_root = (
        Path(args.out).resolve()
        if str(args.out).strip()
        else PROJECT_ROOT / "paper2_fidelity_calibrated" / "results" / "causal_head_attention_patterns" / str(args.pair) / str(args.model)
    )
    out_root.mkdir(parents=True, exist_ok=True)

    eval_rows = list(pair_bundle["eval_rows"][: max(1, int(args.max_items))])
    item_rows: List[Dict[str, Any]] = []

    log(
        f"Running G2 causal head attention patterns: pair={args.pair} model={args.model} "
        f"heads={[(h['layer'], h['head']) for h in top_heads]}"
    )

    for item_idx, word in enumerate(eval_rows, start=1):
        log(f"[{item_idx}/{len(eval_rows)}] {word['ood']} -> {word['hindi']}")
        packet = build_patch_packet(
            model=model,
            tokenizer=tokenizer,
            transcoder=transcoder,
            word=word,
            icl_examples=pair_bundle["icl_examples"],
            stagea_best=stagea_best,
            input_script_name=pair_bundle["input_script_name"],
            source_language=pair_bundle["source_language"],
            output_script_name=pair_bundle["output_script_name"],
            device=device,
        )

        zs_regions = _extract_prompt_regions(
            tokenizer=tokenizer,
            raw_prompt=packet["zs_prompt"],
            rendered_prompt=packet["zs_rendered"],
            query_token=str(word["ood"]),
            icl_examples=[],
        )
        icl_regions = _extract_prompt_regions(
            tokenizer=tokenizer,
            raw_prompt=packet["icl_prompt"],
            rendered_prompt=packet["icl_rendered"],
            query_token=str(word["ood"]),
            icl_examples=pair_bundle["icl_examples"],
        )

        qpos_zs = int(packet["zs_input_ids"].shape[1] - 1)
        qpos_icl = int(packet["icl_input_ids"].shape[1] - 1)

        item_rows.extend(
            _extract_attention_rows(
                model=model,
                input_ids=packet["zs_input_ids"],
                heads=top_heads,
                qpos=qpos_zs,
                prompt_regions=zs_regions,
                condition="zs",
                item_index=int(item_idx - 1),
                word=word,
            )
        )
        item_rows.extend(
            _extract_attention_rows(
                model=model,
                input_ids=packet["icl_input_ids"],
                heads=top_heads,
                qpos=qpos_icl,
                prompt_regions=icl_regions,
                condition="icl64",
                item_index=int(item_idx - 1),
                word=word,
            )
        )
        patch_hook = register_transcoder_feature_patch_hook(
            model,
            transcoder,
            int(stagea_best["layer"]),
            packet["patch_feats"],
            patch_position=int(packet["zs_patch_position"]),
            target_output_norm=packet["target_output_norm"],
        )
        try:
            item_rows.extend(
                _extract_attention_rows(
                    model=model,
                    input_ids=packet["zs_input_ids"],
                    heads=top_heads,
                    qpos=qpos_zs,
                    prompt_regions=zs_regions,
                    condition="patched",
                    item_index=int(item_idx - 1),
                    word=word,
                )
            )
        finally:
            patch_hook.remove()

    head_summary: List[Dict[str, Any]] = []
    if item_rows:
        keys = sorted({(str(r["condition"]), int(r["layer"]), int(r["head"])) for r in item_rows})
        for condition, layer, head in keys:
            rows = [
                r for r in item_rows
                if str(r["condition"]) == condition and int(r["layer"]) == layer and int(r["head"]) == head
            ]
            if not rows:
                continue
            def _m(name: str) -> float:
                vals = np.array([float(r[name]) for r in rows], dtype=np.float64)
                return float(np.nanmean(vals))
            region_means = {
                "instruction_mass": _m("instruction_mass"),
                "icl_source_mass": _m("icl_source_mass"),
                "icl_target_mass": _m("icl_target_mass"),
                "query_mass": _m("query_mass"),
                "other_mass": _m("other_mass"),
            }
            dominant_region = max(region_means.items(), key=lambda kv: kv[1])[0]
            head_summary.append(
                {
                    "condition": str(condition),
                    "layer": int(layer),
                    "head": int(head),
                    "n_items": int(len(rows)),
                    **region_means,
                    "entropy": _m("entropy"),
                    "concentration": _m("concentration"),
                    "dominant_region": str(dominant_region),
                }
            )

    payload = {
        "experiment": "causal_head_attention_patterns",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pair": str(args.pair),
        "model": str(args.model),
        "seed": int(args.seed),
        "stagea_best": stagea_best,
        "top_heads_path": str(top_heads_path),
        "top_heads": top_heads,
        "n_items": int(len(eval_rows)),
        "notes": {
            "query_position": "final prompt position (next-token prediction site)",
            "tracked_regions": ["instruction", "icl_source", "icl_target", "query", "other"],
            "conditions": ["zs", "icl64", "patched"],
            "patch_condition": "Stage-A best feature patch applied on ZS prompt",
        },
        "head_summary": head_summary,
        "items": item_rows,
    }
    _write_json(out_root / "causal_head_attention_patterns.json", payload)
    log(f"Saved: {out_root / 'causal_head_attention_patterns.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
