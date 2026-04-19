#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_model_config
from core import apply_chat_template, build_task_prompt, load_model, split_data_three_way
from paper2_fidelity_calibrated.eval_utils import (
    build_bare_zs_prompt,
    evaluate_prompt_condition,
    mean_metric,
)
from paper2_fidelity_calibrated.protocol_utils import (
    premise_gap_summary,
    prompt_fingerprint,
    prompt_template_fingerprint,
    runtime_identity,
)
from paper2_fidelity_calibrated.run import _load_words, _prompt_naming
from rescue_research.data_pipeline.ingest import get_pair_prompt_metadata


DEFAULT_BEHAVIOR_KEYS = (
    "exact_match",
    "first_entry_correct",
    "akshara_cer",
    "script_compliance",
)


def _normalize_label(label: str) -> str:
    val = str(label or "").strip().lower()
    if val in {"unambiguous", "orthographic_variant", "intrinsically_ambiguous"}:
        return val
    return ""


def _load_ambiguity_labels(path: Path) -> dict[tuple[str, str, str], str]:
    rows: dict[tuple[str, str, str], str] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pair = str(row.get("pair", "")).strip()
            source = str(row.get("source", "")).strip()
            target = str(row.get("target", "")).strip()
            label = _normalize_label(str(row.get("label", "")))
            if pair and source and target and label:
                rows[(pair, source, target)] = label
    return rows


def _attach_labels(
    rows: Iterable[Dict[str, str]],
    *,
    pair_id: str,
    labels: Mapping[tuple[str, str, str], str],
) -> list[Dict[str, str]]:
    out: list[Dict[str, str]] = []
    for row in rows:
        item = dict(row)
        key = (str(pair_id), str(item.get("ood", "")), str(item.get("hindi", "")))
        item["ambiguity_label"] = str(labels.get(key, "") or "")
        out.append(item)
    return out


def _prompt_texts(
    word: Dict[str, str],
    *,
    icl_examples: List[Dict[str, str]],
    prompt_meta: Mapping[str, str],
    prompt_variant: str,
) -> Dict[str, str]:
    source_language, input_script_name, output_script_name = _prompt_naming(dict(prompt_meta))
    query = str(word["ood"])
    return {
        "bare_zs": build_bare_zs_prompt(query),
        "explicit_zs": build_task_prompt(
            query,
            None,
            input_script_name=input_script_name,
            source_language=source_language,
            output_script_name=output_script_name,
            prompt_variant=prompt_variant,
        ),
        "icl8": build_task_prompt(
            query,
            icl_examples[:8],
            input_script_name=input_script_name,
            source_language=source_language,
            output_script_name=output_script_name,
            prompt_variant=prompt_variant,
        ),
        "icl64": build_task_prompt(
            query,
            icl_examples,
            input_script_name=input_script_name,
            source_language=source_language,
            output_script_name=output_script_name,
            prompt_variant=prompt_variant,
        ),
    }


def _evaluate_rows(
    model: Any,
    tokenizer: Any,
    *,
    eval_rows: List[Dict[str, str]],
    icl_examples: List[Dict[str, str]],
    prompt_meta: Mapping[str, str],
    prompt_variant: str,
    device: str,
    max_new_tokens: int,
) -> list[Dict[str, Any]]:
    target_script = str(prompt_meta.get("target_script", "")).strip() or "Devanagari"
    out: list[Dict[str, Any]] = []
    for row in eval_rows:
        prompts = _prompt_texts(
            row,
            icl_examples=icl_examples,
            prompt_meta=prompt_meta,
            prompt_variant=prompt_variant,
        )
        item: Dict[str, Any] = {
            "english": str(row.get("english", "")),
            "source": str(row.get("ood", "")),
            "target": str(row.get("hindi", "")),
            "ambiguity_label": str(row.get("ambiguity_label", "")),
            "conditions": {},
        }
        for name, prompt in prompts.items():
            item["conditions"][name] = evaluate_prompt_condition(
                model,
                tokenizer,
                prompt_text=prompt,
                target_text=str(row["hindi"]),
                target_script=target_script,
                device=device,
                max_new_tokens=max_new_tokens,
            )
        out.append(item)
    return out


def _slice_items(items: List[Dict[str, Any]], *, ambiguity: str | None) -> List[Dict[str, Any]]:
    if not ambiguity:
        return list(items)
    want = str(ambiguity).strip().lower()
    return [item for item in items if str(item.get("ambiguity_label", "")).strip().lower() == want]


def _summarize_condition(items: List[Dict[str, Any]], *, condition: str) -> Dict[str, float]:
    rows = [dict(item["conditions"][condition]) for item in items]
    return {
        "n": int(len(rows)),
        "exact_match": mean_metric(rows, "exact_match"),
        "akshara_cer": mean_metric(rows, "akshara_cer"),
        "script_compliance": mean_metric(rows, "script_compliance"),
        "first_entry_correct": mean_metric(rows, "first_entry_correct"),
        "continuation_akshara_cer": mean_metric(rows, "continuation_akshara_cer"),
        "joint_logprob": mean_metric(rows, "joint_logprob"),
        "target_pos1_nll": mean_metric(rows, "target_pos1_nll"),
        "target_pos2_nll": mean_metric(rows, "target_pos2_nll"),
        "target_pos3_nll": mean_metric(rows, "target_pos3_nll"),
    }


def _gap_block(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not items:
        return out
    for key in DEFAULT_BEHAVIOR_KEYS:
        explicit = [float(item["conditions"]["explicit_zs"][key]) for item in items]
        icl64 = [float(item["conditions"]["icl64"][key]) for item in items]
        summary = premise_gap_summary(explicit, icl64)
        direction = "higher_better" if key != "akshara_cer" else "lower_better"
        if direction == "lower_better":
            summary = premise_gap_summary(icl64, explicit)
        out[key] = summary
    out["clearly_above_floor_noise"] = bool(
        any(bool(out[key].get("ci_excludes_zero", False)) for key in DEFAULT_BEHAVIOR_KEYS)
    )
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the Stage 0 CFOM premise gate.")
    ap.add_argument("--model", type=str, required=True, choices=["270m", "1b", "4b", "12b"])
    ap.add_argument("--pair", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-icl", type=int, default=64)
    ap.add_argument("--n-select", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--prompt-variant", type=str, default="canonical")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--ambiguity-labels", type=str, default="")
    ap.add_argument("--external-only", action="store_true")
    ap.add_argument("--require-external-sources", action="store_true")
    ap.add_argument("--min-pool-size", type=int, default=0)
    ap.add_argument("--out", type=str, default="")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    pair_id = str(args.pair)
    words, provenance = _load_words(
        pair_id,
        external_only=bool(args.external_only),
        require_external_sources=bool(args.require_external_sources),
        min_pool_size=int(args.min_pool_size),
    )
    labels = _load_ambiguity_labels(Path(args.ambiguity_labels).resolve()) if str(args.ambiguity_labels).strip() else {}
    prompt_meta = get_pair_prompt_metadata(pair_id)
    icl, _, ev = split_data_three_way(
        words=words,
        n_icl=int(args.n_icl),
        n_select=int(args.n_select),
        n_eval=int(args.n_eval),
        seed=int(args.seed),
    )
    eval_rows = _attach_labels(ev, pair_id=pair_id, labels=labels)

    cfg = get_model_config(str(args.model))
    model, tokenizer = load_model(str(args.model), device=str(args.device))
    results = _evaluate_rows(
        model,
        tokenizer,
        eval_rows=eval_rows,
        icl_examples=icl,
        prompt_meta=prompt_meta,
        prompt_variant=str(args.prompt_variant),
        device=str(args.device),
        max_new_tokens=int(args.max_new_tokens),
    )

    sample_word = eval_rows[0] if eval_rows else (icl[0] if icl else None)
    prompt_packet = {}
    if sample_word is not None:
        prompts = _prompt_texts(
            sample_word,
            icl_examples=icl,
            prompt_meta=prompt_meta,
            prompt_variant=str(args.prompt_variant),
        )
        prompt_packet = {
            "prompt_template": prompt_template_fingerprint(tokenizer),
            **{
                name: prompt_fingerprint(
                    raw_prompt=prompt,
                    rendered_prompt=apply_chat_template(tokenizer, prompt),
                )
                for name, prompt in prompts.items()
            },
        }

    full_gap = _gap_block(results)
    unambiguous_items = _slice_items(results, ambiguity="unambiguous")
    unambiguous_gap = _gap_block(unambiguous_items)

    summary = {
        "full_set": {
            cond: _summarize_condition(results, condition=cond)
            for cond in ("bare_zs", "explicit_zs", "icl8", "icl64")
        },
        "unambiguous_slice": {
            "available": bool(unambiguous_items),
            "n": int(len(unambiguous_items)),
            "conditions": {
                cond: _summarize_condition(unambiguous_items, condition=cond)
                for cond in ("bare_zs", "explicit_zs", "icl8", "icl64")
            },
        },
        "gaps": {
            "full_set": full_gap,
            "unambiguous_slice": unambiguous_gap,
        },
        "runbook_gate": {
            "metric": "paired bootstrap CI excluding zero on at least one primary behavioral metric",
            "full_set_clearly_above_floor_noise": bool(full_gap.get("clearly_above_floor_noise", False)),
            "unambiguous_slice_clearly_above_floor_noise": bool(
                unambiguous_gap.get("clearly_above_floor_noise", False)
            ),
        },
    }

    payload = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "paper": "paper2_fidelity_calibrated",
        "stage": "stage0_premise_gate",
        "pair": pair_id,
        "model_key": str(args.model),
        "pair_meta": dict(prompt_meta),
        "runtime_identity": runtime_identity(
            model_key=str(args.model),
            hf_id=cfg.hf_id,
            tokenizer=tokenizer,
            model=model,
        ),
        "prompt_packet": prompt_packet,
        "split_sizes": {"icl": len(icl), "eval_blind": len(ev)},
        "provenance": provenance,
        "ambiguity_labels_present": bool(labels),
        "summary": summary,
        "items": results,
    }

    out_path = (
        Path(args.out).resolve()
        if str(args.out).strip()
        else Path(__file__).resolve().parent / "results" / pair_id / str(args.model) / "stage0_premise_gate.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
