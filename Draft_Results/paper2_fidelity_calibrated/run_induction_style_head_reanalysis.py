#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_model_config  # noqa: E402
from core import (  # noqa: E402
    apply_chat_template,
    build_corrupted_icl_prompt,
    load_model,
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
    ap = argparse.ArgumentParser(description="Induction-style matched-example reanalysis of top causal heads.")
    ap.add_argument("--model", type=str, default="4b", choices=["1b", "4b"])
    ap.add_argument("--pair", type=str, default="aksharantar_hin_latin")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-icl", type=int, default=64)
    ap.add_argument("--n-select", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--max-items", type=int, default=30)
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
    multilang = PROJECT_ROOT / "artifacts" / "phase5_attribution" / f"top_heads_{model_key}_{_lang_from_pair(pair_id)}_multilang.json"
    if multilang.exists():
        return multilang
    return PROJECT_ROOT / "artifacts" / "phase5_attribution" / f"top_heads_{model_key}_{_lang_from_pair(pair_id)}.json"


def _find_first_subsequence_after(haystack: Sequence[int], needle: Sequence[int], start: int) -> Optional[Tuple[int, int]]:
    if not needle or len(needle) > len(haystack):
        return None
    n = len(needle)
    for idx in range(max(0, int(start)), len(haystack) - n + 1):
        if list(haystack[idx : idx + n]) == list(needle):
            return (idx, idx + n)
    return None


def _similarity(a: str, b: str) -> float:
    return float(SequenceMatcher(a=str(a), b=str(b)).ratio())


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


def _extract_example_spans(*, tokenizer: Any, rendered_prompt: str, icl_examples: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    prompt_ids = tokenizer(rendered_prompt, return_tensors="pt")["input_ids"][0].tolist()
    spans: List[Dict[str, Any]] = []
    cursor = 0
    for idx, ex in enumerate(icl_examples):
        src_text = str(ex.get("ood", "")).strip()
        tgt_text = str(ex.get("hindi", "")).strip()
        src_ids = tokenizer.encode(src_text, add_special_tokens=False) if src_text else []
        tgt_ids = tokenizer.encode(tgt_text, add_special_tokens=False) if tgt_text else []
        src_span = _find_first_subsequence_after(prompt_ids, src_ids, cursor) if src_ids else None
        if src_span is not None:
            cursor = src_span[1]
        tgt_span = _find_first_subsequence_after(prompt_ids, tgt_ids, cursor) if tgt_ids else None
        if tgt_span is not None:
            cursor = tgt_span[1]
        spans.append(
            {
                "example_index": int(idx),
                "source_text": src_text,
                "target_text": tgt_text,
                "source_span": src_span,
                "target_span": tgt_span,
            }
        )
    return spans


def _build_corrupted_examples(
    *,
    icl_examples: List[Dict[str, str]],
    input_script_name: str,
    output_script_name: str,
    seed: int,
) -> List[Dict[str, str]]:
    outs = [str(ex.get("hindi", "") or "") for ex in icl_examples]
    n = len(outs)
    if n <= 1:
        return [dict(ex) for ex in icl_examples]
    msg = f"derange::{int(seed)}::{n}::{input_script_name}::{output_script_name}".encode("utf-8")
    seed32 = int.from_bytes(hashlib.sha256(msg).digest()[:4], "little", signed=False)
    rng = np.random.default_rng(seed32)
    perm = None
    for _ in range(100):
        cand = rng.permutation(n).tolist()
        if all(int(cand[i]) != i for i in range(n)):
            perm = [int(x) for x in cand]
            break
    if perm is None:
        perm = list(range(1, n)) + [0]
    corrupted: List[Dict[str, str]] = []
    for ex, j in zip(icl_examples, perm):
        ex2 = dict(ex)
        ex2["hindi"] = outs[int(j)]
        corrupted.append(ex2)
    return corrupted


def _span_mass(dist: torch.Tensor, span: Optional[Tuple[int, int]]) -> float:
    if span is None:
        return 0.0
    s, e = int(span[0]), int(span[1])
    s = max(0, s)
    e = min(int(dist.shape[0]), e)
    if e <= s:
        return 0.0
    return float(dist[s:e].sum().item())


def _rank_of_index(sorted_indices: List[int], target_idx: int) -> float:
    try:
        return float(sorted_indices.index(int(target_idx)) + 1)
    except ValueError:
        return float("nan")


def _extract_rows(
    *,
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    heads: List[Dict[str, Any]],
    qpos: int,
    query_text: str,
    icl_examples: List[Dict[str, str]],
    rendered_prompt: str,
    condition: str,
    item_index: int,
    word: Dict[str, str],
) -> List[Dict[str, Any]]:
    example_spans = _extract_example_spans(tokenizer=tokenizer, rendered_prompt=rendered_prompt, icl_examples=icl_examples)
    similarities = [float(_similarity(query_text, str(ex.get("ood", "")))) for ex in icl_examples]
    if similarities:
        matched_idx = int(np.argmax(np.array(similarities, dtype=np.float64)))
        ranked_by_similarity = [
            int(i)
            for i in np.argsort(np.array([-s for s in similarities], dtype=np.float64)).tolist()
        ]
    else:
        matched_idx = -1
        ranked_by_similarity = []

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
        source_masses = [float(_span_mass(dist, ex.get("source_span"))) for ex in example_spans]
        target_masses = [float(_span_mass(dist, ex.get("target_span"))) for ex in example_spans]
        top_source_idx = int(np.argmax(np.array(source_masses, dtype=np.float64))) if source_masses else -1
        top_target_idx = int(np.argmax(np.array(target_masses, dtype=np.float64))) if target_masses else -1
        matched_source_mass = float(source_masses[matched_idx]) if matched_idx >= 0 and matched_idx < len(source_masses) else float("nan")
        matched_target_mass = float(target_masses[matched_idx]) if matched_idx >= 0 and matched_idx < len(target_masses) else float("nan")
        paired_target_mass_at_top_source = float(target_masses[top_source_idx]) if top_source_idx >= 0 and top_source_idx < len(target_masses) else float("nan")
        other_target_masses = [
            float(v) for i, v in enumerate(target_masses) if int(i) != int(top_source_idx)
        ]
        paired_target_margin = (
            float(paired_target_mass_at_top_source - float(np.nanmean(other_target_masses)))
            if other_target_masses and np.isfinite(paired_target_mass_at_top_source)
            else float("nan")
        )
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
                "matched_example_index": int(matched_idx),
                "matched_similarity": float(similarities[matched_idx]) if matched_idx >= 0 else float("nan"),
                "matched_source_mass": matched_source_mass,
                "matched_target_mass": matched_target_mass,
                "matched_pair_mass": float(matched_source_mass + matched_target_mass) if np.isfinite(matched_source_mass) and np.isfinite(matched_target_mass) else float("nan"),
                "top_source_example_index": int(top_source_idx),
                "top_target_example_index": int(top_target_idx),
                "top_source_similarity_rank": float(_rank_of_index(ranked_by_similarity, top_source_idx)) if top_source_idx >= 0 else float("nan"),
                "top_target_similarity_rank": float(_rank_of_index(ranked_by_similarity, top_target_idx)) if top_target_idx >= 0 else float("nan"),
                "top_source_mass": float(source_masses[top_source_idx]) if top_source_idx >= 0 else float("nan"),
                "top_target_mass": float(target_masses[top_target_idx]) if top_target_idx >= 0 else float("nan"),
                "source_target_same_example": float(int(top_source_idx == top_target_idx)) if top_source_idx >= 0 and top_target_idx >= 0 else float("nan"),
                "paired_target_mass_at_top_source": paired_target_mass_at_top_source,
                "paired_target_margin_at_top_source": paired_target_margin,
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
        except Exception:
            pass

    stagea_path = resolve_stagea_path(str(args.pair), str(args.model), str(args.stagea))
    if not stagea_path.exists():
        raise FileNotFoundError(f"Missing Stage A artifact: {stagea_path}")
    stagea_best = load_stagea_best(stagea_path, seed=int(args.seed))
    stagea_best["scope_repo"] = str(cfg.scope_repo)

    top_heads_path = Path(str(args.top_heads_json)).resolve() if str(args.top_heads_json).strip() else _default_top_heads_path(str(args.pair), str(args.model))
    if not top_heads_path.exists():
        raise FileNotFoundError(f"Missing top-heads artifact: {top_heads_path}")
    top_heads = _load_top_heads(top_heads_path, top_n=int(args.top_n_heads))

    transcoder = load_transcoder_for_stagea(model, {**stagea_best, "scope_repo": str(cfg.scope_repo)}, device)

    out_root = (
        Path(str(args.out)).resolve()
        if str(args.out).strip()
        else PROJECT_ROOT / "paper2_fidelity_calibrated" / "results" / "induction_style_head_reanalysis" / str(args.pair) / str(args.model)
    )
    out_root.mkdir(parents=True, exist_ok=True)

    eval_rows = list(pair_bundle["eval_rows"][: max(1, int(args.max_items))])
    if not eval_rows:
        raise RuntimeError("No evaluation rows available for induction-style head reanalysis")

    item_rows: List[Dict[str, Any]] = []
    log(
        f"Running induction-style head reanalysis: pair={args.pair} model={args.model} heads={[(h['layer'], h['head']) for h in top_heads]}"
    )

    for item_idx, word in enumerate(eval_rows, start=1):
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
        qpos_icl = int(packet["icl_input_ids"].shape[1] - 1)
        item_rows.extend(
            _extract_rows(
                model=model,
                tokenizer=tokenizer,
                input_ids=packet["icl_input_ids"],
                heads=top_heads,
                qpos=qpos_icl,
                query_text=str(word["ood"]),
                icl_examples=pair_bundle["icl_examples"],
                rendered_prompt=packet["icl_rendered"],
                condition="icl64",
                item_index=int(item_idx - 1),
                word=word,
            )
        )

        corrupted_examples = _build_corrupted_examples(
            icl_examples=pair_bundle["icl_examples"],
            input_script_name=pair_bundle["input_script_name"],
            output_script_name=pair_bundle["output_script_name"],
            seed=int(args.seed),
        )
        corrupt_prompt = build_corrupted_icl_prompt(
            str(word["ood"]),
            pair_bundle["icl_examples"],
            input_script_name=pair_bundle["input_script_name"],
            source_language=pair_bundle["source_language"],
            output_script_name=pair_bundle["output_script_name"],
            seed=int(args.seed),
        )
        corrupt_rendered = apply_chat_template(tokenizer, corrupt_prompt)
        corrupt_input_ids = tokenizer(corrupt_rendered, return_tensors="pt").to(device).input_ids
        qpos_corrupt = int(corrupt_input_ids.shape[1] - 1)
        item_rows.extend(
            _extract_rows(
                model=model,
                tokenizer=tokenizer,
                input_ids=corrupt_input_ids,
                heads=top_heads,
                qpos=qpos_corrupt,
                query_text=str(word["ood"]),
                icl_examples=corrupted_examples,
                rendered_prompt=corrupt_rendered,
                condition="corrupt_icl",
                item_index=int(item_idx - 1),
                word=word,
            )
        )

    head_summary: List[Dict[str, Any]] = []
    keys = sorted({(str(r["condition"]), int(r["layer"]), int(r["head"])) for r in item_rows})
    for condition, layer, head in keys:
        rows = [r for r in item_rows if str(r["condition"]) == condition and int(r["layer"]) == layer and int(r["head"]) == head]
        if not rows:
            continue
        def _m(name: str) -> float:
            return float(np.nanmean([float(r.get(name, float("nan"))) for r in rows]))
        head_summary.append(
            {
                "condition": str(condition),
                "layer": int(layer),
                "head": int(head),
                "n_items": int(len(rows)),
                "mean_matched_source_mass": _m("matched_source_mass"),
                "mean_matched_target_mass": _m("matched_target_mass"),
                "mean_matched_pair_mass": _m("matched_pair_mass"),
                "mean_top_source_similarity_rank": _m("top_source_similarity_rank"),
                "mean_top_target_similarity_rank": _m("top_target_similarity_rank"),
                "mean_source_target_same_example": _m("source_target_same_example"),
                "mean_paired_target_mass_at_top_source": _m("paired_target_mass_at_top_source"),
                "mean_paired_target_margin_at_top_source": _m("paired_target_margin_at_top_source"),
            }
        )

    payload = {
        "experiment": "induction_style_head_reanalysis",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pair": str(args.pair),
        "model": str(args.model),
        "seed": int(args.seed),
        "stagea_best": stagea_best,
        "top_heads_path": str(top_heads_path),
        "top_heads": top_heads,
        "n_items": int(len(eval_rows)),
        "head_summary": head_summary,
        "item_rows": item_rows,
        "notes": {
            "framing": "Descriptive matched-example routing check inspired by induction-head-style retrieval, not a proof that these heads are canonical induction heads.",
            "conditions": ["icl64", "corrupt_icl"],
        },
    }
    _write_json(out_root / "induction_style_head_reanalysis.json", payload)
    log(f"Saved: {out_root / 'induction_style_head_reanalysis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
