#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import load_model, set_all_seeds, split_data_three_way  # noqa: E402
from paper2_fidelity_calibrated.run import _load_words, _prompt_naming  # noqa: E402
from paper2_fidelity_calibrated.run_attribution_graph_pair import (  # noqa: E402
    _run_attention_attribution,
    log,
)
from rescue_research.data_pipeline.ingest import get_pair_prompt_metadata  # noqa: E402


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
    ap = argparse.ArgumentParser(description="Resample stability of attention-head attribution rankings.")
    ap.add_argument("--pair", required=True)
    ap.add_argument("--model", default="4b", choices=["1b", "4b"])
    ap.add_argument("--batch-words", type=int, default=10)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--resamples", type=int, default=6)
    ap.add_argument("--subset-size", type=int, default=10)
    ap.add_argument("--n-icl", type=int, default=64)
    ap.add_argument("--n-select", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--external-only", action="store_true")
    ap.add_argument("--require-external-sources", action="store_true")
    ap.add_argument("--min-pool-size", type=int, default=500)
    ap.add_argument("--out", type=str, default="")
    return ap.parse_args()


def _lang_from_pair(pair_id: str) -> str:
    parts = str(pair_id).split("_")
    return parts[1] if len(parts) >= 2 else str(pair_id)


def _default_reference_top_heads_path(pair_id: str, model_key: str) -> Optional[Path]:
    lang = _lang_from_pair(pair_id)
    candidates = [
        PROJECT_ROOT / "artifacts" / "phase5_attribution" / f"top_heads_{model_key}_{lang}_multilang.json",
        PROJECT_ROOT / "artifacts" / "phase5_attribution" / f"top_heads_{model_key}_{lang}.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _jaccard_heads(a: Set[Tuple[int, int]], b: Set[Tuple[int, int]]) -> float:
    union = len(a | b)
    if union <= 0:
        return float("nan")
    return float(len(a & b) / union)


def _top_heads_from_scores(scores: torch.Tensor, *, topk: int) -> List[Dict[str, Any]]:
    if scores.ndim != 2 or int(scores.shape[1]) <= 0:
        raise ValueError(f"Expected 2D [layers, heads] attribution scores, got shape={tuple(scores.shape)}")
    flat_scores = scores.flatten()
    k = min(max(1, int(topk)), int(flat_scores.numel()))
    top_indices = torch.topk(flat_scores, k).indices
    num_heads = int(scores.shape[1])
    out: List[Dict[str, Any]] = []
    for rank, flat_idx in enumerate(top_indices.tolist(), start=1):
        layer = int(flat_idx) // int(num_heads)
        head = int(flat_idx) % int(num_heads)
        value = float(flat_scores[int(flat_idx)].item())
        out.append({"rank": int(rank), "layer": int(layer), "head": int(head), "effect": float(value)})
    return out


def _read_reference_set(path: Optional[Path], *, topk: int) -> Tuple[Optional[str], Optional[Set[Tuple[int, int]]]]:
    if path is None or not path.exists():
        return None, None
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return str(path), None
    top_rows = list(rows[: max(1, int(topk))])
    ref_set: Set[Tuple[int, int]] = set()
    for row in top_rows:
        try:
            ref_set.add((int(row["layer"]), int(row["head"])))
        except Exception:
            continue
    return str(path), ref_set if ref_set else None


def main() -> int:
    args = parse_args()
    set_all_seeds(int(args.seed))

    model, tokenizer = load_model(str(args.model), device=str(args.device))
    device = str(next(model.parameters()).device)

    prompt_meta = dict(get_pair_prompt_metadata(str(args.pair)))
    source_lang, input_script, output_script = _prompt_naming(prompt_meta)
    words, provenance = _load_words(
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
    eval_rows = list(eval_rows)
    if not eval_rows:
        raise RuntimeError("No evaluation rows available for attribution-stability analysis")

    out_root = (
        Path(str(args.out)).resolve()
        if str(args.out).strip()
        else PROJECT_ROOT / "paper2_fidelity_calibrated" / "results" / "head_attribution_stability" / str(args.pair) / str(args.model)
    )
    out_root.mkdir(parents=True, exist_ok=True)

    reference_path, reference_set = _read_reference_set(
        _default_reference_top_heads_path(str(args.pair), str(args.model)),
        topk=int(args.topk),
    )

    rng = random.Random(int(args.seed))
    resamples_detail: List[Dict[str, Any]] = []
    head_sets: List[Set[Tuple[int, int]]] = []
    head_frequency: Dict[Tuple[int, int], int] = {}
    ref_overlaps: List[float] = []

    log(
        f"Running head-attribution stability: pair={args.pair} model={args.model} "
        f"resamples={args.resamples} subset_size={args.subset_size} topk={args.topk}"
    )

    for resample_idx in range(max(1, int(args.resamples))):
        if len(eval_rows) <= int(args.subset_size):
            sample = list(eval_rows)
        else:
            sample = rng.sample(eval_rows, int(args.subset_size))
        scores = _run_attention_attribution(
            model,
            tokenizer,
            sample,
            icl_examples,
            source_lang,
            input_script,
            output_script,
            device,
            batch_words=min(int(args.batch_words), len(sample)),
        )
        top_heads = _top_heads_from_scores(scores, topk=int(args.topk))
        head_set = {(int(row["layer"]), int(row["head"])) for row in top_heads}
        head_sets.append(head_set)
        for key in head_set:
            head_frequency[key] = head_frequency.get(key, 0) + 1
        if reference_set is not None:
            ref_overlaps.append(float(_jaccard_heads(head_set, reference_set)))
        resamples_detail.append(
            {
                "resample_index": int(resample_idx),
                "n_items": int(len(sample)),
                "top_heads": top_heads,
                "sample_words": [str(row.get("ood", "")) for row in sample],
            }
        )

    pairwise_jaccard: List[Dict[str, Any]] = []
    for i in range(len(head_sets)):
        for j in range(i + 1, len(head_sets)):
            pairwise_jaccard.append({"i": int(i), "j": int(j), "jaccard": float(_jaccard_heads(head_sets[i], head_sets[j]))})

    stable_heads = [
        {"layer": int(layer), "head": int(head), "count": int(count), "frequency": float(count / max(1, len(head_sets)))}
        for (layer, head), count in sorted(head_frequency.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
    ]

    payload = {
        "experiment": "head_attribution_stability",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pair": str(args.pair),
        "model": str(args.model),
        "seed": int(args.seed),
        "resamples": int(args.resamples),
        "subset_size": int(args.subset_size),
        "topk": int(args.topk),
        "batch_words": int(args.batch_words),
        "provenance": provenance,
        "reference_top_heads_path": reference_path,
        "mean_pairwise_jaccard": float(np.nanmean([row["jaccard"] for row in pairwise_jaccard])) if pairwise_jaccard else float("nan"),
        "mean_reference_overlap_jaccard": float(np.nanmean(ref_overlaps)) if ref_overlaps else float("nan"),
        "pairwise_jaccard": pairwise_jaccard,
        "stable_heads": stable_heads,
        "resamples_detail": resamples_detail,
    }
    _write_json(out_root / "head_attribution_stability.json", payload)
    log(f"Saved: {out_root / 'head_attribution_stability.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
