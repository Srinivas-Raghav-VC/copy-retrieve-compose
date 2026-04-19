#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_model_config  # noqa: E402
from core import load_model, set_all_seeds  # noqa: E402
from paper2_fidelity_calibrated.phase1_common import (  # noqa: E402
    extract_feature_delta_vector,
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
    ap = argparse.ArgumentParser(description="Resampling stability of top-ranked Stage-A features.")
    ap.add_argument("--model", type=str, default="4b", choices=["1b", "4b"])
    ap.add_argument("--pair", type=str, default="aksharantar_hin_latin")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-icl", type=int, default=64)
    ap.add_argument("--n-select", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--resamples", type=int, default=8)
    ap.add_argument("--subset-size", type=int, default=100)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--stagea", type=str, default="")
    ap.add_argument("--external-only", action="store_true")
    ap.add_argument("--require-external-sources", action="store_true")
    ap.add_argument("--min-pool-size", type=int, default=500)
    ap.add_argument("--out", type=str, default="")
    return ap.parse_args()


def _topk_indices(vec: torch.Tensor, k: int) -> List[int]:
    return [int(x) for x in torch.topk(torch.abs(vec), k=min(int(k), int(vec.numel()))).indices.tolist()]


def _jaccard(a: Set[int], b: Set[int]) -> float:
    union = len(a | b)
    if union <= 0:
        return float("nan")
    return float(len(a & b) / union)


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

    stagea_path = resolve_stagea_path(str(args.pair), str(args.model), str(args.stagea))
    if not stagea_path.exists():
        raise FileNotFoundError(f"Missing Stage A artifact: {stagea_path}")
    stagea_best = load_stagea_best(stagea_path, seed=int(args.seed))
    stagea_best["scope_repo"] = str(cfg.scope_repo)
    transcoder = load_transcoder_for_stagea(model, stagea_best, device)

    select_rows = list(pair_bundle["select_rows"])
    if not select_rows:
        raise RuntimeError("No selection rows available for stability analysis")

    out_root = (
        Path(args.out).resolve()
        if str(args.out).strip()
        else PROJECT_ROOT / "paper2_fidelity_calibrated" / "results" / "feature_stability_resamples" / str(args.pair) / str(args.model)
    )
    out_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(int(args.seed))
    resample_rows: List[Dict[str, Any]] = []
    top_sets: List[Set[int]] = []

    log(
        f"Running feature-stability resamples: pair={args.pair} model={args.model} resamples={args.resamples} subset_size={args.subset_size} topk={args.topk}"
    )

    for resample_idx in range(max(1, int(args.resamples))):
        sample = select_rows if len(select_rows) <= int(args.subset_size) else rng.sample(select_rows, int(args.subset_size))
        abs_vals: List[torch.Tensor] = []
        for word in sample:
            delta = extract_feature_delta_vector(
                model=model,
                tokenizer=tokenizer,
                transcoder=transcoder,
                layer=int(stagea_best["layer"]),
                word=word,
                icl_examples=pair_bundle["icl_examples"],
                input_script_name=pair_bundle["input_script_name"],
                source_language=pair_bundle["source_language"],
                output_script_name=pair_bundle["output_script_name"],
                prompt_variant=str(stagea_best.get("prompt_variant", "canonical")),
                selector_reference=str(stagea_best.get("selector_reference", "zs")),
                patch_position_mode=str(stagea_best.get("patch_position_mode", "source_last_subtoken")),
                device=device,
                seed=int(args.seed) + int(resample_idx),
            )
            abs_vals.append(torch.abs(delta.detach().float().cpu()))
        mean_abs = torch.stack(abs_vals, dim=0).mean(dim=0)
        top_idx = _topk_indices(mean_abs, int(args.topk))
        top_set = {int(i) for i in top_idx}
        top_sets.append(top_set)
        resample_rows.append(
            {
                "resample_index": int(resample_idx),
                "n_items": int(len(sample)),
                "top_feature_indices": list(top_idx),
            }
        )

    feature_freq: Dict[int, int] = {}
    for top_set in top_sets:
        for idx in top_set:
            feature_freq[int(idx)] = feature_freq.get(int(idx), 0) + 1

    pairwise_jaccard: List[Dict[str, Any]] = []
    for i in range(len(top_sets)):
        for j in range(i + 1, len(top_sets)):
            pairwise_jaccard.append(
                {
                    "i": int(i),
                    "j": int(j),
                    "jaccard": float(_jaccard(top_sets[i], top_sets[j])),
                }
            )

    payload = {
        "experiment": "feature_stability_resamples",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": str(args.model),
        "pair": str(args.pair),
        "seed": int(args.seed),
        "resamples": int(args.resamples),
        "subset_size": int(args.subset_size),
        "topk": int(args.topk),
        "stagea_best": stagea_best,
        "resamples_detail": resample_rows,
        "feature_frequency": [{"feature_index": int(k), "count": int(v)} for k, v in sorted(feature_freq.items(), key=lambda kv: (-kv[1], kv[0]))],
        "pairwise_jaccard": pairwise_jaccard,
        "mean_pairwise_jaccard": float(np.nanmean([row["jaccard"] for row in pairwise_jaccard])) if pairwise_jaccard else float("nan"),
    }
    _write_json(out_root / "feature_stability_resamples.json", payload)
    log(f"Saved: {out_root / 'feature_stability_resamples.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
