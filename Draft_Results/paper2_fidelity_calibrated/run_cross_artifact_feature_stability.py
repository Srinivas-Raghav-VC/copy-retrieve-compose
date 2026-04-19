#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

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
    ap = argparse.ArgumentParser(description="Compare ranked features across transcoder variants/artifacts.")
    ap.add_argument("--model", type=str, default="4b", choices=["1b", "4b"])
    ap.add_argument("--pair", type=str, default="aksharantar_hin_latin")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-icl", type=int, default=64)
    ap.add_argument("--n-select", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--stats-items", type=int, default=100)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--variants", type=str, default="skipless_or_non_affine,affine_skip")
    ap.add_argument("--stagea", type=str, default="")
    ap.add_argument("--external-only", action="store_true")
    ap.add_argument("--require-external-sources", action="store_true")
    ap.add_argument("--min-pool-size", type=int, default=500)
    ap.add_argument("--out", type=str, default="")
    return ap.parse_args()


def _parse_variants(raw: str) -> List[str]:
    vals = [str(x).strip() for x in str(raw or "").split(",") if str(x).strip()]
    return vals or ["skipless_or_non_affine", "affine_skip"]


def _topk(vec: torch.Tensor, k: int) -> List[int]:
    return [int(x) for x in torch.topk(torch.abs(vec), k=min(int(k), int(vec.numel()))).indices.tolist()]


def _jaccard(a: List[int], b: List[int]) -> float:
    sa = set(int(x) for x in a)
    sb = set(int(x) for x in b)
    union = len(sa | sb)
    return float(len(sa & sb) / union) if union > 0 else float("nan")


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    na = float(torch.norm(a).item())
    nb = float(torch.norm(b).item())
    if not np.isfinite(na) or not np.isfinite(nb) or na <= 0.0 or nb <= 0.0:
        return float("nan")
    return float(torch.dot(a, b).item() / max(1e-12, na * nb))


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
    stagea_best = load_stagea_best(stagea_path, seed=int(args.seed))
    stagea_best["scope_repo"] = str(cfg.scope_repo)
    variants = _parse_variants(str(args.variants))

    rows = list(pair_bundle["select_rows"][: max(1, int(args.stats_items))])
    mean_abs_by_variant: Dict[str, torch.Tensor] = {}
    top_features_by_variant: Dict[str, List[int]] = {}

    log(f"Running cross-artifact feature stability: pair={args.pair} model={args.model} variants={variants} stats_items={len(rows)}")

    for variant in variants:
        best = dict(stagea_best)
        best["variant"] = str(variant)
        transcoder = load_transcoder_for_stagea(model, best, device)
        vals: List[torch.Tensor] = []
        for word in rows:
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
                seed=int(args.seed),
            )
            vals.append(torch.abs(delta.detach().float().cpu()))
        mean_abs = torch.stack(vals, dim=0).mean(dim=0)
        mean_abs_by_variant[str(variant)] = mean_abs
        top_features_by_variant[str(variant)] = _topk(mean_abs, int(args.topk))

    summary: Dict[str, Any] = {
        "top_features_by_variant": top_features_by_variant,
    }
    if len(variants) >= 2:
        for i in range(len(variants)):
            for j in range(i + 1, len(variants)):
                va, vb = str(variants[i]), str(variants[j])
                summary[f"jaccard__{va}__{vb}"] = float(_jaccard(top_features_by_variant[va], top_features_by_variant[vb]))
                summary[f"mean_abs_cosine__{va}__{vb}"] = float(_cos(mean_abs_by_variant[va], mean_abs_by_variant[vb]))

    payload = {
        "experiment": "cross_artifact_feature_stability",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": str(args.model),
        "pair": str(args.pair),
        "seed": int(args.seed),
        "variants": variants,
        "stagea_best": stagea_best,
        "summary": summary,
    }
    _write_json(
        (
            Path(args.out).resolve()
            if str(args.out).strip()
            else PROJECT_ROOT / "paper2_fidelity_calibrated" / "results" / "cross_artifact_feature_stability" / str(args.pair) / str(args.model) / "cross_artifact_feature_stability.json"
        ),
        payload,
    )
    log("Saved cross-artifact feature stability")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
