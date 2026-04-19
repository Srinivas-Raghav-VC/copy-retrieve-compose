#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_model_config  # noqa: E402
from core import load_model, run_patching_experiment, set_all_seeds  # noqa: E402
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
    ap = argparse.ArgumentParser(description="Minimality curve over G6-ranked core features.")
    ap.add_argument("--model", type=str, default="4b", choices=["1b", "4b"])
    ap.add_argument("--pair", type=str, default="aksharantar_hin_latin")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-icl", type=int, default=64)
    ap.add_argument("--n-select", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--max-items", type=int, default=20)
    ap.add_argument("--counts", type=str, default="1,2,4,8")
    ap.add_argument("--stagea", type=str, default="")
    ap.add_argument("--feature-knockout-json", type=str, default="")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--external-only", action="store_true")
    ap.add_argument("--require-external-sources", action="store_true")
    ap.add_argument("--min-pool-size", type=int, default=500)
    ap.add_argument("--out", type=str, default="")
    return ap.parse_args()


def _parse_counts(raw: str) -> List[int]:
    vals = sorted(set(int(x.strip()) for x in str(raw or "").split(",") if x.strip()))
    return vals or [1, 2, 4, 8]


def _resolve_feature_knockout_path(pair_id: str, model_key: str, raw: str) -> Path:
    if str(raw).strip():
        return Path(str(raw)).resolve()
    return PROJECT_ROOT / "paper2_fidelity_calibrated" / "results" / "feature_knockout_panel" / str(pair_id) / str(model_key) / "feature_knockout_panel.json"


def _load_ranked_features(path: Path) -> List[int]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    rows = list(obj.get("summary_by_feature_index") or [])
    rows.sort(key=lambda row: float(row.get("mean_drop_from_full_patch_first_prob", float("-inf"))), reverse=True)
    out = [int(row["feature_index"]) for row in rows]
    if not out:
        raise RuntimeError(f"No ranked features in {path}")
    return out


def _random_matched(reference: List[int], dim: int, *, seed: int) -> List[int]:
    pool = [i for i in range(int(dim)) if int(i) not in set(int(x) for x in reference)]
    rng = random.Random(int(seed))
    rng.shuffle(pool)
    return sorted(pool[: len(reference)])


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

    feature_knockout_path = _resolve_feature_knockout_path(str(args.pair), str(args.model), str(args.feature_knockout_json))
    if not feature_knockout_path.exists():
        raise FileNotFoundError(f"Missing feature-knockout artifact: {feature_knockout_path}")
    ranked_features = _load_ranked_features(feature_knockout_path)
    counts = _parse_counts(str(args.counts))

    out_root = (
        Path(args.out).resolve()
        if str(args.out).strip()
        else PROJECT_ROOT / "paper2_fidelity_calibrated" / "results" / "minimality_curve" / str(args.pair) / str(args.model)
    )
    out_root.mkdir(parents=True, exist_ok=True)

    eval_rows = list(pair_bundle["eval_rows"][: max(1, int(args.max_items))])
    item_rows: List[Dict[str, Any]] = []

    packet0 = build_patch_packet(
        model=model,
        tokenizer=tokenizer,
        transcoder=transcoder,
        word=eval_rows[0],
        icl_examples=pair_bundle["icl_examples"],
        stagea_best=stagea_best,
        input_script_name=pair_bundle["input_script_name"],
        source_language=pair_bundle["source_language"],
        output_script_name=pair_bundle["output_script_name"],
        device=device,
    )
    latent_dim = int(packet0["icl_feats"].numel())

    log(
        f"Running minimality curve: pair={args.pair} model={args.model} items={len(eval_rows)} counts={counts}"
    )

    for item_idx, word in enumerate(eval_rows, start=1):
        log(f"[{item_idx}/{len(eval_rows)}] {word['ood']} -> {word['hindi']}")
        for count in counts:
            selected = list(ranked_features[: max(1, int(count))])
            random_selected = _random_matched(selected, latent_dim, seed=int(args.seed) + int(item_idx) + int(count) * 101)
            for label, chosen in (("core", selected), ("random", random_selected)):
                idx_override = torch.tensor(chosen, device=device, dtype=torch.long)
                result = run_patching_experiment(
                    model,
                    tokenizer,
                    transcoder,
                    int(stagea_best["layer"]),
                    word,
                    pair_bundle["icl_examples"],
                    topk=max(1, len(chosen)),
                    device=device,
                    seed=int(args.seed),
                    input_script_name=pair_bundle["input_script_name"],
                    source_language=pair_bundle["source_language"],
                    output_script_name=pair_bundle["output_script_name"],
                    patch_style=str(stagea_best.get("patch_style", "sparse")),
                    feature_selection=str(stagea_best.get("feature_selection", "topk_abs_delta")),
                    idx_override=idx_override,
                    prompt_variant=str(stagea_best.get("prompt_variant", "canonical")),
                    selector_reference_mode=str(stagea_best.get("selector_reference", "zs")),
                    require_query_span_match=bool(stagea_best.get("require_query_span_match", False)),
                    use_norm_matching=bool(stagea_best.get("norm_matching", True)),
                    eval_generation=True,
                    max_new_tokens=int(args.max_new_tokens),
                )
                row = result.to_dict()
                row.update(
                    {
                        "pair": str(args.pair),
                        "model": str(args.model),
                        "seed": int(args.seed),
                        "item_index": int(item_idx - 1),
                        "word_ood": str(word["ood"]),
                        "word_hindi": str(word["hindi"]),
                        "condition": str(label),
                        "count": int(count),
                        "idx_override": list(chosen),
                    }
                )
                item_rows.append(row)

    summary: Dict[str, Dict[str, float]] = {}
    for label in ("core", "random"):
        for count in counts:
            rows = [row for row in item_rows if str(row.get("condition")) == label and int(row.get("count")) == int(count)]
            summary[f"{label}::{count}"] = {
                "n_items": float(len(rows)),
                "mean_prob_patched_first": float(np.nanmean([float(row.get("prob_patched_first", float("nan"))) for row in rows])),
                "mean_pe_first": float(np.nanmean([float(row.get("pe_first", float("nan"))) for row in rows])),
                "mean_nll_pos1_patched": float(np.nanmean([float(row.get("nll_pos1_patched", float("nan"))) for row in rows])),
                "mean_exact_match": float(np.nanmean([float(str(row.get("gen_patched", "")) == str(row.get("target", ""))) for row in rows])),
            }

    payload = {
        "experiment": "minimality_curve",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": str(args.model),
        "pair": str(args.pair),
        "seed": int(args.seed),
        "counts": counts,
        "ranked_features": ranked_features,
        "summary": summary,
        "item_rows": item_rows,
    }
    _write_json(out_root / "minimality_curve.json", payload)
    log(f"Saved: {out_root / 'minimality_curve.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
