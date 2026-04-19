#!/usr/bin/env python3
"""
Paper 2 utility: measure transcoder fidelity at the *actual intervention target*.

Why this exists:
  The patch hook in `core.register_transcoder_feature_patch_hook` treats
  `decode(encode(mlp_in))` as an approximation to a component of the MLP output.
  Many existing "reconstruction" checks compare recon to `mlp_in`, which can be
  misleading for judging whether patching is well-defined.

This script captures (mlp_in, mlp_out) at a chosen layer and computes:
  - fidelity_to_mlp_out: cosine / relative error / r2 of recon vs mlp_out
  - diagnostic_to_mlp_in: cosine / relative error of recon vs mlp_in
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Ensure project root is importable when running from this subfolder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_model_config
from core import (
    apply_chat_template,
    build_task_prompt,
    get_layer_device,
    get_model_layers,
    load_model,
    load_transcoder,
    split_data_three_way,
)
from rescue_research.data_pipeline.ingest import get_pair_prompt_metadata, load_pair_records_bundle


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _parse_int_list(raw: str) -> List[int]:
    raw = str(raw or "").strip()
    if not raw:
        return []
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _safe_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    na = float(torch.norm(a).item())
    nb = float(torch.norm(b).item())
    if not (np.isfinite(na) and np.isfinite(nb)) or na <= 0.0 or nb <= 0.0:
        return float("nan")
    return float(torch.dot(a, b).item() / max(1e-12, na * nb))


def _rel_l2(err: torch.Tensor, ref: torch.Tensor) -> float:
    err = err.detach().float().reshape(-1)
    ref = ref.detach().float().reshape(-1)
    denom = float(torch.norm(ref).item())
    if not np.isfinite(denom) or denom <= 0.0:
        return float("nan")
    return float(torch.norm(err).item() / max(1e-12, denom))


def _r2(recon: torch.Tensor, target: torch.Tensor) -> float:
    recon = recon.detach().float().reshape(-1)
    target = target.detach().float().reshape(-1)
    var = float(torch.var(target, unbiased=False).item())
    if not np.isfinite(var) or var <= 0.0:
        return float("nan")
    mse = float(torch.mean((recon - target) ** 2).item())
    return float(1.0 - (mse / max(1e-12, var)))


def _get_model_layers(model):
    return get_model_layers(model)


@dataclass(frozen=True)
class FidelityRow:
    pair_id: str
    model_key: str
    variant: str
    layer: int
    n_samples: int
    mean_cos_y: float
    mean_relerr_y: float
    mean_r2_y: float
    mean_cos_x: float
    mean_relerr_x: float


def _capture_mlp_in_out_last_token(
    model,
    tokenizer,
    *,
    prompt: str,
    layer: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    text = apply_chat_template(tokenizer, prompt)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    layers = _get_model_layers(model)

    captured: Dict[str, torch.Tensor] = {}

    def hook(module, inputs_tuple, output):
        x = inputs_tuple[0]
        y = output[0] if isinstance(output, tuple) else output
        # Expect [batch, seq, d_model]
        captured["x"] = x.detach()
        captured["y"] = y.detach()

    handle = layers[int(layer)].mlp.register_forward_hook(hook)
    with torch.inference_mode():
        model(**inputs, use_cache=False)
    handle.remove()

    if "x" not in captured or "y" not in captured:
        raise RuntimeError("Failed to capture MLP (input, output).")

    x_last = captured["x"][0, -1, :]
    y_last = captured["y"][0, -1, :]
    return x_last, y_last


def _load_words(
    pair_id: str,
    *,
    external_only: bool,
    require_external_sources: bool,
    min_pool_size: int,
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    bundle = load_pair_records_bundle(pair_id, include_builtin=not bool(external_only))
    total = int(len(bundle.rows))
    source_names = [s.name for s in bundle.sources]
    external_sources = [n for n in source_names if n and n != "config_multiscript"]

    if bool(require_external_sources) and not external_sources:
        raise RuntimeError(
            f"Pair {pair_id!r} has no external sources (only builtin). "
            "Provide external data under data/transliteration/ or set --require-external-sources=0."
        )
    if int(min_pool_size) > 0 and total < int(min_pool_size):
        raise RuntimeError(f"Pair {pair_id!r} pool too small: total={total} < {int(min_pool_size)}")

    words = [{"english": r["english"], "hindi": r["source"], "ood": r["target"]} for r in bundle.rows]
    meta = {
        "pair_id": pair_id,
        "total_rows": total,
        "sources": [asdict(s) for s in bundle.sources],
        "source_counts": dict(bundle.source_counts),
    }
    return words, meta


def main() -> int:
    ap = argparse.ArgumentParser(description="Measure transcoder fidelity vs MLP output at hookpoint.")
    ap.add_argument("--model", type=str, default="4b", choices=["270m", "1b", "4b", "12b"])
    ap.add_argument("--pair", type=str, required=True)
    ap.add_argument("--variant", type=str, default="affine_skip,skipless_or_non_affine")
    ap.add_argument("--layers", type=str, default="", help="Comma-separated layers. Default: all layers.")
    ap.add_argument("--layer-step", type=int, default=1, help="If --layers is empty, test every Nth layer.")
    ap.add_argument("--n-samples", type=int, default=64)
    ap.add_argument("--n-icl", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--external-only", action="store_true", help="Exclude builtin (config_multiscript) data.")
    ap.add_argument("--require-external-sources", action="store_true")
    ap.add_argument("--min-pool-size", type=int, default=0)
    ap.add_argument("--out-csv", type=str, default="", help="Optional explicit output CSV path.")
    args = ap.parse_args()

    pair_id = str(args.pair).strip()
    model_key = str(args.model).strip()
    cfg = get_model_config(model_key)

    words, provenance = _load_words(
        pair_id,
        external_only=bool(args.external_only),
        require_external_sources=bool(args.require_external_sources),
        min_pool_size=int(args.min_pool_size),
    )

    variants = [v.strip() for v in str(args.variant).split(",") if v.strip()]
    if not variants:
        raise ValueError("No variants provided.")

    model, tokenizer = load_model(model_key, device=str(args.device))

    layers = _get_model_layers(model)
    n_layers = int(len(layers))
    chosen_layers = _parse_int_list(args.layers)
    if not chosen_layers:
        step = max(1, int(args.layer_step))
        chosen_layers = list(range(0, n_layers, step))
    chosen_layers = [l for l in chosen_layers if 0 <= int(l) < n_layers]
    if not chosen_layers:
        raise ValueError("No valid layers selected.")

    # Disjoint split: use ICL bank to build prompts; evaluate fidelity on a separate set.
    icl, sel, _ = split_data_three_way(
        words=words,
        n_icl=int(args.n_icl),
        n_select=min(int(args.n_samples), max(1, len(words) - int(args.n_icl))),
        n_eval=0,
        seed=int(args.seed),
    )
    # Pick a stable subset from selection.
    rng = np.random.default_rng(int(args.seed))
    if len(sel) > int(args.n_samples):
        idx = rng.choice(len(sel), size=int(args.n_samples), replace=False).tolist()
        sel = [sel[i] for i in idx]

    prompt_meta = get_pair_prompt_metadata(pair_id)
    input_script_name = str(prompt_meta.get("source_script", "")).strip() or "Latin"
    output_script_name = str(prompt_meta.get("target_script", "")).strip() or "Devanagari"
    source_language = str(prompt_meta.get("source_language", "")).strip() or "Hindi"

    out_root = Path(__file__).resolve().parent
    results_dir = out_root / "results"
    figs_dir = out_root / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    out_csv = Path(args.out_csv).resolve() if str(args.out_csv).strip() else (
        results_dir / f"fidelity_{pair_id}_{model_key}.csv"
    )
    out_meta = out_csv.with_suffix(".meta.json")

    rows_out: List[Dict[str, Any]] = []
    summary_rows: List[FidelityRow] = []

    log(f"Model: {model_key}  Pair: {pair_id}  Layers: {len(chosen_layers)}  Variants: {variants}")
    log(f"Selection samples for fidelity: {len(sel)}  ICL: {len(icl)}")

    for variant in variants:
        for layer in chosen_layers:
            layer_dev = get_layer_device(model, int(layer))
            tc = load_transcoder(model, cfg.scope_repo, int(layer), layer_dev, variant=str(variant))

            cos_y: List[float] = []
            rel_y: List[float] = []
            r2_y: List[float] = []
            cos_x: List[float] = []
            rel_x: List[float] = []

            for w in sel:
                prompt = build_task_prompt(
                    w["ood"],
                    icl_examples=icl,
                    input_script_name=input_script_name,
                    source_language=source_language,
                    output_script_name=output_script_name,
                )
                x, y = _capture_mlp_in_out_last_token(
                    model,
                    tokenizer,
                    prompt=prompt,
                    layer=int(layer),
                    device=str(args.device),
                )
                feats = tc.encode(x.unsqueeze(0)).squeeze(0)
                recon = tc.decode(feats.unsqueeze(0)).squeeze(0)

                cy = _safe_cos(recon, y)
                ry = _rel_l2(recon - y, y)
                r2v = _r2(recon, y)
                cx = _safe_cos(recon, x)
                rx = _rel_l2(recon - x, x)

                cos_y.append(cy)
                rel_y.append(ry)
                r2_y.append(r2v)
                cos_x.append(cx)
                rel_x.append(rx)

            row = FidelityRow(
                pair_id=pair_id,
                model_key=model_key,
                variant=str(variant),
                layer=int(layer),
                n_samples=int(len(sel)),
                mean_cos_y=float(np.nanmean(cos_y)),
                mean_relerr_y=float(np.nanmean(rel_y)),
                mean_r2_y=float(np.nanmean(r2_y)),
                mean_cos_x=float(np.nanmean(cos_x)),
                mean_relerr_x=float(np.nanmean(rel_x)),
            )
            summary_rows.append(row)

            rows_out.append(
                {
                    **asdict(row),
                    "cos_y_values": cos_y,
                    "relerr_y_values": rel_y,
                    "r2_y_values": r2_y,
                    "cos_x_values": cos_x,
                    "relerr_x_values": rel_x,
                }
            )

            log(
                f"{variant:22s} L{int(layer):02d} "
                f"cos_y={row.mean_cos_y:+.3f} r2_y={row.mean_r2_y:+.3f} rel_y={row.mean_relerr_y:.3f}"
            )

    # Write CSV summary.
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "pair_id",
                "model_key",
                "variant",
                "layer",
                "n_samples",
                "mean_cos_y",
                "mean_r2_y",
                "mean_relerr_y",
                "mean_cos_x",
                "mean_relerr_x",
            ],
        )
        w.writeheader()
        for r in summary_rows:
            w.writerow(
                {
                    "pair_id": r.pair_id,
                    "model_key": r.model_key,
                    "variant": r.variant,
                    "layer": r.layer,
                    "n_samples": r.n_samples,
                    "mean_cos_y": r.mean_cos_y,
                    "mean_r2_y": r.mean_r2_y,
                    "mean_relerr_y": r.mean_relerr_y,
                    "mean_cos_x": r.mean_cos_x,
                    "mean_relerr_x": r.mean_relerr_x,
                }
            )

    # Save a meta JSON with provenance + full per-word values.
    out_meta.write_text(
        json_dumps(
            {
                "pair": prompt_meta,
                "provenance": provenance,
                "config": {
                    "model_key": model_key,
                    "variant": variants,
                    "layers": chosen_layers,
                    "n_samples": int(args.n_samples),
                    "n_icl": int(args.n_icl),
                    "seed": int(args.seed),
                    "device": str(args.device),
                },
                "rows": rows_out,
            }
        ),
        encoding="utf-8",
    )

    # Make a figure: fidelity vs layer per variant.
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 4))
        for variant in variants:
            xs = [r.layer for r in summary_rows if r.variant == variant]
            ys = [r.mean_cos_y for r in summary_rows if r.variant == variant]
            if xs:
                plt.plot(xs, ys, marker="o", linewidth=1.5, label=variant)
        plt.axhline(0.0, color="black", linewidth=1)
        plt.title(f"Fidelity vs layer (cosine vs MLP output) | {pair_id} | {model_key}")
        plt.xlabel("Layer")
        plt.ylabel("Mean cosine(recon, mlp_out)")
        plt.legend()
        fig_path = figs_dir / f"fig_fidelity_vs_layer_{pair_id}_{model_key}.png"
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()
        log(f"Wrote: {fig_path}")
    except Exception as e:
        log(f"Figure skipped: {e}")

    log(f"Wrote: {out_csv}")
    log(f"Wrote: {out_meta}")
    return 0


def json_dumps(obj: Any) -> str:
    import json

    return json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True)


if __name__ == "__main__":
    raise SystemExit(main())
