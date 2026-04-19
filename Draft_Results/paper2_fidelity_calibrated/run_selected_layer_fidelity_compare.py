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

from core import _extract_mlp_io_at_position_from_input_ids, load_model, load_transcoder, set_all_seeds  # noqa: E402
from paper2_fidelity_calibrated.phase1_common import (  # noqa: E402
    _prepare_prompts_and_positions,
    load_pair_split,
    load_stagea_best,
    log,
    resolve_stagea_path,
)

VARIANTS = ["skipless_or_non_affine", "affine_skip"]


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


def _safe_div(num: float, den: float) -> float:
    den_f = float(den)
    if not np.isfinite(den_f) or abs(den_f) <= 1e-12:
        return float("nan")
    return float(num / den_f)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.detach().float()
    b_f = b.detach().float()
    denom = float(torch.norm(a_f).item()) * float(torch.norm(b_f).item())
    if not np.isfinite(denom) or denom <= 1e-12:
        return float("nan")
    return float(torch.dot(a_f, b_f).item() / denom)


def _topk_indices(vec: torch.Tensor, k: int) -> torch.Tensor:
    k_eff = min(max(1, int(k)), int(vec.numel()))
    return torch.topk(torch.abs(vec.detach().float()), k_eff).indices.detach().to(dtype=torch.long)


def _capture_share(vec: torch.Tensor, idx: torch.Tensor) -> float:
    abs_vec = torch.abs(vec.detach().float())
    total = float(abs_vec.sum().item())
    if not np.isfinite(total) or total <= 1e-12:
        return float("nan")
    if idx.numel() <= 0:
        return 0.0
    return float(abs_vec[idx].sum().item() / total)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Selected-layer transcoder fidelity / ceiling comparison across supported families.")
    ap.add_argument("--model", type=str, default="4b", choices=["1b", "4b"])
    ap.add_argument("--pair", type=str, default="aksharantar_hin_latin")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-icl", type=int, default=64)
    ap.add_argument("--n-select", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--max-items", type=int, default=30)
    ap.add_argument("--stagea", type=str, default="")
    ap.add_argument("--external-only", action="store_true")
    ap.add_argument("--require-external-sources", action="store_true")
    ap.add_argument("--min-pool-size", type=int, default=500)
    ap.add_argument("--out", type=str, default="")
    return ap.parse_args()


def _extract_prompt_packet(*, tokenizer: Any, word: Dict[str, str], pair_bundle: Dict[str, Any], stagea_best: Dict[str, Any], device: str) -> Dict[str, Any]:
    return _prepare_prompts_and_positions(
        tokenizer=tokenizer,
        word=word,
        icl_examples=pair_bundle["icl_examples"],
        prompt_variant=str(stagea_best.get("prompt_variant", "canonical")),
        input_script_name=pair_bundle["input_script_name"],
        source_language=pair_bundle["source_language"],
        output_script_name=pair_bundle["output_script_name"],
        device=device,
        selector_reference=str(stagea_best.get("selector_reference", "zs")),
        patch_position_mode=str(stagea_best.get("patch_position_mode", "source_last_subtoken")),
        seed=int(stagea_best.get("seed", 0)),
    )


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

    model, tokenizer = load_model(str(args.model), device=str(args.device))
    device = str(next(model.parameters()).device)

    stagea_path = resolve_stagea_path(str(args.pair), str(args.model), str(args.stagea))
    if not stagea_path.exists():
        raise FileNotFoundError(f"Missing Stage A artifact: {stagea_path}")
    stagea_best = load_stagea_best(stagea_path, seed=int(args.seed))
    scope_repo = str(stagea_best.get("scope_repo", "") or "")
    if not scope_repo:
        # match existing pattern where caller adds scope_repo from config when needed
        from config import get_model_config  # noqa: E402

        cfg = get_model_config(str(args.model))
        scope_repo = str(cfg.scope_repo)
        stagea_best["scope_repo"] = scope_repo

    transcoders = {
        variant: load_transcoder(model, scope_repo, int(stagea_best["layer"]), device, variant=variant)
        for variant in VARIANTS
    }

    eval_rows = list(pair_bundle["eval_rows"][: max(1, int(args.max_items))])
    if not eval_rows:
        raise RuntimeError("No evaluation rows available for selected-layer fidelity comparison")
    out_root = (
        Path(str(args.out)).resolve()
        if str(args.out).strip()
        else PROJECT_ROOT / "paper2_fidelity_calibrated" / "results" / "selected_layer_fidelity_compare" / str(args.pair) / str(args.model)
    )
    out_root.mkdir(parents=True, exist_ok=True)

    item_rows: List[Dict[str, Any]] = []
    log(
        f"Running selected-layer fidelity compare: pair={args.pair} model={args.model} "
        f"layer={stagea_best['layer']} topk={stagea_best['topk']} variants={VARIANTS}"
    )

    for item_idx, word in enumerate(eval_rows, start=1):
        packet = _extract_prompt_packet(
            tokenizer=tokenizer,
            word=word,
            pair_bundle=pair_bundle,
            stagea_best=stagea_best,
            device=device,
        )
        mlp_in_icl, mlp_out_icl = _extract_mlp_io_at_position_from_input_ids(
            model=model,
            input_ids=packet["feature_icl_input_ids"],
            layer=int(stagea_best["layer"]),
            position=int(packet["icl_feature_position"]),
        )
        mlp_in_ref, mlp_out_ref = _extract_mlp_io_at_position_from_input_ids(
            model=model,
            input_ids=packet["feature_selector_input_ids"],
            layer=int(stagea_best["layer"]),
            position=int(packet["selector_ref_position"]),
        )
        for variant, transcoder in transcoders.items():
            icl_feats = transcoder.encode(mlp_in_icl.unsqueeze(0)).squeeze(0)
            ref_feats = transcoder.encode(mlp_in_ref.unsqueeze(0)).squeeze(0)
            rec_icl = transcoder.decode(icl_feats.unsqueeze(0)).squeeze(0).detach()
            rec_ref = transcoder.decode(ref_feats.unsqueeze(0)).squeeze(0).detach()
            delta_feats = (icl_feats - ref_feats).detach()
            idx = _topk_indices(delta_feats, int(stagea_best["topk"]))
            item_rows.append(
                {
                    "pair": str(args.pair),
                    "model": str(args.model),
                    "seed": int(args.seed),
                    "item_index": int(item_idx - 1),
                    "word_ood": str(word["ood"]),
                    "word_hindi": str(word["hindi"]),
                    "layer": int(stagea_best["layer"]),
                    "topk": int(stagea_best["topk"]),
                    "stagea_selected_variant": str(stagea_best["variant"]),
                    "variant": str(variant),
                    "icl_reconstruction_cosine": float(_cosine(rec_icl, mlp_out_icl)),
                    "icl_reconstruction_rel_error": float(_safe_div(torch.norm((rec_icl - mlp_out_icl).float()).item(), torch.norm(mlp_out_icl.float()).item())),
                    "icl_reconstruction_mse": float(torch.mean((rec_icl.float() - mlp_out_icl.float()) ** 2).item()),
                    "ref_reconstruction_cosine": float(_cosine(rec_ref, mlp_out_ref)),
                    "ref_reconstruction_rel_error": float(_safe_div(torch.norm((rec_ref - mlp_out_ref).float()).item(), torch.norm(mlp_out_ref.float()).item())),
                    "ref_reconstruction_mse": float(torch.mean((rec_ref.float() - mlp_out_ref.float()) ** 2).item()),
                    "delta_topk_mass_share": float(_capture_share(delta_feats, idx)),
                    "delta_l1_norm": float(torch.abs(delta_feats.float()).sum().item()),
                    "active_feature_fraction_icl": float(torch.count_nonzero(torch.abs(icl_feats.detach().float()) > 1e-8).item() / max(1, int(icl_feats.numel()))),
                }
            )

    summary_by_variant: List[Dict[str, Any]] = []
    for variant in VARIANTS:
        rows_v = [row for row in item_rows if str(row.get("variant")) == variant]
        if not rows_v:
            continue
        def _m(key: str) -> float:
            return float(np.nanmean([float(row.get(key, float("nan"))) for row in rows_v]))
        summary_by_variant.append(
            {
                "variant": str(variant),
                "n_items": int(len(rows_v)),
                "mean_icl_reconstruction_cosine": _m("icl_reconstruction_cosine"),
                "mean_icl_reconstruction_rel_error": _m("icl_reconstruction_rel_error"),
                "mean_icl_reconstruction_mse": _m("icl_reconstruction_mse"),
                "mean_ref_reconstruction_cosine": _m("ref_reconstruction_cosine"),
                "mean_ref_reconstruction_rel_error": _m("ref_reconstruction_rel_error"),
                "mean_ref_reconstruction_mse": _m("ref_reconstruction_mse"),
                "mean_delta_topk_mass_share": _m("delta_topk_mass_share"),
                "mean_active_feature_fraction_icl": _m("active_feature_fraction_icl"),
            }
        )

    payload = {
        "experiment": "selected_layer_fidelity_compare",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pair": str(args.pair),
        "model": str(args.model),
        "seed": int(args.seed),
        "stagea_best": stagea_best,
        "variants": list(VARIANTS),
        "summary_by_variant": summary_by_variant,
        "item_rows": item_rows,
    }
    _write_json(out_root / "selected_layer_fidelity_compare.json", payload)
    log(f"Saved: {out_root / 'selected_layer_fidelity_compare.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
