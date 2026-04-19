#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_model_config  # noqa: E402
from core import (  # noqa: E402
    _extract_layer_output_at_position_from_input_ids,
    _extract_mlp_io_at_position_from_input_ids,
    _teacher_forced_metrics_from_input_ids,
    load_model,
    register_dense_mlp_output_patch_hook,
    register_layer_output_replace_hook,
    set_all_seeds,
)
from paper2_fidelity_calibrated.eval_utils import (  # noqa: E402
    akshara_cer,
    continuation_akshara_cer,
    first_entry_correct,
    normalize_text,
    script_compliance,
)
from paper2_fidelity_calibrated.phase1_common import (  # noqa: E402
    _prepare_prompts_and_positions,
    load_pair_split,
    load_stagea_best,
    log,
    resolve_stagea_path,
)


ALLOWED_SPACES = {"layer_output", "mlp_output"}


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


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out


def _parse_csv(raw: str) -> List[str]:
    return [x.strip() for x in str(raw or "").split(",") if x.strip()]


def _parse_spaces(raw: str) -> List[str]:
    vals = _parse_csv(raw)
    out = [x for x in vals if x in ALLOWED_SPACES]
    return out or ["layer_output", "mlp_output"]


def _deterministic_shuffle_vector(vec: torch.Tensor, *, seed: int, tag: str) -> torch.Tensor:
    msg = f"actdiff_shuffle::{seed}::{tag}".encode("utf-8")
    seed32 = int.from_bytes(hashlib.sha256(msg).digest()[:4], "little", signed=False)
    rng = np.random.default_rng(seed32)
    perm = rng.permutation(int(vec.numel()))
    perm_t = torch.tensor(perm, device=vec.device, dtype=torch.long)
    return torch.index_select(vec, 0, perm_t)


def _run_teacher_forced(
    *,
    model: Any,
    input_ids: torch.Tensor,
    target_ids: List[int],
    target_id: int,
    device: str,
    hook: Any = None,
) -> Dict[str, float]:
    try:
        return _teacher_forced_metrics_from_input_ids(
            model=model,
            input_ids=input_ids,
            target_ids=list(target_ids),
            target_id=int(target_id),
            device=str(device),
            competitor_id=-1,
        )
    finally:
        if hook is not None:
            try:
                hook.remove()
            except Exception:
                pass


def _generation_metrics(gold_text: str, pred_text: str, target_script: str) -> Dict[str, float]:
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


def _generate_with_hook(
    *,
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    hook: Any = None,
) -> str:
    pad_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", 0)
    try:
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            out = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                pad_token_id=int(pad_id),
            )
        new_tokens = out[0, input_ids.shape[1] :]
        return normalize_text(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    finally:
        if hook is not None:
            try:
                hook.remove()
            except Exception:
                pass


def _extract_space_vector(
    *,
    model: Any,
    input_ids: torch.Tensor,
    layer: int,
    position: int,
    space: str,
) -> torch.Tensor:
    if str(space) == "layer_output":
        return _extract_layer_output_at_position_from_input_ids(
            model=model,
            input_ids=input_ids,
            layer=int(layer),
            position=int(position),
        ).detach().float()
    if str(space) == "mlp_output":
        _, mlp_out = _extract_mlp_io_at_position_from_input_ids(
            model=model,
            input_ids=input_ids,
            layer=int(layer),
            position=int(position),
        )
        return mlp_out.detach().float()
    raise ValueError(f"Unsupported space={space!r}")


def _register_space_hook(
    *,
    model: Any,
    layer: int,
    patch_vector: torch.Tensor,
    patch_position: int,
    space: str,
):
    if str(space) == "layer_output":
        return register_layer_output_replace_hook(
            model,
            int(layer),
            patch_vector,
            patch_position=int(patch_position),
        )
    if str(space) == "mlp_output":
        return register_dense_mlp_output_patch_hook(
            model,
            int(layer),
            patch_vector,
            patch_position=int(patch_position),
        )
    raise ValueError(f"Unsupported space={space!r}")


def _mean_vector(vectors: List[torch.Tensor]) -> torch.Tensor:
    if not vectors:
        raise ValueError("Expected at least one vector to average")
    return torch.stack([v.detach().float().cpu() for v in vectors], dim=0).mean(dim=0)


def _resolve_eval_generation_allowed(packet: Dict[str, Any]) -> bool:
    # When patching at target_pos1, generation does not expose that position yet.
    return int(packet["zs_patch_position"]) < int(packet["zs_input_ids"].shape[1])


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Activation-difference baseline at the selected Stage-A site.")
    ap.add_argument("--model", type=str, default="4b", choices=["1b", "4b"])
    ap.add_argument("--pair", type=str, default="aksharantar_hin_latin")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-icl", type=int, default=64)
    ap.add_argument("--n-select", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--stats-items", type=int, default=100)
    ap.add_argument("--max-items", type=int, default=50)
    ap.add_argument("--spaces", type=str, default="layer_output,mlp_output")
    ap.add_argument("--stagea", type=str, default="")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--external-only", action="store_true")
    ap.add_argument("--require-external-sources", action="store_true")
    ap.add_argument("--min-pool-size", type=int, default=500)
    ap.add_argument("--out", type=str, default="")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    set_all_seeds(int(args.seed))

    spaces = _parse_spaces(str(args.spaces))
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

    calibration_rows = list(pair_bundle["select_rows"][: max(1, int(args.stats_items))])
    eval_rows = list(pair_bundle["eval_rows"][: max(1, int(args.max_items))])
    if not calibration_rows:
        calibration_rows = list(eval_rows)
    if not eval_rows:
        raise RuntimeError("No evaluation rows available for activation-difference baseline")

    log(
        f"Running activation-difference baseline: pair={args.pair} model={args.model} "
        f"items={len(eval_rows)} calibration={len(calibration_rows)} spaces={spaces} "
        f"stageA=(layer={stagea_best['layer']}, topk={stagea_best['topk']}, variant={stagea_best['variant']})"
    )

    deltas_by_space: Dict[str, List[torch.Tensor]] = {space: [] for space in spaces}
    for item_idx, word in enumerate(calibration_rows, start=1):
        if item_idx == 1 or item_idx % 10 == 0:
            log(f"Calibrating mean deltas [{item_idx}/{len(calibration_rows)}]")
        packet = _prepare_prompts_and_positions(
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
            seed=int(args.seed),
        )
        for space in spaces:
            icl_vec = _extract_space_vector(
                model=model,
                input_ids=packet["feature_icl_input_ids"],
                layer=int(stagea_best["layer"]),
                position=int(packet["icl_feature_position"]),
                space=space,
            )
            ref_vec = _extract_space_vector(
                model=model,
                input_ids=packet["feature_selector_input_ids"],
                layer=int(stagea_best["layer"]),
                position=int(packet["selector_ref_position"]),
                space=space,
            )
            deltas_by_space[space].append((icl_vec - ref_vec).detach().cpu())

    mean_delta_by_space = {space: _mean_vector(vs) for space, vs in deltas_by_space.items()}
    shuffled_delta_by_space = {
        space: _deterministic_shuffle_vector(mean_delta_by_space[space].to(device=device), seed=int(args.seed), tag=f"{args.model}:{args.pair}:{space}").detach().cpu()
        for space in spaces
    }

    item_rows: List[Dict[str, Any]] = []
    for item_idx, word in enumerate(eval_rows, start=1):
        log(f"Eval [{item_idx}/{len(eval_rows)}] {word['ood']} -> {word['hindi']}")
        packet = _prepare_prompts_and_positions(
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
            seed=int(args.seed),
        )
        target_ids = list(packet["target_ids"])
        target_id = int(packet["target_id"])
        if target_id < 0 or not target_ids:
            continue

        zs_metrics = _run_teacher_forced(
            model=model,
            input_ids=packet["zs_input_ids"],
            target_ids=target_ids,
            target_id=target_id,
            device=device,
        )
        icl_metrics = _run_teacher_forced(
            model=model,
            input_ids=packet["icl_input_ids"],
            target_ids=target_ids,
            target_id=target_id,
            device=device,
        )
        eval_generation_allowed = _resolve_eval_generation_allowed(packet)
        zs_pred = _generate_with_hook(
            model=model,
            tokenizer=tokenizer,
            input_ids=packet["zs_input_ids"],
            max_new_tokens=int(args.max_new_tokens),
        ) if eval_generation_allowed else ""
        icl_pred = _generate_with_hook(
            model=model,
            tokenizer=tokenizer,
            input_ids=packet["icl_input_ids"],
            max_new_tokens=int(args.max_new_tokens),
        ) if eval_generation_allowed else ""
        zs_gen = _generation_metrics(str(word["hindi"]), zs_pred, pair_bundle["output_script_name"]) if eval_generation_allowed else {}
        icl_gen = _generation_metrics(str(word["hindi"]), icl_pred, pair_bundle["output_script_name"]) if eval_generation_allowed else {}

        patch_position_mode = str(stagea_best.get("patch_position_mode", "source_last_subtoken")).strip().lower()
        if patch_position_mode == "target_pos1_teacher_forced":
            patch_position_mode = "target_pos1"
        zs_base_extract_ids = packet["zs_tf_input_ids"] if patch_position_mode == "target_pos1" else packet["zs_input_ids"]

        for space in spaces:
            zs_base_vec = _extract_space_vector(
                model=model,
                input_ids=zs_base_extract_ids,
                layer=int(stagea_best["layer"]),
                position=int(packet["zs_patch_position"]),
                space=space,
            )
            mean_delta = mean_delta_by_space[space].to(device=zs_base_vec.device, dtype=zs_base_vec.dtype)
            shuffled_delta = shuffled_delta_by_space[space].to(device=zs_base_vec.device, dtype=zs_base_vec.dtype)
            patch_vec = (zs_base_vec + mean_delta).detach()
            shuffled_patch_vec = (zs_base_vec + shuffled_delta).detach()

            patched_hook = _register_space_hook(
                model=model,
                layer=int(stagea_best["layer"]),
                patch_vector=patch_vec,
                patch_position=int(packet["zs_patch_position"]),
                space=space,
            )
            patched_metrics = _run_teacher_forced(
                model=model,
                input_ids=packet["zs_input_ids"],
                target_ids=target_ids,
                target_id=target_id,
                device=device,
                hook=patched_hook,
            )
            shuffled_hook = _register_space_hook(
                model=model,
                layer=int(stagea_best["layer"]),
                patch_vector=shuffled_patch_vec,
                patch_position=int(packet["zs_patch_position"]),
                space=space,
            )
            shuffled_metrics = _run_teacher_forced(
                model=model,
                input_ids=packet["zs_input_ids"],
                target_ids=target_ids,
                target_id=target_id,
                device=device,
                hook=shuffled_hook,
            )

            patched_pred = ""
            patched_gen = {}
            if eval_generation_allowed:
                patched_gen_hook = _register_space_hook(
                    model=model,
                    layer=int(stagea_best["layer"]),
                    patch_vector=patch_vec,
                    patch_position=int(packet["zs_patch_position"]),
                    space=space,
                )
                patched_pred = _generate_with_hook(
                    model=model,
                    tokenizer=tokenizer,
                    input_ids=packet["zs_input_ids"],
                    max_new_tokens=int(args.max_new_tokens),
                    hook=patched_gen_hook,
                )
                patched_gen = _generation_metrics(str(word["hindi"]), patched_pred, pair_bundle["output_script_name"])

            denom = _safe_float(icl_metrics.get("first_prob")) - _safe_float(zs_metrics.get("first_prob"))
            rescue_frac_first = (
                (_safe_float(patched_metrics.get("first_prob")) - _safe_float(zs_metrics.get("first_prob"))) / denom
                if np.isfinite(denom) and abs(denom) > 1e-9
                else float("nan")
            )
            item_rows.append(
                {
                    "pair": str(args.pair),
                    "model": str(args.model),
                    "seed": int(args.seed),
                    "space": str(space),
                    "item_index": int(item_idx - 1),
                    "word_english": str(word["english"]),
                    "word_target": str(word["hindi"]),
                    "word_source_romanized": str(word["ood"]),
                    "patch_layer": int(stagea_best["layer"]),
                    "patch_position": int(packet["zs_patch_position"]),
                    "selector_reference": str(stagea_best.get("selector_reference", "zs")),
                    "patch_position_mode": str(stagea_best.get("patch_position_mode", "source_last_subtoken")),
                    "zs_first_prob": _safe_float(zs_metrics.get("first_prob")),
                    "icl_first_prob": _safe_float(icl_metrics.get("first_prob")),
                    "patched_first_prob": _safe_float(patched_metrics.get("first_prob")),
                    "shuffled_first_prob": _safe_float(shuffled_metrics.get("first_prob")),
                    "zs_first_logit": _safe_float(zs_metrics.get("first_logit")),
                    "icl_first_logit": _safe_float(icl_metrics.get("first_logit")),
                    "patched_first_logit": _safe_float(patched_metrics.get("first_logit")),
                    "shuffled_first_logit": _safe_float(shuffled_metrics.get("first_logit")),
                    "zs_joint_logprob": _safe_float(zs_metrics.get("joint_logprob")),
                    "icl_joint_logprob": _safe_float(icl_metrics.get("joint_logprob")),
                    "patched_joint_logprob": _safe_float(patched_metrics.get("joint_logprob")),
                    "shuffled_joint_logprob": _safe_float(shuffled_metrics.get("joint_logprob")),
                    "pe_first": _safe_float(patched_metrics.get("first_prob")) - _safe_float(zs_metrics.get("first_prob")),
                    "pe_shuffled_first": _safe_float(shuffled_metrics.get("first_prob")) - _safe_float(zs_metrics.get("first_prob")),
                    "pe_logit": _safe_float(patched_metrics.get("first_logit")) - _safe_float(zs_metrics.get("first_logit")),
                    "rescue_frac_first": float(rescue_frac_first),
                    "mean_delta_norm": float(torch.norm(mean_delta.float()).item()),
                    "zs_base_norm": float(torch.norm(zs_base_vec.float()).item()),
                    "patch_vec_norm": float(torch.norm(patch_vec.float()).item()),
                    "eval_generation_allowed": bool(eval_generation_allowed),
                    "exact_match_zs": _safe_float(zs_gen.get("exact_match", float("nan"))),
                    "exact_match_icl": _safe_float(icl_gen.get("exact_match", float("nan"))),
                    "exact_match_patched": _safe_float(patched_gen.get("exact_match", float("nan"))),
                    "first_entry_correct_zs": _safe_float(zs_gen.get("first_entry_correct", float("nan"))),
                    "first_entry_correct_icl": _safe_float(icl_gen.get("first_entry_correct", float("nan"))),
                    "first_entry_correct_patched": _safe_float(patched_gen.get("first_entry_correct", float("nan"))),
                    "akshara_cer_zs": _safe_float(zs_gen.get("akshara_cer", float("nan"))),
                    "akshara_cer_icl": _safe_float(icl_gen.get("akshara_cer", float("nan"))),
                    "akshara_cer_patched": _safe_float(patched_gen.get("akshara_cer", float("nan"))),
                    "script_compliance_zs": _safe_float(zs_gen.get("script_compliance", float("nan"))),
                    "script_compliance_icl": _safe_float(icl_gen.get("script_compliance", float("nan"))),
                    "script_compliance_patched": _safe_float(patched_gen.get("script_compliance", float("nan"))),
                }
            )

    summary_by_space: Dict[str, Dict[str, Any]] = {}
    for space in spaces:
        rows = [row for row in item_rows if str(row.get("space")) == str(space)]
        pe = np.array([_safe_float(r.get("pe_first", float("nan"))) for r in rows], dtype=np.float64)
        pe_shuf = np.array([_safe_float(r.get("pe_shuffled_first", float("nan"))) for r in rows], dtype=np.float64)
        rescue = np.array([_safe_float(r.get("rescue_frac_first", float("nan"))) for r in rows], dtype=np.float64)
        em_zs = np.array([_safe_float(r.get("exact_match_zs", float("nan"))) for r in rows], dtype=np.float64)
        em_patch = np.array([_safe_float(r.get("exact_match_patched", float("nan"))) for r in rows], dtype=np.float64)
        fec_zs = np.array([_safe_float(r.get("first_entry_correct_zs", float("nan"))) for r in rows], dtype=np.float64)
        fec_patch = np.array([_safe_float(r.get("first_entry_correct_patched", float("nan"))) for r in rows], dtype=np.float64)
        finite_pe = pe[np.isfinite(pe)]
        if finite_pe.size:
            rng = np.random.default_rng(int(args.seed) + (1 if space == "layer_output" else 17))
            boots = []
            for _ in range(2000):
                samp = rng.choice(finite_pe, size=finite_pe.size, replace=True)
                boots.append(float(np.mean(samp)))
            ci_low, ci_high = float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
        else:
            ci_low = ci_high = float("nan")
        summary_by_space[str(space)] = {
            "n_items": int(len(rows)),
            "n_calibration": int(len(calibration_rows)),
            "mean_delta_norm": float(torch.norm(mean_delta_by_space[space].float()).item()),
            "mean_pe_first": float(np.nanmean(pe)),
            "ci_pe_first_low": ci_low,
            "ci_pe_first_high": ci_high,
            "mean_pe_shuffled_first": float(np.nanmean(pe_shuf)),
            "specificity_margin_vs_shuffle": float(np.nanmean(pe) - np.nanmean(pe_shuf)),
            "mean_rescue_frac_first": float(np.nanmean(rescue)),
            "mean_exact_match_delta": float(np.nanmean(em_patch) - np.nanmean(em_zs)),
            "mean_first_entry_delta": float(np.nanmean(fec_patch) - np.nanmean(fec_zs)),
        }

    best_space = None
    if summary_by_space:
        def _score(space_name: str) -> float:
            row = summary_by_space[str(space_name)]
            pe = _safe_float(row.get("mean_pe_first"))
            margin = _safe_float(row.get("specificity_margin_vs_shuffle"))
            if not np.isfinite(pe):
                return -1e9
            if not np.isfinite(margin):
                margin = -1e9
            return min(pe, margin)
        best_space = max(summary_by_space, key=_score)

    payload = {
        "experiment": "activation_difference_baseline",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": str(args.model),
        "pair": str(args.pair),
        "seed": int(args.seed),
        "stagea_best": stagea_best,
        "spaces": spaces,
        "stats_items": int(len(calibration_rows)),
        "max_items": int(len(eval_rows)),
        "summary_by_space": summary_by_space,
        "best_space": str(best_space) if best_space is not None else None,
        "item_rows": item_rows,
    }

    out_root = (
        Path(args.out).resolve()
        if str(args.out).strip()
        else PROJECT_ROOT / "paper2_fidelity_calibrated" / "results" / "activation_difference_baseline" / str(args.pair) / str(args.model)
    )
    out_root.mkdir(parents=True, exist_ok=True)
    _write_json(out_root / "activation_difference_baseline.json", payload)
    _write_rows(out_root / "activation_difference_baseline_item_level", item_rows)
    log(f"Saved: {out_root / 'activation_difference_baseline.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
