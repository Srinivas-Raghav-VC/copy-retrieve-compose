"""
Prompt-format robustness probe.

Runs a lightweight baseline-style audit across meaning-preserving prompt variants
on the same split. This is exploratory and intended to quantify format spread,
not to replace the locked canonical prompt for confirmatory claims.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List

from rescue_research.config import RunConfig
from rescue_research.data import get_words, three_way_split
from rescue_research.data_pipeline.ingest import get_pair_prompt_metadata
from rescue_research.data_pipeline.runtime_splits import (
    load_prepared_split_for_seed,
    runtime_three_way_from_prepared,
)
from rescue_research.prompts.templates import PROMPT_FORMAT_VARIANTS


def run_prompt_robustness(
    config: RunConfig,
    *,
    n_eval_sample: int = 24,
    variants: Iterable[str] | None = None,
) -> Dict:
    """
    Evaluate ICL lift across prompt format variants.

    Returns a compact payload with per-variant means and spread statistics.
    """
    ref_root = Path(__file__).resolve().parent.parent.parent
    if str(ref_root) not in sys.path:
        sys.path.insert(0, str(ref_root))

    from core import build_task_prompt, get_first_token_prob, load_model
    import numpy as np

    pair_id = str(config.pair or "hindi_telugu").strip()
    seed = int(config.seeds[0]) if config.seeds else 42

    split_source = "runtime_random_split"
    split_meta: Dict[str, object] = {
        "split_source": split_source,
        "use_blind_eval": False,
    }

    prepared_split_dir = str(getattr(config, "prepared_split_dir", "") or "").strip()
    if prepared_split_dir:
        payload, split_path = load_prepared_split_for_seed(Path(prepared_split_dir), int(seed))
        icl, _select, eval_samples, split_meta = runtime_three_way_from_prepared(
            payload=payload,
            n_icl=int(config.n_icl),
            n_select=int(config.n_select),
            n_eval=int(config.n_eval),
            use_blind_eval=bool(getattr(config, "use_blind_eval", False)),
            split_path=split_path,
        )
        split_source = str(split_meta.get("split_source", "prepared_protocol_split"))
    else:
        words = get_words(pair_id=pair_id)
        if len(words) < config.n_icl + config.n_select + config.n_eval:
            raise ValueError(
                f"Not enough words for prompt robustness split: need "
                f"{config.n_icl + config.n_select + config.n_eval}, have {len(words)}"
            )
        icl, _select, eval_samples = three_way_split(
            words, config.n_icl, config.n_select, config.n_eval, seed
        )

    meta = get_pair_prompt_metadata(pair_id)
    input_script = str(meta.get("target_script", "")).strip() or "Telugu"
    source_lang = str(meta.get("source_language", "")).strip() or "Hindi"
    output_script = str(meta.get("source_script", "")).strip() or "Devanagari"

    eval_subset = list(eval_samples)[: max(1, min(int(n_eval_sample), len(eval_samples)))]

    model, tokenizer = load_model(config.model, device=config.device)
    device = str(config.device)

    use_variants = [
        str(v).strip().lower()
        for v in (list(variants) if variants is not None else list(PROMPT_FORMAT_VARIANTS))
        if str(v).strip()
    ]
    if not use_variants:
        use_variants = ["canonical"]

    per_variant: Dict[str, Dict[str, float | int]] = {}
    for variant in use_variants:
        zs_vals: List[float] = []
        icl_vals: List[float] = []
        lifts: List[float] = []

        for sample in eval_subset:
            ood = str(sample.get("ood", sample.get("telugu", "")))
            tgt = str(sample.get("hindi", ""))
            if not ood or not tgt:
                continue

            zs_prompt = build_task_prompt(
                ood,
                None,
                input_script_name=input_script,
                source_language=source_lang,
                output_script_name=output_script,
                prompt_variant=variant,
            )
            icl_prompt = build_task_prompt(
                ood,
                icl,
                input_script_name=input_script,
                source_language=source_lang,
                output_script_name=output_script,
                prompt_variant=variant,
            )

            p_zs, _ = get_first_token_prob(model, tokenizer, zs_prompt, tgt, device)
            p_icl, _ = get_first_token_prob(model, tokenizer, icl_prompt, tgt, device)
            if math.isfinite(p_zs) and math.isfinite(p_icl):
                zs_vals.append(float(p_zs))
                icl_vals.append(float(p_icl))
                lifts.append(float(p_icl - p_zs))

        arr_zs = np.array(zs_vals, dtype=np.float64)
        arr_icl = np.array(icl_vals, dtype=np.float64)
        arr_lift = np.array(lifts, dtype=np.float64)
        per_variant[variant] = {
            "n_eval": int(arr_lift.shape[0]),
            "mean_prob_zs": float(np.mean(arr_zs)) if arr_zs.size else float("nan"),
            "mean_prob_icl": float(np.mean(arr_icl)) if arr_icl.size else float("nan"),
            "mean_icl_lift": float(np.mean(arr_lift)) if arr_lift.size else float("nan"),
            "positive_lift_rate": float(np.mean(arr_lift > 0.0)) if arr_lift.size else float("nan"),
        }

    lift_means = [
        float(v.get("mean_icl_lift", float("nan")))
        for v in per_variant.values()
        if isinstance(v, dict)
    ]
    lift_finite = [x for x in lift_means if math.isfinite(x)]
    spread = (max(lift_finite) - min(lift_finite)) if lift_finite else float("nan")

    # Best-effort cleanup (important when looping over many pair/model runs).
    try:
        import torch  # local import to avoid hard dependency at module import time

        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return {
        "pair_id": pair_id,
        "model": str(config.model),
        "seed": int(seed),
        "n_eval_sample": int(len(eval_subset)),
        "split_source": split_source,
        "split_meta": split_meta,
        "variants": per_variant,
        "format_spread_mean_icl_lift": float(spread),
    }
