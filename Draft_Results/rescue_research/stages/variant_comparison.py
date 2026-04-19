"""
Variant comparison: run patching at best layer with affine vs skipless transcoders.

Writes artifacts/variants/{variant}/{model}/{pair_id}.json with mean_pe, n_samples.
Used by Figure 4 (affine vs skip).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

from rescue_research.config import RunConfig
from rescue_research.data import get_words, three_way_split
from rescue_research.data_pipeline.ingest import get_pair_prompt_metadata
from rescue_research.data_pipeline.runtime_splits import (
    load_prepared_split_for_seed,
    runtime_three_way_from_prepared,
)

VARIANTS = ["affine_skip", "skipless_or_non_affine"]


def run_variant_comparison(config: RunConfig, n_eval_sample: int = 20) -> List[Dict]:
    """
    Run patching at best layer for each variant; return list of {variant, mean_pe, n_samples}.
    """
    ref_root = Path(__file__).resolve().parent.parent.parent
    if str(ref_root) not in sys.path:
        sys.path.insert(0, str(ref_root))

    from config import get_model_config
    from core import (
        load_model,
        load_transcoder,
        get_layer_device,
        run_patching_experiment,
        compute_statistics,
    )

    config.ensure_out_dir()
    best_path = config.out_dir / "best_layer.txt"
    if best_path.exists():
        layer = int(best_path.read_text(encoding="utf-8").strip())
    else:
        layer = config.layer

    pair_id = str(config.pair or "hindi_telugu").strip()

    try:
        pair_meta = get_pair_prompt_metadata(pair_id)
    except ValueError:
        return []
    input_script = pair_meta.get("target_script", "Telugu")
    output_script = pair_meta.get("source_script", "Devanagari")
    source_lang = pair_meta.get("source_language", "Hindi")

    seed = config.seeds[0]
    prepared_split_dir = str(getattr(config, "prepared_split_dir", "") or "").strip()
    if prepared_split_dir:
        payload, split_path = load_prepared_split_for_seed(Path(prepared_split_dir), int(seed))
        icl, _select, eval_samples, _split_meta = runtime_three_way_from_prepared(
            payload=payload,
            n_icl=int(config.n_icl),
            n_select=int(config.n_select),
            n_eval=int(config.n_eval),
            use_blind_eval=bool(getattr(config, "use_blind_eval", False)),
            split_path=split_path,
        )
    else:
        words = get_words(pair_id=pair_id)
        if len(words) < config.n_icl + config.n_select + config.n_eval:
            return []
        icl, _select, eval_samples = three_way_split(
            words, config.n_icl, config.n_select, config.n_eval, seed
        )

    eval_subset = eval_samples[: min(n_eval_sample, len(eval_samples))]

    model, tokenizer = load_model(config.model)
    cfg = get_model_config(config.model)
    device = str(get_layer_device(model, layer))
    topk = config.topk_values[-1] if config.topk_values else 25

    results: List[Dict] = []
    for variant in VARIANTS:
        transcoder = load_transcoder(
            model, cfg.scope_repo, layer, device, variant=variant
        )
        rows: List[Dict] = []
        for sample in eval_subset:
            sw = {"english": sample["english"], "hindi": sample["hindi"], "telugu": sample.get("ood", sample.get("telugu", "")), "ood": sample.get("ood", sample.get("telugu", ""))}
            try:
                r = run_patching_experiment(
                    model=model,
                    tokenizer=tokenizer,
                    transcoder=transcoder,
                    layer=layer,
                    test_word=sw,
                    icl_examples=icl,
                    topk=topk,
                    device=device,
                    seed=seed,
                    input_script_name=input_script,
                    source_language=source_lang,
                    output_script_name=output_script,
                    patch_style="substitute",
                    feature_selection="topk_abs_icl",
                )
                rows.append(r)
            except Exception:
                pass
        del transcoder
        if sys.modules.get("torch") and hasattr(sys.modules["torch"].cuda, "empty_cache"):
            sys.modules["torch"].cuda.empty_cache()

        stats = compute_statistics(rows) if rows else {}
        mean_pe = float(stats.get("mean_pe", 0.0) or 0.0)
        results.append({
            "variant": variant,
            "mean_pe": mean_pe,
            "n_samples": len(rows),
        })
    return results
