"""
Stage 4: Causal mediation (NIE/NDE, triangulation) at best layer.

Integrates causal_mediation into the main pipeline. Loads model, transcoder,
uses three-way split (eval set as test words), runs run_causal_mediation_experiment,
saves to config.out_dir.

Band mode: when band_size > 1, runs mediation at each of top-K layers from
layer sweep (see configs/stats.yaml mediation_band_size).
"""

from __future__ import annotations

import json
import math
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


def _run_mediation_at_layer(config: RunConfig, layer: int) -> Dict:
    """Run causal mediation at a single layer; return per-seed and aggregate results."""
    ref_root = Path(__file__).resolve().parent.parent.parent
    if str(ref_root) not in sys.path:
        sys.path.insert(0, str(ref_root))

    from config import get_model_config
    from core import load_model, load_transcoder, get_layer_device
    from causal_mediation import run_causal_mediation_experiment

    pair_id = str(config.pair or "hindi_telugu").strip()
    prepared_split_dir = str(getattr(config, "prepared_split_dir", "") or "").strip()
    words: List[Dict] = []
    if not prepared_split_dir:
        words = get_words(pair_id=pair_id)
        if len(words) < config.n_icl + config.n_select + config.n_eval:
            raise ValueError(
                f"Not enough words for three-way split: need {config.n_icl + config.n_select + config.n_eval}, have {len(words)}"
            )

    prompt_meta = get_pair_prompt_metadata(pair_id)
    input_script_name = str(prompt_meta.get("target_script", "")).strip() or "Telugu"
    source_language = str(prompt_meta.get("source_language", "")).strip() or "Hindi"
    output_script_name = str(prompt_meta.get("source_script", "")).strip() or "Devanagari"

    print(f"[rescue_research] Loading model {config.model}...", flush=True)
    model, tokenizer = load_model(config.model)
    cfg = get_model_config(config.model)
    device = str(get_layer_device(model, layer))
    transcoder = load_transcoder(model, cfg.scope_repo, layer, device)

    seed_results: List[Dict] = []
    for seed in list(config.seeds):
        split_meta: Dict[str, object] = {
            "split_source": "runtime_random_split",
            "use_blind_eval": False,
        }
        if prepared_split_dir:
            payload, split_path = load_prepared_split_for_seed(
                Path(prepared_split_dir), int(seed)
            )
            icl, _select, eval_samples, split_meta = runtime_three_way_from_prepared(
                payload=payload,
                n_icl=int(config.n_icl),
                n_select=int(config.n_select),
                n_eval=int(config.n_eval),
                use_blind_eval=bool(getattr(config, "use_blind_eval", False)),
                split_path=split_path,
            )
        else:
            icl, _select, eval_samples = three_way_split(
                words, config.n_icl, config.n_select, config.n_eval, int(seed)
            )
        test_words = []
        for w in eval_samples:
            source_token = str(w.get("hindi", w.get("source", "")))
            ood_token = str(w.get("ood", w.get("telugu", w.get("input", ""))))
            test_words.append(
                {
                    "english": str(w.get("english", "")),
                    "source": source_token,
                    "target": source_token,
                    "ood": ood_token,
                    # Backward-compatible aliases used by causal_mediation internals.
                    "hindi": source_token,
                    "telugu": ood_token,
                }
            )
        print(
            f"[rescue_research] Running causal mediation at layer {layer}, seed={seed}, n_words={len(test_words)}",
            flush=True,
        )
        res = run_causal_mediation_experiment(
            model=model,
            tokenizer=tokenizer,
            transcoder=transcoder,
            layer=layer,
            test_words=test_words,
            icl_examples=icl,
            device=device,
            topk=config.topk_values[-1] if config.topk_values else 25,
            n_icl=config.n_icl,
            run_triangulation_test=True,
            input_script_name=input_script_name,
            source_language=source_language,
            output_script_name=output_script_name,
        )
        if isinstance(res, dict):
            res.setdefault(
                "prompt_meta",
                {
                    "pair_id": pair_id,
                    "input_script_name": input_script_name,
                    "source_language": source_language,
                    "output_script_name": output_script_name,
                },
            )
            res.setdefault("split_meta", dict(split_meta))
        seed_results.append({"seed": int(seed), "result": res, "split_meta": dict(split_meta)})

    if not seed_results:
        raise ValueError("No mediation results produced: config.seeds is empty.")

    # Keep top-level schema backward-compatible (first-seed payload),
    # then attach explicit per-seed outputs and aggregate summary.
    primary = dict(seed_results[0]["result"])
    mean_nie_vals: List[float] = []
    for sr in seed_results:
        rr = sr.get("result", {})
        if not isinstance(rr, dict):
            continue
        agg = rr.get("aggregate_stats", {})
        if isinstance(agg, dict):
            v = agg.get("mean_nie")
            if v is not None:
                try:
                    mean_nie_vals.append(float(v))
                except (TypeError, ValueError):
                    pass

    mean_nie_across = sum(mean_nie_vals) / len(mean_nie_vals) if mean_nie_vals else None
    std_nie_across = None
    if len(mean_nie_vals) >= 2:
        mu = float(mean_nie_across)
        std_nie_across = math.sqrt(
            sum((x - mu) ** 2 for x in mean_nie_vals) / (len(mean_nie_vals) - 1)
        )

    primary["seed_results"] = seed_results
    primary["seed_aggregate"] = {
        "n_seeds": len(seed_results),
        "mean_nie_across_seeds": mean_nie_across,
        "std_nie_across_seeds": std_nie_across,
    }

    del transcoder
    if hasattr(sys.modules.get("torch", None), "cuda") and sys.modules.get("torch"):
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return primary


def run_mediation(config: RunConfig) -> None:
    """Run mediation at best layer only (legacy single-layer path)."""
    config.ensure_out_dir()
    best_path = config.out_dir / "best_layer.txt"
    if best_path.exists():
        layer = int(best_path.read_text(encoding="utf-8").strip())
    else:
        layer = config.layer
        print(f"[rescue_research] No best_layer.txt; using config layer {layer}", flush=True)

    results = _run_mediation_at_layer(config, layer)
    out_path = config.out_dir / f"mediation_{config.model}_L{layer}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"[rescue_research] Mediation results: {out_path}", flush=True)


def run_mediation_band(config: RunConfig, band_size: int) -> List[Dict]:
    """
    Run mediation at top-K layers from layer sweep.
    Returns list of {layer, results} dicts; also writes per-layer JSON files.
    """
    config.ensure_out_dir()
    layer_path = config.out_dir / f"layer_sweep_cv_{config.model}.json"
    top_layers: List[int] = []
    if layer_path.exists():
        try:
            data = json.loads(layer_path.read_text(encoding="utf-8"))
            top_layers = (data.get("summary") or {}).get("top_layers") or []
        except Exception:
            pass
    if not top_layers:
        best_path = config.out_dir / "best_layer.txt"
        if best_path.exists():
            top_layers = [int(best_path.read_text(encoding="utf-8").strip())]
        else:
            top_layers = [config.layer]

    layers_to_run = top_layers[: max(1, band_size)]
    band_results: List[Dict] = []
    for layer in layers_to_run:
        results = _run_mediation_at_layer(config, int(layer))
        out_path = config.out_dir / f"mediation_{config.model}_L{layer}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
        print(f"[rescue_research] Mediation L{layer} -> {out_path}", flush=True)
        band_results.append({"layer": int(layer), "results": results})
    return band_results
