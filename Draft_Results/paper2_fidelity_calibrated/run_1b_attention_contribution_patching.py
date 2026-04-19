#!/usr/bin/env python3
"""
CORRECTED multi-layer patching: replace ATTENTION CONTRIBUTION only.

Uses the same o_proj pre-hook as head attribution, but patches ALL heads
at the target layer simultaneously. This replaces the attention output
(the contribution before residual addition) without touching the residual
stream directly.

Subsets tested:
  - all_global_attn: L5, L11, L17, L23
  - all_local_attn: all non-global layers
  - all_attn: all 26 layers
  - single_L11, single_L14, single_L23
  - first_two_global: L5, L11
  - last_two_global: L17, L23
"""
from __future__ import annotations
import json, sys, time, random
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import (
    apply_chat_template, build_task_prompt, get_model_layers,
    load_model, set_all_seeds,
)

OUT = ROOT / "paper2_fidelity_calibrated" / "results" / "1b_definitive"
OUT.mkdir(parents=True, exist_ok=True)


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def load_pair(pair_id, seed=42, n_icl=16):
    try:
        from paper2_fidelity_calibrated.phase1_common import load_pair_split
        return load_pair_split(pair_id, seed=seed, n_icl=n_icl, n_select=300, n_eval=200,
                               external_only=True, require_external_sources=True, min_pool_size=100)
    except (ValueError, KeyError, TypeError):
        pass
    data_path = ROOT / "data" / "transliteration" / ("%s.jsonl" % pair_id)
    meta_path = data_path.with_suffix(".jsonl.meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    ds = meta.get("dataset", {})
    words = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line.strip())
            src = str(row.get("source", row.get("english word", ""))).strip()
            tgt = str(row.get("target", row.get("native word", ""))).strip()
            if src and tgt:
                words.append({"ood": src, "hindi": tgt, "english": src})
    rng = random.Random(seed)
    rng.shuffle(words)
    return {
        "icl_examples": words[:n_icl], "eval_rows": words[n_icl+300:n_icl+300+200],
        "input_script_name": str(ds.get("source_script", "Latin")),
        "source_language": str(ds.get("source_language", pair_id.split("_")[1].capitalize())),
        "output_script_name": str(ds.get("target_script", "Devanagari")),
    }


def bootstrap_ci(vals, n_boot=2000):
    vals = np.array(vals, dtype=float)
    if len(vals) < 3:
        return {"mean": float(np.mean(vals)), "ci_lo": None, "ci_hi": None,
                "std": float(np.std(vals)), "n": len(vals)}
    boots = np.array([np.mean(np.random.choice(vals, len(vals), replace=True))
                      for _ in range(n_boot)])
    return {"mean": float(np.mean(vals)), "ci_lo": float(np.percentile(boots, 2.5)),
            "ci_hi": float(np.percentile(boots, 97.5)), "std": float(np.std(vals)),
            "n": len(vals)}


def jsafe(v):
    if isinstance(v, (str, int, bool)) or v is None: return v
    if isinstance(v, float): return v if np.isfinite(v) else None
    if isinstance(v, np.floating): return float(v) if np.isfinite(v) else None
    if isinstance(v, np.integer): return int(v)
    if isinstance(v, dict): return {str(k): jsafe(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)): return [jsafe(x) for x in v]
    if isinstance(v, np.ndarray): return jsafe(v.tolist())
    return str(v)


def main():
    set_all_seeds(42)
    model, tokenizer = load_model("1b", device="cuda")
    device = str(next(model.parameters()).device)
    layers = get_model_layers(model)
    n_layers = len(layers)
    cfg = getattr(model.config, "text_config", model.config)
    n_heads = cfg.num_attention_heads
    head_dim = cfg.head_dim
    ltypes = [str(x) for x in getattr(cfg, "layer_types", [])]
    gl = [i for i, lt in enumerate(ltypes) if "full" in lt]
    local = [i for i in range(n_layers) if i not in gl]

    log("Architecture: %d layers, %d heads, global=%s" % (n_layers, n_heads, gl))
    log("=== CORRECTED ATTENTION-CONTRIBUTION PATCHING ===")
    log("Method: patch ALL heads at target layers via o_proj pre-hook")

    PAIRS = [
        ("aksharantar_hin_latin", "Hindi", "Latin", "Hindi", "Devanagari"),
        ("aksharantar_tel_latin", "Telugu", "Latin", "Telugu", "Telugu"),
    ]

    subsets = {
        "single_L05": [5],
        "single_L11": [11],
        "single_L14": [14],
        "single_L17": [17],
        "single_L23": [23],
        "first_two_global": gl[:2],          # L5, L11
        "last_two_global": gl[2:],            # L17, L23
        "all_global_attn": gl,                # L5, L11, L17, L23
        "all_local_attn": local,              # 22 local layers
        "all_attn": list(range(n_layers)),    # all 26
    }

    all_results = {}
    for pair_id, lang, src_s, src_l, tgt_s in PAIRS:
        log("--- %s ---" % lang)
        pb = load_pair(pair_id)
        icl_ex, evals = pb["icl_examples"], list(pb["eval_rows"][:50])

        lang_results = {}
        for sname, slayers in subsets.items():
            pe_vals = []
            for word in evals:
                query, target = str(word["ood"]), str(word["hindi"])
                tids = tokenizer.encode(target, add_special_tokens=False)
                if not tids:
                    continue
                gold = int(tids[0])

                icl_r = build_task_prompt(query, icl_ex, input_script_name=src_s,
                                          source_language=src_l, output_script_name=tgt_s)
                zs_r = build_task_prompt(query, None, input_script_name=src_s,
                                         source_language=src_l, output_script_name=tgt_s)
                ids_i = tokenizer(apply_chat_template(tokenizer, icl_r), return_tensors="pt").to(device).input_ids
                ids_z = tokenizer(apply_chat_template(tokenizer, zs_r), return_tensors="pt").to(device).input_ids
                pi, pz = int(ids_i.shape[1] - 1), int(ids_z.shape[1] - 1)

                # Step 1: Capture ICL attention outputs at target layers (all heads)
                # Use o_proj pre-hook to capture the concatenated head outputs
                icl_head_outs = {}

                def make_cap(li):
                    def hook(m, args, kwargs):
                        # args[0] is the concatenated head output [batch, seq, hidden]
                        icl_head_outs[li] = args[0][0, pi, :].detach().clone()
                        return args, kwargs
                    return hook

                handles = []
                for li in slayers:
                    h = layers[li].self_attn.o_proj.register_forward_pre_hook(
                        make_cap(li), with_kwargs=True)
                    handles.append(h)
                with torch.inference_mode():
                    model(input_ids=ids_i, use_cache=False)
                for h in handles:
                    h.remove()

                # Step 2: ZS baseline
                with torch.inference_mode():
                    zo = model(input_ids=ids_z, use_cache=False)
                zp = float(torch.softmax(zo.logits[0, pz], dim=-1)[gold].item())

                # Step 3: Patched — replace ALL heads at target layers
                def make_patch(li, replacement):
                    def hook(m, args, kwargs):
                        patched = args[0].clone()
                        patched[0, pz, :] = replacement.to(device=patched.device, dtype=patched.dtype)
                        return (patched,) + args[1:], kwargs
                    return hook

                handles = []
                for li in slayers:
                    if li in icl_head_outs:
                        h = layers[li].self_attn.o_proj.register_forward_pre_hook(
                            make_patch(li, icl_head_outs[li]), with_kwargs=True)
                        handles.append(h)
                with torch.inference_mode():
                    po = model(input_ids=ids_z, use_cache=False)
                pp = float(torch.softmax(po.logits[0, pz], dim=-1)[gold].item())
                for h in handles:
                    h.remove()

                pe_vals.append(pp - zp)

            ci = bootstrap_ci(pe_vals)
            frac_pos = sum(1 for v in pe_vals if v > 0) / len(pe_vals) if pe_vals else 0
            lang_results[sname] = {
                "layers": slayers, "n_layers": len(slayers), "n_items": len(pe_vals),
                "pe": ci, "frac_positive": frac_pos,
            }
            log("  %-22s (%2d layers): pe=%.4f [%.4f, %.4f] frac_pos=%.0f%%" % (
                sname, len(slayers), ci["mean"],
                ci.get("ci_lo", 0) or 0, ci.get("ci_hi", 0) or 0, frac_pos * 100))

        all_results[lang] = lang_results

    p = OUT / "attention_contribution_patching.json"
    p.write_text(json.dumps(jsafe({"experiment": "attention_contribution_patching",
                                    "method": "o_proj_pre_hook_all_heads",
                                    "global_layers": gl, "results": all_results}),
                             indent=2, ensure_ascii=False), encoding="utf-8")
    log("Saved: %s" % p.name)
    log("=== DONE ===")


if __name__ == "__main__":
    main()
