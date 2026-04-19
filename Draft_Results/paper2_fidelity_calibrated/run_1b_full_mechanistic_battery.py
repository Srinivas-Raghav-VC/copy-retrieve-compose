#!/usr/bin/env python3
"""
1B complete mechanistic battery — publication-ready N.
Runs ALL 1B experiments sequentially with N=50 items per condition.

Experiments:
  1. Matched-control logit lens (5 languages × 50 items)
  2. Head attribution (Hindi, Telugu × 30 items)
  3. Multi-layer activation patching with GENERATION verification (Hindi × 50 items)
  4. MLP layer contribution (Hindi × 30 items)
  5. Cross-language head attribution (5 languages × 20 items)
  6. Sliding window attention verification (1 item, detailed)
"""
from __future__ import annotations
import argparse, json, sys, time, random
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import Counter

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import (
    apply_chat_template, build_task_prompt, build_corrupted_icl_prompt,
    build_random_icl_prompt, build_null_icl_prompt,
    get_model_layers, load_model, set_all_seeds,
    _extract_layer_output_at_position_from_input_ids,
    _extract_mlp_io_at_position_from_input_ids,
    register_layer_output_replace_hook,
    register_dense_mlp_output_patch_hook,
)

OUT_DIR = PROJECT_ROOT / "paper2_fidelity_calibrated" / "results" / "1b_mechanistic_analysis"


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def _json_safe(v):
    if isinstance(v, (str, int, bool)) or v is None: return v
    if isinstance(v, float): return v if np.isfinite(v) else None
    if isinstance(v, dict): return {str(k): _json_safe(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)): return [_json_safe(x) for x in v]
    return v


def save(name, data):
    p = OUT_DIR / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_json_safe(data), indent=2, ensure_ascii=False), encoding="utf-8")
    log("Saved: %s" % p)


def normalize_text(s):
    return s.strip().split("\n")[0].strip()


def load_pair(pair_id, seed=42, n_icl=16, n_select=300, n_eval=200):
    """Load pair with fallback for unregistered pairs."""
    try:
        from paper2_fidelity_calibrated.phase1_common import load_pair_split
        return load_pair_split(pair_id, seed=seed, n_icl=n_icl, n_select=n_select,
                               n_eval=n_eval, external_only=True,
                               require_external_sources=True, min_pool_size=100)
    except (ValueError, KeyError):
        pass
    data_path = PROJECT_ROOT / "data" / "transliteration" / ("%s.jsonl" % pair_id)
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
        "pair": pair_id, "words": words,
        "source_language": str(ds.get("source_language", "Hindi")),
        "input_script_name": str(ds.get("source_script", "Latin")),
        "output_script_name": str(ds.get("target_script", "Devanagari")),
        "icl_examples": words[:n_icl],
        "select_rows": words[n_icl:n_icl+n_select],
        "eval_rows": words[n_icl+n_select:n_icl+n_select+n_eval],
    }


PAIRS = [
    ("aksharantar_hin_latin", "Hindi"),
    ("aksharantar_tel_latin", "Telugu"),
    ("aksharantar_ben_latin", "Bengali"),
    ("aksharantar_kan_latin", "Kannada"),
    ("aksharantar_guj_latin", "Gujarati"),
]


def get_final_norm(model):
    for chain in [("model","norm"),("language_model","norm"),("model","model","norm")]:
        cur = model
        ok = True
        for name in chain:
            if not hasattr(cur, name): ok = False; break
            cur = getattr(cur, name)
        if ok: return cur
    raise AttributeError("no final norm")


# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Matched-control logit lens (all languages, N=50)
# ══════════════════════════════════════════════════════════════════════

def run_logit_lens(model, tokenizer, device, pair_id, lang_name, n_items=50):
    log("=== LOGIT LENS: %s (N=%d) ===" % (lang_name, n_items))
    pb = load_pair(pair_id, n_eval=n_items + 50)
    icl_ex = pb["icl_examples"]
    eval_rows = list(pb["eval_rows"][:n_items])
    src_s, src_l, tgt_s = pb["input_script_name"], pb["source_language"], pb["output_script_name"]

    layers_mod = get_model_layers(model)
    n_layers = len(layers_mod)
    lm_head = model.lm_head if hasattr(model, "lm_head") else model.get_output_embeddings()
    final_norm = get_final_norm(model)
    lm_dtype = next(lm_head.parameters()).dtype

    config = getattr(model.config, "text_config", model.config)
    layer_types = getattr(config, "layer_types", [])

    all_items = []
    for idx, word in enumerate(eval_rows):
        query, target = str(word["ood"]), str(word["hindi"])
        target_ids = tokenizer.encode(target, add_special_tokens=False)
        if not target_ids: continue
        gold_id = int(target_ids[0])

        conds = {}
        helpful_raw = build_task_prompt(query, icl_ex, input_script_name=src_s, source_language=src_l, output_script_name=tgt_s)
        conds["helpful_icl"] = apply_chat_template(tokenizer, helpful_raw)
        corrupt_raw = build_corrupted_icl_prompt(query, icl_ex, input_script_name=src_s, source_language=src_l, output_script_name=tgt_s, seed=42)
        conds["corrupt_icl"] = apply_chat_template(tokenizer, corrupt_raw)
        zs_raw = build_task_prompt(query, None, input_script_name=src_s, source_language=src_l, output_script_name=tgt_s)
        conds["zs"] = apply_chat_template(tokenizer, zs_raw)

        item = {"idx": idx, "ood": query, "hindi": target, "gold_id": gold_id, "trajectories": {}, "lengths": {}}
        for cname, rendered in conds.items():
            ids = tokenizer(rendered, return_tensors="pt").to(device).input_ids
            last_pos = int(ids.shape[1] - 1)
            item["lengths"][cname] = int(ids.shape[1])

            hidden = {}
            def mkhook(li):
                def hook(m, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    hidden[li] = h[0, last_pos, :].detach()
                return hook
            handles = [layers_mod[i].register_forward_hook(mkhook(i)) for i in range(n_layers)]
            with torch.inference_mode():
                outputs = model(input_ids=ids, use_cache=False)
            for h in handles: h.remove()

            traj = []
            for li in range(n_layers):
                h = hidden[li].to(dtype=lm_dtype)
                normed = final_norm(h.unsqueeze(0).unsqueeze(0)).squeeze()
                logits = lm_head(normed.unsqueeze(0)).squeeze().float()
                probs = torch.softmax(logits, dim=-1)
                rank = int((torch.argsort(logits, descending=True) == gold_id).nonzero(as_tuple=True)[0].item()) + 1
                traj.append({"layer": li, "rank": rank, "prob": float(probs[gold_id].item())})
            item["trajectories"][cname] = traj
        all_items.append(item)
        if (idx+1) % 10 == 0:
            log("  %s: %d/%d" % (lang_name, idx+1, len(eval_rows)))

    # Summary
    summary = []
    for cname in ["helpful_icl", "corrupt_icl", "zs"]:
        for li in range(n_layers):
            ranks = [it["trajectories"][cname][li]["rank"] for it in all_items if cname in it["trajectories"]]
            probs = [it["trajectories"][cname][li]["prob"] for it in all_items if cname in it["trajectories"]]
            if ranks:
                summary.append({"condition": cname, "layer": li, "n": len(ranks),
                                "mean_rank": float(np.mean(ranks)), "median_rank": float(np.median(ranks)),
                                "mean_prob": float(np.mean(probs))})

    save("logit_lens_%s_N%d.json" % (lang_name.lower(), len(all_items)),
         {"experiment": "matched_control_logit_lens", "lang": lang_name, "pair": pair_id,
          "n_items": len(all_items), "n_layers": n_layers, "layer_types": [str(lt) for lt in layer_types],
          "summary": summary, "items": all_items})
    return summary


# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Multi-layer patching WITH generation (Hindi, N=50)
# ══════════════════════════════════════════════════════════════════════

def run_multi_layer_patching(model, tokenizer, device, n_items=50):
    log("=== MULTI-LAYER PATCHING WITH GENERATION (N=%d) ===" % n_items)
    pb = load_pair("aksharantar_hin_latin", n_eval=n_items + 50)
    icl_ex = pb["icl_examples"]
    eval_rows = list(pb["eval_rows"][:n_items])

    config = getattr(model.config, "text_config", model.config)
    layer_types = getattr(config, "layer_types", [])
    global_layers = [i for i, lt in enumerate(layer_types) if "full" in str(lt)]
    n_layers = len(get_model_layers(model))

    subsets = {
        "single_best_global_L11": [11],
        "global_only": global_layers,
        "late_half": list(range(13, 26)),
        "all_layers": list(range(26)),
    }

    results = {}
    for sname, slayers in subsets.items():
        pe_vals, gen_results = [], []
        for widx, word in enumerate(eval_rows):
            query, target = str(word["ood"]), str(word["hindi"])
            target_ids = tokenizer.encode(target, add_special_tokens=False)
            if not target_ids: continue
            gold_id = int(target_ids[0])

            icl_raw = build_task_prompt(query, icl_ex, input_script_name="Latin", source_language="Hindi", output_script_name="Devanagari")
            zs_raw = build_task_prompt(query, None, input_script_name="Latin", source_language="Hindi", output_script_name="Devanagari")
            ids_icl = tokenizer(apply_chat_template(tokenizer, icl_raw), return_tensors="pt").to(device).input_ids
            ids_zs = tokenizer(apply_chat_template(tokenizer, zs_raw), return_tensors="pt").to(device).input_ids
            pos_icl, pos_zs = int(ids_icl.shape[1]-1), int(ids_zs.shape[1]-1)

            # Extract ICL layer outputs
            icl_outs = {}
            for li in slayers:
                icl_outs[li] = _extract_layer_output_at_position_from_input_ids(model, ids_icl, li, pos_icl).detach()

            # ZS baseline prob
            with torch.inference_mode():
                zs_out = model(input_ids=ids_zs, use_cache=False)
            zs_prob = float(torch.softmax(zs_out.logits[0, pos_zs], dim=-1)[gold_id].item())

            # ICL baseline prob
            with torch.inference_mode():
                icl_out = model(input_ids=ids_icl, use_cache=False)
            icl_prob = float(torch.softmax(icl_out.logits[0, pos_icl], dim=-1)[gold_id].item())

            # Patch all layers simultaneously
            hooks = [register_layer_output_replace_hook(model, li, icl_outs[li], patch_position=pos_zs) for li in slayers]
            with torch.inference_mode():
                patched_out = model(input_ids=ids_zs, use_cache=False)
            patched_prob = float(torch.softmax(patched_out.logits[0, pos_zs], dim=-1)[gold_id].item())

            # Also GENERATE with patching (the key missing verification)
            pad_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", 0)
            with torch.inference_mode():
                gen_out = model.generate(ids_zs, attention_mask=torch.ones_like(ids_zs),
                                          max_new_tokens=16, do_sample=False, pad_token_id=int(pad_id))
            gen_text = normalize_text(tokenizer.decode(gen_out[0, ids_zs.shape[1]:], skip_special_tokens=True))

            for h in hooks: h.remove()

            pe = patched_prob - zs_prob
            rescue_frac = pe / max(icl_prob - zs_prob, 1e-8) if (icl_prob - zs_prob) > 1e-8 else 0.0
            em = float(normalize_text(gen_text) == normalize_text(target))

            pe_vals.append(pe)
            gen_results.append({"ood": query, "gold": target, "pred": gen_text, "em": em,
                                "zs_prob": zs_prob, "icl_prob": icl_prob, "patched_prob": patched_prob,
                                "pe": pe, "rescue_frac": rescue_frac})

        mean_pe = float(np.mean(pe_vals)) if pe_vals else 0
        mean_em = float(np.mean([r["em"] for r in gen_results])) if gen_results else 0
        mean_rescue = float(np.mean([r["rescue_frac"] for r in gen_results])) if gen_results else 0
        results[sname] = {"layers": slayers, "n_layers": len(slayers), "n_items": len(gen_results),
                          "mean_pe": mean_pe, "mean_em": mean_em, "mean_rescue_frac": mean_rescue,
                          "items": gen_results}
        log("  %s (%d layers): pe=%.4f EM=%.1f%% rescue=%.1f%%" % (sname, len(slayers), mean_pe, mean_em*100, mean_rescue*100))

    save("multi_layer_patching_with_generation_N%d.json" % len(eval_rows),
         {"experiment": "multi_layer_patching_with_generation", "global_layers": global_layers, "results": results})
    return results


# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: Head attribution (Hindi + Telugu, N=30)
# ══════════════════════════════════════════════════════════════════════

def run_head_attribution(model, tokenizer, device, pair_id, lang_name, n_items=30):
    log("=== HEAD ATTRIBUTION: %s (N=%d) ===" % (lang_name, n_items))
    pb = load_pair(pair_id, n_eval=n_items + 50)
    icl_ex = pb["icl_examples"]
    eval_rows = list(pb["eval_rows"][:n_items])
    src_s, src_l, tgt_s = pb["input_script_name"], pb["source_language"], pb["output_script_name"]

    layers_mod = get_model_layers(model)
    n_layers = len(layers_mod)
    config = getattr(model.config, "text_config", model.config)
    n_heads = config.num_attention_heads
    head_dim = config.head_dim
    layer_types = getattr(config, "layer_types", [])
    global_set = {i for i, lt in enumerate(layer_types) if "full" in str(lt)}

    head_effects = torch.zeros(n_layers, n_heads, dtype=torch.float32)
    n_valid = 0

    for widx, word in enumerate(eval_rows):
        query, target = str(word["ood"]), str(word["hindi"])
        target_ids = tokenizer.encode(target, add_special_tokens=False)
        if not target_ids: continue
        gold_id = int(target_ids[0])

        icl_raw = build_task_prompt(query, icl_ex, input_script_name=src_s, source_language=src_l, output_script_name=tgt_s)
        zs_raw = build_task_prompt(query, None, input_script_name=src_s, source_language=src_l, output_script_name=tgt_s)
        ids_icl = tokenizer(apply_chat_template(tokenizer, icl_raw), return_tensors="pt").to(device).input_ids
        ids_zs = tokenizer(apply_chat_template(tokenizer, zs_raw), return_tensors="pt").to(device).input_ids
        pos_icl, pos_zs = int(ids_icl.shape[1]-1), int(ids_zs.shape[1]-1)

        clean_h = {}
        def mkcap(li):
            def hook(m, a, kw):
                clean_h[li] = a[0][0, pos_icl, :].detach().clone().view(n_heads, head_dim)
                return a, kw
            return hook
        handles = [layers_mod[i].self_attn.o_proj.register_forward_pre_hook(mkcap(i), with_kwargs=True) for i in range(n_layers)]
        with torch.inference_mode():
            out_icl = model(input_ids=ids_icl, use_cache=False)
        icl_logit = float(out_icl.logits[0, pos_icl, gold_id].item())
        for h in handles: h.remove()

        with torch.inference_mode():
            out_zs = model(input_ids=ids_zs, use_cache=False)
        zs_logit = float(out_zs.logits[0, pos_zs, gold_id].item())

        te = icl_logit - zs_logit
        if te <= 0: continue
        n_valid += 1

        for li in range(n_layers):
            for hi in range(n_heads):
                def mkp(cv, th):
                    def hook(m, a, kw):
                        h = a[0].clone()
                        r = h[0, pos_zs, :].view(n_heads, head_dim)
                        r[th] = cv.to(device=r.device, dtype=r.dtype)
                        h[0, pos_zs, :] = r.view(-1)
                        return (h,) + a[1:], kw
                    return hook
                handle = layers_mod[li].self_attn.o_proj.register_forward_pre_hook(mkp(clean_h[li][hi], hi), with_kwargs=True)
                with torch.inference_mode():
                    po = model(input_ids=ids_zs, use_cache=False)
                handle.remove()
                head_effects[li, hi] += (float(po.logits[0, pos_zs, gold_id].item()) - zs_logit) / te

        if (widx+1) % 10 == 0:
            log("  %s: %d/%d (valid=%d)" % (lang_name, widx+1, len(eval_rows), n_valid))

    if n_valid > 0:
        head_effects /= n_valid

    flat = head_effects.flatten()
    topk = torch.topk(flat, 15)
    top_heads = []
    for rank, (sc, idx_flat) in enumerate(zip(topk.values.tolist(), topk.indices.tolist()), 1):
        li, hi = idx_flat // n_heads, idx_flat % n_heads
        top_heads.append({"rank": rank, "layer": li, "head": hi,
                          "type": "GLOBAL" if li in global_set else "local",
                          "effect": round(sc, 6)})

    save("head_attribution_%s_N%d.json" % (lang_name.lower(), n_valid),
         {"experiment": "head_attribution", "lang": lang_name, "pair": pair_id,
          "n_valid": n_valid, "n_layers": n_layers, "n_heads": n_heads,
          "global_layers": sorted(global_set), "top_heads": top_heads,
          "full_matrix": head_effects.tolist()})
    return top_heads


# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 4: MLP layer contribution (Hindi, N=30)
# ══════════════════════════════════════════════════════════════════════

def run_mlp_contribution(model, tokenizer, device, n_items=30):
    log("=== MLP CONTRIBUTION (N=%d) ===" % n_items)
    pb = load_pair("aksharantar_hin_latin", n_eval=n_items + 50)
    icl_ex = pb["icl_examples"]
    eval_rows = list(pb["eval_rows"][:n_items])

    config = getattr(model.config, "text_config", model.config)
    n_layers = config.num_hidden_layers
    layer_types = getattr(config, "layer_types", [])

    mlp_results = []
    for layer in range(n_layers):
        pe_vals = []
        for word in eval_rows:
            query, target = str(word["ood"]), str(word["hindi"])
            target_ids = tokenizer.encode(target, add_special_tokens=False)
            if not target_ids: continue
            gold_id = int(target_ids[0])

            icl_raw = build_task_prompt(query, icl_ex, input_script_name="Latin", source_language="Hindi", output_script_name="Devanagari")
            zs_raw = build_task_prompt(query, None, input_script_name="Latin", source_language="Hindi", output_script_name="Devanagari")
            ids_icl = tokenizer(apply_chat_template(tokenizer, icl_raw), return_tensors="pt").to(device).input_ids
            ids_zs = tokenizer(apply_chat_template(tokenizer, zs_raw), return_tensors="pt").to(device).input_ids
            pos_icl, pos_zs = int(ids_icl.shape[1]-1), int(ids_zs.shape[1]-1)

            _, mlp_out_icl = _extract_mlp_io_at_position_from_input_ids(model, ids_icl, layer, pos_icl)
            with torch.inference_mode():
                zs_out = model(input_ids=ids_zs, use_cache=False)
            zs_prob = float(torch.softmax(zs_out.logits[0, pos_zs], dim=-1)[gold_id].item())

            hook = register_dense_mlp_output_patch_hook(model, layer, mlp_out_icl.detach(), patch_position=pos_zs)
            with torch.inference_mode():
                patched_out = model(input_ids=ids_zs, use_cache=False)
            patched_prob = float(torch.softmax(patched_out.logits[0, pos_zs], dim=-1)[gold_id].item())
            hook.remove()
            pe_vals.append(patched_prob - zs_prob)

        lt = "GLOBAL" if "full" in str(layer_types[layer]) if layer < len(layer_types) else "" else "local"
        mean_pe = float(np.mean(pe_vals)) if pe_vals else 0
        mlp_results.append({"layer": layer, "type": lt, "mean_pe": round(mean_pe, 6), "n": len(pe_vals)})
        if layer % 5 == 0:
            log("  L%02d: pe=%.6f" % (layer, mean_pe))

    save("mlp_contribution_hindi_N%d.json" % n_items,
         {"experiment": "mlp_contribution", "layers": mlp_results})
    return mlp_results


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--logit-lens-n", type=int, default=50)
    ap.add_argument("--patching-n", type=int, default=50)
    ap.add_argument("--attribution-n", type=int, default=30)
    ap.add_argument("--mlp-n", type=int, default=30)
    args = ap.parse_args()

    set_all_seeds(42)
    model, tokenizer = load_model("1b", device=args.device)
    device = str(next(model.parameters()).device)
    if hasattr(model, "set_attn_implementation"):
        try: model.set_attn_implementation("eager")
        except: pass

    log("=" * 70)
    log("1B COMPLETE MECHANISTIC BATTERY")
    log("=" * 70)

    # 1. Logit lens for all languages
    for pair_id, lang in PAIRS:
        try:
            run_logit_lens(model, tokenizer, device, pair_id, lang, n_items=args.logit_lens_n)
        except Exception as e:
            log("FAILED %s: %s" % (lang, str(e)[:200]))

    # 2. Multi-layer patching with generation
    run_multi_layer_patching(model, tokenizer, device, n_items=args.patching_n)

    # 3. Head attribution for Hindi and Telugu
    for pair_id, lang in PAIRS[:2]:
        run_head_attribution(model, tokenizer, device, pair_id, lang, n_items=args.attribution_n)

    # 4. MLP contribution
    run_mlp_contribution(model, tokenizer, device, n_items=args.mlp_n)

    log("=" * 70)
    log("ALL 1B EXPERIMENTS COMPLETE")
    log("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
