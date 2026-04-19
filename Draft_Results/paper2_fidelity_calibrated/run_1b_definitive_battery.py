#!/usr/bin/env python3
"""
1B DEFINITIVE MECHANISTIC BATTERY
==================================
The complete, final set of 1B experiments at publication-ready sample sizes.

Experiments (in order):
  1. Matched-control logit lens — 5 languages × N=50
  2. Head attribution — 5 languages × N=20
  3. Multi-layer patching with GENERATION — Hindi N=50
  4. Cross-language multi-layer patching (global-only) — 4 langs × N=30
  5. MLP layer contribution — Hindi N=30
  6. Density-dependent attention dilution analysis — Hindi, 5 density levels
  7. Telugu L25 anomaly investigation — N=50 token-level analysis
  8. Key head attention patterns — Hindi, what do L11H0/L14H0 attend to?

Output: JSON files in paper2_fidelity_calibrated/results/1b_definitive/
"""
from __future__ import annotations
import json, sys, time, random, traceback
from pathlib import Path
from collections import Counter

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import (
    apply_chat_template, build_task_prompt, build_corrupted_icl_prompt,
    get_model_layers, load_model, set_all_seeds,
    _extract_layer_output_at_position_from_input_ids,
    _extract_mlp_io_at_position_from_input_ids,
    register_layer_output_replace_hook,
    register_dense_mlp_output_patch_hook,
)

OUT = ROOT / "paper2_fidelity_calibrated" / "results" / "1b_definitive"
OUT.mkdir(parents=True, exist_ok=True)

PAIRS = [
    ("aksharantar_hin_latin", "Hindi", "Latin", "Hindi", "Devanagari"),
    ("aksharantar_tel_latin", "Telugu", "Latin", "Telugu", "Telugu"),
    ("aksharantar_ben_latin", "Bengali", "Latin", "Bengali", "Bangla"),
    ("aksharantar_kan_latin", "Kannada", "Latin", "Kannada", "Kannada"),
    ("aksharantar_guj_latin", "Gujarati", "Latin", "Gujarati", "Gujarati"),
]


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def jsafe(v):
    if isinstance(v, (str, int, bool)) or v is None: return v
    if isinstance(v, float): return v if np.isfinite(v) else None
    if isinstance(v, np.floating): return float(v) if np.isfinite(v) else None
    if isinstance(v, np.integer): return int(v)
    if isinstance(v, dict): return {str(k): jsafe(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)): return [jsafe(x) for x in v]
    if isinstance(v, np.ndarray): return jsafe(v.tolist())
    return str(v)


def save(name, data):
    p = OUT / name
    p.write_text(json.dumps(jsafe(data), indent=2, ensure_ascii=False), encoding="utf-8")
    log("  Saved %s" % p.name)


def load_pair(pair_id, seed=42, n_icl=16, n_pool=600):
    """Load pair data with fallback for unregistered pairs."""
    try:
        from paper2_fidelity_calibrated.phase1_common import load_pair_split
        pb = load_pair_split(pair_id, seed=seed, n_icl=n_icl, n_select=300, n_eval=200,
                             external_only=True, require_external_sources=True, min_pool_size=100)
        return pb
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


def norm(s):
    return s.strip().split("\n")[0].strip()


def bootstrap_ci(vals, n_boot=2000, ci=0.95):
    """Bootstrap confidence interval."""
    vals = np.array(vals, dtype=float)
    if len(vals) < 3:
        return {"mean": float(np.mean(vals)), "ci_lo": float(np.min(vals)),
                "ci_hi": float(np.max(vals)), "std": float(np.std(vals)), "n": len(vals)}
    boots = np.array([np.mean(np.random.choice(vals, size=len(vals), replace=True))
                      for _ in range(n_boot)])
    lo = float(np.percentile(boots, (1-ci)/2 * 100))
    hi = float(np.percentile(boots, (1+ci)/2 * 100))
    return {"mean": float(np.mean(vals)), "ci_lo": lo, "ci_hi": hi,
            "std": float(np.std(vals)), "n": len(vals)}


# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: MATCHED-CONTROL LOGIT LENS (5 langs × N=50)
# ══════════════════════════════════════════════════════════════════════
def exp1_logit_lens(model, tokenizer, device, pair_id, lang, src_s, src_l, tgt_s, N=50):
    log("EXP1 LOGIT-LENS: %s N=%d" % (lang, N))
    pb = load_pair(pair_id)
    icl_ex, evals = pb["icl_examples"], list(pb["eval_rows"][:N])
    layers = get_model_layers(model)
    n_layers = len(layers)
    cfg = getattr(model.config, "text_config", model.config)
    ltypes = [str(x) for x in getattr(cfg, "layer_types", [])]
    lm_head = model.lm_head if hasattr(model, "lm_head") else model.get_output_embeddings()
    lm_dtype = next(lm_head.parameters()).dtype

    # Find final norm
    for chain in [("model","norm"),("language_model","model","norm")]:
        cur = model
        ok = True
        for nm in chain:
            if not hasattr(cur, nm): ok=False; break
            cur = getattr(cur, nm)
        if ok: final_norm = cur; break

    items = []
    for idx, word in enumerate(evals):
        query, target = str(word["ood"]), str(word["hindi"])
        tids = tokenizer.encode(target, add_special_tokens=False)
        if not tids: continue
        gold = int(tids[0])

        prompts = {}
        # Helpful ICL
        r = build_task_prompt(query, icl_ex, input_script_name=src_s, source_language=src_l, output_script_name=tgt_s)
        prompts["helpful"] = apply_chat_template(tokenizer, r)
        # Corrupt ICL (shuffled outputs)
        r2 = build_corrupted_icl_prompt(query, icl_ex, input_script_name=src_s, source_language=src_l, output_script_name=tgt_s, seed=42)
        prompts["corrupt"] = apply_chat_template(tokenizer, r2)
        # Zero-shot
        r3 = build_task_prompt(query, None, input_script_name=src_s, source_language=src_l, output_script_name=tgt_s)
        prompts["zs"] = apply_chat_template(tokenizer, r3)

        item = {"idx": idx, "ood": query, "target": target, "gold_id": gold, "trajectories": {}, "lengths": {}}
        for cname, rendered in prompts.items():
            ids = tokenizer(rendered, return_tensors="pt").to(device).input_ids
            lp = int(ids.shape[1] - 1)
            item["lengths"][cname] = int(ids.shape[1])

            hidden = {}
            def mkhk(li):
                def hook(m, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    hidden[li] = h[0, lp, :].detach()
                return hook
            hs = [layers[i].register_forward_hook(mkhk(i)) for i in range(n_layers)]
            with torch.inference_mode():
                model(input_ids=ids, use_cache=False)
            for h in hs: h.remove()

            traj = []
            for li in range(n_layers):
                h = hidden[li].to(dtype=lm_dtype)
                normed = final_norm(h.unsqueeze(0).unsqueeze(0)).squeeze()
                logits = lm_head(normed.unsqueeze(0)).squeeze().float()
                probs = torch.softmax(logits, dim=-1)
                rank = int((torch.argsort(logits, descending=True) == gold).nonzero(as_tuple=True)[0].item()) + 1
                prob = float(probs[gold].item())
                traj.append({"layer": li, "rank": rank, "prob": prob})
            item["trajectories"][cname] = traj
        items.append(item)
        if (idx+1) % 10 == 0: log("  %s %d/%d done" % (lang, idx+1, len(evals)))

    # Aggregate summary
    summary = {}
    for cname in ["helpful", "corrupt", "zs"]:
        by_layer = []
        for li in range(n_layers):
            ranks = [it["trajectories"][cname][li]["rank"] for it in items if cname in it["trajectories"]]
            probs = [it["trajectories"][cname][li]["prob"] for it in items if cname in it["trajectories"]]
            by_layer.append({
                "layer": li, "type": ltypes[li] if li < len(ltypes) else "?",
                "rank": bootstrap_ci(ranks), "prob": bootstrap_ci(probs),
            })
        summary[cname] = by_layer

    save("logit_lens_%s.json" % lang.lower(), {
        "lang": lang, "pair": pair_id, "n_items": len(items), "n_layers": n_layers,
        "layer_types": ltypes, "summary": summary, "items": items,
    })
    return summary


# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: HEAD ATTRIBUTION (5 langs × N=20)
# ══════════════════════════════════════════════════════════════════════
def exp2_head_attribution(model, tokenizer, device, pair_id, lang, src_s, src_l, tgt_s, N=20):
    log("EXP2 HEAD-ATTR: %s N=%d" % (lang, N))
    pb = load_pair(pair_id)
    icl_ex, evals = pb["icl_examples"], list(pb["eval_rows"][:N])
    layers = get_model_layers(model)
    n_layers = len(layers)
    cfg = getattr(model.config, "text_config", model.config)
    n_heads, hdim = cfg.num_attention_heads, cfg.head_dim
    ltypes = [str(x) for x in getattr(cfg, "layer_types", [])]
    gset = {i for i, lt in enumerate(ltypes) if "full" in lt}

    effects = torch.zeros(n_layers, n_heads)
    item_effects = []  # per-item for bootstrap
    n_valid = 0

    for widx, word in enumerate(evals):
        query, target = str(word["ood"]), str(word["hindi"])
        tids = tokenizer.encode(target, add_special_tokens=False)
        if not tids: continue
        gold = int(tids[0])

        icl_r = build_task_prompt(query, icl_ex, input_script_name=src_s, source_language=src_l, output_script_name=tgt_s)
        zs_r = build_task_prompt(query, None, input_script_name=src_s, source_language=src_l, output_script_name=tgt_s)
        ids_i = tokenizer(apply_chat_template(tokenizer, icl_r), return_tensors="pt").to(device).input_ids
        ids_z = tokenizer(apply_chat_template(tokenizer, zs_r), return_tensors="pt").to(device).input_ids
        pi, pz = int(ids_i.shape[1]-1), int(ids_z.shape[1]-1)

        # Capture ICL head outputs
        ch = {}
        def mc(li):
            def hook(m, a, kw):
                ch[li] = a[0][0, pi, :].detach().clone().view(n_heads, hdim)
                return a, kw
            return hook
        hs = [layers[i].self_attn.o_proj.register_forward_pre_hook(mc(i), with_kwargs=True) for i in range(n_layers)]
        with torch.inference_mode():
            oi = model(input_ids=ids_i, use_cache=False)
        il = float(oi.logits[0, pi, gold].item())
        for h in hs: h.remove()

        with torch.inference_mode():
            oz = model(input_ids=ids_z, use_cache=False)
        zl = float(oz.logits[0, pz, gold].item())
        te = il - zl
        if te <= 0: continue
        n_valid += 1
        item_eff = torch.zeros(n_layers, n_heads)

        for li in range(n_layers):
            for hi in range(n_heads):
                def mp(cv, th):
                    def hook(m, a, kw):
                        h = a[0].clone()
                        r = h[0, pz, :].view(n_heads, hdim)
                        r[th] = cv.to(device=r.device, dtype=r.dtype)
                        h[0, pz, :] = r.view(-1)
                        return (h,)+a[1:], kw
                    return hook
                hh = layers[li].self_attn.o_proj.register_forward_pre_hook(mp(ch[li][hi], hi), with_kwargs=True)
                with torch.inference_mode():
                    po = model(input_ids=ids_z, use_cache=False)
                hh.remove()
                eff = (float(po.logits[0, pz, gold].item()) - zl) / te
                effects[li, hi] += eff
                item_eff[li, hi] = eff

        item_effects.append(item_eff.tolist())
        if (widx+1) % 5 == 0: log("  %s %d/%d (valid=%d)" % (lang, widx+1, len(evals), n_valid))

    if n_valid > 0: effects /= n_valid

    # Top heads with bootstrap CI
    flat = effects.flatten()
    topk = torch.topk(flat, min(20, flat.numel()))
    top = []
    for rank, (sc, fi) in enumerate(zip(topk.values.tolist(), topk.indices.tolist()), 1):
        li, hi = fi // n_heads, fi % n_heads
        # Bootstrap from per-item effects
        if item_effects:
            vals = [ie[li][hi] for ie in item_effects]
            ci = bootstrap_ci(vals)
        else:
            ci = {"mean": sc, "ci_lo": sc, "ci_hi": sc, "std": 0, "n": 0}
        top.append({"rank": rank, "layer": li, "head": hi,
                     "type": "GLOBAL" if li in gset else "local",
                     "effect": round(sc, 6), "bootstrap": ci})

    save("head_attribution_%s.json" % lang.lower(), {
        "lang": lang, "pair": pair_id, "n_valid": n_valid, "n_total": len(evals),
        "n_layers": n_layers, "n_heads": n_heads, "global_layers": sorted(gset),
        "top_heads": top, "full_matrix": effects.tolist(),
    })
    return top


# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: MULTI-LAYER PATCHING + GENERATION (Hindi N=50)
# ══════════════════════════════════════════════════════════════════════
def exp3_multilayer_patching(model, tokenizer, device, pair_id, lang, src_s, src_l, tgt_s, N=50):
    log("EXP3 MULTI-LAYER-PATCH+GEN: %s N=%d" % (lang, N))
    pb = load_pair(pair_id)
    icl_ex, evals = pb["icl_examples"], list(pb["eval_rows"][:N])
    cfg = getattr(model.config, "text_config", model.config)
    ltypes = [str(x) for x in getattr(cfg, "layer_types", [])]
    gl = [i for i, lt in enumerate(ltypes) if "full" in lt]
    n_layers = cfg.num_hidden_layers

    subsets = {
        "single_L05": [5],
        "single_L11": [11],
        "single_L17": [17],
        "single_L23": [23],
        "global_only": gl,
        "late_half": list(range(13, 26)),
        "all_layers": list(range(n_layers)),
    }

    results = {}
    for sname, slayers in subsets.items():
        pe_vals, rescue_vals, gen_items = [], [], []
        for word in evals:
            query, target = str(word["ood"]), str(word["hindi"])
            tids = tokenizer.encode(target, add_special_tokens=False)
            if not tids: continue
            gold = int(tids[0])

            icl_r = build_task_prompt(query, icl_ex, input_script_name=src_s, source_language=src_l, output_script_name=tgt_s)
            zs_r = build_task_prompt(query, None, input_script_name=src_s, source_language=src_l, output_script_name=tgt_s)
            ids_i = tokenizer(apply_chat_template(tokenizer, icl_r), return_tensors="pt").to(device).input_ids
            ids_z = tokenizer(apply_chat_template(tokenizer, zs_r), return_tensors="pt").to(device).input_ids
            pi, pz = int(ids_i.shape[1]-1), int(ids_z.shape[1]-1)

            # Get ICL layer outputs at each target layer
            icl_outs = {}
            for li in slayers:
                icl_outs[li] = _extract_layer_output_at_position_from_input_ids(model, ids_i, li, pi).detach()

            # ZS baseline
            with torch.inference_mode():
                zo = model(input_ids=ids_z, use_cache=False)
            zp = float(torch.softmax(zo.logits[0, pz], dim=-1)[gold].item())
            # ICL baseline
            with torch.inference_mode():
                io = model(input_ids=ids_i, use_cache=False)
            ip = float(torch.softmax(io.logits[0, pi], dim=-1)[gold].item())

            # Patch
            hooks = [register_layer_output_replace_hook(model, li, icl_outs[li], patch_position=pz) for li in slayers]

            with torch.inference_mode():
                po = model(input_ids=ids_z, use_cache=False)
            pp = float(torch.softmax(po.logits[0, pz], dim=-1)[gold].item())

            # Generate with patches active
            pad = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", 0)
            try:
                with torch.inference_mode():
                    go = model.generate(ids_z, attention_mask=torch.ones_like(ids_z),
                                        max_new_tokens=20, do_sample=False, pad_token_id=int(pad))
                gen = norm(tokenizer.decode(go[0, ids_z.shape[1]:], skip_special_tokens=True))
            except Exception:
                gen = "<gen_error>"

            for h in hooks: h.remove()

            # Also generate with ICL (baseline)
            with torch.inference_mode():
                go_icl = model.generate(ids_i, attention_mask=torch.ones_like(ids_i),
                                        max_new_tokens=20, do_sample=False, pad_token_id=int(pad))
            gen_icl = norm(tokenizer.decode(go_icl[0, ids_i.shape[1]:], skip_special_tokens=True))

            # Also generate ZS (baseline)
            with torch.inference_mode():
                go_zs = model.generate(ids_z, attention_mask=torch.ones_like(ids_z),
                                       max_new_tokens=20, do_sample=False, pad_token_id=int(pad))
            gen_zs = norm(tokenizer.decode(go_zs[0, ids_z.shape[1]:], skip_special_tokens=True))

            pe = pp - zp
            rf = pe / max(ip - zp, 1e-8) if (ip - zp) > 1e-8 else 0
            tgt_norm = norm(target)
            em_patched = float(gen == tgt_norm)
            em_icl = float(gen_icl == tgt_norm)
            em_zs = float(gen_zs == tgt_norm)

            pe_vals.append(pe)
            rescue_vals.append(rf)
            gen_items.append({
                "ood": query, "gold": target, "gen_patched": gen, "gen_icl": gen_icl, "gen_zs": gen_zs,
                "em_patched": em_patched, "em_icl": em_icl, "em_zs": em_zs,
                "zs_prob": zp, "icl_prob": ip, "patched_prob": pp, "pe": pe, "rescue_frac": rf,
            })

        r = {
            "layers": slayers, "n_layers": len(slayers), "n_items": len(gen_items),
            "pe": bootstrap_ci(pe_vals), "rescue_frac": bootstrap_ci(rescue_vals),
            "em_patched": bootstrap_ci([g["em_patched"] for g in gen_items]),
            "em_icl": bootstrap_ci([g["em_icl"] for g in gen_items]),
            "em_zs": bootstrap_ci([g["em_zs"] for g in gen_items]),
            "items": gen_items,
        }
        results[sname] = r
        log("  %s (%d layers): pe=%.4f±%.4f EM_patched=%.1f%% EM_icl=%.1f%% EM_zs=%.1f%%" % (
            sname, len(slayers), r["pe"]["mean"], r["pe"]["std"],
            r["em_patched"]["mean"]*100, r["em_icl"]["mean"]*100, r["em_zs"]["mean"]*100))

    save("multilayer_patching_%s.json" % lang.lower(), {
        "lang": lang, "global_layers": gl, "results": results,
    })
    return results


# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 4: MLP CONTRIBUTION (Hindi N=30)
# ══════════════════════════════════════════════════════════════════════
def exp4_mlp_contribution(model, tokenizer, device, N=30):
    log("EXP4 MLP-CONTRIB Hindi N=%d" % N)
    pb = load_pair("aksharantar_hin_latin")
    icl_ex, evals = pb["icl_examples"], list(pb["eval_rows"][:N])
    cfg = getattr(model.config, "text_config", model.config)
    n_layers = cfg.num_hidden_layers
    ltypes = [str(x) for x in getattr(cfg, "layer_types", [])]

    layer_results = []
    for layer in range(n_layers):
        pe_vals = []
        for word in evals:
            query, target = str(word["ood"]), str(word["hindi"])
            tids = tokenizer.encode(target, add_special_tokens=False)
            if not tids: continue
            gold = int(tids[0])

            icl_r = build_task_prompt(query, icl_ex, input_script_name="Latin", source_language="Hindi", output_script_name="Devanagari")
            zs_r = build_task_prompt(query, None, input_script_name="Latin", source_language="Hindi", output_script_name="Devanagari")
            ids_i = tokenizer(apply_chat_template(tokenizer, icl_r), return_tensors="pt").to(device).input_ids
            ids_z = tokenizer(apply_chat_template(tokenizer, zs_r), return_tensors="pt").to(device).input_ids
            pi, pz = int(ids_i.shape[1]-1), int(ids_z.shape[1]-1)

            _, mlp_out = _extract_mlp_io_at_position_from_input_ids(model, ids_i, layer, pi)
            with torch.inference_mode():
                zo = model(input_ids=ids_z, use_cache=False)
            zp = float(torch.softmax(zo.logits[0, pz], dim=-1)[gold].item())
            hook = register_dense_mlp_output_patch_hook(model, layer, mlp_out.detach(), patch_position=pz)
            with torch.inference_mode():
                po = model(input_ids=ids_z, use_cache=False)
            pp = float(torch.softmax(po.logits[0, pz], dim=-1)[gold].item())
            hook.remove()
            pe_vals.append(pp - zp)

        lt = ltypes[layer] if layer < len(ltypes) else "?"
        layer_results.append({"layer": layer, "type": lt, "pe": bootstrap_ci(pe_vals)})
        if layer % 5 == 0: log("  L%02d pe=%.6f" % (layer, float(np.mean(pe_vals)) if pe_vals else 0))

    save("mlp_contribution_hindi.json", {"experiment": "mlp_contribution", "layers": layer_results})
    return layer_results


# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 5: DENSITY-DEPENDENT ATTENTION DILUTION (Hindi)
# ══════════════════════════════════════════════════════════════════════
def exp5_density_attention(model, tokenizer, device):
    log("EXP5 DENSITY-ATTENTION-DILUTION Hindi")
    pb = load_pair("aksharantar_hin_latin", n_icl=64)
    all_examples = pb["icl_examples"][:64]
    evals = list(pb["eval_rows"][:10])  # 10 items per density
    cfg = getattr(model.config, "text_config", model.config)
    ltypes = [str(x) for x in getattr(cfg, "layer_types", [])]
    gl = [i for i, lt in enumerate(ltypes) if "full" in lt]
    layers = get_model_layers(model)
    n_layers = len(layers)
    n_heads = cfg.num_attention_heads

    # Force eager attention
    if hasattr(model, "set_attn_implementation"):
        try: model.set_attn_implementation("eager")
        except: pass

    densities = [8, 16, 32, 48, 64]
    density_results = []

    for n_ex in densities:
        examples = all_examples[:n_ex]
        attn_data = []
        for word in evals[:5]:  # 5 items per density for attention analysis
            query = str(word["ood"])
            raw = build_task_prompt(query, examples, input_script_name="Latin",
                                    source_language="Hindi", output_script_name="Devanagari")
            rendered = apply_chat_template(tokenizer, raw)
            ids = tokenizer(rendered, return_tensors="pt").to(device).input_ids
            seq_len = int(ids.shape[1])
            lp = seq_len - 1

            # Identify token regions: find where ICL examples vs query/instruction are
            # ICL examples are in the first ~(n_ex * 13) tokens
            # Approximate: each ICL pair is ~13 tokens
            icl_region_end = min(n_ex * 13, seq_len - 50)  # leave room for query

            try:
                with torch.inference_mode():
                    out = model(input_ids=ids, use_cache=False, output_attentions=True)
                attns = out.attentions
                if attns is None: continue

                item = {"n_examples": n_ex, "seq_len": seq_len, "icl_region_end": icl_region_end, "layers": {}}
                for li in gl:  # only global layers
                    a = attns[li]  # [1, heads, seq, seq]
                    head_data = []
                    for hi in range(min(a.shape[1], n_heads)):
                        dist = a[0, hi, lp, :].detach().float().cpu().numpy()
                        icl_mass = float(dist[:icl_region_end].sum())
                        last512_mass = float(dist[max(0,seq_len-512):].sum())
                        total = float(dist.sum())
                        head_data.append({
                            "head": hi, "icl_mass": icl_mass,
                            "last512_mass": last512_mass, "total": total,
                        })
                    item["layers"]["L%02d" % li] = head_data
                attn_data.append(item)
            except Exception as e:
                log("  Attention extraction failed at n_ex=%d: %s" % (n_ex, str(e)[:100]))

        # Also measure behavioral performance at this density
        pe_vals = []
        for word in evals:
            query, target = str(word["ood"]), str(word["hindi"])
            tids = tokenizer.encode(target, add_special_tokens=False)
            if not tids: continue
            gold = int(tids[0])
            raw = build_task_prompt(query, examples, input_script_name="Latin",
                                    source_language="Hindi", output_script_name="Devanagari")
            ids = tokenizer(apply_chat_template(tokenizer, raw), return_tensors="pt").to(device).input_ids
            with torch.inference_mode():
                out = model(input_ids=ids, use_cache=False)
            prob = float(torch.softmax(out.logits[0, -1], dim=-1)[gold].item())
            pe_vals.append(prob)

        density_results.append({
            "n_examples": n_ex, "mean_prob": float(np.mean(pe_vals)),
            "attention_data": attn_data,
        })
        log("  n_ex=%d seq_len≈%d mean_target_prob=%.4f" % (
            n_ex, attn_data[0]["seq_len"] if attn_data else 0, float(np.mean(pe_vals))))

    save("density_attention_dilution.json", {
        "experiment": "density_attention_dilution", "global_layers": gl, "results": density_results,
    })
    return density_results


# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 6: TELUGU L25 ANOMALY INVESTIGATION (N=50)
# ══════════════════════════════════════════════════════════════════════
def exp6_telugu_anomaly(model, tokenizer, device, N=50):
    log("EXP6 TELUGU-L25-ANOMALY N=%d" % N)
    pb = load_pair("aksharantar_tel_latin")
    icl_ex, evals = pb["icl_examples"], list(pb["eval_rows"][:N])
    lm_head = model.lm_head if hasattr(model, "lm_head") else model.get_output_embeddings()
    lm_dtype = next(lm_head.parameters()).dtype
    cfg = getattr(model.config, "text_config", model.config)
    layers = get_model_layers(model)
    n_layers = len(layers)

    for chain in [("model","norm"),("language_model","model","norm")]:
        cur = model
        ok = True
        for nm in chain:
            if not hasattr(cur, nm): ok=False; break
            cur = getattr(cur, nm)
        if ok: final_norm = cur; break

    items = []
    for idx, word in enumerate(evals):
        query, target = str(word["ood"]), str(word["hindi"])
        tids = tokenizer.encode(target, add_special_tokens=False)
        if not tids: continue
        gold = int(tids[0])

        for cname, exs in [("helpful", icl_ex), ("corrupt", None)]:
            if cname == "helpful":
                raw = build_task_prompt(query, exs, input_script_name="Latin", source_language="Telugu", output_script_name="Telugu")
            else:
                raw = build_corrupted_icl_prompt(query, icl_ex, input_script_name="Latin", source_language="Telugu", output_script_name="Telugu", seed=42)
            ids = tokenizer(apply_chat_template(tokenizer, raw), return_tensors="pt").to(device).input_ids
            lp = int(ids.shape[1]-1)

            # Get final layer hidden state
            hidden_final = {}
            def mkhk(li):
                def hook(m, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    hidden_final[li] = h[0, lp, :].detach()
                return hook
            hs = [layers[n_layers-1].register_forward_hook(mkhk(n_layers-1))]
            with torch.inference_mode():
                out = model(input_ids=ids, use_cache=False)
            for h in hs: h.remove()

            # Get full logits at L25
            h = hidden_final[n_layers-1].to(dtype=lm_dtype)
            normed = final_norm(h.unsqueeze(0).unsqueeze(0)).squeeze()
            logits = lm_head(normed.unsqueeze(0)).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            rank = int((torch.argsort(logits, descending=True) == gold).nonzero(as_tuple=True)[0].item()) + 1
            prob = float(probs[gold].item())

            # Get TOP-5 predicted tokens
            top5_idx = torch.topk(probs, 5).indices.tolist()
            top5_probs = torch.topk(probs, 5).values.tolist()
            top5_tokens = [tokenizer.decode([t]) for t in top5_idx]

            items.append({
                "idx": idx, "ood": query, "target": target, "condition": cname,
                "gold_rank": rank, "gold_prob": prob,
                "top5": [{"token": t, "prob": p, "id": i}
                         for t, p, i in zip(top5_tokens, top5_probs, top5_idx)],
            })

        if (idx+1) % 10 == 0: log("  Telugu %d/%d" % (idx+1, len(evals)))

    # Analyze: compare helpful vs corrupt top-5
    save("telugu_anomaly.json", {"experiment": "telugu_l25_anomaly", "n_items": len(items) // 2, "items": items})
    return items


# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 7: KEY HEAD ATTENTION PATTERNS (Hindi, detailed)
# ══════════════════════════════════════════════════════════════════════
def exp7_head_attention_patterns(model, tokenizer, device, N=10):
    log("EXP7 HEAD-ATTENTION-PATTERNS Hindi N=%d" % N)
    pb = load_pair("aksharantar_hin_latin")
    icl_ex, evals = pb["icl_examples"], list(pb["eval_rows"][:N])
    cfg = getattr(model.config, "text_config", model.config)
    ltypes = [str(x) for x in getattr(cfg, "layer_types", [])]
    layers = get_model_layers(model)
    n_layers = len(layers)
    n_heads = cfg.num_attention_heads
    key_heads = [(5,0),(5,1),(11,0),(11,1),(14,0),(17,0),(17,1),(23,0),(23,1)]

    if hasattr(model, "set_attn_implementation"):
        try: model.set_attn_implementation("eager")
        except: pass

    items = []
    for idx, word in enumerate(evals):
        query = str(word["ood"])
        for cname, exs in [("helpful_icl", icl_ex), ("zs", None)]:
            raw = build_task_prompt(query, exs, input_script_name="Latin",
                                    source_language="Hindi", output_script_name="Devanagari")
            rendered = apply_chat_template(tokenizer, raw)
            ids = tokenizer(rendered, return_tensors="pt").to(device).input_ids
            tokens = tokenizer.convert_ids_to_tokens(ids[0].tolist())
            seq_len = int(ids.shape[1])
            lp = seq_len - 1

            try:
                with torch.inference_mode():
                    out = model(input_ids=ids, use_cache=False, output_attentions=True)
                attns = out.attentions

                head_patterns = {}
                for (li, hi) in key_heads:
                    if attns[li].shape[1] <= hi: continue
                    dist = attns[li][0, hi, lp, :].detach().float().cpu().numpy()
                    # Top-10 attended positions
                    top10_pos = np.argsort(dist)[::-1][:10].tolist()
                    top10_mass = [float(dist[p]) for p in top10_pos]
                    top10_tok = [tokens[p] if p < len(tokens) else "?" for p in top10_pos]
                    head_patterns["L%02dH%d" % (li, hi)] = {
                        "top10_pos": top10_pos, "top10_mass": top10_mass, "top10_tokens": top10_tok,
                        "total_mass_first_half": float(dist[:seq_len//2].sum()),
                        "total_mass_second_half": float(dist[seq_len//2:].sum()),
                    }

                items.append({
                    "idx": idx, "ood": query, "condition": cname,
                    "seq_len": seq_len, "head_patterns": head_patterns,
                })
            except Exception as e:
                log("  Attn extraction failed: %s" % str(e)[:100])

    save("head_attention_patterns.json", {
        "experiment": "head_attention_patterns", "key_heads": key_heads, "items": items,
    })
    return items


# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 8: ICL BASELINE GENERATION (all 5 langs, N=50)
# ══════════════════════════════════════════════════════════════════════
def exp8_icl_baseline_generation(model, tokenizer, device, N=50):
    log("EXP8 ICL-BASELINE-GENERATION all languages N=%d" % N)
    results = {}
    for pair_id, lang, src_s, src_l, tgt_s in PAIRS:
        pb = load_pair(pair_id)
        icl_ex, evals = pb["icl_examples"], list(pb["eval_rows"][:N])

        gen_items = []
        for word in evals:
            query, target = str(word["ood"]), str(word["hindi"])

            for cname, exs in [("icl", icl_ex), ("zs", None)]:
                raw = build_task_prompt(query, exs, input_script_name=src_s,
                                        source_language=src_l, output_script_name=tgt_s)
                ids = tokenizer(apply_chat_template(tokenizer, raw), return_tensors="pt").to(device).input_ids
                pad = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", 0)
                with torch.inference_mode():
                    go = model.generate(ids, attention_mask=torch.ones_like(ids),
                                        max_new_tokens=30, do_sample=False, pad_token_id=int(pad))
                gen = norm(tokenizer.decode(go[0, ids.shape[1]:], skip_special_tokens=True))
                tgt_norm = norm(target)
                em = float(gen == tgt_norm)
                gen_items.append({"ood": query, "target": target, "condition": cname,
                                  "generated": gen, "em": em})

        icl_em = [g["em"] for g in gen_items if g["condition"] == "icl"]
        zs_em = [g["em"] for g in gen_items if g["condition"] == "zs"]
        results[lang] = {
            "icl_em": bootstrap_ci(icl_em), "zs_em": bootstrap_ci(zs_em),
            "items": gen_items,
        }
        log("  %s: ICL_EM=%.1f%% ZS_EM=%.1f%%" % (lang, np.mean(icl_em)*100, np.mean(zs_em)*100))

    save("icl_baseline_generation.json", {"experiment": "icl_generation_baseline", "results": results})
    return results


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    set_all_seeds(42)
    model, tokenizer = load_model("1b", device="cuda")
    device = str(next(model.parameters()).device)

    log("=" * 70)
    log("1B DEFINITIVE MECHANISTIC BATTERY — START")
    log("=" * 70)

    # EXP 8 first — lightweight, gives us behavioral baselines immediately
    try: exp8_icl_baseline_generation(model, tokenizer, device, N=50)
    except Exception as e: log("EXP8 FAILED: %s" % traceback.format_exc()[-300:])

    # EXP 1: Logit lens for all languages
    for pair_id, lang, src_s, src_l, tgt_s in PAIRS:
        try: exp1_logit_lens(model, tokenizer, device, pair_id, lang, src_s, src_l, tgt_s, N=50)
        except Exception as e: log("EXP1 %s FAILED: %s" % (lang, traceback.format_exc()[-300:]))

    # EXP 3: Multi-layer patching for Hindi first (the key finding)
    try: exp3_multilayer_patching(model, tokenizer, device, *PAIRS[0], N=50)
    except Exception as e: log("EXP3 Hindi FAILED: %s" % traceback.format_exc()[-300:])

    # EXP 3b: Multi-layer patching for other languages (global-only subset)
    for pair_id, lang, src_s, src_l, tgt_s in PAIRS[1:]:
        try: exp3_multilayer_patching(model, tokenizer, device, pair_id, lang, src_s, src_l, tgt_s, N=30)
        except Exception as e: log("EXP3 %s FAILED: %s" % (lang, traceback.format_exc()[-300:]))

    # EXP 2: Head attribution for all languages
    for pair_id, lang, src_s, src_l, tgt_s in PAIRS:
        try: exp2_head_attribution(model, tokenizer, device, pair_id, lang, src_s, src_l, tgt_s, N=20)
        except Exception as e: log("EXP2 %s FAILED: %s" % (lang, traceback.format_exc()[-300:]))

    # EXP 4: MLP contribution
    try: exp4_mlp_contribution(model, tokenizer, device, N=30)
    except Exception as e: log("EXP4 FAILED: %s" % traceback.format_exc()[-300:])

    # EXP 5: Density attention dilution
    try: exp5_density_attention(model, tokenizer, device)
    except Exception as e: log("EXP5 FAILED: %s" % traceback.format_exc()[-300:])

    # EXP 6: Telugu anomaly
    try: exp6_telugu_anomaly(model, tokenizer, device, N=50)
    except Exception as e: log("EXP6 FAILED: %s" % traceback.format_exc()[-300:])

    # EXP 7: Head attention patterns
    try: exp7_head_attention_patterns(model, tokenizer, device, N=10)
    except Exception as e: log("EXP7 FAILED: %s" % traceback.format_exc()[-300:])

    elapsed = time.time() - t0
    log("=" * 70)
    log("ALL EXPERIMENTS COMPLETE in %.1f minutes" % (elapsed / 60))
    log("=" * 70)

    # List all output files
    for f in sorted(OUT.glob("*.json")):
        log("  OUTPUT: %s (%.1f KB)" % (f.name, f.stat().st_size / 1024))


if __name__ == "__main__":
    raise SystemExit(main())
