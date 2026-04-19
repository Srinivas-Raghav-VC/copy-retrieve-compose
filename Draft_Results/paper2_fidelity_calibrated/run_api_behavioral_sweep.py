#!/usr/bin/env python3
"""
Behavioral sweep: ZS vs ICL across languages and models via Google AI Studio API.
Measures text generation only (exact match, first-entry). No logprobs for Gemma.
"""
import argparse, json, os, sys, time, urllib.request
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LANGUAGES = {
    "hindi": {
        "src_script": "Latin", "tgt_script": "Devanagari", "language": "Hindi",
        "icl": [("dilli","दिल्ली"),("mumbai","मुम्बई"),("bharat","भारत"),("paani","पानी"),("kitaab","किताब"),("sundar","सुंदर"),("samay","समय"),("duniya","दुनिया")],
        "test": [("namaste","नमस्ते"),("ghar","घर"),("bachcha","बच्चा"),("sapna","सपना"),("aakash","आकाश"),("rasta","रास्ता"),("khushi","खुशी"),("kahani","कहानी"),("pyaar","प्यार"),("barish","बारिश"),("pariwaar","परिवार"),("zindagi","ज़िंदगी")],
    },
    "telugu": {
        "src_script": "Latin", "tgt_script": "Telugu", "language": "Telugu",
        "icl": [("namaskaram","నమస్కారం"),("vijayawada","విజయవాడ"),("telangana","తెలంగాణ"),("pustakam","పుస్తకం"),("vidyarthi","విద్యార్థి"),("manishi","మనిషి"),("prabhavam","ప్రభావం"),("paristhiti","పరిస్థితి")],
        "test": [("hyderabad","హైదరాబాద్"),("andhra","ఆంధ్ర"),("krishna","కృష్ణ"),("guntur","గుంటూరు"),("nellore","నెల్లూరు"),("tirupati","తిరుపతి"),("kakinada","కాకినాడ"),("warangal","వరంగల్")],
    },
    "bengali": {
        "src_script": "Latin", "tgt_script": "Bangla", "language": "Bengali",
        "icl": [("kolkata","কলকাতা"),("rabindranath","রবীন্দ্রনাথ"),("sundorbon","সুন্দরবন"),("mishti","মিষ্টি"),("boi","বই"),("shiksha","শিক্ষা"),("swadhinata","স্বাধীনতা"),("jibon","জীবন")],
        "test": [("dhaka","ঢাকা"),("chittagong","চট্টগ্রাম"),("shanti","শান্তি"),("poribar","পরিবার"),("brishti","বৃষ্টি"),("asha","আশা"),("somoy","সময়"),("manush","মানুষ")],
    },
    "tamil": {
        "src_script": "Latin", "tgt_script": "Tamil", "language": "Tamil",
        "icl": [("chennai","சென்னை"),("tamilnadu","தமிழ்நாடு"),("vanakkam","வணக்கம்"),("madurai","மதுரை"),("kovil","கோவில்"),("neer","நீர்"),("kalam","காலம்"),("ulakam","உலகம்")],
        "test": [("coimbatore","கோயம்புத்தூர்"),("nandri","நன்றி"),("palli","பள்ளி"),("maram","மரம்"),("vaanam","வானம்"),("kadal","கடல்"),("malai","மலை"),("amma","அம்மா")],
    },
    "marathi": {
        "src_script": "Latin", "tgt_script": "Devanagari", "language": "Marathi",
        "icl": [("mumbai","मुंबई"),("pune","पुणे"),("maharashtra","महाराष्ट्र"),("sahyadri","सह्याद्री"),("namaskar","नमस्कार"),("paani","पाणी"),("shala","शाळा"),("ghar","घर")],
        "test": [("nagpur","नागपूर"),("nashik","नाशिक"),("kolhapur","कोल्हापूर"),("aai","आई"),("baba","बाबा"),("diwali","दिवाळी"),("ganpati","गणपती"),("wada","वडा")],
    },
}

def api_call(model, prompt, key, max_tok=32):
    url = "https://generativelanguage.googleapis.com/v1beta/%s:generateContent?key=%s" % (model, key)
    body = json.dumps({"contents":[{"parts":[{"text":prompt}]}],"generationConfig":{"maxOutputTokens":max_tok,"temperature":0}}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type":"application/json"})
    try:
        resp = urllib.request.urlopen(req, timeout=60)
        d = json.loads(resp.read())
        return d["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        return "ERROR:" + str(e)[:100]

def build_prompt(cfg, query, n_icl):
    lines = ["Task: Convert %s text written in %s script into %s in %s script." % (cfg["language"],cfg["src_script"],cfg["language"],cfg["tgt_script"]), ""]
    for src, tgt in cfg["icl"][:n_icl]:
        lines += ["Input (%s): %s" % (cfg["src_script"],src), "Output (%s): %s" % (cfg["tgt_script"],tgt), ""]
    lines += ["Input (%s): %s" % (cfg["src_script"],query), "Output (%s):" % cfg["tgt_script"]]
    return "\n".join(lines)

def norm(s):
    return s.strip().split("\n")[0].strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", default=os.environ.get("GOOGLE_API_KEY",""))
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    key = args.api_key
    if not key:
        print("Set GOOGLE_API_KEY"); return 1

    models = ["models/gemma-3-1b-it","models/gemma-3-4b-it"]
    icl_counts = [0, 2, 4, 8]
    rows = []

    for lang_name, cfg in LANGUAGES.items():
        for model in models:
            mname = model.split("/")[-1]
            for n_icl in icl_counts:
                ems, fes = [], []
                for src, gold in cfg["test"]:
                    prompt = build_prompt(cfg, src, n_icl)
                    pred = norm(api_call(model, prompt, key))
                    gold_n = norm(gold)
                    em = float(pred == gold_n)
                    fe = float(gold_n != "" and pred.startswith(gold_n[0])) if len(gold_n) > 0 else 0.0
                    ems.append(em); fes.append(fe)
                    rows.append({"lang":lang_name,"model":mname,"n_icl":n_icl,"src":src,"gold":gold_n,"pred":pred,"em":em,"fe":fe})
                    time.sleep(0.3)
                avg_em = sum(ems)/len(ems) if ems else 0
                avg_fe = sum(fes)/len(fes) if fes else 0
                print("%s %-10s icl=%d: EM=%.1f%% FE=%.1f%% (n=%d)" % (mname, lang_name, n_icl, avg_em*100, avg_fe*100, len(ems)))

    out_path = Path(args.out) if args.out else PROJECT_ROOT/"paper2_fidelity_calibrated"/"results"/"api_behavioral_sweep.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"],r["lang"],r["n_icl"])].append(r)
    summary = []
    for (m,l,n), rs in sorted(groups.items()):
        summary.append({"model":m,"language":l,"n_icl":n,"n":len(rs),
                        "exact_match":sum(r["em"] for r in rs)/len(rs),
                        "first_entry":sum(r["fe"] for r in rs)/len(rs)})
    payload = {"experiment":"api_behavioral_sweep","created":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
               "summary":summary,"items":rows}
    out_path.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding="utf-8")
    print("Saved:", out_path)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
