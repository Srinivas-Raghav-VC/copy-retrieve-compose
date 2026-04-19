#!/usr/bin/env python3
"""
Emit a lightweight Stage 0 packet for the frozen CFOM workshop protocol.

This packet freezes prompt/template fingerprints, runtime identity, and the
intended hook-site metadata before any expensive mainline run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_model_config
from core import load_model, split_data_three_way
from paper2_fidelity_calibrated.protocol_utils import runtime_identity, site_alignment_verdict
from paper2_fidelity_calibrated.run import _load_words, _make_protocol_prompt_packet
from rescue_research.data_pipeline.ingest import get_pair_prompt_metadata


def main() -> int:
    ap = argparse.ArgumentParser(description="Write Stage 0 protocol freeze packet.")
    ap.add_argument("--model", type=str, required=True, choices=["270m", "1b", "4b", "12b"])
    ap.add_argument("--pair", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-icl", type=int, default=64)
    ap.add_argument("--n-select", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--prompt-variant", type=str, default="canonical")
    ap.add_argument("--external-only", action="store_true")
    ap.add_argument("--require-external-sources", action="store_true")
    ap.add_argument("--min-pool-size", type=int, default=0)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    words, provenance = _load_words(
        str(args.pair),
        external_only=bool(args.external_only),
        require_external_sources=bool(args.require_external_sources),
        min_pool_size=int(args.min_pool_size),
    )
    icl, sel, ev = split_data_three_way(
        words=words,
        n_icl=int(args.n_icl),
        n_select=int(args.n_select),
        n_eval=int(args.n_eval),
        seed=int(args.seed),
    )
    sample_word = sel[0] if sel else (ev[0] if ev else icl[0])
    prompt_meta = get_pair_prompt_metadata(str(args.pair))
    cfg = get_model_config(str(args.model))
    model, tokenizer = load_model(str(args.model), device=str(args.device))

    packet = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pair": str(args.pair),
        "model_key": str(args.model),
        "runtime_identity": runtime_identity(
            model_key=str(args.model),
            hf_id=cfg.hf_id,
            tokenizer=tokenizer,
            model=model,
        ),
        "prompt_packet": _make_protocol_prompt_packet(
            tokenizer,
            sample_word=sample_word,
            icl_examples=icl,
            prompt_meta=prompt_meta,
            prompt_variant=str(args.prompt_variant),
        ),
        "site_sanity_packet": {
            "hook_registration": "core.register_transcoder_feature_patch_hook -> layers[layer].mlp.register_forward_hook",
            "hook_tensor": "forward output of layers[layer].mlp",
            "pre_post_norm_status": "UNVERIFIED_STAGE0_REQUIRED",
            "pre_post_projection_status": "UNVERIFIED_STAGE0_REQUIRED",
            "artifact_site_target": "Gemma Scope 2 compatible MLP-output site",
            "alignment_verdict": site_alignment_verdict(
                exact_match=False,
                family_match=True,
            ),
            "note": "Update verdict after tensor-level Stage 0 inspection; do not claim exact artifact alignment from this packet alone.",
        },
        "protocol": {
            "claim_level": "intervention_only",
            "prompt_template_frozen_after_stage0": True,
            "no_silent_span_fallback": True,
            "fidelity_proxy_not_sufficient": True,
        },
        "split_sizes": {"icl": len(icl), "selection": len(sel), "eval_blind": len(ev)},
        "provenance": provenance,
    }

    out_path = (
        Path(args.out).resolve()
        if str(args.out).strip()
        else Path(__file__).resolve().parent / "results" / str(args.pair) / str(args.model) / "stage0_packet.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
