from __future__ import annotations

from pathlib import Path

from .common import RESULTS_ROOT, parse_common_args, write_json


def main() -> int:
    ap = parse_common_args("Emit the attention-probe diagnostic packet.")
    args = ap.parse_args()
    payload = {
        "lane": "attention_probes",
        "status": "packetized",
        "models": ["270m", "1b", "4b"],
        "purpose": "Use probes as layer-selection heuristics before causal patching.",
        "probe_recommendation": {
            "heads": 8,
            "baselines": ["mean_pool", "last_token"],
            "report": ["accuracy", "head_entropy", "max_activating_examples"],
        },
        "interpretation_rule": "Probe decodability is hypothesis-generating evidence, not mechanistic proof.",
        "blog_anchor": "https://blog.eleuther.ai/attention-probes/",
    }
    out = (
        Path(str(args.emit)).resolve()
        if str(args.emit).strip()
        else RESULTS_ROOT / "packets" / "attention_probes_packet.json"
    )
    write_json(out, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
