from __future__ import annotations

from pathlib import Path

from .common import RESULTS_ROOT, parse_common_args, write_json


def main() -> int:
    ap = parse_common_args("Emit the multiscale logit-lens experiment packet.")
    args = ap.parse_args()
    payload = {
        "lane": "logit_lens",
        "status": "packetized",
        "supported_models": ["1b", "4b"],
        "deferred_models": ["270m"],
        "recommended_controls": ["helpful", "corrupt", "zero_shot", "format_only"],
        "required_n": {"core": 100, "breadth": 50},
        "attention_probes_note": "Use probe heatmaps only as layer hypotheses, never as causal proof.",
    }
    out = (
        Path(str(args.emit)).resolve()
        if str(args.emit).strip()
        else RESULTS_ROOT / "packets" / "logit_lens_packet.json"
    )
    write_json(out, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
