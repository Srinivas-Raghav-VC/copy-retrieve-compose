from __future__ import annotations

from pathlib import Path

from .common import RESULTS_ROOT, parse_common_args, write_json


def main() -> int:
    ap = parse_common_args("Emit the multiscale head-attribution packet.")
    args = ap.parse_args()
    payload = {
        "lane": "head_attribution",
        "status": "packetized",
        "primary_models": ["1b"],
        "followup_models": ["4b", "270m"],
        "required_n": 100,
        "seed_family": [42, 123, 456],
        "oracle_constraint": "derive candidate heads from architecture-derived global/local roles, not 1b hardcoded assumptions",
    }
    out = (
        Path(str(args.emit)).resolve()
        if str(args.emit).strip()
        else RESULTS_ROOT / "packets" / "head_attribution_packet.json"
    )
    write_json(out, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
