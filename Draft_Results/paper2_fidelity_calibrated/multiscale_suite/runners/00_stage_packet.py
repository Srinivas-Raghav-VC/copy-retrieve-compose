from __future__ import annotations

from pathlib import Path

from .common import RESULTS_ROOT, parse_common_args, stage_packet, write_json


def main() -> int:
    ap = parse_common_args("Emit the frozen multiscale stage packet.")
    ap.add_argument("--lanes", type=str, default="")
    args = ap.parse_args()
    lanes = [x.strip() for x in str(args.lanes).split(",") if x.strip()]
    payload = stage_packet(lanes or None)
    out = (
        Path(str(args.emit)).resolve()
        if str(args.emit).strip()
        else RESULTS_ROOT / "packets" / "stage_packet.json"
    )
    write_json(out, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
