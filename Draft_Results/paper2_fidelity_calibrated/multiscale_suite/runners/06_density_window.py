from __future__ import annotations

from pathlib import Path

from .common import RESULTS_ROOT, parse_common_args, write_json


def main() -> int:
    ap = parse_common_args("Emit the density/window stress packet.")
    args = ap.parse_args()
    payload = {
        "lane": "density_window",
        "status": "mixed",
        "required_counts": [4, 8, 16, 32, 48, 64],
        "normalization_rule": "window analyses must use each model's actual architecture, not transplanted 1b constants",
        "plot_targets": ["within_window_dilution", "boundary_cliff", "delta_from_zs"],
    }
    out = (
        Path(str(args.emit)).resolve()
        if str(args.emit).strip()
        else RESULTS_ROOT / "packets" / "density_window_packet.json"
    )
    write_json(out, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
