from __future__ import annotations

import subprocess
from pathlib import Path

from ...multiscale_modal_suite.verify_suite import run_verification
from .common import RESULTS_ROOT, parse_common_args, write_json


def main() -> int:
    ap = parse_common_args("Run the architecture + behavioral premise lane.")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    packet = {
        "lane": "premise_behavior",
        "status": "plan_only",
        "selected_lanes": ["architecture", "behavioral"],
        "results_root": str(RESULTS_ROOT),
        "smoke": bool(args.smoke),
    }
    out = (
        Path(str(args.emit)).resolve()
        if str(args.emit).strip()
        else RESULTS_ROOT / "packets" / "premise_behavior_packet.json"
    )
    write_json(out, packet)
    command = [
        "python3",
        "-m",
        "Draft_Results.paper2_fidelity_calibrated.multiscale_modal_suite.runner",
        "--lanes",
        "architecture,behavioral",
        "--emit-plan",
        str(RESULTS_ROOT / "packets" / "premise_behavior_plan.json"),
        "--execute",
    ]
    if args.smoke:
        command.append("--smoke")
    if args.force:
        command.append("--force")
    if not args.execute:
        return 0

    preflight = run_verification()
    if not preflight.get("preflight_ok", False):
        raise RuntimeError(
            "Refusing premise lane execution because required research modules are missing: "
            + ", ".join(preflight.get("missing_research_modules", []))
        )

    completed = subprocess.run(command, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
