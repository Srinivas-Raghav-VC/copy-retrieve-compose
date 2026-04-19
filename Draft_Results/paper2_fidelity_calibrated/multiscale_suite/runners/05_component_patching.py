from __future__ import annotations

from pathlib import Path

from .common import RESULTS_ROOT, parse_common_args, write_json


def main() -> int:
    ap = parse_common_args("Emit the component-patching packet.")
    args = ap.parse_args()
    payload = {
        "lane": "component_patching",
        "status": "mixed",
        "executable_now": ["1b_final_bundle"],
        "packetized_followups": ["4b_joint_groups", "270m_floor_patch"],
        "required_subruns": ["attention_only", "mlp_only", "joint_attention_mlp"],
        "critical_risk": "If the isolated negative remains split-sensitive, freeze the claim as split-sensitive rather than universal zero.",
    }
    out = (
        Path(str(args.emit)).resolve()
        if str(args.emit).strip()
        else RESULTS_ROOT / "packets" / "component_patching_packet.json"
    )
    write_json(out, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
