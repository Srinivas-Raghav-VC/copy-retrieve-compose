from __future__ import annotations

from pathlib import Path

from .common import RESULTS_ROOT, parse_common_args, write_json


def main() -> int:
    ap = parse_common_args("Emit the proxy-bridge packet for judge-vs-proxy analysis.")
    args = ap.parse_args()
    payload = {
        "lane": "proxy_bridge",
        "status": "packetized",
        "supported_models": ["1b", "4b"],
        "unsupported_models": ["270m"],
        "required_existing_scripts": [
            "run_first_token_proxy_validation.py",
            "analyze_first_token_proxy_validation.py",
            "generate_behavioral_problem_statement_report.py",
        ],
        "review_note": "Do not compare mechanistic proxies across scales until this bridge exists on the same items.",
    }
    out = (
        Path(str(args.emit)).resolve()
        if str(args.emit).strip()
        else RESULTS_ROOT / "packets" / "proxy_bridge_packet.json"
    )
    write_json(out, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
