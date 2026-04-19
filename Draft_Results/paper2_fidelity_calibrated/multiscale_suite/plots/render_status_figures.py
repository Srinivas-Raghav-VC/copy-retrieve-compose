from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from ...multiscale_modal_suite.plot_style import apply_research_style


PAPER_TABLE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "multiscale_suite"
    / "paper_tables"
)


def _load_summary() -> dict:
    path = PAPER_TABLE_ROOT / "execution_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing execution summary: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def render() -> tuple[Path, Path]:
    apply_research_style()
    summary = _load_summary()
    status_counts = summary.get("status_counts", {})
    lane_counts = summary.get("lane_counts", {})

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    statuses = list(status_counts.keys())
    counts = [status_counts[s] for s in statuses]
    axes[0].bar(
        statuses,
        counts,
        color=["#577590", "#43aa8b", "#f94144", "#f9c74f"][: len(statuses)],
    )
    axes[0].set_title("Execution Status Overview")
    axes[0].set_ylabel("Task count")
    axes[0].tick_params(axis="x", rotation=25)

    lanes = list(lane_counts.keys())
    ready = [lane_counts[l].get("success", 0) for l in lanes]
    blocked = [
        sum(v for k, v in lane_counts[l].items() if k != "success") for l in lanes
    ]
    axes[1].barh(lanes, ready, color="#2a9d8f", label="success")
    axes[1].barh(lanes, blocked, left=ready, color="#adb5bd", label="non-success")
    axes[1].set_title("Lane Readiness")
    axes[1].set_xlabel("Task count")
    axes[1].legend(frameon=False)

    fig.suptitle("Multiscale Suite Execution Snapshot", fontsize=14)
    fig.tight_layout()

    out_dir = PAPER_TABLE_ROOT.parent / "paper_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "fig_execution_snapshot.png"
    pdf = out_dir / "fig_execution_snapshot.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> int:
    png, pdf = render()
    print(json.dumps({"png": str(png), "pdf": str(pdf)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
