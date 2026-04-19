import json
from pathlib import Path

import marimo as mo
import pandas as pd


app = mo.App(width="full")


@app.cell
def _():
    root = (
        Path(__file__).resolve().parent.parent
        / "results"
        / "multiscale_modal_suite"
        / "telemetry"
    )
    manifests = sorted(root.glob("*/manifest.json"))
    rows = []
    for path in manifests:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "task_id": payload.get("task_id", path.parent.name),
                "lane": payload.get("lane", ""),
                "status": payload.get("status", ""),
                "model": payload.get("model", ""),
                "pair": payload.get("pair", ""),
                "smoke": bool(payload.get("smoke", False)),
                "exit_code": payload.get("exit_code", None),
                "started_at_utc": payload.get("started_at_utc", ""),
                "finished_at_utc": payload.get("finished_at_utc", ""),
                "stdout_path": payload.get("telemetry", {}).get("stdout_path", ""),
                "stderr_path": payload.get("telemetry", {}).get("stderr_path", ""),
            }
        )
    df = pd.DataFrame(rows)
    return df, root


@app.cell
def _(df):
    status_counts = (
        df.groupby("status", dropna=False).size().reset_index(name="count")
        if not df.empty
        else pd.DataFrame(columns=["status", "count"])
    )
    return status_counts


@app.cell
def _(df, status_counts):
    mo.md(
        """
        # Multiscale Modal Suite Dashboard

        This marimo app surfaces task telemetry for the 270M/1B/4B suite.
        Use it as a launch checklist and as a quick audit of failures before synthesis.
        """
    )
    summary = mo.vstack(
        [
            mo.md(f"**Tasks discovered:** {len(df)}"),
            mo.ui.table(status_counts),
        ]
    )
    table = mo.ui.table(df, pagination=True, page_size=20)
    return summary, table


@app.cell
def _(summary, table):
    mo.vstack([summary, mo.md("## Task manifests"), table])
    return


if __name__ == "__main__":
    app.run()
