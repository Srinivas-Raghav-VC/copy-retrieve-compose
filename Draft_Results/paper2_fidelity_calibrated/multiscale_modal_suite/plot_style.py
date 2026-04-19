from __future__ import annotations

import matplotlib as mpl


LANG_COLORS = {
    "Hindi": "#c23b22",
    "Telugu": "#1f5aa6",
    "Bengali": "#2b8a3e",
    "Gujarati": "#ff7f11",
    "Kannada": "#6a4c93",
    "Arabic": "#2d728f",
    "Cyrillic": "#7a9e7e",
}

MODEL_COLORS = {
    "270m": "#8ecae6",
    "1b": "#219ebc",
    "4b": "#023047",
}

CONDITION_COLORS = {
    "helpful": "#20639b",
    "corrupt": "#b33f62",
    "zs": "#7d8597",
}


def apply_research_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.6,
            "figure.autolayout": False,
        }
    )
