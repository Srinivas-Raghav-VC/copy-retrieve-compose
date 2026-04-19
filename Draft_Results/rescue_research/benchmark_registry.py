from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List


BENCHMARK_SOURCES: List[Dict] = [
    {
        "id": "dakshina",
        "task": "cross-script transliteration",
        "url": "https://github.com/google-research-datasets/dakshina",
        "citation": "Roark et al., 2020, Processing South Asian Languages Written in the Latin Script: the Dakshina Dataset",
        "arxiv": "https://arxiv.org/abs/2007.01176",
        "license_notes": "See dataset repository for license and redistribution terms.",
    },
    {
        "id": "aksharantar",
        "task": "indic transliteration",
        "url": "https://github.com/AI4Bharat/IndicXlit",
        "citation": "Madhani et al., 2022, Aksharantar",
        "arxiv": "https://arxiv.org/abs/2205.03018",
        "license_notes": "See dataset repository for license and redistribution terms.",
    },
    {
        "id": "indonlp_2025_transliteration",
        "task": "real-time reverse transliteration",
        "url": "https://arxiv.org/abs/2501.05816",
        "citation": "IndoNLP 2025 shared task resources",
        "arxiv": "https://arxiv.org/abs/2501.05816",
        "license_notes": "Task-specific; verify release constraints before redistribution.",
    },
]


EVAL_METRICS: List[Dict] = [
    {
        "name": "top1_accuracy",
        "type": "task_metric",
        "justification": "Standard transliteration exact-match metric used in Dakshina/Aksharantar reporting.",
    },
    {
        "name": "cer",
        "type": "task_metric",
        "justification": "Character-level robustness metric for noisy transliteration outputs.",
    },
    {
        "name": "wer",
        "type": "task_metric",
        "justification": "Word-level error decomposition for phrase/sentence transliteration.",
    },
    {
        "name": "mean_pe",
        "type": "mechanistic_metric",
        "justification": "Primary mechanistic sufficiency signal (patched - zero-shot).",
    },
    {
        "name": "mean_ae",
        "type": "mechanistic_metric",
        "justification": "Necessity/ablation signal in intervention analysis.",
    },
]


def write_benchmark_registry(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmark_sources": BENCHMARK_SOURCES,
        "evaluation_metrics": EVAL_METRICS,
        "notes": [
            "Registry is a protocol manifest for reproducibility and paper-writing traceability.",
            "Update citations/URLs in one place before submission freeze.",
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
