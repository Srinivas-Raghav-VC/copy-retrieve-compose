"""
Data loading and splits for the rescue pipeline.

- Words are loaded via `rescue_research.data_pipeline.ingest` (canonical source).
- Splits use three-way (ICL / selection / eval) whenever selection is involved.
- Disjointness is enforced by a unique key (english); we validate uniqueness.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

from rescue_research.config import N_ICL, N_SELECT, N_EVAL, MIN_WORDS_FOR_THREE_WAY
from rescue_research.data_pipeline.ingest import load_pair_words_for_experiment


def _add_reference_path() -> None:
    """Ensure parent package is on path so we can import project modules."""
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def get_words(pair_id: str = "hindi_telugu") -> List[Dict[str, Any]]:
    """
    Load word list from canonical data ingestion pipeline.

    Returns compatibility keys used by legacy experiment code:
      - english: stable lemma key
      - hindi: source-script token (historical name kept for compatibility)
      - telugu/ood: target-script token (historical names kept for compatibility)
    """
    words = []
    for row in load_pair_words_for_experiment(pair_id):
        words.append(
            {
                "english": str(row.get("english", "")).strip(),
                "hindi": str(row.get("hindi", "")).strip(),
                "telugu": str(row.get("ood", "")).strip(),
                "ood": str(row.get("ood", "")).strip(),
            }
        )
    # Enforce unique key for disjointness
    seen: set[str] = set()
    for w in words:
        key = w.get("english", "")
        if key in seen:
            raise ValueError(f"Duplicate 'english' in word list: {key!r}")
        seen.add(key)
    return words


def set_seed(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)


def three_way_split(
    words: List[Dict[str, Any]],
    n_icl: int,
    n_select: int,
    n_eval: int,
    seed: int,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split into disjoint ICL / selection / evaluation sets.
    Use for any step that does selection (layer or feature) so eval is held out.
    """
    if n_icl + n_select + n_eval > len(words):
        raise ValueError(
            f"Not enough words: need {n_icl + n_select + n_eval}, have {len(words)}. "
            f"Use at least {MIN_WORDS_FOR_THREE_WAY} words or reduce n_icl/n_select/n_eval."
        )
    set_seed(seed)
    shuffled = list(words)
    random.shuffle(shuffled)
    icl = shuffled[:n_icl]
    select = shuffled[n_icl : n_icl + n_select]
    eval_ = shuffled[n_icl + n_select : n_icl + n_select + n_eval]
    # Validate disjointness
    key = "english"
    s_icl = {w[key] for w in icl}
    s_sel = {w[key] for w in select}
    s_eval = {w[key] for w in eval_}
    if s_icl & s_sel or s_icl & s_eval or s_sel & s_eval:
        raise RuntimeError("Three-way split overlap detected (bug).")
    return icl, select, eval_


def two_way_split(
    words: List[Dict[str, Any]],
    n_icl: int,
    n_eval: int,
    seed: int,
) -> Tuple[List[Dict], List[Dict]]:
    """Split into ICL and eval only. Use when no selection is done (e.g. baseline)."""
    if n_icl + n_eval > len(words):
        raise ValueError(
            f"Not enough words: need {n_icl + n_eval}, have {len(words)}."
        )
    set_seed(seed)
    shuffled = list(words)
    random.shuffle(shuffled)
    icl = shuffled[:n_icl]
    eval_ = shuffled[n_icl : n_icl + n_eval]
    s_icl = {w["english"] for w in icl}
    s_eval = {w["english"] for w in eval_}
    if s_icl & s_eval:
        raise RuntimeError("Two-way split overlap detected (bug).")
    return icl, eval_
