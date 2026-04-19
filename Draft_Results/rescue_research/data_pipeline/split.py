from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ProtocolSplit:
    icl_bank: List[Dict[str, str]]
    selection: List[Dict[str, str]]
    eval_open: List[Dict[str, str]]
    eval_blind: List[Dict[str, str]]


def deterministic_protocol_split(
    records: List[Dict[str, str]],
    *,
    seed: int,
    n_icl_bank: int,
    n_selection: int,
    n_eval_open: int,
    n_eval_blind: int,
) -> ProtocolSplit:
    needed = n_icl_bank + n_selection + n_eval_open + n_eval_blind
    if len(records) < needed:
        raise ValueError(
            f"Not enough records for protocol split: need {needed}, have {len(records)}."
        )
    rows = list(records)
    rng = random.Random(seed)
    rng.shuffle(rows)

    i = 0
    icl_bank = rows[i : i + n_icl_bank]
    i += n_icl_bank
    selection = rows[i : i + n_selection]
    i += n_selection
    eval_open = rows[i : i + n_eval_open]
    i += n_eval_open
    eval_blind = rows[i : i + n_eval_blind]

    # Lemma-level uniqueness guard via english IDs.
    sets = [
        {r["english"] for r in icl_bank},
        {r["english"] for r in selection},
        {r["english"] for r in eval_open},
        {r["english"] for r in eval_blind},
    ]
    for a in range(len(sets)):
        for b in range(a + 1, len(sets)):
            if sets[a].intersection(sets[b]):
                raise RuntimeError("Split overlap detected; expected strict disjointness.")

    return ProtocolSplit(
        icl_bank=icl_bank,
        selection=selection,
        eval_open=eval_open,
        eval_blind=eval_blind,
    )

