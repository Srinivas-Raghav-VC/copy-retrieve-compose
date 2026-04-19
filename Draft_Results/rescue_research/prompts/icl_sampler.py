from __future__ import annotations

import random
from typing import Dict, Iterable, List


def _length_bucket(token: str) -> int:
    n = len(token)
    if n <= 4:
        return 0
    if n <= 8:
        return 1
    if n <= 12:
        return 2
    return 3


def sample_icl_examples(
    *,
    icl_bank: Iterable[Dict[str, str]],
    query_english: str,
    k: int,
    seed: int,
) -> List[Dict[str, str]]:
    """
    Deterministic stratified sampling from ICL bank.

    We exclude the same english lemma as the query to avoid leakage.
    """
    rows = [r for r in icl_bank if r.get("english") != query_english]
    if len(rows) < k:
        raise ValueError(f"Not enough ICL candidates after leakage filter: {len(rows)} < {k}.")

    grouped: Dict[int, List[Dict[str, str]]] = {0: [], 1: [], 2: [], 3: []}
    for row in rows:
        grouped[_length_bucket(str(row.get("source", "")))].append(row)

    rng = random.Random(seed)
    for g in grouped.values():
        rng.shuffle(g)

    picked: List[Dict[str, str]] = []
    bucket_order = [0, 1, 2, 3]
    cursor = 0
    while len(picked) < k:
        b = bucket_order[cursor % len(bucket_order)]
        if grouped[b]:
            rec = grouped[b].pop()
            picked.append(
                {
                    # Demonstrations follow source-script input -> target-script output.
                    "input": rec["source"],
                    "output": rec["target"],
                    "english": rec["english"],
                }
            )
        cursor += 1
        if cursor > (k * 20):
            # Fallback to random among remaining rows.
            remaining = []
            for g in grouped.values():
                remaining.extend(g)
            rng.shuffle(remaining)
            picked.extend(
                {"input": r["source"], "output": r["target"], "english": r["english"]}
                for r in remaining[: k - len(picked)]
            )
            break
    return picked[:k]

