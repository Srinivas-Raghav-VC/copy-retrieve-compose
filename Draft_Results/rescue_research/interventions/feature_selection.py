from __future__ import annotations

from typing import Iterable, List


def select_feature_indices(scores: Iterable[float], *, topk: int, strategy: str) -> List[int]:
    pairs = list(enumerate(float(x) for x in scores))
    s = strategy.strip().lower()
    if s.startswith("bottomk"):
        pairs.sort(key=lambda kv: abs(kv[1]))
    else:
        pairs.sort(key=lambda kv: abs(kv[1]), reverse=True)
    return [idx for idx, _ in pairs[:topk]]

