from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Iterable, List


def _length_distribution(tokens: Iterable[str]) -> Dict[int, float]:
    counts: Counter[int] = Counter(len(t) for t in tokens)
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in counts.items()}


def _char_distribution(tokens: Iterable[str]) -> Dict[str, float]:
    counts: Counter[str] = Counter()
    for t in tokens:
        counts.update(list(t))
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in counts.items()}


def _js_divergence(p: Dict, q: Dict) -> float:
    keys = set(p.keys()).union(q.keys())
    m = {k: 0.5 * p.get(k, 0.0) + 0.5 * q.get(k, 0.0) for k in keys}

    def _kl(a: Dict, b: Dict) -> float:
        val = 0.0
        for k, pa in a.items():
            if pa <= 0.0:
                continue
            pb = max(1e-12, b.get(k, 0.0))
            val += pa * math.log(pa / pb)
        return val

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def compute_ood_profile(
    *,
    selection_tokens: List[str],
    eval_tokens: List[str],
) -> Dict[str, float]:
    sel_len = _length_distribution(selection_tokens)
    ev_len = _length_distribution(eval_tokens)
    sel_char = _char_distribution(selection_tokens)
    ev_char = _char_distribution(eval_tokens)
    return {
        "js_length": _js_divergence(sel_len, ev_len),
        "js_char": _js_divergence(sel_char, ev_char),
    }

