from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple


def _as_rows(payload: object, key: str) -> List[Dict[str, str]]:
    if not isinstance(payload, dict):
        return []
    raw = payload.get(key, [])
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        english = str(item.get("english", "")).strip()
        source = str(item.get("source", "")).strip()
        target = str(item.get("target", "")).strip()
        if not english or not source or not target:
            continue
        out.append({"english": english, "source": source, "target": target})
    return out


def _legacy_word(row: Dict[str, str]) -> Dict[str, str]:
    src = str(row.get("source", "")).strip()
    tgt = str(row.get("target", "")).strip()
    return {
        "english": str(row.get("english", "")).strip(),
        "hindi": src,
        "source": src,
        "ood": tgt,
        "telugu": tgt,
        "target": tgt,
    }


def _assert_disjoint(parts: List[List[Dict[str, str]]]) -> None:
    sets = []
    for rows in parts:
        sets.append({str(r.get("english", "")).strip() for r in rows})
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            if sets[i].intersection(sets[j]):
                raise ValueError("Prepared split overlap detected across runtime partitions.")


def load_prepared_split_file(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Prepared split file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to parse prepared split JSON at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Prepared split payload must be a JSON object: {path}")
    return payload


def load_prepared_split_for_seed(prepared_split_dir: Path, seed: int) -> Tuple[Dict, Path]:
    split_path = prepared_split_dir / f"split_seed_{int(seed)}.json"
    payload = load_prepared_split_file(split_path)
    return payload, split_path


def runtime_three_way_from_prepared(
    *,
    payload: Dict,
    n_icl: int,
    n_select: int,
    n_eval: int,
    use_blind_eval: bool,
    split_path: Path | None = None,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]], Dict[str, object]]:
    icl_bank = _as_rows(payload, "icl_bank")
    selection = _as_rows(payload, "selection")
    eval_open = _as_rows(payload, "eval_open")
    eval_blind = _as_rows(payload, "eval_blind")
    eval_pool = list(eval_open) + (list(eval_blind) if bool(use_blind_eval) else [])

    if len(icl_bank) < int(n_icl):
        raise ValueError(
            f"Prepared split has insufficient icl_bank rows: need {int(n_icl)}, have {len(icl_bank)}"
        )
    if len(selection) < int(n_select):
        raise ValueError(
            f"Prepared split has insufficient selection rows: need {int(n_select)}, have {len(selection)}"
        )
    if len(eval_pool) < int(n_eval):
        mode = "eval_open+eval_blind" if bool(use_blind_eval) else "eval_open"
        raise ValueError(
            f"Prepared split has insufficient {mode} rows: need {int(n_eval)}, have {len(eval_pool)}"
        )

    icl_rows = [_legacy_word(r) for r in icl_bank[: int(n_icl)]]
    sel_rows = [_legacy_word(r) for r in selection[: int(n_select)]]
    eval_rows = [_legacy_word(r) for r in eval_pool[: int(n_eval)]]

    _assert_disjoint([icl_rows, sel_rows, eval_rows])

    meta: Dict[str, object] = {
        "split_source": "prepared_protocol_split",
        "split_path": str(split_path) if split_path is not None else "",
        "use_blind_eval": bool(use_blind_eval),
        "available": {
            "icl_bank": len(icl_bank),
            "selection": len(selection),
            "eval_open": len(eval_open),
            "eval_blind": len(eval_blind),
            "eval_pool": len(eval_pool),
        },
        "used": {
            "icl": int(n_icl),
            "selection": int(n_select),
            "eval": int(n_eval),
        },
    }
    return icl_rows, sel_rows, eval_rows, meta


def runtime_two_way_from_prepared(
    *,
    payload: Dict,
    n_icl: int,
    n_eval: int,
    use_blind_eval: bool,
    split_path: Path | None = None,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], Dict[str, object]]:
    icl_bank = _as_rows(payload, "icl_bank")
    eval_open = _as_rows(payload, "eval_open")
    eval_blind = _as_rows(payload, "eval_blind")
    eval_pool = list(eval_open) + (list(eval_blind) if bool(use_blind_eval) else [])

    if len(icl_bank) < int(n_icl):
        raise ValueError(
            f"Prepared split has insufficient icl_bank rows: need {int(n_icl)}, have {len(icl_bank)}"
        )
    if len(eval_pool) < int(n_eval):
        mode = "eval_open+eval_blind" if bool(use_blind_eval) else "eval_open"
        raise ValueError(
            f"Prepared split has insufficient {mode} rows: need {int(n_eval)}, have {len(eval_pool)}"
        )

    icl_rows = [_legacy_word(r) for r in icl_bank[: int(n_icl)]]
    eval_rows = [_legacy_word(r) for r in eval_pool[: int(n_eval)]]

    _assert_disjoint([icl_rows, eval_rows])

    meta: Dict[str, object] = {
        "split_source": "prepared_protocol_split",
        "split_path": str(split_path) if split_path is not None else "",
        "use_blind_eval": bool(use_blind_eval),
        "available": {
            "icl_bank": len(icl_bank),
            "eval_open": len(eval_open),
            "eval_blind": len(eval_blind),
            "eval_pool": len(eval_pool),
        },
        "used": {
            "icl": int(n_icl),
            "eval": int(n_eval),
        },
    }
    return icl_rows, eval_rows, meta
