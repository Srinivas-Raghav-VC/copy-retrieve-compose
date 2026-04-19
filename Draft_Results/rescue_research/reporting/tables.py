from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List


MANDATORY_TABLES = [
    "table_1_data_composition.csv",
    "table_2_confirmatory_tests.csv",
    "table_3_control_sanity.csv",
    "table_4_icl_bank_composition.csv",
    "table_5_mediation_band.csv",
    "table_6_quality_metrics.csv",
    "table_7_experimental_design.csv",
    "table_8_attention_controls.csv",
    "table_9_transcoder_variants.csv",
]


def _read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _iter_json_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return sorted(root.glob("**/*.json"))


def _write_csv(path: Path, headers: List[str], rows: List[List[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(row)


def _table_1_data_composition(out_dir: Path, out_path: Path) -> None:
    audit_root = out_dir / "artifacts" / "audit"
    rows: List[List[object]] = []
    for p in _iter_json_files(audit_root):
        if not p.name.startswith("data_quality_report_"):
            continue
        payload = _read_json(p)
        pair_id = p.stem.replace("data_quality_report_", "")
        rows.append(
            [
                pair_id,
                payload.get("total"),
                payload.get("kept"),
                payload.get("dropped_empty"),
                payload.get("dropped_script_mismatch"),
                payload.get("dropped_length_bounds"),
            ]
        )
    _write_csv(
        out_path,
        [
            "pair_id",
            "total",
            "kept",
            "dropped_empty",
            "dropped_script_mismatch",
            "dropped_length_bounds",
        ],
        rows,
    )


def _table_2_confirmatory(out_dir: Path, out_path: Path) -> None:
    payload = _read_json(out_dir / "artifacts" / "stats" / "confirmatory_results.json")
    rows_in = payload.get("rows", []) if isinstance(payload, dict) else []
    rows: List[List[object]] = []
    for row in rows_in if isinstance(rows_in, list) else []:
        if not isinstance(row, dict):
            continue
        rows.append(
            [
                row.get("pair_id"),
                row.get("model"),
                row.get("seed"),
                row.get("primary_passed"),
                row.get("selected_topk"),
                row.get("mean_pe"),
                row.get("mean_pe_logit"),
                row.get("p_holm"),
                row.get("mean_pe_corrupt"),
                row.get("mean_pe_random"),
                row.get("mean_pe_shuffle"),
                row.get("mean_pe_gauss"),
                row.get("mean_pe_attention"),
                row.get("mean_pe_basis"),
                row.get("mean_ae"),
                row.get("ci_pe_95"),
                row.get("ci_pe_corrupt_95"),
                row.get("ci_ae_95"),
                row.get("p_ae_lt_0_one_tailed"),
            ]
        )
    _write_csv(
        out_path,
        [
            "pair_id",
            "model",
            "seed",
            "primary_passed",
            "selected_topk",
            "mean_pe",
            "mean_pe_logit",
            "p_holm",
            "mean_pe_corrupt",
            "mean_pe_random",
            "mean_pe_shuffle",
            "mean_pe_gauss",
            "mean_pe_attention",
            "mean_pe_basis",
            "mean_ae",
            "ci_pe_95",
            "ci_pe_corrupt_95",
            "ci_ae_95",
            "p_ae_lt_0_one_tailed",
        ],
        rows,
    )


def _table_3_control_sanity(out_dir: Path, out_path: Path) -> None:
    interventions_root = out_dir / "artifacts" / "interventions"
    rows: List[List[object]] = []
    for p in _iter_json_files(interventions_root):
        payload = _read_json(p)
        pair_id = payload.get("pair_id")
        model = payload.get("model")
        seed = payload.get("seed")
        topk_agg = payload.get("topk_aggregate", {})
        if not isinstance(topk_agg, dict):
            continue
        for topk, stats in topk_agg.items():
            if not isinstance(stats, dict):
                continue
            rows.append(
                [
                    pair_id,
                    model,
                    seed,
                    topk,
                    stats.get("mean_pe"),
                    stats.get("mean_pe_corrupt"),
                    stats.get("mean_pe_random"),
                    stats.get("mean_pe_shuffle"),
                    stats.get("mean_pe_gauss"),
                    stats.get("mean_pe_attention"),
                    stats.get("mean_pe_basis"),
                    (stats.get("task_matched_control", {}) or {}).get(
                        "mean_pe_minus_corrupt"
                    ),
                ]
            )
    _write_csv(
        out_path,
        [
            "pair_id",
            "model",
            "seed",
            "topk",
            "mean_pe",
            "mean_pe_corrupt",
            "mean_pe_random",
            "mean_pe_shuffle",
            "mean_pe_gauss",
            "mean_pe_attention",
            "mean_pe_basis",
            "mean_pe_minus_corrupt",
        ],
        rows,
    )


def _table_4_icl_bank_composition(out_dir: Path, out_path: Path) -> None:
    data_root = out_dir / "data" / "processed"
    rows: List[List[object]] = []
    if data_root.exists():
        for pair_dir in sorted([p for p in data_root.iterdir() if p.is_dir()]):
            for split_file in sorted(pair_dir.glob("split_seed_*.json")):
                payload = _read_json(split_file)
                seed = split_file.stem.replace("split_seed_", "")
                icl_bank = payload.get("icl_bank", [])
                rows.append(
                    [
                        pair_dir.name,
                        seed,
                        len(icl_bank) if isinstance(icl_bank, list) else 0,
                    ]
                )
    _write_csv(out_path, ["pair_id", "seed", "icl_bank_count"], rows)


def _table_5_mediation_band(out_dir: Path, out_path: Path) -> None:
    payload = _read_json(out_dir / "artifacts" / "stats" / "mediation_band_summary.json")
    rows_in = payload.get("rows", []) if isinstance(payload, dict) else []
    rows: List[List[object]] = []
    for row in rows_in if isinstance(rows_in, list) else []:
        if not isinstance(row, dict):
            continue
        pair_id = row.get("pair_id")
        model = row.get("model")
        best_layer = row.get("best_layer")
        top_k = row.get("top_k_layers", [])
        per_layer = row.get("per_layer", [])
        for pl in per_layer if isinstance(per_layer, list) else []:
            if isinstance(pl, dict):
                ci = pl.get("ci_95")
                ci_str = str(ci) if ci is not None else ""
                rows.append(
                    [
                        pair_id,
                        model,
                        best_layer,
                        ",".join(str(x) for x in top_k) if top_k else "",
                        pl.get("layer"),
                        pl.get("mean_nie"),
                        ci_str,
                    ]
                )
    _write_csv(
        out_path,
        ["pair_id", "model", "best_layer", "top_k_layers", "layer", "mean_nie", "ci_95"],
        rows if rows else [["(no band data)", "", "", "", "", "", ""]],
    )


def _table_6_quality_metrics(out_dir: Path, out_path: Path) -> None:
    baseline_root = out_dir / "artifacts" / "baseline"
    rows: List[List[object]] = []
    for p in _iter_json_files(baseline_root):
        payload = _read_json(p)
        pair_id = str(payload.get("pair_id", p.parent.name))
        model = str(payload.get("model", p.parent.parent.name))
        seed = str(payload.get("seed", p.stem))
        stage = (payload.get("stage_output") or {}) if isinstance(payload, dict) else {}
        stats = (stage.get("stats") or {}) if isinstance(stage, dict) else {}
        cer_z = stats.get("mean_cer_zs")
        cer_i = stats.get("mean_cer_icl")
        em_z = stats.get("exact_match_zs")
        em_i = stats.get("exact_match_icl")
        rows.append([
            pair_id, model, seed,
            cer_z if cer_z is not None else "", cer_i if cer_i is not None else "",
            em_z if em_z is not None else "", em_i if em_i is not None else "",
        ])
    _write_csv(
        out_path,
        ["pair_id", "model", "seed", "mean_cer_zs", "mean_cer_icl", "exact_match_zs", "exact_match_icl"],
        rows if rows else [["(no quality data)", "", "", "", "", "", ""]],
    )


def _table_7_experimental_design(out_dir: Path, out_path: Path) -> None:
    payload = _read_json(out_dir / "artifacts" / "audit" / "experimental_design.json")
    pairs = payload.get("pairs", {}) if isinstance(payload, dict) else {}
    split_policy = payload.get("split_policy", "") if isinstance(payload, dict) else ""
    rows: List[List[object]] = []
    if isinstance(pairs, dict):
        for pair_id, row in pairs.items():
            if not isinstance(row, dict):
                continue
            four = row.get("four_way_plan", {}) if isinstance(row, dict) else {}
            three = row.get("three_way_runtime_plan", {}) if isinstance(row, dict) else {}
            rows.append([
                pair_id,
                split_policy,
                row.get("available_records"),
                four.get("n_icl_bank"),
                four.get("n_selection"),
                four.get("n_eval_open"),
                four.get("n_eval_blind"),
                three.get("n_icl"),
                three.get("n_selection"),
                three.get("n_eval"),
                "|".join(three.get("warnings", []) if isinstance(three.get("warnings"), list) else []),
                "|".join(four.get("warnings", []) if isinstance(four.get("warnings"), list) else []),
            ])
    _write_csv(
        out_path,
        [
            "pair_id",
            "split_policy",
            "available_records",
            "four_way_icl_bank",
            "four_way_selection",
            "four_way_eval_open",
            "four_way_eval_blind",
            "runtime_n_icl",
            "runtime_n_selection",
            "runtime_n_eval",
            "runtime_warnings",
            "four_way_warnings",
        ],
        rows if rows else [["(no design data)", "", "", "", "", "", "", "", "", "", "", ""]],
    )


def _table_8_attention_controls(out_dir: Path, out_path: Path) -> None:
    payload = _read_json(out_dir / "artifacts" / "stats" / "attention_control_summary.json")
    rows_in = payload.get("by_pair_model", []) if isinstance(payload, dict) else []
    rows: List[List[object]] = []
    for r in rows_in if isinstance(rows_in, list) else []:
        if not isinstance(r, dict):
            continue
        rows.append(
            [
                r.get("pair_id"),
                r.get("model"),
                r.get("mean_pe"),
                r.get("mean_pe_corrupt"),
                r.get("mean_pe_attention"),
                r.get("mean_pe_random"),
                r.get("mean_pe_shuffle"),
                r.get("mean_pe_gauss"),
                r.get("mean_pe_basis"),
                r.get("mean_pe_minus_corrupt"),
            ]
        )
    _write_csv(
        out_path,
        [
            "pair_id",
            "model",
            "mean_pe",
            "mean_pe_corrupt",
            "mean_pe_attention",
            "mean_pe_random",
            "mean_pe_shuffle",
            "mean_pe_gauss",
            "mean_pe_basis",
            "mean_pe_minus_corrupt",
        ],
        rows if rows else [["(no attention control data)", "", "", "", "", "", "", "", "", ""]],
    )


def _table_9_transcoder_variants(out_dir: Path, out_path: Path) -> None:
    payload = _read_json(out_dir / "artifacts" / "stats" / "transcoder_variant_summary.json")
    rows_in = payload.get("pair_model_deltas", []) if isinstance(payload, dict) else []
    rows: List[List[object]] = []
    for r in rows_in if isinstance(rows_in, list) else []:
        if not isinstance(r, dict):
            continue
        rows.append(
            [
                r.get("pair_id"),
                r.get("model"),
                r.get("affine_skip_mean_pe"),
                r.get("skipless_mean_pe"),
                r.get("delta_affine_minus_skipless"),
            ]
        )
    _write_csv(
        out_path,
        [
            "pair_id",
            "model",
            "affine_skip_mean_pe",
            "skipless_mean_pe",
            "delta_affine_minus_skipless",
        ],
        rows if rows else [["(no transcoder variant data)", "", "", "", ""]],
    )


def generate_mandatory_tables(out_dir: Path) -> List[Path]:
    table_dir = out_dir / "artifacts" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    paths = [table_dir / name for name in MANDATORY_TABLES]
    _table_1_data_composition(out_dir, paths[0])
    _table_2_confirmatory(out_dir, paths[1])
    _table_3_control_sanity(out_dir, paths[2])
    _table_4_icl_bank_composition(out_dir, paths[3])
    _table_5_mediation_band(out_dir, paths[4])
    _table_6_quality_metrics(out_dir, paths[5])
    _table_7_experimental_design(out_dir, paths[6])
    _table_8_attention_controls(out_dir, paths[7])
    _table_9_transcoder_variants(out_dir, paths[8])
    return paths

