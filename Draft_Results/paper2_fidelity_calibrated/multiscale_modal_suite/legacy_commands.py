from __future__ import annotations

from pathlib import Path

from .suite_spec import PAPER2_DIR, SuiteTask, resolve_results_root


def _script_path(name: str) -> str:
    return str(PAPER2_DIR / name)


def build_command(task: SuiteTask, *, smoke: bool = False) -> list[str]:
    n_eval = task.smoke_n_eval if smoke else task.n_eval
    out_root = Path(task.out_dir)

    if task.command_kind == "phase0_token_visibility":
        return [
            "python3",
            _script_path("run_phase0_token_visibility.py"),
            "--models",
            str(task.model),
            "--pairs",
            str(task.pair),
            "--seed",
            "42",
            "--n-icl",
            str(task.n_icl),
            "--n-select",
            str(task.n_select),
            "--n-eval",
            str(n_eval),
            "--out-root",
            str(out_root / "phase0"),
            "--external-only",
            "--require-external-sources",
        ]

    if task.command_kind == "premise_gate":
        model = _required(task.model, "model")
        pair = _required(task.pair, "pair")
        return [
            "python3",
            _script_path("run_premise_gate.py"),
            "--model",
            model,
            "--pair",
            pair,
            "--seed",
            "42",
            "--n-icl",
            str(task.n_icl),
            "--n-select",
            str(task.n_select),
            "--n-eval",
            str(n_eval),
            "--external-only",
            "--require-external-sources",
            "--out",
            str(out_root / "premise_gate" / model / f"{pair}.json"),
        ]

    if task.command_kind == "fidelity_sanity":
        model = _required(task.model, "model")
        pair = _required(task.pair, "pair")
        return [
            "python3",
            _script_path("fidelity_sanity_check.py"),
            "--model",
            model,
            "--pair",
            pair,
            "--n-samples",
            str(n_eval),
            "--device",
            "cuda",
            "--out-csv",
            str(out_root / "fidelity" / model / f"{pair}.csv"),
        ]

    if task.command_kind == "script_space_map":
        return [
            "python3",
            _script_path("run_script_space_map.py"),
            "--tasks",
            ",".join(
                f"{model}:{pair}"
                for model in str(task.model).split(",")
                for pair in str(task.pair).split(",")
            ),
            "--seed",
            "42",
            "--n-icl",
            str(task.n_icl),
            "--n-select",
            str(task.n_select),
            "--n-eval",
            str(n_eval),
            "--max-words",
            str(min(50, n_eval)),
            "--device",
            "cuda",
            "--external-only",
            "--require-external-sources",
            "--out-root",
            str(out_root / "script_space"),
        ]

    if task.command_kind == "paper2_cfom":
        model = _required(task.model, "model")
        pair = _required(task.pair, "pair")
        seeds = "42,123" if smoke else "42,123,456"
        return [
            "python3",
            _script_path("run.py"),
            "--model",
            model,
            "--pair",
            pair,
            "--seeds",
            seeds,
            "--device",
            "cuda",
            "--n-icl",
            str(task.n_icl),
            "--n-select",
            str(task.n_select),
            "--n-eval",
            str(n_eval),
            "--external-only",
            "--require-external-sources",
            "--min-pool-size",
            str(max(500, task.n_icl + task.n_select + n_eval)),
            "--out",
            str(out_root / "paper2" / model / f"{pair}.json"),
        ]

    if task.command_kind == "one_b_final_bundle":
        return [
            "python3",
            _script_path("run_1b_final_gpu_bundle.py"),
            "--tasks",
            "mlp,head,attn-only,density,joint,content-count,seed-robust",
            "--device",
            "cuda",
            "--out-root",
            str(out_root / "1b_final"),
            "--skip-existing",
        ] + (["--smoke"] if smoke else [])

    raise KeyError(f"Unsupported command kind: {task.command_kind}")


def expected_script_paths() -> list[Path]:
    names = [
        "run_phase0_token_visibility.py",
        "run_premise_gate.py",
        "fidelity_sanity_check.py",
        "run_script_space_map.py",
        "run.py",
        "run_1b_final_gpu_bundle.py",
    ]
    return [PAPER2_DIR / name for name in names]


def _required(value: str | None, label: str) -> str:
    if value is None or not str(value).strip():
        raise ValueError(f"Task missing {label}")
    return str(value)


def default_results_root() -> Path:
    return resolve_results_root()
