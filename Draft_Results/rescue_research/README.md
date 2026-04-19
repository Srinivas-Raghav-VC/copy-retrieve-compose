# Rescue Research — Modular Pipeline

Research-grade reimplementation of the cross-script rescue pipeline with:

- **Single config** — all `n_icl`, `n_select`, `n_eval`, seeds in `config.py` (no inconsistent defaults).
- **No selection bias** — layer choice uses **layer_sweep_cv** (three-way split: ICL / selection / eval).
- **Causal mediation in pipeline** — NIE/NDE and triangulation run as stage 4 and write to `out_dir`.
- **One primary outcome** — pre-registered: PE > 0 and PE > corrupt on held-out eval; written to `primary_outcome.json`.

This package **orchestrates** the reference code in `../experiments/` and `../core.py`; it does not duplicate patching/transcoder logic.

---

## Verification (run before claiming pipeline works)

From the **workspace root** (`fresh_experiments`):

```bash
# 1. Tests pass
python3 -m pytest tests/ -v

# 2. Entry point works
python3 -m rescue_research.run --help

# 3. Transliteration data audit writes readiness/coverage report
python3 scripts/audit_transliteration_pairs.py
```

Only claim the pipeline is working after both commands succeed (verification-before-completion).

---

## Quick start

From the **workspace root** (`fresh_experiments`):

```bash
# Bootstrap env (installs requirements + verifies core imports)
bash scripts/bootstrap_submission_env.sh

# Run full pipeline (baseline → layer_sweep_cv → comprehensive → mediation → primary outcome)
python -m rescue_research.run --stage full --out-dir rescue_research/results

# Run a single stage
python -m rescue_research.run --stage baseline --out-dir rescue_research/results
python -m rescue_research.run --stage layer_sweep_cv --out-dir rescue_research/results
python -m rescue_research.run --stage comprehensive --out-dir rescue_research/results
python -m rescue_research.run --stage mediation --out-dir rescue_research/results
```

---

## Full confirmatory pipeline (new)

Run the locked pipeline DAG:

```bash
python -m rescue_research.run \
  --pipeline full_confirmatory \
  --split-policy adaptive \
  --backend local \
  --out-dir rescue_research/results_maintrack
```

Strict submission preset (high sample-size gate defaults):

```bash
bash scripts/run_submission_strict.sh rescue_research/results_submission_strict
```

Faster wall-clock (same code path, reduced matrix for iterative runs):

```bash
PAIRS="hindi_telugu,english_arabic" MODELS="1b,4b" \
  bash scripts/run_submission_strict.sh rescue_research/results_fast_iter
```

The strict script now sets CUDA allocator/thread defaults:
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- `OMP_NUM_THREADS` / `MKL_NUM_THREADS`

Single-command Modal run (A100-80GB):

```bash
modal run scripts/run_full_confirmatory_modal.py
```

`full_confirmatory` enforces the locked pair matrix by default.
`--allow-custom-pairs` has two modes:
- same length as locked matrix: only pre-approved substitutions are accepted;
- different length (subset/expanded): run is explicitly exploratory and cannot support confirmatory claims.

Dry-run (contracts/manifests only, no heavy model execution):

```bash
python -m rescue_research.run \
  --pipeline full_confirmatory \
  --split-policy adaptive \
  --backend local \
  --no-execute \
  --out-dir rescue_research/results_dryrun
```

Modal job-manifest mode:

```bash
python -m rescue_research.run \
  --pipeline full_confirmatory \
  --split-policy adaptive \
  --backend modal \
  --out-dir rescue_research/results_modal
```

Split sizing is now policy-controlled:

- `--split-policy adaptive` (default): scales protocol counts to available data with explicit minima.
- `--split-policy strict`: requires full protocol counts; errors if data is insufficient.

Adaptive choices are audited in `artifacts/audit/experimental_design.json` and exported as `artifacts/tables/table_7_experimental_design.csv`.
A benchmark/citation manifest is written to `artifacts/manifests/benchmark_registry.json`.

Execution matrix (pair×model):

```bash
python3 scripts/run_execution_matrix.py \
  --out-root rescue_research/results_matrix \
  --pairs hindi_telugu,bengali_odia,tamil_kannada,marathi_gujarati \
  --models 270m,1b,4b,12b \
  --run-quality-eval \
  --compare-variants \
  --run-blind-eval
```

Matrix runner passes `--allow-custom-pairs` by default (exploratory).
Set `--strict-locked-pairs` to disable that behavior.

Matrix outputs:
- `results_matrix/matrix/execution_matrix.json`
- `results_matrix/matrix/matrix_results.jsonl`
- `results_matrix/matrix/summary.json`

Readiness guardrails:
- Confirmatory pair-readiness defaults are set to runnable minima (`pool>=40`, `icl>=4`, `selection>=12`, `eval>=24`).
- These minima are still below conference-grade evidence floors; use the paper-readiness audit to distinguish runnable pilots from publishable runs.
- Audit report: `artifacts/audit/pair_readiness.json`
- Exploratory override only: `--allow-underpowered-pairs`

Protocol/reproducibility audits:
- `artifacts/audit/protocol_compliance.json`
- `artifacts/audit/reproducibility_check.json`
- `artifacts/audit/transcoder_fidelity_gate.json`
- `artifacts/audit/submission_gates.json`
- `artifacts/manifests/pipeline_timing.json`

---

## Data flow

- **Canonical word source:** experiments (baseline, layer_sweep, comprehensive) load through `rescue_research.data_pipeline.ingest` (built-ins + optional external transliteration files).
- **Main-track evidence rule:** confirmatory paper claims should use provenance-tracked external transliteration files under `data/transliteration/` or configured in `rescue_research/configs/datasets.yaml`. Built-ins are acceptable for smoke/pilot/fallback analyses only.
- **Prepared splits (execution source of truth):** `stage_prepare_data` writes deterministic splits to `data/processed/<pair_id>/split_seed_<seed>.json`, and downstream stages consume those files directly. This closes ad-hoc random split drift and keeps blind slices sealed unless `--run-blind-eval` is explicitly enabled.

External dataset build helper (provenance + deterministic sampling):

```bash
python3 scripts/build_confirmatory_external_data.py \
  --pair-id english_arabic \
  --hf-dataset moha/Arabic-English-Transliteration-Dataset \
  --hf-split train \
  --max-rows 5000 \
  --seed 42
```

Then verify pool readiness with `python3 scripts/audit_transliteration_pairs.py`.

---

## Stages (order matters for `--stage full`)

| Stage | What it does | Outputs |
|-------|----------------|--------|
| **baseline** | ICL vs ZS (reference exp1) | `baseline_{model}.json` |
| **layer_sweep_cv** | Layer ranking on **selection** split; eval on **eval** split | `layer_sweep_cv_{model}.json`, `best_layer.txt` |
| **comprehensive** | Patching at best layer with controls (reference exp3, three-way) | `comprehensive_{model}_L{layer}.json` |
| **mediation** | NIE/NDE, triangulation (reference causal_mediation) | `mediation_{model}_L{layer}.json` |
| *(after full)* | Primary outcome | `primary_outcome.json` |

---

## Config

Edit `rescue_research/config.py` for:

- `N_ICL`, `N_SELECT`, `N_EVAL`, `SEEDS`
- `DEFAULT_MODEL`, `DEFAULT_LAYER`, `TOP_K_VALUES`
- `PRIMARY_OUTCOME_DESCRIPTION`, `PRIMARY_ALPHA`

Or override via CLI: `--n-icl`, `--n-select`, `--n-eval`, `--seeds`, `--model`, `--layer`.

---

## Debugging

- **Single stage:** run one stage at a time and inspect JSON under `--out-dir`.
- **Logs:** all stages print to stderr; use `--log-file PATH` to capture (if wired in run.py).
- **Reference scripts:** each stage calls `experiments/exp1_baseline.py`, `exp2_layer_sweep_cv.py`, `exp3_comprehensive.py`, or `causal_mediation.run_causal_mediation_experiment`; run those scripts directly with the same args for step-through debugging.
- **Data:** pair records flow through `rescue_research.data_pipeline.ingest`, and deterministic protocol splits are written by `stage_prepare_data`.

---

## Design choices (vs original codebase)

1. **Layer choice:** only **exp2_layer_sweep_cv** (held-out eval); original exp2 (no CV) is not used.
2. **Comprehensive:** uses **n_select > 0** so selection and eval are disjoint; no silent fallback to two-way.
3. **Causal mediation:** **stage 4** runs `run_causal_mediation_experiment` and saves results; no longer optional/unused.
4. **Primary outcome:** one pre-registered check written to `primary_outcome.json`.
5. **Single entry point:** `python -m rescue_research.run`; no separate shell pipeline script.

---

## Eval and direction

See [docs/README.md](../docs/README.md) for eval (CER, LLM-as-judge, sample size) and direction (fix vs pivot).
Main-track protocol assets are in `../docs/MAIN_TRACK_PROTOCOL.md` and `../docs/MODAL_PIPELINE_RUNBOOK.md`.

## Extended analysis outputs

`stage_report_bundle` now writes:
- `artifacts/stats/attention_control_summary.json`
- `artifacts/stats/transcoder_variant_summary.json`
- `artifacts/stats/prompt_format_robustness.json` (format-spread audit from robustness stage)
- `artifacts/tables/table_8_attention_controls.csv`
- `artifacts/tables/table_9_transcoder_variants.csv`
- `artifacts/figures/figure_8_attention_controls.png`
- `artifacts/figures/figure_9_transcoder_variant_deltas.png`

If `matplotlib` is unavailable, figure generation is skipped but stats/tables/contracts still complete.
