# Autoresearch Session: Loop 1 cross-scale transliteration anchor

_Date started: 2026-03-27_

## Honors north star

Understand how Gemma 3 models perform multilingual transliteration, how in-context learning changes that computation, and which candidate mechanisms or candidate circuits are shared or changed across `270M`, `1B`, and `4B`.

## Why Loop 1 exists

Loop 1 is **not** the full thesis.

It is the bounded search for the first **behavioral anchor** strong enough to justify later mechanistic work.

The question for this loop is:

> Across `270M`, `1B`, and `4B`, where do we see a real transliteration-specific ICL gap that is worth opening up mechanistically?

## Approved environment

- **Modal**
- **new git branch**: `autoresearch-loop1-cross-scale-anchor`

## Current executable benchmark

Because the current cross-scale Modal suite already supports `premise_gate` cleanly, the initial executable benchmark for Loop 1 is:

- **metric:** `premise_gap_exact_mean`
- **unit:** exact-match points
- **direction:** higher is better

Definition:
- for each of the 6 model-language tasks (`270M/1B/4B × Hindi/Telugu`), compute
  - `exact_match(icl64) - exact_match(explicit_zs)`
- then average across tasks

This is a **behavioral anchor metric**, not the final thesis metric.

## Benchmark command

```bash
bash autoresearch.sh smoke
```

A later confirmation run can use:

```bash
bash autoresearch.sh full
```

## Fixed scope for Loop 1

### Models
- `270m`
- `1b`
- `4b`

### Languages
- `aksharantar_hin_latin`
- `aksharantar_tel_latin`

### Tasks
- `premise_gate__270m__aksharantar_hin_latin`
- `premise_gate__270m__aksharantar_tel_latin`
- `premise_gate__1b__aksharantar_hin_latin`
- `premise_gate__1b__aksharantar_tel_latin`
- `premise_gate__4b__aksharantar_hin_latin`
- `premise_gate__4b__aksharantar_tel_latin`

## Files in scope

- `autoresearch.md`
- `autoresearch.sh`
- `experiments/score_cross_scale_anchor.py`
- `Draft_Results/paper2_fidelity_calibrated/multiscale_modal_suite/*`
- supporting config / infra files needed to make the benchmark honest and runnable

## Guardrails

Track, but do not optimize as the main score:
- `premise_gap_cer_mean` higher is better
- count of tasks whose exact-match CI excludes zero
- count of tasks passing the runbook gate

## Max iterations

- **8** for Loop 1

## Stop conditions

### Stop with GO if
- at least one model-language pair shows a meaningful premise gap
- and the cross-scale pattern is strong enough to choose a first mechanistic target

### Stop with NO-GO if
- after bounded iterations, the cross-scale premise gap remains too weak, too noisy, or too broken to justify mechanism-first work

## Initial implementation note

The originally discussed ideal metric was a stricter helpful-vs-random/corrupt anchor across all three models.

The **current executable baseline** starts from `premise_gate` because that is the cross-scale lane already wired for `270M`, `1B`, and `4B`. If Loop 1 stabilizes, later iterations can strengthen the benchmark toward richer controls.

## Iteration log

### Iteration 0 — initialization
- Created fresh autoresearch session files.
- Chosen branch/environment: Modal + git branch.
- Began wiring the multiscale Modal suite into a runnable baseline for the cross-scale premise gate.
- First smoke launch failed immediately because `modal_app.py` was being executed as a script by `modal run`, while the suite file assumed package-relative imports.
- Patched `modal_app.py` to support direct path execution with absolute-import fallback.
- Second smoke launch got through image build, but Modal aborted because the benchmark harness was writing `modal_run.log` into the workspace while the local workspace snapshot was still being used for build/input hashing.
- Patched `autoresearch.sh` so the live Modal log is written under `/tmp` during execution and only moved into the workspace after the run completes.
- Third smoke launch passed local preflight and Modal image build, but the remote container still failed before any `premise_gate` task executed.
- Current blocker: the remote Modal container cannot import `Draft_Results`, so the suite entrypoint fails with `ModuleNotFoundError: No module named 'Draft_Results'` and retries.
- I terminated the stuck local wrapper process after confirming it was not making forward progress.
- Patched `modal_app.py` so it bootstraps `/workspace` import paths before module imports and resolves the workspace root safely in both local and remote contexts.
- The next smoke run got further: local preflight passed, the Modal app started, and tasks launched, but the task payloads still used the local results root (`/mnt/d/...`) instead of the remote Modal volume root.
- Patched the multiscale suite result-root handling so task expansion and plan writing resolve `MULTISCALE_RESULTS_ROOT` at runtime rather than freezing the local path at import time.
- After that fix, the remote tasks finally executed and exposed the first real benchmark blocker: `run_premise_gate.py` could not find usable external Aksharantar records in the Modal workspace.
- Materialized explicit external JSONL sources plus provenance sidecars for Hindi and Telugu under `Draft_Results/data/transliteration/` so the smoke benchmark has real external data to consume.
- Also fixed `autoresearch.sh` to download the Modal volume from the volume root (`/`) rather than the in-container mount path.
- Smoke baseline completed successfully across all 6 premise-gate tasks.
- Baseline smoke findings:
  - `premise_gap_exact_mean = 0.0347`
  - `premise_gap_cer_mean = 0.3036`
  - strongest positive anchor: `4b × aksharantar_tel_latin` with `exact_match 0.000 -> 0.292` and positive CER gap, both CI-supported
  - `270m` appears flat on both languages
  - `1b × aksharantar_hin_latin` worsens under `icl64`
  - `4b × aksharantar_hin_latin` is already fairly strong zero-shot and shows little exact-match rescue
- Full six-task premise-gate run (`n_eval=200`) completed successfully.
- Full findings:
  - `premise_gap_exact_mean = 0.065`
  - `premise_gap_cer_mean = 0.2567`
  - strongest positive anchor remains `4b × aksharantar_tel_latin` with `exact_match 0.000 -> 0.335` and strong CER improvement (`2.380 -> 0.247`), CI-supported
  - `270m` remains flat on both languages across explicit-ZS, `icl8`, and `icl64`
  - `1b × aksharantar_hin_latin` shows high-N fragility: explicit-ZS beats `icl64`, while `icl8` is less harmful, suggesting a long-context problem rather than a total inability to use examples
  - `1b × aksharantar_tel_latin` does not gain exact match, and `icl64` is slightly worse than explicit-ZS on CER
  - `4b × aksharantar_hin_latin` is already strong under explicit-ZS and gains only moderately from ICL (`0.300 -> 0.360` EM)
- Interpretation status:
  - established from this iteration: the cross-scale premise landscape is structured, not random
  - supported but provisional: `4b × Telugu` is the best positive mechanistic anchor
  - supported but provisional: `1b × Hindi` is a useful fragility / high-N anchor
- Recommended next iteration: targeted robustness / control comparisons on `4b × Telugu` and `1b × Hindi`, especially low-shot vs high-shot and helpful-vs-control variants.
