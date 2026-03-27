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
- Pending rerun: `bash autoresearch.sh smoke`
