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

## Loop 2 — VM control verification

_Date approved: 2026-03-29_

### Loop 2 objective

Verify the two-language behavioral story before any language expansion or mechanistic probing.

The core question for this loop is:

> For the `1B` and `4B` models on Hindi and Telugu, do genuinely helpful examples beat matched controls, and does `1B` specifically show a high-N fragility pattern rather than a total inability to use examples?

### Approved environment

- **Shared A100 VM**
- approved by user after Loop 1 to favor interactive inspection and continuity into later mechanistic work

### Core panel

- `1b × aksharantar_hin_latin`
- `1b × aksharantar_tel_latin`
- `4b × aksharantar_hin_latin`
- `4b × aksharantar_tel_latin`

### Conditions for this loop

Run `run_neutral_filler_recency_controls.py` on the core panel with:

- `n_icl = 8`
- `n_icl = 64`
- helpful examples
- corrupted examples
- random Indic controls
- null filler controls
- recency / ordering variants already emitted by the script

### Current executable benchmark

Primary metric:

- **`helpful_control_exact_margin_mean`**
- unit: exact-match points
- direction: higher is better

Definition:
- for each `(model, pair, n_icl)` cell, compute
  - `exact_match(icl_helpful) - max(exact_match(icl_corrupt), exact_match(icl_random_indic), exact_match(icl_null_filler))`
- then average across the 8 expected cells (`1B/4B × Hindi/Telugu × n_icl {8,64}`)

### Guardrails

Track, but do not optimize as the main score:

- `helpful_control_cer_margin_mean` higher is better
- `helpful_minus_zs_exact_mean` higher is better
- `one_b_highN_helpful_cer_regret_mean` higher means `1B` gets worse at `n_icl=64` than at `n_icl=8`
- `four_b_highN_helpful_cer_gain_mean` higher means `4B` benefits from `n_icl=64` over `n_icl=8`

### Benchmark commands

Smoke baseline:

```bash
VM_PASS='***' bash autoresearch.sh loop2_smoke
```

Full baseline:

```bash
VM_PASS='***' bash autoresearch.sh loop2_full
```

### Files in scope

- `autoresearch.md`
- `autoresearch.sh`
- `experiments/score_loop2_controls.py`
- `Draft_Results/paper2_fidelity_calibrated/run_neutral_filler_recency_controls.py`
- small supporting harness/config files needed to make the VM benchmark honest and runnable

### Max iterations

- **6** for Loop 2

### Stop conditions

#### Stop with GO if
- helpful examples beat matched controls on at least one clean anchor, especially `4B × Telugu`
- and the `1B` vs `4B` contrast is sharp enough to justify later expansion and mechanistic probing

#### Stop with NO-GO if
- the helpful-vs-control advantage collapses under direct controls
- or VM execution remains too brittle to produce a reliable baseline within the bounded iteration budget

### Iteration log

#### Iteration 0 — initialization and VM bring-up
- User approved the Loop 2 order explicitly: verify the `1B/4B × Hindi/Telugu` panel first, then expand languages, then do mechanistic probing.
- Created a fresh branch for Loop 2: `autoresearch-loop2-vm-controls`.
- Added a dedicated Loop 2 scorer at `experiments/score_loop2_controls.py`.
- Extended `autoresearch.sh` with VM-backed Loop 2 modes: `loop2_smoke` and `loop2_full`.
- Immediate blocker on first bring-up attempt: the shared VM was unreachable from this environment (`ssh` timeout, port 22 closed), so the benchmark could not be launched yet.
- Retried VM connectivity and the host became reachable again (`hostname=2f399cd77c0c`, `Python 3.8.5`).
- First smoke launch on the reachable VM failed immediately because the remote machine does not have `rsync`; the old sync path assumed remote `rsync` availability.
- Patched `autoresearch.sh` to use a tar-over-SSH sync/fetch path instead of `rsync`, which is more robust for this VM.
- Second smoke launch got into the remote benchmark body and failed at the first `1b × Hindi × n_icl=8` cell because the VM had no Hugging Face authentication for gated Gemma 3 checkpoints (`401 gated repo` on `google/gemma-3-1b-it`).
- Verified that a secure local HF login already exists on this machine, copied that existing token file to the VM's standard Hugging Face token path, and confirmed remote access to `google/gemma-3-1b-it` via `huggingface_hub` without using the token that was exposed in chat.
- Third smoke launch completed all 8 remote cells successfully on the VM, so the scientific benchmark itself now runs end-to-end for the `1B/4B × Hindi/Telugu × n_icl {8,64}` panel.
- That run then crashed only at the local scoring stage because the harness passed `--out .../neutral_filler_recency_controls.json` to a script that interprets `--out` as an output directory, creating a nested `.../neutral_filler_recency_controls.json/neutral_filler_recency_controls.json` path.
- Patched both the Loop 2 harness and the scorer to handle the directory-style output contract, then rescored the already-downloaded smoke artifacts locally without rerunning the VM benchmark.
- Loop 2 smoke baseline result (8 items per cell):
  - `helpful_control_exact_margin_mean = -0.0469`
  - `helpful_control_cer_margin_mean = 0.0363`
  - `helpful_minus_zs_exact_mean = -0.0313`
  - `helpful_minus_zs_cer_mean = 0.5182`
  - `positive_helpful_control_tasks = 0 / 8`
  - `positive_helpful_vs_zs_tasks = 1 / 8`
- Current interpretation status:
  - established from smoke: the Loop 2 control benchmark runs end-to-end on the VM
  - supported but provisional: under this small smoke sample, helpful examples do **not** beat the best matched controls on exact match
  - supported but provisional: helpful prompts still improve CER on average relative to zero-shot, so the effect may be partially non-exact or partially explained by generic task framing / non-specific control prompts
- Next step in progress: run the full Loop 2 control baseline (`loop2_full`) now that the harness path bug is fixed, because the smoke sample is too small to settle the helpful-vs-control question honestly.
