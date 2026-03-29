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
- Full Loop 2 baseline (`loop2_full`, 30 items per cell) completed successfully.
- Full baseline result:
  - `helpful_control_exact_margin_mean = 0.0417`
  - `helpful_control_cer_margin_mean = 0.0721`
  - `helpful_minus_zs_exact_mean = 0.0500`
  - `helpful_minus_zs_cer_mean = 0.4939`
  - `positive_helpful_control_tasks = 4 / 8`
  - `positive_helpful_vs_zs_tasks = 3 / 8`
- Important per-cell findings from the full baseline:
  - `4b × Telugu × n_icl=64` is the cleanest positive anchor: `helpful_exact = 0.300`, `zs_exact = 0.000`, best matched control exact = `0.133`, and helpful also wins strongly on CER.
  - `4b × Telugu × n_icl=8` is also positive, but weaker.
  - `4b × Hindi × n_icl=64` is modestly positive against matched controls, but not above explicit zero-shot; this looks like a capable-model / low-dependence regime, not a pure rescue regime.
  - `1b × Hindi × n_icl=64` remains the clearest fragility cell: `helpful_exact = 0.000` versus `zs_exact = 0.067`, and helpful is also worse on CER than both zero-shot and the best matched control.
  - `1b × Telugu` shows CER gains without exact-match gains, especially at `n_icl=64`, so it remains a weak / partial-help cell rather than a clean positive anchor.
- Additional structural findings from the full controls:
  - `helpful_minus_reversed_exact_mean = -0.0125` and `desc_minus_asc_exact_mean ≈ 0`, so order and near-to-far helpful sorting are not showing a strong exact-match advantage overall.
  - This weakens any simplistic recency-order story and suggests the stronger distinction is between meaningful examples versus clearly non-helpful controls, especially in `4b × Telugu`.
- Verification note: while reviewing the scorer, I found that the sign on `one_b_highN_helpful_cer_regret_mean` was backwards. I fixed the formula and rescored smoke + full artifacts locally before interpreting them.
- Current interpretation status after the full baseline:
  - established: the `1B/4B × Hindi/Telugu × n_icl {8,64}` control benchmark now runs cleanly end-to-end on the VM
  - supported but provisional: `4b × Telugu`, especially at `n_icl=64`, is the best positive helpful-vs-control anchor
  - supported but provisional: `1b × Hindi` is a real high-N fragility anchor
  - supported but provisional: the broader helpful effect is not universal; it is concentrated in the stronger 4B regime rather than uniformly present across the whole 2×2 panel
- Next step in progress: run seed-robustness replications on the same full Loop 2 benchmark (starting with additional seeds beyond 42) before deciding whether the control story is stable enough to justify language expansion.
- Seed-robustness replication on seeds `11` and `101` completed successfully, and a 3-seed aggregate now exists at `research/results/autoresearch/loop2_vm_controls/seed_aggregate.json`.
- 3-seed aggregate summary:
  - `helpful_control_exact_margin_mean = 0.0347 ± 0.0098`
  - `helpful_control_cer_margin_mean = 0.0478 ± 0.0304`
  - `helpful_minus_zs_exact_mean = 0.0625 ± 0.0090`
  - `positive_helpful_control_tasks = 4 / 8` on every seed
  - `one_b_highN_helpful_cer_regret_mean = 0.6910 ± 0.3622`
  - `four_b_highN_helpful_cer_gain_mean = 0.0935 ± 0.0685`
- Stable cross-seed conclusions:
  - `4b × Telugu × n_icl=64` is the strongest and cleanest helpful-vs-control anchor across all three seeds.
  - `4b × Telugu × n_icl=8` is also consistently positive, but weaker.
  - `1b × Hindi × n_icl=64` remains the strongest fragility anchor: helpful exact remains near zero and CER is much worse than zero-shot.
  - `1b × Telugu` remains an unstable / partial-help regime with no exact-match gains.
  - `4b × Hindi × n_icl=64` is mildly positive against matched controls but not a rescue-above-zero-shot story; treat it as a strong-base-capability comparison cell, not the main anchor.
- Decision after seed replication: the 2-language verification phase is strong enough to move to bounded language expansion before mechanistic probing.
- Next step in progress: prepare additional Aksharantar breadth languages (`Marathi`, `Bengali`, `Tamil`) under the same external-data / control-benchmark contract, then run a bounded expansion benchmark at the most informative high-shot setting.
- Expansion prep completed:
  - generalized `autoresearch.sh` and `experiments/score_loop2_controls.py` so the same Loop 2 harness can run arbitrary model/pair/n-shot grids
  - added `experiments/build_external_aksharantar_pairs.py`
  - extended `pair_registry.json` with `aksharantar_mar_latin` and `aksharantar_ben_latin`
  - materialized external JSONL + meta sidecars for `Marathi`, `Bengali`, and `Tamil`
  - locally verified that `aksharantar_ben_latin`, `aksharantar_mar_latin`, and `aksharantar_tam_latin` resolve through the ingestion stack with external sources present
- Planned bounded expansion benchmark:
  - models: `1b`, `4b`
  - pairs: `aksharantar_mar_latin`, `aksharantar_ben_latin`, `aksharantar_tam_latin`
  - shot setting: `n_icl = 64` only
  - rationale: this is the most informative high-shot condition given the verified `1B` fragility and strongest `4B` positive anchor at high shot
- Bounded expansion benchmark (`Marathi`, `Bengali`, `Tamil`, seed 42, `n_icl=64`) completed successfully.
- Expansion result summary:
  - `helpful_control_exact_margin_mean = 0.0389`
  - `helpful_control_cer_margin_mean = 0.2923`
  - `helpful_minus_zs_exact_mean = 0.0444`
  - `positive_helpful_control_tasks = 3 / 6`
- Per-cell interpretation:
  - `4b × Marathi`: modest positive (`helpful_control_exact_margin = 0.0333`, positive CER margin)
  - `4b × Bengali`: clear positive (`helpful_control_exact_margin = 0.1000`, positive CER margin)
  - `4b × Tamil`: clear positive (`helpful_control_exact_margin = 0.1000`, very large improvement over zero-shot on CER)
  - `1b × Marathi`: no exact-match gain but positive CER margin versus controls
  - `1b × Bengali`: no exact-match gain and worse than zero-shot on CER despite beating weak controls
  - `1b × Tamil`: no exact-match gain and worse than zero-shot on CER, though still better than the best matched control on CER
- Current expansion-phase conclusion:
  - supported but provisional: the positive helpful-vs-control story generalizes across multiple additional languages for `4B`
  - supported but provisional: the weak / fragile `1B` regime also generalizes beyond Hindi/Telugu and is not just a two-language artifact
- Decision after bounded expansion:
  - mechanistic probing should stay focused on the cleanest validated anchor (`4b × Telugu × n_icl=64`) and the clearest fragility comparison (`1b × Hindi × n_icl=64`), with `4b × Hindi` as an optional strong-base-capability comparison cell.
- Next step in progress: run a cheap mechanistic screening pass (representation-level, not full causal claims yet) on the validated anchor cells before any heavier intervention pipeline.
- Mechanistic screening plan chosen:
  - tool: `run_script_space_map.py`
  - tasks: `4b:aksharantar_tel_latin`, `1b:aksharantar_hin_latin`, `4b:aksharantar_hin_latin`
  - purpose: compare a clean positive anchor (`4B Telugu`), a clear fragility anchor (`1B Hindi`), and a strong-base-capability comparison cell (`4B Hindi`) before any heavier causal intervention work
  - budget: `max_words=30`, `seed=42`, `n_icl=64`
- A dedicated VM launcher was added at `experiments/run_vm_script_space_screen.sh`, and the first screening run completed successfully.
- Mechanistic screening observations (representation-level only; not a causal claim):
  - `4b × Telugu`: the biggest ICL-vs-explicit-ZS change is a sharp late-layer Telugu-script ramp centered on layers `29–33`, with the strongest gains at local layers `30–33` after the final global layer `29`. Example: layer `33` script-mass rises from `0.031` to `1.000`, and the gold-token rank proxy improves from `250.2` to `1.0`.
  - `4b × Hindi`: explicit-ZS already has substantial late-layer Devanagari mass, and ICL mainly amplifies the same late block (`29–32`) rather than creating a totally new pattern. This matches the strong-base-capability interpretation from behavior.
  - `1b × Hindi`: ICL increases Hindi-script mass through a mid/late block around layers `17–24`, but the final layer `25` looks unstable rather than cleanly improved: script mass falls from `0.914` under explicit-ZS to `0.040` under `icl64`, even though some intermediate rank proxies improve. Treat this as a candidate late-stage collapse to test, not as a settled mechanism.
- Decision after script-space screening:
  - candidate positive layer range for `4b × Telugu`: `29–33`
  - candidate comparison range for `4b × Hindi`: `29–32` (with layer `33` saturated in both conditions)
  - candidate fragility range for `1b × Hindi`: `17–24`, with special caution on final layer `25`
- Next step in progress: run a cheap token-visibility audit on the same family of cells to test whether the high-N `1B` fragility is plausibly related to architectural visibility limits before moving to heavier causal interventions.
- Token-visibility audit completed successfully for `1B/4B × Hindi/Telugu × {explicit_zs, icl8, icl64}`.
- Visibility findings:
  - `icl8` is fully visible for both `1B` and `4B` on both languages.
  - `4B` has full visibility even at `icl64`: all 64 examples remain fully visible for both Hindi and Telugu at both the source-query and target-pos1 loci.
  - `1B × Hindi × icl64` exceeds the local window: about `82.6` ICL tokens fall outside the target-pos1 local window on average, with roughly `54` fully visible examples, `1` partial example, and `9` fully invisible examples.
  - `1B × Telugu × icl64` is much more severely truncated: about `305.6` ICL tokens fall outside the target-pos1 local window on average, with roughly `39` fully visible examples, `0.9` partial examples, and `24.1` fully invisible examples.
- Interpretation after visibility audit:
  - supported but provisional: architectural visibility is a real part of the `1B` vs `4B` difference at high shot, especially for Telugu.
  - supported but provisional: visibility alone does **not** fully explain `1B × Hindi` fragility, because Hindi still has ~54 visible examples under `icl64` yet remains behaviorally poor and shows a late-layer collapse in the script-space screen.
  - established from this check: the clean `4B` high-shot regime is not bottlenecked by local-window truncation on these prompts.
- Decision after visibility audit:
  - do **not** jump straight to heavyweight causal patching yet.
  - first run a small behavioral threshold test on `1B` (`Hindi`, `Telugu`) across intermediate `n_icl` values to test whether fitting more of the ICL bank inside the local window actually recovers behavior.
- Next step in progress: run a bounded `1B` visibility-threshold benchmark on `Hindi/Telugu` with intermediate `n_icl` settings (`48`, `56`, `64`) under the same helpful-vs-control setup.
- The bounded `1B` visibility-threshold benchmark completed successfully at `research/results/autoresearch/loop2_vm_controls/threshold_1b_seed42/`.
- Threshold benchmark summary:
  - `helpful_control_exact_margin_mean = -0.0167`
  - `helpful_control_cer_margin_mean = -0.2386`
  - `helpful_minus_zs_exact_mean = -0.0222`
  - `helpful_minus_zs_cer_mean = -0.8150`
  - `positive_helpful_control_tasks = 0 / 6`
  - `positive_helpful_vs_zs_tasks = 0 / 6`
- Per-language interpretation from the threshold test:
  - `1b × Hindi`: lowering `n_icl` from `64` to `56` or `48` does **not** recover exact-match wins; all three cells stay at or below zero-shot and at or below the best matched control on exact match. The `n_icl=48` cell is especially pathological on CER.
  - `1b × Telugu`: exact match stays at `0.0` for `n_icl=48`, `56`, and `64`, so reducing prompt length does not unlock exact-match recovery. However, CER and script metrics improve monotonically as `n_icl` increases, despite the visibility audit showing *more* truncation at larger `n_icl`.
- Interpretation after the threshold test:
  - supported but provisional: local-window visibility is **not** the dominant explanation for the `1B` failure regime.
  - supported but provisional: visibility contributes to the `1B` vs `4B` contrast, but the main remaining question is computational: how the visible examples are (or are not) turned into the correct target continuation.
  - supported but provisional: `1B × Hindi` remains the stronger candidate for a downstream instability / late-stage failure story than for a pure context-window story.
- Decision after the threshold test:
  - retire the simple "just too many examples fell out of the window" explanation as the primary story.
  - keep visibility as a contributing architectural factor, especially for `1B × Telugu`.
  - next bounded mechanistic step should focus on **where** the `1B` computation fails given visible evidence, not on squeezing prompt length further.
- Follow-up analysis completed from existing Loop 2 raw control artifacts, written to `outputs/loop2_failure_modes_2026-03-29.md` and `outputs/loop2_failure_modes_2026-03-29.json`.
- Decomposition result from that follow-up:
  - `1B × Hindi × n_icl=64` is already failing at the **first target token**: helpful ICL lowers both target first-token probability and first-entry correctness relative to zero-shot, and many outputs are Latin/source-like (`50%` Latin-script, `40%` exact source copies).
  - `1B × Telugu × n_icl=64` is **not** mainly a first-token failure: helpful ICL drives first-token probability from near-zero to `0.869` and first-entry correctness to `0.900`, but exact match stays at `0.000`.
  - `1B × Telugu × n_icl=64` often emits a prompt-bank target string instead of the query-specific answer: `80%` of helpful predictions are exact copies of one of the 64 ICL-bank targets, and `88.9%` of the `first-correct but exact-wrong` cases are bank copies.
  - `4B × Telugu × n_icl=64` remains the clean positive anchor because it gets both early target selection and later whole-word continuation mostly right; its remaining errors are mostly near-misses rather than bank copies.
- Interpretation after the failure-mode follow-up:
  - established: the `1B` failure story is **not unitary** across languages.
  - established: `1B Hindi` looks like an early routing / target-selection problem.
  - established: `1B Telugu` looks like a later retrieval-composition / continuation problem.
- Recommended next bounded experiments now differ by language:
  - `1B Hindi`: small first-token competition audit to inspect the top wrong competitor token and whether it is Latin/source-like or a wrong Devanagari bank token.
  - `1B Telugu`: prompt-bank copy audit with nearest-neighbor controls to test whether the model is retrieving a similar in-context target rather than composing the correct query-specific continuation.
- Added new bounded audit code for the first-token step:
  - `experiments/first_token_competition_audit.py`
  - `experiments/run_vm_first_token_competition_audit.sh`
- Current run in progress:
  - process: `proc_11`
  - name: `first-token-competition-v1`
  - tasks: `1b:aksharantar_hin_latin:64`, `1b:aksharantar_tel_latin:64`, `4b:aksharantar_tel_latin:64`
  - budget: `max_items=30`, `seed=42`
- Oracle for this audit:
  - for each task and each condition (`zs`, `icl_helpful`, `icl_corrupt`), record the correct first target token probability, the top-1 predicted first token, its script bucket, and the target-vs-best-competitor logit gap.
  - minimal decision rule: if `1B Hindi` helpful prompts often lose the first target token to Latin/source-like top-1 competitors, the early-routing hypothesis gains support; if `1B Telugu` already wins the first token cleanly, that reinforces the later continuation / retrieval-composition hypothesis.
- The first-token competition audit completed successfully at `research/results/autoresearch/first_token_competition_v1/results/summary.json`.
- Audit result:
  - `1B × Hindi × n_icl=64`: the early-routing hypothesis is supported. Under `icl_helpful`, top-1 target rate falls to `0.467` (vs `0.600` in zero-shot), and the top-1 token is Latin-script in `50%` of items. However, `icl_corrupt` is similarly bad (`0.433`), so the early failure is not strongly content-specific.
  - `1B × Telugu × n_icl=64`: the first-token stage is largely fixed under high-shot ICL. `icl_helpful` reaches top-1 target rate `0.900` and mean target-token probability `0.869`; even `icl_corrupt` reaches `0.767` / `0.791`.
  - `4B × Telugu × n_icl=64`: the first-token stage is nearly saturated for both `icl_helpful` and `icl_corrupt` (`0.967` top-1 target rate each), so the helpful-vs-control difference that remains must be downstream of the first token.
- Additional bounded follow-up completed locally from existing raw control artifacts:
  - `experiments/analyze_prompt_bank_copy_ranks.py`
  - output: `outputs/loop2_bank_copy_rank_2026-03-29.json`
- Prompt-bank copy rank result:
  - `1B × Telugu × n_icl=64` copies an exact prompt-bank target on `24/30` items (`80%`). Those copied targets are usually drawn from the similarity neighborhood of the query: median source-similarity rank `2.5`, `62.5%` from the top 5, `75%` from the top 10.
  - As `n_icl` increases from `48 -> 56 -> 64`, the `1B × Telugu` bank-copy rate rises from `53.3% -> 70.0% -> 80.0%`.
  - `4B × Telugu × n_icl=64` has only `2/30` exact bank copies.
- Interpretation after these follow-ups:
  - established: the `1B` failure story splits by language *and by stage*.
  - established: `1B Hindi` is best framed as an early routing problem that is not strongly content-specific under high-shot ICL.
  - established: `1B Telugu` is best framed as a nearest-neighbor-style retrieval/composition problem rather than a first-token failure.
- Recommended next bounded experiment:
  - do **not** rerun more generic high-shot sweeps.
  - instead run a narrow Telugu prompt-bank retrieval audit that explicitly links each copied output to the matched in-context example identity / similarity rank and compares helpful vs corrupt vs reversed ordering on the same fixed items.
- That Telugu retrieval audit was then completed **without another model run** by analyzing the existing `1B × Telugu × n_icl=64` control artifact.
- Added analysis script:
  - `experiments/analyze_telugu_retrieval_conditions.py`
- Output:
  - `outputs/loop2_telugu_retrieval_conditions_2026-03-29.json`
- Retrieval-audit result:
  - `icl_helpful`: bank-copy rate `80.0%`, median copied-target similarity rank `2.5`.
  - `icl_helpful_similarity_desc`: bank-copy rate `73.3%`, with copied targets even more concentrated on nearest neighbors (median rank `1.0`).
  - `icl_helpful_similarity_asc`: bank-copy rate drops sharply to `23.3%`, but first-token quality and CER both worsen.
  - `icl_helpful_reversed`: intermediate bank-copy rate `50.0%`.
  - `icl_corrupt`: bank-copy rate stays very high (`83.3%`), but copied targets are *not* nearest neighbors under the true helpful similarity ranking (median rank `22`, top-5 share `12%`).
- Interpretation after the Telugu retrieval audit:
  - established: there are **two** Telugu high-shot failure pressures in `1B`:
    1. a query-conditioned nearest-neighbor retrieval tendency when source-target alignment is preserved,
    2. a more generic high-shot bank-copy tendency when the prompt is long and target strings are available, even if source-target alignment is broken.
  - supported but provisional: the helpful-condition failure seems to be the superposition of these two effects, with nearest-neighbor retrieval being the more query-specific one.
- Decision after this audit:
  - the next mechanistic step should stay **Telugu-specific** and test why aligned similar examples trigger wrong-bank continuation in `1B`, while `4B` escapes that trap.
  - do not spend more budget on broad behavioral sweeps before that targeted follow-up.
- Manual spot-check audit completed against the raw per-item artifacts and the first-token audit outputs.
- Manual audit verdict:
  - the automated conclusions are directionally correct.
  - `1B Hindi` really does show many early Latin/source-like first-token failures under high-shot ICL, but the cell is not *purely* first-token-limited because some items recover the first token and still miss later.
  - `1B Telugu` really is mostly a later retrieval/composition failure, with many concrete cases where the first token is corrected to `ఆ` yet the whole output becomes a wrong prompt-bank target; a minority of Telugu items still fail early.
  - `4B Telugu` manual spot-checks look like near-miss whole-word errors rather than the `1B` prompt-bank-copy regime.
- Wording adjustment after manual audit:
  - prefer "substantial early routing failure" over "pure first-token failure" for `1B Hindi`.
  - prefer "mostly later retrieval/composition failure" over "entirely later failure" for `1B Telugu`.
- Thesis / paper strategy review completed in `outputs/thesis_strategy_grander_goal_2026-03-29.md`.
- Main strategic decision from that review:
  - do **not** redefine the project around a judge or around a broader but vaguer benchmark.
  - instead upgrade the project into a stronger thesis around **algorithmic regimes of ICL**: copying, nearest-neighbor retrieval, and genuine composition in multilingual transliteration.
- Judgment after literature and strategy review:
  - there is enough evidence to justify **narrow mechanistic work now**, especially on `1B Hindi` and `4B Telugu`.
  - there is **not** enough evidence yet for broad causal claims about all languages or a single unified `1B` mechanism.
- Role of free VRAM in the new plan:
  - worthwhile for a **calibrated local verifier stack** (correctness / acceptability judge, failure-taxonomy judge, human-audited calibration set).
  - not worthwhile as a replacement for deterministic evaluation or as the main source of truth.
- Recommended thesis north star:
  - use multilingual transliteration as a **model organism** for studying when ICL behaves like copying, retrieval, or composition, and how those regimes change with scale.
- Recommended next practical step:
  - build the verifier stack and audited error taxonomy first,
  - then consolidate the current behavioral findings into a phase-diagram-style thesis framing before doing the next targeted mechanistic experiment.

## Loop 3 — thesis-scale expansion program

_Date approved: 2026-03-29_

### New user instruction

The project is now explicitly authorized to move beyond the bounded discovery posture and into a deeper, end-to-end thesis program:

- do the broader autoresearch loop,
- do not stay artificially small if the next uncertainty requires more breadth,
- use papers, search, and implementation evidence to stay grounded,
- keep moving through code bugs while re-verifying after fixes,
- aim for a full understanding story rather than a mediocre narrow benchmark.

### Thesis-scale north star

Use multilingual transliteration as a **model organism for ICL algorithms** and map when the models behave like:

1. source copying / wrong-script fallback,
2. generic prompt-bank copying,
3. nearest-neighbor retrieval,
4. genuine query-specific composition.

### Working thesis panel

Promote a 4-language confirmatory panel for the next program stage:

- `aksharantar_hin_latin`
- `aksharantar_tel_latin`
- `aksharantar_ben_latin`
- `aksharantar_tam_latin`

Reason for this panel:
- `Hindi` and `Telugu` are the strongest already-opened anchors,
- `Bengali` and `Tamil` broaden script/family coverage and were cleaner than `Marathi` in the first bounded expansion,
- `Marathi` stays useful as a reserve same-script control rather than being dropped conceptually.

### Program phases

#### Phase A — broad behavioral map
- Run the 4-language `1B/4B × n_icl {8,64}` helpful-vs-control panel across seeds `{42,11,101}`.
- Keep the current deterministic metrics as the hard anchor.
- Use this phase to decide which regimes generalize cleanly across more than the original Hindi/Telugu core.

#### Phase B — verifier hardening
- Build a calibrated local verifier stack for:
  - transliteration acceptability / variant equivalence,
  - failure taxonomy,
  - disagreement analysis versus deterministic metrics.
- Human-audit a calibration subset before trusting judge outputs.

#### Phase C — mechanistic localization
- `1B Hindi`: first-token / early-routing localization.
- `1B Telugu`: later retrieval/composition localization on copied-bank cases.
- `4B Telugu`: positive-control comparison.
- Only after localization survives checks should heavier causal interventions expand.

#### Phase D — cross-scale synthesis
- Bring `270M` back in as the floor / capability-boundary comparison in the final thesis synthesis, even if the rich helpful-vs-control panel remains `1B/4B` only.
- End goal: a phase-diagram-style story about copying vs retrieval vs composition across scale and language.

### Active executable campaign

The first thesis-scale campaign to launch is:

- script: `experiments/run_vm_four_lang_thesis_panel.sh`
- scope: `1B/4B × {Hindi, Telugu, Bengali, Tamil} × n_icl {8,64} × seeds {42,11,101}`
- outputs root: `research/results/autoresearch/four_lang_thesis_panel/`

### Manual audit policy

Do not rely only on aggregates or automated labels.

Manual spot-checks are now mandatory at the following checkpoints:

1. after each seed-level four-language panel finishes,
2. before trusting any verifier / judge output as a thesis-facing metric,
3. before escalating a behavioral pattern into a mechanistic claim,
4. after any bug fix that could change prompt construction, scoring, or artifact interpretation.

Prefer small, explicit audit slices with saved notes over vague "looks good" judgments.

### Current decision rule

Do **not** let the new broader mandate collapse into aimless sprawl.

Broaden the work only when the additional breadth reduces real uncertainty. The current broadening is justified because the project now needs a **paper-grade four-language phase map**, not just more localized anecdotes.

### Current incident log

- The first long four-language thesis-panel run (`proc_12`) crashed locally with SSH exit code `255` and stderr `client_loop: send disconnect: Broken pipe`.
- Manual diagnosis showed this was a **transport failure**, not an immediate science failure:
  - remote artifacts persisted on the VM,
  - seed `42` had already completed `14/16` cells,
  - all `1B` cells were present,
  - `4B Hindi`, `4B Telugu`, and `4B Bengali` were present,
  - only `4B Tamil × n_icl {8,64}` remained missing.
- No active research python process remained on the VM after the disconnect, so the run needs an explicit resume rather than passive waiting.
- Recovery plan:
  1. add SSH keepalive options to the VM harness,
  2. resume only the missing `seed42` `4B Tamil` cells into the same remote result root,
  3. rescore the full seed-42 panel locally,
  4. rebuild manual audit packets for the recovered seed,
  5. continue seeds `11` and `101`,
  6. aggregate all three seeds at the end.
- Recovery status update after checking the VM again:
  - `seed42` is now fully complete (`16/16` files on the VM and a full local `score.json`).
  - `seed11` has begun on the VM and has at least `3` raw artifacts written so far.
  - the local recovery wrapper is no longer attached, but a remote transliteration process is still running on the VM (`1b × Telugu × n_icl=64` at the time of inspection).
  - `seed101` has not started yet.
- A dedicated fast-direction / technique memo now exists at `outputs/mechanistic_technique_playbook_and_fast_direction_2026-03-29.md`.
- If time pressure becomes dominant, the thesis should collapse to:
  - 4-language multi-seed phase map,
  - manual audit packets,
  - `1B Hindi` early-routing case study,
  - `1B Telugu` vs `4B Telugu` later-continuation comparison,
  - and a final claim about copying vs retrieval vs composition across scale.
- Deeper mechanistic planning update from further paper/code review:
  - organizing principle: localize **stage** before localizing **site**.
  - primary stage split remains `early routing` vs `later continuation / composition`.
  - broadening languages suggest a provisional 2-axis map for `1B` high-shot behavior:
    - Bengali: catastrophic early collapse,
    - Hindi: substantial early routing failure,
    - Tamil: mixed regime,
    - Telugu: late retrieval/composition collapse.
  - primary mechanistic tools should therefore be:
    1. teacher-forced token competition,
    2. logit-lens trajectories,
    3. activation patching / narrow ablation,
    with transcoders and larger attribution-graph tools treated as secondary instruments rather than the main proof.
