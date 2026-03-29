# CHANGELOG

## 2026-03-27
- Initialized Loop 1 autoresearch for the honors transliteration project.
- Fixed the plan around a bounded cross-scale behavioral anchor rather than open-ended score chasing.
- Set the initial executable benchmark to the multiscale `premise_gate` suite across `270M`, `1B`, and `4B` on Hindi/Telugu.
- Added session files (`autoresearch.md`, `autoresearch.sh`) and a scorer (`experiments/score_cross_scale_anchor.py`).
- Began repairing the multiscale Modal suite so a smoke baseline can run before deeper iterations.
- First smoke launch failed immediately because `modal run path/to/modal_app.py` executed the file as a script and the suite used package-relative imports.
- Patched `modal_app.py` to support direct path execution with absolute-import fallback.
- Second smoke launch got through image build, but `modal run` aborted because `autoresearch.sh` was teeing build logs into a file inside the workspace, and Modal detected that the workspace changed during build.
- Patched `autoresearch.sh` to write the live Modal log to `/tmp` during build and move it into the workspace only after the command completes.
- Third smoke launch passed local preflight and Modal image build, but the remote container still failed before any benchmark task ran.
- Current blocking error in the remote container: `ModuleNotFoundError: No module named 'Draft_Results'` while importing the multiscale Modal suite entrypoint.
- I terminated the stuck local wrapper process after confirming it was only retrying the same remote import failure, so we do not keep burning time on a non-progressing run.
- Patched `modal_app.py` again so it bootstraps `/workspace` import paths before module imports and resolves the workspace root safely in both local and remote contexts.
- The next smoke run got further: local preflight passed, the Modal app started, and tasks were launched, but the task payloads still carried the local results root (`/mnt/d/...`) instead of the remote volume root.
- Patched the multiscale suite result-root handling so task expansion and plan writing resolve `MULTISCALE_RESULTS_ROOT` at runtime instead of freezing the local path at import time.
- After that fix, the remote tasks finally executed, and the real scientific blocker surfaced: `run_premise_gate.py` failed because the Aksharantar pairs had no usable external records inside the Modal workspace.
- Materialized explicit external JSONL sources plus provenance sidecars for `aksharantar_hin_latin` and `aksharantar_tel_latin` under `Draft_Results/data/transliteration/` so the smoke benchmark has real external data to consume.
- Also fixed `autoresearch.sh` to download the Modal volume from the correct root path (`/`) instead of the mount path inside the container.
- Smoke baseline then completed successfully across all 6 cross-scale premise-gate tasks.
- Baseline smoke result: `premise_gap_exact_mean = 0.0347`, `premise_gap_cer_mean = 0.3036`, with the clearest positive anchor at `4b × Telugu` (`exact_match: 0.000 -> 0.292`, CI excludes zero).
- Negative/flat signals also matter: `270m` is flat on both languages, `1b × Hindi` worsens under `icl64`, and `4b × Hindi` is already strong zero-shot with almost no exact-match gain from ICL.
- Full six-task premise-gate run (`n_eval=200`) completed successfully.
- Full result strengthened the same overall picture: `premise_gap_exact_mean = 0.065`, `premise_gap_cer_mean = 0.2567`, `ci_positive_exact_tasks = 1`, `runbook_gate_positive_tasks = 4`.
- The most robust positive anchor is still `4b × Telugu` (`exact_match: 0.000 -> 0.335`, CER `2.380 -> 0.247`, both CI-supported).
- Additional insight from the full packets: `1b × Hindi` is not uniformly bad at low shot; `icl8` is closer to explicit-ZS, but `icl64` is materially worse on CER and script compliance, suggesting high-N fragility rather than a blanket inability to use examples.
- `270m` remains effectively flat under explicit-ZS, `icl8`, and `icl64`, which supports a capability-floor interpretation for this setup.
- `4b × Hindi` is already strong under explicit-ZS and gets only a moderate boost from ICL (`0.300 -> 0.360` EM), consistent with reduced ICL dependence once the base capability is present.
- Loop 1 therefore achieved its behavioral-goal milestone: we now have a structured cross-scale anchor landscape rather than noise.
- Recommended next step: move to a targeted robustness / control phase centered on `4b × Telugu` (positive anchor) and `1b × Hindi` (high-N fragility anchor), using low-shot vs high-shot and helpful-vs-control comparisons.

## 2026-03-29
- Started Loop 2 on a fresh branch: `autoresearch-loop2-vm-controls`.
- Fixed the Loop 2 objective around a two-language control-verification panel first: `1B/4B × Hindi/Telugu`, with later language expansion and mechanistic probing only after the control story is clean.
- Added a dedicated scorer `experiments/score_loop2_controls.py` for the helpful-vs-control benchmark.
- Extended `autoresearch.sh` with VM-backed Loop 2 modes (`loop2_smoke`, `loop2_full`) that sync the needed code to the shared VM, run the 2×2 control matrix at `n_icl ∈ {8,64}`, pull back artifacts, and score them locally.
- Immediate blocker on the first VM bring-up attempt: the shared VM was unreachable from this environment (`ssh` timed out and port 22 appeared closed), so the Loop 2 baseline has not launched yet.
- Next step: retry VM connectivity, then launch `bash autoresearch.sh loop2_smoke` with the approved VM environment before making any scientific changes.
