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
- Next step: rerun the smoke baseline and see whether the remote Modal tasks can now write into the remote artifacts volume and complete cleanly.
