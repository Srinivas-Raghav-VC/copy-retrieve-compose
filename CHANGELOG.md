# CHANGELOG

## 2026-03-27
- Initialized Loop 1 autoresearch for the honors transliteration project.
- Fixed the plan around a bounded cross-scale behavioral anchor rather than open-ended score chasing.
- Set the initial executable benchmark to the multiscale `premise_gate` suite across `270M`, `1B`, and `4B` on Hindi/Telugu.
- Added session files (`autoresearch.md`, `autoresearch.sh`) and a scorer (`experiments/score_cross_scale_anchor.py`).
- Began repairing the multiscale Modal suite so a smoke baseline can run before deeper iterations.
- First smoke launch failed immediately because `modal run path/to/modal_app.py` executed the file as a script and the suite used package-relative imports.
- Patched `modal_app.py` to support direct path execution with absolute-import fallback.
- Next step: rerun the smoke baseline and see whether the current suite wiring and Modal environment are actually operational.
