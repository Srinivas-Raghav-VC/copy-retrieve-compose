# spec_journal.md

## Project Header
- Project: Mechanistic Interpretability of ICL Transliteration Rescue and Degradation in Gemma 3
- Spec version: copied from `.sisyphus/drafts/spec.md` on 2026-03-25
- Build phase: Phase 0A
- Model(s): Gemma 3 1B IT
- Language(s): Hindi, Telugu

---

## Phase Entry

### Objective
Run only the Phase 0A de-risking packet: validate frozen Hindi/Telugu snapshots, port deterministic + judge evaluation stack, run a behavioral sweep for N={0,4,8,16,32,48,64,96,128,192,256} on unique Aksharantar snapshots across prompt templates and ICL variants, execute rescue/degradation/visibility/judge/transcoder checks, and stop with go/no-go.

### Inputs
- Prompt families: `canonical`, `output_only`, `task_tagged`
- ICL variants: `helpful`, `random`, `shuffled_targets`, `corrupted_targets`
- N values: 0, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256
- Language codes are now parameterized (default `hin,tel`; can include `ben,tam,mar` if snapshots exist)
- Data split IDs: validation against frozen snapshots in `Draft_Results/paper2_fidelity_calibrated/split_snapshots/` plus run snapshots in `research/results/phase0/snapshots/`
- Default multi-seed manifest: `research/config/phase0a_run_config.json` (paper-level source of truth)
- Manifest uses `cross_seed_disjoint=true` so seed pools/evals are non-overlapping across seeds.
- Judge default in manifest: disabled (`judge.enabled=false`) to avoid free-tier rate-limit artifacts; deterministic stack remains primary
- Modal scripts:
  - single-seed packet: `research/modules/modal/modal_phase0_packet.py`
  - manifest runner: `research/modules/modal/run_phase0a_from_config.py`
- Evaluation stack version: `research/modules/eval/*` (Phase 0A port + aggregation/stat tests)

---

Pending execution results will be appended after the run completes.
