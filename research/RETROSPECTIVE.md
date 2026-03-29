# Retrospective Log

Use this file after meaningful sessions.

Each entry should include:
- date / session identifier
- task attempted
- what worked
- what failed
- what slowed the agent down
- what confused the evaluation or implementation
- what should change in prompts
- what should change in repo structure
- what should become a skill
- what should become an example

## Context-engineering retrospective

For each major session, record:
- Which global skills were actually useful
- Which skills added noise or complexity
- Where context got bloated
- What should have been offloaded to files earlier
- Whether subagents helped or just fragmented attention
- Whether a new local project skill should be created

The purpose is compound improvement.
Do not let sessions disappear into chat history.

## 2026-03-25 (Phase 0A data-source hardening)
- Worked: bypassing `datasets` config instability by ingesting Aksharantar language zips directly (`hin.zip`, `tel.zip`) and creating deterministic unique snapshots for high-N ICL.
- Worked: extending snapshot validator to support both legacy and new phase0a snapshot schemas.
- Failed/blocked: Modal runs still interrupted when local client session is unstable; detached workflow must be used consistently.
- Slowdown: uncertainty from mixed schema inputs caused repeated debugging churn; explicit offline snapshot build script reduced this.
- Reusable pattern candidate: "zip-first HF dataset ingestion for unstable dataset scripts" as a repo skill/example.
- Follow-up hardening: switched canonical prompt separator from unicode arrow to ASCII `->` to avoid tokenizer/Unicode confounds in visibility accounting.
- Added N=256 support in both snapshot presets and Phase 0A sweep so degradation tests can extend to longer ICL contexts.
- Added condition-factorized experimentation support (prompt templates + ICL variants) to reduce reviewer concerns about prompt-specific artifacts.
- Added seed-aware output filenames to prevent accidental overwrites when running multi-seed sweeps.
- Improved snapshot builder performance by preloading Aksharantar rows once per language and reusing them across seeds, enabling practical multi-seed snapshot generation.
- Removed an avoidable hard-coding bottleneck by parameterizing language codes and adding an explicit `icl_bank` snapshot field to clarify semantics.
- Added a manifest-first orchestration path to reduce hidden environment-driven behavior and make experiment settings auditable in one JSON source.
- Added explicit judge-disable mode to avoid free-tier Gemini rate-limit confounds while keeping deterministic verification fully operational.
- Added cross-seed-disjoint snapshot generation in manifest runner to eliminate seed-overlap criticism and strengthen statistical independence assumptions.
- Added strict output extraction + newline stop criteria after noticing risk of boilerplate outputs; this improves metric validity when the model emits extra text.
- Added batch generation support to better utilize A100-40GB and reduce run-time overhead for large condition matrices.

## 2026-03-27 (Bounded problem-discovery review)
- Worked: comparing the current clean `research/` behavioral stack against older `Draft_Results/` artifacts and the external `gemma-rescue-study` codebase made it much easier to separate real current evidence from legacy narrative momentum.
- Worked: the local evaluation stack is in noticeably better shape than the mechanistic stack; running the current tests gave a clean anchor before ranking paper stories.
- Worked: checking the shared VM early reduced hand-waving about compute; the machine is reachable and the old repo is present, but the current repo is not synced there and the default Python env lacks `torch`.
- Failed/blocked: web-search providers were unavailable, so current-source gathering had to fall back to alphaXiv, GitHub direct access, and repo/VM inspection.
- Slowdown: a large amount of prior writing already assumes a stronger mechanistic story than the fresh behavioral evidence supports; this required explicit claim downgrading instead of incremental polishing.
- What should change in prompts: future planning prompts should ask for a ranked problem-statement slate before any mechanism-heavy execution, and should explicitly force a "what would make this story lose?" section.
- What should change in repo structure: keep the clean `research/` stack as the canonical behavioral spine and treat old `Draft_Results/` and external codebases as priors / migration sources, not equal-status claim stores.
- Reusable lesson candidate: "old artifact trees are priors, not truth" should become a local reusable checklist item whenever a repo contains both legacy results and a rebuilt clean pipeline.
- New planning lesson: when the user's real goal is honors-level understanding rather than workshop-style paper positioning, a more ambitious mechanistic question can be the right north star as long as the repo still enforces a small behavioral anchor before interpretability claims.

## 2026-03-29 (Visibility audit and 1B threshold follow-up)
- Worked: a cheap token-visibility audit sharply separated `4B` from `1B` at `icl64`; the result was simple enough to interpret directly from prompt-token accounting rather than from noisy behavioral proxies.
- Worked: the bounded `1B` threshold follow-up (`n_icl=48,56,64`) was the right falsification test for the simple visibility hypothesis.
- Main finding: reducing `n_icl` from `64` to `56` or `48` did **not** restore `1B` exact-match wins. So visibility is part of the story, but not the whole story.
- Stronger nuance: `1B × Telugu` actually improves on CER and script metrics as `n_icl` increases, despite worse visibility. That argues against a naive "examples fell out of window, therefore performance must drop" explanation.
- Stronger nuance: `1B × Hindi` stays behaviorally weak even when prompt length is reduced, and one mid-shot cell (`n_icl=48`) is pathologically bad on CER; this points to an additional computation-level instability, not just windowing.
- Slowdown: raw artifact directories now contain results from multiple adjacent runs, which makes quick inspection harder and increases the risk of reading the wrong cell unless the scorer output is treated as canonical.
- Repo-structure improvement: create a small run-manifest sidecar in each autoresearch result directory listing the intended `(model, pair, n_icl, seed)` grid for that run, so mixed raw folders are easier to audit.
- Reusable lesson candidate: when a mechanistic story suggests a simple architectural bottleneck, the next experiment should be a tiny threshold/falsification sweep before any heavier intervention work.
