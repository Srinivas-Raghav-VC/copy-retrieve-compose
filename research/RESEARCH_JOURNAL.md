# Research Journal

## Established
- Phase 0A frozen Hindi/Telugu snapshots (`Draft_Results/paper2_fidelity_calibrated/split_snapshots/...`) pass schema, disjointness, and script-validity checks.
- Aksharantar direct loading through `datasets` in this environment is unreliable for Hindi/Telugu due to mixed-row schema behavior (`score` field mismatch), producing cast/generation failures.
- Direct language zip ingestion from HF (`hin.zip`, `tel.zip`) is stable and reproducible.

## Supported but provisional
- Unique run snapshots are generated at `research/results/phase0/snapshots/`.
- Seed-11 multilingual verification run completed at `research/results/phase0_clean_modal/verification_v1/` with strict output auditing enabled.
- In that run, low-N rescue is present in Bengali, Hindi, Marathi, and Tamil, but the current degradation criterion is not met in any language at `N=256`; the packet therefore marks `problem_statement_check=false`.
- The same run confirms multilingual output-hygiene failures at baseline and beyond (boilerplate tails, leading fragments, and mixed-script outputs), so `strict_output_check=false` and `mechanistic_readiness_check=false`.
- GPU execution was verified on Modal A100 (`cuda_available=True`, `NVIDIA A100-SXM4-40GB`, `hf_device_map={'': device(type='cuda')}`).
- Current manifest-backed set covers seeds `{11,42,101}` across `{hin,tel,ben,tam,mar}` (`n_candidate=300`, `n_eval=50`) and validates cleanly (15 snapshots).
- Snapshot builder now materializes presets through `N=256` (`[2,4,8,16,32,48,64,96,128,192,256]`) and the Phase 0A runner sweep now includes `N=256` (`[0,4,8,16,32,48,64,96,128,192,256]`).
- Runner now supports prompt-template and ICL-variant condition matrices:
  - prompt templates: `canonical`, `output_only`, `task_tagged`
  - ICL variants: `helpful`, `random`, `shuffled_targets`, `corrupted_targets`
- Snapshot schema now includes `icl_bank` (preferred semantic name) with backward-compatible `candidate_pool` alias.
- Runner language scope is now parameterized (`PHASE0_LANGUAGE_CODES`) instead of being fixed to only Hindi/Telugu.
- Additional unique snapshots for `ben`, `tam`, and `mar` were built for seeds `{11,42,101}` and pass snapshot validation.
- Current manifest snapshots are generated with `cross_seed_disjoint=true`, yielding zero pool/eval overlap across seeds for all configured languages.
- Paper-level run-config manifest support was added via `research/config/phase0a_run_config.json` and loader/orchestrator modules:
  - `research/modules/infra/phase0a_run_config.py`
  - `research/modules/modal/run_phase0a_from_config.py`
- Statistical post-processing scripts are now available:
  - `research/modules/eval/aggregate_phase0_seeds.py`
  - `research/modules/eval/statistical_tests_phase0.py`
- Judge is now configurable and can be fully disabled (default manifest setting) to avoid free-tier rate-limit artifacts while preserving deterministic evaluation.
- Generation/output strictness was tightened for all scripts: newline-aware stop IDs, strict transliteration extraction, and standalone-answer tracking (`standalone_answer_rate`, `hit_max_new_tokens_rate`).
- Runtime row handling is now language-agnostic at the semantic level: code reads canonical `target` text via helper accessors and keeps `hindi` only as a backward-compatible snapshot alias.
- Raw-output auditing is now integrated across scripts/languages, with per-condition rates for `raw_strict_word_only_rate`, `leading_text_rate`, and `trailing_text_rate`, plus example capture for boilerplate-before-word cases.
- Deterministic metric sanity now spans the Phase 0A scripts currently in scope (Devanagari, Telugu, Bengali, Tamil) instead of only Hindi/Telugu examples.
- Phase 0A packets now separate `problem_statement_check` from `mechanistic_readiness_check`, so rescue/degradation evidence can be judged separately from intervention-tool readiness and output-hygiene readiness.
- Detached Modal seed runs with volume persistence now work via `run_phase0a_seed_to_volume`, writing packets/tables/status files into the `phase0a-verification-artifacts` volume.
- The Gemma Scope transcoder smoke now uses a valid asset and passes on detached runs:
  - release: `gemma-scope-2-1b-it-transcoders`
  - sae_id: `transcoder/layer_17_width_262k_l0_medium`
- On detached seeds 42 and 101, `problem_statement_check=true` but `mechanistic_readiness_check=false`; output hygiene remains the blocking gate.
- A100 optimization pass added batched generation (`batch_size` in manifest, default 16), TF32 matmul, and SDPA attention path.
- These new snapshots validate cleanly with the updated validator (mode=`phase0a`) and contain enough unique ICL rows for high-N tests without repetition.
- Bounded discovery review across the current repo, old `gemma-rescue-study` codebase, and the shared VM suggests the strongest current paper direction is a strict-evaluation task-mode / output-hygiene story, with low-shot rescue/high-N instability as the main backup and cross-scale fragility still unvalidated.
- The shared VM is reachable and the old repo already exists there, but the current `Honors` repo is not yet synced and the default Python environment lacks `torch`; Modal remains the cleaner behavioral execution path for the next narrowing pass.
- Project framing is now being reconsidered toward an honors-first, understanding-first cross-scale study: use the current behavioral stack as a sanity anchor, then compare candidate transliteration mechanisms and candidate circuits across Gemma 3 `270M`, `1B`, and `4B`.
- Token-visibility audit (`research/results/autoresearch/token_visibility_v1/`) shows that `icl8` is fully visible for both `1B` and `4B`, while at `icl64` the `4B` prompts still keep all 64 examples fully visible but `1B` loses part of the ICL bank from the local window (about `9` full examples lost on Hindi and about `24` on Telugu at the target-decoding locus).
- A bounded `1B` threshold test at `n_icl ∈ {48,56,64}` on Hindi/Telugu (`research/results/autoresearch/loop2_vm_controls/threshold_1b_seed42/`) does **not** restore exact-match wins by shrinking the prompt. All six threshold cells remain non-positive on `helpful_control_exact_margin`, and none beat zero-shot on exact match.
- Within that threshold test, Telugu soft metrics improve as `n_icl` increases from `48 -> 56 -> 64` even though visibility should worsen, suggesting that partial truncation is not the dominant explanation for the `1B` failure regime.
- Within that same threshold test, Hindi remains weak even when the prompt is shortened, and the `n_icl=48` cell is pathologically bad on CER, reinforcing the suspicion that `1B × Hindi` has an additional instability beyond pure window visibility.
- Re-analysis of the existing raw Loop 2 control artifacts (`outputs/loop2_failure_modes_2026-03-29.md`) shows that the `1B` failure regime is not unitary across languages:
  - `1B × Hindi × n_icl=64` already fails at early target selection (helpful ICL reduces first-token probability and first-entry correctness versus zero-shot, with many Latin/source-like outputs).
  - `1B × Telugu × n_icl=64` succeeds at early target selection but fails at later query-specific continuation (helpful ICL makes the first token highly likely and usually script-correct, yet exact match stays at zero because the model often outputs a prompt-bank target string).
- `4B × Telugu × n_icl=64` remains the clean positive anchor under the same decomposition: it improves both early target selection and whole-word continuation, and its residual mistakes are mostly near-misses rather than prompt-bank copies.
- First-token competition audit (`research/results/autoresearch/first_token_competition_v1/results/summary.json`) strengthens the stage split:
  - `1B × Hindi × n_icl=64` has a genuine early first-token routing failure under high-shot ICL, but that failure is not strongly content-specific because `icl_corrupt` is similarly bad.
  - `1B × Telugu × n_icl=64` largely fixes the first-token stage under both helpful and corrupt ICL, so its main failure is later than the first token.
  - `4B × Telugu × n_icl=64` is almost saturated at the first-token stage even under corrupt ICL, which localizes the helpful-vs-control advantage to later continuation / composition.
- Prompt-bank copy rank analysis (`outputs/loop2_bank_copy_rank_2026-03-29.json`) shows that `1B × Telugu` copied targets are usually not arbitrary prompt-bank items: they come disproportionately from the query's nearest-neighbor similarity neighborhood, and the copy rate increases monotonically with `n_icl`.
- A broader thesis/paper strategy review (`outputs/thesis_strategy_grander_goal_2026-03-29.md`) suggests the strongest honest framing is not a narrow transliteration score story but a study of **ICL algorithmic regimes** — copying, nearest-neighbor retrieval, and composition — using multilingual transliteration as a model organism.
- The next thesis-scale behavioral requirement is now a **4-language phase map** rather than another two-language anecdote; the active confirmatory panel is `Hindi`, `Telugu`, `Bengali`, and `Tamil`, with `Marathi` held as a reserve same-script control.
- `research/spec.md` has now been rewritten to match that broader thesis identity: transliteration as a model organism for ICL regimes (copying, prompt-bank retrieval, nearest-neighbor retrieval, composition) with bounded mechanistic case studies rather than a one-model rescue/degradation story.
- Condition-wise Telugu retrieval analysis (`outputs/loop2_telugu_retrieval_conditions_2026-03-29.json`) further separates two effects:
  - when source-target alignment is preserved, nearest-neighbor structure matters strongly (`helpful_similarity_desc` keeps copy rate high and concentrates copies on top-ranked neighbors; `helpful_similarity_asc` sharply reduces copying but also hurts quality);
  - when alignment is broken (`icl_corrupt`), bank-copying still stays high but is no longer concentrated on true nearest neighbors.

## Open questions
- Does the phenomenon genuinely show strong high-N degradation under the stricter multilingual Phase 0A setup, or does the current `>=0.10` degradation threshold need recalibration before declaring the problem statement verified?
- Is the current negative result for degradation seed-specific, or does it persist across the planned seeds `{11,42,101}`?
- Which output-hygiene intervention best removes boilerplate/mixed-script contamination without changing the underlying rescue/degradation curve?
- Which transcoder IDs/releases are the correct mechanistic assets for Gemma 3 1B in this repo, given the current smoke-load ID mismatch?
- Manual disagreement audit for deterministic-vs-judge conflicts is not yet executed (especially needed if judge is later re-enabled).
- Does the task-mode / generic-continuation story survive prompt-template and decoding cleanup, or is it mainly a prompt-format artifact?
- If output hygiene is tightened, does the low-shot rescue / high-N instability curve remain across seeds and languages?
- Under one frozen strict protocol, does `270M -> 1B -> 4B` actually yield a meaningful fragile-middle-regime story, or does scale add less explanatory power than language/script and formatting factors?
- Are the new `1B` threshold findings (`48/56/64`) seed-robust, or is the apparent mid-shot Hindi instability partly driven by seed-specific word selection?
- Why does `1B × Telugu` improve substantially on CER / script compliance as `n_icl` grows even while more of the bank falls outside the local window, yet still fail to convert those gains into exact-match wins?
- For `1B × Hindi`, where in the computation does the visible ICL signal fail: target-token selection, whole-word continuation, late-layer normalization, or something else?
- For `1B × Hindi`, what is the dominant wrong first-token competitor under helpful ICL: a Latin/source-copy token, a script-valid but wrong Devanagari token, or an in-context-bank retrieval token?
- For `1B × Telugu`, beyond the already-supported nearest-neighbor tendency, what specific retrieval heuristic dominates the copied-bank behavior: source-form similarity, prompt position/recency, target-subtoken overlap, or some interaction among them?
- For `1B × Telugu`, how much of the remaining failure is explained by a generic long-prompt target-copy prior versus a query-conditioned aligned-neighbor retrieval mechanism?
- For `1B × Telugu`, why do corrupt high-shot prompts still largely fix the first token while failing on the whole word — is the early signal just generic target-script/task-mode activation, with query-specific selection collapsing later?

## Retired claims
- Retired: "N up to 96 can be evaluated from frozen 16-example snapshots by deterministic repetition" as research-valid evidence. This is now considered debug-only and not acceptable for scientific claims.
- Retired for now: a mechanism-first paper framing is the main story before behavioral cleanup. Mechanistic work remains downstream of stricter output hygiene and a cleaner behavioral contrast.

## Priority next experiments
1. Run the thesis-scale 4-language helpful-vs-control panel on the shared VM: `1B/4B × {Hindi, Telugu, Bengali, Tamil} × n_icl {8,64} × seeds {42,11,101}` and aggregate the scores into a paper-grade phase map.
2. Build the calibrated local verifier stack with a human-audited calibration set so variant-equivalence and failure-taxonomy labels can augment exact-match/CER without replacing them.
3. On the now-stable anchors, perform mechanistic localization:
   - `1B Hindi` early-routing / first-token competition localizer,
   - `1B Telugu` later retrieval/composition localizer on copied-bank cases,
   - `4B Telugu` positive-control comparison.
4. Only after localization survives checks, run heavier causal tests (patching / ablation / recency-position controls) on selected components.
5. Bring `270M` back into the final synthesis as the capability-floor comparison even if the rich helpful-vs-control matrix stays focused on `1B/4B`.

## Failure cases worth inspecting
- Modal local-client disconnect interruptions (`modal run` without stable detach workflow).
- `modal run --detach` on the local entrypoint can keep the remote GPU job alive but still drop local packet-writing/postprocess steps; use the manifest orchestrator directly for durable local artifacts.
- Transcoder smoke currently fails because the requested SAE ID is invalid for the chosen release (`layer_12_width_262k_l0_medium` missing from `gemma-scope-2-1b-it-transcoders-all`).
- Mixed-script outputs remain common well beyond `N=0` for several languages, especially Hindi/Bengali/Marathi/Tamil.
- Any Unicode delimiter/tokenization sensitivity in prompts (now switched to explicit ASCII `->` separator for consistency).
