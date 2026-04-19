# Research Journal

## Established
- Phase 0A frozen Hindi/Telugu snapshots (`Draft_Results/paper2_fidelity_calibrated/split_snapshots/...`) pass schema, disjointness, and script-validity checks.
- Aksharantar direct loading through `datasets` in this environment is unreliable for Hindi/Telugu due to mixed-row schema behavior (`score` field mismatch), producing cast/generation failures.
- Direct language zip ingestion from HF (`hin.zip`, `tel.zip`) is stable and reproducible.

## Supported but provisional
- The current ICML manuscript claim boundary is now calibrated more tightly against stored artifacts: the Telugu single-competitor diagnostic caveat is quantified (`8/24` exact-copy top-1 matches on the 57-item helpful temperature panel; `22/81` on the 200-item helpful prompt-composition panel), the Telugu negative intervention wording is explicitly limited to the tested full-state mean-shift family, and the Hindi first-step causal objective is now named `target-vs-majority-Latin/ASCII` to match the actual `V_{Latin}` implementation.
- The Hindi token-level causal readout is now partially bridged to the paper's akshara-level early-start axis on held-out patch-eval items: in `hindi_1b_practical_patch_eval.json`, the itemwise correlation between patch-induced `ΔL` and itemwise change in first-entry correctness is `0.40`, and rescued baseline failures show larger mean `ΔL` gain than non-rescued failures (`+8.26` vs `+5.65`). This is still not an exact surrogate relation, but it materially strengthens the mechanistic-to-behavioral bridge.
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
- The repaired Hindi mechanistic localizer (`research/results/autoresearch/mech_localizer_v1/1b/aksharantar_hin_latin/nicl64/layerwise_routing_trace.json`) now matches the trusted first-token audit at the final layer on the same audited split:
  - `zs`: mean target prob `0.5776` vs audit `0.5793`, top1 target rate `0.600`
  - `icl_helpful`: `0.4824` vs `0.4813`, top1 target rate `0.467`
  - `icl_corrupt`: `0.4362` vs `0.4312`, top1 target rate `0.433`
- Gemma-3 localizer implementation lesson is now established for this repo: the final hidden state should **not** be passed through the final norm a second time before unembedding; installed `transformers` already returns final-normalized text hidden states.
- Prompt-bank copy rank analysis (`outputs/loop2_bank_copy_rank_2026-03-29.json`) shows that `1B × Telugu` copied targets are usually not arbitrary prompt-bank items: they come disproportionately from the query's nearest-neighbor similarity neighborhood, and the copy rate increases monotonically with `n_icl`.
- A broader thesis/paper strategy review (`outputs/thesis_strategy_grander_goal_2026-03-29.md`) suggests the strongest honest framing is not a narrow transliteration score story but a study of **ICL algorithmic regimes** — copying, nearest-neighbor retrieval, and composition — using multilingual transliteration as a model organism.
- The next thesis-scale behavioral requirement is now a **4-language phase map** rather than another two-language anecdote; the active confirmatory panel is `Hindi`, `Telugu`, `Bengali`, and `Tamil`, with `Marathi` held as a reserve same-script control.
- `research/spec.md` has now been rewritten to match that broader thesis identity: transliteration as a model organism for ICL regimes (copying, prompt-bank retrieval, nearest-neighbor retrieval, composition) with bounded mechanistic case studies rather than a one-model rescue/degradation story.
- The spec now also carries an explicit **reviewer acceptance bar**: the work should not be treated as thesis-acceptable unless it survives four-language multi-seed checks, manual audit, verifier calibration, and at least one bounded mechanistic case study.
- Condition-wise Telugu retrieval analysis (`outputs/loop2_telugu_retrieval_conditions_2026-03-29.json`) further separates two effects:
  - when source-target alignment is preserved, nearest-neighbor structure matters strongly (`helpful_similarity_desc` keeps copy rate high and concentrates copies on top-ranked neighbors; `helpful_similarity_asc` sharply reduces copying but also hurts quality);
  - when alignment is broken (`icl_corrupt`), bank-copying still stays high but is no longer concentrated on true nearest neighbors.
- Supported but provisional mechanistic readout from the repaired Hindi localizer:
  - both helpful and corrupt ICL raise the correct first-target token probability above zero-shot in layers roughly `18–25`,
  - the strongest pre-final band is `23–25`,
  - but the final output step still often flips toward Latin/source-like competitors (`a`, `aa`, `aad`, `ah`).
- First Hindi causal patching update:
  - query-position late-layer patching was the wrong direct test for first-token competition and produced near-zero effects; this is now treated as a methodology lesson, not as a scientific negative result.
  - with **last-token** patching, the cleanest current site is `layer 25 MLP output`.
  - helpful `<-` zs at `L25 MLP` gives `delta_mean_gap_latin = +3.358`, `delta_top1_target_rate = +0.200`, `rescue_rate_on_base_failures = 0.375`, and `harm_rate_on_base_successes = 0.000`.
  - zs `<-` helpful at `L25 MLP` gives `delta_mean_gap_latin = -4.708`, `delta_top1_target_rate = -0.133`.
- Broader last-token `layer_output` replacements at `20/23/24/25` move the regime strongly but less cleanly: they rescue many failures and also damage many already-correct items, so they currently look like coarse late-state moves rather than the cleanest bottleneck.
- Final-state (`final_norm` output) last-token replacement matches the aggregate effect of `layer 25 layer_output` almost exactly, supporting the view that the harmful/helpful distinction is already encoded in the last-token readout state by the end of layer 25.
- Held-out `L25 MLP` raw-coordinate subspace patching update (`hindi_mlp_subspace_patch_v1`):
  - the `L25 MLP` bottleneck survives a stricter selection-split → eval-split test,
  - but it is **not** a tiny raw-coordinate story under naive absolute-delta ranking,
  - helpful `<-` zs recovers about `44%` of the dense margin effect at `k=64`, about `81%` at `k=256`, and about `88%` at `k=512`, while `k=1/4/16` are actually harmful.
  - exact self-patch no-op confirms the partial-coordinate hook is mechanically correct.
- Held-out sign-split subspace update (`hindi_mlp_subspace_signsplit_v1`):
  - splitting the raw coordinates by the sign of `zs - helpful` reveals a strong asymmetry.
  - for helpful `<-` zs, **negative-signed** coordinates are the cleaner small-k rescue direction:
    - `neg k=4`: `delta_mean_gap_latin = +1.571`, bootstrap CI `[+1.135, +2.027]`
    - `neg k=16`: `+1.352`, bootstrap CI `[+0.600, +2.092]`
  - by contrast, **positive-signed** coordinates are strongly harmful at small k:
    - `pos k=4`: `-2.921`, bootstrap CI `[-3.306, -2.496]`
    - `pos k=16`: `-2.535`, bootstrap CI `[-3.071, -1.996]`
  - for zs `<-` helpful, the same negative-signed family cleanly harms zero-shot (`k=4`: `-1.125`, CI `[-1.483, -0.773]`; `k=16`: `-2.089`, CI `[-2.498, -1.675]`).
- Better-basis Hindi update (`hindi_mlp_channel_patch_v1`): moving from the raw output basis to the **actual L25 MLP channel basis** (`down_proj` input channels, size `6912`) makes the localization substantially cleaner.
  - helpful `<-` zs in channel basis:
    - `abs k=4`: `delta_mean_gap_latin = +1.935`, bootstrap CI `[+1.635, +2.225]`
    - `abs k=16`: `+2.744`, CI `[+2.396, +3.092]`
    - dense `k=6912`: `+3.356`, `delta_top1_target_rate = +0.200`
  - this is a major contrast with the raw output basis, where `abs k=4/16` were harmful.
- Strongest current channel-basis clue: the **negative-signed channel subset** carries most of the harmful high-shot state.
  - helpful `<-` zs:
    - `neg k=4`: `+3.506`, CI `[+2.900, +4.154]`, `delta_top1_target_rate = +0.167`
    - `neg k=64`: `+4.052`, CI `[+3.158, +4.896]`, `delta_top1_target_rate = +0.167`
  - zs `<-` helpful:
    - `neg k=4`: `-2.108`, CI `[-2.450, -1.775]`, `delta_top1_target_rate = -0.233`
    - `neg k=64`: `-4.064`, CI `[-4.644, -3.481]`, `delta_top1_target_rate = -0.233`
- Positive-signed channel subsets are not the main harmful direction: for helpful `<-` zs they remain harmful or near-zero at small/medium `k`, and for zs `<-` helpful they are much less damaging than the negative-signed family.
- Channel-basis random-control confirmation (`hindi_mlp_channel_randomctrl_v1`): matched random channel subsets stay near zero while selected subsets remain strongly non-random.
  - helpful `<-` zs selected-minus-random:
    - `abs k=4`: `+1.961`, bootstrap CI `[+1.669, +2.263]`
    - `abs k=16`: `+2.728`, CI `[+2.389, +3.090]`
    - `neg k=4`: `+3.532`, CI `[+2.967, +4.138]`
    - `neg k=64`: `+4.250`, CI `[+3.398, +5.129]`
  - this substantially reduces the risk that the channel-basis result is just a generic partial-perturbation artifact.
- Exploratory top-negative pairwise panel (`hindi_mlp_channel_pair_panel_v1`) now suggests a much smaller core inside the negative-signed channel set.
  - strongest eval-panel pair so far is **`[5486, 2299]`**:
    - helpful `<-` zs: `delta_mean_gap_latin = +3.838`, CI `[+3.283, +4.404]`, `delta_top1_target_rate = +0.167`, `harm_rate_on_base_successes = 0.0`
    - zs `<-` helpful: `delta_mean_gap_latin = -2.318`, CI `[-2.609, -2.022]`, `delta_top1_target_rate = -0.167`
  - pairwise structure suggests `5486` is the most important channel, `2299` is the strongest secondary contributor, `789` is not necessary, and `6015` is weak on its own.
- Held-out pair-selection verification (`hindi_mlp_channel_pair_select_eval_v1`) now removes the main methodological objection to the pair claim.
  - the selection split independently chose the same pair `[5486,2299]`.
  - held-out eval with random-pair controls:
    - helpful `<-` zs: `delta_mean_gap_latin = +3.838`, chosen-minus-random `+3.836`, bootstrap CI `[+3.256, +4.425]`, `delta_top1_target_rate = +0.167`
    - zs `<-` helpful: chosen-minus-random `delta_mean_gap_latin = -2.310`, CI `[-2.597, -2.030]`, `delta_top1_target_rate = -0.167`
- This is now the strongest bounded Hindi claim in the repo: a **held-out, random-controlled two-channel core candidate** `[5486,2299]` inside the negative-signed `L25` MLP channel subspace.
- Channel value audit + sensitivity audit sharpened the explanation of the Hindi channel story.
  - simple helpful-vs-ZS value shift does **not** explain causal importance: `6015` shifts strongly but is nearly causally inert.
  - local first-order sensitivity explains the ranking much better:
    - `5486`: grad `≈ -0.212`, predicted `≈ +2.362`, actual singleton effect `≈ +2.381`
    - `2299`: grad `≈ -0.233`, predicted `≈ +1.869`, actual `≈ +1.644`
    - `6015`: grad `≈ -0.0079`, predicted `≈ +0.087`, actual `≈ +0.017`
    - `789`: grad `≈ +0.0204`, predicted `≈ -0.131`, actual `≈ -0.163`
- Pair interpretation sharpened too:
  - `[5486,2299]` remains the best held-out two-channel core candidate,
  - but the current margin evidence looks more **mostly additive / mostly linear** than sharply superadditive.
- Final Hindi closure runs both passed.
  - Readout geometry directly supports the channel ranking:
    - `5486`: dot `≈ -0.212`, cosine `≈ -0.084`, predicted `≈ +2.361`, actual `≈ +2.381`
    - `2299`: dot `≈ -0.233`, cosine `≈ -0.075`, predicted `≈ +1.868`, actual `≈ +1.644`
    - `6015`: dot `≈ -0.0079`, cosine `≈ -0.004`, predicted `≈ +0.087`, actual `≈ +0.017`
    - `789`: dot `≈ +0.0204`, cosine `≈ +0.011`, predicted `≈ -0.131`, actual `≈ -0.163`
  - One extra held-out seed repeat on `seed11` again selected `[5486,2299]` and preserved a large chosen-minus-random effect (`+4.004`, CI `[+3.429, +4.589]`).
- Decision: the bounded Hindi mechanism story is now strong enough to **freeze** and hand off to the next mechanistic branch.
- Updated best Hindi wording: **late target-vs-Latin competition failure under high-shot ICL, with the clearest current implementational bottleneck at the last-token layer-25 MLP; within that site, the current best mechanistic clue is a small, sign-asymmetric harmful channel subspace, with held-out pair candidate `[5486,2299]` carrying a large share of the effect, and with channel importance explained better by local readout sensitivity × state shift than by activation magnitude alone**.

- Matched Telugu continuation audits now give a clean positive-control comparison.
  - In this nearest-bank competitor setup, `same_first_token_rate = 1.0` for both `1B` and `4B`, so the discriminating signal lies in later continuation.
  - `1B Telugu` turns negative under helpful ICL:
    - `icl_helpful` sequence/continuation gap `≈ -6.720`
    - nearest-bank generation rate `≈ 0.267`
    - `similarity_desc` worsens further (`≈ -8.150`, bank generation `≈ 0.433`)
  - `4B Telugu` stays strongly positive under the same protocol:
    - `icl_helpful` gap `≈ +21.670`
    - nearest-bank generation rate `≈ 0.067`
    - `icl_corrupt` weakens but does not erase the preference (`≈ +5.194`)
- This strengthens the broader phase-map story:
  - `1B Hindi`: late first-token competition failure
  - `1B Telugu`: later continuation / nearest-bank retrieval failure
  - `4B Telugu`: preserved query-specific continuation preference.

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
- For `1B × Hindi`, is the newly identified **negative-signed overshoot-like raw subspace** best explained by a cleaner feature basis (e.g. MLP-aligned features / transcoders) or is it still mostly a basis artifact of the raw hidden space?
- For `1B × Hindi`, which upstream paths write the harmful state into the last-token late MLP / readout state: direct late attention, earlier layer-output mediation, or specific query-to-last-token paths best tested with path patching?
- For `1B × Hindi`, where in the computation does the visible ICL signal fail beyond the first token: target-token selection, whole-word continuation, or a later continuation/composition step?
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
4. For `1B Hindi`, the next exact mechanistic step is now narrower than before: move from raw-coordinate patching to a **better MLP-aligned basis** (feature / subspace / neuron-level if justified) using the same target-vs-Latin-competitor metric.
5. Add a **path-patching style** Hindi test if the question is specifically whether query-position information mediates the final last-token decision through late layers.
6. Only after localization survives checks, run heavier causal tests on selected components.
7. Bring `270M` back into the final synthesis as the capability-floor comparison even if the rich helpful-vs-control matrix stays focused on `1B/4B`.

## Failure cases worth inspecting
- Modal local-client disconnect interruptions (`modal run` without stable detach workflow).
- `modal run --detach` on the local entrypoint can keep the remote GPU job alive but still drop local packet-writing/postprocess steps; use the manifest orchestrator directly for durable local artifacts.
- Transcoder smoke currently fails because the requested SAE ID is invalid for the chosen release (`layer_12_width_262k_l0_medium` missing from `gemma-scope-2-1b-it-transcoders-all`).
- Mixed-script outputs remain common well beyond `N=0` for several languages, especially Hindi/Bengali/Marathi/Tamil.
- Any Unicode delimiter/tokenization sensitivity in prompts (now switched to explicit ASCII `->` separator for consistency).

## 2026-03-29 — larger-N four-cell confirmation

### Established
- The four-cell stage map survives beyond the original 30-item analyses.
- `1B Hindi` remains the clearest harmful first-token competition anchor under high-shot ICL.
- `4B Hindi` remains a stable already-Hindi regime with within-script errors rather than Latin collapse.
- `1B Telugu` vs `4B Telugu` remains the clearest continuation-stage scale contrast.

### Supported but provisional
- Hindi continuation degradation under high-shot ICL is real, but the nearest-bank continuation instrument is less cleanly continuation-isolated than Telugu because `same_first_token_rate` is only about `0.79`.
- For the thesis core, `4B Hindi` probably does not need immediate deeper causal localization.

### Open questions
- Exact implementational pathway for the `1B` vs `4B` Telugu continuation contrast.
- Whether the four-language seed aggregate reveals any contradiction to the current `2 deep + 2 showing` plan.

### Priority next experiments
- `4B Telugu` continuation localizer.
- matched `1B Telugu` continuation localizer.
- finish four-language `seed101` and aggregate.

## 2026-03-29 — Telugu continuation localizer design

### Established
- The next mechanistic bottleneck is the `1B` vs `4B` Telugu continuation contrast.
- The cleanest next state-localizer is not token 1, but the first divergence token after the shared gold-vs-bank prefix.

### Supported but provisional
- If the new localizer behaves well, the right next causal step will likely be Telugu continuation patching rather than deeper `4B Hindi` work.

### Priority next experiments
- inspect smoke results for `telugu_continuation_localizer.py`
- if clean, launch full `1B` and `4B` Telugu continuation localizer runs
- after that, decide between dense patching and a narrower component screen

### Telugu localizer update
- Established:
  - `1B Telugu` late continuation failure is now localized to a band roughly `L18–26`, strongest at `L26`.
  - `4B Telugu` helpful-specific late continuation advantage is now localized to a band roughly `L30–34`, strongest at `L34`.
- Supported but provisional:
  - the remaining Telugu uncertainty is now mostly about **component/site** rather than stage.
- Priority next experiments:
  - divergence-token component localization for `1B Telugu` and `4B Telugu`
  - then bounded causal patching of the strongest component/site
- Telugu divergence-token component localization is now materially clearer.
  - `1B Telugu` full component panel (`research/results/autoresearch/telugu_continuation_component_v1/1b/aksharantar_tel_latin/seed42/nicl64/telugu_continuation_component_panel.json`) shows the strongest non-final-state movement in a **broad late layer-output band** across roughly `L18–26`, with the clearest row at `icl_helpful <- zs @ L26 layer_output` (`delta_mean_gap ≈ +9.424`, `delta_gold_top1 ≈ +0.179`, `delta_competitor_top1 ≈ -0.429`).
  - `1B Telugu` also shows a cleaner but weaker secondary attention contribution at `L18 attention_output` (`delta_mean_gap ≈ +6.523`, no observed base-gold harms in the 28-item panel), while `mlp_output` remains weak throughout.
  - `4B Telugu` full component panel (`research/results/autoresearch/telugu_continuation_component_v1/4b/aksharantar_tel_latin/seed42/nicl64/telugu_continuation_component_panel.json`) shows the strongest non-final-state movement in a **broad late layer-output positive band** across roughly `L30–34`, with the clearest row at `icl_corrupt <- icl_helpful @ L34 layer_output` (`delta_mean_gap ≈ +10.258`).
  - In both Telugu models, `final_state` rows now act mainly as readout confirmations rather than clean site localizations.
- Current best Telugu wording is therefore:
  - `1B Telugu`: broad late layer-output failure band, with some possible earlier-late attention contribution.
  - `4B Telugu`: broad late layer-output positive band, with weaker attention/MLP contributions.
- This weakens a Hindi-style narrow-MLP analogy for Telugu and pushes the next validation step toward larger-`N` checks plus bounded causal patching of the strongest layer-output candidates rather than immediate basis decomposition.
- Marathi same-script control probe (`research/results/autoresearch/marathi_first_token_60_v1/results/summary.json` and `research/results/autoresearch/marathi_continuation_60_v1/...`) now provides a direct test of the simple “Hindi vs non-Hindi” story.
  - `1B Marathi` is a hybrid: early first-token routing is much stronger than `1B Hindi` under helpful ICL (`top1_target_rate ≈ 0.75`), but later continuation still turns negative (`helpful seq_gap ≈ -1.41`, `nearest_bank_generation_rate ≈ 0.15`, `same_first_token_rate = 1.0`).
  - `4B Marathi` stays strongly positive in continuation (`helpful seq_gap ≈ +27.83`, `corrupt ≈ +8.63`).
- Current best interpretation of the Marathi result:
  - same-script support likely helps the early routing / first-token stage,
  - but same script does not guarantee a Hindi-like overall regime,
  - which further weakens any one-axis explanation in terms of “Hindi dominant vs all other languages Telugu-like.”
- Endgame synthesis update:
  - Current strongest cross-language intuition is a two-axis map: (1) early target/task entry and (2) late query-specific continuation against bank attractors.
  - Marathi same-script probe strengthens the claim that these axes are partially dissociable: stronger early routing than `1B Hindi`, but still weak late continuation.
  - This makes the current thesis-level story less compatible with a one-axis “Hindi dominant vs others” explanation and more compatible with a stage-based ICL regime account.
- Telugu late-band persistence probe (smoke) supports a more specific reading of the broad layer-output result:
  - `1B`: patching `L26` layer output already matches the tested multi-layer late-band combinations on the smoke slice.
  - `4B`: patching `L34` layer output already matches the tested multi-layer late-band combinations on the smoke slice.
- Supported but still provisional:
  - the Telugu late band may be a **persistent carried continuation state** whose strongest readable expression is near the final late layer, rather than a strongly additive set of many late mediators.
- Telugu layer-band full confirmation:
  - `1B`: `L26` alone matches all tested combinations containing `L26` on the full panel; earlier late layers carry partial signal but add no extra rescue once `L26` is patched.
  - `4B`: `L34` alone matches all tested combinations containing `L34` on the full panel; earlier late layers carry partial signal but add no extra rescue once `L34` is patched.
- Claim status:
  - Supported but now much stronger: the broad Telugu late `layer_output` story is best read as a **persistent carried continuation state** with strongest readable expression near the final late layer.
  - Still open: whether earlier candidate sites causally write through that mediator state.
- New `1B Telugu` mediation-smoke evidence:
  - `L18 attention_output` and `L18 layer_output` both rescue when donor-patched into the helpful recipient,
  - but the rescue disappears when `L26 layer_output` is overwritten back to the recipient state,
  - while off-band overwrite does not remove the rescue.
- Supported interpretation:
  - the earlier `1B Telugu` late candidates act through the final late mediator state (`L26 layer_output`).
- Telugu mediation smoke now supports the same qualitative story on both `1B` and `4B`:
  - earlier late writer candidates help on their own,
  - that help is removed by overwriting the final late mediator back to the recipient state,
  - and is not removed by overwriting an off-band site.
- Claim status:
  - Supported but still provisional until larger-`N` confirmation: the Telugu late continuation story is not just a broad redundant band, but a **mediated carried-state structure**.
- Full `1B Telugu` mediation confirmation strengthens the mechanistic story from smoke-level to panel-level:
  - earlier late writer candidates still help on their own,
  - their help is eliminated by mediator overwrite at `L26 layer_output`,
  - and is preserved under off-band overwrite.
- Claim status:
  - Supported but still not final: `1B Telugu` now has a strong mediated carried-state interpretation centered on `L26 layer_output`.
  - Remaining gap before stronger closure: confirm the same at larger `N` for `4B` and then test late-site sufficiency directly.
- Telugu mediation full confirmation is now complete on both `1B` and `4B`.
- Current claim status:
  - Stronger than before: the Telugu late continuation story is best read as a **mediated carried-state bottleneck** centered on `L26` (`1B`) / `L34` (`4B`).
  - Still provisional: full sufficiency awaits the larger-`N` final-site transplant confirmation now running as `proc_16`.
- Telugu full final-site sufficiency confirmation:
  - `1B L26`: site donor > off-band > random, with positive bootstrap CIs for site-minus-off-band and site-minus-random gap differences.
  - `4B L34`: site donor > off-band > random, with positive bootstrap CIs for site-minus-off-band and site-minus-random gap differences.
- Claim status update:
  - Strong and close to established for this task/model family: Telugu late continuation is mediated through a dominant late `layer_output` bottleneck (`L26` / `L34`) that also carries meaningful content on its own.
  - Still open but non-blocking: exact fine-grained circuit identity and broader transfer/generalization.

## 2026-03-30 — practical Hindi patch and non-core early-axis follow-up

### Established
- A fixed practical inference-time patch derived from the bounded Hindi mechanism improves held-out `1B Hindi` generation quality relative to baseline and controls.
- The effect is directional rather than generic:
  - chosen-pair positive patch helps,
  - sign-flip hurts,
  - random-pair and zero-ablation controls stay near baseline.
- Bengali/Tamil non-core first-token probes now reduce uncertainty in the broader phase map:
  - `1B Bengali` is strongly weak-early,
  - `1B Tamil` is intermediate-early rather than fully Telugu-like,
  - `4B Bengali/Tamil` both improve strongly on the early axis.

### Supported but provisional
- The practical Hindi patch gives a real end-to-end gain, but the exact-match improvement remains modest and still statistically weak on the current 60-item eval panel; CER and first-entry improvements are much cleaner.
- The current out-of-core predictive check is now much stronger than before, but it is still a qualitative/panel-level predictor rather than a fully learned held-out classifier.

### Open questions
- Whether the practical Hindi patch exact-match gain remains positive and sharper at larger eval N.
- Whether a Telugu practical steering patch can achieve a similarly clear mechanism-to-performance bridge on the late continuation side.
- Whether a formal two-axis predictor can be fit and evaluated cleanly on held-out language/model cells without overfitting the current story.

### 2026-03-30 addendum — Hindi single-channel vs signed-patch intervention panel

#### Established
- The bounded Hindi mechanism is better characterized as a **signed harmful two-channel state** than as a simple “turn these channels off” lesion target.
- On the held-out 60-item panel, the calibrated signed two-channel mean shift remains much stronger than single-channel or dual-channel zero-ablation.
- Channel `5486` appears more individually consequential than `2299`, but neither single-channel suppression matches the calibrated signed edit.

#### Supported but provisional
- The practical patch does improve exact match numerically, but the exact-match gain remains modest; the stronger support is still CER / first-entry / first-token competition.
- Harm-on-already-correct cases for the practical patch is hard to read from this panel because baseline exact-match successes are only `2/60`; the observed `0.5` harm rate there is therefore high-variance and should not be overinterpreted.

#### Open questions
- Whether the same signed-subspace-vs-lesion distinction remains as clear at larger eval N.
- Whether a dynamic Telugu continuation steering intervention can produce an equally practical mechanism->performance bridge, or whether Telugu fundamentally needs state-conditional late correction rather than a static prompt-time edit.

### 2026-03-30 addendum — first Telugu practical steering smoke

#### Established
- A first practical `1B Telugu` late-state steering design was implemented and smoke-tested: fixed additive mean delta at `L26 layer_output`, learned on a selection split and evaluated on held-out items from shared-prefix continuation contexts.
- After fixing continuation extraction so trailing junk does not dominate CER, this first design is **not yet a clean practical win**.

#### Supported but provisional
- The weak outcome suggests that Telugu's late bottleneck may be harder to exploit with a static additive patch than Hindi's bounded signed subspace.
- A likely reason is that first-divergence local selection is not sufficient to choose the best practical continuation patch family.

#### Open questions
- Whether a better practical Telugu family is a fixed blend/replace toward the mean donor state rather than a pure additive delta.
- Whether selection should optimize continuation-generation metrics rather than only local divergence-gap metrics.
- Whether `4B Telugu` is worth a practical robustness-style patch, or whether its main value should remain mechanistic rather than interventional.

### 2026-03-30 addendum — full Telugu practical patch evaluation

#### Established
- Full held-out practical Telugu patch evaluations are now complete for both `1B` and `4B` using fixed late-state interventions at the mechanistic bottleneck sites.
- The current static patch family is much weaker than the Hindi practical patch.
  - `1B Telugu`: chosen mean shift is essentially neutral on full exact match and full CER, with only modest bank-copy reduction.
  - `4B Telugu`: chosen mean shift is also near-neutral on end-to-end metrics; sign flip is not clearly worse on generation metrics, so the result is not directionally specific enough for a strong practical claim.
- Stronger bank-copy suppression alone is not sufficient for practical rescue in Telugu; several controls reduce bank-copy while also worsening overall generation quality.

#### Supported but provisional
- The best current interpretation is that Telugu's late bottleneck is real but more conditional than Hindi's bounded signed subspace, so a single fixed vector is probably the wrong practical intervention family.

#### Open questions
- Whether one more small practical Telugu iteration should test a more conditional or blended intervention family.
- Whether the workshop/thesis should preserve the current Telugu practical result as a negative/contrastive finding and stop there.

### 2026-03-30 addendum — preserve Telugu practical patching as contrast, not rescue

#### Established
- The current Telugu practical patch results should be preserved as a negative/contrastive finding rather than pushed into a stronger practical claim.
- A skeptical paper audit identified the main remaining wording risks and those were corrected in the draft: Hindi practical gains are now framed more carefully, Telugu practicality is explicitly weak/contrastive, and oracle donor-state mechanistic patching is separated more clearly from fixed practical patching.
- The paper now contains a compact Telugu practical-results table and explicit robustness notes for that analysis.

#### Supported but provisional
- A more conditional Telugu intervention family may still be interesting later, but it is no longer required for the thesis-safe core story.

### 2026-03-30 addendum — compile closure and final endgame status

#### Established
- The current paper source compiles locally to PDF after provisioning a Tectonic-based compile path.
- The end-to-end project state is now coherent enough to treat the core thesis loop as closed: the remaining questions are real but no longer blockers.

#### Open questions (non-blocking)
- direct pretraining-distribution causality,
- a formal held-out regime predictor,
- a more conditional Telugu practical intervention family,
- cross-model generality beyond Gemma 3.

### 2026-03-30 addendum — prose rewrite and figure redesign

#### Established
- The draft now reads much more like a paper than a research memo after rewriting the abstract, introduction, and conclusion.
- The figure set is materially stronger after redesigning the chart types and layouts to foreground the actual scientific comparisons rather than just dumping bars.
- The updated draft compiles successfully after the prose and figure changes.

### 2026-03-30 addendum — Altair-style figures and Excalidraw conceptual diagrams

#### Established
- The paper now has two new Excalidraw-style conceptual figures that communicate the regime map and the Hindi-vs-Telugu intervention contrast more clearly than the old hand-built TikZ figure.
- A reproducible local Excalidraw export loop is working in this workspace using `mcp_excalidraw` on port `3010` plus a headless Chrome frontend client.
- The updated paper compiles successfully with the new conceptual diagrams and the newer plot style.

## 2026-03-30 — phase 2/3 compute restart and queue launch

### Established
- The new non-paper experiment stack is in place and syntactically validated:
  - `hindi_1b_patch_safety_audit.py`
  - `kshot_regime_sweep.py`
  - `telugu_temperature_sweep.py`
  - `telugu_writer_head_probe.py`
- Remote smoke runs now pass for all four new experiment families.
- A real implementation bug in the first Telugu writer-head probe was found and fixed:
  - human layer labels (`L26`, `L34`) had been passed directly into zero-based internal hook/extraction APIs,
  - and the script assumed `model.model.layers` even on `4B`,
  - both were repaired before the full queue launch.
- The Phase 2/3 compute queue is now live in background as `proc_5` (`phase23-compute-queue`).

### Supported but provisional
- Smoke-scale writer probing is at least directionally informative:
  - `1B Telugu` smoke surfaced strongest positive ablation effects around `L18` heads, especially `L18H0` / `L18H2`, with top-group ablation larger than matched-random on the tiny 2-item smoke.
  - `4B Telugu` smoke surfaced strongest negative-gap ablation at `L30H7`, again on only a 1-item smoke.
- The Hindi safety smoke already suggests the practical patch may have nontrivial off-task effects on unrelated prompts, so the full safety result matters and should not be assumed benign.

### Open questions
- Do the full writer-head probe runs preserve the smoke ordering once `N` is increased beyond the tiny smoke items?
- Does the full Hindi safety audit confirm real side effects, or were the smoke prompts too brittle / underpowered?
- Do the full fixed-split k-shot curves show clean monotonic Telugu bank drift and a readable Hindi early-entry trajectory?
- Does Telugu bank-copy remain strong at `T=0.2` / `T=0.7` in the full temperature sweep?

### Priority next experiments
1. Let the background queue finish and inspect `proc_5` artifacts.
2. If the queue fails mid-run, resume from the last completed phase rather than restarting all four jobs blindly.
3. After result inspection, decide whether any follow-up should be an extension run (more seeds / more items) or whether the paper can move to writing-only mode.

## 2026-04-01 — reviewer follow-up sync and synthesis

### Established
- The remaining reviewer-followup artifact families are now present locally after direct VM sync.
- Behavioral cross-family support now extends beyond Gemma 3:
  - `Qwen 2.5 1.5B`, `Qwen 2.5 3B`, and `Llama 3.2 3B` reproduce the broad Hindi-vs-Telugu asymmetry and helpful-over-corrupt advantage.
- `Llama 3.2 1B` does not replicate Gemma 3 `1B`'s structured regime map cleanly; it behaves more like a capability-floor model.
- The larger-`N` prompt-composition ablation strengthens the `1B Telugu` retrieval/attractor reading while ruling out a single-exemplar-only explanation.
  - Similarity-forward ordering improves entry and CER but increases nearest-bank copy.
  - Similarity-back ordering suppresses copy but badly harms entry.
  - Nearest/top-2 drop ablations reduce copy pressure without restoring exact match.
- The paper draft and compiled PDF now reflect these new results.

### Supported but provisional
- The cleanest external claim is now: the regime map generalizes behaviorally across multiple families once the model is above a capability floor, but not as a simple per-parameter law.
- The best current Telugu wording is that the late failure is a distributed similarity-sensitive attractor basin, not a one-nearest-neighbor copy rule.

### Open questions
- Whether Qwen/Llama share any of the Gemma late-state mediation structure internally.
- Why `Llama 3.2 1B` sits below the apparent floor for the structured regime map.
- Whether a conditional prompt-selection or dynamic intervention family can exploit the Telugu similarity signal without collapsing early entry.

## 2026-04-01 addendum — end-to-end reverification and reviewer-fix pass

### Established
- The current local copies of the paper-critical artifacts match the VM originals on the bounded oracle set checked in `outputs/end_to_end_reverification_2026-04-01.md`.
- The cross-family behavioral claim is now stronger and better bounded: Qwen 2.5 `1.5B/3B` and Llama 3.2 `3B` support the Hindi-vs-Telugu regime asymmetry, while Llama 3.2 `1B` remains a capability-floor counterexample.
- The larger-`N` prompt-composition ablation now strongly supports a distributed similarity-sensitive attractor reading for `1B Telugu`, not a single-nearest-exemplar story.
- The paper now uses the 200-item reviewer rerun as the canonical Hindi practical-patch panel.

### Supported but provisional
- The regime map appears behaviorally cross-family above the capability floor, but the exact family-specific floor likely depends on tokenizer/training/instruction details rather than parameter count alone.

### Open questions
- Cross-family mechanistic replication remains open.
- The Telugu late state is still mechanistically localized more strongly than it is semantically interpreted.
- A conditional Telugu intervention strategy remains the most interesting practical follow-up.

## 2026-04-01 addendum — alpha-grounded literature review and claim audit

### Established
- The paper's current mechanistic framing is better grounded when mapped onto three distinct literature lines rather than one blurred story:
  1. script barriers / transliteration / romanization,
  2. staged or implicit ICL mechanisms,
  3. retrieval-based ICL and prompt-composition sensitivity.
- `RomanLens` provides a useful but bounded bridge hypothesis for early script-entry effects; it should be used as consistency evidence, not as direct proof of our Gemma mechanism.
- Retrieval-based ICL literature strongly supports the claim that similarity and order of demonstrations can materially alter behavior, which improves the grounding of the Telugu prompt-composition section.
- The paper had real bibliography-accuracy issues before this pass; these are now corrected for the main relevant citations.

### Supported but provisional
- The early entry axis may relate to internal romanized or script-bridge representations, but that remains a cross-paper synthesis rather than a directly localized Gemma 3 finding.

### Open questions
- Whether Gemma 3 itself shows latent-romanization-like intermediate states on the Hindi/Telugu tasks remains open.
- Whether the Telugu late attractor can be linked to a more general retrieved-demonstration interaction mechanism across families remains open.

## 2026-04-01 addendum — conference-style paper variant

### Established
- The project now has two paper-shaped artifacts with the same scientific core but different presentation styles:
  - `gemma_1b_icl_paper_v11.tex` as the fuller thesis-leaning draft,
  - `gemma_1b_icl_paper_conference_v1.tex` as a cleaner workshop/conference-style variant.
- The conference variant improves page-1/page-2 signposting without changing the underlying scientific claim set.

### Supported but provisional
- The conference variant is likely the better starting point for external readers or workshop-style review because it foregrounds one central claim and adds clearer visual cues.

### Open questions
- Whether the conference variant should become the new canonical draft depends on final advisor/user preference after direct PDF reading.

## 2026-04-01 addendum — visual verification and venue fit

### Established
- The conference-style paper variant is now materially cleaner than the original conference rewrite after direct page-image inspection.
- The main behavioral regime figure is better as a native TikZ figure than as the previous raster plot in the conference variant.
- The conference variant reads more coherently after merging the Hindi/Telugu mechanism story into one comparative section rather than two separate case-study sections.
- The paper is clearly workshop-worthy and plausibly conference-worthy; the limiting factors for main-track competitiveness are now mostly statistical/packaging rather than absence of a core contribution.

### Supported but provisional
- BlackboxNLP 2026 and the ICML 2026 Mechanistic Interpretability Workshop look like the strongest currently visible venue fits from the official web pass.
- A future main-track ACL-family attempt still looks plausible, but probably benefits from one more hardening pass on uncertainty reporting and claim packaging.

### Open questions
- Whether to convert additional quantitative figures (Hindi patch, cross-family heatmap) to native TikZ for style consistency remains open.

## 2026-04-01 addendum — separate 8+2 submission variant approved

### Established
- The project no longer needs to force one artifact to serve both the fuller conference-style narrative and the stricter compressed submission target.
- A separate dedicated submission file now exists:
  - `Paper Template and Paper/Paper/gemma_1b_icl_paper_submission_8plus2_v1.tex`

### Open questions
- Exact page-budget strategy for the final 8+2 compression will be chosen after the prompt-robustness runs finish and their results are integrated.

## 2026-04-01 addendum — submission endgame queue staged

### Established
- No new major experimental branch is currently justified beyond the two already-running robustness jobs.
- The remaining missing work is now explicitly staged as a paper-completion queue: robustness summaries, claim-boundary text, qualitative examples, appendix linkage, and final compile/verification.

### Open questions
- Whether the prompt-format sensitivity and Hindi tagged patch variant preserve the current story cleanly enough to be folded into the short submission paper without new caveats.

## 2026-04-01 addendum — curiosity follow-up queued

### Established
- The Hindi fixed-patch compact prompt variant remains directionally positive.
- The tagged prompt variant is mixed: the first-token target-vs-Latin gap still improves, but held-out CER does not improve under that template.
- A same-script Marathi follow-up is now queued as a curiosity experiment to test whether the Hindi practical-patch site/channel family behaves more like a broader Devanagari-entry bottleneck or a Hindi-specific one.

### Open questions
- Whether the Marathi same-site patch shows a bounded transfer-style signal at the Hindi site/channels, or whether it comes back null / weak and thereby sharpens Hindi specificity.

## 2026-04-01 addendum — second curiosity follow-up staged

### Established
- The patch-eval runner can now execute fixed-vector transfer experiments by loading an external patch payload directly, without re-fitting a new mean delta on the target language.
- This enables a sharper curiosity decomposition for Marathi:
  - same-site re-fit on Marathi,
  - exact Hindi-vector transfer to Marathi.

### Open questions
- Whether Marathi supports the Hindi site only, or the Hindi direction itself.
- Whether the resulting pair of curiosity experiments changes the paper-level interpretation materially, or remains appendix-only boundary-setting.

## 2026-04-01 addendum — first-principles understanding pass

### Established
- The strongest current first-principles picture is a two-computation decomposition:
  - output-manifold / script entry
  - query-conditioned continuation against prompt-bank attractors
- The Hindi fixed-patch direction is nearly invariant across `canonical`, `compact`, and `tagged` prompt variants (pairwise cosine `> 0.998`).
- The tagged Hindi prompt variant preserves local first-token-gap rescue on nearly all items but does not reliably convert that rescue into generation-level entry or CER gains.

### Supported but provisional
- The current evidence fits recent Gemma / ICL literature best when read as a stage-structured story rather than a single ICL mechanism.
- Script / romanization interventions likely align more strongly with the early-entry axis than with full continuation.
- Static-vector intervention failure in Telugu is increasingly consistent with a higher-rank or conditional late-state hypothesis.

### Open questions
- Why does the tagged prompt format decouple local Hindi rescue from generation-level rescue?
- Does Marathi share only the Hindi patch site or the Hindi patch direction itself?
- How much of the Telugu late-state story requires dynamic rather than static intervention?

## 2026-04-01 addendum — Hindi cross-template transfer queued

### Established
- A deeper follow-up is now staged to test the operating-point hypothesis directly via cross-template Hindi vector transfer.
- The queued design uses stored external patch artifacts rather than re-fitting, so it directly probes vector portability across prompt templates.

### Open questions
- If the canonical vector still fails on tagged prompts while the tagged vector works on canonical prompts, how strongly should the paper frame prompt-conditioned global state geometry as the main explanation?
- If transfer is more symmetric than expected, does the current prompt-sensitivity interpretation need to be softened?

## 2026-04-01 addendum — curiosity queue restart fix

### Established
- The Marathi curiosity queue failure was operational: the long-lived background queue shell did not retain the VM credential needed once it woke up after waiting.
- The failure does not currently count as evidence against the Marathi same-site hypothesis.

### Open questions
- Whether the restarted Marathi same-site and transfer runs complete cleanly with the queue-wrapper fix.

## 2026-04-04 addendum — final honors deliverables review pass

### Established
- The new combined final Honors report now compiles successfully with Tectonic and keeps the thesis story coherent from Phase 1 negative results through Phase 2 mechanistic findings.
- The new final presentation also compiles successfully with Tectonic after fixing Beamer/TikZ frame-macro issues.
- Additional PDF screenshot inspection caught and corrected real clarity issues in newly added TikZ summary figures (timeline overflow, first-principles slide clipping, intervention-pipeline overlap).

### Supported but provisional
- The current final report is now materially clearer on the first-principles decomposition (entry vs continuation), the bounded status of the Hindi patch claim, and the literature-grounded boundary between Gemma-specific mechanism and cross-family behavior.

### Open questions
- Whether to do one more content pass specifically for supervisor-specific institutional formatting or chapter-length expectations once advisor feedback arrives.

## 2026-04-04 addendum — second final-deliverables recheck

### Established
- Both final Honors deliverables still build successfully after the latest edits:
  - `Final_Honors_Report/report.pdf`
  - `Final_Honors_Presentation/presentation.pdf`
- A whole-deck presentation review found one real remaining issue (summary-slide text clipping), and that issue has now been fixed and rechecked.
- No obvious placeholder text or replacement-character corruption was found in rendered PDF text extraction for the final report or presentation.

### Supported but provisional
- Remaining `tectonic` warnings now appear mostly cosmetic/non-blocking on the pages inspected (appendix layout warnings in the report; small `nullfont` digit warnings in the presentation).

## 2026-04-04 addendum — template-aesthetic repair

### Established
- The mismatch with the official institute report template was primarily a global-style issue, not a broken compile: main font override, custom headers, custom geometry, and other preamble-level styling had moved the report away from the stock template look.
- The final Honors report has now been moved back toward the actual template aesthetic while preserving the thesis content, figures, and multilingual support.
- Devanagari snippets that would have broken under the restored template-like Latin text font now render correctly via explicit local `\indic{...}` wrapping.

### Supported but provisional
- The report is now much closer to the official template aesthetically, though it still intentionally keeps a few modern research conveniences (TikZ figures, tcolorboxes, richer tables) that the bare stock template does not provide.

## 2026-04-09 — ICML v7 reviewer-proofing pass

### Established
- The early behavioral axis in the paper should be described as **first-akshara correctness / early target start**, not as a loose script-family entry check; Hindi often manifests as failure to enter Devanagari, but the operational metric is stricter.
- Held-out Telugu static patch evidence is now explicitly summarized from the existing 191-item review panel:
  - selected patch = `L26_layer_output` mean-shift with `alpha=0.25`
  - full exact match stays `0.0 -> 0.0`
  - full CER improvement delta is only `+0.0001` with 95% CI `[-0.0037, +0.0034]`
  - bank-copy rate is unchanged
  - first-divergence gold-vs-bank gap worsens by `-0.139` with 95% CI `[-0.240, -0.036]`
- The 200-item Telugu prompt-composition ablation gives a clean front/back tradeoff that is worth showing directly rather than only narrating:
  - helpful: `E_ak=0.540`, `CER=0.777`, exact bank-copy `0.405`
  - similarity-front: `E_ak=0.620`, `CER=0.708`, exact bank-copy `0.450`
  - similarity-back: `E_ak=0.215`, `CER=0.927`, exact bank-copy `0.135`

### Supported but provisional
- Cross-family follow-up panels now have an explicit continuation-sensitive proxy on hand from existing JSONs: helpful-condition fuzzy bank-copy is modest but nonzero for stronger Telugu rows (`Qwen 2.5 1.5B: 0.120`, `Qwen 2.5 3B: 0.065`, `Llama 3.2 3B: 0.130`). This is bounded behavioral support for harder continuations after strong starts, not a full reproduction of the Gemma 3 1B Telugu regime.
- The small Hindi patch safety audit is useful for calibration but should stay clearly secondary: it is a pair of 24-item one-word cloze sets, not a broad safety benchmark.

### Open questions
- Whether the final workshop submission should keep the new continuation-sensitive cross-family column, or narrow the cross-family claim even further despite having the column available.
- Whether the appendix now contains the right amount of follow-up detail, or whether one more compact study-design table would still materially help reviewers.

## 2026-04-09 — ICML v8 reviewer-proofing pass

### Established
- The paper's second behavioral axis no longer needs to be only proxy-defined: the existing evaluation stack already exposes a direct conditioned continuation metric via `continuation_fidelity`, i.e. continuation akshara CER computed only when the first akshara is correct.
- Recomputed stage-sensitive 1B `k=64` values from the canonical raw artifacts:
  - Hindi: `E_ak 0.60 -> 0.50`, `CER_tail_help = 0.754`, `bank_help = 0.267`
  - Marathi: `E_ak 0.367 -> 0.733`, `CER_tail_help = 0.648`, `bank_help = 0.367`
  - Telugu: `E_ak 0.000 -> 0.900`, `CER_tail_help = 0.674`, `bank_help = 0.800`
- The existing fixed-split `k`-shot sweep artifact supports a bounded appendix display for Hindi/Telugu without rerunning experiments:
  - Hindi helpful condition is strongest around `k=32` (`E_ak = 0.633`, `CER = 0.659`)
  - Telugu helpful exact bank-copy is `0.150 / 0.133 / 0.167 / 0.383 / 0.083` at `k = 8 / 16 / 32 / 64 / 128`
- The Telugu static patch is now specified in-paper as a full residual-state mean-shift at `L26_layer_output` on the first divergence token, with `alpha` selected on a disjoint split by the mean first-divergence gold-vs-bank gap.

### Supported but provisional
- The fixed-split `k`-shot sweep is still single-seed and 60-item, so it is good appendix robustness evidence rather than a major standalone pillar.
- `CER_tail` is a cleaner direct continuation metric than unconditional bank-copy, but bank-copy remains useful as a concrete diagnostic of the retrieval-like late-drift subtype rather than of all late failures.

### Open questions
- Whether the final workshop package should stop at `v8`, or whether one last packaging-only pass should rename it to the canonical submission filename.
- Whether any reviewer would still insist on a compact study-design table despite the now-expanded appendix support.

## 2026-04-09 — ICML v9 comparability / reporting pass

### Established
- The Telugu held-out null is not deployment-analogous to the Hindi patch: its insertion point is oracle-defined by the gold target and nearest-bank competitor via `t_div`, so the negative result should be interpreted as failure even under a favorable oracle-positioned static edit rather than as failure of a fully fixed deployable patch.
- The helpful-condition `CER_tail` denominators on the matched 30-item diagnostic panels are:
  - Hindi: `15/30`
  - Marathi: `22/30`
  - Telugu: `27/30`
- Recomputed bootstrap support for those conditional means from the raw artifacts:
  - Hindi: `0.754`, bootstrap 95% CI `[0.591, 0.914]`
  - Marathi: `0.648`, bootstrap 95% CI `[0.505, 0.790]`
  - Telugu: `0.674`, bootstrap 95% CI `[0.571, 0.784]`

### Supported but provisional
- The main-text paper now reports the eligible counts directly, which materially improves interpretability of the stage table. The bootstrap intervals are saved in a supporting artifact but not promoted into the table because the count fix was the highest-value low-space change.

### Open questions
- Whether `v9` should now be frozen as the canonical workshop submission draft, or whether a final packaging-only rename / cleanup pass is still desirable.

## 2026-04-09 — ICML v10 temperature-sweep anchoring pass

### Established
- The Telugu temperature sweep cited in Sec. 4.2 is not the matched 30-item diagnostic panel from Table 1. It is a separate single-seed helpful Telugu 1B follow-up panel with `n_items = 57`.
- The sweep uses one sample per item at `T=0.0` and three sampled continuations per item at `T=0.2` and `T=0.7`, giving `n_samples = 57, 171, 171` respectively.
- Helpful-condition sample-level rates from the stored artifact are:
  - `T=0.0`: `E_ak = 0.544`, `CER = 0.763`, exact bank-copy `0.421`, fuzzy bank-copy `0.474`
  - `T=0.2`: `E_ak = 0.614`, `CER = 0.735`, exact bank-copy `0.503`, fuzzy bank-copy `0.550`
  - `T=0.7`: `E_ak = 0.708`, `CER = 0.710`, exact bank-copy `0.591`, fuzzy bank-copy `0.637`

### Supported but provisional
- The temperature sweep remains a bounded single-seed follow-up, but its role in the paper is now clearer: it is evidence that Telugu's copy attractor is not purely a greedy-decoding artifact, not a replacement for the main 30-item diagnostic slice.

### Open questions
- Whether `v10` should now become the canonical frozen submission draft, or whether the remaining work is only packaging / metadata / submission assembly.

## 2026-04-09 — ICML v11 Telugu follow-up evidence pass

### Established
- The `L26/L34` Telugu summary can now be tied to visible appendix evidence rather than only to unstated internal artifacts.
- Direct late-site follow-up numbers from the stored seed-42 panels:
  - `1B`: true site `L26_layer_output`, `n=57`, `Δ\mathcal{L}_{div}` shift `+7.253`; off-band `L10` gives `+2.575`; random-direction control gives `-0.128`; no-op gives `0.000`.
  - `4B`: true site `L34_layer_output`, `n=57`, `Δ\mathcal{L}_{div}` shift `+9.096`; off-band `L20` gives `+4.296`; random-direction control gives `-0.236`; no-op gives `0.000`.
  - Separate `4B` layer-band follow-up also identifies `L34` as the strongest tested late residual site among `L20/L30/L32/L34` (`L30=+7.887`, `L32=+7.895`, `L34=+9.096`).
- Exploratory writer-head probe numbers now surfaced in appendix form:
  - `1B` usable items `=38`; top heads by writer score are `L18H0`, `L18H2`, `L19H1`; grouped top-head mean `Δ\mathcal{L}_{div} = +4.048` vs matched-random `-1.543`.
  - `4B` usable items `=38`; top heads are `L30H6`, `L34H2`, `L30H7`; grouped top-head mean `Δ\mathcal{L}_{div} = -2.418` vs matched-random `-0.062`.
- The Telugu temperature-sweep table's auxiliary `E_ak` and CER columns are indeed sample-level quantities, not item-level aggregates. This is now stated explicitly in the caption.

### Supported but provisional
- The `4B` and writer-head Telugu follow-ups remain appendix-level and exploratory relative to the core `1B` intervention story. They support a same-pattern reading, but the strongest directly intervention-tested negative result is still the `1B` oracle-positioned static patch.

### Open questions
- Whether `v11` should now be treated as the canonical frozen submission draft for packaging, with any remaining work limited to metadata / submission assembly / bibliography spot-checking.

## 2026-04-09 — ICML v12 method-label / sign-semantics pass

### Established
- The Telugu mechanistic summary no longer needs the undefined label `mediation`; the visible evidence stack in the manuscript is adequately described as patching panels plus donor-replacement follow-ups, with exploratory writer-head screens reported separately.
- The writer-head appendix is easier to audit when the grouped comparison is normalized into a regime-aligned effect:
  - `1B`: top `+4.05` vs random `-1.54`
  - `4B`: top `+2.42` vs random `+0.06`
- The introduction's top-level Hindi/Telugu contrast is clearer when the `1B` negative intervention result is detached from the `4B` `L34` follow-up clause.

### Supported but provisional
- The writer-head appendix remains exploratory. The regime-aligned normalization improves readability, but it does not upgrade those head-level results into the same evidentiary tier as the direct `1B` late-site intervention result.

### Open questions
- Whether `v12` should now be treated as the canonical frozen workshop draft, with any remaining work limited to bibliography spot-checking, file naming, and submission assembly.

## 2026-04-09 — ICML v13 appendix consistency / writer-score definition pass

### Established
- The `4B` late-site appendix row now includes the no-op value that was already present in the underlying final-site sufficiency artifact (`0.00`), so the caption and row content are aligned.
- The writer-head table now defines the ranking quantity explicitly: for a head `h`, aligned writer score is the sign-aligned mean change in first-divergence gold-vs-bank gap under single-head ablation over the usable evaluation subset.
- The `38` usable items in the writer-head probe come from the first 40 evaluation words after excluding cases where the first-divergence comparison is undefined.

### Supported but provisional
- The writer-head appendix is now operationally interpretable, but it remains exploratory and should still be read below the evidentiary tier of the direct late-site intervention-style follow-ups.

### Open questions
- Whether `v13` should now be frozen as the canonical submission draft, with any remaining work limited to bibliography / metadata spot-checking and submission assembly.

## 2026-04-09 — ICML v14 final appendix layout polish

### Established
- No new technical content changes were needed after the latest review; the remaining issue was purely presentational.
- Switching the writer-head appendix table to non-floating placement consolidates both `A.6` tables onto one appendix page and removes the previously sparse final page.

### Supported but provisional
- A tiny overfull-box warning remains on the compact late-site appendix table, but visual inspection shows the table renders cleanly and the issue appears cosmetic rather than reader-facing.

### Open questions
- Whether to freeze `v14` as the canonical submission draft and proceed to final packaging / submission assembly.

## 2026-04-09 — ICML v15 regime-map calibration pass

### Established
- The manuscript no longer overclaims Bengali/Tamil positions in the two-axis regime map relative to the stage-sensitive evidence shown on the page.
- The five-language behavioral figure is now framed explicitly as a broad phase map, with direct axis-defining evidence limited to Hindi, Marathi, and Telugu via Table 1.
- The Telugu mechanistic paragraph now maps evidence sources more cleanly: the `L34` late-site claim points to the late-site appendix table, while the upstream writer-head statement points to the writer-head appendix table.

### Supported but provisional
- Bengali and Tamil still contribute to the broader qualitative picture through the five-language behavioral summary, but the paper now avoids treating them as equally stage-resolved with Hindi/Marathi/Telugu.

### Open questions
- Whether `v15` should be frozen as the canonical final submission draft, with any remaining work limited to submission packaging and file naming.

## 2026-04-10 — canonical final freeze

### Established
- A fresh skeptical pass on `v16` found the repeated Bengali/Tamil overreach and Telugu citation-scope complaints to be stale relative to the live manuscript.
- `v16` has been frozen as the canonical review/submission target:
  - source: `Paper Template and Paper/Paper/icml2026/gemma_1b_icl_paper_submission_8plus2_icml2026_final.tex`
  - PDF: `Paper Template and Paper/Paper/icml2026/gemma_1b_icl_paper_submission_8plus2_icml2026_final.pdf`

### Supported but provisional
- A tiny overfull-box warning remains on the compact late-site appendix table, but visual inspection suggests it is cosmetic rather than reader-facing.

### Open questions
- Remaining work is submission assembly and any final external metadata spot-check, not further manuscript-content revision.

## 2026-04-10 — ICML v17 implementation-specification and audit-strengthening pass

### Established
- The latest review was partly live and partly stale: the most substantive remaining gaps were implementation specification, figure notation consistency, and direct auditability of already-run results rather than missing new science.
- The manuscript now specifies the akshara segmentation procedure used for `CER`, `E_ak`, and `CER_tail` directly in the metrics section.
- The Telugu oracle-patch protocol is now explicit about teacher-forced shared-prefix conditioning, the `191/200` usable-item filter, and the fact that the usable shared prefix already contains the gold first akshara in `191/191` cases.
- The Hindi localization section now includes direct last-token sweep numbers and explicit per-condition prompt-final alignment wording.
- Figure 5 no longer uses the undefined `Δentry` notation; it now aligns with `ΔE_ak` / first-akshara correctness.
- The Telugu temperature-sweep interpretation is now strengthened by an item-level `any exact bank copy` view (`0.421 -> 0.614 -> 0.719`), not only the sample-level summary already in the table.
- `v17` has been promoted into the canonical final manuscript file.

### Supported but provisional
- The remaining manuscript risks are now mostly ordinary reviewer-preference questions about how much extra robustness work one might still wish to see, not clear manuscript-internal support defects.

### Open questions
- Whether to keep the current anonymous-source setup as-is for submission packaging, or to make anonymity intent even more explicit in the source comments before external handoff.

## 2026-04-10 — ICML v18 support/bookkeeping/hygiene pass

### Established
- The latest review's strongest remaining manuscript-internal issues were support calibration and bookkeeping, not missing new experiments.
- The Hindi-versus-Telugu intervention contrast is now explicitly framed as a **non-matched intervention comparison** rather than as a controlled apples-to-apples tractability test.
- Figure 3 is now described honestly:
  - Hindi/Bengali/Tamil/Telugu cells are three-seed means
  - Marathi is a seed-42 targeted same-script diagnostic row, not a three-seed aggregate
- The paper now exposes cell-wise seed ranges for Figure 3 and a compact panel/split summary table for the main text and appendix evidence.
- The Hindi patch-tuning objective is now explicit and reproducible in-paper (`alpha` maximizes mean `ΔL`; `p_target` tie-break only).
- The canonical submission source is now source-anonymous as well as PDF-anonymous.
- `v18` has been promoted into the canonical final submission file.

### Supported but provisional
- The paper is now stronger against reviewer attacks about bookkeeping, notation, and overclaimed cross-panel comparisons, but it still remains bounded by the actual intervention family explored for Telugu and by the lack of mechanistic cross-family replication.

### Open questions
- Remaining work is submission packaging only; no further manuscript-internal science or specification gap is currently the bottleneck.

## 2026-04-10 — ICML v19 review-integration pass

### Established
- The new April 10 review's live manuscript issues were all about claim calibration, tensor-bridge clarity, and figure evidence presentation, not missing algebra or broken implementation.
- The Telugu negative result is now summarized consistently as a **shared-prefix-conditioned oracle diagnostic** rather than as a directly parallel full-task rescue comparison.
- The Hindi mechanistic story now explicitly bridges the `L25 MLP output` localization result to the deployed edit on the same block's pre-down-projection channel basis.
- Figure 3 now visibly marks Marathi as a seed-42-only diagnostic row, and the surrounding prose now says `seed ranges` rather than `uncertainty ranges`.
- The cross-family contribution no longer uses the unsupported `capability floor` phrasing.
- The prompt-composition ablation is now described as suggestive / consistent-with rather than as ruling out generic order effects.
- `v19` has been promoted into the canonical final submission file.

### Supported but provisional
- The paper is now more review-proof against protocol-comparability and evidence-presentation criticisms, but the remaining limitations are still real scientific ones: non-matched Hindi/Telugu edit families, single-seed prompt-composition follow-up, and behavioral-only cross-family checks.

### Open questions
- No new manuscript-internal issue emerged from this review pass that justifies more paper churn before submission packaging.

## 2026-04-10 — ICML v20 final review-integration pass

### Established
- The Telugu divergence-step notation is now unambiguous: $t_{\mathrm{pref}}$ is the final shared-prefix token (where the patch is applied), and $t_{\mathrm{div}} = t_{\mathrm{pref}} + 1$ is the first divergence token (where logits are read).
- The Hindi $E_{\mathrm{tok}}$ / $E_{\mathrm{ak}}$ distinction is now explicitly flagged at the start of the mechanistic subsection to prevent reader confusion.
- Marathi is now consistently framed as a same-script, different-language probe whose evidence is consistent with but does not uniquely isolate the shared-script hypothesis.
- The literature-positioning claim is now domain-scoped rather than making a broad absence claim about prior work.
- Appendix Table A.2 is now readable and no longer produces an overfull hbox warning.

### Supported but provisional
- The paper's main findings are unchanged; these fixes are about implementation clarity and claim calibration, not new science.

### Open questions
- No further manuscript-internal issue is blocking submission packaging.

## 2026-04-10 — ICML v21 final calibration pass

### Established
- The Telugu oracle-patch panel is now described in a stage-matched way: continuation-only outcomes are primary because the shared prefix is oracle-supplied on all 191 usable items.
- The Telugu negative result remains genuinely negative under those continuation-only outcomes: continuation exact match stays at 0.0 and continuation CER does not improve.
- `Δℒ_div` is now explicitly framed as a single-competitor diagnostic, not a competitor-agnostic continuation metric.
- The Hindi `prompt-state routing` interpretation is now claim-bounded to what the current mechanistic evidence actually isolates.
- The `4B` Telugu `L34` wording is now consistently described as bounded strongest-among-tested-sites evidence rather than as a full scale-parallel localization.

### Supported but provisional
- A stronger Telugu competitor-robustness claim would still require additional robustness analyses (e.g. top-k or actual-copied-exemplar variants), which were not added here.
- A stronger Hindi prompt-state interpretation would still require a corrupt-condition mechanistic control, which was not added here.

### 2026-04-11 — v21 scope/reproducibility tightening
- Established:
  - The manuscript now states the strongest honest scope more clearly: broad multilingual behavioral mapping, with direct mechanistic localization only for Hindi and Telugu.
  - The Hindi activation-patching equation is now implementation-aligned rather than suggestive of shared-position replacement.
  - The corrupt-control discussion now matches the actual evidentiary role it plays in the paper.
- Supported but provisional:
  - A stronger methodological use of the corrupt condition would still require additional stage-sensitive or mechanistic corrupt-condition panels.

### 2026-04-11 — v22 indexing and diagnostic-notation cleanup
- Established:
  - The paper's previous `L26 layer_output` / `L34 layer_output` shorthand was vulnerable to off-by-one misreading; main-text terminology now uses explicit residual-boundary names instead.
  - Gemma~3 `4B` depth in the paper is now aligned with the architecture packet (`34` layers, not `42`).
  - The Hindi activation-patching equation is now explicit enough for reproduction without relying on prose to override the index notation.
  - The `t_{\mathrm{pref}}` / `t_{\mathrm{div}}` definition is now implementation-facing and object-based rather than relying on an ambiguous phrase like `index of the final shared target-token prefix`.
- Supported but provisional:
  - The appendix still preserves some raw artifact labels for auditability, but the main-text interpretation is now much harder to misread.
