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

## 2026-04-15 (ICML v28/v29 review-calibration passes)
- Worked: treating each fresh review as a live manuscript audit rather than as a prose-only memo immediately separated already-fixed points from genuinely live ones.
- Worked: the stored item-level Telugu follow-up artifacts were enough to quantify the competitor-definition mismatch directly (`8/24` and `22/81`) without rerunning experiments.
- Worked: the stored Hindi held-out patch-eval item rows were rich enough to answer the token→akshara bridge criticism one level better than before; a simple item-level bridge (`corr = 0.40`, rescued-vs-non-rescued `ΔL` split) was much more useful than repeating the earlier high-level caveat.
- Worked: when a metric name is slightly cleaner than the implementation (`target-vs-Latin`), renaming it to the literal implementation-facing version is often better than defending the cleaner label.
- Slowdown: `tectonic` was flaky when writing PDFs directly back into the mounted workspace; compiling to `/tmp` and then copying the validated PDF back in Python chunks was the reliable workaround.
- What should change in prompts: future late-cycle manuscript audits should explicitly ask ``which reviewer claims can be answered from existing item-level artifacts?'' before opening any new experimental branch.
- What should change in repo structure: keep a tiny reproducible manuscript-audit note or script for derived reviewer-facing counts like competitor top-1 match rates and token→akshara bridge statistics, so these numbers do not have to be recomputed ad hoc in chat.
- Reusable lesson candidate: ``if a reviewer questions a heuristic competitor definition, first compute heuristic-top1 versus actual-copied match rates from existing exact-copy artifacts before proposing new localization reruns.''
- Reusable lesson candidate: ``if a reviewer says the mechanistic readout is only partially aligned with the behavioral axis, look for per-item held-out intervention files before proposing new experiments; often a lightweight bridge analysis is already possible.''

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
- Worked: after the threshold sweep weakened the visibility-only story, the best next move was not another big model run; a cheap re-analysis of existing raw control traces was enough to reveal that `1B Hindi` and `1B Telugu` are failing at different stages.
- Worked: adding a small reproducible analysis script (`experiments/analyze_loop2_failure_modes.py`) was a better investment than continuing to do one-off notebook-style inspection in chat.
- New lesson: once exact-match behavior splits by language, do not keep talking about a single `1B` mechanism. Separate the candidate failure modes before proposing interventions.
- Worked: the first-token competition audit was cheap, decisive, and fast enough to run interactively on the VM; it confirmed `1B Hindi` as an early-stage problem and ruled out a first-token explanation for `1B Telugu`.
- Worked: the next smallest useful step after that was not another VM run but a local rank-analysis of copied prompt-bank targets, which showed a nearest-neighbor-style retrieval tendency for `1B Telugu`.
- Prompt lesson: when a behavioral artifact already stores per-item prompt-bank metadata, exploit it before scheduling another model run.
- Worked: the Telugu retrieval-condition comparison was answerable entirely from existing artifacts; no new GPU time was required.
- New lesson: once you suspect bank-copy errors, compare `helpful`, `similarity_desc`, `similarity_asc`, `reversed`, and `corrupt` on the same fixed items before reaching for mechanistic patching. Those condition contrasts can already distinguish nearest-neighbor retrieval from generic bank-copy pressure.
- Strategic lesson: if the work starts to feel too narrow or mediocre, do a first-principles reframing review before expanding the experiment matrix. In this case, the better move was to widen the **question** (ICL regimes: copying vs retrieval vs composition) rather than merely widen the **benchmark**.
- Follow-on lesson: once the broader question is chosen, re-expand deliberately rather than everywhere at once. Here that means a 4-language confirmatory panel first, not an undifferentiated many-language sprawl.
- Spec discipline matters: once the project identity changes, rewrite `research/spec.md` early so future experiments are judged against the new thesis and not stale rescue/degradation language.
- Reviewer discipline matters too: if a result would not convince a skeptical reviewer without raw examples, build a manual-audit helper immediately instead of hoping future prose will rescue it.
- Worked: directly inspecting the installed `transformers` source on the VM was the fastest way to resolve the Gemma-3 logit-lens bug; guessing from generic transformer intuition would have kept the localizer wrong.
- New lesson: when a mechanistic readout disagrees sharply with an already trusted behavioral audit, first check whether the model family's final hidden-state contract differs from the generic "apply final norm then unembed" assumption.
- Repo/process improvement: future mechanistic localizers should default to the exact audited split and prompt-construction path first, then simplify only after they reproduce the trusted behavioral endpoint.
- Worked: the first Hindi patch panel failed informatively. Near-zero query-position effects forced a re-check of the causal locus, which revealed that next-token competition must be patched at the **last prompt position** for a direct test.
- New lesson: for generation-time first-token analyses, late same-position patching at the query token and late same-position patching at the final prompt token answer different questions. The former is about mediation from source-token state; the latter is the direct readout-state intervention.
- Worked: once the patch locus was corrected to the last token, the signal became strong enough to localize a cleaner implementational site (`layer 25 MLP output`) rather than just a broad late band.
- Worked: the held-out selection-split → eval-split subspace patch test made the Hindi `L25 MLP` claim sharper. It preserved the site-level bottleneck while ruling out a naive "tiny raw-neuron subset" story.
- New lesson: raw hidden coordinates can be mechanistically misleading even when the site is correct. In this case, small absolute-ranked coordinate sets were harmful, and only the sign-split follow-up revealed that the cleaner small-k rescue direction lives in the **negative-signed** raw subspace.
- Prompt for future agents: when a dense patch is clean but top-k raw-coordinate patches look paradoxical, do not immediately conclude the site is wrong; test **sign-split** and **better-basis** hypotheses before abandoning the localization.
- New lesson: when a pair or subset looks strongest on the eval panel, do not promote it directly to the main claim. Add a fast **selection→eval** confirmation pass immediately so the final statement is not contaminated by eval-side subset choice.
- What worked: moving from dense patching -> held-out channel-basis patching -> subset/pair/singleton decomposition produced a much cleaner mechanistic story than trying to interpret raw coordinates directly.
- What worked: forcing selection->eval separation before trusting the pair result prevented an avoidable overclaim.
- What to keep: when a candidate pair looks strong, test explicit singletons before calling it synergy; in this case the better wording is a two-channel core with asymmetric contributions, not a proved minimal circuit.
- Another useful pattern: after finding a small causal channel set, separate **state shift magnitude** from **local sensitivity**. The value audit alone would have been misleading because a strongly shifting negative-control channel (`6015`) was almost causally inert; the sensitivity audit explained why.
- Good stopping rule: once the story has site localization, model-native basis, held-out controls, singleton/pair decomposition, and a readout-geometry explanation, stop digging and move to the comparison branch.

## 2026-03-29 — larger-N four-cell pass
- What worked: repeating the stage-localizing audits at `N=100` materially reduced uncertainty without needing a large new method build.
- What failed / stayed weak: Hindi nearest-bank continuation is not as clean a stage-isolation probe as Telugu because first-token overlap with the competitor is only partial.
- What to reuse: when a 30-item mechanistic story looks promising, rerun the same stage-localizing instrument at larger `N` before opening a new mechanistic branch.

## 2026-03-29 — Telugu component localization bring-up and full pass
- What worked: iterating through the launcher/runtime bugs quickly was worth it; by the time the smoke passed, the experiment body itself produced a useful answer rather than a null due to harness issues.
- What worked: treating `final_state` rows as readout confirmations prevented overclaiming them as site localizations.
- What worked: the full 30-item component panel was the right next experiment after the smoke. It clarified that Telugu is currently a broad late layer-output story, not a clean Hindi-style narrow-MLP story.
- What failed / slowed progress: repeated shell/path/wrapper mismatches in the new Telugu component launcher (`PATCH_PAIRS` quoting, 1-based vs 0-based layer indexing, `~`-prefixed out-root handling, custom 4B layer locator weaker than `core.get_model_layers(...)`).
- Reusable lesson: for new mechanistic launchers, prefer repo-native helpers (`core.get_model_layers`) over ad hoc structure discovery, and treat human-facing layer labels explicitly as a separate interface from internal indices.
- Reusable lesson: once a smoke panel succeeds, inspect the actual JSON immediately and promote only the strongest non-final-state rows; otherwise it is too easy to keep talking about final-state effects that are really just readout confirmations.
- Method lesson: if a component panel shows broad late `layer_output` effects but weak `mlp_output`, the safe reading is a distributed late-state story. Do not force a narrow-basis decomposition too early.

## 2026-03-29 — Marathi same-script control probe
- What worked: reusing the already-built first-token and continuation instruments on Marathi was the right low-cost way to test the broader hypothesis about Hindi vs Telugu vs language dominance.
- Main finding: Marathi does not simply collapse into either the Hindi bucket or the Telugu bucket. At `1B` it looks hybrid: stronger early routing than Hindi, but later continuation still degrades like a bank-attractor problem.
- Reusable lesson: once a mechanistic story starts sounding like a one-axis training-data story, run a same-script control before telling that story to yourself too often.
- Manual-audit value: inspecting the worst Marathi continuation cases showed concrete bank-like targets (`आदेशाप्रमाणे`, `आदर्शांकडे`, `आबासाहेबानी`) rather than relying only on the mean continuation gap.

## 2026-03-29 — sharpening the endgame synthesis
- What worked: using Marathi as a same-script control materially improved the scientific story; it constrained the explanation more than another broad rerun would have.
- Main synthesis gain: the current best story is now explicitly two-axis (early routing vs late continuation), not just “small model bad / large model good”.
- Wording discipline: cross-model claims should target regime classes more than exact mechanisms; exact bands/channels/sites remain Gemma-family evidence.

## 2026-03-30 — Telugu layer-band persistence probe
- What worked: instead of jumping straight to a harder path-patching story, a simpler multi-layer layer-output redundancy test gave a strong early answer about the shape of the late-state story.
- Main finding from smoke: in both `1B` and `4B` Telugu, patching the latest late layer already matches the tested multi-layer combinations on the smoke slice.
- Method lesson: when many adjacent late layers all look causal, first test whether they are additive or just different readout points on the same carried state.

## 2026-03-30 — Telugu full layer-band confirmation
- What worked: scaling the redundancy probe from smoke to the full 57-item usable panels gave a much stronger answer than a new broad sweep would have.
- Main finding: the latest late layer (`L26` for `1B`, `L34` for `4B`) exactly matches the tested multi-layer combinations on the current panel, down to per-item patched outputs in the spot checks.
- Practical method lesson: when a late band looks broad, test whether the band is additive or simply different readable slices of one carried state before escalating to more exotic decomposition tools.

## 2026-03-30 — 1B Telugu mediation smoke
- What worked: once the layer-band redundancy result was in hand, the right next move really was a mediator-overwrite test rather than another broad localization pass.
- Main finding so far: the `1B Telugu` smoke already shows a clean mediation signature — earlier writer rescue is killed by overwriting the final late mediator, but not by overwriting an off-band site.

## 2026-03-30 — 4B Telugu mediation smoke
- What worked: the same mediation design generalized cleanly from the `1B` failure branch to the `4B` positive-control branch.
- Main gain: this reduces the chance that the mediation signature is a one-off artifact of the `1B` failure mode.
- Updated belief: the carried-state picture now looks like the right abstraction for both Telugu branches, with model size changing the sign/quality of that state more than the existence of the mediation structure itself.

## 2026-03-30 — 1B Telugu full mediation confirmation
- What worked: the mediation design scaled cleanly from smoke to the 57-item usable panel on the `1B` branch.
- Main implication: `1B Telugu` is now materially beyond a broad localization story; it has a credible mediated late-state explanation.
- Remaining honesty constraint: mediation and necessity are stronger than sufficiency, so wording should still stop short of “full circuit found.”

## 2026-03-30 — Telugu mediation full confirmation and sufficiency smoke
- What worked: the full mediation panel confirmed the smoke-level story instead of weakening it, which means the next uncertainty really is sufficiency rather than another localization pass.
- Method lesson: once mediation holds at larger `N`, the right next experiment is not more screening but a controlled sufficiency test against off-band and random-direction controls.

## 2026-03-30 — Telugu full sufficiency confirmation
- What worked: the final-site sufficiency panel provided the last compact stress test the reviewer-style critique asked for, without requiring a much heavier path-patching project.
- Main conclusion: the Telugu mediator is now supported not only as a necessary bottleneck but as a content-carrying dominant site relative to off-band and random controls.
- Important caution retained: dominant is not identical to exclusive; item-level exceptions and positive off-band effects still matter for honest wording.

## 2026-03-30 — practical patching and paper rewrite
- What worked: converting the bounded Hindi mechanism into a fixed-vector practical patch was the right final stress test; it produced a real held-out gain and made the paper story much stronger.
- What also worked: a small Bengali/Tamil first-token follow-up was cheaper than a new mechanistic branch but materially tightened the non-core language discussion.
- What remains annoying: LaTeX compilation could not be checked in-session because the environment has no TeX engine; future paper-writing sessions should either provision a TeX toolchain early or route preview through a dedicated rendering environment.
- The follow-up Hindi intervention panel was worth doing: it ruled out an oversimplified “just suppress the bad channels” story and strengthened the more precise signed-subspace interpretation.
- This was a good example of a small extra experiment materially improving the honesty of the final claim without reopening the whole project.
- The first Telugu practical steering smoke was worth doing even though it was not yet a win: it prevented premature symmetry with Hindi and exposed that the practical design problem is different on the late continuation side.
- Good lesson: for continuation interventions, naive local-gap optimization may not align well enough with end-to-end continuation quality; selection metrics may need to be closer to the actual generation objective.
- The full Telugu practical evaluation was scientifically useful even without a clean win: it sharpened the contrast between Hindi and Telugu and prevented an over-symmetric story.
- The project now has a better endgame shape because “mechanistically real but not simply editable” is a more interesting and more honest result than forcing a weak practical patch claim.
- Good endgame move: once the full Telugu practical run came back weak, switching from “keep trying to win” to “preserve the negative contrast and tighten the paper” improved the scientific story and likely the workshop readability.
- Figure polish was worth doing now rather than postponing everything to a final formatting pass, because it also forced a better plotting design for the Hindi practical result.
- Provisioning a small local Tectonic path was the right move; it closed a real paper-readiness blocker without forcing a heavier system-wide TeX install.
- Updating the canonical summary artifact in place was also the right choice, because it avoided proliferating user-facing markdown files late in the project.
- User feedback on the writing quality was correct. The earlier draft was scientifically careful but still too memo-like. Doing a real prose pass late was necessary, not cosmetic.
- Figure redesign was also worth more than minor styling tweaks; changing chart type and layout improved the story much more than palette changes alone.
- Excalidraw works well here for thesis communication, but only after treating it as a diagram system rather than as a plot system. The right split is now clear: Altair-like charts for quantitative evidence, Excalidraw for conceptual story structure.
- The easiest reliable export path was not MCP-tool integration inside this agent; it was a local Excalidraw canvas server plus headless Chrome and REST export. That is a useful reusable pattern.

## 2026-04-01 — reviewer follow-up completion, sync, and paper integration
- What worked: switching the unfinished VM jobs onto a real remote `tmux` queue was the right operational fix; it survived local terminal resets and produced the remaining artifacts cleanly.
- What worked scientifically: the external-family behavioral tranche improved the paper more than expected because it upgraded the story from Gemma-only behavior to cross-family behavioral support without forcing unsupported mechanistic claims.
- Important honesty gain: `Llama 3.2 1B` was useful precisely because it was not a clean replicate; it prevented an over-simple “all 1B models behave like Gemma 1B” story.
- The prompt-composition ablation was especially valuable because it refined the Telugu story rather than just confirming it. The key lesson is that similarity mass can help entry and hurt continuation at the same time, so the right abstraction is a distributed attractor basin rather than a single copied exemplar.
- Good workflow pattern: once reviewer-hardening experiments land, immediately sync them, write one canonical synthesis memo, and patch the paper before the stale wording becomes the new source of confusion.

## 2026-04-01 — end-to-end reverification and reviewer-fix hardening
- What worked: turning the final verification into a script (`experiments/reverify_end_to_end_artifacts.py`) was much better than doing ad hoc spot checks; it produced a durable local-vs-VM integrity record and forced a minimal oracle set.
- Important paper-writing lesson: reviewer comments often expose stale prose more than weak science. The biggest fixes here were harmonizing panel sizes/numbers, aligning methods text to the actual implementation, and correcting an outdated Telugu prompt-order sentence.
- Figure lesson: the two most valuable new plots were not more mechanism plots, but cross-family behavior and prompt-composition tradeoff plots. They improved the story because they compressed the reviewer-hardening evidence into readable geometry.
- Scope lesson: the right strengthened claim is “cross-family behavioral support with Gemma-only mechanism,” not “general mechanism across families.” The negative `Llama 3.2 1B` result materially improved the honesty of the paper.

## 2026-04-01 — alpha-grounding and literature-audit pass
- What worked: using `alpha get` and `alpha ask` directly on a small, claim-targeted paper set was much better than broad unguided search. It surfaced not only useful new citations (`RomanLens`, retrieval-based ICL survey) but also a real citation-mismatch bug: `learningwithouttraining` was being used as if it were a multi-phase paper.
- Important writing lesson: literature review should not just add more citations; it should sharpen the mapping from citation to claim. The best changes here were actually subtractions or separations of claims.
- Important reviewer lesson: bibliography-quality errors are easy to miss late in a project and look worse than they are. A dedicated lit-audit pass was worth it because it fixed several wrong author labels before delivery.
- Reusable pattern: for thesis-stage hardening, a good sequence is (1) artifact reverification, (2) alpha-based claim audit, (3) bibliography correction, (4) skeptical re-review, (5) recompile.

## 2026-04-01 — conference-style submission rewrite
- What worked: separating “science hardening” from “submission-style rewriting” was the right move. Once the claims were already tightened, a dedicated writing pass could focus on title, abstract, intro, visual signposts, and section naming instead of reopening the science.
- Claude was most useful as an editor, not as an authority: the strongest suggestions were about structure, density, and reviewer reading behavior rather than facts.
- The most effective presentation changes were simple ones: a contributions box, an early teaser figure, section-level takeaway lines, and removing revision-history phrasing.
- Reusable pattern: for research papers near closure, keeping a thesis-leaning draft and a submission-leaning variant in parallel can be better than forcing one file to serve both purposes immediately.

## 2026-04-01 — image-based verification and venue scan
- What worked: actually rendering the PDF pages to images and inspecting the screenshots caught a real figure-layout bug that plain text/compile logs did not make obvious. This should be a standard late-stage paper workflow.
- Important figure lesson: converting a plot to TikZ is not automatically an improvement. The first TikZ version had overlapping panels; the improvement came from visual verification plus a second geometry pass.
- Important venue lesson: the paper's content fit is strongest for interpretability-focused NLP venues and workshops; deadline timing matters as much as topic fit.
- Reusable pattern: for final paper polish, use the sequence compile -> screenshot pages -> inspect rendered pages -> fix -> re-render, rather than trusting the TeX source or parsed text alone.

## 2026-04-04 — final deliverables review retrospective

### What worked
- Re-reading the project synthesis markdown before polishing the report prevented drift back into older over-broad framings.
- `document_parse` screenshot inspection was genuinely useful for catching layout/pathology in newly added TikZ figures that a plain compile log would not catch.
- A short alpha refresh on the key papers was enough to tighten wording without reopening the science branch.

### What failed or slowed things down
- New TikZ summary figures looked fine in code but still overflowed on the rendered page/slide; visual inspection remains necessary.
- Beamer frame bodies plus inline TikZ style parameters (`#1`) caused macro-expansion issues; pushing reusable styles to safer forms or avoiding parameterized inline styles is more robust.

### What should change next time
- For future final-deliverable work, default earlier to a render-review loop: compile, screenshot, adjust, repeat.
- Keep first-principles summary slides/figures compact; dense explanatory text should usually stay in the report, not the slide canvas.

## 2026-04-04 — second final recheck note

### What worked
- Rendering the entire slide deck and then inspecting a contact-sheet view was a fast way to catch one last real layout bug that compile logs missed.
- Shortening text, not just resizing boxes, was the right fix for the summary slide.

### What remains annoying
- Beamer/fontspec under Tectonic still emits a few hard-to-localize `nullfont` digit warnings even when the rendered slides look normal.
- Report longtables continue to generate appendix box warnings even though the visible pages are acceptable.

## 2026-04-07 (Final honors report / presentation finishing pass)
- Worked: a line-by-line citation-order audit plus targeted prose cleanup was a much better final pass than continuing to do vague “polish” requests in chat. It made the deliverable measurably cleaner.
- Worked: a small scripted audit for labels/references caught many figure/table/equation ordering mistakes quickly and prevented manual drift.
- Worked: converting only the nonessential list-heavy blocks to prose preserved technical clarity while making the report read more like a thesis and less like notes.
- Worked: visual spot-checking compiled PDF pages after the edits was necessary; several layout concerns were easier to judge from screenshots than from TeX warnings.
- Slowdown: LaTeX warning streams are noisy. Overfull/underfull boxes in appendices and code blocks can dominate the log even when the main document is visually fine.
- What should change in repo structure: add a lightweight LaTeX QA helper for (1) pre-citation checks, (2) missing labels, and (3) likely overfull-table hotspots before the final compile pass.
- Reusable pattern candidate: “final thesis cleanup = scripted citation audit + prose de-bulleting + screenshot QA” would make a good local skill/example for future long-form writeup sessions.

## 2026-04-09 (ICML v7 reviewer-proofing pass)
- Worked: triaging the new review against the current `v6` manuscript before editing prevented stale reviewer complaints from reopening already-fixed issues.
- Worked: extracting the exact Telugu negative-patch, prompt-composition, safety-audit, and cross-family continuation-proxy numbers into a small reproducible summary script (`experiments/summarize_icml_v7_followups.py`) reduced the risk of hand-copying inconsistent numbers into the paper.
- Worked: the strongest remaining reviewer concerns were addressable without new experiments: tighten the early-axis language, quantify the Telugu null intervention, and expose prose-only follow-ups in appendix tables.
- Slowdown: several review variants were concatenated together and referenced older filenames (`v3`, `submisson.tex`), which made it easy to chase already-resolved issues unless every point was checked against the live draft.
- What should change in prompts: when the user pastes a review memo, default to an explicit "live issues vs stale issues" triage before editing.
- What should change in repo structure: keep small follow-up-summary scripts near the paper endgame so appendix numbers can be regenerated quickly from canonical JSON outputs.

## 2026-04-09 (ICML v8 reviewer-proofing pass)
- Worked: instead of treating the review's request for a direct late-axis metric as requiring new experiments, checking the old evaluation code revealed that `continuation_fidelity` already implemented the right conditioned continuation metric. That turned a potentially expensive branch into a bounded manuscript fix.
- Worked: recomputing the stage-sensitive table values directly from the raw JSON artifacts was worthwhile; it confirmed the existing Hindi/Marathi/Telugu bank-copy values and gave a reproducible basis for adding `CER_tail`.
- Worked: the `k`-shot criticism exposed a more important issue than missing appendix material: the main-text numbers should match the actual stored sweep artifacts. Recomputing from the JSON and then aligning both the prose and a new appendix table avoided carrying forward stale or hard-to-trace numbers.
- Slowdown: review-driven paper passes can easily drift into patching prose around unverified remembered numbers. Writing a tiny summary script first was much safer than editing the TeX from memory.
- Reusable pattern candidate: when a reviewer asks for a more direct metric or appendix display, first check whether the current raw artifact schema already contains the needed quantity before launching new experiments.

## 2026-04-09 (ICML v9 comparability / reporting pass)
- Worked: the latest review was worth treating as live even though it was narrow. The oracle-position issue was real and easy to understate in the abstract if left unfixed.
- Worked: adding conditional-sample counts directly into the stage table was a high-leverage fix. It improved reviewer trust without requiring a new experiment or a large layout change.
- Worked: computing bootstrap intervals into a supporting artifact was a good compromise. It gave a quantitative check on noise without overloading the already-dense main-text table.
- Lesson: when a metric is conditional on another event, report the effective denominator somewhere visible. Otherwise even a correct number will feel under-specified to a reviewer.

## 2026-04-09 (ICML v10 temperature-sweep anchoring pass)
- Worked: the review caught a real reader-trust problem rather than a science problem. The temperature-sweep numbers were correct, but they were not visually anchored to a panel, so they could be misread as contradicting the 30-item stage table.
- Worked: adding a tiny appendix table was the cleanest fix. It preserved the main-text flow while making the follow-up auditable in the same style as the other appendix summaries.
- Lesson: if a follow-up number differs materially from a nearby headline table, name the panel and the denominator immediately, even when the number itself is already correct.

## 2026-04-09 (ICML v11 Telugu follow-up evidence pass)
- Worked: the latest review again surfaced a real evidentiary-presentation issue rather than a flaw in the science itself. The `L26/L34` shorthand had become cleaner than the visible support behind it.
- Worked: appendix tables were the right repair. They made the `4B` same-pattern claim and the writer-head references auditable without dragging the main text into another dense methods detour.
- Worked: clarifying the temperature-sweep unit of analysis in the caption was a low-cost, high-trust fix. The numbers were already correct; the missing piece was telling the reader what kind of average they were looking at.
- Lesson: once a manuscript reaches late-stage polish, the main failure mode is often not a wrong number but an unanchored number or an over-compressed summary phrase. At that stage, tiny appendix tables can buy a lot of reviewer trust.

## 2026-04-09 (ICML v12 method-label / sign-semantics pass)
- Worked: the latest review was again about reader trust, not missing science. Matching the method labels to the actual visible analyses was a straightforward but worthwhile calibration fix.
- Worked: sign-normalizing the grouped writer-head comparison was better than adding another paragraph of explanation. A tiny representation change made the appendix more readable than prose alone would have.
- Lesson: when an appendix comparison needs a model-specific sign convention, normalize it into a common human-readable direction if possible. That is usually better than asking the reader to mentally invert one row.

## 2026-04-09 (ICML v13 appendix consistency / writer-score definition pass)
- Worked: the latest review again found a real trust issue in the appendix rather than a science flaw. Fixing the caption-row mismatch and defining writer score closed two easy-to-question gaps.
- Worked: splitting the Telugu mechanistic summary into direct late-site evidence versus exploratory writer-head follow-up made the evidence hierarchy more visible without changing the substance.
- Lesson: once appendix tables start carrying important review-proofing work, even small caption/row mismatches matter more than they normally would. At that stage, table semantics need the same rigor as equations.

## 2026-04-09 (ICML v14 final appendix layout polish)
- Worked: once the draft was technically stable, a small float-placement change delivered a visibly cleaner appendix with almost no risk to content.
- Lesson: after substantive review issues are exhausted, the right final move is often not more wording but one conservative layout fix that improves the paper's finish without perturbing claims.

## 2026-04-09 (ICML v15 regime-map calibration pass)
- Worked: the last live issue was again claim calibration, not missing data. Softening the Bengali/Tamil language was better than inventing appendix evidence that the paper did not need.
- Worked: separating the late-site citation from the writer-head citation made the Telugu mechanistic section easier to audit with almost no cost.
- Lesson: once a paper is nearly done, the most valuable remaining edits are often the ones that align the granularity of claims to the granularity of the evidence actually displayed.

## 2026-04-10 (canonical final freeze)
- Worked: the repeated rereview issue was partly process, not just prose. Creating one canonical final source/PDF should reduce the chance that later review passes target stale versioned drafts.
- Lesson: once a version chain gets long, freezing a single non-versioned review target is worth it. Otherwise reviewers and scripts can keep finding already-fixed issues in older files.

## 2026-04-10 (ICML v17 audit-strengthening pass)
- Worked: this review was useful once decomposed. The strongest move was not to obey every sentence mechanically, but to isolate the real remaining manuscript debts: implementation specification, figure notation drift, and protocol auditability.
- Worked: existing artifacts were rich enough to answer the hard reproducibility questions without rerunning experiments. The temperature item-level summary and the Telugu `191/191` boundary-alignment fact were both already latent in stored JSON.
- Lesson: late-stage paper reviews often ask for more "science" when the actual missing piece is sharper exposure of the science already done. Reproducibility scripts and small appendix clarifications can convert a weak-accept review into a stronger one more reliably than opening new experiments.

## 2026-04-10 (ICML v18 support/bookkeeping/hygiene pass)
- Worked: the latest review was most useful when treated as a pressure test on the paper's evidence ledger. The best fixes were support calibration, panel bookkeeping, and source-level submission hygiene, not new computation.
- Worked: a deeper audit surfaced one real hidden issue the review did not call out directly: Figure 3 was described as three-seed for all five languages even though Marathi was actually a seed-42 diagnostic row. Fixing that made the paper more honest and more review-proof.
- Lesson: when a paper mixes broad maps, targeted diagnostics, and held-out interventions, the split ledger itself becomes part of the scientific result. If the panel relationships are not visible, reviewers will correctly treat the paper as weaker than the underlying work actually is.

## 2026-04-10 (ICML v19 review-integration pass)
- Worked: the best response to this review was not to add new analyses, but to tighten the paper's interface between evidence and reader inference. The biggest gains came from naming the Telugu protocol honestly, showing the Hindi tensor bridge explicitly, and making the figure visually encode Marathi's weaker support.
- Worked: a tiny visual cue (`Marathi†`) did more to fix the comparability problem than another paragraph alone would have. This is a good reminder that figure semantics are part of the evidence ledger.
- Lesson: when a reviewer says a result is easier to read than to justify, the fix is often to make the task interface or tensor interface more explicit, not to rewrite the conclusion more aggressively.

## 2026-04-10 (ICML v20 final review-integration pass)
- Worked: the most important fix was the Telugu $t_{\mathrm{pref}}$ / $t_{\mathrm{div}}$ notation, which was a real implementation-facing ambiguity that could have caused a reproduction failure. Introducing two indices and keeping them separate clarified exactly which hidden state is patched and which logits are measured.
- Worked: adding a metric-switch note in the Hindi section directly addresses the most likely source of reader confusion when comparing prose numbers to Table 1.
- Lesson: when a notation defines both an intervention site and a measurement location, check that the prose description is consistent with the equation's index. In autoregressive generation, "divergence token" versus "hidden state used to predict divergence token" are one step apart, and that difference matters for implementation.

## 2026-04-10 (ICML v21 final calibration pass)
- Worked: the latest review was most useful where it forced a tighter match between evaluation design and summary metrics. Making continuation-only outcomes primary for the Telugu oracle patch materially improved the honesty and readability of the contrast.
- Worked: demoting `4B L34` from an apparently scale-general localization claim to `strongest tested site in a bounded follow-up` preserved the real finding while removing a reviewer-facing overstatement risk.
- Lesson: when a panel is oracle-conditioned or otherwise partially constrained, make the stage-matched outcome primary in the prose even if broader whole-word metrics are also available. Reviewers notice when the evaluation target and the reported metric do not line up cleanly.

## 2026-04-11 (ICML v21 scope/reproducibility tightening)
- Worked: narrowing the top-level framing improved the paper without weakening the real contribution. The manuscript is stronger when it says plainly that the multilingual map is behavioral and the mechanistic story comes from two case studies.
- Worked: the Hindi patch equation was indeed under-specified for reproduction; making the donor/recipient positions explicit was a high-value low-cost fix.
- Lesson: when a review says "this claim is broader than the evidence," the best response is often not to defend the sentence but to rewrite the paper around the actual evidence boundary.

## 2026-04-11 (ICML v22 indexing and notation cleanup)
- Worked: this review surfaced a genuinely important issue that previous rounds had missed — the paper's text had become more careful than the underlying site-label notation. Replacing ambiguous `L26/L34` shorthand with explicit residual-boundary names materially improved reproducibility.
- Worked: checking the repo state before acting mattered. The filename-typo complaint was not live in the current canonical lineage, so there was no reason to churn filenames just because a review mentioned them.
- Lesson: in mechanistic papers, notation drift around layer labels and boundary states is as dangerous as claim overstatement. Readers implementing interventions will follow the math and labels literally, so site naming has to be as reproducible as the numbers.
