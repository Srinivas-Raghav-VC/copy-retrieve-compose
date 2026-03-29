# spec.md — Cross-scale in-context transliteration as a microscope for copying, retrieval, and composition

_Last rewritten: 2026-03-29_

---

## 1. Working title

**From Copying to Composition: Cross-Scale In-Context Transliteration in Gemma 3**

Alternate title:

**Multilingual Transliteration as a Model Organism for In-Context Learning**

---

## 2. Thesis-scale research question

### Primary question

How does in-context learning move Gemma 3 models between **source copying**, **prompt-bank retrieval**, and **query-specific composition** in multilingual transliteration, and how do those regimes change with scale?

### Secondary questions

1. Which parts of the computation fail first when a model does **not** successfully use in-context examples?
2. Are the `1B` failures explained mainly by context visibility, or by different internal algorithms across languages?
3. Does a larger model (`4B`) escape the same failure modes by better early routing, better later continuation, or both?
4. How much of the behavioral story generalizes across a paper-grade four-language panel rather than only Hindi/Telugu anecdotes?
5. Can we produce at least one bounded mechanistic explanation that survives strong behavioral and causal scrutiny?

---

## 3. Why this is the right problem

This project is **not** mainly about transliteration as an application benchmark.

The stronger framing is that multilingual transliteration is a **clean experimental system** for studying in-context learning because it is:

- cheap to run,
- easy to score deterministically,
- easy to manually audit,
- multilingual,
- and rich enough to separate **early token routing** from **later whole-word continuation**.

That makes it a good microscope for a more general scientific question:

> when a language model sees many examples in-context, when does it behave like copying, retrieval, or composition?

This is honors-first and understanding-first. The goal is the strongest honest story, not a pre-selected mech-interp narrative.

---

## 4. Current evidence snapshot

## Established

1. `4B × Telugu × n_icl=64` is the cleanest current positive anchor.
2. `1B × Hindi × n_icl=64` is the clearest current fragility anchor.
3. The `1B` failure story is **not unitary** across languages.
4. Visibility contributes to the `1B` vs `4B` contrast, but a simple visibility-only explanation is inadequate.
5. The project now has enough behavioral structure to justify **narrow mechanistic work**.

## Supported but provisional

1. `1B × Hindi` shows **substantial early routing failure** under high-shot ICL.
2. `1B × Telugu` shows **mostly later retrieval/composition failure**, often producing a wrong prompt-bank target after getting the first token mostly right.
3. `1B × Telugu` bank-copy failures are often concentrated in the query's nearest-neighbor region under helpful ordering.
4. `4B` gains generalize beyond Hindi/Telugu to additional languages in the bounded breadth check.
5. `1B` weaknesses also generalize beyond Hindi/Telugu, though not in a perfectly identical form.

## Retired or downgraded

1. Retired as the primary thesis: a narrow rescue/degradation story centered on one model and one threshold.
2. Retired as the primary explanation: pure visibility-threshold failure for `1B`.
3. Downgraded: any single-mechanism story for all `1B` languages.
4. Downgraded: any claim that judge scores or SAE/transcoder metrics alone should carry the scientific burden.

---

## 5. Working hypothesis map

## H1 — regime-shift hypothesis

As scale increases, Gemma 3 moves from weaker or unstable high-shot behavior toward more reliable **query-specific composition**.

### Observable predictions

- `270M` stays near the capability floor.
- `1B` enters mixed regimes where some tasks look like copying or retrieval rather than composition.
- `4B` more often converts in-context examples into correct task-conditioned outputs.

### Current status

**Supported but provisional.**

---

## H2 — Hindi early-routing failure hypothesis

For `1B × Hindi × n_icl=64`, the main failure begins early: high-shot ICL worsens the competition for the **first target token**, often toward Latin/source-like alternatives.

### Observable predictions

- first-token target probability falls under helpful high-shot ICL relative to zero-shot,
- top-1 token often becomes Latin/source-like,
- corrupt high-shot prompts are similarly harmful, implying a prompt-state/routing pathology more than useful content-specific retrieval.

### Current status

**Supported but provisional**, and already strong enough for targeted mechanistic localization.

---

## H3 — Telugu late-retrieval failure hypothesis

For `1B × Telugu × n_icl=64`, the model usually fixes the first-token stage but then fails later by emitting a wrong **prompt-bank target**, often from a similar example.

### Observable predictions

- helpful ICL greatly improves first-token correctness relative to zero-shot,
- exact-match still stays poor because later continuation collapses,
- copied targets cluster near the query's similarity neighborhood under helpful ordering,
- alignment-breaking conditions preserve generic bank-copy pressure but weaken true nearest-neighbor concentration.

### Current status

**Supported but provisional**, but it still needs one more clean localization step before heavy causal claims.

---

## H4 — positive-control composition hypothesis

`4B × Telugu × n_icl=64` is not just better because of first-token activation; it also continues more correctly after the first token, escaping the prompt-bank-copy regime that traps `1B`.

### Observable predictions

- first-token success is already high,
- residual helpful-vs-control gains localize to later continuation rather than the first token,
- wrong outputs are mostly near misses rather than exact prompt-bank copies.

### Current status

**Supported but provisional**, and suitable as the main positive control.

---

## H5 — visibility-is-partial hypothesis

Architectural/local-window visibility matters, especially in the `1B` vs `4B` contrast, but it is only one contributor.

### Observable predictions

- `1B` loses more prompt-bank visibility than `4B` at `n_icl=64`,
- but reducing `n_icl` alone does not fully restore `1B` exact-match gains,
- and some `1B` regimes improve soft metrics even as visibility worsens.

### Current status

**Established as a partial constraint; retired as a full explanation.**

---

## 6. Claim-discipline table

| Claim | Status |
|---|---|
| Cross-scale transliteration behavior is structured rather than noise | established |
| `4B Telugu` is the strongest current positive anchor | established |
| `1B Hindi` is a clean high-shot fragility anchor | established |
| `1B` failures differ materially across languages | established |
| `1B Hindi` is mainly an early-routing failure | supported but provisional |
| `1B Telugu` is mainly a late retrieval/composition failure | supported but provisional |
| `4B` operates in a more compositional regime than `1B` | supported but provisional |
| visibility alone explains the `1B` failure regime | retired |
| a single unified `1B` mechanism explains all failures | retired |
| we will recover a full transliteration circuit | speculative and not a project promise |

---

## 7. Confirmatory scope

## Main language panel (current confirmatory 4)

- `aksharantar_hin_latin`
- `aksharantar_tel_latin`
- `aksharantar_ben_latin`
- `aksharantar_tam_latin`

### Why these four

- `Hindi` and `Telugu` are already behaviorally opened.
- `Bengali` and `Tamil` widen both script and language-family coverage.
- They were cleaner than `Marathi` in the first bounded expansion.

## Reserve language

- `aksharantar_mar_latin`

`Marathi` remains valuable as a same-script control relative to Hindi, but it is reserve rather than part of the main confirmatory four-language panel.

## Model scope

### Main mechanistic panel

- `1B`
- `4B`

### Cross-scale floor / synthesis panel

- `270M`
- `1B`
- `4B`

Use `270M` as the floor / capability-boundary comparison in the final story, even if the richest helpful-vs-control grid is concentrated on `1B/4B`.

---

## 8. Measurement philosophy

## Hard anchors

These are the primary sources of truth:

- exact match,
- akshara-aware CER,
- script validity,
- source-copy rate,
- prompt-bank-copy rate,
- nearest-neighbor copy diagnostics,
- first-token correctness / probability / competitor gap,
- manually auditable raw outputs.

## Soft labels

These are useful but secondary:

- local judge for acceptable transliteration variants,
- local judge for failure taxonomy,
- human-vs-judge agreement tables.

## Rule

Deterministic metrics and manually auditable outputs remain the **hard anchor**. Judges help with richer labeling, not with replacing the truth criterion.

---

## 9. Thesis program phases

## Phase A — paper-grade four-language behavioral map

Run the main confirmatory panel:

- models: `1B`, `4B`
- languages: `Hindi`, `Telugu`, `Bengali`, `Tamil`
- shots: `8`, `64`
- seeds: `42`, `11`, `101`
- conditions: helpful + matched controls from the current Loop 2 pipeline

### Goal

Produce a **four-language phase map** strong enough for thesis-level claims about copying / retrieval / composition regimes.

### Current executable launcher

```bash
VM_PASS='***' bash experiments/run_vm_four_lang_thesis_panel.sh
```

---

## Phase B — calibrated verifier stack

Build a local verifier pipeline with three layers:

1. correctness / acceptability judge,
2. failure-taxonomy judge,
3. human-audited calibration subset.

### Goal

Capture variant-equivalence and failure categories without letting judge behavior become ungrounded.

### Minimum bar

Report disagreement against humans and deterministic metrics before using judge outputs in thesis-defining figures.

---

## Phase C — mechanistic localization

### Case study A: `1B Hindi`

Target: early-routing / first-token failure.

Desired outputs:
- correct-token vs competitor trajectories,
- layer / position localization,
- comparison against corrupt and zero-shot prompts,
- explicit identification of whether failure is Latin/source-like, wrong-target-script, or bank-conditioned.

### Case study B: `1B Telugu`

Target: later retrieval/composition failure.

Desired outputs:
- token-position-localized divergence after the first token,
- comparison between gold continuation and copied-bank continuation,
- identification of where aligned-neighbor retrieval takes over.

### Case study C: `4B Telugu`

Target: positive-control comparison.

Desired outputs:
- same measurements as the `1B Telugu` path,
- explicit contrast showing what later-stage computation succeeds in `4B` but not `1B`.

### Rule

Localization before heavy interventions.

---

## Phase D — bounded causal tests

After localization is clean, run targeted causal work only where the localization already points:

- ablations,
- patching,
- recency / position manipulations,
- candidate feature or component interventions.

### Rule

No heavy causal claim should be made from diffuse, weak, or under-localized behavior.

---

## Phase E — final synthesis

Combine:

1. the four-language behavioral map,
2. the verifier stack,
3. the bounded mechanistic case studies,
4. the `270M -> 1B -> 4B` synthesis.

### Intended thesis contribution

A careful account of when in-context transliteration behaves like copying, retrieval, or composition, plus at least one mechanistic explanation that survives strong checks.

---

## 10. Decision rules

## Behavioral GO

Proceed with a strong thesis story if the four-language panel shows:

- stable positive `4B` regimes,
- stable fragile or retrieval-heavy `1B` regimes,
- and enough structure to support a real phase map rather than isolated anecdotes.

## Mechanistic GO

Proceed with stronger mechanistic claims only if:

- the behavior is stable across seeds,
- the failure mode is localized in stage and position,
- the positive control is comparably localized,
- and the intervention effect is specific enough to be scientifically legible.

## Mechanistic NO-GO

If localization stays diffuse or interventions only yield weak global nudges, downgrade the thesis to a strong behavioral + representational study rather than forcing a circuit claim.

---

## 11. Compute / tooling plan

## VM

Use the shared A100 VM for:

- multi-seed behavioral panels,
- mechanistic probes that need repeated local inspection,
- long-running thesis-scale experiments.

## Modal

Use Modal when:

- detached sweeps are cleaner than VM orchestration,
- `270M/1B/4B` cross-scale premise-gate runs are the main task,
- or reproducible stateless sweeps are preferable to interactive debugging.

## AlphaXiv / GitHub / papers

Stay grounded in:

- current papers for ICL phase structure,
- mech-interp caution papers,
- implementation repos for transcoders / CLTs / analysis code,
- and direct code inspection when using a method from the literature.

No latest/current implementation claim should be made from memory alone.

---

## 12. Main risks

1. **Judge overreach** — solved by calibration and separating hard vs soft evidence.
2. **Mechanistic overclaiming** — solved by claim-discipline and localization-before-intervention.
3. **Thesis sprawl** — solved by keeping one main question: copying vs retrieval vs composition.
4. **Bug-driven false stories** — solved by keeping intuition active, but re-verifying after every meaningful fix.
5. **Breadth without depth** — solved by using the four-language panel for phase mapping, not endless language accumulation.

---

## 13. What success looks like

A successful thesis from this repo would produce:

1. a **paper-grade four-language behavioral phase map**,
2. a **calibrated verifier stack** with human-audited labels,
3. a **bounded mechanistic explanation** for at least one failure path and one positive-control path,
4. a clean final story about when ICL behaves like copying, retrieval, or composition in Gemma 3 transliteration.

That is ambitious, but it is still honest and testable.

---

## 14. Reviewer acceptance bar

A recurring planning question for this project is:

> **Would I accept this if I were a skeptical reviewer, and why should anyone else accept it?**

### I would not accept the thesis if it only provided

- top-level exact-match plots with weak manual auditing,
- a judge-heavy story without calibration,
- language anecdotes that do not survive multi-seed checks,
- mechanistic language without stage localization,
- or causal claims that remain broad, weak, and post-hoc.

### I would accept the thesis if it provides all of the following

1. **A real four-language confirmatory result**, not just Hindi/Telugu anecdotes.
2. **Multi-seed stability** for the main behavioral claims.
3. **Manual audit evidence** showing that the claimed failure modes are visible in raw examples and not just in derived metrics.
4. **A calibrated verifier stack** whose disagreements with humans and deterministic metrics are explicitly reported.
5. **At least one bounded mechanistic case study** that is localized enough to support a specific computational claim.
6. **Explicit negative results and retired claims**, so the work is clearly not just story-fitting.

### Manual-audit rule under this bar

At any stage where a result looks cleaner or more surprising than expected, perform explicit spot-checks before upgrading the claim.

A reusable helper for this now exists:

```bash
python3 experiments/build_manual_audit_packet.py \
  --input path/to/neutral_filler_recency_controls.json \
  --out-json notes/audit_packet.json \
  --out-md notes/audit_packet.md
```

### Current gap to acceptance

The project is not yet at the acceptance bar because it still needs:

- the full four-language multi-seed panel,
- the verifier calibration package,
- and the next layer of mechanistic localization for `1B Hindi` and `1B Telugu`.

That gap is exactly what the current thesis-scale program is meant to close.
