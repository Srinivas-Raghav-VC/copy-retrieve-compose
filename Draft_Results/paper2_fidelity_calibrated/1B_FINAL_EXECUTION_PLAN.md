# 1B FINAL EXECUTION PLAN — The Definitive Checklist

**Date:** 2026-03-18  
**Goal:** Complete publication-ready 1B analysis end-to-end.  
**Design:** Deep causal on Hindi+Telugu. Breadth on 5 languages. Behavioral on all 5.  
**Compute:** Modal A100-40GB for GPU work. Gemini API for judge work. Local for writing.

---

## WHAT WE ALREADY HAVE (locked, do not re-run)

| ID | Experiment | Languages | N | Status | Confidence |
|----|-----------|-----------|---|--------|------------|
| D1 | Matched-control logit lens | 5 (Hi/Te/Be/Ka/Gu) | 50 each | ✅ done | HIGH |
| D2 | Head attribution | 5 languages | 20 each | ✅ done | MODERATE (CIs wide) |
| D3 | Corrected attention-only patching | Hindi + Telugu | 50 each | ✅ done (A6000, older split) | HIGH |
| D4 | MLP single-layer contribution | Hindi only | 30 | ✅ done | MODERATE |
| D5 | Density degradation + attention mass | Hindi only | 10/density | ✅ done | MODERATE (small N) |
| D6 | Telugu L25 anomaly analysis | Telugu | 50 | ✅ done | HIGH |
| D7 | Head attention patterns | Hindi | 10 | ✅ done | MODERATE |
| D8 | Generation baseline (greedy) | 5 languages | 50 each | ✅ done | HIGH (EM≈0 expected) |
| D9 | API behavioral sweep (repaired) | 5 languages | 352 rows | ✅ done | HIGH |
| D10 | Judge pipeline + anchor validation | 5 languages | pilot+full | ✅ done | HIGH (auxiliary) |
| D11 | M1 first-token proxy validation | Hindi + Telugu | 50×5 counts | ✅ done | HIGH |
| D12 | 8 publication figures (PNG+PDF) | — | — | ✅ done | — |
| D13 | 9 analysis markdown docs | — | — | ✅ done | — |

**These are frozen. We do NOT touch them unless a bug is found.**

---

## TASK LIST (18 tasks, strict execution order)

---

### PHASE 1: Strengthen existing weak spots (Modal GPU)

These upgrade existing experiments from "moderate confidence" to "high confidence."

---

#### TASK 1 — Telugu MLP contribution (N=50)

**Why:** MLP contribution currently exists only for Hindi (N=30). We need Telugu to claim MLP role is cross-language, not Hindi-specific.

**What:** For each of 26 layers, replace single-layer MLP output from ICL run into ZS run. Measure first-token PE (patch effect = patched_prob − zs_prob).

**Spec:**
- Model: Gemma 3 1B IT
- Languages: Telugu
- N: 50 items (same eval split, seed=42)
- ICL count: 16 (standard)
- Layers: all 26
- Metric: first-token probability PE + bootstrap 95% CI

**Pass gate:** At least 2 layers show significant non-zero PE (CI excludes zero). Pattern should show global-layer MLPs (L17/L23) positive, post-global local MLPs (L12/L16) negative or near-zero.

**Output:** `results/1b_final/telugu_mlp_contribution.json`

---

#### TASK 2 — Hindi MLP contribution upgrade (N=50)

**Why:** Current Hindi MLP is N=30. Upgrade to N=50 for consistency and tighter CIs.

**Spec:** Same as Task 1 but for Hindi.

**Pass gate:** Same pattern as current N=30 results, but with tighter CIs.

**Output:** `results/1b_final/hindi_mlp_contribution_n50.json`

---

#### TASK 3 — Density degradation (Hindi+Telugu, N=30)

**Why:** Current density experiment is Hindi-only, N=10 per density level. Too small for publication. Need Telugu too.

**Spec:**
- Languages: Hindi + Telugu
- N: 30 items per density level
- Density levels: 4, 8, 16, 32, 48, 64
- Measure: first-token probability + global-layer attention mass per example

**Pass gate:**
- Clear monotonic decline from 8→48 (dilution)
- Sharp drop at 48→64 (window boundary for 1B with sliding_window=512)
- Telugu should show similar architecture-driven collapse even if absolute levels differ

**Output:** `results/1b_final/density_degradation_hindi_telugu_n30.json`

---

#### TASK 4 — Final same-split attention-only rerun (Hindi+Telugu, N=50)

**Why:** The corrected attention-only result is central and strong, but currently lives in an older artifact tree / split context. For a final 1B freeze, we should replicate it on the exact final Hindi/Telugu evaluation setup.

**Spec:**
- Languages: Hindi + Telugu
- N: 50 items
- Subsets: single layers, global-only, local-only, all-attention
- Metric: first-token probability PE + rank change + fraction recovered

**Pass gate:** `all_global_attn` and `all_attn` remain near zero relative to ICL on the final split. If not, we debug before trusting any joint-group conclusion.

**Output:** `results/1b_final/attention_only_contribution_hindi.json`, `results/1b_final/attention_only_contribution_telugu.json`

---

#### TASK 5 — Head attribution upgrade (Hindi+Telugu, N=50)

**Why:** Current N=20 gives wide CIs. N=50 is standard for mechanistic papers.

**Spec:**
- Languages: Hindi + Telugu
- N: 50 items
- 26 layers × 4 heads = 104 single-head patches per item
- Metric: fraction of ICL-ZS logit gap recovered, bootstrap 95% CI

**Pass gate:** Same top heads emerge (L14H0, L11H0, L17H3, L21H1) with tighter CIs. Top-3 heads should have CI_lo > 0.

**Output:** `results/1b_final/head_attribution_hindi_n50.json`, `results/1b_final/head_attribution_telugu_n50.json`

---

### PHASE 2: The key closure experiment (Modal GPU)

This is the single most important missing experiment.

---

#### TASK 6 — Joint attention+MLP grouped intervention (Hindi+Telugu, N=50)

**Why:** We know attention-only ≈ 0. We know single-MLP is small. The claim is they work TOGETHER. We must test this directly.

**What:** Simultaneously replace attention output AND MLP output at selected layer groups.

**Spec:**
- Languages: Hindi + Telugu
- N: 50 items
- Groups to test:

| Group | What's replaced | Expected |
|-------|----------------|----------|
| `attn_global_only` | Attention at L05,L11,L17,L23 | ~0 (already shown) |
| `mlp_global_only` | MLP at L05,L11,L17,L23 | Small positive |
| `both_global` | Attention+MLP at L05,L11,L17,L23 | **Should be > either alone** |
| `both_L11_L17` | Attention+MLP at L11+L17 | Key pair test |
| `both_L17_L23` | Attention+MLP at L17+L23 | Key pair test |
| `both_all_layers` | Attention+MLP at all 26 layers | Upper bound |
| `mlp_all_layers` | MLP at all 26 layers | Reference |

- Metric: first-token PE + rank change + bootstrap 95% CI

**Pass gate:** `both_global` PE significantly > `attn_global_only` PE AND > `mlp_global_only` PE. This proves coupled mechanism.

**CRITICAL: If `both_global` is ALSO near zero, the mechanism is even more distributed than we thought. That's still a valid finding but changes the claim.**

**Output:** `results/1b_final/joint_attn_mlp_grouped_hindi.json`, `results/1b_final/joint_attn_mlp_grouped_telugu.json`

---

### PHASE 3: Behavioral grounding (Gemini API, local)

---

#### TASK 7 — Full 5-language behavioral judge sweep (API)

**Why:** Need complete behavioral picture across all 5 languages with the repaired + structured judge.

**Spec:**
- Input: use generation outputs from D8 (ICL+ZS, N=50, 5 languages) + M1 generation outputs (Hindi+Telugu, counts 0/2/4/8/16)
- Primary judge: Gemini 3 Flash
- Secondary judge: Gemini 2.0 Flash Lite (audit only)
- Anchors: automatic (positive exact + negative empty/copy/explanation/wrong-ref)
- Deterministic metrics: EM, CER, first_char_match, target_script_ratio

**Pass gate:**
- Primary anchor accuracy ≥ 95%
- Hindi ICL acceptable rate > Hindi ZS acceptable rate (confirms problem statement)
- Cross-language ordering roughly follows: Hindi > Telugu > Bengali > Gujarati > Kannada

**Output:** `results/1b_final/behavioral_judge_5lang_full.json`

---

#### TASK 8 — Behavioral analysis + plots (local)

**Why:** Generate publication figures from Task 7 output.

**Spec:** Python script producing:
- `fig_behavioral_5lang_accept_rate.png` — bar chart: acceptable% by language × condition
- `fig_behavioral_icl_vs_zs.png` — paired comparison
- `fig_behavioral_judge_vs_em.png` — scatter showing judge adds value over strict EM
- `fig_behavioral_count_curve.png` — acceptable% vs ICL count (Hindi+Telugu from M1)

**Pass gate:** Plots are clean, interpretable, match numerical expectations.

**Output:** `results/1b_final/behavioral_figures/`

---

### PHASE 4: Robustness checks (Modal GPU)

---

#### TASK 9 — Content-specificity at multiple densities (Hindi+Telugu, N=30)

**Why:** Currently we show helpful > corrupt at N=16 ICL examples. A reviewer may ask: does this hold at other counts?

**Spec:**
- Matched-control logit lens at ICL counts: 4, 8, 16, 32
- Languages: Hindi + Telugu
- N: 30 items per count
- Compare: helpful vs corrupt rank at L17 and L25

**Pass gate:** Content-specificity ratio (corrupt/helpful) > 1.0 for Hindi at all counts. Telugu shows mid-layer specificity that converges at output.

**Output:** `results/1b_final/content_specificity_by_count.json`

---

#### TASK 10 — Seed robustness for head attribution (Hindi, 3 seeds)

**Why:** Show top heads are stable across random ICL example selections, not cherry-picked.

**Spec:**
- Language: Hindi
- N: 30 items
- Seeds: 42, 123, 456
- Same head attribution procedure per seed

**Pass gate:** Top-3 heads overlap across all 3 seeds (at least 2 of top-3 shared).

**Output:** `results/1b_final/head_attribution_seed_robustness.json`

---

### PHASE 5: Synthesis and writing (local)

---

#### TASK 11 — Update 1B_PAPER_REFERENCE.md with final numbers

**What:** Replace all "moderate confidence" entries with final high-N results from Tasks 1-10. Add new tables for Task 6 (grouped intervention) and Task 9 (multi-density content specificity).

**Pass gate:** Every table in the doc has N ≥ 30, two languages for claim-bearing experiments.

---

#### TASK 12 — Generate final figure set

**What:** Rerun `generate_1b_figures.py` (or updated version) with final data.

**Figures needed (minimum 10):**

| Fig | Content | Source |
|-----|---------|--------|
| 1 | Logit lens trajectories (5 languages) | D1 |
| 2 | Content-specificity ratio bars | D1 |
| 3 | Head attribution heatmap (5 languages) | Task 5 |
| 4 | Attention-only patching = zero | Task 4 + D3 |
| 5 | MLP contribution (Hindi+Telugu) | Tasks 1-2 |
| 6 | **Joint attention+MLP grouped intervention** | **Task 6** |
| 7 | Density degradation (Hindi+Telugu) | Task 3 |
| 8 | Architecture diagram with mechanism | existing |
| 9 | Cross-language head universality table | Task 5 |
| 10 | Telugu anomaly (format dominance) | D6 |
| 11 | Behavioral 5-language judge comparison | Task 8 |
| 12 | Content-specificity by ICL count | Task 9 |

**Pass gate:** All figures render cleanly, no placeholder data.

---

#### TASK 13 — Rewrite Results section in LaTeX

**What:** Replace current `04_results.tex` (which is 4B CFOM transcoder focused) with new 1B-focused results using `1B_PAPER_REFERENCE.md` as source.

**Structure:**
1. §4.1 — Behavioral premise (ICL helps, ZS fails, language hierarchy)
2. §4.2 — Logit lens: content-specificity is real and language-dependent
3. §4.3 — Head attribution: shared heads across languages
4. §4.4 — Attention-only replacement fails (the key negative)
5. §4.5 — MLP contribution is non-trivial
6. §4.6 — **Joint attention+MLP recovers what neither can alone** (the key positive)
7. §4.7 — Density degradation: dilution + window boundary
8. §4.8 — Telugu format dominance and cross-language hierarchy

**Pass gate:** Every claim cites specific N, CI, and artifact path.

---

#### TASK 14 — Update Related Work with grounded citations

**What:** Use `1B_ALPHAXIV_GROUNDED_CLAIM_MAP.md` to ensure every citation use is Tier A or B, and wording follows safe templates.

**Pass gate:** No overclaims. Every "consistent with" has a cited source. Every "novel" claim is genuinely not in prior work.

---

#### TASK 15 — Update Introduction and Discussion

**What:** Reframe for 1B positive story:
- Intro: "We study how ICL rescue works in a sliding-window architecture"
- Discussion: "Coupled attention-MLP mechanism under locality constraints"

**Pass gate:** No mention of 4B as primary. 4B mentioned only as "future scale extension."

---

#### TASK 16 — Write Appendix sections

**What:**
- Appendix A: Full 5-language logit lens + head attribution tables
- Appendix B: Seed robustness (Task 10)
- Appendix C: Judge configuration, anchors, validation details
- Appendix D: Generation examples (qualitative)

---

#### TASK 17 — Final internal audit

**What:** Read the complete paper. For every claim, check:
- [ ] Is there an artifact file backing it?
- [ ] Is N ≥ 30?
- [ ] Is the CI reported?
- [ ] Is the wording "consistent with" (not "proves")?
- [ ] Does the figure match the table?

**Pass gate:** Zero ungrounded claims.

---

#### TASK 18 — Compile and submit

**What:** `pdflatex` / `bibtex` cycle. Visual check. PDF ready.

---

## DEPENDENCY MAP

```
PHASE 1 (GPU, parallel)          PHASE 2 (GPU)         PHASE 3 (API)
┌──────────┐                     ┌──────────┐          ┌──────────┐
│ Task 1   │──┐                  │ Task 5   │          │ Task 6   │
│ Tel MLP  │  │                  │ JOINT    │          │ 5L Judge │
├──────────┤  │                  │ GROUPED  │          └────┬─────┘
│ Task 2   │  ├─── all feed ───▶│ (depends │               │
│ Hin MLP  │  │    into T5      │  on T1-4 │          ┌────▼─────┐
├──────────┤  │                  │  results │          │ Task 7   │
│ Task 3   │  │                  │  for     │          │ Beh Figs │
│ Density  │  │                  │  layer   │          └──────────┘
├──────────┤  │                  │  select) │
│ Task 4   │──┘                  └────┬─────┘
│ Head N50 │                          │
└──────────┘                          │
                                      │
PHASE 4 (GPU, parallel with P3)       │
┌──────────┐                          │
│ Task 8   │                          │
│ CS×count │                          │
├──────────┤                          │
│ Task 9   │                          │
│ Seed rob │                          │
└──────────┘                          │
                                      │
PHASE 5 (local, sequential)           │
┌──────────┐                          │
│ Task 10  │◀────── all GPU done ─────┘
│ Update   │
│ ref doc  │
├──────────┤
│ Task 11  │ figures
├──────────┤
│ Task 12  │ results LaTeX
├──────────┤
│ Task 13  │ related work
├──────────┤
│ Task 14  │ intro + discussion
├──────────┤
│ Task 15  │ appendix
├──────────┤
│ Task 16  │ audit
├──────────┤
│ Task 17  │ compile + submit
└──────────┘
```

---

## SAMPLE SIZE JUSTIFICATION

| N | Justification |
|---|---------------|
| 50 | Standard for mechanistic interp papers (matches Olsson 2022, Yin 2025) |
| 30 | Minimum for CLT-based CIs; used where GPU cost is high (density × 6 levels) |
| 20 | Below ideal; only acceptable for existing runs we don't want to re-run unless needed |
| 10 | Too small for main text; existing density data at N=10 will be superseded by Task 3 |

---

## COST ESTIMATE (Modal A100-40GB @ ~$2.10/hr)

| Task | Est. GPU time | Est. cost |
|------|--------------|-----------|
| Task 1 (Tel MLP) | ~15 min | $0.50 |
| Task 2 (Hin MLP) | ~15 min | $0.50 |
| Task 3 (Density) | ~30 min | $1.05 |
| Task 4 (Final attn-only rerun) | ~25 min | $0.90 |
| Task 5 (Head N50) | ~40 min | $1.40 |
| Task 6 (Joint) | ~25 min | $0.90 |
| Task 9 (CS×count) | ~30 min | $1.05 |
| Task 10 (Seed rob) | ~30 min | $1.05 |
| **TOTAL GPU** | **~3.5 hrs** | **~$7.35** |
| Tasks 6-7 (API) | — | ~$0.10 |

**Total: ~$6.55** — well within budget (~$29 remaining).

---

## FILE STRUCTURE (final)

```
paper2_fidelity_calibrated/
├── results/
│   ├── 1b_definitive/          ← existing (frozen)
│   ├── 1b_final/               ← NEW: all final high-N results
│   │   ├── hindi_mlp_contribution_n50.json
│   │   ├── telugu_mlp_contribution.json
│   │   ├── density_degradation_hindi_telugu_n30.json
│   │   ├── head_attribution_hindi_n50.json
│   │   ├── head_attribution_telugu_n50.json
│   │   ├── joint_attn_mlp_grouped_hindi.json
│   │   ├── joint_attn_mlp_grouped_telugu.json
│   │   ├── behavioral_judge_5lang_full.json
│   │   ├── content_specificity_by_count.json
│   │   ├── head_attribution_seed_robustness.json
│   │   └── behavioral_figures/
│   ├── 1b_mechanistic_analysis/ ← existing docs (will be updated)
│   └── 1b_figures/              ← will be regenerated
└── 1B_FINAL_EXECUTION_PLAN.md   ← THIS FILE
```

---

## SUCCESS CRITERIA (paper-level)

The paper is done when ALL of these are true:

1. [ ] Joint attention+MLP intervention shows significantly more rescue than either alone
2. [ ] Density collapse is replicated at N≥30 for both Hindi and Telugu
3. [ ] Top attribution heads are stable across seeds
4. [ ] Behavioral judge confirms ICL > ZS across all 5 languages
5. [ ] Every main-text claim has N≥30, two languages, reported CIs
6. [ ] Every citation follows the grounded claim map (Tier A or B only for main text)
7. [ ] LaTeX compiles cleanly with all figures
8. [ ] Internal audit passes with zero ungrounded claims
