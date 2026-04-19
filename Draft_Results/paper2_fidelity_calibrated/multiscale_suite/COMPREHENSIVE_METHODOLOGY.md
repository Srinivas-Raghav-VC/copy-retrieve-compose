# Comprehensive Experimental Methodology: Mechanistic Understanding of ICL in Gemma 31B

**Date:** 2026-03-20
**Status:** Planning Document
**Scope:** Deep mechanistic analysis of in-context learning (ICL) in Gemma 3 1B transliteration tasks

---

## Executive Summary

This document presents a comprehensive experimental plan to achieve **complete mechanistic understanding** of how Gemma 3 1B implements transliteration ICL. The plan integrates:

1. **Literature grounding** from Wurgaft et al. (NeurIPS 2025), EleutherAI attention probes, and the broader ICL interpretability literature
2. **Structural dimension exploration** across prompt structures, sampling strategies, and languages
3. **Causal intervention methodology** following rigorous patching protocols
4. **Visual analysis framework** for publication-quality mechanistic plots

---

## Part 1: Literature Foundation

### 1.1 Theoretical Framework: Wurgaft et al. (NeurIPS 2025)

**Key insight:** ICL strategies emerge from a hierarchical Bayesian framework:

```
ICL behavior = posterior-weighted average over strategies
            = P(memorizing | data) × memorizing_predictor 
            + P(generalizing | data) × generalizing_predictor
```

**Strategic predictions for our work:**

| Prediction | Implication for Gemma 3 1B Transliteration |
|------------|-------------------------------------------|
| Strategies trade off loss vs complexity | Smaller models (1B) should prefer memorization over generalization |
| Memorizing predictor assumes discrete prior | 1B may encode ICL mappings as discrete lookuptables rather than algorithms |
| Generalizing predictor matches task distribution | 4B maylearn compositional transliteration rules |
| Posterior updates with more examples | More ICL examples shift weight toward memorizing OR generalizing depending on which better fits |

**How this changes our experimental design:**

1. **Measure strategy mixture:** Does 1B switch from memorizing to generalizing asexamples increase?
2. **Intervention on strategies:** Can we intervene on specific heads to shift the strategy balance?
3. **Cross-scale comparison:** Does 270M use different strategies than 1B/4B?

### 1.2 EleutherAI Attention Probes: Aggregation Methodology

**Problem:** Standard probing has an aggregation problem - how to reduce `[seq_len, d_model]` to a single vector?

| Method | What it assumes | When it fails |
|--------|-----------------|---------------|
| Mean pooling | Information is diffuse across sequence | Localizedsignals get diluted |
| Last-token | Model aggregates to final position | Early-sequence signals ignored |
| Attention probe | Learns which positions matter | Adds complexity, may overfit |

**Application to our work:**

For logit lens and content-specificity analysis, we need to decide aggregation strategy. Current approach (last-token for autoregressive models) is defensible but may miss signals at other positions.

**Proposed addition:**
- Add attention-probe aggregation as a diagnostic layer
- Compare last-token vs learned-attention aggregation for content-specificity detection
- This can reveal whether the model deposits task information at specific positions

### 1.3 ICL Representations (Park et al., ICLR 2025)

**Key finding:** Models can in-context learn novel semantics, overriding pretrained meaning through representational reorganization.

**Implications:**
- The "format signal" vs "content signal" in our analysis aligns with this framework
- Low-resource languages (Kannada, Gujarati) may lack pretrained character mappings, so ICL cannot override
- High-resource languages (Hindi) have pretrained mappings that ICL refines

### 1.4 Evidence Accumulation Model (Bigelow et al., ICLR 2024)

**Key finding:** ICL = posterior updating with each example as evidence.

**Mathematical model:**
```
p(rescue) = σ(b + γ · N^(1-α))
```

where N = number of examples, and each example contributes `γ · N^(-α)` to the evidence.

**Our density degradation data fits this model:**
- Examples contribute diminishing returns (each additional example adds less)
- Filler examples dilute effective evidence
- The 33×drop at 64 examples (crossing sliding window) is NOT explained by accumulation alone- requires architectural bottleneck

---

## Part 2: Experimental Dimensions

### 2.1 Prompt Structure Variations [NEW - Not Yet Done]

**Motivation:** Current experiments use a single prompt format. The Wurgaft framework suggests different prompts may shift strategy weights.

#### Experiment PS-1: Prompt Format Taxonomy

**Test 4 prompt structures:**

| Format | Example | What it tests |
|--------|----------|---------------|
| Standard | `namaste -> नमस्ते\nword -> SCRIPT` | Baseline (current design) |
| Instructional | `Transliterate to Hindi:\nnamaste -> नमस्ते` | Explicit task specification |
| Few-shot without arrow | `नमस्ते\nनमस्ते` (output only) | Removes explicit mapping cue |
| Chain-of-thought | `namaste: na-ma-ste, each sound maps to...` | Encourages algorithmic processing |

**Hypotheses:**
- H1: Instructional prompts shift weight toward generalizing predictor
- H2: Arrow-free prompts reduce memorization (harder to pattern-match)
- H3: CoT prompts encourage algorithmic (generalizing) strategies

**Measurements:**
- First-token probability/rank (mechanistic)
- Exact match (behavioral)
- Head attribution patterns (do they shift?)
- Content-specificity ratio (helpful vs corrupt)

**Output:** `prompts/format_taxonomy_{format}.json`

#### Experiment PS-2: Example Positioning

**Test 3 positioning strategies:**

| Position | Description | What it tests |
|----------|-------------|---------------|
| Start-near | Examples at positions 0-200, query at end | Current design (global layers cansee) |
| End-near | Query first, examples at end | Local layers can see examples too |
| Interspersed | Query and examples interleaved | Tests integration under different conditions |

**Critical test:** If end-near positioning reduces the 64-example collapse (because local layers can see examples), this confirms the architectural bottleneck hypothesis.

**Output:** `prompts/positioning_{strategy}.json`

#### Experiment PS-3: Task Clarity Manipulation

**Test 3 clarity levels:**

| Clarity | Prompt design | What it tests |
|---------|--------------|---------------|
| Explicit | "Transliterate: Latin -> Devanagari" | Clear task specification |
| Implicit | No instruction, just examples | Model must infer task |
| Ambiguous | Mixed-language examples | Tests generalization vs memorization |

**Hypothesis:** Ambiguous prompts force the model to generalize, revealingdifferent strategy mixture.

**Output:** `prompts/clarity_{level}.json`

### 2.2 Sampling Strategy Variations [NEW - Not Yet Done]

**Motivation:** Temperature and sampling affect the distribution over strategies in the Wurgaft framework.

#### Experiment SS-1: Temperature Sweep

**Test temperatures:** 0.0, 0.3, 0.7, 1.0, 1.5

**What this reveals:**
- Low temperature (0.0): Greedy decoding - reveals dominant strategy
- High temperature (1.5): More exploration - reveals strategy diversity

**For each temperature:**
- Generate N=50 outputs per language
- Measure: first-token probability, exact match, character error rate
- Plot: success rate vs temperature curves

**Hypothesis:** If the model has multiple strategies ("memorize" vs "generalize"), high temperature should reveal this diversity in outputs.

**Output:** `sampling/temperature_{t}.json`

#### Experiment SS-2: Top-k/Top-p Sampling

**Test decoding strategies:**

| Strategy | Setting | What it tests |
|----------|---------|---------------|
| Greedy | top_k=1 | Dominant strategy |
| Top-k | k=10, k=50 | Strategy diversity |
| Top-p | p=0.9 | Typical nucleus sampling |
| Beam | beam=5 | Comprehensive search |

**For beam search:**
- Examine whether beam candidates show consistent strategies
- If beams diverge (some correct, some wrong), this suggests strategy multiplicity

**Output:** `sampling/decoding_{strategy}.json`

#### Experiment SS-3: Repetition Penalty Variation

**Test:** repetition_penalty ∈ {1.0, 1.2, 1.5}

**Hypothesis:** Higher repetition penalty may force the model to explore alternative strategies, revealing how many valid transliteration paths exist in internal representations.

**Output:** `sampling/repetition_{penalty}.json`

### 2.3 Language and Data Variations [PARTIALLY DONE]

**Current state:** 5 languages tested (Hindi, Telugu, Bengali, Kannada, Gujarati)

#### Experiment LD-1: Language Resource Gradient

**Expand to 8 languages:**

| Language | Resource Level | Script | Hypothesis|
|----------|---------------|--------|-----------|
| Hindi | Very High | Devanagari | Strongest content-specific ICL |
| Telugu | High | Telugu | Strong mid-layer content-specificity |
| Bengali | Medium | Bangla | Weak content-specificity |
| Marathi | Medium | Devanagari | Shared script with Hindi - transfer? |
| Kannada | Low | Kannada | Format-dominated |
| Gujarati | Low | Gujarati | Format-dominated |
| Tamil | Low | Tamil | Format-dominated, different script family |
| Malayalam | Very Low | Malayalam | Pure format signal expected |

**This gradient tests the hypothesis:**
> ICL content-specificity correlates with pretraining resource level. Languages with more pretraining data have stronger pretrained character mappings, enabling the model to USE ICL content rather than relying on format signals.

**Output:** `languages/resource_gradient.json`

#### Experiment LD-2: Script Sharing Analysis

**Test:** Marathi (Devanagari) vs Hindi (Devanagari)

**What this tests:**
- If Marathi ICL works as well as Hindi, the model transfers script knowledge across languages
- If Marathi ICL is weaker, the model's transliteration knowledge is language-specific, not script-specific

**Output:** `languages/script_sharing.json`

#### Experiment LD-3: Character Frequency Control

**Design pairs with:**
- High-frequency characters (top-100 in corpus)
- Medium-frequency characters (100-1000)
- Low-frequency characters (1000+)

**Hypothesis:** Characters with higher pretraining frequency should show stronger content-specificity (model already knows the mapping, ICL refines it).

**Output:** `languages/char_frequency.json`

### 2.4 N-shot Curve Variations [PARTIALLY DONE]

**Current state:** Density experiments at 8, 16, 32, 48, 64 examples

#### Experiment NS-1: Fine-grained Early Curve

**Add density points:** 2, 4, 6, 8, 12, 16,24, 32 examples

**Why:** The Wurgaft framework predicts strategy mixing. Fine-grained early curve may reveal:
- Phase transitions (sudden jumps)
- Evidence accumulation slopes
- Strategy weight shifts

**Output:** `density/early_curve.json`

#### Experiment NS-2: Long-context Extension

**Test prompt lengths:** 100, 200, 400, 800, 1600, 3200, 6400 tokens

**What this tests:**
- At 512 tokens: local layers lose access (confirmed)
- At 3200 tokens: How does ICL behavior extend?
- At 6400 tokens: Full context window test

**Hypothesis:** Long contexts suffer from both attention dilution AND architectural bottleneck. The degradation should be non-linear.

**Output:** `density/long_context.json`

#### Experiment NS-3: Example Quality vs Quantity

**Design:**
- 16 correct examples (baseline)
- 8 correct + 8 distractor (same script, wrong mapping)
- 16correct + 16 distractor
- 16 correct + 32 distractor (current "filler" condition)

**What this tests:** Does the model distinguish helpful examples from distractors?

**Wurgaft framework prediction:** The posterior should down-weight distractor information if the model is generalizing. If memorizing, distractors should corrupt performance directly.

**Output:** `density/quality_vs_quantity.json`

---

## Part 3: Causal Intervention Methodology

### 3.1 Core Intervention Types

**Status:** Attention-only patching and MLP contribution have been run. Joint intervention is PLANNED.

#### Intervention I-1: Per-Head Attribution [DONE - N=20]

**What it measures:** Fraction of ICL-ZS logit gap recovered by patching a single head's output.

**Current finding:** L14H0 and L11H0 are the most universal across languages.

**Upgrade needed:** Re-run at N=50 for tighter CIs.

#### Intervention I-2: Attention-Contribution Patching [DONE - N=50]

**What it measures:** Effect of replacing ONLY attention outputs (not full residual stream).

**Current finding:** Near-zero effect for all layer subsets. This proves attention-alone is insufficient.

**Critical:** This is the strongest negative finding.

#### Intervention I-3: MLP Single-Layer Contribution [DONE - N=30 Hindi]

**What it measures:** Effect of replacing single-layer MLP output.

**Current finding:**
- Destructive: L12, L15, L16 (post-global local layers)
- Constructive: L17, L23, L25 (global layers + readout)

**Upgrade needed:** N=50 for Hindi, N=50 for Telugu.

#### Intervention I-4: Joint Attention+MLP Grouping [PLANNED - HIGHEST PRIORITY]

**The key closure experiment:**

| Group | Components replaced | Expected |
|-------|---------------------|----------|
| A_global | Attention at L05,L11,L17,L23 | ~0 |
| M_global | MLP at L05,L11,L17,L23 | Small + |
| A+M_global | Both at global layers | **Significantly > A_global + M_global** |
| A+M_L11_L17 | Both at L11+L17 | Key pair test |
| A+M_all | Both at all26 layers | Upper bound |

**Why this matters:** If A+M_global is significantly more than either alone, this proves the coupled mechanism. If A+M_global is still near zero, the mechanism is more distributed than we thought.

**Design:**
- N=50 per language (Hindi, Telugu)
- Bootstrap 95% CIs
- Metric: first-token probability PE + rank change

#### Intervention I-5: Head-Group Intervention [NEW]

**Test intervening on groups of heads:**

| Group | Heads | Rationale |
|-------|-------|-----------|
| Universal | L14H0, L11H0, L17H3, L21H1 | Top4 across languages |
| Global-only | L05H0, L11H0, L17H0, L23H0 | One head per global layer |
| All-global | All 4heads× 4 layers | Complete global attention |

**Measurement:** Compare single-head patching to group patching. Does the effect compound linearly or super-linearly?

**Output:** `interventions/head_groups.json`

#### Intervention I-6: Mediation Analysis [NEW]

**Goal:** Trace the causal path from ICL examples to output.

**Method:**
1. Patch head H_i from ICL→ZS
2. Measure effect at all downstream residual stream positions
3. Identify which downstream positions carry the mediated signal

**Hypothesis:** L11H0 writes to residual stream positions that L14H0 reads from. This would establish:
```
ICL examples → L05/11H0 (global attention reads) → residual stream → L14H0 (local amplifies) → output
```

**Output:** `interventions/mediation_path.json`

### 3.2 Layer-wise Attribution Analysis

**For each layer:**
| Metric | How to measure |
|--------|----------------|
| Attention contribution | Patch attention output only |
| MLP contribution | Patch MLP output only |
| Residual stream change | Compare L05→L11→L17→L23 residual stream magnitudes |
| Head importance distribution | Variance across 4 heads per layer |

### 3.3 Attention Pattern Analysis

**For key heads:**

| Head | What to measure |
|------|------------------|
| L05H0, L05H1 | % attention on ICL examples, % on `<bos>`, % on query |
| L11H0, L11H1 | Distribution over ICL tokens - which tokens get most attention |
| L17H0, L17H1 | Balance between ICL region and query region |
| L23H0, L23H1 | Final consolidation pattern |

**Compare ICL vs ZS:**
- Does the attention pattern change?
- Which positions gain/lose attention?

---

## Part 4: Plot Framework

### 4.1 Core Mechanistic Plots

#### Plot M1: Logit Lens Trajectories

**What it shows:** Target token rank/probability progression across layers for helpful, corrupt, and ZS conditions.

**Current:**
```
fig1_logit_lens_trajectories.png/pdf
```

**Planned additions:**
- Add confidence bands (bootstrap)
- Add language gradient color coding
- Annotate global layer positions

#### Plot M2: Content-Specificity Ratio

**What it shows:** Ratio of corrupt_rank / helpful_rank at each layer, per language.

**Current:**
```
fig2_content_specificity.png
```

**Planned additions:**
- Add ratio-by-ICL-count curves (Experiment NS-1)
- Highlight global layer positions

#### Plot M3: Head Attribution Heatmap

**What it shows:** Attribution effect for each head, across languages.

**Current:**
```
fig3_head_attribution_heatmap.png
```

**Planned additions:**
- Add N=50 vs N=20 comparison (CIs)
- Cluster heads by effect strength

#### Plot M4: Attention-Contribution = Near Zero

**What it shows:** Bar chart showing PE for different layer subsets, all near zero.

**Current:**
```
fig4_attention_only_patching.png
```

**This is the key negative finding plot.**

#### Plot M5: MLP Contribution

**What it shows:** Per-layer MLP PE with CIs.

**Current:**
```
fig5_mlp_contribution.png
```

**Planned additions:**
- Add Telugu
- Annotate destructive vs constructive layers

#### Plot M6: Joint Intervention [NEW - HIGHEST PRIORITY]

**What it shows:** Bar chart comparing:
- Attention-only (global layers)
- MLP-only (global layers)
- Attention+MLP (global layers)
- Full model (all layers)

**This is the key positive finding plot that proves the coupled mechanism.**

#### Plot M7: Density Degradation

**What it shows:** Target probability and attention-per-example vs number of examples.

**Current:**
```
fig5_density_degradation.png
```

**Planned additions:**
- Add Telugu
- Add confidence bands
- Annotate sliding window boundary (512 tokens)

#### Plot M8: Architecture Mechanism Diagram

**What it shows:** Layer-by-layer diagram with:
- Global vs local layer distinction
- Attention pattern percentages
- Head attribution effects
- MLP contribution signs

**Current:**
```
fig6_architecture_mechanism.png
```

### 4.2 Behavioral Plots

#### Plot B1: Language Hierarchy Bar Chart

**What it shows:** Acceptable rate by language and condition.

**Metric:** Judge acceptable%, exact match%

#### Plot B2: ICL Count Curves

**What it shows:** Success rate vs number of examples.

**For each language:**
- Curve for helpful ICL
- Curve for corrupt ICL
- Curve for ZS

#### Plot B3: First-Token vs Behavioral Correlation

**What it shows:** Scatter plot of first-token probability vs judge acceptable.

**Purpose:** Validate that mechanistic proxy correlates with behavioral outcome.

###4.3 New Plots from Extended Experiments

#### Plot E1: Prompt Format Comparison

**What it shows:** Success rates for each prompt format (PS-1).

**Hypothesis:** Instructional prompts may improve performance by shifting strategy.

#### Plot E2: Temperature Sensitivity Curves

**What it shows:** Success rate vs temperature for each language.

**Hypothesis:** If multiple strategies exist, high temperature reveals diversity.

#### Plot E3: Language Resource Gradient

**What it shows:** Content-specificity ratio vs pretraining corpus size (log scale).

**Hypothesis:** Linear relationship - more resources → stronger content-specific ICL.

#### Plot E4: Joint Intervention Confirmatory Plot

**What it shows:**
- X-axis: Intervened components (A_global, M_global, A+M_global, A+M_all)
- Y-axis: Fraction of ICL-ZS gap recovered
- Error bars: Bootstrap 95% CIs

**Key comparison:** Is A+M_global significantly > A_global + M_global?

---

## Part 5: Execution Plan

### Phase 1: Core Closure [PRIORITY 1]

**Goal:** Prove coupled attention-MLP mechanism.

**Experiments:**
1. **TASK 1:** Joint attention+MLP intervention (Hindi, Telugu, N=50)
2. **TASK 2:** MLP contribution upgrade (Hindi N=50, Telugu N=50)
3. **TASK 3:** Head attribution upgrade (Hindi, Telugu, N=50)

**Deliverable:** Two key plots:
- M4: Attention-only contribution (= 0)
- M6: Joint intervention (A+M > either alone)

**Timeline:** 1 week GPU time

### Phase 2: Behavioral Grounding [PRIORITY 2]

**Goal:** Validate mechanistic proxy against behavioral outcomes.

**Experiments:**
1. **TASK 4:** First-token vs behavioral correlation (all 5 languages, N=50)
2. **TASK 5:** Full 5-language judge sweep

**Deliverable:**
- Plot B3: First-token vs behavioral correlation
- Plot B1: Language hierarchy

**Timeline:** 2 days API time

### Phase 3: Extended Dimensions [PRIORITY 3]

**Goal:** Test generalization across prompt structures, sampling, languages.

**Experiments:**
1. **PS-1:** Prompt format taxonomy
2. **SS-1:** Temperature sweep
3. **LD-1:** Language resource gradient (3 new languages)

**Timeline:** 1 week GPU time

### Phase 4: Robustness Checks [PRIORITY 4]

**Goal:** Reviewer-proof the core claims.

**Experiments:**
1. Seed robustness (3 seeds for head attribution)
2. Content-specificity at multiple ICL counts
3. Cross-validation splits

**Timeline:** 3 days GPU time

---

## Part 6: Success Criteria

### Minimum Viable Paper (What we have now)

| Claim | Evidence | Status |
|-------|----------|--------|
| C1: Behavior | API sweep + judge | ✅ Done |
| C2: Content-specificity | Logit lens matched control | ✅ Done |
| C3: Head universality | Cross-language attribution | ✅ Done (CIs wide)|
| C4: Attention-alone fails | Corrected patching | ✅ Done |
| C5: Density bottleneck | 8-64 example curve | ✅ Done (small N) |

### Complete Paper (What we need)

| Claim | Evidence | Required Experiments |
|-------|----------|---------------------|
| C1: Behavior | ✅ | - |
| C2: Content-specificity | ✅ | Add multi-count analysis |
| C3: Head universality | ✅ | Upgrade N to50 |
| C4: Attention-alone fails | ✅ | - |
| C5: Density bottleneck | ✅ | Add Telugu, upgrade N to 30 |
| **C6: Coupled mechanism** | **PLANNED** | **Joint A+M intervention** |
| **C7: Strategy mixture** | **PLANNED** | **Temperature, prompt format** |
| **C8: Language gradient** | **PLANNED** | **Expand to 8 languages** |
| **C9: Proxy-behavior link** | **PLANNED** | **First-token correlation** |

---

## Part 7: Literature Integration

### How Each Paper Informs Our Analysis

| Paper | Key Insight | How We Use It |
|-------|------------|---------------|
| Wurgaft et al. (NeurIPS 2025) | ICL = posterior over strategies | Test for strategy mixture via temperature, prompt format |
| EleutherAI Attention Probes | Aggregation method matters | Add attention-probe comparison for logit lens |
| Park et al. (ICLR 2025) | ICL representations override pretraining | Format vs content signal decomposition |
| Bigelow et al. (ICLR 2024) | Evidence accumulation model | Fit density curves to accumulation formula |
| Olsson et al. (2022) | Induction heads for ICL | Our heads are FV-type, not induction |
| Yin & Steinhardt (2025) | FV heads > induction heads | Supports our head attribution interpretation |
| Heimersheim & Nanda (2024) | Patching methodology | Our corrected attention-only patching |
| Kahardipraja et al. (2025) | In-context vs parametric heads | L11H0 = in-context, L14H0 = parametric |

---

## Part 8: Architecture-Specific Mechanism (Gemma 3 1B)

### The Sliding Window Constraint

```
Gemma 3 1B Architecture:
- 26 layers
- 4 attention heads per layer
- 1 KV head (GQA 4:1)
- sliding_window = 512
- sliding_window_pattern = 6 (every 6th layer is global)

Global layers: L05, L11, L17, L23 (can attend to all positions)
Local layers: all others (can only attend to last 512 positions)
```

### Mechanism Hypothesis

```
ICL Mechanism in Gemma 3 1B:

Step 1: Global attention reads ICL examples
┌─────────────────────────────────────────────────────────────────────┐
│ L05 (GLOBAL): First pass - "What is this task?"                    │
│   - Attends 92-99% to ICL examples                                │
│   - Extracts: format, script, instruction structure                │
│   - Writes: task-type representation to residual stream            │
├─────────────────────────────────────────────────────────────────────┤
│ L06-L10 (LOCAL): Local processing                                  │
│   - Cannot see ICL examples (beyond 512 tokens for long prompts)   │
│   - Amplifies L05's signal via residual stream                     │
├─────────────────────────────────────────────────────────────────────┤
│ L11 (GLOBAL): Second pass - "What is the mapping?"                 │
│   - L11H0: 92% attention on ICL examples                           │
│   - Extracts: specific character mappings                          │
│   - Writes: mapping content to residual stream                     │
│   - THIS IS THE UNIVERSAL RESCUE HEAD (top-3 in 3/5 languages)      │
├─────────────────────────────────────────────────────────────────────┤
│ L12-L16 (LOCAL): Amplification zone                                │
│   - L14H0: Strongest single-head effect                            │
│   - L15H3: Secondary amplifier                                     │
│   - MLP at L15: DESTRUCTIVE (-0.093 PE)                            │
├─────────────────────────────────────────────────────────────────────┤
│ L17 (GLOBAL): Third pass - "Integrate mapping with query"          │
│   - Mixed attention: 16-84% on ICL vs query                        │
│   - Combines mapping signal with query context                     │
│   - First layer with clear helpful > corrupt separation            │
├─────────────────────────────────────────────────────────────────────┤
│ L18-L22 (LOCAL): Refinement zone                                   │
│   - L21H1: Strong for Telugu/Kannada/Gujarati                      │
│   - MLP at L16: DESTRUCTIVE (-0.061 PE)                             │
├─────────────────────────────────────────────────────────────────────┤
│ L23 (GLOBAL): Fourth pass - "Final consolidation"                  │
│   - ~60% attention on ICL examples                                 │
│   - Final check and integration                                     │
├─────────────────────────────────────────────────────────────────────┤
│ L24-L25 (LOCAL): Readout zone                                      │
│   - MLP at L25: Constructive (+0.046 PE)                            │
│   - Converts residual signal to output logits                       │
└─────────────────────────────────────────────────────────────────────┘
                            ↓
                     Output: "नमस्ते"
```

### Key Insight: Coupled Attention→MLP Mechanism

**Why attention-alone patching gives zero:**
- Each head's output is calibrated to work with the downstream MLP
- Replacing all attention outputs simultaneously produces an activation pattern the MLP has never seen
- The MLP expects a specific residual stream composition from training

**Why single-head patching works:**
- Only 25% of the attention output changes
- The remaining 3 heads provide anchor points
- The MLP can adjust to the perturbation

**The mechanism is emergent:**
- Not localizable to attention OR MLP independently
- Not localizable to a single layer
- Requires the full computational cascade from L05→L25

---

## Appendix A: Experiment Specifications

### A.1 Model Specification

```
Model: google/gemma-3-1b-it
Precision: bf16
Architecture:
  -num_hidden_layers: 26
  - num_attention_heads: 4
  - num_key_value_heads: 1 (GQA)
  - head_dim: 256
  - hidden_size: 1152
  - intermediate_size: 6912
  - vocab_size: 262144
  - max_position_embeddings: 32768
  - sliding_window: 512
  - sliding_window_pattern: 6
```

### A.2 Data Specification

```
Source: Aksharantar (AI4Bharat) + HuggingFace transliteration pairs
Languages: Hindi (hin), Telugu (tel), Bengali (ben), Kannada (kan), Gujarati (guj)
Splits:
  - ICL examples: 16 randomly selected pairs (seed=42)
  - Evaluation: 50 items (offset 316-366 from shuffled pool)
Matched controls:
  - Corrupt ICL: shuffle outputs, same prompt length
  - Zero-shot: no examples
```

### A.3 Metric Specifications

```
Primary mechanistic metric: first-token probability
  - At the generation position
  - Target token probability
  - Target token rank (among 262K vocabulary)

Primary behavioral metric: judge acceptable rate
  - Automatic guardrails for obvious cases
  - Gemini 2.5 Flash for ambiguous cases
  - Binary classification: acceptable / not acceptable

Supplementary metrics:
  - Exact match (strict)
  - Character error rate (CER)
  - First-character match
  - Target script ratio
```

### A.4 Statistical Specifications

```
Bootstrap confidence intervals:
  - N_bootstrap = 1000
  - CI level = 95%
  - Aggregation: mean over items

Significance tests:
  - Paired sign test for ICL vs ZS
  - Wilcoxon signed-rank for matched comparisons
  - Fisher's exact test for binary comparisons

Effect size reporting:
  - Cohen's d for continuous metrics
  - Odds ratio for binary metrics
```

---

## Appendix B: File Naming Convention

```
results/
├── 1b_final/
│   ├── attention_only_contribution_{language}.json
│   ├── head_attribution_{language}_n50.json
│   ├── mlp_contribution_{language}.json
│   ├── joint_intervention_{language}.json
│   ├── density_degradation_{language}_n30.json
│   ├── content_specificity_by_count.json
│   ├── prompts/
│   │   ├── format_taxonomy_{format}.json
│   │   ├── positioning_{strategy}.json
│   │   └── clarity_{level}.json
│   ├── sampling/
│   │   ├── temperature_{t}.json
│   │   ├── decoding_{strategy}.json
│   │   └── repetition_{penalty}.json
│   ├── languages/
│   │   ├── resource_gradient.json
│   │   ├── script_sharing.json
│   │   └── char_frequency.json
│   └── figures/
│       ├── fig_m1_logit_lens.pdf
│       ├── fig_m6_joint_intervention.pdf
│       └── ...
└── 1b_mechanistic_analysis/
    └── [existing analysis documents]
```

---

## Appendix C: Claim Mapping to Experiments

| Claim | Primary Experiment | Secondary Experiments | Status |
|-------|-------------------|----------------------|--------|
| C1: Behavioral premise | API sweep | Judge validation | ✅ |
| C2: Content-specificity | Matched-control logit lens | Multi-count analysis | ✅ (needs N upgrade) |
| C3: Head universality | Cross-language attribution | Seed robustness | ✅ (needs N upgrade) |
| C4: Attention-alone fails | Corrected patching | - | ✅ |
| C5: Density bottleneck | 8-64 density curve | Long-context extension | ✅ (needs N upgrade) |
| C6: Coupled mechanism | Joint A+M intervention | Mediation analysis | PLANNED |
| C7: Strategy mixture | Temperature sweep, prompt formats | - | PLANNED |
| C8: Language gradient | 8-language sweep | Script sharing | PLANNED |
| C9: Proxy-behavior link | First-token correlation | - | PLANNED |

---

**Document End**