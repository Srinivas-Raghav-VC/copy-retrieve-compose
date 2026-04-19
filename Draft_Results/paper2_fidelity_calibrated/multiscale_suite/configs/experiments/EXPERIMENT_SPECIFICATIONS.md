# Experiment Specifications: Mechanistic Understanding of ICL in Gemma 3 1B

**Version:** 2.0 (Literature-Grounded)
**Date:** 2026-03-21
**Status:** Planning - Awaiting Kernel Restoration

---

## Overview

This document specifies all experiments for the mechanistic understanding of ICL in Gemma 3 1B transliteration. Each experiment is grounded in the theoretical framework from the literature review.

---

## Critical Experiments (Tier 1)

These experiments test core hypotheses and are essential for publication.

---

### EXP-CRIT-01: Joint Attention+MLP Intervention

**Theoretical Basis:**
- Coupled mechanism hypothesis (our finding)
- Phase transition theory (Park et al., ICLR 2025)
- Strategy framework (Wurgaft et al., NeurIPS 2025)

**Hypothesis:**
> Joint attention+MLP intervention should show significantly more rescue than either component alone, proving the coupled mechanism.

**Design:**

```yaml
id: EXP-CRIT-01
name: "Joint Attention+MLP Grouped Intervention"
family: "causal_intervention"
priority: "CRITICAL"

languages:
  - hindi
  - telugu

n_items: 50
icl_count: 16
seeds: [42]

model: "google/gemma-3-1b-it"
precision: "bf16"

intervention_groups:
  - id: "attn_global_only"
    description: "Replace attention output at global layers L05,L11,L17,L23"
    layers: [5, 11, 17, 23]
    component: "attention_output"
    source: "icl_run"
    target: "zs_run"
    expected: "~0 PE"

  - id: "mlp_global_only"
    description: "Replace MLP output at global layers L05,L11,L17,L23"layers: [5, 11, 17, 23]
    component: "mlp_output"
    source: "icl_run"
    target: "zs_run"
    expected: "small positive PE"

  - id: "both_global"
    description: "Replace BOTH attention and MLP at global layers"
    layers: [5, 11, 17, 23]
    component: "both"
    source: "icl_run"
    target: "zs_run"
    expected: "SIGNIFICANTLY > attn_global_only + mlp_global_only"

  - id: "both_L11_L17"
    description: "Replace both at key pair L11+L17"
    layers: [11, 17]
    component: "both"
    source: "icl_run"
    target: "zs_run"
    expected: "Key pair test"

  - id: "both_L17_L23"
    description: "Replace both at key pair L17+L23"
    layers: [17, 23]
    component: "both"
    source: "icl_run"
    target: "zs_run"
    expected: "Key pair test"

  - id: "both_all_layers"
    description: "Replace both at all 26 layers (upper bound)"
    layers: "all"
    component: "both"
    source: "icl_run"
    target: "zs_run"
    expected: "Upper bound"

  - id: "mlp_all_layers"
    description: "Replace MLP at all 26 layers (reference)"
    layers: "all"
    component: "mlp_output"
    source: "icl_run"
    target: "zs_run"
    expected: "Reference"

metrics:
  primary: "first_token_probability_PE"
  secondary:
    - "first_token_rank_change"
    - "fraction_icl_gap_recovered"
  bootstrap_ci: true
  n_bootstrap: 1000
  ci_level: 0.95

pass_criteria:
  - "both_global PE significantly > attn_global_only PE (p < 0.05)"
  - "both_global PE significantly > mlp_global_only PE (p < 0.05)"
  - "Bootstrap CI excludes zero for both_global"

failure_interpretation:
  "If both_global ≈ 0: Mechanism more distributed than hypothesized; requires circuit tracing, not component patching"

outputs:
  - "results/1b_final/joint_intervention_{language}.json"
  - "results/1b_final/figures/fig_m6_joint_intervention.pdf"
```

---

### EXP-CRIT-02: Context-Length Geometry Sweep (Phase Transition Detection)

**Theoretical Basis:**
- Representational phase transition (Park et al., ICLR 2025)
- Evidence accumulation model (Bigelow et al., ICLR 2024)

**Hypothesis:**
> ICL behavior should show discontinuous jumps at critical context lengths, indicating phase transition in representation geometry.

**Design:**

```yaml
id: EXP-CRIT-02
name: "Context-Length Geometry Sweep for Phase Transition"
family: "phase_transition"
priority: "CRITICAL"

languages:
  - hindi
  - telugu

n_items: 30
icl_counts: [2, 4, 6, 8, 12, 16, 24, 32, 48, 64]
model: "google/gemma-3-1b-it"precision: "bf16"

measurements:
  behavioral:
    - "exact_match"
    - "judge_acceptable"
    - "first_token_probability"
  
  geometric:
    - "layerwise_pca_variance_explained"
    - "dirichlet_energy_token_graph"
    - "representation_alignment_to_task"
    - "intra_class_distance_vs_inter_class_distance"

analysis:
  - "Detect phase transition: discontinuous jump in accuracy"
  - "Detect geometry transition: sharp change in PCA structure"
  - "Correlate behavioral transition with geometry transition"
  - "Identify critical context length for each language"

geometry_analysis:
  layers_to_analyze: [0, 5, 11, 17, 23, 25]
  pca_dimensions: 50
  energy_normalization: "per_layer"

pass_criteria:
  - "Identify at least one phase transition per language"
  - "Geometry transition precedes or coincides with behavioral transition"

outputs:
  - "results/1b_final/context_length_geometry_{language}.json"
  - "results/1b_final/figures/fig_phase_transition.pdf"
```

---

### EXP-CRIT-03: Cross-Language Strategy Analysis

**Theoretical Basis:**
- Strategy selection framework (Wurgaft et al., NeurIPS 2025)
- Pretrained prior interaction

**Hypothesis:**
> High-resource languages use generalization strategy; low-resource languages use memorization or fall back to format signal.

**Design:**

```yaml
id: EXP-CRIT-03
name: "Cross-Language Pretrained Prior Interaction"
family: "strategy_analysis"
priority: "CRITICAL"

languages:
  - hindi        # Very High resource
  - telugu       # High resource
  - bengali      # Medium resource
  - marathi      # Medium resource, shared script with Hindi
  - kannada      # Low resource
  - tamil        # Low resource, different scriptfamily
  - gujarati     # Low resource
  - malayalam    # Very Low resource [NEW]

n_items: 50
icl_count: 16
model: "google/gemma-3-1b-it"
precision: "bf16"

measurements:
  behavioral:
    - "exact_match"
    - "judge_acceptable"
    - "content_specificity_ratio"  # (corrupt - zs) / (helpful - zs)
  
  mechanistic:
    - "head_attribution_top_heads"
    - "logit_lens_separation_trajectory"
    - "attention_mass_on_icl_examples"

analysis:
  - "Correlate content_specificity with pretraining corpus size"
  - "Identify strategy boundary: where does content-specificity drop?"
  - "Test script sharing: Marathi vs Hindi"
  - "Test script family: Tamil vs others"

pass_criteria:
  - "Content-specificity correlates with pretraining resource level"
  - "Script sharing (Marathi/Hindi) shows transfer"

outputs:
  - "results/1b_final/language_strategy_analysis.json"
  - "results/1b_final/figures/fig_language_gradient.pdf"
```

---

## Important Experiments (Tier 2)

These experiments extend understanding and provide robustness.

---

### EXP-IMP-01: Prompt Structure Variations

**Theoretical Basis:**
- Strategy framework: prompts affect posterior P(task|context)

**Design:**

```yaml
id: EXP-IMP-01
name: "Prompt Structure Variations"
family: "prompt_engineering"
priority: "HIGH"

languages:
  - hindi
  - telugu

n_items: 30
icl_count: 16
model: "google/gemma-3-1b-it"

prompt_variations:
  - id: "standard"
    format: "namaste -> नमस्ते\nword -> SCRIPT"
    description: "Baseline (current design)"
  
  - id: "instructional"
    format: "Transliterate to Hindi:\nnamaste -> नमस्ते\n..."
    description: "Explicit task specification"
    hypothesis: "Shifts posterior toward generalization"
  
  - id: "arrow_free"
    format: "नमस्ते\nनमस्ते\n..." (output only)
    description: "Removes explicit mapping cue"
    hypothesis: "Forces model to infer task"
  
  - id: "chain_of_thought"
    format: "namaste: na-ma-ste, each sound maps to..."
    description: "Encourages algorithmic processing"
    hypothesis: "Reveals strategy mixture"

measurements:
  - "first_token_probability"
  - "exact_match"
  - "judge_acceptable"
  - "head_attribution_pattern_shift"
  - "logit_lens_trajectory_shift"

analysis:
  - "Compare behavioral metrics across formats"
  - "Detect strategy shifts in mechanistic measurements"

outputs:
  - "results/1b_final/prompt_variations.json"
```

---

### EXP-IMP-02: Temperature Sensitivity (Strategy Mixture)

**Theoretical Basis:**
- Strategy mixture revealed by sampling diversity (Wurgaft et al.)

**Design:**

```yaml
id: EXP-IMP-02
name: "Temperature Sensitivity Sweep"
family: "sampling_analysis"
priority: "HIGH"

languages:
  - hindi
  - telugu

n_items: 50
icl_counts: [4, 8, 16]
temperatures: [0.0, 0.3, 0.7, 1.0, 1.5]
model: "google/gemma-3-1b-it"

measurements:
  - "output_diversity" (entropy over repeated samples)
  - "exact_match_variance"
  - "strategy_indicators":  # Binary presence of memorization vs generalization patterns
    - "output_matches_example_exactly"
    - "output_uses_example_mapping_consistently"
    - "output_is_novel_correct_transliteration"

analysis:
  - "High T should reveal strategy diversity"
  - "Low T should show dominant strategy"
  - "Entropy vs temperature curve reveals mixture weight"

outputs:
  - "results/1b_final/temperature_sensitivity.json"
```

---

### EXP-IMP-03: MLP Contribution Upgrade

**Theoretical Basis:**
- MLP transformation role in coupled mechanism

**Design:**

```yaml
id: EXP-IMP-03
name: "MLP Contribution (N=50, Hindi+Telugu)"
family: "causal_intervention"
priority: "HIGH"

languages:
  - hindi
  - telugu

n_items: 50
icl_count: 16
model: "google/gemma-3-1b-it"

intervention:
  type: "single_layer_mlp_patch"
  layers: [0, 1, 2, ..., 25]  # All 26 layers
  source: "icl_run"
  target: "zs_run"

metrics:
  primary: "first_token_probability_PE"
  bootstrap_ci: true
  n_bootstrap: 1000

pass_criteria:
  - "At least 2 layers show significant non-zero PE (CI excludes zero)"
  - "Global layers (L05, L11, L17, L23) show positive MLP contribution"
  - "Post-global local layers (L12, L15, L16) show negative or near-zero MLP contribution"

outputs:
  - "results/1b_final/mlp_contribution_hindi_n50.json"
  - "results/1b_final/mlp_contribution_telugu_n50.json"
```

---

### EXP-IMP-04: Head Attribution Upgrade

**Theoretical Basis:**
- Induction/FV head identification

**Design:**

```yaml
id: EXP-IMP-04
name: "Head Attribution (N=50)"
family: "causal_intervention"
priority: "HIGH"

languages:
  - hindi
  - telugu

n_items: 50
icl_count: 16
model: "google/gemma-3-1b-it"

intervention:
  type: "single_head_patch"
  # 26 layers × 4 heads = 104 patches per item
  source: "icl_run"
  target: "zs_run"

metrics:
  primary: "fraction_icl_gap_recovered"
  bootstrap_ci: true
  n_bootstrap: 1000

pass_criteria:
  - "Top-3 heads have CI excluding zero"
  - "L11H0 or L14H0 appears in top-5 across both languages"
  - "Cross-language overlap ≥ 2 heads in top-5"

outputs:
  - "results/1b_final/head_attribution_hindi_n50.json"
  - "results/1b_final/head_attribution_telugu_n50.json"
```

---

### EXP-IMP-05: Seed Robustness

**Design:**

```yaml
id: EXP-IMP-05
name: "Head Attribution Seed Robustness"
family: "robustness"
priority: "MEDIUM"

language: hindi
n_items: 30
seeds: [42, 123, 456]
icl_count: 16
model: "google/gemma-3-1b-it"

measurement: "top_heads_per_seed"

pass_criteria:
  - "At least 2 of top-3 heads overlap across all 3 seeds"

outputs:
  - "results/1b_final/head_attribution_seed_robustness.json"
```

---

## Exploratory Experiments (Tier 3)

---

### EXP-EXP-01: Mediation Analysis

**Theoretical Basis:**
- Signal propagation path detection

**Design:**

```yaml
id: EXP-EXP-01
name: "Head-to-MLP Mediation Analysis"
family: "causal_tracing"
priority: "MEDIUM"

language: hindi
n_items: 30
icl_count: 16
model: "google/gemma-3-1b-it"

procedure:
  - "Patch head H_i from ICL→ZS"
  - "Measure effect at all downstream residual stream positions"
  - "Identify which positions carry mediated signal"
  - "Test: L11H0 → residual → L14H0"

analysis:
  - "Compute mediation fraction for each head→position path"
  - "Identify causal chain from attention to output"

outputs:
  - "results/1b_final/mediation_analysis.json"
```

---

### EXP-EXP-02: Attention Pattern Analysis

**Design:**

```yaml
id: EXP-EXP-02
name: "Key Head Attention Patterns"
family: "attention_visualization"
priority: "MEDIUM"

language: hindi
n_items: 10
icl_count: 16
model: "google/gemma-3-1b-it"

heads_to_analyze:
  - L05H0, L05H1
  - L11H0, L11H1
  - L17H0, L17H1
  - L23H0, L23H1

measurements:
  - "% attention_on_icl_examples"
  - "% attention_on_query"
  - "% attention_on_special_tokens"
  - "attention_distribution_across_icl_positions"

comparison:
  - "ICL vs ZS attention pattern changes"
  - "Which positions gain/lose attention"

outputs:
  - "results/1b_final/attention_patterns.json"
```

---

### EXP-EXP-03: ICL Order Shuffling

**Design:**

```yaml
id: EXP-EXP-03
name: "ICL Order Independence"
family: "robustness"
priority: "LOW"

language: hindi
n_items: 30
icl_count: 16
conditions:
  - "original_order"
  - "shuffled_order_seed_42"
  - "shuffled_order_seed_123"
  - "reversed_order"

measurements:
  - "first_token_probability"
  - "exact_match"
  - "judge_acceptable"

hypothesis: "If ICL uses all examples equally, order should not matter"

outputs:
  - "results/1b_final/icl_order_robustness.json"
```

---

### EXP-EXP-04: Density Degradation (Hindi+Telugu, N=30)

**Design:**

```yaml
id: EXP-EXP-04
name: "Density Degradation with Attention Mass"
family: "architectural_analysis"
priority: "HIGH"

languages:
  - hindi
  - telugu

n_items: 30
icl_counts: [4, 8, 16, 32, 48, 64]
model: "google/gemma-3-1b-it"

measurements:
  - "first_token_probability"
  - "global_layer_attention_per_example"
  - "attention_dilution_factor"

pass_criteria:
  - "Clear monotonic decline from 8→48 (dilution)"
  - "Sharp drop at 48→64 (window boundary)"
  - "Telugu shows similar but attenuated pattern"

outputs:
  - "results/1b_final/density_degradation_hindi_telugu_n30.json"
```

---

## Implementation Notes

### Dependencies

All experiments depend on:
```
missing_modules: ["core", "config", "rescue_research"]
```

Before execution, restore or reimplement:
- `core/model_loading.py` - Load Gemma 3 1B with hooks
- `core/activation_hooks.py` - Capture attention/MLP outputs
- `core/patching.py` - Activation patching functions
- `config/evaluation_sets.py` - Evaluation data loading
- `rescue_research/runners/` - Experiment runners

### Modal Deployment

All experiments target Modal A100-40GB:
```python
# Modal app configuration
GPU_TYPE = "A100-40GB"
COMMIT_AFTER_EACH = True
LOG_TO_MODAL = True
```

### Preflight Validation

Before any experiment runs:
```python
def validate_preflight():
    assert module_exists("core"), "Missing: core"
    assert module_exists("config"), "Missing: config"
    assert module_exists("rescue_research"), "Missing: rescue_research"
    assert hf_token_valid(), "Invalid HF token"
    assert modal_authenticated(), "Modal not authenticated"
    return True
```

---

## Summary Table

| ID | Name | Priority | Languages | N | Key Hypothesis |
|----|------|----------|-----------|---|----------------|
| EXP-CRIT-01 | Joint A+M Intervention | CRITICAL | Hi,Te | 50 | Coupled mechanism |
| EXP-CRIT-02 | Context-Length Geometry | CRITICAL | Hi,Te | 30 | Phase transition |
| EXP-CRIT-03 | Cross-Language Strategy | CRITICAL | 8 langs | 50 | Pretrained prior |
| EXP-IMP-01 | Prompt Variations | HIGH | Hi,Te | 30 | Strategy shift |
| EXP-IMP-02 | Temperature Sweep | HIGH | Hi,Te | 50 | Strategy mixture |
| EXP-IMP-03 | MLP Contribution | HIGH | Hi,Te | 50 | MLP role |
| EXP-IMP-04 | Head Attribution | HIGH | Hi,Te | 50 | Universal heads |
| EXP-IMP-05 | Seed Robustness | MEDIUM | Hi | 30 | Stability |
| EXP-EXP-01 | Mediation Analysis | MEDIUM | Hi | 30 | Signal path |
| EXP-EXP-02 | Attention Patterns | MEDIUM | Hi | 10 | Head function |
| EXP-EXP-03 | ICL Order | LOW | Hi | 30 | Order independence |
| EXP-EXP-04 | Density Degradation | HIGH | Hi,Te | 30 | Architectural bottleneck |

---

**Document End**