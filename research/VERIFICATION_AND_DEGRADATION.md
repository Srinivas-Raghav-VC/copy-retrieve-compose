# Verification and Degradation

## Verification philosophy
Behavioral verification must be mostly deterministic.
LLM judging is for the ambiguous middle, not the entire evaluation pipeline.

## Verification stack

### Tier 0: normalization
- use script-aware normalization
- log normalization choices
- keep language-specific distinctions unless they are deliberately collapsed

### Tier 1: deterministic hard checks
- exact match after canonical normalization
- target-script validity
- empty / refusal / explanation / copied-input detection
- standalone-answer check

### Tier 2: variant-aware deterministic checks
- grapheme-aware CER
- normalized edit distance
- optional shared-Latin comparison if robust
- prefix correctness only as a diagnostic
- syllable or grapheme-cluster error analysis where feasible

### Tier 3: ambiguous-middle judge
Only unresolved cases should go to an LLM judge.
Judge labels:
- exact
- acceptable_variant
- script_correct_but_wrong
- invalid_or_non_answer

### Tier 4: disagreement audit
Manually audit cases where:
- deterministic metrics disagree
- judge disagrees with deterministic checks
- first-token proxy disagrees with full-output behavior

## Behavioral metric hierarchy

### Primary metrics
- exact_match_rate
- acceptable_rate
- grapheme_CER
- script_validity_rate
- label_mix distribution

### Diagnostic metrics
- first_token_probability
- first_token_rank / inverse rank
- prefix_match_length
- first-entry correctness

Rule:
Never elevate first-entry correctness to a main behavioral metric.

## Degradation decomposition
Treat degradation as a decomposed object, not one vague complaint.

Axes:
1. example count
2. prompt token length
3. prompt density
4. example position
5. local/global visibility regime
6. attention dilution
7. prompt confusion
8. example quality
9. continuation failure after early success

## Required degradation outputs
Always try to produce:
- performance vs N
- performance vs token length
- performance vs density
- visibility tables by layer and prompt regime
- example-position sensitivity
- divergence plots for first-token vs full-output metrics
- error buckets for high-confidence first-token / bad-full-output cases

## Control families
At minimum, consider:
- zero-shot
- helpful ICL
- corrupt ICL length-matched
- format-only ICL
- target-script-only exemplars
- source shuffle
- target shuffle
- ordering controls
- same N / different token length
- same token length / different N
- local-window-accessible vs global-only-accessible examples
- repeated helpful example
- mixed-quality examples
- mixed-language interference

For each control, state:
- what nuisance it isolates
- what remains confounded
