# Multiscale Research Execution Blueprint

## Purpose

This blueprint turns the current project state into a final research program for:

- `270m` as the low-capacity contrast model
- `1b` as the main mechanistic target
- `4b` as the scale-up comparison

The goal is not to produce a single blob of experiments. The goal is to create a claim-bearing evidence map that can survive top-tier ML review.

## What is already established

### Hard facts

- Gemma 3 1B architecture has been confirmed from authenticated config extraction
- `1b` uses a `512` token local sliding window with global layers at `L05`, `L11`, `L17`, `L23`
- the current repo already contains a strong 1B evidence base and a dedicated final execution plan

### Strong empirical footing

- 1B has a real behavioral premise gap under ICL
- 1B shows content-sensitive internal rescue for at least some languages
- first-token metrics are useful but not equivalent to full transliteration behavior
- the best current mechanism framing is distributed attention plus downstream computation under locality constraints

### What remains unresolved

- the strongest attention-only negative claim still needs careful final freeze language
- scaling behavior from `270m -> 1b -> 4b` is not yet integrated into one evidence map
- non-Indic extension is desirable, but must not be allowed to dilute the core causal lane unless data provenance is strong

## Research lanes

### Lane A - Architecture and premise

1. `phase0_token_visibility`
2. `premise_gate`
3. `script_space_map`

These tasks answer:

- what each model can see
- whether the pair is mechanistically worth studying
- where native-script preference appears in the stack

### Lane B - 1B closure

Use the existing `run_1b_final_gpu_bundle.py` task family on Modal A100-40GB to generate:

- high-N MLP contribution for Hindi and Telugu
- high-N head attribution
- density degradation replication
- grouped joint attention+MLP interventions
- content-specificity by count
- seed robustness

This lane is the core of the final paper story.

### Lane C - Multiscale intervention evidence

Use the existing `run.py` paper2 intervention lane for `270m`, `1b`, and `4b` on the same core pairs.

This lane does **not** replace the 1B closure lane. It supplies the scaling story around:

- whether the premise gap grows or shrinks with scale
- whether fidelity-calibrated interventions become cleaner with scale
- whether the 1B story appears as a transition point between weak small-scale behavior and stronger 4B organization

### Lane D - Optional breadth and non-Indic extension

Only run this lane after core causal tasks are green.

- Indic breadth: Bengali, Gujarati, Kannada
- Non-Indic extension: Arabic, Cyrillic, or other externally sourced transliteration pairs

This lane is support, not the main claim-bearing center.

## Suggested sample sizes

These choices are deliberately stronger than the current minimums.

| Lane | Core pairs | N | Notes |
|---|---:|---:|---|
| Premise gate | Hi/Te x 3 models | 200 | reject dead lanes early |
| Phase 0 visibility | Hi/Te x 3 models | 200 | architecture sanity with real prompts |
| Script-space map | Hi/Te x 3 models | 50 words | representation lane, not main statistical claim |
| 1B closure bundle | Hindi + Telugu | 50 / 30 by task | inherits current final plan |
| Paper2 scaling lane | Hi/Te x 3 models | 200 eval | identical settings across scales |
| Breadth Indic | 3 languages | 50-100 | only after core lane |
| Non-Indic extension | 2 languages | 50-100 | only with external provenance |

## Evidence policy

Every final claim should be labeled as one of:

- `architecture fact`
- `behavioral fact`
- `proxy measurement`
- `causal intervention result`
- `interpretive synthesis`

The paper should never blur these categories.

## Final paper-ready endpoint

The multiscale story is ready when all of the following are true:

1. the 1B closure lane is green with clean manifests
2. the multiscale premise lane shows where `270m`, `1b`, and `4b` genuinely differ
3. the intervention lane supplies a scaling comparison that is not cherry-picked
4. every figure and dashboard panel points back to a concrete artifact file
5. the final discussion says exactly what is known, what is only suggestive, and what remains unresolved

## Recommended final interpretation shape

- `270m` = weak or noisy rescue, useful lower-bound contrast
- `1b` = best regime for seeing rescue emerge under strong architectural constraints
- `4b` = scale-up test of whether the 1B mechanism generalizes, diffuses, or changes character

That is the right path to a complete picture without pretending we already have one.
