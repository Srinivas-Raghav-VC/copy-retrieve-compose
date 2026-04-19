# Stage 0 / Stage A Runbook

This runbook implements the frozen 2026 workshop protocol for the CFOM paper.

## Non-negotiables

- No silent query-span fallback.
- No selector changes after looking at dev results.
- No broadening Stage A beyond the tiny `attn_out` symmetry check.
- No cutting skeptical controls to save compute.
- No claim stronger than intervention-level evidence from patching/ablation.
- No artifact-aligned wording unless the site sanity packet says `exact_artifact_match`.

## Stage 0

Stage 0 findings are interpreted as **intervention-level evidence only**.
Fidelity and reconstruction summaries are necessary but never sufficient on
their own.

### 1. Site sanity packet

Record:

- exact code-level hook identity
- whether the hooked tensor is pre/post any internal norm
- whether the hooked tensor is pre/post any internal projection
- tensor shape
- norm distribution
- small reconstruction summary

Verdict must be exactly one of:

- `exact_artifact_match`
- `artifact_family_match_but_operational_offset`
- `mismatch_do_not_claim_artifact_alignment`

### 2. Prompt freeze packet

Freeze and save:

- prompt variant
- prompt-template hash / rendered-prompt checksums
- tokenizer name/version
- tokenizer revision
- model name/version
- model revision

Prompt template is frozen after Stage 0.

### 3. Span-localization audit

Report by language:

- number attempted
- number localized successfully
- number invalid under fail-closed rule

Failed items remain in the tokenizer/prompt-audit denominator.

### 4. Tokenizer burden audit

Report:

- fertility
- token-per-akshara ratio
- unbroken ratio
- source-span fragmentation
- target first-token fragmentation
- raw task-text length
- rendered prompt length after chat template
- `local_window_exceeded`

`local_window_exceeded` is a routing-pressure flag, not proof that distant ICL examples are inaccessible.

### 5. Norm-matching sanity

Run:

- norm matching ON
- norm matching OFF
- matched-norm random decoded-vector control

If the effect appears only with norm matching ON, interpret as fidelity/steering-sensitive.

### 6. Premise gate

On Hindi and Telugu, for both 1B and 4B, run:

- `bare_zs`
- `explicit_zs`
- `icl8`
- `icl64`

Operationalize “clearly above floor noise” in the internal runbook as:

- on at least one primary behavioral metric,
- the paired `icl64 - explicit_zs` gap on the unambiguous slice
- has a paired bootstrap CI excluding zero

This is an execution rule, not a paper threshold.

### 7. Stage 0 / Stage A wrapper

Preferred entrypoint:

```bash
python3 paper2_fidelity_calibrated/run_stage0_stagea.py --dry-run
```

This writes:

- `paper2_fidelity_calibrated/results/stage0_stagea_manifest.json`

and prints the exact Stage 0 / Stage A commands without executing them.

## Stage A

### Scope

- languages: Hindi, Telugu
- models: 1B, 4B
- prompt sweep
- coarse layer bracket
- tiny structured `attn_out` symmetry check

### `attn_out` symmetry check

Must use:

- same frozen selector rule
- same fail-closed span rule
- same norm ON/OFF handling

And is capped to:

- 1 anchor language
- 1 or 2 matched layers

### Local stability / mini-consensus check

Reporting only:

- layers `L-1`, `L`, `L+1`
- neighboring top-k values from fixed ladder

This is not a reselection pass.

### Anti-bloat rule

Do not introduce in Stage A:

- crosscoders
- CLT attribution graphs
- activation oracles as evidence
- any new selector rule

## Artifact filenames to preserve

Keep these files under versioned results storage:

- `paper2_fidelity_calibrated/results/stage0_stagea_manifest.json`
- `paper2_fidelity_calibrated/results/stage0_packet_*`
- `paper2_fidelity_calibrated/results/paper2_fidelity_calibrated_*.json`
- `paper2_fidelity_calibrated/results/matrix_summary.json`

## Final paper outcomes

- Positive mechanism-validity paper
- Only affine survives
- Only CLT survives
- Skeptical negative result
