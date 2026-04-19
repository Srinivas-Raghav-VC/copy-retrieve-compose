# Claim Ledger (Re-Verification)

Date: 2026-03-22
Scope: 1B/4B transliteration ICL mechanistic claims

## Frozen Protocol Decisions
- Treat prior JSON artifacts as exploratory unless re-confirmed in the current frozen run.
- Use full behavioral metrics as primary outcomes; first-token metrics are mechanistic support only.
- Keep sample-size policy explicit per experiment in run manifests.

## Confirmed by Audit
- Data schema convention is stable in this codebase: rows are normalized to `{"ood", "hindi", "english"}` where `hindi` is a legacy key representing TARGET text for all languages.
- Prompt templates are language-agnostic with dynamic substitutions for source language and output script.
- Five-language coverage is implemented in major runners (hin/tel/ben/kan/guj), with caveats that some pairs rely on external dataset registration/fallback paths.

## Contradictions / Risks Requiring Re-Run
- Attention-only sufficiency claim is contradictory across runs (near-zero in one run vs materially positive in another run context).
- Head attribution universality claim for L11H0 is not consistently supported across all five languages in available prior artifacts.
- Inconsistent N across prior scripts/runs can change conclusions; harmonized N is required for confirmatory claims.

## Result Integrity Cross-Check (Docs vs JSON)

### C1 — "L11H0 is universal cross-language rescue head"
- **Doc claims (overstated):**
  - `results/1b_mechanistic_analysis/1B_COMPLETE_STORY.md` claims L11H0 is universal/top-5 across all 5.
  - `results/1b_mechanistic_analysis/1B_DEEP_ANALYSIS.md` similarly states all 5.
- **JSON evidence (contradicts all-5 wording):**
  - `results/vm_backup/1b_definitive/head_attribution_hindi.json`: L11H0 rank #2 (top-5)
  - `results/vm_backup/1b_definitive/head_attribution_telugu.json`: L11H0 rank #2 (top-5)
  - `results/vm_backup/1b_definitive/head_attribution_bengali.json`: L11H0 rank #3 (top-5)
  - `results/vm_backup/1b_definitive/head_attribution_kannada.json`: L11H0 rank #12 (NOT top-5)
  - `results/vm_backup/1b_definitive/head_attribution_gujarati.json`: L11H0 rank #6 (NOT top-5)
- **Verdict:** **CONTRADICTED** (all-5 universality false). Safer wording: L11H0 is top-5 in 3/5.

### C2 — "L14H0 is most universal head"
- **Doc claims (moderate):** `results/1b_mechanistic_analysis/1B_PAPER_REFERENCE.md` states L14H0 appears in 4/5.
- **JSON evidence:** L14H0 in top-5 for Hindi/Telugu/Bengali/Kannada, not Gujarati.
- **Verdict:** **CONFIRMED** (4/5).

### C3 — "Attention-only replacement is effectively zero"
- **Older evidence (supports near-zero):**
  - `results/1b_definitive/attention_contribution_patching.json`
  - Hindi `all_attn` PE mean = `2.5538e-05` (N=50)
  - Telugu `all_attn` PE mean = `-4.3139e-04` (N=46)
- **Newer evidence (contradicts near-zero):**
  - `results/1b_final_modal_diag/attention_only_contribution_hindi.json`: `all_attn` prob_effect mean = `0.0687`
  - `results/1b_final_modal_diag/attention_only_contribution_telugu.json`: `all_attn` prob_effect mean = `0.1877`
  - This contradiction is explicitly documented in `results/1b_mechanistic_analysis/1B_SCOPE_REVALIDATION_20260318.md`.
- **Verdict:** **UNRESOLVED / SPLIT-SENSITIVE** (cannot freeze as definitive without rerun under a single frozen protocol).

### C4 — MLP layer roles (Hindi)
- **JSON evidence (`results/vm_backup/1b_definitive/mlp_contribution_hindi.json`):**
  - L12: mean `-0.1653`, 95% CI `[-0.2615, -0.0792]` (destructive)
  - L16: mean `-0.0607`, 95% CI `[-0.1209, -0.0083]` (destructive)
  - L17: mean `+0.0402`, 95% CI `[+0.0104, +0.0733]` (constructive)
  - L23: mean `+0.0331`, 95% CI `[+0.0012, +0.0727]` (weak constructive)
  - L25: mean `+0.0459`, 95% CI `[+0.0152, +0.0800]` (constructive/readout)
  - L15 CI crosses zero (`[-0.1159, +0.0127]`).
- **Verdict:** **PARTIALLY CONFIRMED** (core L12/L17/L25 story supported; avoid strong claims on L15).

### C5 — Density degradation / cliff
- **JSON evidence (`results/vm_backup/1b_definitive/density_attention_dilution.json`):**
  - 8 → `0.2703`
  - 16 → `0.2344`
  - 32 → `0.1714`
  - 48 → `0.1316`
  - 64 → `0.0035` (sharp cliff)
- **Verdict:** **CONFIRMED** for this run family; still requires harmonized rerun policy when comparing across papers/scripts.

### C6 — First-token proxy adequacy
- **Docs with direct validation:**
  - `results/1b_mechanistic_analysis/1B_M1_FIRST_TOKEN_PROXY_VALIDATION.md`
  - conclusion: useful mechanistic signal but partial behavioral proxy.
- **Verdict:** **CONFIRMED** (must not present first-token as full behavior metric).

## Bugs / Risk Flags
- **Semantic key hazard:** target text is frequently stored under legacy key `hindi` for all languages. Functionally consistent in current pipeline, but high confusion risk for new scripts/runners.
- **Run-family drift:** different scripts and splits can yield materially different conclusions (especially attention-only claim), so run-manifest freezing is mandatory.

## Proxy Metric Status
- First-token metrics contain signal but are a partial proxy for full transliteration quality.
- Required complementary metrics for all confirmatory claims:
  - full-string exact match
  - normalized CER / edit distance
  - script compliance
  - teacher-forced sequence likelihood metrics

## Detached Re-Verification Run (Current)
- Job: Modal full clean verification battery
- Script: `paper2_fidelity_calibrated/modal_verify_1b.py`
- Launch mode: detached (`nohup`)
- PID: `1520`
- Log: `paper2_fidelity_calibrated/modal_verify_1b.nohup.log`

## Next Interpretation Gates
1. Validate run manifest and language configs in generated artifacts.
2. Recompute claim table with bootstrap CIs using only current frozen run outputs.
3. Mark each claim as `Confirmed`, `Partial`, or `Rejected` with evidence paths.
