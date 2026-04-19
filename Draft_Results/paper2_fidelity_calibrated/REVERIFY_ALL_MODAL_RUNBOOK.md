# Re-Verify All (Modal-Only) Runbook

Date: 2026-03-22

## Objective
Run one coordinated, publication-rigor re-verification pass on Modal (no VM dependency), then refresh the claim ledger from fresh artifacts only.

## Why this path
- VM path is excluded due GPU reliability concerns.
- Direct HF Aksharantar loader is currently brittle on Modal (`default` config + schema mismatch).
- The multiscale Modal suite already mounts the project workspace and runs project-native tasks.

## Launched job
- App URL: `https://modal.com/apps/academic/main/ap-62h5kBkGelrOcByLNEC63p`
- Local launcher log: `paper2_fidelity_calibrated/modal_one_b_final_bundle.nohup.log`
- Command (detached):

```bash
modal run -m paper2_fidelity_calibrated.multiscale_modal_suite.modal_app \
  --task-ids one_b_final_bundle --force
```

## Bundle scope (one_b_final_bundle)
Expected outputs under Modal artifacts:
- `1b_final/hindi_mlp_contribution_n50.json`
- `1b_final/telugu_mlp_contribution_n50.json`
- `1b_final/density_degradation_hindi_telugu_n30.json`
- `1b_final/head_attribution_hindi_n50.json`
- `1b_final/head_attribution_telugu_n50.json`
- `1b_final/joint_attn_mlp_grouped_hindi.json`
- `1b_final/joint_attn_mlp_grouped_telugu.json`
- `1b_final/content_specificity_by_count.json`
- `1b_final/head_attribution_seed_robustness.json`

## Acceptance gates (from Oracle + audit)
1. **Target-label integrity gate**
   - Confirm target extraction is correct for each pair in manifests.
2. **Metric validity gate**
   - First-token metrics must be reported as support metrics only.
   - Full behavioral metrics required in interpretation.
3. **Causal claim gate**
   - Coupled attention+MLP claim requires joint intervention evidence.
4. **Statistical gate**
   - Report bootstrap CIs and do not mix incompatible N across comparisons.

## Post-run checklist
1. Pull artifacts from Modal volume.
2. Recompute a fresh claim table from new outputs only.
3. Update `CLAIM_LEDGER_REVERIFY_2026-03-22.md` with `Confirmed/Partial/Rejected` per claim.
4. Regenerate summary figures and markdown narrative with conservative wording.
