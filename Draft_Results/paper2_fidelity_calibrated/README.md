# Paper 2: Fidelity-Calibrated Feature Interventions

This folder contains the publication-oriented runner for the Paper 2 workshop lane:

- external-data build to `fresh_experiments/data/transliteration/`
- provenance enforcement (no builtin-only runs)
- fidelity sanity-check (decode vs intervention target at hookpoint)
- fidelity-aware joint selection over `(variant, layer, top-k)` on a held-out selection split
- held-out evaluation and plotting

This is the strongest current path to a publishable workshop paper in this
repo. It is not a guarantee of main-track evidence by itself; that still
depends on powered external datasets, fidelity passing at the actual hookpoint,
and stable held-out effects.

The frozen protocol and Stage 0 / Stage A execution guardrails now live in:

- `paper2_fidelity_calibrated/STAGE0_STAGE_A_RUNBOOK.md`

Preferred wrapper for Stage 0 / Stage A:

```bash
python3 paper2_fidelity_calibrated/run_stage0_stagea.py --dry-run
```

Then run the same command without `--dry-run` on the target GPU host.

Bounded Stage A.5 diagnostic tranche:

```bash
python3 paper2_fidelity_calibrated/run_stagea5.py \
  --model 4b \
  --pairs aksharantar_hin_latin,aksharantar_tel_latin \
  --device cuda \
  --external-only \
  --require-external-sources \
  --min-pool-size 500
```

This tranche does not reopen selection or expand languages. It reuses the
best Stage A configuration per anchor pair and tests only:
- source-side `last_subtoken` vs first target-position patching
- raw vs clipped vs sign-normalized patch geometry
- entry-vs-continuation diagnostics for target positions 1–3
- a tiny structured `attn_out` fairness check

Current-scope one-shot GPU upgrade entry point (for the paper-strengthening queue after the base artifacts exist):

```bash
bash paper2_fidelity_calibrated/run_current_scope_paper_upgrade_pipeline.sh
```

This wrapper keeps the current paper scope fixed and does the following end to end:
- repairs missing Stage-A prerequisites if needed,
- materializes missing 1B/4B head-attribution artifacts,
- runs the queued Phase-2 / Phase-3 / Phase-4 stacks,
- runs the dense positive-control and activation-difference controls,
- and finishes with the Phase-5 audit / synthesis pass.

Optional bounded multilingual appendix lane (post-Stage A / post-Stage A.5):

```bash
bash paper2_fidelity_calibrated/run_multilang_head_group_appendix.sh
python3 paper2_fidelity_calibrated/summarize_head_group_appendix.py
```

This appendix lane is intentionally separate from the core workshop claim. It
adds a narrow set of attribution-patching and localization experiments:
- pair-specific head attribution ranking
- shared-vs-specific grouped head ablation
- shared-head sufficiency and additive-synergy patch panels
- bounded component-localization and layer-output alignment checks

Interpret these as supportive mechanistic structure probes, not as broader
claim expansion beyond the intervention-level Paper 2 framing.

Authoritative final research spec:

- `docs/PAPER2_FINAL_RESEARCH_READY_SPEC.md`

## Install deps (in your CUDA env)

From `fresh_experiments/`:

```bash
pip install -r requirements.txt
```

## Step 1: Build external datasets (writes JSONL)

This script downloads and materializes a dataset split into JSONL files that the
existing ingestion pipeline will auto-discover under `data/transliteration/`.

```bash
python3 paper2_fidelity_calibrated/build_external_dataset.py --help
```

Example (requires you to specify columns for your dataset):

```bash
python3 paper2_fidelity_calibrated/build_external_dataset.py \
  --pair-id hindi_telugu \
  --hf-dataset ai4bharat/Aksharantar \
  --hf-split train \
  --source-col source \
  --target-col target \
  --id-col english \
  --out data/transliteration/hindi_telugu.jsonl
```

### Alternative: build synthetic datasets for debugging only

If you need a local stress test, you can build larger synthetic pools via
`wordfreq` + transliteration libraries:

```bash
python3 paper2_fidelity_calibrated/build_synthetic_datasets.py \
  --pairs hindi_telugu,hindi_tamil,english_arabic,english_cyrillic \
  --out-dir data/transliteration \
  --min-rows 500 \
  --max-rows 5000
```

These synthetic datasets are useful for dry-runs and failure analysis. They are
not strong primary evidence for a publishable claim.

## Step 2: Verify ingestion sees external sources (recommended)

```bash
python3 paper2_fidelity_calibrated/inspect_pair_data.py \
  --pair hindi_telugu \
  --external-only
```

## Step 3: Fidelity sanity-check (fast)

```bash
python3 paper2_fidelity_calibrated/fidelity_sanity_check.py \
  --model 4b \
  --pair hindi_telugu \
  --layers 0,8,16,24,32 \
  --n-samples 64 \
  --device cuda
```

Outputs:
- `paper2_fidelity_calibrated/results/fidelity_<pair>_<model>.csv`
- `paper2_fidelity_calibrated/figures/fig_fidelity_vs_layer_<pair>_<model>.png`

## Step 4: Run Paper 2 (fidelity-aware adaptive selection)

```bash
python3 paper2_fidelity_calibrated/run.py \
  --model 4b \
  --pair hindi_telugu \
  --seeds 42,123,456 \
  --device cuda \
  --require-external-sources \
  --external-only \
  --min-pool-size 500
```

Recommended workshop-budget profile:
- models: `1b,4b`
- pairs: 2-3 externally sourced pairs
- seeds: `42,123,456`
- variants: `skipless_or_non_affine` by default; treat affine as a later artifact-ladder comparison
- ICL bank / selection / eval defaults now target the frozen workshop protocol:
  - `n_icl = 64`
  - `n_select = 300`
  - `n_eval = 200`
- default top-k ladder: `4,8,16,32`
- selector baseline: `icl64` vs plausibility-matched `corrupt_icl64`
- mainline patch policy: `last_subtoken`
- span-localization: fail closed
- norm matching: explicit ON/OFF ablation
- claim: reliability of sparse intervention evidence, not broad multilingual mechanism discovery

## Step 5: Matrix run + synthesis

```bash
python3 paper2_fidelity_calibrated/run_matrix.py --help
python3 paper2_fidelity_calibrated/summarize_matrix.py
python3 paper2_fidelity_calibrated/make_figures.py
```

For publication-facing matrix runs, override `--pairs` with the externally
backed, powered pair set you actually intend to claim on. The matrix runner now
defaults to the registered workshop-oriented Aksharantar pairs rather than the
old legacy coverage template.

## Final paper outcomes

- Positive mechanism-validity paper
- Only affine survives
- Only CLT survives
- Skeptical negative result
