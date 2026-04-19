# Phase 0A Runbook (Manifest-first)

This runbook is the paper-facing entrypoint for reproducible Phase 0A runs.

## 1) Config source of truth

Use:

- `research/config/phase0a_run_config.json`

Key sections:
- seeds
- language_codes
- n_values
- prompt_templates
- icl_variants
- judge settings
- `cross_seed_disjoint` (set `true` for seed-to-seed non-overlap)

## 2) Build/refresh snapshots only (clean stage)

```bash
python research/modules/modal/run_phase0a_from_config.py \
  --config research/config/phase0a_run_config.json \
  --build-only
```

## 3) Run all seeds on Modal

```bash
python research/modules/modal/run_phase0a_from_config.py \
  --config research/config/phase0a_run_config.json
```

Recommended for durable local artifacts: run the manifest orchestrator directly (foreground or under `nohup`/tmux).
Avoid relying on `modal run --detach` for the local entrypoint when you need local post-processing files, because detached local-entrypoint mode only guarantees the last remote Modal function keeps running after disconnect; local packet writing/aggregation may not finish.

This will:
1. materialize unique Aksharantar snapshots for each (seed, language)
2. execute Modal Phase 0A runs per seed
3. write per-seed result JSONs into `packets/`
4. write per-seed tables into `tables/`
5. run aggregation + statistical tests
6. generate plots into `plots/`

## 4) Judge rate-limit safe default

In the default config:

```json
"judge": {
  "enabled": false,
  "model": "gemini-2.0-flash-lite",
  "probe_per_condition": 0
}
```

This keeps deterministic evaluation fully active while avoiding free-tier API failures.

## 5) Optional judge-enabled subset run

Enable judge only for a smaller matrix (e.g., one seed, one language) by editing config:
- set `judge.enabled=true`
- keep `probe_per_condition` small (e.g., 3-5)

## 6) Clean output layout

Default output root in current config:

- `research/results/phase0_clean_modal/run_v1/`

Generated structure:
- `run_config_source.json`
- `run_config_resolved.json`
- `packets/phase0a_packet_results_seed*.json`
- `tables/table_phase0_*_seed*.csv`
- `tables/table_phase0_seed_aggregate.csv`
- `stats/phase0_stat_tests.json`
- `stats/table_phase0_stat_tests.csv`
- `plots/seed*/fig_phase0_*.png`

## 7) Optional manual plotting for a single packet

```bash
python research/modules/eval/plot_phase0_curves.py \
  --input research/results/phase0_clean_modal/run_v1/packets/phase0a_packet_results_seed42.json \
  --prompt-template canonical \
  --icl-variant helpful \
  --out-dir research/results/phase0_clean_modal/run_v1/plots/manual
```
