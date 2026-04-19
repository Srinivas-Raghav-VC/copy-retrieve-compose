# copy-retrieve-compose

Research code for analyzing copy, retrieval, and composition regimes in multilingual in-context learning.

## What this repo contains

This repository combines the code used to study multilingual transliteration as a model organism for in-context learning (ICL), with a focus on:

- behavioral regime analysis
- mechanistic localization
- intervention and patching experiments
- reproducible evaluation utilities

The current codebase is organized around a few main areas:

- `experiments/` — runnable analysis and experiment scripts
- `research/modules/` — cleaner reusable modules for data, prompts, evaluation, and infrastructure
- `tests/` — lightweight objective checks for the reusable evaluation and prompt stack
- `Draft_Results/` — older or supporting pipelines, legacy scripts, and dataset preparation utilities retained for reproducibility and comparison
- `research/` — project specs, journals, and supporting research-operating-system documents

The main code paths today are in `experiments/`, `research/modules/`, and `tests/`.

## Quick checks

From the repo root, a small validation subset is:

```bash
pytest -q \
  tests/test_eval_pipeline.py \
  tests/test_output_extraction.py \
  tests/test_prompt_variants.py \
  tests/test_run_config.py
```

The repository excludes local caches, virtual environments, agent state, and most generated results / logs.
