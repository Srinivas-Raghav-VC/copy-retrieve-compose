# Multiscale Suite

This is the config-driven, reviewer-facing experiment suite for the three-model program:

- `270m`
- `1b`
- `4b`

It is intentionally more modular than `multiscale_modal_suite/`.

## Design principles

- keep `1b` closure and scaling analysis separate
- gate every mechanistic lane behind a behavioral premise check
- keep proxy results separate from behavioral truth
- never hardcode 1B architecture assumptions into the multiscale folder
- make every lane emit an explicit packet or manifest before GPU spend

## Layout

- `configs/` - model facts, split packs, and claim registry
- `runners/` - family-level entrypoints
- `plots/` - paper/static plot builders
- `analysis/` - aggregation and claim-status builders
- `modal/` - Modal deployment entrypoint

## Runner order

1. `runners/00_stage_packet.py`
2. `runners/01_premise_behavior.py`
3. `runners/02_proxy_bridge.py`
4. `runners/03_logit_lens.py`
5. `runners/03b_attention_probes.py`
6. `runners/04_head_attribution.py`
7. `runners/05_component_patching.py`
8. `runners/06_density_window.py`
9. `runners/07_scale_synthesis.py`

## What is executable today

- architecture and premise gating via existing repo kernels
- `1b` final bundle via `run_1b_final_gpu_bundle.py`
- fidelity-calibrated multiscale paper2 lane via `run.py`
- Modal orchestration, manifests, and telemetry
- static execution-summary plots via `plots/render_status_figures.py`

## What remains packetized

- full cross-scale proxy-bridge parity for `270m`
- fully generic multiscale logit-lens/head-attribution/component-patching kernels

Those are represented here as explicit plan packets rather than hidden TODOs.

## Attention probes

The EleutherAI attention-probes idea is included only as a diagnostic family:

- use probes to locate where features are decodable
- use them to narrow the search space for later patching
- do not use them as causal proof of model mechanism
