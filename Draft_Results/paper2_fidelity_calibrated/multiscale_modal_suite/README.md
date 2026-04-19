# Multiscale Modal Suite

This folder turns the current research state into a reproducible execution layer for three model scales:

- `270m`
- `1b`
- `4b`

It does **not** replace the existing scientific kernels in `paper2_fidelity_calibrated/`. Instead, it wraps them into:

- a source-grounded task registry
- a Modal A100-40GB launch path
- a telemetry and manifest layer
- a marimo dashboard for run inspection
- a verification script that catches broken wiring before GPU spend

## What this suite is for

The suite is organized around two goals:

1. **Close the 1B story** with the high-N final bundle already specified in `1B_FINAL_EXECUTION_PLAN.md`
2. **Build a clean scaling lane** for `270m -> 1b -> 4b` using the existing repo runners for premise-gating, token visibility, script-space maps, fidelity checks, and paper2 intervention runs

## Folder contents

- `suite_spec.py` - canonical task registry and experiment plan
- `legacy_commands.py` - maps each task to an existing repo script
- `runner.py` - emits plans and executes tasks with manifests and logs
- `modal_app.py` - Modal app for A100-40GB runs
- `verify_suite.py` - static verification of the suite wiring
- `dashboard_marimo.py` - marimo dashboard for telemetry inspection
- `plot_style.py` - shared figure palette and rcParams
- `RESEARCH_EXECUTION_BLUEPRINT.md` - the step-by-step research program

## Quick start

Plan the full suite:

```bash
python3 -m Draft_Results.paper2_fidelity_calibrated.multiscale_modal_suite.runner \
  --emit-plan "Draft_Results/paper2_fidelity_calibrated/results/multiscale_modal_suite/plan.json"
```

Verify the suite wiring:

```bash
python3 -m Draft_Results.paper2_fidelity_calibrated.multiscale_modal_suite.verify_suite
```

Launch all tasks on Modal:

```bash
modal run Draft_Results/paper2_fidelity_calibrated/multiscale_modal_suite/modal_app.py
```

Launch only the 1B closure lane:

```bash
modal run Draft_Results/paper2_fidelity_calibrated/multiscale_modal_suite/modal_app.py \
  --lanes 1b_closure
```

Smoke test the suite on Modal:

```bash
modal run Draft_Results/paper2_fidelity_calibrated/multiscale_modal_suite/modal_app.py \
  --smoke
```

Open the marimo dashboard locally:

```bash
marimo run Draft_Results/paper2_fidelity_calibrated/multiscale_modal_suite/dashboard_marimo.py
```

## Required secrets for Modal

Create these before GPU runs:

- `huggingface-secret` with `HF_TOKEN`
- `google-generative-ai` with your Gemini API key for judge-linked lanes

## Important note about repo dependencies

The current workspace snapshot does **not** contain `core.py`, `config.py`, or the `rescue_research` package tree that the legacy scripts import. The suite therefore includes:

- `verify_suite.py` import health checks
- manifest logging for every task
- a clean failure boundary so Modal jobs fail early if the scientific kernel environment is incomplete

In other words: this folder makes the orchestration robust, but the legacy scientific kernels still need the research environment they were originally written against.
