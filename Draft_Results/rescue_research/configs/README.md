# Pipeline Configs

These files mirror the locked main-track protocol in machine-readable form.

- `models.yaml`: model roles + affine/skip comparison policy.
- `pairs.yaml`: locked 4-pair matrix and approved backups.
- `stats.yaml`: hypotheses, estimands, correction policy, practical floor.
- `modal.yaml`: Modal execution guardrails and retry/checkpoint defaults.
- `prompts.yaml`: prompt/ICL policy and decoding constraints.
- `datasets.yaml`: optional external transliteration sources and data-root policy.

The runtime pipeline currently reads config through `PipelineConfig` (Python
dataclass), and these YAML files serve as auditable protocol snapshots.

