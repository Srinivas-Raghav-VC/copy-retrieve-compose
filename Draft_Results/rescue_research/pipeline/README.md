# Pipeline Package

Main modules:

- `dag.py`: stage order and top-level pipeline execution.
- `stages.py`: concrete stage implementations.
- `artifact_contracts.py`: required directories and artifact paths.
- `schema.py`: manifest schema requirements.
- `validator.py`: post-stage artifact validation.

The pipeline is invoked from:

`python -m rescue_research.run --pipeline full_confirmatory ...`

