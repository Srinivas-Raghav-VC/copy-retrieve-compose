#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
OUTDIR="${2:-research/results/autoresearch/loop1_cross_scale_anchor/${MODE}}"
FORCE_FLAG="${FORCE_FLAG:---force}"
TASK_IDS="premise_gate__270m__aksharantar_hin_latin,premise_gate__270m__aksharantar_tel_latin,premise_gate__1b__aksharantar_hin_latin,premise_gate__1b__aksharantar_tel_latin,premise_gate__4b__aksharantar_hin_latin,premise_gate__4b__aksharantar_tel_latin"
VOLUME_NAME="gemma-multiscale-results"
REMOTE_RESULTS_ROOT="/artifacts/multiscale_modal_suite"

mkdir -p "$OUTDIR"

echo "[loop1] verifying multiscale suite wiring"
python3 -m Draft_Results.paper2_fidelity_calibrated.multiscale_modal_suite.verify_suite \
  --out "$OUTDIR/verify_suite.json"

echo "[loop1] launching Modal premise-gate tasks ($MODE)"
MODAL_ARGS=(
  run Draft_Results/paper2_fidelity_calibrated/multiscale_modal_suite/modal_app.py
  --task-ids "$TASK_IDS"
  --wait
  $FORCE_FLAG
)
if [[ "$MODE" == "smoke" ]]; then
  MODAL_ARGS+=(--smoke)
elif [[ "$MODE" != "full" ]]; then
  echo "Unknown mode: $MODE (expected smoke or full)" >&2
  exit 2
fi

modal "${MODAL_ARGS[@]}" | tee "$OUTDIR/modal_run.log"

echo "[loop1] downloading Modal artifacts"
mkdir -p "$OUTDIR/volume"
modal volume get "$VOLUME_NAME" "$REMOTE_RESULTS_ROOT" "$OUTDIR/volume" --force

echo "[loop1] scoring cross-scale premise gate"
python3 experiments/score_cross_scale_anchor.py \
  --results-root "$OUTDIR/volume" \
  --out "$OUTDIR/score.json" | tee "$OUTDIR/score.log"

echo "[loop1] done -> $OUTDIR/score.json"
