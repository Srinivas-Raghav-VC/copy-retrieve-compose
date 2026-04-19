#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DEFAULT_PYTHON="$ROOT/.venv/bin/python"
if [[ -x "$DEFAULT_PYTHON" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi
DEVICE="${PAPER2_DEVICE:-cuda}"
LOG="${LOG:-paper2_fidelity_calibrated/results/phase3_anchor_pipeline.log}"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

phase2_done() {
  ! pgrep -f 'run_phase2_anchor_pipeline.sh|run_head_to_mlp_edge_attribution.py|run_feature_knockout_panel.py|run_belief_dynamics_fit.py' >/dev/null 2>&1
}

wait_for_file() {
  local path="$1"
  while [[ ! -f "$path" ]]; do
    echo "[$(date +%H:%M:%S)] waiting for $path"
    sleep 30
  done
}

run_step() {
  local name="$1"
  shift
  echo "[$(date +%H:%M:%S)] START $name"
  "$@"
  echo "[$(date +%H:%M:%S)] DONE  $name"
}

echo "[$(date +%H:%M:%S)] Phase-3 anchor pipeline queued"
while ! phase2_done; do
  echo "[$(date +%H:%M:%S)] waiting for Phase-2 queue to finish"
  sleep 60
done

wait_for_file paper2_fidelity_calibrated/results/feature_knockout_panel/aksharantar_hin_latin/4b/feature_knockout_panel.json
wait_for_file paper2_fidelity_calibrated/results/feature_knockout_panel/aksharantar_tel_latin/4b/feature_knockout_panel.json
wait_for_file paper2_fidelity_calibrated/results/feature_knockout_panel/aksharantar_hin_latin/1b/feature_knockout_panel.json
wait_for_file paper2_fidelity_calibrated/results/feature_knockout_panel/aksharantar_tel_latin/1b/feature_knockout_panel.json

echo "[$(date +%H:%M:%S)] Phase-2 prerequisites satisfied; launching Phase-3"

echo "[$(date +%H:%M:%S)] Phase-3 preflight smokes"
run_step smoke_g11_4b "$PYTHON_BIN" paper2_fidelity_calibrated/run_cfom_function_vector_tests.py \
  --model 4b \
  --pairs aksharantar_hin_latin,aksharantar_tel_latin \
  --device "$DEVICE" \
  --max-items 1 \
  --donor-items 4 \
  --external-only \
  --require-external-sources
run_step smoke_g4_4b_hin "$PYTHON_BIN" paper2_fidelity_calibrated/run_circuit_sufficiency.py \
  --model 4b \
  --pair aksharantar_hin_latin \
  --device "$DEVICE" \
  --max-items 1 \
  --core-features 4 \
  --top-n-heads 4 \
  --external-only \
  --require-external-sources
run_step smoke_g5_4b_hin "$PYTHON_BIN" paper2_fidelity_calibrated/run_icl_contribution_curve.py \
  --model 4b \
  --pair aksharantar_hin_latin \
  --device "$DEVICE" \
  --max-items 1 \
  --counts 0,4,64 \
  --external-only \
  --require-external-sources
run_step smoke_g12_4b "$PYTHON_BIN" paper2_fidelity_calibrated/run_sparse_feature_circuit.py \
  --model 4b \
  --pairs aksharantar_hin_latin,aksharantar_tel_latin \
  --device "$DEVICE" \
  --max-items 1 \
  --top-features 4 \
  --external-only \
  --require-external-sources
run_step smoke_g11_1b "$PYTHON_BIN" paper2_fidelity_calibrated/run_cfom_function_vector_tests.py \
  --model 1b \
  --pairs aksharantar_hin_latin,aksharantar_tel_latin \
  --device "$DEVICE" \
  --max-items 1 \
  --donor-items 4 \
  --external-only \
  --require-external-sources
run_step smoke_g4_1b_hin "$PYTHON_BIN" paper2_fidelity_calibrated/run_circuit_sufficiency.py \
  --model 1b \
  --pair aksharantar_hin_latin \
  --device "$DEVICE" \
  --max-items 1 \
  --core-features 4 \
  --top-n-heads 4 \
  --external-only \
  --require-external-sources
run_step smoke_g5_1b_hin "$PYTHON_BIN" paper2_fidelity_calibrated/run_icl_contribution_curve.py \
  --model 1b \
  --pair aksharantar_hin_latin \
  --device "$DEVICE" \
  --max-items 1 \
  --counts 0,4,64 \
  --external-only \
  --require-external-sources
run_step smoke_g12_1b "$PYTHON_BIN" paper2_fidelity_calibrated/run_sparse_feature_circuit.py \
  --model 1b \
  --pairs aksharantar_hin_latin,aksharantar_tel_latin \
  --device "$DEVICE" \
  --max-items 1 \
  --top-features 4 \
  --external-only \
  --require-external-sources

run_step g11_4b "$PYTHON_BIN" paper2_fidelity_calibrated/run_cfom_function_vector_tests.py \
  --model 4b \
  --pairs aksharantar_hin_latin,aksharantar_tel_latin \
  --device "$DEVICE" \
  --external-only \
  --require-external-sources

run_step g4_4b_hin "$PYTHON_BIN" paper2_fidelity_calibrated/run_circuit_sufficiency.py \
  --model 4b \
  --pair aksharantar_hin_latin \
  --device "$DEVICE" \
  --external-only \
  --require-external-sources

run_step g4_4b_tel "$PYTHON_BIN" paper2_fidelity_calibrated/run_circuit_sufficiency.py \
  --model 4b \
  --pair aksharantar_tel_latin \
  --device "$DEVICE" \
  --external-only \
  --require-external-sources

run_step g5_4b_hin "$PYTHON_BIN" paper2_fidelity_calibrated/run_icl_contribution_curve.py \
  --model 4b \
  --pair aksharantar_hin_latin \
  --device "$DEVICE" \
  --external-only \
  --require-external-sources

run_step g5_4b_tel "$PYTHON_BIN" paper2_fidelity_calibrated/run_icl_contribution_curve.py \
  --model 4b \
  --pair aksharantar_tel_latin \
  --device "$DEVICE" \
  --external-only \
  --require-external-sources

run_step g12_4b "$PYTHON_BIN" paper2_fidelity_calibrated/run_sparse_feature_circuit.py \
  --model 4b \
  --pairs aksharantar_hin_latin,aksharantar_tel_latin \
  --device "$DEVICE" \
  --external-only \
  --require-external-sources

run_step g11_1b "$PYTHON_BIN" paper2_fidelity_calibrated/run_cfom_function_vector_tests.py \
  --model 1b \
  --pairs aksharantar_hin_latin,aksharantar_tel_latin \
  --device "$DEVICE" \
  --external-only \
  --require-external-sources

run_step g4_1b_hin "$PYTHON_BIN" paper2_fidelity_calibrated/run_circuit_sufficiency.py \
  --model 1b \
  --pair aksharantar_hin_latin \
  --device "$DEVICE" \
  --external-only \
  --require-external-sources

run_step g4_1b_tel "$PYTHON_BIN" paper2_fidelity_calibrated/run_circuit_sufficiency.py \
  --model 1b \
  --pair aksharantar_tel_latin \
  --device "$DEVICE" \
  --external-only \
  --require-external-sources

run_step g5_1b_hin "$PYTHON_BIN" paper2_fidelity_calibrated/run_icl_contribution_curve.py \
  --model 1b \
  --pair aksharantar_hin_latin \
  --device "$DEVICE" \
  --external-only \
  --require-external-sources

run_step g5_1b_tel "$PYTHON_BIN" paper2_fidelity_calibrated/run_icl_contribution_curve.py \
  --model 1b \
  --pair aksharantar_tel_latin \
  --device "$DEVICE" \
  --external-only \
  --require-external-sources

run_step g12_1b "$PYTHON_BIN" paper2_fidelity_calibrated/run_sparse_feature_circuit.py \
  --model 1b \
  --pairs aksharantar_hin_latin,aksharantar_tel_latin \
  --device "$DEVICE" \
  --external-only \
  --require-external-sources

run_step minimality_4b_hin "$PYTHON_BIN" paper2_fidelity_calibrated/run_minimality_curve.py \
  --model 4b \
  --pair aksharantar_hin_latin \
  --device "$DEVICE" \
  --external-only \
  --require-external-sources
run_step minimality_4b_tel "$PYTHON_BIN" paper2_fidelity_calibrated/run_minimality_curve.py \
  --model 4b \
  --pair aksharantar_tel_latin \
  --device "$DEVICE" \
  --external-only \
  --require-external-sources
run_step minimality_1b_hin "$PYTHON_BIN" paper2_fidelity_calibrated/run_minimality_curve.py \
  --model 1b \
  --pair aksharantar_hin_latin \
  --device "$DEVICE" \
  --external-only \
  --require-external-sources
run_step minimality_1b_tel "$PYTHON_BIN" paper2_fidelity_calibrated/run_minimality_curve.py \
  --model 1b \
  --pair aksharantar_tel_latin \
  --device "$DEVICE" \
  --external-only \
  --require-external-sources

echo "[$(date +%H:%M:%S)] Phase-3 anchor pipeline complete"
