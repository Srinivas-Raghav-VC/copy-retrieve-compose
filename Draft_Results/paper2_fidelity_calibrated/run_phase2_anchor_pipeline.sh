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
LOG="${LOG:-paper2_fidelity_calibrated/results/phase2_anchor_pipeline.log}"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

phase1_done() {
  ! pgrep -f 'paper2_fidelity_calibrated/run_stage0_stagea.py|paper2_fidelity_calibrated/run_attribution_graph_pair.py|paper2_fidelity_calibrated/run_logit_lens_rescue_trajectory.py|paper2_fidelity_calibrated/run_causal_head_attention_patterns.py|paper2_fidelity_calibrated/run_language_script_feature_suppression.py|phase1_followup_1b_20260311.sh' >/dev/null 2>&1
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

echo "[$(date +%H:%M:%S)] Phase-2 anchor pipeline queued"
while ! phase1_done; do
  echo "[$(date +%H:%M:%S)] waiting for Phase-1 queue to finish"
  sleep 60
done

wait_for_file paper2_fidelity_calibrated/results/aksharantar_hin_latin/1b/paper2_fidelity_calibrated_1b.json
wait_for_file paper2_fidelity_calibrated/results/aksharantar_tel_latin/1b/paper2_fidelity_calibrated_1b.json
wait_for_file artifacts/phase5_attribution/top_heads_1b_hin_multilang.json
wait_for_file artifacts/phase5_attribution/top_heads_1b_tel_multilang.json

echo "[$(date +%H:%M:%S)] Phase-1 prerequisites satisfied; launching Phase-2"

echo "[$(date +%H:%M:%S)] Phase-2 preflight smokes"
run_step smoke_g3_4b_hin "$PYTHON_BIN" paper2_fidelity_calibrated/run_head_to_mlp_edge_attribution.py \
  --model 4b \
  --pair aksharantar_hin_latin \
  --device "$DEVICE" \
  --top-heads-json artifacts/phase5_attribution/top_heads_4b_hin_multilang.json \
  --top-n-heads 4 \
  --max-items 1 \
  --external-only \
  --require-external-sources
run_step smoke_g6_4b_hin "$PYTHON_BIN" paper2_fidelity_calibrated/run_feature_knockout_panel.py \
  --model 4b \
  --pair aksharantar_hin_latin \
  --device "$DEVICE" \
  --max-items 1 \
  --max-features 2 \
  --external-only \
  --require-external-sources
run_step smoke_g3_1b_hin "$PYTHON_BIN" paper2_fidelity_calibrated/run_head_to_mlp_edge_attribution.py \
  --model 1b \
  --pair aksharantar_hin_latin \
  --device "$DEVICE" \
  --top-heads-json artifacts/phase5_attribution/top_heads_1b_hin_multilang.json \
  --top-n-heads 4 \
  --max-items 1 \
  --external-only \
  --require-external-sources
run_step smoke_g6_1b_hin "$PYTHON_BIN" paper2_fidelity_calibrated/run_feature_knockout_panel.py \
  --model 1b \
  --pair aksharantar_hin_latin \
  --device "$DEVICE" \
  --max-items 1 \
  --max-features 2 \
  --external-only \
  --require-external-sources
run_step smoke_g10 "$PYTHON_BIN" paper2_fidelity_calibrated/run_belief_dynamics_fit.py \
  --out /tmp/belief_dynamics_fit_phase2_smoke.json

run_step g3_4b_hin "$PYTHON_BIN" paper2_fidelity_calibrated/run_head_to_mlp_edge_attribution.py \
  --model 4b \
  --pair aksharantar_hin_latin \
  --device "$DEVICE" \
  --top-heads-json artifacts/phase5_attribution/top_heads_4b_hin_multilang.json \
  --top-n-heads 8 \
  --max-items 30 \
  --external-only \
  --require-external-sources

run_step g6_4b_hin "$PYTHON_BIN" paper2_fidelity_calibrated/run_feature_knockout_panel.py \
  --model 4b \
  --pair aksharantar_hin_latin \
  --device "$DEVICE" \
  --max-items 30 \
  --max-features 8 \
  --external-only \
  --require-external-sources

run_step g3_4b_tel "$PYTHON_BIN" paper2_fidelity_calibrated/run_head_to_mlp_edge_attribution.py \
  --model 4b \
  --pair aksharantar_tel_latin \
  --device "$DEVICE" \
  --top-heads-json artifacts/phase5_attribution/top_heads_4b_tel_multilang.json \
  --top-n-heads 8 \
  --max-items 30 \
  --external-only \
  --require-external-sources

run_step g6_4b_tel "$PYTHON_BIN" paper2_fidelity_calibrated/run_feature_knockout_panel.py \
  --model 4b \
  --pair aksharantar_tel_latin \
  --device "$DEVICE" \
  --max-items 30 \
  --max-features 8 \
  --external-only \
  --require-external-sources

run_step g3_1b_hin "$PYTHON_BIN" paper2_fidelity_calibrated/run_head_to_mlp_edge_attribution.py \
  --model 1b \
  --pair aksharantar_hin_latin \
  --device "$DEVICE" \
  --top-heads-json artifacts/phase5_attribution/top_heads_1b_hin_multilang.json \
  --top-n-heads 8 \
  --max-items 30 \
  --external-only \
  --require-external-sources

run_step g6_1b_hin "$PYTHON_BIN" paper2_fidelity_calibrated/run_feature_knockout_panel.py \
  --model 1b \
  --pair aksharantar_hin_latin \
  --device "$DEVICE" \
  --max-items 30 \
  --max-features 8 \
  --external-only \
  --require-external-sources

run_step g3_1b_tel "$PYTHON_BIN" paper2_fidelity_calibrated/run_head_to_mlp_edge_attribution.py \
  --model 1b \
  --pair aksharantar_tel_latin \
  --device "$DEVICE" \
  --top-heads-json artifacts/phase5_attribution/top_heads_1b_tel_multilang.json \
  --top-n-heads 8 \
  --max-items 30 \
  --external-only \
  --require-external-sources

run_step g6_1b_tel "$PYTHON_BIN" paper2_fidelity_calibrated/run_feature_knockout_panel.py \
  --model 1b \
  --pair aksharantar_tel_latin \
  --device "$DEVICE" \
  --max-items 30 \
  --max-features 8 \
  --external-only \
  --require-external-sources

run_step g10_belief_dynamics "$PYTHON_BIN" paper2_fidelity_calibrated/run_belief_dynamics_fit.py

run_step shift_4b_hin "$PYTHON_BIN" paper2_fidelity_calibrated/run_position_shift_sanity.py \
  --model 4b \
  --pair aksharantar_hin_latin \
  --device "$DEVICE" \
  --max-items 20 \
  --external-only \
  --require-external-sources
run_step shift_4b_tel "$PYTHON_BIN" paper2_fidelity_calibrated/run_position_shift_sanity.py \
  --model 4b \
  --pair aksharantar_tel_latin \
  --device "$DEVICE" \
  --max-items 20 \
  --external-only \
  --require-external-sources
run_step shift_1b_hin "$PYTHON_BIN" paper2_fidelity_calibrated/run_position_shift_sanity.py \
  --model 1b \
  --pair aksharantar_hin_latin \
  --device "$DEVICE" \
  --max-items 20 \
  --external-only \
  --require-external-sources
run_step shift_1b_tel "$PYTHON_BIN" paper2_fidelity_calibrated/run_position_shift_sanity.py \
  --model 1b \
  --pair aksharantar_tel_latin \
  --device "$DEVICE" \
  --max-items 20 \
  --external-only \
  --require-external-sources

run_step stability_4b_hin "$PYTHON_BIN" paper2_fidelity_calibrated/run_feature_stability_resamples.py \
  --model 4b \
  --pair aksharantar_hin_latin \
  --device "$DEVICE" \
  --external-only \
  --require-external-sources
run_step stability_4b_tel "$PYTHON_BIN" paper2_fidelity_calibrated/run_feature_stability_resamples.py \
  --model 4b \
  --pair aksharantar_tel_latin \
  --device "$DEVICE" \
  --external-only \
  --require-external-sources
run_step stability_1b_hin "$PYTHON_BIN" paper2_fidelity_calibrated/run_feature_stability_resamples.py \
  --model 1b \
  --pair aksharantar_hin_latin \
  --device "$DEVICE" \
  --external-only \
  --require-external-sources
run_step stability_1b_tel "$PYTHON_BIN" paper2_fidelity_calibrated/run_feature_stability_resamples.py \
  --model 1b \
  --pair aksharantar_tel_latin \
  --device "$DEVICE" \
  --external-only \
  --require-external-sources

echo "[$(date +%H:%M:%S)] Phase-2 anchor pipeline complete"
