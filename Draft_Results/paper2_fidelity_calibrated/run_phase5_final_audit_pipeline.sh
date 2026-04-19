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
LOG="${LOG:-paper2_fidelity_calibrated/results/phase5_final_audit_pipeline.log}"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

phase4_done() {
  ! pgrep -f 'run_phase4_robustness_pipeline.sh|run_transcoder_family_consensus.py|run_direct_icl_feature_necessity.py|run_patch_geometry_robustness.py|run_cross_artifact_feature_stability.py|run_target_competitor_logit_gap.py|run_leave_k_out_icl_contribution.py|run_decoded_vs_latent_equivalence.py|run_neighbor_layer_causality.py|run_head_attribution_stability.py|run_neutral_filler_recency_controls.py|run_induction_style_head_reanalysis.py' >/dev/null 2>&1
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

echo "[$(date +%H:%M:%S)] Phase-5 final audit pipeline queued"
while ! phase4_done; do
  echo "[$(date +%H:%M:%S)] waiting for Phase-4 queue to finish"
  sleep 60
done

wait_for_file paper2_fidelity_calibrated/results/transcoder_family_consensus/aksharantar_hin_latin/4b/transcoder_family_consensus.json
wait_for_file paper2_fidelity_calibrated/results/transcoder_family_consensus/aksharantar_tel_latin/4b/transcoder_family_consensus.json
wait_for_file paper2_fidelity_calibrated/results/transcoder_family_consensus/aksharantar_hin_latin/1b/transcoder_family_consensus.json
wait_for_file paper2_fidelity_calibrated/results/transcoder_family_consensus/aksharantar_tel_latin/1b/transcoder_family_consensus.json
wait_for_file paper2_fidelity_calibrated/results/dense_mlp_sweep/aksharantar_hin_latin/4b/dense_layer_sweep_results.json
wait_for_file paper2_fidelity_calibrated/results/dense_mlp_sweep/aksharantar_tel_latin/4b/dense_layer_sweep_results.json
wait_for_file paper2_fidelity_calibrated/results/dense_mlp_sweep/aksharantar_hin_latin/1b/dense_layer_sweep_results.json
wait_for_file paper2_fidelity_calibrated/results/dense_mlp_sweep/aksharantar_tel_latin/1b/dense_layer_sweep_results.json
wait_for_file paper2_fidelity_calibrated/results/activation_difference_baseline/aksharantar_hin_latin/4b/activation_difference_baseline.json
wait_for_file paper2_fidelity_calibrated/results/activation_difference_baseline/aksharantar_tel_latin/4b/activation_difference_baseline.json
wait_for_file paper2_fidelity_calibrated/results/activation_difference_baseline/aksharantar_hin_latin/1b/activation_difference_baseline.json
wait_for_file paper2_fidelity_calibrated/results/activation_difference_baseline/aksharantar_tel_latin/1b/activation_difference_baseline.json
wait_for_file paper2_fidelity_calibrated/results/head_attribution_stability/aksharantar_hin_latin/4b/head_attribution_stability.json
wait_for_file paper2_fidelity_calibrated/results/head_attribution_stability/aksharantar_tel_latin/4b/head_attribution_stability.json
wait_for_file paper2_fidelity_calibrated/results/head_attribution_stability/aksharantar_hin_latin/1b/head_attribution_stability.json
wait_for_file paper2_fidelity_calibrated/results/head_attribution_stability/aksharantar_tel_latin/1b/head_attribution_stability.json
wait_for_file paper2_fidelity_calibrated/results/neutral_filler_recency_controls/aksharantar_hin_latin/4b/neutral_filler_recency_controls.json
wait_for_file paper2_fidelity_calibrated/results/neutral_filler_recency_controls/aksharantar_tel_latin/4b/neutral_filler_recency_controls.json
wait_for_file paper2_fidelity_calibrated/results/neutral_filler_recency_controls/aksharantar_hin_latin/1b/neutral_filler_recency_controls.json
wait_for_file paper2_fidelity_calibrated/results/neutral_filler_recency_controls/aksharantar_tel_latin/1b/neutral_filler_recency_controls.json
wait_for_file paper2_fidelity_calibrated/results/induction_style_head_reanalysis/aksharantar_hin_latin/4b/induction_style_head_reanalysis.json
wait_for_file paper2_fidelity_calibrated/results/induction_style_head_reanalysis/aksharantar_tel_latin/4b/induction_style_head_reanalysis.json
wait_for_file paper2_fidelity_calibrated/results/induction_style_head_reanalysis/aksharantar_hin_latin/1b/induction_style_head_reanalysis.json
wait_for_file paper2_fidelity_calibrated/results/induction_style_head_reanalysis/aksharantar_tel_latin/1b/induction_style_head_reanalysis.json

run_step selected_fidelity_4b_hin "$PYTHON_BIN" paper2_fidelity_calibrated/run_selected_layer_fidelity_compare.py --model 4b --pair aksharantar_hin_latin --max-items 30 --external-only --require-external-sources
run_step selected_fidelity_4b_tel "$PYTHON_BIN" paper2_fidelity_calibrated/run_selected_layer_fidelity_compare.py --model 4b --pair aksharantar_tel_latin --max-items 30 --external-only --require-external-sources
run_step selected_fidelity_1b_hin "$PYTHON_BIN" paper2_fidelity_calibrated/run_selected_layer_fidelity_compare.py --model 1b --pair aksharantar_hin_latin --max-items 30 --external-only --require-external-sources
run_step selected_fidelity_1b_tel "$PYTHON_BIN" paper2_fidelity_calibrated/run_selected_layer_fidelity_compare.py --model 1b --pair aksharantar_tel_latin --max-items 30 --external-only --require-external-sources

run_step artifact_integrity "$PYTHON_BIN" paper2_fidelity_calibrated/run_artifact_integrity_validator.py
run_step uncertainty "$PYTHON_BIN" paper2_fidelity_calibrated/run_uncertainty_estimates.py
run_step synthesis "$PYTHON_BIN" paper2_fidelity_calibrated/run_cross_experiment_synthesis.py
run_step claim_audit "$PYTHON_BIN" paper2_fidelity_calibrated/run_claim_audit.py
run_step spotcheck "$PYTHON_BIN" paper2_fidelity_calibrated/run_spotcheck_packet.py

echo "[$(date +%H:%M:%S)] Phase-5 final audit pipeline complete"
