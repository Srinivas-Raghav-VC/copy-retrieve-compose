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
LOG="${LOG:-paper2_fidelity_calibrated/results/phase4_robustness_pipeline.log}"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

phase3_done() {
  ! pgrep -f 'run_phase3_anchor_pipeline.sh|run_cfom_function_vector_tests.py|run_circuit_sufficiency.py|run_icl_contribution_curve.py|run_sparse_feature_circuit.py|run_minimality_curve.py' >/dev/null 2>&1
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

echo "[$(date +%H:%M:%S)] Phase-4 robustness pipeline queued"
while ! phase3_done; do
  echo "[$(date +%H:%M:%S)] waiting for Phase-3 queue to finish"
  sleep 60
done

wait_for_file paper2_fidelity_calibrated/results/head_to_mlp_edge_attribution/aksharantar_hin_latin/4b/head_to_mlp_edge_attribution.json
wait_for_file paper2_fidelity_calibrated/results/head_to_mlp_edge_attribution/aksharantar_tel_latin/4b/head_to_mlp_edge_attribution.json
wait_for_file paper2_fidelity_calibrated/results/head_to_mlp_edge_attribution/aksharantar_hin_latin/1b/head_to_mlp_edge_attribution.json
wait_for_file paper2_fidelity_calibrated/results/head_to_mlp_edge_attribution/aksharantar_tel_latin/1b/head_to_mlp_edge_attribution.json
wait_for_file paper2_fidelity_calibrated/results/feature_knockout_panel/aksharantar_hin_latin/4b/feature_knockout_panel.json
wait_for_file paper2_fidelity_calibrated/results/feature_knockout_panel/aksharantar_tel_latin/4b/feature_knockout_panel.json
wait_for_file paper2_fidelity_calibrated/results/feature_knockout_panel/aksharantar_hin_latin/1b/feature_knockout_panel.json
wait_for_file paper2_fidelity_calibrated/results/feature_knockout_panel/aksharantar_tel_latin/1b/feature_knockout_panel.json

echo "[$(date +%H:%M:%S)] Phase-4 preflight smokes"
run_step smoke_consensus_4b_hin "$PYTHON_BIN" paper2_fidelity_calibrated/run_transcoder_family_consensus.py --model 4b --pair aksharantar_hin_latin --device "$DEVICE" --max-items 1 --external-only --require-external-sources
run_step smoke_icl_necessity_4b_hin "$PYTHON_BIN" paper2_fidelity_calibrated/run_direct_icl_feature_necessity.py --model 4b --pair aksharantar_hin_latin --device "$DEVICE" --max-items 1 --core-features 4 --external-only --require-external-sources
run_step smoke_geometry_4b_hin "$PYTHON_BIN" paper2_fidelity_calibrated/run_patch_geometry_robustness.py --model 4b --pair aksharantar_hin_latin --device "$DEVICE" --max-items 1 --external-only --require-external-sources
run_step smoke_artifact_stability_4b_hin "$PYTHON_BIN" paper2_fidelity_calibrated/run_cross_artifact_feature_stability.py --model 4b --pair aksharantar_hin_latin --device "$DEVICE" --stats-items 8 --topk 4 --external-only --require-external-sources
run_step smoke_gap_4b_hin "$PYTHON_BIN" paper2_fidelity_calibrated/run_target_competitor_logit_gap.py --model 4b --pair aksharantar_hin_latin --device "$DEVICE" --max-items 1 --external-only --require-external-sources
run_step smoke_leavek_4b_hin "$PYTHON_BIN" paper2_fidelity_calibrated/run_leave_k_out_icl_contribution.py --model 4b --pair aksharantar_hin_latin --device "$DEVICE" --max-items 1 --ks 1 --subsets-per-k 1 --external-only --require-external-sources
run_step smoke_equiv_4b_hin "$PYTHON_BIN" paper2_fidelity_calibrated/run_decoded_vs_latent_equivalence.py --model 4b --pair aksharantar_hin_latin --device "$DEVICE" --max-items 1 --external-only --require-external-sources
run_step smoke_neighbor_4b_hin "$PYTHON_BIN" paper2_fidelity_calibrated/run_neighbor_layer_causality.py --model 4b --pair aksharantar_hin_latin --device "$DEVICE" --max-items 1 --external-only --require-external-sources
run_step smoke_headstab_4b_hin "$PYTHON_BIN" paper2_fidelity_calibrated/run_head_attribution_stability.py --model 4b --pair aksharantar_hin_latin --device "$DEVICE" --resamples 2 --subset-size 4 --batch-words 4 --topk 4 --external-only --require-external-sources
run_step smoke_neutralrecency_4b_hin "$PYTHON_BIN" paper2_fidelity_calibrated/run_neutral_filler_recency_controls.py --model 4b --pair aksharantar_hin_latin --device "$DEVICE" --max-items 1 --external-only --require-external-sources
run_step smoke_induction_4b_hin "$PYTHON_BIN" paper2_fidelity_calibrated/run_induction_style_head_reanalysis.py --model 4b --pair aksharantar_hin_latin --device "$DEVICE" --max-items 1 --top-n-heads 4 --external-only --require-external-sources
run_step smoke_consensus_1b_hin "$PYTHON_BIN" paper2_fidelity_calibrated/run_transcoder_family_consensus.py --model 1b --pair aksharantar_hin_latin --device "$DEVICE" --max-items 1 --external-only --require-external-sources
run_step smoke_icl_necessity_1b_hin "$PYTHON_BIN" paper2_fidelity_calibrated/run_direct_icl_feature_necessity.py --model 1b --pair aksharantar_hin_latin --device "$DEVICE" --max-items 1 --core-features 4 --external-only --require-external-sources
run_step smoke_geometry_1b_hin "$PYTHON_BIN" paper2_fidelity_calibrated/run_patch_geometry_robustness.py --model 1b --pair aksharantar_hin_latin --device "$DEVICE" --max-items 1 --external-only --require-external-sources
run_step smoke_artifact_stability_1b_hin "$PYTHON_BIN" paper2_fidelity_calibrated/run_cross_artifact_feature_stability.py --model 1b --pair aksharantar_hin_latin --device "$DEVICE" --stats-items 8 --topk 4 --external-only --require-external-sources
run_step smoke_gap_1b_hin "$PYTHON_BIN" paper2_fidelity_calibrated/run_target_competitor_logit_gap.py --model 1b --pair aksharantar_hin_latin --device "$DEVICE" --max-items 1 --external-only --require-external-sources
run_step smoke_leavek_1b_hin "$PYTHON_BIN" paper2_fidelity_calibrated/run_leave_k_out_icl_contribution.py --model 1b --pair aksharantar_hin_latin --device "$DEVICE" --max-items 1 --ks 1 --subsets-per-k 1 --external-only --require-external-sources
run_step smoke_equiv_1b_hin "$PYTHON_BIN" paper2_fidelity_calibrated/run_decoded_vs_latent_equivalence.py --model 1b --pair aksharantar_hin_latin --device "$DEVICE" --max-items 1 --external-only --require-external-sources
run_step smoke_neighbor_1b_hin "$PYTHON_BIN" paper2_fidelity_calibrated/run_neighbor_layer_causality.py --model 1b --pair aksharantar_hin_latin --device "$DEVICE" --max-items 1 --external-only --require-external-sources
run_step smoke_headstab_1b_hin "$PYTHON_BIN" paper2_fidelity_calibrated/run_head_attribution_stability.py --model 1b --pair aksharantar_hin_latin --device "$DEVICE" --resamples 2 --subset-size 4 --batch-words 4 --topk 4 --external-only --require-external-sources
run_step smoke_neutralrecency_1b_hin "$PYTHON_BIN" paper2_fidelity_calibrated/run_neutral_filler_recency_controls.py --model 1b --pair aksharantar_hin_latin --device "$DEVICE" --max-items 1 --external-only --require-external-sources
run_step smoke_induction_1b_hin "$PYTHON_BIN" paper2_fidelity_calibrated/run_induction_style_head_reanalysis.py --model 1b --pair aksharantar_hin_latin --device "$DEVICE" --max-items 1 --top-n-heads 4 --external-only --require-external-sources

for MODEL in 4b 1b; do
  for PAIR in aksharantar_hin_latin aksharantar_tel_latin; do
    run_step consensus_${MODEL}_$(echo $PAIR | cut -d_ -f2) "$PYTHON_BIN" paper2_fidelity_calibrated/run_transcoder_family_consensus.py --model "$MODEL" --pair "$PAIR" --device "$DEVICE" --external-only --require-external-sources
    run_step icl_necessity_${MODEL}_$(echo $PAIR | cut -d_ -f2) "$PYTHON_BIN" paper2_fidelity_calibrated/run_direct_icl_feature_necessity.py --model "$MODEL" --pair "$PAIR" --device "$DEVICE" --external-only --require-external-sources
    run_step geometry_${MODEL}_$(echo $PAIR | cut -d_ -f2) "$PYTHON_BIN" paper2_fidelity_calibrated/run_patch_geometry_robustness.py --model "$MODEL" --pair "$PAIR" --device "$DEVICE" --external-only --require-external-sources
    run_step artifact_stability_${MODEL}_$(echo $PAIR | cut -d_ -f2) "$PYTHON_BIN" paper2_fidelity_calibrated/run_cross_artifact_feature_stability.py --model "$MODEL" --pair "$PAIR" --device "$DEVICE" --external-only --require-external-sources
    run_step gap_${MODEL}_$(echo $PAIR | cut -d_ -f2) "$PYTHON_BIN" paper2_fidelity_calibrated/run_target_competitor_logit_gap.py --model "$MODEL" --pair "$PAIR" --device "$DEVICE" --external-only --require-external-sources
    run_step leavek_${MODEL}_$(echo $PAIR | cut -d_ -f2) "$PYTHON_BIN" paper2_fidelity_calibrated/run_leave_k_out_icl_contribution.py --model "$MODEL" --pair "$PAIR" --device "$DEVICE" --external-only --require-external-sources
    run_step equiv_${MODEL}_$(echo $PAIR | cut -d_ -f2) "$PYTHON_BIN" paper2_fidelity_calibrated/run_decoded_vs_latent_equivalence.py --model "$MODEL" --pair "$PAIR" --device "$DEVICE" --external-only --require-external-sources
    run_step neighbor_${MODEL}_$(echo $PAIR | cut -d_ -f2) "$PYTHON_BIN" paper2_fidelity_calibrated/run_neighbor_layer_causality.py --model "$MODEL" --pair "$PAIR" --device "$DEVICE" --external-only --require-external-sources
    run_step headstab_${MODEL}_$(echo $PAIR | cut -d_ -f2) "$PYTHON_BIN" paper2_fidelity_calibrated/run_head_attribution_stability.py --model "$MODEL" --pair "$PAIR" --device "$DEVICE" --resamples 6 --subset-size 10 --batch-words 10 --topk 8 --external-only --require-external-sources
    run_step neutralrecency_${MODEL}_$(echo $PAIR | cut -d_ -f2) "$PYTHON_BIN" paper2_fidelity_calibrated/run_neutral_filler_recency_controls.py --model "$MODEL" --pair "$PAIR" --device "$DEVICE" --max-items 30 --external-only --require-external-sources
    run_step induction_${MODEL}_$(echo $PAIR | cut -d_ -f2) "$PYTHON_BIN" paper2_fidelity_calibrated/run_induction_style_head_reanalysis.py --model "$MODEL" --pair "$PAIR" --device "$DEVICE" --max-items 30 --top-n-heads 8 --external-only --require-external-sources
    run_step g3_mediation_${MODEL}_$(echo $PAIR | cut -d_ -f2) "$PYTHON_BIN" paper2_fidelity_calibrated/run_g3_mediation_fraction_summary.py --input paper2_fidelity_calibrated/results/head_to_mlp_edge_attribution/$PAIR/$MODEL/head_to_mlp_edge_attribution.json
  done
done

echo "[$(date +%H:%M:%S)] Phase-4 robustness pipeline complete"
