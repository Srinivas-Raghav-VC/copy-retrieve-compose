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
export PYTHON_BIN
DEVICE="${PAPER2_DEVICE:-cuda}"
LOG="${LOG:-paper2_fidelity_calibrated/results/current_scope_paper_upgrade_pipeline.log}"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

PAIRS=(aksharantar_hin_latin aksharantar_tel_latin)
MODELS=(4b 1b)
ATTR_BATCH_WORDS="${ATTR_BATCH_WORDS:-15}"
ATTR_TOPK="${ATTR_TOPK:-20}"
CONTROL_N_SELECT="${CONTROL_N_SELECT:-300}"
CONTROL_N_EVAL="${CONTROL_N_EVAL:-50}"
ACTDIFF_STATS_ITEMS="${ACTDIFF_STATS_ITEMS:-100}"

run_step() {
  local name="$1"
  shift
  echo "[$(date +%H:%M:%S)] START $name"
  "$@"
  echo "[$(date +%H:%M:%S)] DONE  $name"
}

wait_for_file() {
  local path="$1"
  while [[ ! -f "$path" ]]; do
    echo "[$(date +%H:%M:%S)] waiting for $path"
    sleep 30
  done
}

phase1_done() {
  ! pgrep -f 'paper2_fidelity_calibrated/run_stage0_stagea.py|paper2_fidelity_calibrated/run_attribution_graph_pair.py|paper2_fidelity_calibrated/run_logit_lens_rescue_trajectory.py|paper2_fidelity_calibrated/run_causal_head_attention_patterns.py|paper2_fidelity_calibrated/run_language_script_feature_suppression.py|phase1_followup_1b_20260311.sh' >/dev/null 2>&1
}

stagea_path() {
  local pair="$1"
  local model="$2"
  printf 'paper2_fidelity_calibrated/results/%s/%s/paper2_fidelity_calibrated_%s.json' "$pair" "$model" "$model"
}

top_heads_path() {
  local pair="$1"
  local model="$2"
  local lang
  lang="$(echo "$pair" | cut -d_ -f2)"
  printf 'artifacts/phase5_attribution/top_heads_%s_%s_multilang.json' "$model" "$lang"
}

stagea_layer() {
  local pair="$1"
  local model="$2"
  "$PYTHON_BIN" - <<PY
import json
from pathlib import Path
p = Path('${ROOT}') / 'paper2_fidelity_calibrated' / 'results' / '${pair}' / '${model}' / 'paper2_fidelity_calibrated_${model}.json'
obj = json.loads(p.read_text(encoding='utf-8'))
print(int(obj['seeds']['42']['best']['layer']))
PY
}

ensure_stagea_artifacts() {
  local missing_models=()
  for model in "${MODELS[@]}"; do
    local model_missing=0
    for pair in "${PAIRS[@]}"; do
      local path
      path="$(stagea_path "$pair" "$model")"
      if [[ ! -f "$path" ]]; then
        model_missing=1
      fi
    done
    if [[ "$model_missing" -eq 1 ]]; then
      missing_models+=("$model")
    fi
  done

  if [[ "${#missing_models[@]}" -gt 0 ]]; then
    echo "[$(date +%H:%M:%S)] missing Stage-A artifacts for models: ${missing_models[*]}"
    run_step stagea_repair "$PYTHON_BIN" paper2_fidelity_calibrated/run_stage0_stagea.py \
      --models "$(IFS=,; echo "${missing_models[*]}")" \
      --skip-stage0 \
      --stagea-pairs "$(IFS=,; echo "${PAIRS[*]}")" \
      --device "$DEVICE"
  fi

  for model in "${MODELS[@]}"; do
    for pair in "${PAIRS[@]}"; do
      wait_for_file "$(stagea_path "$pair" "$model")"
    done
  done
}

ensure_top_heads_artifacts() {
  for model in "${MODELS[@]}"; do
    for pair in "${PAIRS[@]}"; do
      local out
      out="$(top_heads_path "$pair" "$model")"
      if [[ ! -f "$out" ]]; then
        run_step attribution_${model}_$(echo "$pair" | cut -d_ -f2) \
          "$PYTHON_BIN" paper2_fidelity_calibrated/run_attribution_graph_pair.py \
          --model "$model" \
          --pair "$pair" \
          --device "$DEVICE" \
          --batch-words "$ATTR_BATCH_WORDS" \
          --topk "$ATTR_TOPK" \
          --external-only \
          --require-external-sources \
          --out-tag multilang
      fi
      wait_for_file "$out"
    done
  done
}

run_dense_control() {
  local model="$1"
  local pair="$2"
  local layer
  layer="$(stagea_layer "$pair" "$model")"
  run_step dense_${model}_$(echo "$pair" | cut -d_ -f2) \
    "$PYTHON_BIN" paper2_fidelity_calibrated/run_dense_mlp_sweep.py \
    --model "$model" \
    --pair "$pair" \
    --device "$DEVICE" \
    --n-select "$CONTROL_N_SELECT" \
    --n-eval "$CONTROL_N_EVAL" \
    --layer-start "$layer" \
    --layer-end "$layer" \
    --external-only \
    --require-external-sources
}

run_dense_smoke() {
  local model="$1"
  local pair="$2"
  local layer
  layer="$(stagea_layer "$pair" "$model")"
  run_step smoke_dense_${model}_$(echo "$pair" | cut -d_ -f2) \
    "$PYTHON_BIN" paper2_fidelity_calibrated/run_dense_mlp_sweep.py \
    --model "$model" \
    --pair "$pair" \
    --device "$DEVICE" \
    --n-select 32 \
    --n-eval 1 \
    --layer-start "$layer" \
    --layer-end "$layer" \
    --external-only \
    --require-external-sources
}

run_actdiff_control() {
  local model="$1"
  local pair="$2"
  run_step actdiff_${model}_$(echo "$pair" | cut -d_ -f2) \
    "$PYTHON_BIN" paper2_fidelity_calibrated/run_activation_difference_baseline.py \
    --model "$model" \
    --pair "$pair" \
    --device "$DEVICE" \
    --stats-items "$ACTDIFF_STATS_ITEMS" \
    --max-items "$CONTROL_N_EVAL" \
    --external-only \
    --require-external-sources
}

run_actdiff_smoke() {
  local model="$1"
  local pair="$2"
  run_step smoke_actdiff_${model}_$(echo "$pair" | cut -d_ -f2) \
    "$PYTHON_BIN" paper2_fidelity_calibrated/run_activation_difference_baseline.py \
    --model "$model" \
    --pair "$pair" \
    --device "$DEVICE" \
    --stats-items 8 \
    --max-items 1 \
    --external-only \
    --require-external-sources
}

echo "[$(date +%H:%M:%S)] Current-scope paper-upgrade pipeline queued"
while ! phase1_done; do
  echo "[$(date +%H:%M:%S)] waiting for active Phase-1 work to finish"
  sleep 60
done

ensure_stagea_artifacts
ensure_top_heads_artifacts

run_step phase2_anchor bash paper2_fidelity_calibrated/run_phase2_anchor_pipeline.sh

# Decisive reviewer-facing controls.
run_dense_smoke 4b aksharantar_hin_latin
run_actdiff_smoke 4b aksharantar_hin_latin
run_dense_control 4b aksharantar_hin_latin
run_actdiff_control 4b aksharantar_hin_latin
run_dense_control 4b aksharantar_tel_latin
run_actdiff_control 4b aksharantar_tel_latin
run_dense_smoke 1b aksharantar_hin_latin
run_actdiff_smoke 1b aksharantar_hin_latin
run_dense_control 1b aksharantar_hin_latin
run_actdiff_control 1b aksharantar_hin_latin
run_dense_control 1b aksharantar_tel_latin
run_actdiff_control 1b aksharantar_tel_latin

run_step phase3_anchor bash paper2_fidelity_calibrated/run_phase3_anchor_pipeline.sh
run_step phase4_robustness bash paper2_fidelity_calibrated/run_phase4_robustness_pipeline.sh
run_step phase5_final_audit bash paper2_fidelity_calibrated/run_phase5_final_audit_pipeline.sh

echo "[$(date +%H:%M:%S)] Current-scope paper-upgrade pipeline complete"
