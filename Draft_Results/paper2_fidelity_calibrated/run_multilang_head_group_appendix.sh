#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${PAPER2_DEVICE:-cuda}"
LOG="${LOG:-paper2_fidelity_calibrated/results/multilang_head_group_appendix.log}"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date +%H:%M:%S)] Multilingual head-group appendix pipeline start"

echo "[$(date +%H:%M:%S)] Step 1: 4B Hindi attribution graph"
"$PYTHON_BIN" paper2_fidelity_calibrated/run_attribution_graph_pair.py \
  --pair aksharantar_hin_latin \
  --model 4b \
  --device "$DEVICE" \
  --batch-words 15 \
  --topk 20 \
  --n-icl 64 \
  --n-select 300 \
  --n-eval 50 \
  --external-only \
  --require-external-sources \
  --min-pool-size 500 \
  --out-tag multilang

echo "[$(date +%H:%M:%S)] Step 2: 4B Telugu attribution graph"
"$PYTHON_BIN" paper2_fidelity_calibrated/run_attribution_graph_pair.py \
  --pair aksharantar_tel_latin \
  --model 4b \
  --device "$DEVICE" \
  --batch-words 15 \
  --topk 20 \
  --n-icl 64 \
  --n-select 300 \
  --n-eval 50 \
  --external-only \
  --require-external-sources \
  --min-pool-size 500 \
  --out-tag multilang

echo "[$(date +%H:%M:%S)] Step 3: bounded attention analysis"
"$PYTHON_BIN" paper2_fidelity_calibrated/run_bounded_attention_analysis.py \
  --external-only \
  --require-external-sources \
  --pairs aksharantar_hin_latin,aksharantar_tel_latin \
  --top-layers 2 \
  --top-heads 3

echo "[$(date +%H:%M:%S)] Step 4: shared-vs-specific head ablation"
"$PYTHON_BIN" paper2_fidelity_calibrated/run_shared_specific_head_ablation.py \
  --model 4b \
  --pairs aksharantar_hin_latin,aksharantar_tel_latin \
  --device "$DEVICE"

echo "[$(date +%H:%M:%S)] Step 5: shared-head sufficiency panel"
"$PYTHON_BIN" paper2_fidelity_calibrated/run_shared_head_sufficiency_panel.py \
  --pairs aksharantar_hin_latin,aksharantar_tel_latin \
  --device "$DEVICE"

echo "[$(date +%H:%M:%S)] Step 6: additive synergy patch panel"
"$PYTHON_BIN" paper2_fidelity_calibrated/run_additive_synergy_patch_panel.py \
  --pairs aksharantar_hin_latin,aksharantar_tel_latin \
  --device "$DEVICE"

echo "[$(date +%H:%M:%S)] Step 7: component localization panel (Hindi)"
"$PYTHON_BIN" paper2_fidelity_calibrated/run_component_localization_panel.py \
  --model 4b \
  --pair aksharantar_hin_latin \
  --alt-pair aksharantar_tel_latin \
  --layers 24,25 \
  --device "$DEVICE" \
  --external-only \
  --require-external-sources \
  --min-pool-size 500

echo "[$(date +%H:%M:%S)] Step 8: component localization panel (Telugu)"
"$PYTHON_BIN" paper2_fidelity_calibrated/run_component_localization_panel.py \
  --model 4b \
  --pair aksharantar_tel_latin \
  --alt-pair aksharantar_hin_latin \
  --layers 24,25 \
  --device "$DEVICE" \
  --external-only \
  --require-external-sources \
  --min-pool-size 500

echo "[$(date +%H:%M:%S)] Step 9: layer-output alignment panel (Hindi)"
"$PYTHON_BIN" paper2_fidelity_calibrated/run_layer_output_alignment_panel.py \
  --model 4b \
  --pair aksharantar_hin_latin \
  --alt-pair aksharantar_tel_latin \
  --layers 23,24 \
  --device "$DEVICE" \
  --external-only \
  --require-external-sources \
  --min-pool-size 500

echo "[$(date +%H:%M:%S)] Step 10: layer-output alignment panel (Telugu)"
"$PYTHON_BIN" paper2_fidelity_calibrated/run_layer_output_alignment_panel.py \
  --model 4b \
  --pair aksharantar_tel_latin \
  --alt-pair aksharantar_hin_latin \
  --layers 23,24 \
  --device "$DEVICE" \
  --external-only \
  --require-external-sources \
  --min-pool-size 500

echo "[$(date +%H:%M:%S)] Multilingual head-group appendix pipeline complete"
