#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
OUTDIR="${2:-}"
FORCE_FLAG="${FORCE_FLAG:---force}"

loop1_default_outdir() {
  local mode="$1"
  printf 'research/results/autoresearch/loop1_cross_scale_anchor/%s' "$mode"
}

run_loop1() {
  local mode="$1"
  local outdir="${2:-$(loop1_default_outdir "$mode")}"
  local task_ids="premise_gate__270m__aksharantar_hin_latin,premise_gate__270m__aksharantar_tel_latin,premise_gate__1b__aksharantar_hin_latin,premise_gate__1b__aksharantar_tel_latin,premise_gate__4b__aksharantar_hin_latin,premise_gate__4b__aksharantar_tel_latin"
  local volume_name="gemma-multiscale-results"
  local remote_volume_path="/"

  mkdir -p "$outdir"

  echo "[loop1] verifying multiscale suite wiring"
  python3 -m Draft_Results.paper2_fidelity_calibrated.multiscale_modal_suite.verify_suite \
    --out "$outdir/verify_suite.json"

  echo "[loop1] launching Modal premise-gate tasks ($mode)"
  local -a modal_args=(
    run Draft_Results/paper2_fidelity_calibrated/multiscale_modal_suite/modal_app.py
    --task-ids "$task_ids"
    --wait
    "$FORCE_FLAG"
  )
  if [[ "$mode" == "smoke" ]]; then
    modal_args+=(--smoke)
  elif [[ "$mode" != "full" ]]; then
    echo "Unknown Loop 1 mode: $mode (expected smoke or full)" >&2
    exit 2
  fi

  local tmp_modal_log
  tmp_modal_log="$(mktemp /tmp/loop1-modal-run-XXXXXX.log)"
  modal "${modal_args[@]}" | tee "$tmp_modal_log"
  mv "$tmp_modal_log" "$outdir/modal_run.log"

  echo "[loop1] downloading Modal artifacts"
  mkdir -p "$outdir/volume"
  modal volume get "$volume_name" "$remote_volume_path" "$outdir/volume" --force

  echo "[loop1] scoring cross-scale premise gate"
  python3 experiments/score_cross_scale_anchor.py \
    --results-root "$outdir/volume" \
    --out "$outdir/score.json" | tee "$outdir/score.log"

  echo "[loop1] done -> $outdir/score.json"
}

loop2_default_outdir() {
  local mode="$1"
  printf 'research/results/autoresearch/loop2_vm_controls/%s' "$mode"
}

require_vm_secret() {
  if [[ -z "${VM_PASS:-}" ]]; then
    echo "Set VM_PASS in the environment before running Loop 2 on the VM." >&2
    exit 2
  fi
}

vm_ssh() {
  sshpass -p "$VM_PASS" ssh \
    -o StrictHostKeyChecking=no \
    -o ConnectTimeout="${VM_CONNECT_TIMEOUT:-15}" \
    "${VM_HOST:-srinivasr@10.10.0.215}" "$@"
}

vm_sync_to_workdir() {
  local dest="$1"
  tar \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.git/' \
    --exclude='.cache/' \
    --exclude='.venv*/' \
    --exclude='Draft_Results/paper2_fidelity_calibrated/results/' \
    --exclude='Draft_Results/results/' \
    --exclude='research/results/autoresearch/' \
    -czf - \
    Draft_Results \
    experiments \
    autoresearch.md \
    autoresearch.sh \
    CHANGELOG.md | vm_ssh "mkdir -p '$dest' && tar xzf - -C '$dest'"
}

vm_fetch_dir() {
  local remote_dir="$1"
  local local_dir="$2"
  mkdir -p "$local_dir"
  vm_ssh "tar czf - -C '$remote_dir' ." | tar xzf - -C "$local_dir"
}

run_loop2() {
  local mode="$1"
  local outdir="${2:-$(loop2_default_outdir "$mode")}"
  local vm_workdir="${VM_WORKDIR:-/home/srinivasr/Research/Honors}"
  local remote_results_rel="${LOOP2_REMOTE_RESULTS_REL:-$outdir/raw}"
  local max_items n_eval

  case "$mode" in
    loop2_smoke)
      max_items="8"
      n_eval="24"
      ;;
    loop2_full)
      max_items="30"
      n_eval="50"
      ;;
    *)
      echo "Unknown Loop 2 mode: $mode (expected loop2_smoke or loop2_full)" >&2
      exit 2
      ;;
  esac

  require_vm_secret
  mkdir -p "$outdir"

  echo "[loop2] checking VM reachability"
  vm_ssh "echo connected: \\$(hostname)"

  echo "[loop2] preparing VM workspace at $vm_workdir"
  vm_ssh "mkdir -p $vm_workdir"

  echo "[loop2] syncing code to VM"
  vm_sync_to_workdir "$vm_workdir"

  echo "[loop2] running 2x2 helpful-vs-control panel on VM ($mode)"
  vm_ssh "cd $vm_workdir && LOOP2_REMOTE_RESULTS_REL='$remote_results_rel' LOOP2_MAX_ITEMS='$max_items' LOOP2_N_EVAL='$n_eval' PAPER2_DEVICE='${PAPER2_DEVICE:-cuda}' bash -s" <<'REMOTE' | tee "$outdir/vm_run.log"
set -euo pipefail
WORKDIR="$(pwd)"
DEVICE="${PAPER2_DEVICE:-cuda}"
REMOTE_RESULTS_REL="${LOOP2_REMOTE_RESULTS_REL}"
MAX_ITEMS="${LOOP2_MAX_ITEMS}"
N_EVAL="${LOOP2_N_EVAL}"

PYTHON_BIN=""
for cand in \
  "$WORKDIR/.venv/bin/python" \
  "$WORKDIR/.venv-phase0a/bin/python" \
  "$HOME/Research/gemma-rescue-study/.venv/bin/python" \
  "$HOME/Research/gemma-rescue-study/.venv-phase0a/bin/python" \
  "$(command -v python3)"

do
  if [[ -n "$cand" && -x "$cand" ]]; then
    PYTHON_BIN="$cand"
    break
  fi
done
if [[ -z "$PYTHON_BIN" ]]; then
  echo "[loop2][remote] no usable python found" >&2
  exit 3
fi

echo "[loop2][remote] workdir=$WORKDIR"
echo "[loop2][remote] python=$PYTHON_BIN"
"$PYTHON_BIN" - <<'PY'
import importlib
for name in ('torch', 'transformers', 'numpy'):
    importlib.import_module(name)
print('python_import_health: ok')
PY

mkdir -p "$REMOTE_RESULTS_REL"
for model in 1b 4b; do
  for pair in aksharantar_hin_latin aksharantar_tel_latin; do
    for n_icl in 8 64; do
      out_dir="$REMOTE_RESULTS_REL/$model/$pair/nicl${n_icl}"
      mkdir -p "$out_dir"
      echo "[loop2][remote] START model=$model pair=$pair n_icl=$n_icl"
      "$PYTHON_BIN" Draft_Results/paper2_fidelity_calibrated/run_neutral_filler_recency_controls.py \
        --model "$model" \
        --pair "$pair" \
        --device "$DEVICE" \
        --seed 42 \
        --n-icl "$n_icl" \
        --n-select 300 \
        --n-eval "$N_EVAL" \
        --max-items "$MAX_ITEMS" \
        --external-only \
        --require-external-sources \
        --out "$out_dir"
      echo "[loop2][remote] DONE  model=$model pair=$pair n_icl=$n_icl"
    done
  done
done
REMOTE

  echo "[loop2] downloading Loop 2 artifacts from VM"
  mkdir -p "$outdir/raw"
  vm_fetch_dir "$vm_workdir/$remote_results_rel" "$outdir/raw"

  echo "[loop2] scoring helpful-vs-control panel"
  python3 experiments/score_loop2_controls.py \
    --results-root "$outdir/raw" \
    --out "$outdir/score.json" | tee "$outdir/score.log"

  echo "[loop2] done -> $outdir/score.json"
}

export LOOP2_REMOTE_RESULTS_REL="${LOOP2_REMOTE_RESULTS_REL:-}"
export LOOP2_MAX_ITEMS="${LOOP2_MAX_ITEMS:-}"
export LOOP2_N_EVAL="${LOOP2_N_EVAL:-}"

case "$MODE" in
  smoke|full)
    run_loop1 "$MODE" "$OUTDIR"
    ;;
  loop2_smoke|loop2_full)
    if [[ -z "${LOOP2_REMOTE_RESULTS_REL:-}" ]]; then
      export LOOP2_REMOTE_RESULTS_REL="$(loop2_default_outdir "$MODE")/raw"
    fi
    if [[ -z "${LOOP2_MAX_ITEMS:-}" || -z "${LOOP2_N_EVAL:-}" ]]; then
      if [[ "$MODE" == "loop2_smoke" ]]; then
        export LOOP2_MAX_ITEMS="8"
        export LOOP2_N_EVAL="24"
      else
        export LOOP2_MAX_ITEMS="30"
        export LOOP2_N_EVAL="50"
      fi
    fi
    run_loop2 "$MODE" "$OUTDIR"
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    echo "Expected one of: smoke, full, loop2_smoke, loop2_full" >&2
    exit 2
    ;;
esac
