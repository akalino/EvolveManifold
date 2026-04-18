#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_pipeline.sh
#   ROOT_DIR=/path/to/data PH_MODE=online_landmark_dynamic_support bash run_pipeline.sh
#   RUN_STAGE=0 MEASURE_STAGE=1 SUMMARIZE_STAGE=1 ASSET_STAGE=1 bash run_pipeline.sh

ROOT_DIR="${ROOT_DIR:-/media/alex/WD_BLACK/evolve_collapse}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$ROOT_DIR/evolve_checkpoints}"
METRIC_ROOT="${METRIC_ROOT:-$ROOT_DIR/metric_outputs}"
SUMMARY_ROOT="${SUMMARY_ROOT:-$ROOT_DIR/metric_summaries}"
ASSET_ROOT="${ASSET_ROOT:-$ROOT_DIR/summary_assets}"
LOG_ROOT="${LOG_ROOT:-$ROOT_DIR/logs}"

PH_MODE="${PH_MODE:-online_landmark_dynamic_support}"

RUN_STAGE="${RUN_STAGE:-1}"
MEASURE_STAGE="${MEASURE_STAGE:-1}"
SUMMARIZE_STAGE="${SUMMARIZE_STAGE:-1}"
ASSET_STAGE="${ASSET_STAGE:-1}"

mkdir -p "$CHECKPOINT_ROOT" "$METRIC_ROOT" "$SUMMARY_ROOT" "$ASSET_ROOT" "$LOG_ROOT"

echo "[INFO] ROOT_DIR=$ROOT_DIR"
echo "[INFO] CHECKPOINT_ROOT=$CHECKPOINT_ROOT"
echo "[INFO] METRIC_ROOT=$METRIC_ROOT"
echo "[INFO] SUMMARY_ROOT=$SUMMARY_ROOT"
echo "[INFO] ASSET_ROOT=$ASSET_ROOT"
echo "[INFO] PH_MODE=$PH_MODE"

run_cmd() {
  local stage="$1"
  shift
  echo
  echo "========== $stage =========="
  echo "[CMD] $*"
  "$@" 2>&1 | tee "$LOG_ROOT/${stage}.log"
}

if [[ "$RUN_STAGE" == "1" ]]; then
  run_cmd "01_run_generate_checkpoints" \
    python - <<PY
import os
import run

# Temporary override of path globals if run.py still uses module-level dirs.
if hasattr(run, "EXTERNAL_ROOT"):
    run.EXTERNAL_ROOT = r"$ROOT_DIR"
if hasattr(run, "CHECKPOINT_ROOT"):
    run.CHECKPOINT_ROOT = r"$CHECKPOINT_ROOT"
if hasattr(run, "METRIC_ROOT"):
    run.METRIC_ROOT = r"$METRIC_ROOT"
if hasattr(run, "SUMMARY_ROOT"):
    run.SUMMARY_ROOT = r"$SUMMARY_ROOT"
if hasattr(run, "ASSET_ROOT"):
    run.ASSET_ROOT = r"$ASSET_ROOT"

# Assumes run.py executes from its normal __main__ path or exposes helpers.
# First preference: call a main() if it exists.
if hasattr(run, "main"):
    run.main()
else:
    # Fallback: execute the file body by importing; if your run.py
    # only works through __main__, replace this block with a subprocess call:
    #   python run.py
    print("[WARN] run.py has no main(); falling back to import-only behavior.")
PY
fi

if [[ "$MEASURE_STAGE" == "1" ]]; then
  run_cmd "02_measure_checkpoints" \
    python - <<PY
import measure_checkpoints as mc

mc.EXTERNAL_ROOT = r"$ROOT_DIR"
mc.CHECKPOINT_ROOT = r"$CHECKPOINT_ROOT"
mc.METRIC_ROOT = r"$METRIC_ROOT"
mc.SUMMARY_ROOT = r"$SUMMARY_ROOT"
mc.ASSET_ROOT = r"$ASSET_ROOT"

mc.main(
    _root_dir=r"$CHECKPOINT_ROOT",
    _out_dir=r"$METRIC_ROOT",
    _ph_mode=r"$PH_MODE",
)
PY
fi

if [[ "$SUMMARIZE_STAGE" == "1" ]]; then
  run_cmd "03_summarize_metrics" \
    python summarize_metric_results.py \
      --input_dir "$METRIC_ROOT" \
      --output_dir "$SUMMARY_ROOT"
fi

if [[ "$ASSET_STAGE" == "1" ]]; then
  run_cmd "04_generate_assets" \
    python generate_assets.py \
      --summary_dir "$SUMMARY_ROOT" \
      --out_dir "$ASSET_ROOT"
fi

echo
echo "[DONE] Pipeline finished."
echo "[DONE] Logs are in: $LOG_ROOT"