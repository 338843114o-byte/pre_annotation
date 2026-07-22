#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASET_ROOT="${DATASET_ROOT:-/home/data_manager/DataPipes/Yolo_Detetcion_Datapipe/Storage/data/labeled/basic_hand_minrect}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/data_manager/jiangfan/basic_hand_minrect_cropped}"
PYTHON_BIN="${PYTHON_BIN:-python}"

RECT_LABEL="${RECT_LABEL:-最小外接矩形}"
PRODUCT_LABELS="${PRODUCT_LABELS:-罐装,瓶装,袋装,盒装,桶装,条装}"
CROP_MODE="${CROP_MODE:-oriented}"
PAD="${PAD:-0}"
SKIP_EMPTY_PRODUCTS="${SKIP_EMPTY_PRODUCTS:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
PROGRESS_EVERY="${PROGRESS_EVERY:-100}"
DRY_RUN="${DRY_RUN:-0}"
DETACH="${DETACH:-0}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/${RUN_ID}}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/run.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/run.pid}"

run_job() {
  if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "[ERROR] 未找到 Python：$PYTHON_BIN" >&2
    exit 1
  fi
  if ! "$PYTHON_BIN" -c "import cv2, numpy" >/dev/null 2>&1; then
    echo "[ERROR] 当前 Python 缺少 cv2/numpy：$($PYTHON_BIN -c 'import sys; print(sys.executable)')" >&2
    exit 1
  fi
  if [[ ! -d "$DATASET_ROOT" ]]; then
    echo "[ERROR] 数据集目录不存在：$DATASET_ROOT" >&2
    exit 1
  fi

  mkdir -p "$LOG_DIR" "$OUTPUT_ROOT"
  local args=(
    --dataset_root "$DATASET_ROOT"
    --output_root "$OUTPUT_ROOT"
    --rect_label "$RECT_LABEL"
    --product_labels "$PRODUCT_LABELS"
    --crop_mode "$CROP_MODE"
    --pad "$PAD"
    --progress_every "$PROGRESS_EVERY"
  )
  if [[ "$SKIP_EMPTY_PRODUCTS" == "1" ]]; then
    args+=(--skip_empty_products)
  else
    args+=(--no-skip_empty_products)
  fi
  if [[ "$SKIP_EXISTING" == "1" ]]; then
    args+=(--skip_existing)
  else
    args+=(--no-skip_existing)
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    args+=(--dry_run)
  fi

  echo "================================================================================"
  echo "按最小外接矩形裁切 + 仅保留商品标注"
  echo "================================================================================"
  echo "Python：     $($PYTHON_BIN -c 'import sys; print(sys.executable)')"
  echo "输入：       $DATASET_ROOT"
  echo "输出：       $OUTPUT_ROOT"
  echo "矩形标签：   $RECT_LABEL"
  echo "商品标签：   $PRODUCT_LABELS"
  echo "裁切模式：   $CROP_MODE"
  echo "外扩像素：   $PAD"
  echo "日志：       $LOG_FILE"
  echo "================================================================================"

  "$PYTHON_BIN" "${SCRIPT_DIR}/crop_by_min_rect.py" "${args[@]}"
}

if [[ "$DETACH" == "1" && "${CROP_CHILD:-}" != "1" ]]; then
  mkdir -p "$LOG_DIR"
  nohup env \
    CROP_CHILD=1 \
    DATASET_ROOT="$DATASET_ROOT" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    PYTHON_BIN="$PYTHON_BIN" \
    RECT_LABEL="$RECT_LABEL" \
    PRODUCT_LABELS="$PRODUCT_LABELS" \
    CROP_MODE="$CROP_MODE" \
    PAD="$PAD" \
    SKIP_EMPTY_PRODUCTS="$SKIP_EMPTY_PRODUCTS" \
    SKIP_EXISTING="$SKIP_EXISTING" \
    PROGRESS_EVERY="$PROGRESS_EVERY" \
    DRY_RUN="$DRY_RUN" \
    RUN_ID="$RUN_ID" \
    LOG_DIR="$LOG_DIR" \
    LOG_FILE="$LOG_FILE" \
    PID_FILE="$PID_FILE" \
    bash "$0" >>"$LOG_FILE" 2>&1 &
  child_pid=$!
  echo "$child_pid" >"$PID_FILE"
  echo "[INFO] 已在后台启动，PID=$child_pid"
  echo "[INFO] 查看日志：tail -f $LOG_FILE"
  exit 0
fi

if [[ "${CROP_CHILD:-}" == "1" ]]; then
  run_job
else
  mkdir -p "$LOG_DIR"
  run_job 2>&1 | tee -a "$LOG_FILE"
fi
