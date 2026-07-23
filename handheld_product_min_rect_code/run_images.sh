#!/usr/bin/env bash
# 模式 2：已有图片目录 → 最小外接矩形生成
# 输入：DATASET_ROOT（含 images/；json 模式还需同级 json_labels/）
# 输出：OUTPUT_ROOT
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=run_min_rect_lib.sh
source "${SCRIPT_DIR}/run_min_rect_lib.sh"

DATASET_ROOT="${DATASET_ROOT:-/home/data_manager/jiangfan/1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/data_manager/jiangfan/1_yolo_minrect}"
LABEL_SOURCE="${LABEL_SOURCE:-yolo}"

RUN_ID="${RUN_ID:-images_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/${RUN_ID}}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/run.log}"
STATUS_FILE="${STATUS_FILE:-${LOG_DIR}/progress.json}"
PID_FILE="${PID_FILE:-${LOG_DIR}/run.pid}"

run_images_job() {
  echo "[MODE] 已有图片 → 最小外接矩形"
  run_min_rect_job "$DATASET_ROOT" "$OUTPUT_ROOT" "已有图片：手拿商品最小外接矩形"

  local result_json_dir="${OUTPUT_ROOT}/${JSON_DIR_NAME:-json_labels}"
  if [[ "$VALIDATE" == "1" && -d "$result_json_dir" ]]; then
    echo "================================================================================"
    echo "步骤：覆盖校验（同一结果 JSON：手持商品 vs 最小外接矩形）"
    echo "================================================================================"
    local validate_args=(
      --result_root "$OUTPUT_ROOT"
      --product_labels "$PRODUCT_LABELS"
      --rect_label "$RECT_LABEL"
      --coverage_threshold "$VALIDATE_COVERAGE"
      --report_json "${OUTPUT_ROOT}/validation_uncovered.json"
      --report_txt "${OUTPUT_ROOT}/validation_uncovered.txt"
    )
    set +e
    "$PYTHON_BIN" "${SCRIPT_DIR}/validate_gt_min_rect_coverage.py" "${validate_args[@]}"
    local validate_rc=$?
    set -e
    if [[ "$validate_rc" -eq 1 ]]; then
      echo "[WARN] 存在未被最小外接矩形覆盖的手持商品，详见："
      echo "  ${OUTPUT_ROOT}/validation_uncovered.txt"
    elif [[ "$validate_rc" -ne 0 ]]; then
      echo "[ERROR] 校验脚本失败，exit=$validate_rc" >&2
      exit "$validate_rc"
    else
      echo "[INFO] 校验通过：所有手持商品均被某个最小外接矩形覆盖（≥${VALIDATE_COVERAGE}）"
    fi
  elif [[ "$VALIDATE" == "1" ]]; then
    echo "[INFO] 未找到结果 json_labels（$result_json_dir），跳过校验"
  fi
}

if [[ "$DETACH" == "1" && "${MIN_RECT_CHILD:-}" != "1" ]]; then
  mkdir -p "$LOG_DIR"
  nohup env \
    MIN_RECT_CHILD=1 \
    DATASET_ROOT="$DATASET_ROOT" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    WEIGHTS="$WEIGHTS" \
    HAND_WEIGHTS="$HAND_WEIGHTS" \
    LABEL_SOURCE="$LABEL_SOURCE" \
    PYTHON_BIN="$PYTHON_BIN" \
    DEVICE="$DEVICE" \
    IMGSZ="$IMGSZ" \
    CONF="$CONF" \
    IOU="$IOU" \
    PRODUCT_CLASSES="$PRODUCT_CLASSES" \
    PRODUCT_LABELS="$PRODUCT_LABELS" \
    IGNORE_LABELS="$IGNORE_LABELS" \
    HAND_LABELS="$HAND_LABELS" \
    AUXILIARY_LABELS="$AUXILIARY_LABELS" \
    IGNORE_YOLO_CLASS_NAMES="$IGNORE_YOLO_CLASS_NAMES" \
    MAX_MATCH_AREA_RATIO="$MAX_MATCH_AREA_RATIO" \
    MAX_SIDE_RATIO="$MAX_SIDE_RATIO" \
    MAX_RECT_AREA_RATIO="$MAX_SIDE_RATIO" \
    MIN_MATCH_IOU="$MIN_MATCH_IOU" \
    MIN_MATCH_OVERLAP="$MIN_MATCH_OVERLAP" \
    HAND_EXPAND_RATIO="$HAND_EXPAND_RATIO" \
    HAND_NEAR_EXPAND_RATIO="$HAND_NEAR_EXPAND_RATIO" \
    HAND_MAX_CENTER_DIST_RATIO="$HAND_MAX_CENTER_DIST_RATIO" \
    INCLUDE_NEARBY_HANDS="$INCLUDE_NEARBY_HANDS" \
    RECTANGLE_MODE="$RECTANGLE_MODE" \
    MARGIN="$MARGIN" \
    MARGIN_RATIO="$MARGIN_RATIO" \
    RECT_LABEL="$RECT_LABEL" \
    PROGRESS_EVERY="$PROGRESS_EVERY" \
    RESUME="$RESUME" \
    OVERWRITE="$OVERWRITE" \
    SKIP_EXISTING="$SKIP_EXISTING" \
    VALIDATE="$VALIDATE" \
    VALIDATE_COVERAGE="$VALIDATE_COVERAGE" \
    OUTPUT_ZIP="$OUTPUT_ZIP" \
    RUN_ID="$RUN_ID" \
    LOG_DIR="$LOG_DIR" \
    LOG_FILE="$LOG_FILE" \
    STATUS_FILE="$STATUS_FILE" \
    PID_FILE="$PID_FILE" \
    bash "$0" >>"$LOG_FILE" 2>&1 &
  echo $! >"$PID_FILE"
  echo "[INFO] 已在后台启动，PID=$(cat "$PID_FILE")"
  echo "[INFO] 查看日志：tail -f $LOG_FILE"
  exit 0
fi

if [[ "${MIN_RECT_CHILD:-}" == "1" ]]; then
  run_images_job
else
  mkdir -p "$LOG_DIR"
  run_images_job 2>&1 | tee -a "$LOG_FILE"
fi
