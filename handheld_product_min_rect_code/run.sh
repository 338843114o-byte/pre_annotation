#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 只需按实际情况修改/覆盖下面几个路径；本脚本不会创建虚拟环境。
DATASET_ROOT="${DATASET_ROOT:-/home/data_manager/DataPipes/Yolo_Detetcion_Datapipe/Storage/data/labeled/basic_hand/simple/2025_year_hand/08_month/train/train_split1}"
WEIGHTS="${WEIGHTS:-/home/data_manager/jiangfan/for_skus.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/data_manager/jiangfan/test}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# 推理与匹配参数。
DEVICE="${DEVICE:-0}"
IMGSZ="${IMGSZ:-1024}"
CONF="${CONF:-0.25}"
IOU="${IOU:-0.70}"
PRODUCT_CLASSES="${PRODUCT_CLASSES:-}"
PRODUCT_LABELS="${PRODUCT_LABELS:-罐装,瓶装,袋装,盒装,桶装,条装,未定义包装,严重遮挡,过于模糊}"
IGNORE_LABELS="${IGNORE_LABELS:-手,头,售货柜,最小外接矩形}"
HAND_LABELS="${HAND_LABELS:-手}"
AUXILIARY_LABELS="${AUXILIARY_LABELS:-遮挡}"
IGNORE_YOLO_CLASS_NAMES="${IGNORE_YOLO_CLASS_NAMES:-vending_machine,phone,too_blurry,insufficient_info}"
MAX_MATCH_AREA_RATIO="${MAX_MATCH_AREA_RATIO:-8}"
MAX_SIDE_RATIO="${MAX_SIDE_RATIO:-${MAX_RECT_AREA_RATIO:-0.5}}"
MIN_MATCH_IOU="${MIN_MATCH_IOU:-0.05}"
MIN_MATCH_OVERLAP="${MIN_MATCH_OVERLAP:-0.20}"
HAND_EXPAND_RATIO="${HAND_EXPAND_RATIO:-0.15}"
HAND_NEAR_EXPAND_RATIO="${HAND_NEAR_EXPAND_RATIO:-0.35}"
HAND_MAX_CENTER_DIST_RATIO="${HAND_MAX_CENTER_DIST_RATIO:-1.25}"
INCLUDE_NEARBY_HANDS="${INCLUDE_NEARBY_HANDS:-1}"
RECTANGLE_MODE="${RECTANGLE_MODE:-min_area}"
MARGIN="${MARGIN:-12}"
MARGIN_RATIO="${MARGIN_RATIO:-0}"
RECT_LABEL="${RECT_LABEL:-最小外接矩形}"

# 运行控制。
PROGRESS_EVERY="${PROGRESS_EVERY:-100}"
RESUME="${RESUME:-0}"
OVERWRITE="${OVERWRITE:-0}"
DETACH="${DETACH:-0}"
OUTPUT_ZIP="${OUTPUT_ZIP:-}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/${RUN_ID}}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/run.log}"
STATUS_FILE="${STATUS_FILE:-${LOG_DIR}/progress.json}"
PID_FILE="${PID_FILE:-${LOG_DIR}/run.pid}"

run_job() {
  if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "[ERROR] 未找到 Python：$PYTHON_BIN" >&2
    exit 1
  fi
  if ! "$PYTHON_BIN" -c "import torch, ultralytics, cv2, numpy, PIL" >/dev/null 2>&1; then
    echo "[ERROR] 当前 Python 缺少依赖：$($PYTHON_BIN -c 'import sys; print(sys.executable)')" >&2
    echo "请在现有环境中执行：$PYTHON_BIN -m pip install -r ${SCRIPT_DIR}/requirements.txt" >&2
    exit 1
  fi
  if [[ ! -d "$DATASET_ROOT" ]]; then
    echo "[ERROR] 数据集目录不存在：$DATASET_ROOT" >&2
    exit 1
  fi
  if [[ ! -f "$WEIGHTS" ]]; then
    echo "[ERROR] 商品 YOLO 权重不存在：$WEIGHTS" >&2
    echo "请设置：export WEIGHTS=/你的/商品模型/best.pt" >&2
    exit 1
  fi

  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  local py_device
  if [[ "$DEVICE" == "cpu" ]]; then
    unset CUDA_VISIBLE_DEVICES || true
    py_device="cpu"
  else
    export CUDA_VISIBLE_DEVICES="$DEVICE"
    py_device="0"
    if ! "$PYTHON_BIN" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1; then
      echo "[ERROR] 当前 Python 无法使用 CUDA；可设置 DEVICE=cpu 使用 CPU。" >&2
      exit 1
    fi
  fi

  mkdir -p "$LOG_DIR"
  local args=(
    --dataset_root "$DATASET_ROOT"
    --weights "$WEIGHTS"
    --output_root "$OUTPUT_ROOT"
    --device "$py_device"
    --imgsz "$IMGSZ"
    --conf "$CONF"
    --iou "$IOU"
    --product_classes "$PRODUCT_CLASSES"
    --product_labels "$PRODUCT_LABELS"
    --ignore_labels "$IGNORE_LABELS"
    --hand_labels "$HAND_LABELS"
    --auxiliary_labels "$AUXILIARY_LABELS"
    --ignore_yolo_class_names "$IGNORE_YOLO_CLASS_NAMES"
    --max_match_area_ratio "$MAX_MATCH_AREA_RATIO"
    --max_side_ratio "$MAX_SIDE_RATIO"
    --min_match_iou "$MIN_MATCH_IOU"
    --min_match_overlap "$MIN_MATCH_OVERLAP"
    --hand_expand_ratio "$HAND_EXPAND_RATIO"
    --hand_near_expand_ratio "$HAND_NEAR_EXPAND_RATIO"
    --hand_max_center_dist_ratio "$HAND_MAX_CENTER_DIST_RATIO"
    --rectangle_mode "$RECTANGLE_MODE"
    --margin "$MARGIN"
    --margin_ratio "$MARGIN_RATIO"
    --rect_label "$RECT_LABEL"
    --skip_existing
    --progress_every "$PROGRESS_EVERY"
    --status_file "$STATUS_FILE"
  )
  if [[ "$INCLUDE_NEARBY_HANDS" == "1" ]]; then
    args+=(--include_nearby_hands)
  else
    args+=(--no-include_nearby_hands)
  fi
  if [[ "$RESUME" == "1" ]]; then
    args+=(--resume)
  fi
  if [[ "$OVERWRITE" == "1" ]]; then
    args+=(--overwrite)
  fi
  if [[ -n "$OUTPUT_ZIP" ]]; then
    args+=(--output_zip "$OUTPUT_ZIP")
  fi

  echo "================================================================================"
  echo "手拿商品最小外接矩形批量标注"
  echo "================================================================================"
  echo "Python：         $($PYTHON_BIN -c 'import sys; print(sys.executable)')"
  echo "数据集：         $DATASET_ROOT"
  echo "商品模型：       $WEIGHTS"
  echo "输出目录：       $OUTPUT_ROOT"
  echo "物理显卡：       $DEVICE"
  echo "推理 device：    $py_device"
  echo "商品 JSON 标签： ${PRODUCT_LABELS:-自动排除非商品标签}"
  echo "商品模型类别：   ${PRODUCT_CLASSES:-未限制 id}"
  echo "忽略 YOLO 类名： ${IGNORE_YOLO_CLASS_NAMES}"
  echo "最大匹配面积比： ${MAX_MATCH_AREA_RATIO}"
  echo "外接矩形边长比： ${MAX_SIDE_RATIO}"
  echo "纳入重叠手：     ${INCLUDE_NEARBY_HANDS}"
  echo "手靠近扩展比：   ${HAND_NEAR_EXPAND_RATIO}"
  echo "手中心距上限：   ${HAND_MAX_CENTER_DIST_RATIO}"
  echo "矩形形式：       $RECTANGLE_MODE"
  echo "外扩像素/比例：  ${MARGIN} / ${MARGIN_RATIO}"
  echo "新增 label：     $RECT_LABEL"
  echo "日志：           $LOG_FILE"
  echo "================================================================================"

  "$PYTHON_BIN" "${SCRIPT_DIR}/add_handheld_product_min_rect.py" "${args[@]}"
}

if [[ "$DETACH" == "1" && "${MIN_RECT_CHILD:-}" != "1" ]]; then
  mkdir -p "$LOG_DIR"
  nohup env \
    MIN_RECT_CHILD=1 \
    DATASET_ROOT="$DATASET_ROOT" \
    WEIGHTS="$WEIGHTS" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
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
    OUTPUT_ZIP="$OUTPUT_ZIP" \
    RUN_ID="$RUN_ID" \
    LOG_DIR="$LOG_DIR" \
    LOG_FILE="$LOG_FILE" \
    STATUS_FILE="$STATUS_FILE" \
    PID_FILE="$PID_FILE" \
    bash "$0" >>"$LOG_FILE" 2>&1 &
  child_pid=$!
  echo "$child_pid" >"$PID_FILE"
  echo "[INFO] 已在后台启动，PID=$child_pid"
  echo "[INFO] 查看日志：tail -f $LOG_FILE"
  exit 0
fi

if [[ "${MIN_RECT_CHILD:-}" == "1" ]]; then
  run_job
else
  mkdir -p "$LOG_DIR"
  run_job 2>&1 | tee -a "$LOG_FILE"
fi
