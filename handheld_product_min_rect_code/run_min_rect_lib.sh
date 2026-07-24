#!/usr/bin/env bash
# 公共逻辑：供 run_images.sh / run_video.sh source，不要直接执行。
# shellcheck disable=SC2034

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

WEIGHTS="${WEIGHTS:-/home/data_manager/jiangfan/for_skus.pt}"
HAND_WEIGHTS="${HAND_WEIGHTS:-/home/data_manager/jiangfan/for_hands.pt}"
LABEL_SOURCE="${LABEL_SOURCE:-yolo}"
PYTHON_BIN="${PYTHON_BIN:-python}"

DEVICE="${DEVICE:-0}"
IMGSZ="${IMGSZ:-1024}"
CONF="${CONF:-0.25}"
IOU="${IOU:-0.70}"
PRODUCT_CLASSES="${PRODUCT_CLASSES:-}"
PRODUCT_LABELS="${PRODUCT_LABELS:-罐装,瓶装,袋装,盒装,桶装,条装,未定义包装,严重遮挡,过于模糊,信息不足}"
IGNORE_LABELS="${IGNORE_LABELS:-手,头,售货柜,手机,最小外接矩形,最小外接矩形（仅物品）}"
HAND_LABELS="${HAND_LABELS:-手}"
AUXILIARY_LABELS="${AUXILIARY_LABELS:-遮挡}"
IGNORE_YOLO_CLASS_NAMES="${IGNORE_YOLO_CLASS_NAMES:-vending_machine,phone,too_blurry}"
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
ADD_PRODUCT_ONLY_RECT="${ADD_PRODUCT_ONLY_RECT:-1}"
PRODUCT_ONLY_RECT_LABEL="${PRODUCT_ONLY_RECT_LABEL:-最小外接矩形（仅物品）}"

PROGRESS_EVERY="${PROGRESS_EVERY:-100}"
RESUME="${RESUME:-0}"
OVERWRITE="${OVERWRITE:-0}"
DETACH="${DETACH:-0}"
OUTPUT_ZIP="${OUTPUT_ZIP:-}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
# 输入侧存在 json_labels 时，跑完后做真值覆盖校验（1=开启，0=关闭）
VALIDATE="${VALIDATE:-1}"
VALIDATE_COVERAGE="${VALIDATE_COVERAGE:-0.9}"

ensure_python_ready() {
  if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "[ERROR] 未找到 Python：$PYTHON_BIN" >&2
    exit 1
  fi
  if ! "$PYTHON_BIN" -c "import torch, ultralytics, cv2, numpy, PIL" >/dev/null 2>&1; then
    echo "[ERROR] 当前 Python 缺少依赖：$($PYTHON_BIN -c 'import sys; print(sys.executable)')" >&2
    echo "请在现有环境中执行：$PYTHON_BIN -m pip install -r ${SCRIPT_DIR}/requirements.txt" >&2
    exit 1
  fi
  if [[ ! -f "$WEIGHTS" ]]; then
    echo "[ERROR] 商品 YOLO 权重不存在：$WEIGHTS" >&2
    echo "请设置：export WEIGHTS=/你的/商品模型/best.pt" >&2
    exit 1
  fi
  if [[ "$LABEL_SOURCE" == "yolo" && ! -f "$HAND_WEIGHTS" ]]; then
    echo "[ERROR] LABEL_SOURCE=yolo 时手部 YOLO 权重不存在：$HAND_WEIGHTS" >&2
    echo "请设置：export HAND_WEIGHTS=/你的/手部模型/for_hands.pt" >&2
    exit 1
  fi
}

setup_device() {
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  if [[ "$DEVICE" == "cpu" ]]; then
    unset CUDA_VISIBLE_DEVICES || true
    PY_DEVICE="cpu"
  else
    export CUDA_VISIBLE_DEVICES="$DEVICE"
    PY_DEVICE="0"
    if ! "$PYTHON_BIN" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1; then
      echo "[ERROR] 当前 Python 无法使用 CUDA；可设置 DEVICE=cpu 使用 CPU。" >&2
      exit 1
    fi
  fi
}

run_min_rect_job() {
  local dataset_root="$1"
  local output_root="$2"
  local title="${3:-手拿商品最小外接矩形批量标注}"

  if [[ ! -d "$dataset_root" ]]; then
    echo "[ERROR] 数据集目录不存在：$dataset_root" >&2
    exit 1
  fi

  ensure_python_ready
  setup_device
  mkdir -p "$LOG_DIR"

  local args=(
    --dataset_root "$dataset_root"
    --weights "$WEIGHTS"
    --label_source "$LABEL_SOURCE"
    --output_root "$output_root"
    --device "$PY_DEVICE"
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
    --product_only_rect_label "$PRODUCT_ONLY_RECT_LABEL"
    --progress_every "$PROGRESS_EVERY"
    --status_file "$STATUS_FILE"
  )
  if [[ "$LABEL_SOURCE" == "yolo" ]]; then
    args+=(--hand_weights "$HAND_WEIGHTS")
  fi
  if [[ "$SKIP_EXISTING" == "1" ]]; then
    args+=(--skip_existing)
  else
    args+=(--no-skip_existing)
  fi
  if [[ "$INCLUDE_NEARBY_HANDS" == "1" ]]; then
    args+=(--include_nearby_hands)
  else
    args+=(--no-include_nearby_hands)
  fi
  if [[ "$ADD_PRODUCT_ONLY_RECT" == "1" ]]; then
    args+=(--add_product_only_rect)
  else
    args+=(--no-add_product_only_rect)
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
  echo "$title"
  echo "================================================================================"
  echo "Python：         $($PYTHON_BIN -c 'import sys; print(sys.executable)')"
  echo "数据集：         $dataset_root"
  echo "标注来源：       $LABEL_SOURCE"
  echo "商品模型：       $WEIGHTS"
  if [[ "$LABEL_SOURCE" == "yolo" ]]; then
    echo "手部模型：       $HAND_WEIGHTS"
  fi
  echo "输出目录：       $output_root"
  echo "物理显卡：       $DEVICE"
  echo "推理 device：    $PY_DEVICE"
  echo "商品 JSON 标签： ${PRODUCT_LABELS:-自动排除非商品标签}"
  echo "忽略 YOLO 类名： ${IGNORE_YOLO_CLASS_NAMES}"
  echo "矩形形式：       $RECTANGLE_MODE"
  echo "仅物品外接矩形： $ADD_PRODUCT_ONLY_RECT ($PRODUCT_ONLY_RECT_LABEL)"
  echo "日志：           $LOG_FILE"
  echo "================================================================================"

  "$PYTHON_BIN" "${SCRIPT_DIR}/add_handheld_product_min_rect.py" "${args[@]}"
}

detach_or_run() {
  echo "[WARN] detach_or_run 已弃用；请直接使用 run_images.sh / run_video.sh" >&2
  exit 1
}
