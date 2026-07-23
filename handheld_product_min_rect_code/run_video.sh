#!/usr/bin/env bash
# 模式 1：视频 URL/本地视频/URL 列表 → 抽帧 → 最小外接矩形生成
# 输入（二选一）：VIDEO_URL 或 VIDEO_URL_FILE
#   VIDEO_URL_FILE 支持：
#     - 一行一个 URL 的 .txt
#     - 「视频URL」列的 .xlsx
#     - 「订单视频」JSON 数组列的重叠订单 .xlsx
# 中间：FRAMES_ROOT/（单视频为 images/；批量则为 <job>/images/）
# 输出：OUTPUT_ROOT（仅单视频/flat 时在此脚本内直接衔接 min_rect）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=run_min_rect_lib.sh
source "${SCRIPT_DIR}/run_min_rect_lib.sh"

VIDEO_URL="${VIDEO_URL:-}"
VIDEO_URL_FILE="${VIDEO_URL_FILE:-}"
FRAMES_ROOT="${FRAMES_ROOT:-/home/data_manager/jiangfan/video_frames_ds}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/data_manager/jiangfan/video_minrect_out}"
LABEL_SOURCE="${LABEL_SOURCE:-yolo}"

FPS="${FPS:-0}"
EVERY_N="${EVERY_N:-1}"
MAX_FRAMES="${MAX_FRAMES:-0}"
START_SEC="${START_SEC:-0}"
END_SEC="${END_SEC:--1}"
FRAME_PREFIX="${FRAME_PREFIX:-frame_}"
IMAGE_EXT="${IMAGE_EXT:-.jpg}"
JPEG_QUALITY="${JPEG_QUALITY:-92}"
KEEP_VIDEO="${KEEP_VIDEO:-0}"
FRAMES_OVERWRITE="${FRAMES_OVERWRITE:-${OVERWRITE:-0}}"
VIDEO_ROLE="${VIDEO_ROLE:-all}"
MAX_VIDEOS="${MAX_VIDEOS:-0}"
FLAT_OUTPUT="${FLAT_OUTPUT:-0}"

RUN_ID="${RUN_ID:-video_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/${RUN_ID}}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/run.log}"
STATUS_FILE="${STATUS_FILE:-${LOG_DIR}/progress.json}"
PID_FILE="${PID_FILE:-${LOG_DIR}/run.pid}"

run_video_job() {
  if [[ -z "$VIDEO_URL" && -z "$VIDEO_URL_FILE" ]]; then
    echo "[ERROR] 请设置视频输入（二选一）：" >&2
    echo "  export VIDEO_URL='https://.../a.mp4'  # 或本地路径" >&2
    echo "  export VIDEO_URL_FILE=/path/to/一行一个URL.txt|.xlsx|重叠订单.xlsx" >&2
    exit 1
  fi
  if [[ -n "$VIDEO_URL" && -n "$VIDEO_URL_FILE" ]]; then
    echo "[ERROR] VIDEO_URL 与 VIDEO_URL_FILE 只能二选一" >&2
    exit 1
  fi

  ensure_python_ready
  if ! "$PYTHON_BIN" -c "import cv2" >/dev/null 2>&1; then
    echo "[ERROR] 抽帧需要 opencv（cv2）" >&2
    exit 1
  fi
  if [[ -n "$VIDEO_URL_FILE" ]] && [[ "$VIDEO_URL_FILE" == *.xlsx || "$VIDEO_URL_FILE" == *.xlsm ]]; then
    if ! "$PYTHON_BIN" -c "import openpyxl" >/dev/null 2>&1; then
      echo "[ERROR] 读取 xlsx 需要 openpyxl：pip install openpyxl" >&2
      exit 1
    fi
  fi

  echo "[MODE] 视频抽帧 → 最小外接矩形"
  if [[ -n "$VIDEO_URL" ]]; then
    echo "视频：           $VIDEO_URL"
  else
    echo "URL 列表：       $VIDEO_URL_FILE"
  fi
  echo "抽帧目录：       $FRAMES_ROOT"
  echo "min_rect 输出：  $OUTPUT_ROOT"
  echo "video_role：     $VIDEO_ROLE"

  local frame_args=(
    --output_root "$FRAMES_ROOT"
    --frame_prefix "$FRAME_PREFIX"
    --image_ext "$IMAGE_EXT"
    --every_n "$EVERY_N"
    --fps "$FPS"
    --max_frames "$MAX_FRAMES"
    --start_sec "$START_SEC"
    --end_sec "$END_SEC"
    --jpeg_quality "$JPEG_QUALITY"
    --video_role "$VIDEO_ROLE"
  )
  if [[ -n "$VIDEO_URL" ]]; then
    frame_args+=(--video_url "$VIDEO_URL")
  else
    frame_args+=(--url_file "$VIDEO_URL_FILE")
  fi
  if [[ "$MAX_VIDEOS" != "0" ]]; then
    frame_args+=(--max_videos "$MAX_VIDEOS")
  fi
  if [[ "$FLAT_OUTPUT" == "1" ]]; then
    frame_args+=(--flat_output)
  fi
  if [[ "$FRAMES_OVERWRITE" == "1" ]]; then
    frame_args+=(--overwrite)
  fi
  if [[ "$KEEP_VIDEO" == "1" ]]; then
    frame_args+=(--keep_video)
  fi

  echo "================================================================================"
  echo "步骤 1/2：抽帧"
  echo "================================================================================"
  "$PYTHON_BIN" "${SCRIPT_DIR}/video_url_to_frames.py" "${frame_args[@]}"

  # 批量且非 flat：每个子目录需各自作为 DATASET_ROOT；本脚本仅对单根 images/ 衔接
  local images_dir="${FRAMES_ROOT}/images"
  if [[ -d "$images_dir" ]]; then
    echo "================================================================================"
    echo "步骤 2/2：最小外接矩形"
    echo "================================================================================"
    run_min_rect_job "$FRAMES_ROOT" "$OUTPUT_ROOT" "视频帧：手拿商品最小外接矩形"
  else
    echo "================================================================================"
    echo "步骤 2/2：跳过统一 min_rect（批量子目录模式）"
    echo "================================================================================"
    echo "[INFO] 批量抽帧已写入 ${FRAMES_ROOT}/<job_name>/images/"
    echo "[INFO] 请对每个子目录单独跑 run_images.sh，或设置 FLAT_OUTPUT=1 合并后再衔接。"
    echo "[INFO] 批次清单：${FRAMES_ROOT}/video_frames_batch_manifest.json"
  fi
}

if [[ "$DETACH" == "1" && "${MIN_RECT_CHILD:-}" != "1" ]]; then
  mkdir -p "$LOG_DIR"
  nohup env \
    MIN_RECT_CHILD=1 \
    VIDEO_URL="$VIDEO_URL" \
    VIDEO_URL_FILE="$VIDEO_URL_FILE" \
    VIDEO_ROLE="$VIDEO_ROLE" \
    MAX_VIDEOS="$MAX_VIDEOS" \
    FLAT_OUTPUT="$FLAT_OUTPUT" \
    FRAMES_ROOT="$FRAMES_ROOT" \
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
    OUTPUT_ZIP="$OUTPUT_ZIP" \
    FPS="$FPS" \
    EVERY_N="$EVERY_N" \
    MAX_FRAMES="$MAX_FRAMES" \
    START_SEC="$START_SEC" \
    END_SEC="$END_SEC" \
    FRAME_PREFIX="$FRAME_PREFIX" \
    IMAGE_EXT="$IMAGE_EXT" \
    JPEG_QUALITY="$JPEG_QUALITY" \
    KEEP_VIDEO="$KEEP_VIDEO" \
    FRAMES_OVERWRITE="$FRAMES_OVERWRITE" \
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
  run_video_job
else
  mkdir -p "$LOG_DIR"
  run_video_job 2>&1 | tee -a "$LOG_FILE"
fi
