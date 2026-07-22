#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASET_ROOT="${DATASET_ROOT:-/home/data_manager/DataPipes/Yolo_Detetcion_Datapipe/Storage/data/labeled/incremental_hand}"
WEIGHTS="${WEIGHTS:-/home/data_manager/jiangfan/for_hands.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/data_manager/DataPipes/Yolo_Detetcion_Datapipe/Storage/data/labeled/incramental_hand1}"
DEVICE="${DEVICE:-0}"
CONF="${CONF:-0.25}"
IOU="${IOU:-0.70}"
IMGSZ="${IMGSZ:-1024}"
PYTHON_BIN="${PYTHON_BIN:-/home/data_manager/.conda/envs/hand_head_label/bin/python}"
PROGRESS_EVERY="${PROGRESS_EVERY:-100}"
RESUME="${RESUME:-0}"
DETACH="${DETACH:-0}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/${RUN_ID}}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/run.log}"
STATUS_FILE="${STATUS_FILE:-${LOG_DIR}/progress.json}"
PID_FILE="${PID_FILE:-${LOG_DIR}/run.pid}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [[ "$DEVICE" == "cpu" ]]; then
  unset CUDA_VISIBLE_DEVICES
  PY_DEVICE="cpu"
else
  export CUDA_VISIBLE_DEVICES="$DEVICE"
  PY_DEVICE="0"
fi

run_job() {
  if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "[ERROR] 未找到 Python：$PYTHON_BIN" >&2
    exit 1
  fi

  if ! "$PYTHON_BIN" -c "import torch, ultralytics" >/dev/null 2>&1; then
    echo "[ERROR] 当前 Python 缺少 torch 或 ultralytics：$("$PYTHON_BIN" -c 'import sys; print(sys.executable)')" >&2
    echo "可尝试：export PYTHON_BIN=/home/data_manager/.conda/envs/hand_head_label/bin/python" >&2
    exit 1
  fi

  if [[ "$PY_DEVICE" != "cpu" ]]; then
    if ! "$PYTHON_BIN" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1; then
      echo "[ERROR] 当前 Python 无法使用 CUDA（device=${DEVICE}）。" >&2
      "$PYTHON_BIN" -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available())" >&2 || true
      echo "可尝试：export PYTHON_BIN=/home/data_manager/.conda/envs/hand_head_label/bin/python" >&2
      echo "或改用 CPU：export DEVICE=cpu" >&2
      exit 1
    fi
  fi

  if [[ ! -d "$DATASET_ROOT" ]]; then
    echo "[ERROR] 数据集目录不存在：$DATASET_ROOT" >&2
    exit 1
  fi

  if [[ ! -f "$WEIGHTS" ]]; then
    echo "[ERROR] YOLO 模型不存在：$WEIGHTS" >&2
    exit 1
  fi

  if [[ ! -f "${SCRIPT_DIR}/add_hand_head_yolo_labels.py" ]]; then
    echo "[ERROR] 处理脚本不存在：${SCRIPT_DIR}/add_hand_head_yolo_labels.py" >&2
    exit 1
  fi

  mkdir -p "$LOG_DIR"

  local resume_args=()
  if [[ -n "${OUTPUT_ROOT}" && -e "$OUTPUT_ROOT" ]]; then
    if [[ "$RESUME" == "1" ]]; then
      resume_args+=(--resume)
      echo "[INFO] 检测到已有输出目录，启用 --resume 继续处理：$OUTPUT_ROOT"
    else
      echo "[ERROR] 输出目录已经存在：$OUTPUT_ROOT" >&2
      echo "如需断点续跑：export RESUME=1" >&2
      echo "如需覆盖重来：先删除该目录，或自行备份后删除。" >&2
      exit 1
    fi
  fi

  echo "================================================================================"
  echo "手部 YOLO 标注"
  echo "================================================================================"
  echo "Python：       $("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"
  echo "数据集路径：  $DATASET_ROOT"
  echo "模型路径：    $WEIGHTS"
  echo "输出路径：    $OUTPUT_ROOT"
  echo "物理显卡：    ${DEVICE}"
  echo "推理 device： ${PY_DEVICE}"
  echo "置信度 conf： ${CONF}"
  echo "日志文件：    $LOG_FILE"
  echo "进度文件：    $STATUS_FILE"
  echo "类别 0：      手"
  echo "类别 1：      忽略，不写入 JSON"
  echo "================================================================================"

  local output_args=()
  if [[ -n "${OUTPUT_ROOT}" ]]; then
    output_args+=(--output_root "$OUTPUT_ROOT")
  else
    echo "[INFO] 未设置 OUTPUT_ROOT：将原地修改 DATASET_ROOT 中的 JSON"
  fi

  "$PYTHON_BIN" "${SCRIPT_DIR}/add_hand_head_yolo_labels.py" \
    --dataset_root "$DATASET_ROOT" \
    --weights "$WEIGHTS" \
    "${output_args[@]}" \
    --device "$PY_DEVICE" \
    --imgsz "$IMGSZ" \
    --conf "$CONF" \
    --iou "$IOU" \
    --hand_label "手" \
    --skip_existing \
    --progress_every "$PROGRESS_EVERY" \
    --status_file "$STATUS_FILE" \
    "${resume_args[@]}"

  echo "================================================================================"
  echo "处理完成"
  if [[ -n "${OUTPUT_ROOT}" ]]; then
    echo "结果目录：$OUTPUT_ROOT"
  else
    echo "结果目录：原地修改 $DATASET_ROOT"
  fi
  echo "日志文件：$LOG_FILE"
  echo "进度文件：$STATUS_FILE"
  echo "================================================================================"
}

if [[ "$DETACH" == "1" && "${HAND_HEAD_CHILD:-}" != "1" ]]; then
  mkdir -p "$LOG_DIR"
  echo "[INFO] 后台模式启动，日志：$LOG_FILE"
  echo "[INFO] 查看进度：tail -f $LOG_FILE"
  echo "[INFO] 状态 JSON：cat $STATUS_FILE"
  nohup env \
    HAND_HEAD_CHILD=1 \
    DATASET_ROOT="$DATASET_ROOT" \
    WEIGHTS="$WEIGHTS" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    DEVICE="$DEVICE" \
    CONF="$CONF" \
    IOU="$IOU" \
    IMGSZ="$IMGSZ" \
    PYTHON_BIN="$PYTHON_BIN" \
    PROGRESS_EVERY="$PROGRESS_EVERY" \
    RESUME="$RESUME" \
    RUN_ID="$RUN_ID" \
    LOG_DIR="$LOG_DIR" \
    LOG_FILE="$LOG_FILE" \
    STATUS_FILE="$STATUS_FILE" \
    PID_FILE="$PID_FILE" \
    bash "$0" >>"$LOG_FILE" 2>&1 &
  child_pid=$!
  echo "$child_pid" >"$PID_FILE"
  echo "[INFO] 已启动后台任务，PID=$child_pid"
  echo "[INFO] PID 文件：$PID_FILE"
  exit 0
fi

if [[ "${HAND_HEAD_CHILD:-}" == "1" ]]; then
  run_job
else
  mkdir -p "$LOG_DIR"
  run_job 2>&1 | tee -a "$LOG_FILE"
fi
