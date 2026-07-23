#!/usr/bin/env bash
# 入口说明：请按场景选用下面两个脚本之一（本文件仅作指引，不执行任务）。
set -euo pipefail

cat <<'EOF'
请选用对应运行脚本：

1) 视频抽帧 → 最小外接矩形
   bash run_video.sh

   示例：
   export VIDEO_URL='https://example.com/a.mp4'   # 或本地 /path/to.mp4
   export FRAMES_ROOT=/home/data_manager/jiangfan/video_frames_ds
   export OUTPUT_ROOT=/home/data_manager/jiangfan/video_minrect_out
   export FPS=2
   export FRAMES_OVERWRITE=1
   export OVERWRITE=1
   export DEVICE=0
   bash run_video.sh

2) 已有图片 → 最小外接矩形
   bash run_images.sh

   有真值 json_labels 时：只追加最小外接矩形到真值 JSON，不另写 YOLO 标注。
   跑完后默认对 OUTPUT_ROOT 做单文件覆盖校验（VALIDATE=1）。

   示例：
   export DATASET_ROOT=/home/data_manager/jiangfan/1
   export OUTPUT_ROOT=/home/data_manager/jiangfan/1_yolo_minrect
   export LABEL_SOURCE=yolo
   export OVERWRITE=1
   export DEVICE=0
   export VALIDATE=1
   export VALIDATE_COVERAGE=0.9
   bash run_images.sh

   单独跑校验：
   python validate_gt_min_rect_coverage.py \
     --result_root /home/data_manager/jiangfan/1_yolo_minrect \
     --coverage_threshold 0.9

公共参数（权重、conf、product_labels 等）见 run_min_rect_lib.sh。
EOF
exit 1
