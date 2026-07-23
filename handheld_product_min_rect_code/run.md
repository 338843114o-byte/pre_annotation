#!/usr/bin/env bash
# 入口说明：请按场景选用下面两个脚本之一（本文件仅作指引，不执行任务）。
set -euo pipefail

cat <<'EOF'
请选用对应运行脚本：

1) 视频抽帧 → 最小外接矩形
   bash run_video.sh

   输入二选一：
   - VIDEO_URL：单个 URL / 本地视频
   - VIDEO_URL_FILE：列表文件（vedio_url 三种格式均支持）
       · 一行一个 URL 的 .txt
       · 「视频URL」列的 .xlsx
       · 「订单视频」JSON 数组列的重叠订单 .xlsx

   示例（单个）：
   export VIDEO_URL='https://example.com/a.mp4'   # 或本地 /path/to.mp4
   export FRAMES_ROOT=/home/data_manager/jiangfan/video_frames_ds
   export OUTPUT_ROOT=/home/data_manager/jiangfan/video_minrect_out
   export FPS=2
   export FRAMES_OVERWRITE=1
   export OVERWRITE=1
   export DEVICE=0
   bash run_video.sh

   示例（URL 列表）：
   export VIDEO_URL_FILE=/home/data_manager/jiangfan/vedio_url/一行一个URL.txt
   # export VIDEO_URL_FILE=/home/data_manager/jiangfan/vedio_url/一行一个URL.xlsx
   # export VIDEO_URL_FILE=/home/data_manager/jiangfan/vedio_url/重叠订单_查询结果.xlsx
   export VIDEO_ROLE=main          # all|main|sub
   export MAX_VIDEOS=2             # 试跑可限条数；0=不限制
   export FRAMES_ROOT=/home/data_manager/jiangfan/video_frames_ds
   export FRAMES_OVERWRITE=1
   export FPS=2
   bash run_video.sh
   # 批量默认每个视频一个子目录；FLAT_OUTPUT=1 可合并到同一 images/

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
