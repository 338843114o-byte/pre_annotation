# 手拿商品最小外接矩形标注脚本

本脚本支持两种标注来源：

- **json（默认）**：商品 YOLO + 图片同名 JSON 的原有标注，追加手拿商品最小外接矩形。
- **yolo**：无现成 JSON 时，用手数模（`for_hands.pt`）+ 商品模（`for_skus.pt`）检测，先写出完整 LabelMe 标签，再按同一套几何逻辑生成最小外接矩形。

## 1. JSON 修改保证（label_source=json）

- 原 JSON 顶层字段和值不变。
- 原 `shapes` 中每个 shape 的字段、值、顺序均不变。
- 只在 `shapes` 数组末尾追加一个 `label` 为 `最小外接矩形_N` 的新 shape。
- 新 shape 沿用当前 JSON 已有 shape 的字段结构，`shape_type` 为 `rotation`，`points` 为四个顶点。
- 默认把完整数据集复制到新输出目录后再修改，输入目录不受影响。
- 默认检测到已有 `最小外接矩形` 时跳过，防止断点续跑产生重复标注。

## 2. 无 JSON / 有真值 JSON 时的 yolo 模式

与纯 json 模式的差异：最小外接矩形的几何由 YOLO 检测计算。

- **有真值 JSON**（`images/` 同级 `json_labels/`）：**不覆盖、不重写**原有标注，只把生成的 `最小外接矩形_N` 追加到真值 JSON 末尾。
- **无真值 JSON**：先写出 YOLO 检测 LabelMe 标签，再追加最小外接矩形。

写入前对重叠商品框去重（同一物品多检只留高分）；关闭手旁补检。

类名映射示例：`bottle→瓶装`、`can→罐装`、`hand→手`、`vending_machine→售货柜`、`undefined_pack→未定义包装`、`occluded→严重遮挡`。

运行入口已拆成两个脚本（`bash run.sh` 仅打印用法）：

1. **视频抽帧 → 最小外接矩形**：`bash run_video.sh`
2. **已有图片 → 最小外接矩形**：`bash run_images.sh`

```bash
# 已有图片
export DATASET_ROOT=/你的/仅有images的数据集根目录
export WEIGHTS=/home/data_manager/jiangfan/for_skus.pt
export HAND_WEIGHTS=/home/data_manager/jiangfan/for_hands.pt
export LABEL_SOURCE=yolo
export OUTPUT_ROOT=/home/data_manager/jiangfan/yolo_minrect_out
bash run_images.sh

# 视频 URL / 本地视频
export VIDEO_URL='https://example.com/a.mp4'
export FRAMES_ROOT=/home/data_manager/jiangfan/video_frames_ds
export OUTPUT_ROOT=/home/data_manager/jiangfan/video_minrect_out
export FPS=2
export FRAMES_OVERWRITE=1
export OVERWRITE=1
bash run_video.sh
```

### 真值覆盖校验（已有图片 + json_labels）

有真值时结果 JSON = 真值 shapes + 追加的最小外接矩形。`run_images.sh` 默认对 **OUTPUT_ROOT** 做单文件校验（`VALIDATE=1`）：

- 同一 JSON 内：手持商品（`PRODUCT_LABELS`）相对各 `最小外接矩形` 计算 `交面积/商品面积`
- 最大覆盖率 &lt; `VALIDATE_COVERAGE`（默认 0.9）则上报图片路径
- 报告：`OUTPUT_ROOT/validation_uncovered.json` 与 `.txt`

```bash
python validate_gt_min_rect_coverage.py \
  --result_root /home/data_manager/jiangfan/1_yolo_minrect \
  --coverage_threshold 0.9
```

或直接调用：

```bash
python add_handheld_product_min_rect.py \
  --dataset_root /你的/数据集 \
  --weights /home/data_manager/jiangfan/for_skus.pt \
  --hand_weights /home/data_manager/jiangfan/for_hands.pt \
  --label_source yolo \
  --output_root /home/data_manager/jiangfan/yolo_minrect_out \
  --device 0 \
  --skip_existing
```

## 3. 如何确定“手拿商品”（json 模式）

默认流程（JSON 锚点 + YOLO 手旁补检，重叠去重）：

1. 从 JSON 中取出商品 shape 作为手拿商品锚点。若没有显式设置 `PRODUCT_LABELS`，脚本自动把“手、头、售货柜、遮挡、最小外接矩形”之外的 shape 作为商品锚点。推荐把「未定义包装」「严重遮挡」「过于模糊」「信息不足」等业务标签写进 `PRODUCT_LABELS`。
2. 对每个 JSON 商品锚点，从商品 YOLO 结果中选择匹配度最高的检测框，用于细化边界。
3. 若某个 JSON 商品没有被模型检出，使用原 JSON 商品框兜底，保证原来已标注的手拿商品不会消失。
4. 若 JSON 中有“手”标注，仅对尚未与任何 JSON 商品重叠的手，纳入与之相交的 YOLO 商品框，用来补充漏标；已有商品的手不再拉 YOLO，避免错框。
5. “遮挡”仅在与某个商品框空间重叠时扩充该商品；与所有商品都不重叠时单独成单元。
6. 对最终商品点计算最小面积旋转矩形，并追加到 JSON（label 后缀为框内手持商品数）。

这个逻辑不会把商品 YOLO 检出的全部货架商品直接放进矩形。

## 4. 数据目录要求

```text
数据集根目录/
└── 任意子目录/
    ├── images/
    │   ├── a.jpg
    │   └── b.jpg
    └── json_labels/          # yolo 模式可缺失，脚本会创建
        ├── a.json
        └── b.json
```

json 模式：图片与 JSON 必须同名。yolo 模式：只需 `images/`。

## 4.1 从视频 URL 抽帧再衔接

```bash
cd /脚本目录/handheld_product_min_rect_code

# 只抽帧 → 得到可直接作为 DATASET_ROOT 的目录（含 images/）
python video_url_to_frames.py \
  --video_url 'https://example.com/a.mp4' \
  --output_root /home/data_manager/jiangfan/video_frames_ds \
  --fps 2 \
  --overwrite

# 抽帧后立刻跑 YOLO 最小外接矩形
python video_url_to_frames.py \
  --video_url 'https://example.com/a.mp4' \
  --output_root /home/data_manager/jiangfan/video_frames_ds \
  --fps 2 \
  --overwrite \
  --run_min_rect \
  --weights /home/data_manager/jiangfan/for_skus.pt \
  --hand_weights /home/data_manager/jiangfan/for_hands.pt \
  --min_rect_output_root /home/data_manager/jiangfan/video_minrect_out \
  --device 0
```

常用抽帧参数：`--every_n 5`（每 5 帧取 1）、`--fps 2`、`--max_frames 200`、`--start_sec 10 --end_sec 60`。
本地视频也可：`--video_url /path/to/local.mp4`。

## 5. 直接运行（有 JSON）

```bash
cd /脚本解压目录/handheld_product_min_rect_code

export DATASET_ROOT=/home/data_manager/jiangfan/1
export WEIGHTS=/home/data_manager/jiangfan/for_skus.pt
export OUTPUT_ROOT=/home/data_manager/jiangfan/handheld_product_min_rect_labeled
export PYTHON_BIN=python
export DEVICE=0
export LABEL_SOURCE=json

bash run_images.sh
```

缺依赖时：

```bash
python -m pip install -r requirements.txt
```

## 6. 推荐明确设置商品标签

```bash
export PRODUCT_LABELS='瓶装,罐装,袋装,盒装,未定义包装,严重遮挡,过于模糊,信息不足'
bash run_images.sh
```

若模型还检测手、头或其他目标，应只填写商品类别编号：

```bash
export PRODUCT_CLASSES='0,2,3'
bash run_images.sh
```

## 7. 断点续跑与后台运行

```bash
export RESUME=1
bash run_images.sh
```

```bash
export RESUME=1
export DETACH=1
bash run_images.sh
```

完全覆盖：

```bash
export OVERWRITE=1
bash run_images.sh
```

## 8. 水平矩形、边距和其他参数

```bash
export RECTANGLE_MODE=min_area   # 或 axis_aligned
export MARGIN=5
export IMGSZ=1024
export CONF=0.25
export IOU=0.70
export MIN_MATCH_IOU=0.05
export MIN_MATCH_OVERLAP=0.20
export HAND_EXPAND_RATIO=0.15
```

## 9. 无新增最小外接矩形时的排查

- 日志提示“既没有商品锚点，也没有手部标注”：设置正确的 `PRODUCT_LABELS`，或检查 JSON / YOLO 写出的 `label` 名称。
- YOLO 模型还含非商品类别：设置 `PRODUCT_CLASSES` / `IGNORE_YOLO_CLASS_NAMES`。
- 商品框与 JSON 锚点偏差很大：适当降低 `MIN_MATCH_IOU` 或 `MIN_MATCH_OVERLAP`。
- JSON 中手部与商品相距较远：适当增大 `HAND_EXPAND_RATIO`，但过大可能引入货架商品。
