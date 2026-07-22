# 手拿商品最小外接矩形标注脚本

本脚本使用“商品 YOLO 检测结果 + 图片同名 JSON 的原有标注”，为每张图追加一个覆盖全部手拿商品的最小外接矩形。

## 1. JSON 修改保证

- 原 JSON 顶层字段和值不变。
- 原 `shapes` 中每个 shape 的字段、值、顺序均不变。
- 只在 `shapes` 数组末尾追加一个 `label` 为 `最小外接矩形` 的新 shape。
- 新 shape 沿用当前 JSON 已有 shape 的字段结构，`shape_type` 为 `rotation`，`points` 为四个顶点。
- 默认把完整数据集复制到新输出目录后再修改，输入目录不受影响。
- 默认检测到已有 `最小外接矩形` 时跳过，防止断点续跑产生重复标注。

新增内容示意：

```json
{
  "label": "最小外接矩形",
  "score": null,
  "points": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
  "group_id": null,
  "description": "",
  "difficult": false,
  "shape_type": "rotation",
  "flags": {},
  "attributes": {},
  "kie_linking": [],
  "direction": 0.0
}
```

实际字段及字段顺序会以每个 JSON 中已有的 shape 为模板，不会强制变成上面固定顺序。

## 2. 如何确定“手拿商品”

默认流程：

1. **生成几何以 YOLO 为准**：有手时只保留与手重叠的 YOLO 商品框；无手时使用全部 YOLO 商品框。重叠 YOLO 去重。JSON 的瓶装/未定义包装/严重遮挡/过于模糊等**不参与生成几何**。
2. **聚类**：商品↔商品 AABB 重叠可传递合并；手↔商品真实相交则并入同一类（一只手可桥接多件不相交商品）；**手↔手永不合并**，也不能单独成类。
3. **缩圈**：每类先框全部（商品+接触手）→ 去手 → 逐商品；含 12px 边距后 OBB 边长不超过原图对应边一半；贴边侧去掉边距；单品仍超则取消限制。
4. **覆盖校验**：每个 JSON 手持物品相对各最小外接矩形取最大「交面积/物品面积」，&lt;0.9 则上报路径与 label。

这个逻辑不会把货架上未与手关联的 YOLO 框（在有手时）直接放进矩形。

## 3. 数据目录要求

脚本递归查找成对目录：

```text
数据集根目录/
└── 任意子目录/
    ├── images/
    │   ├── a.jpg
    │   └── b.jpg
    └── json_labels/
        ├── a.json
        └── b.json
```

图片与 JSON 必须同名，仅扩展名不同。

## 4. 直接运行

本压缩包不创建新环境，直接使用你已有的 Python/Conda 环境。

```bash
cd /脚本解压目录/handheld_product_min_rect_code

export DATASET_ROOT=/home/data_manager/jiangfan/1
export WEIGHTS=/你的/商品YOLO模型/best.pt
export OUTPUT_ROOT=/home/data_manager/jiangfan/handheld_product_min_rect_labeled
export PYTHON_BIN=python
export DEVICE=0

bash run.sh
```

如果当前环境缺依赖：

```bash
python -m pip install -r requirements.txt
```

## 5. 推荐明确设置商品标签

如果 JSON 里除商品外还有其他业务标签，推荐明确列出商品标签，避免自动识别包含无关 shape：

```bash
export PRODUCT_LABELS='瓶装,罐装,袋装,盒装,未定义包装,严重遮挡,过于模糊'
bash run.sh
```

如果商品 YOLO 模型中所有类别都是商品，`PRODUCT_CLASSES` 留空即可。若模型还检测手、头或其他目标，应只填写商品类别编号：

```bash
export PRODUCT_CLASSES='0,2,3'
bash run.sh
```

## 6. 断点续跑与后台运行

输出目录已存在时继续：

```bash
export RESUME=1
bash run.sh
```

后台运行：

```bash
export RESUME=1        # 首次运行可不设置
export DETACH=1
bash run.sh
```

启动后终端会显示日志路径，可使用 `tail -f 日志路径` 查看进度。每个 JSON 已有 `最小外接矩形` 时会自动跳过。

完全覆盖已有输出目录重新运行：

```bash
export OVERWRITE=1
bash run.sh
```

请确认 `OUTPUT_ROOT` 路径正确后再使用覆盖模式。

## 7. 水平矩形、边距和其他参数

默认生成最小面积旋转矩形：

```bash
export RECTANGLE_MODE=min_area
```

如需水平矩形：

```bash
export RECTANGLE_MODE=axis_aligned
```

外扩 5 像素：

```bash
export MARGIN=5
```

常用推理参数：

```bash
export IMGSZ=1024
export CONF=0.25
export IOU=0.70
export MIN_MATCH_IOU=0.05
export MIN_MATCH_OVERLAP=0.20
export HAND_EXPAND_RATIO=0.15
```

## 8. 直接调用 Python

```bash
python add_handheld_product_min_rect.py \
  --dataset_root /home/data_manager/jiangfan/1 \
  --weights /你的/商品YOLO模型/best.pt \
  --output_root /home/data_manager/jiangfan/handheld_product_min_rect_labeled \
  --device 0 \
  --imgsz 1024 \
  --conf 0.25 \
  --iou 0.70 \
  --product_labels '瓶装,罐装,袋装,盒装' \
  --rectangle_mode min_area \
  --skip_existing
```

只测试推理和统计、不写 JSON：

```bash
python add_handheld_product_min_rect.py \
  --dataset_root /home/data_manager/jiangfan/1 \
  --weights /你的/商品YOLO模型/best.pt \
  --device 0 \
  --dry_run
```

## 9. 无新增结果时的排查

- 日志提示“既没有商品锚点，也没有手部标注”：设置正确的 `PRODUCT_LABELS`，或检查 JSON 的 `label` 名称。
- YOLO 模型还含非商品类别：设置 `PRODUCT_CLASSES`。
- 商品框与 JSON 锚点偏差很大：适当降低 `MIN_MATCH_IOU` 或 `MIN_MATCH_OVERLAP`。
- JSON 中手部与商品相距较远：适当增大 `HAND_EXPAND_RATIO`，但过大可能引入货架商品。
- 脚本不会在没有 JSON 商品锚点且没有手部标注时使用全部 YOLO 框，以免误标整排货架商品。
