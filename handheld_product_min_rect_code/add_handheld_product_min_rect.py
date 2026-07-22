#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用商品 YOLO 模型和图片对应 JSON 中的已有标注，追加覆盖手拿商品的
最小外接矩形（按 AABB 重叠连通聚类；必要时缩圈/拆分）。

核心保证：
1. 原 JSON 顶层字段、原 shapes 内容和顺序完全不变。
2. 只在 shapes 数组末尾追加一个或多个 label=“最小外接矩形_N”的 shape（N 为框内手持商品数）。
3. 新 shape 沿用原 JSON 的字段结构；不写入模型路径、类别编号等额外元数据。
4. 支持普通 YOLO 检测框和 YOLO OBB 旋转框。

默认识别逻辑（始终以 JSON 标注为主）：
- JSON 中除“手/头/售货柜/遮挡类/最小外接矩形”外的已有 shape，视为手拿
  商品标注锚点；也可用 --product_labels 明确指定。商品数量与外接矩形单元
  一律由这些 JSON 商品框决定。
- YOLO 仅用于细化已有 JSON 商品锚点的边界；模型漏检、多检、误检都不改变
  JSON 商品单元数量。
- 仅当 JSON 没有任何商品锚点、但有“手”标注时，才回退用与手相交的 YOLO
  商品框补商品（并对重叠检测去重），避免无标注可依。
- 默认忽略 YOLO 中的售货柜/手机等非商品类别，并拒绝远大于锚点的检测框。
- “严重遮挡/遮挡”等辅助区域会参与外接矩形计算，但不会被当成独立商品。
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np


IMAGE_EXTS_DEFAULT = ".jpg,.jpeg,.png,.bmp,.webp"
DEFAULT_IGNORE_LABELS = "手,头,售货柜,最小外接矩形"
DEFAULT_HAND_LABELS = "手"
DEFAULT_AUXILIARY_LABELS = "严重遮挡,遮挡"
# for_skus.pt 等模型里这些类别不是手上商品，默认排除。
DEFAULT_IGNORE_YOLO_CLASS_NAMES = (
    "vending_machine,phone,too_blurry,insufficient_info"
)
# 检测框面积相对 JSON 锚点过大时视为货架级误匹配。
DEFAULT_MAX_MATCH_AREA_RATIO = 8.0
DEFAULT_MARGIN = 12.0
DEFAULT_MARGIN_RATIO = 0.0
DEFAULT_INCLUDE_NEARBY_HANDS = True
# 商品外接框按该比例外扩后，与手有交集则视为接触/持货手。
# 注：纳入最小外接矩形时已改为要求手框与商品框真实 AABB 重叠；下列参数仅兼容保留。
DEFAULT_HAND_NEAR_EXPAND_RATIO = 0.35
# 手中心到商品中心距离 / 商品对角线 的上限；超过则视为远处的手。
DEFAULT_HAND_MAX_CENTER_DIST_RATIO = 1.25
# 含边距后最小外接矩形按 OBB 真实边长：较长边/较短边相对原图较长边/较短边上限（默认 0.5）。
DEFAULT_MAX_SIDE_RATIO = 0.5
# 兼容旧参数名：语义已改为边长比，不再表示面积比。
DEFAULT_MAX_RECT_AREA_RATIO = DEFAULT_MAX_SIDE_RATIO

# 一个商品单元：同一商品的若干多边形（JSON 锚点 + 匹配 YOLO / 遮挡等）。
ProductUnit = List[List[List[float]]]


@dataclass(frozen=True)
class Detection:
    points: List[List[float]]
    score: float
    class_id: int
    class_name: str


@dataclass(frozen=True)
class PolygonMatch:
    matched: bool
    score: float
    iou: float
    overlap_min: float
    anchor_coverage: float
    detection_coverage: float


@dataclass(frozen=True)
class ProcessResult:
    rectangle_added: int
    skipped_existing: bool
    anchor_count: int
    yolo_detection_count: int
    selected_yolo_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "使用商品 YOLO + 原 JSON 标注，向 shapes 末尾追加覆盖手拿商品的"
            "最小外接矩形（必要时拆成多个）。"
        )
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--dataset_root",
        type=str,
        help="已解压数据集根目录；递归查找 images/json_labels 配对。",
    )
    input_group.add_argument(
        "--input_zip",
        type=str,
        help="输入数据集 zip；先解压到 --work_dir，再处理。",
    )

    parser.add_argument("--weights", type=str, required=True, help="商品 YOLO 权重路径。")
    parser.add_argument(
        "--work_dir", type=str, default="./min_rect_work", help="input_zip 解压目录。"
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help=(
            "输出目录。推荐填写：脚本先完整复制数据集，再只修改副本 JSON；"
            "不填写则原地修改。"
        ),
    )
    parser.add_argument("--output_zip", type=str, default=None, help="可选结果 zip 路径。")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖输出目录/zip。")
    parser.add_argument(
        "--resume", action="store_true", help="输出目录已存在时继续处理，不重新复制。"
    )

    parser.add_argument("--image_dir_name", type=str, default="images")
    parser.add_argument("--json_dir_name", type=str, default="json_labels")
    parser.add_argument("--image_exts", type=str, default=IMAGE_EXTS_DEFAULT)

    parser.add_argument("--device", type=str, default="0", help="例如 0、1 或 cpu。")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument(
        "--product_classes",
        type=str,
        default="",
        help=(
            "可选 YOLO 商品 class_id，逗号分隔；留空表示除 ignore_yolo_class_names "
            "外的全部类别都可作为商品。"
        ),
    )
    parser.add_argument(
        "--ignore_yolo_class_names",
        type=str,
        default=DEFAULT_IGNORE_YOLO_CLASS_NAMES,
        help=(
            "按类别名忽略的 YOLO 检测，逗号分隔。"
            "默认排除售货柜/手机/模糊等非手上商品类别。"
        ),
    )
    parser.add_argument(
        "--max_match_area_ratio",
        type=float,
        default=DEFAULT_MAX_MATCH_AREA_RATIO,
        help=(
            "YOLO 框面积 / JSON 锚点面积 的上限；超过则拒绝匹配，"
            "避免巨型售货柜框包住商品后被选中。默认 8。"
        ),
    )

    parser.add_argument(
        "--product_labels",
        type=str,
        default="",
        help=(
            "JSON 中代表手拿商品的 label，逗号分隔，如 瓶装,罐装,袋装。"
            "支持前缀匹配：袋装_SKU名 也会被当成袋装。"
            "留空时自动使用排除非商品标签后的所有已有 shape。"
        ),
    )
    parser.add_argument(
        "--ignore_labels",
        type=str,
        default=DEFAULT_IGNORE_LABELS,
        help="自动识别商品锚点时忽略的 label，逗号分隔。",
    )
    parser.add_argument(
        "--hand_labels",
        type=str,
        default=DEFAULT_HAND_LABELS,
        help="JSON 中手部 label，逗号分隔。",
    )
    parser.add_argument(
        "--auxiliary_labels",
        type=str,
        default=DEFAULT_AUXILIARY_LABELS,
        help="参与包围范围但不算独立商品的辅助 label，逗号分隔。",
    )

    parser.add_argument(
        "--min_match_iou",
        type=float,
        default=0.05,
        help="YOLO 框与 JSON 商品锚点匹配的最小 IoU。",
    )
    parser.add_argument(
        "--min_match_overlap",
        type=float,
        default=0.20,
        help="YOLO 框与 JSON 锚点交集/较小面积的最小比例。",
    )
    parser.add_argument(
        "--hand_expand_ratio",
        type=float,
        default=0.15,
        help="使用手部框补充商品时，手部外接框向四周扩展比例。",
    )
    parser.add_argument(
        "--include_hand_matched_detections",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "仅当 JSON 无商品锚点但有手部标注时，是否用与手相交的 YOLO 商品框回退补商品。"
            "JSON 已有商品锚点时始终以 JSON 为准，不会因 YOLO 另开商品单元。"
        ),
    )
    parser.add_argument(
        "--include_nearby_hands",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_INCLUDE_NEARBY_HANDS,
        help=(
            "是否把手纳入聚类：与商品框 AABB 重叠的手进入同一类；"
            "无重叠的手不进入任何类。"
        ),
    )
    parser.add_argument(
        "--hand_near_expand_ratio",
        type=float,
        default=DEFAULT_HAND_NEAR_EXPAND_RATIO,
        help="兼容保留；聚类已改为手/商品框真实 AABB 重叠。",
    )
    parser.add_argument(
        "--hand_max_center_dist_ratio",
        type=float,
        default=DEFAULT_HAND_MAX_CENTER_DIST_RATIO,
        help="兼容保留；聚类已改为手/商品框真实 AABB 重叠。",
    )
    parser.add_argument(
        "--max_side_ratio",
        type=float,
        default=DEFAULT_MAX_SIDE_RATIO,
        help=(
            "含边距后最小外接矩形按 OBB 真实边长："
            "较长边≤原图较长边*ratio，较短边≤原图较短边*ratio（默认 0.5）。"
            "每类先框全部→去手→逐商品；单商品仍超则取消限制（例外）。"
        ),
    )
    parser.add_argument(
        "--max_rect_area_ratio",
        type=float,
        default=None,
        help="已弃用：等同 --max_side_ratio（边长比，不再是面积比）。",
    )

    parser.add_argument(
        "--rectangle_mode",
        choices=("min_area", "axis_aligned"),
        default="min_area",
        help="min_area=最小面积旋转矩形；axis_aligned=水平最小外接矩形。",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=DEFAULT_MARGIN,
        help="外接矩形向外扩展像素（默认 12；所有路径都加）。",
    )
    parser.add_argument(
        "--margin_ratio",
        type=float,
        default=DEFAULT_MARGIN_RATIO,
        help="相对边长外扩比例；默认 0（只使用固定 margin 像素）。",
    )
    parser.add_argument(
        "--rect_label",
        type=str,
        default="最小外接矩形",
        help="新增 shape 的 label 基础名；实际写入为「基础名_手持商品数」，如 最小外接矩形_2。",
    )
    parser.add_argument(
        "--skip_existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="JSON 已有 rect_label 时跳过，防止重复追加；默认启用。",
    )
    parser.add_argument(
        "--require_yolo_match",
        action="store_true",
        help="启用后，模型没有任何匹配商品时不追加；默认会用 JSON 商品框兜底。",
    )
    parser.add_argument(
        "--backup_json", action="store_true", help="原地修改时为 JSON 创建一次 .bak。"
    )
    parser.add_argument("--dry_run", action="store_true", help="只推理统计，不写文件。")
    parser.add_argument("--progress_every", type=int, default=100)
    parser.add_argument("--status_file", type=str, default=None)
    return parser.parse_args()


def parse_string_set(value: str) -> Set[str]:
    return {part.strip() for part in value.split(",") if part.strip()}


def label_matches(label: str, labels: Set[str]) -> bool:
    """精确匹配，或前缀匹配（如 袋装_SKU名 匹配 袋装）。"""
    if not label or not labels:
        return False
    if label in labels:
        return True
    for item in labels:
        if label.startswith(item + "_") or label.startswith(item + "-"):
            return True
    return False


def parse_int_list(value: str) -> Optional[List[int]]:
    values = [part.strip() for part in value.split(",") if part.strip()]
    if not values:
        return None
    result = sorted({int(part) for part in values})
    if any(item < 0 for item in result):
        raise ValueError("product_classes 不能包含负数")
    return result


def load_json_text(path: Path) -> Tuple[str, Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        text = file.read()
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"JSON 顶层必须是对象：{path}")
    return text, data


def get_image_size_from_json_or_file(
    json_data: Dict[str, Any], image_path: Path
) -> Tuple[int, int]:
    width = json_data.get("imageWidth")
    height = json_data.get("imageHeight")
    if (
        isinstance(width, int)
        and not isinstance(width, bool)
        and isinstance(height, int)
        and not isinstance(height, bool)
        and width > 0
        and height > 0
    ):
        return width, height

    from PIL import Image

    with Image.open(image_path) as image:
        return int(image.width), int(image.height)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def clamp_points(
    points: Sequence[Sequence[float]], width: int, height: int
) -> List[List[float]]:
    max_x = max(0, width - 1)
    max_y = max(0, height - 1)
    return [
        [
            round(clamp(float(point[0]), 0, max_x), 3),
            round(clamp(float(point[1]), 0, max_y), 3),
        ]
        for point in points
    ]


def xyxy_to_points(
    xyxy: Sequence[float], width: int, height: int
) -> List[List[float]]:
    x1, y1, x2, y2 = [float(value) for value in xyxy]
    return clamp_points(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], width, height
    )


def shape_polygon(shape: Any, width: int, height: int) -> Optional[List[List[float]]]:
    if not isinstance(shape, dict):
        return None
    raw_points = shape.get("points")
    if not isinstance(raw_points, list) or len(raw_points) < 2:
        return None
    points: List[List[float]] = []
    for point in raw_points:
        if (
            not isinstance(point, (list, tuple))
            or len(point) < 2
            or isinstance(point[0], bool)
            or isinstance(point[1], bool)
        ):
            return None
        try:
            x = float(point[0])
            y = float(point[1])
        except (TypeError, ValueError):
            return None
        if not math.isfinite(x) or not math.isfinite(y):
            return None
        points.append([x, y])

    # 兼容 LabelMe rectangle 的两个对角点。
    if len(points) == 2:
        x1, y1 = points[0]
        x2, y2 = points[1]
        points = [
            [min(x1, x2), min(y1, y2)],
            [max(x1, x2), min(y1, y2)],
            [max(x1, x2), max(y1, y2)],
            [min(x1, x2), max(y1, y2)],
        ]
    if len(points) < 3:
        return None
    return clamp_points(points, width, height)


def collect_json_polygons(
    shapes: List[Any],
    width: int,
    height: int,
    *,
    product_labels: Set[str],
    ignore_labels: Set[str],
    hand_labels: Set[str],
    auxiliary_labels: Set[str],
    rect_label: str,
) -> Tuple[List[List[List[float]]], List[List[List[float]]], List[List[List[float]]]]:
    anchors: List[List[List[float]]] = []
    hands: List[List[List[float]]] = []
    auxiliary: List[List[List[float]]] = []
    explicit_product_labels = bool(product_labels)

    for shape in shapes:
        if not isinstance(shape, dict):
            continue
        label = str(shape.get("label", "")).strip()
        polygon = shape_polygon(shape, width, height)
        if polygon is None:
            continue
        if label in hand_labels:
            hands.append(polygon)
            continue
        if label in auxiliary_labels:
            auxiliary.append(polygon)
            continue
        if is_rect_label(label, rect_label):
            continue
        if explicit_product_labels:
            if label_matches(label, product_labels):
                anchors.append(polygon)
        elif label and label not in ignore_labels:
            anchors.append(polygon)
    return anchors, hands, auxiliary


def load_yolo_model(weights: str):
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(
            "无法导入 ultralytics，请先在当前 Python 环境安装 requirements.txt"
        ) from exc
    return YOLO(weights)


def lookup_class_name(result: Any, class_id: int) -> str:
    names = getattr(result, "names", None)
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def detections_from_result(
    result: Any, width: int, height: int
) -> List[Detection]:
    detections: List[Detection] = []
    obb = getattr(result, "obb", None)
    if obb is not None and len(obb) > 0:
        polygons = obb.xyxyxyxy.detach().cpu().numpy().reshape(-1, 4, 2)
        confidences = obb.conf.detach().cpu().numpy()
        class_ids = obb.cls.detach().cpu().numpy().astype(int)
        for polygon, confidence, class_id in zip(polygons, confidences, class_ids):
            class_id = int(class_id)
            detections.append(
                Detection(
                    points=clamp_points(polygon.tolist(), width, height),
                    score=float(confidence),
                    class_id=class_id,
                    class_name=lookup_class_name(result, class_id),
                )
            )
    else:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return detections
        xyxys = boxes.xyxy.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()
        class_ids = boxes.cls.detach().cpu().numpy().astype(int)
        for xyxy, confidence, class_id in zip(xyxys, confidences, class_ids):
            class_id = int(class_id)
            detections.append(
                Detection(
                    points=xyxy_to_points(xyxy, width, height),
                    score=float(confidence),
                    class_id=class_id,
                    class_name=lookup_class_name(result, class_id),
                )
            )
    detections.sort(
        key=lambda item: (
            item.points[0][1],
            item.points[0][0],
            item.class_id,
            -item.score,
        )
    )
    return detections


def as_convex_polygon(points: Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    hull = cv2.convexHull(array).reshape(-1, 2)
    return hull.astype(np.float32)


def polygon_area(points: Sequence[Sequence[float]]) -> float:
    polygon = as_convex_polygon(points)
    return abs(float(cv2.contourArea(polygon)))


def polygon_center(points: Sequence[Sequence[float]]) -> Tuple[float, float]:
    array = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    return float(array[:, 0].mean()), float(array[:, 1].mean())


def point_inside_polygon(
    point: Tuple[float, float], polygon: Sequence[Sequence[float]]
) -> bool:
    hull = as_convex_polygon(polygon)
    return cv2.pointPolygonTest(hull, point, False) >= 0


def compare_polygons(
    anchor: Sequence[Sequence[float]],
    detection: Sequence[Sequence[float]],
    min_iou: float,
    min_overlap: float,
    *,
    max_area_ratio: Optional[float] = None,
) -> PolygonMatch:
    anchor_poly = as_convex_polygon(anchor)
    detection_poly = as_convex_polygon(detection)
    anchor_area = abs(float(cv2.contourArea(anchor_poly)))
    detection_area = abs(float(cv2.contourArea(detection_poly)))
    if anchor_area <= 1e-6 or detection_area <= 1e-6:
        return PolygonMatch(False, 0.0, 0.0, 0.0, 0.0, 0.0)

    # 巨型框（如整排售货柜）完全包住小商品时，overlap_min 会接近 1，
    # 必须用面积比直接拒绝，否则会把货架圈进最小外接矩形。
    if max_area_ratio is not None and max_area_ratio > 0:
        area_ratio = max(detection_area / anchor_area, anchor_area / detection_area)
        if area_ratio > max_area_ratio:
            return PolygonMatch(False, 0.0, 0.0, 0.0, 0.0, 0.0)

    intersection_area, _ = cv2.intersectConvexConvex(anchor_poly, detection_poly)
    intersection = max(0.0, float(intersection_area))
    union = anchor_area + detection_area - intersection
    iou = intersection / union if union > 0 else 0.0
    overlap_min = intersection / min(anchor_area, detection_area)
    anchor_coverage = intersection / anchor_area
    detection_coverage = intersection / detection_area
    center_hit = point_inside_polygon(polygon_center(anchor), detection) or point_inside_polygon(
        polygon_center(detection), anchor
    )
    matched = iou >= min_iou or overlap_min >= min_overlap or center_hit
    # 优先尺寸接近、彼此覆盖率都高的匹配；单纯“大框包小框”不再占优。
    score = 0.45 * iou + 0.35 * min(anchor_coverage, detection_coverage) + 0.20 * overlap_min
    if center_hit:
        score += 0.05
    return PolygonMatch(
        matched=matched,
        score=score,
        iou=iou,
        overlap_min=overlap_min,
        anchor_coverage=anchor_coverage,
        detection_coverage=detection_coverage,
    )


def filter_product_detections(
    detections: Sequence[Detection],
    *,
    product_classes: Optional[Set[int]],
    ignore_class_names: Set[str],
) -> List[Detection]:
    """只保留真正的商品检测，排除售货柜等背景类。"""
    ignored = {name.strip().lower() for name in ignore_class_names if name.strip()}
    filtered: List[Detection] = []
    for detection in detections:
        if product_classes is not None and detection.class_id not in product_classes:
            continue
        if detection.class_name.strip().lower() in ignored:
            continue
        filtered.append(detection)
    return filtered


def expanded_axis_aligned_polygon(
    points: Sequence[Sequence[float]],
    ratio: float,
    width: int,
    height: int,
) -> List[List[float]]:
    array = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    x1, y1 = array.min(axis=0)
    x2, y2 = array.max(axis=0)
    expand_x = max(1.0, float(x2 - x1) * max(0.0, ratio))
    expand_y = max(1.0, float(y2 - y1) * max(0.0, ratio))
    return clamp_points(
        [
            [x1 - expand_x, y1 - expand_y],
            [x2 + expand_x, y1 - expand_y],
            [x2 + expand_x, y2 + expand_y],
            [x1 - expand_x, y2 + expand_y],
        ],
        width,
        height,
    )


def polygons_axis_aligned_bounds(
    polygons: Sequence[Sequence[Sequence[float]]],
) -> Optional[Tuple[float, float, float, float]]:
    arrays = [
        np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        for polygon in polygons
        if polygon
    ]
    arrays = [array for array in arrays if len(array) > 0]
    if not arrays:
        return None
    all_points = np.concatenate(arrays, axis=0)
    x1, y1 = all_points.min(axis=0)
    x2, y2 = all_points.max(axis=0)
    return float(x1), float(y1), float(x2), float(y2)


def aabb_intersects(
    a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]
) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def polygons_intersect(
    poly_a: Sequence[Sequence[float]],
    poly_b: Sequence[Sequence[float]],
    *,
    min_area: float = 1e-3,
) -> bool:
    """
    两多边形是否真实相交（面积 > min_area）或一方顶点落入另一方。
    先用 AABB 快筛，避免斜框 AABB 假重叠被当成真重叠。
    """
    a = np.asarray(poly_a, dtype=np.float32).reshape(-1, 2)
    b = np.asarray(poly_b, dtype=np.float32).reshape(-1, 2)
    if len(a) < 3 or len(b) < 3:
        return False
    ba = polygons_axis_aligned_bounds([a.tolist()])
    bb = polygons_axis_aligned_bounds([b.tolist()])
    if ba is None or bb is None or not aabb_intersects(ba, bb):
        return False

    def _point_in(poly: np.ndarray, pt: np.ndarray) -> bool:
        return cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False) >= 0

    try:
        _ret, inter = cv2.intersectConvexConvex(a, b)
    except cv2.error:
        inter = None
    if inter is not None and len(inter) >= 3:
        if abs(float(cv2.contourArea(inter))) > float(min_area):
            return True
    for pt in a:
        if _point_in(b, pt):
            return True
    for pt in b:
        if _point_in(a, pt):
            return True
    return False


def is_hand_near_products(
    hand: Sequence[Sequence[float]],
    product_polygons: Sequence[Sequence[Sequence[float]]],
    *,
    near_expand_ratio: float = 0.0,
    max_center_dist_ratio: float = 0.0,
) -> bool:
    """
    判断手框是否与任一商品多边形真实相交（非 AABB 虚高）。

    near_expand_ratio / max_center_dist_ratio 保留兼容，不再用于放宽判定。
    """
    del near_expand_ratio, max_center_dist_ratio
    return hand_directly_overlaps_products(hand, product_polygons)


def select_nearby_hands(
    hands: Sequence[Sequence[Sequence[float]]],
    product_polygons: Sequence[Sequence[Sequence[float]]],
    *,
    near_expand_ratio: float = 0.0,
    max_center_dist_ratio: float = 0.0,
) -> List[List[List[float]]]:
    """只保留与至少一个商品多边形真实相交的手；无重叠则不纳入任何手。"""
    nearby: List[List[List[float]]] = []
    for hand in hands:
        if is_hand_near_products(
            hand,
            product_polygons,
            near_expand_ratio=near_expand_ratio,
            max_center_dist_ratio=max_center_dist_ratio,
        ):
            nearby.append(list(hand))
    return nearby


def select_handheld_geometry(
    anchors: List[List[List[float]]],
    hands: List[List[List[float]]],
    auxiliary: List[List[List[float]]],
    detections: List[Detection],
    *,
    width: int,
    height: int,
    min_iou: float,
    min_overlap: float,
    hand_expand_ratio: float,
    include_hand_matched: bool,
    max_area_ratio: float = DEFAULT_MAX_MATCH_AREA_RATIO,
    include_nearby_hands: bool = DEFAULT_INCLUDE_NEARBY_HANDS,
    hand_near_expand_ratio: float = DEFAULT_HAND_NEAR_EXPAND_RATIO,
    hand_max_center_dist_ratio: float = DEFAULT_HAND_MAX_CENTER_DIST_RATIO,
) -> Tuple[List[ProductUnit], Set[int], int]:
    """
    返回商品单元列表、选中的检测索引、未匹配锚点数。

    以 JSON 商品锚点为主：每个锚点对应一个商品单元；YOLO 仅可匹配细化边界。
    仅当没有任何 JSON 商品锚点时，才允许用手旁 YOLO 框回退生成商品单元（并去重）。
    遮挡等辅助区域挂到最近的商品单元。手不在此并入，改由面积策略决定。
    """
    del include_nearby_hands, hand_near_expand_ratio, hand_max_center_dist_ratio
    product_units: List[ProductUnit] = []
    selected_detection_indices: Set[int] = set()
    unmatched_anchor_count = 0

    for anchor in anchors:
        unit: ProductUnit = [list(anchor)]
        best_index: Optional[int] = None
        best_score = -1.0
        for index, detection in enumerate(detections):
            match = compare_polygons(
                anchor,
                detection.points,
                min_iou=min_iou,
                min_overlap=min_overlap,
                max_area_ratio=max_area_ratio,
            )
            if not match.matched:
                continue
            candidate_score = match.score + detection.score * 0.01
            if candidate_score > best_score:
                best_score = candidate_score
                best_index = index
        if best_index is None:
            unmatched_anchor_count += 1
        else:
            selected_detection_indices.add(best_index)
            unit.append(detections[best_index].points)
        product_units.append(unit)

    def detection_covered_by_existing_units(detection_points: Sequence[Sequence[float]]) -> bool:
        """与已有商品单元任一多边形已匹配，则视为同一商品，不再新建单元。"""
        for unit in product_units:
            for poly in unit:
                match = compare_polygons(
                    poly,
                    detection_points,
                    min_iou=min_iou,
                    min_overlap=min_overlap,
                    max_area_ratio=max_area_ratio,
                )
                if match.matched:
                    return True
        return False

    def detection_overlaps_kept(
        detection_points: Sequence[Sequence[float]], kept_indices: Sequence[int]
    ) -> bool:
        for kept in kept_indices:
            match = compare_polygons(
                detections[kept].points,
                detection_points,
                min_iou=min_iou,
                min_overlap=min_overlap,
                max_area_ratio=max_area_ratio,
            )
            if match.matched:
                return True
        return False

    # JSON 已有商品锚点：只吸收与现有单元重叠的 YOLO（便于边界细化统计），不另开商品。
    if anchors:
        for index, detection in enumerate(detections):
            if index in selected_detection_indices:
                continue
            if detection_covered_by_existing_units(detection.points):
                selected_detection_indices.add(index)
    # 无 JSON 商品锚点时，才用手旁 YOLO 回退；重叠检测去重，避免多检抬高商品数。
    elif include_hand_matched and hands:
        expanded_hands = [
            expanded_axis_aligned_polygon(hand, hand_expand_ratio, width, height)
            for hand in hands
        ]
        hand_candidates: List[int] = []
        for index, detection in enumerate(detections):
            for hand in expanded_hands:
                match = compare_polygons(
                    hand,
                    detection.points,
                    min_iou=min_iou,
                    min_overlap=min_overlap,
                    max_area_ratio=max_area_ratio,
                )
                if match.matched:
                    hand_candidates.append(index)
                    break
        hand_candidates.sort(
            key=lambda idx: float(detections[idx].score), reverse=True
        )
        kept_hand_indices: List[int] = []
        for index in hand_candidates:
            if detection_overlaps_kept(detections[index].points, kept_hand_indices):
                selected_detection_indices.add(index)
                continue
            kept_hand_indices.append(index)
            selected_detection_indices.add(index)
            product_units.append([detections[index].points])

    if product_units and auxiliary:
        for aux in auxiliary:
            aux_center = polygons_center([aux])
            if aux_center is None:
                continue
            best_unit = 0
            best_dist = float("inf")
            for unit_index, unit in enumerate(product_units):
                center = polygons_center(unit)
                if center is None:
                    continue
                dist = float(
                    np.hypot(aux_center[0] - center[0], aux_center[1] - center[1])
                )
                if dist < best_dist:
                    best_dist = dist
                    best_unit = unit_index
            product_units[best_unit].append(list(aux))

    return product_units, selected_detection_indices, unmatched_anchor_count


def flatten_product_units(
    product_units: Sequence[Sequence[Sequence[Sequence[float]]]],
) -> List[List[List[float]]]:
    polygons: List[List[List[float]]] = []
    for unit in product_units:
        polygons.extend(list(unit))
    return polygons


def polygons_center(
    polygons: Sequence[Sequence[Sequence[float]]],
) -> Optional[Tuple[float, float]]:
    bounds = polygons_axis_aligned_bounds(polygons)
    if bounds is None:
        return None
    x1, y1, x2, y2 = bounds
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def effective_rectangle_margin(
    polygons: Sequence[Sequence[Sequence[float]]],
    *,
    margin: float,
    margin_ratio: float,
) -> float:
    """取固定像素与相对尺寸外扩的较大值，避免框线贴住商品。"""
    base = max(0.0, float(margin))
    ratio = max(0.0, float(margin_ratio))
    if ratio <= 0:
        return base
    bounds = polygons_axis_aligned_bounds(polygons)
    if bounds is None:
        return base
    x1, y1, x2, y2 = bounds
    long_side = max(x2 - x1, y2 - y1, 1.0)
    return max(base, ratio * long_side)


def format_rect_label(rect_label: str, product_count: int) -> str:
    """最小外接矩形 label = 基础名 + '_' + 框内手持商品数量。"""
    return f"{rect_label}_{max(0, int(product_count))}"


def is_rect_label(label: str, rect_label: str) -> bool:
    """匹配基础名或带数量后缀的最小外接矩形 label（如 最小外接矩形_2）。"""
    text = str(label).strip()
    base = str(rect_label).strip()
    if not base:
        return False
    if text == base:
        return True
    prefix = f"{base}_"
    if not text.startswith(prefix):
        return False
    suffix = text[len(prefix) :]
    return suffix.isdigit()


# 可视化用英文短名；「最小外接矩形」保留后缀，如 最小外接矩形_2 → min_rect_2
VIZ_LABEL_ALIASES: Dict[str, str] = {
    "罐装": "can",
    "瓶装": "bottle",
    "袋装": "bag",
    "盒装": "box",
    "桶装": "bucket",
    "条装": "stick",
    "手": "hand",
    "头": "head",
    "头部": "head",
    "手机": "phone",
    "售货柜": "cabinet",
    "最小外接矩形": "min_rect",
}


def viz_label_text(label: str, aliases: Optional[Dict[str, str]] = None) -> str:
    """
    将 JSON label 转为可视化短名。
    最小外接矩形 / 最小外接矩形_N → min_rect / min_rect_N。
    """
    text = str(label).strip()
    mapping = aliases if aliases is not None else VIZ_LABEL_ALIASES
    rect_base = "最小外接矩形"
    rect_alias = mapping.get(rect_base, "min_rect")
    if text == rect_base or is_rect_label(text, rect_base):
        return rect_alias + text[len(rect_base) :]
    if text in mapping:
        return mapping[text]
    # 前缀匹配商品类（如 瓶装_SKU）；较长键优先，避免「手机」误匹配「手」
    for key, value in sorted(mapping.items(), key=lambda kv: len(kv[0]), reverse=True):
        if key == rect_base:
            continue
        if text.startswith(key):
            return value
    return text


def rectangle_area(points: Sequence[Sequence[float]]) -> float:
    """旋转矩形四点面积（contourArea）。"""
    array = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(array) < 3:
        return 0.0
    return abs(float(cv2.contourArea(array)))


def rectangle_aabb_size(points: Sequence[Sequence[float]]) -> Tuple[float, float]:
    """外接矩形四点的轴对齐宽、高。"""
    array = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(array) == 0:
        return 0.0, 0.0
    width = float(array[:, 0].max() - array[:, 0].min())
    height = float(array[:, 1].max() - array[:, 1].min())
    return width, height


def rectangle_obb_size(points: Sequence[Sequence[float]]) -> Tuple[float, float]:
    """旋转外接矩形自身的两边长（与轴向无关）。"""
    array = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(array) < 3:
        return 0.0, 0.0
    _center, size, _angle = cv2.minAreaRect(array)
    return float(size[0]), float(size[1])


def rect_passes_side_limit(
    points: Sequence[Sequence[float]],
    *,
    image_width: int,
    image_height: int,
    max_side_ratio: float,
) -> bool:
    """
    含边距后的外接矩形按 OBB 真实边长判定：
    较长边 ≤ 原图较长边 * ratio，且较短边 ≤ 原图较短边 * ratio。
    """
    side_a, side_b = rectangle_obb_size(points)
    rect_long = max(side_a, side_b)
    rect_short = min(side_a, side_b)
    image_long = float(max(1, image_width, image_height))
    image_short = float(min(max(1, image_width), max(1, image_height)))
    ratio = max(0.0, float(max_side_ratio))
    return (
        rect_long <= image_long * ratio + 1e-6
        and rect_short <= image_short * ratio + 1e-6
    )


def hand_directly_overlaps_products(
    hand: Sequence[Sequence[float]],
    product_polygons: Sequence[Sequence[Sequence[float]]],
) -> bool:
    """手框是否与任一商品多边形真实相交（排除 AABB 假重叠）。"""
    if not hand:
        return False
    for poly in product_polygons:
        if poly and polygons_intersect(hand, poly):
            return True
    return False


def filter_hands_overlapping_products(
    hands: Sequence[Sequence[Sequence[float]]],
    product_polygons: Sequence[Sequence[Sequence[float]]],
) -> List[List[List[float]]]:
    """只保留与给定商品集合真实多边形相交的手。"""
    return [
        list(hand)
        for hand in hands
        if hand and hand_directly_overlaps_products(hand, product_polygons)
    ]


def cluster_product_units_by_aabb_overlap(
    product_units: Sequence[ProductUnit],
    hand_polygons: Sequence[Sequence[Sequence[float]]],
) -> List[Dict[str, Any]]:
    """
    商品单元按 AABB 重叠做连通分量聚类（仅商品↔商品，传递性）。
    手不参与商品聚类、不通过手桥接商品；仅当手与该类内某商品多边形真实相交时纳入该类。
    返回 [{'units': [...], 'hands': [...]}]；与任何商品都不重叠的手丢弃。
    """
    units = [list(unit) for unit in product_units if unit]
    hands = [list(hand) for hand in hand_polygons if hand]
    if not units:
        return []

    unit_bounds = [polygons_axis_aligned_bounds(unit) for unit in units]
    n_units = len(units)
    parent = list(range(n_units))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_l = find(left)
        root_r = find(right)
        if root_l != root_r:
            parent[root_r] = root_l

    for i in range(n_units):
        if unit_bounds[i] is None:
            continue
        for j in range(i + 1, n_units):
            if unit_bounds[j] is None:
                continue
            if aabb_intersects(unit_bounds[i], unit_bounds[j]):
                union(i, j)

    grouped: Dict[int, Dict[str, Any]] = {}
    for index in range(n_units):
        root = find(index)
        bucket = grouped.setdefault(root, {"units": [], "hands": []})
        bucket["units"].append(units[index])

    for hand in hands:
        attached_roots: Set[int] = set()
        for index, unit in enumerate(units):
            # 与该商品单元内任一多边形真实相交才挂上
            if hand_directly_overlaps_products(hand, unit):
                attached_roots.add(find(index))
        for root in attached_roots:
            grouped[root]["hands"].append(hand)

    return [bucket for bucket in grouped.values() if bucket["units"]]


def cluster_boxes_by_aabb_overlap(
    product_polygons: Sequence[Sequence[Sequence[float]]],
    hand_polygons: Sequence[Sequence[Sequence[float]]],
) -> List[Dict[str, List[List[List[float]]]]]:
    """测试/兼容：每个多边形视为一个商品单元。"""
    units: List[ProductUnit] = [[list(poly)] for poly in product_polygons if poly]
    clustered = cluster_product_units_by_aabb_overlap(units, hand_polygons)
    return [
        {
            "products": flatten_product_units(item["units"]),
            "hands": item["hands"],
        }
        for item in clustered
    ]


def build_sized_min_rectangles(
    product_units: Sequence[ProductUnit],
    hand_polygons: Sequence[Sequence[Sequence[float]]],
    *,
    width: int,
    height: int,
    mode: str,
    margin: float,
    margin_ratio: float = 0.0,
    max_side_ratio: float = DEFAULT_MAX_SIDE_RATIO,
    max_rect_area_ratio: Optional[float] = None,
    include_contact_hands: bool = True,
    hand_near_expand_ratio: float = DEFAULT_HAND_NEAR_EXPAND_RATIO,
    hand_max_center_dist_ratio: float = DEFAULT_HAND_MAX_CENTER_DIST_RATIO,
) -> List[Tuple[List[List[float]], str, int]]:
    """
    重叠连通分量聚类后，对每一类生成最小外接矩形（一律含边距）。

    2.1 框住该类全部商品 + 与这些商品多边形真实相交的手；OBB 达标则通过
    2.2 I 去手，仅该类全部商品；仍超则
    2.2 II 该类每个商品单元单独一框：
         - 先商品+相交手；超限则对该单品去手再算
         - 仅商品仍超 → 例外取消大小限制（例外框不含手）

    返回 [(矩形四点, 策略名, 框内手持商品单元数), ...]
    """
    del hand_near_expand_ratio, hand_max_center_dist_ratio
    if max_rect_area_ratio is not None:
        max_side_ratio = float(max_rect_area_ratio)
    units = [list(unit) for unit in product_units if unit]
    if not units:
        raise ValueError("没有可用于计算最小外接矩形的商品点")

    hands = list(hand_polygons) if include_contact_hands else []
    clusters = cluster_product_units_by_aabb_overlap(units, hands)
    if not clusters:
        raise ValueError("没有可用于计算最小外接矩形的商品点")

    results: List[Tuple[List[List[float]], str, int]] = []
    for cluster in clusters:
        cluster_units: List[ProductUnit] = cluster["units"]
        cluster_products = flatten_product_units(cluster_units)
        n_products = len(cluster_units)
        # 最终纳入的手：必须与本框商品直接重叠（聚类阶段已按此过滤，此处再滤一次兜底）
        contact_hands = filter_hands_overlapping_products(
            cluster.get("hands") or [], cluster_products
        )

        all_polys = list(cluster_products) + list(contact_hands)
        rect_all = min_enclosing_rectangle(
            all_polys,
            width=width,
            height=height,
            mode=mode,
            margin=margin,
            margin_ratio=margin_ratio,
        )
        if rect_passes_side_limit(
            rect_all,
            image_width=width,
            image_height=height,
            max_side_ratio=max_side_ratio,
        ):
            policy = "cluster_all" if contact_hands else "cluster_products"
            results.append((rect_all, policy, n_products))
            continue

        rect_products = min_enclosing_rectangle(
            cluster_products,
            width=width,
            height=height,
            mode=mode,
            margin=margin,
            margin_ratio=margin_ratio,
        )
        if rect_passes_side_limit(
            rect_products,
            image_width=width,
            image_height=height,
            max_side_ratio=max_side_ratio,
        ):
            results.append((rect_products, "cluster_products", n_products))
            continue

        for unit in cluster_units:
            # 单品框：先商品+相交手；超限则去手；仅商品仍超才例外（例外不含手）
            unit_hands = filter_hands_overlapping_products(contact_hands, unit)
            rect_with_hands = min_enclosing_rectangle(
                list(unit) + list(unit_hands),
                width=width,
                height=height,
                mode=mode,
                margin=margin,
                margin_ratio=margin_ratio,
            )
            if unit_hands and rect_passes_side_limit(
                rect_with_hands,
                image_width=width,
                image_height=height,
                max_side_ratio=max_side_ratio,
            ):
                results.append((rect_with_hands, "single_product", 1))
                continue

            rect_product_only = min_enclosing_rectangle(
                list(unit),
                width=width,
                height=height,
                mode=mode,
                margin=margin,
                margin_ratio=margin_ratio,
            )
            if rect_passes_side_limit(
                rect_product_only,
                image_width=width,
                image_height=height,
                max_side_ratio=max_side_ratio,
            ):
                policy = (
                    "single_product_drop_hand"
                    if unit_hands
                    else "single_product"
                )
                results.append((rect_product_only, policy, 1))
            else:
                results.append(
                    (rect_product_only, "single_product_exception", 1)
                )
    return results


def build_sized_min_rectangle(
    product_polygons: Sequence[Sequence[Sequence[float]]],
    hand_polygons: Sequence[Sequence[Sequence[float]]],
    *,
    width: int,
    height: int,
    mode: str,
    margin: float,
    margin_ratio: float = 0.0,
    max_side_ratio: float = DEFAULT_MAX_SIDE_RATIO,
    max_rect_area_ratio: Optional[float] = None,
    include_contact_hands: bool = True,
    hand_near_expand_ratio: float = DEFAULT_HAND_NEAR_EXPAND_RATIO,
    hand_max_center_dist_ratio: float = DEFAULT_HAND_MAX_CENTER_DIST_RATIO,
) -> Tuple[List[List[float]], str]:
    """兼容接口：每个多边形视为独立商品框，返回第一个结果框（不含数量）。"""
    units: List[ProductUnit] = [[list(poly)] for poly in product_polygons if poly]
    results = build_sized_min_rectangles(
        units,
        hand_polygons,
        width=width,
        height=height,
        mode=mode,
        margin=margin,
        margin_ratio=margin_ratio,
        max_side_ratio=max_side_ratio,
        max_rect_area_ratio=max_rect_area_ratio,
        include_contact_hands=include_contact_hands,
        hand_near_expand_ratio=hand_near_expand_ratio,
        hand_max_center_dist_ratio=hand_max_center_dist_ratio,
    )
    rect, policy, _count = results[0]
    return rect, policy


def detect_out_of_bounds_borders(
    points: Sequence[Sequence[float]],
    width: int,
    height: int,
    *,
    eps: float = 1e-3,
) -> Set[str]:
    """返回扩边后矩形触碰/超出的图像边界：left/right/top/bottom。"""
    max_x = float(width - 1)
    max_y = float(height - 1)
    touched: Set[str] = set()
    for point in points:
        x, y = float(point[0]), float(point[1])
        if x < -eps:
            touched.add("left")
        if x > max_x + eps:
            touched.add("right")
        if y < -eps:
            touched.add("top")
        if y > max_y + eps:
            touched.add("bottom")
    return touched


def axis_aligned_rect_with_border_constraints(
    content_points: np.ndarray,
    *,
    width: int,
    height: int,
    margin: float,
    locked_borders: Set[str],
) -> List[List[float]]:
    """
    以内容点 AABB 重建轴对齐外接矩形：
    已锁定的图像边界作为矩形边（该侧不加 margin）；其余侧保留 margin。
    若重建后仍越界，继续锁定新越界侧。
    """
    max_x = float(max(0, width - 1))
    max_y = float(max(0, height - 1))
    m = max(0.0, float(margin))
    locked = set(locked_borders)
    min_xy = content_points.min(axis=0)
    max_xy = content_points.max(axis=0)
    c_min_x, c_min_y = float(min_xy[0]), float(min_xy[1])
    c_max_x, c_max_y = float(max_xy[0]), float(max_xy[1])

    for _ in range(4):
        x1 = 0.0 if "left" in locked else c_min_x - m
        y1 = 0.0 if "top" in locked else c_min_y - m
        x2 = max_x if "right" in locked else c_max_x + m
        y2 = max_y if "bottom" in locked else c_max_y + m
        # 保证覆盖内容且不倒置
        x1 = min(x1, c_min_x)
        y1 = min(y1, c_min_y)
        x2 = max(x2, c_max_x)
        y2 = max(y2, c_max_y)
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        extra = detect_out_of_bounds_borders(pts, width, height)
        if not extra - locked:
            # 贴边侧夹到精确边界；未锁侧若数值略出界也夹回（不应发生）
            x1 = 0.0 if "left" in locked else clamp(x1, 0.0, max_x)
            y1 = 0.0 if "top" in locked else clamp(y1, 0.0, max_y)
            x2 = max_x if "right" in locked else clamp(x2, 0.0, max_x)
            y2 = max_y if "bottom" in locked else clamp(y2, 0.0, max_y)
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        locked |= extra
    return clamp_points([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], width, height)


def min_enclosing_rectangle(
    polygons: Iterable[Sequence[Sequence[float]]],
    *,
    width: int,
    height: int,
    mode: str,
    margin: float,
    margin_ratio: float = 0.0,
) -> List[List[float]]:
    arrays = [np.asarray(polygon, dtype=np.float32).reshape(-1, 2) for polygon in polygons]
    arrays = [array for array in arrays if len(array) > 0]
    if not arrays:
        raise ValueError("没有可用于计算最小外接矩形的商品点")
    polygon_list = [array.tolist() for array in arrays]
    all_points = np.concatenate(arrays, axis=0)
    applied_margin = effective_rectangle_margin(
        polygon_list, margin=margin, margin_ratio=margin_ratio
    )

    if mode == "axis_aligned":
        min_xy = all_points.min(axis=0)
        max_xy = all_points.max(axis=0)
        x1 = float(min_xy[0]) - applied_margin
        y1 = float(min_xy[1]) - applied_margin
        x2 = float(max_xy[0]) + applied_margin
        y2 = float(max_xy[1]) + applied_margin
        points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    elif mode == "min_area":
        center, size, angle = cv2.minAreaRect(all_points)
        expanded_size = (
            max(1e-3, float(size[0]) + 2.0 * applied_margin),
            max(1e-3, float(size[1]) + 2.0 * applied_margin),
        )
        points = cv2.boxPoints((center, expanded_size, angle)).tolist()
    else:
        raise ValueError(f"未知 rectangle_mode：{mode}")

    touched = detect_out_of_bounds_borders(points, width, height)
    if touched:
        points = axis_aligned_rect_with_border_constraints(
            all_points,
            width=width,
            height=height,
            margin=applied_margin,
            locked_borders=touched,
        )
    return canonicalize_rectangle(points)


def canonicalize_rectangle(points: Sequence[Sequence[float]]) -> List[List[float]]:
    """将四点按顺时针排列，并从 y 最小、再 x 最小的点开始。"""
    array = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    center = array.mean(axis=0)
    angles = np.arctan2(array[:, 1] - center[1], array[:, 0] - center[0])
    ordered = array[np.argsort(angles)]
    # 图像坐标中正面积对应顺时针；若方向相反则翻转。
    signed_twice_area = 0.0
    for index in range(len(ordered)):
        x1, y1 = ordered[index]
        x2, y2 = ordered[(index + 1) % len(ordered)]
        signed_twice_area += x1 * y2 - x2 * y1
    if signed_twice_area < 0:
        ordered = ordered[::-1]
    start = min(range(len(ordered)), key=lambda i: (ordered[i][1], ordered[i][0]))
    ordered = np.concatenate([ordered[start:], ordered[:start]], axis=0)
    return [[round(float(x), 3), round(float(y), 3)] for x, y in ordered]


def rectangle_direction(points: Sequence[Sequence[float]]) -> float:
    if len(points) < 2:
        return 0.0
    dx = float(points[1][0]) - float(points[0][0])
    dy = float(points[1][1]) - float(points[0][1])
    return float(math.atan2(dy, dx) % (2.0 * math.pi))


def neutral_value_like(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return False
    if isinstance(value, str):
        return ""
    if isinstance(value, (int, float)):
        return 0
    if isinstance(value, list):
        return []
    if isinstance(value, dict):
        return {}
    return None


def choose_shape_template(
    shapes: List[Any], product_labels: Set[str], ignore_labels: Set[str]
) -> Optional[Dict[str, Any]]:
    for shape in shapes:
        if (
            isinstance(shape, dict)
            and "label" in shape
            and "points" in shape
            and (
                (product_labels and label_matches(str(shape.get("label", "")), product_labels))
                or (not product_labels and str(shape.get("label")) not in ignore_labels)
            )
        ):
            return shape
    for shape in shapes:
        if isinstance(shape, dict) and "label" in shape and "points" in shape:
            return shape
    return None


def make_rectangle_shape(
    points: List[List[float]],
    label: str,
    template: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    direction = rectangle_direction(points)
    if template is None:
        return {
            "kie_linking": [],
            "score": None,
            "direction": direction,
            "label": label,
            "points": points,
            "group_id": None,
            "description": "",
            "difficult": False,
            "shape_type": "rotation",
            "flags": {},
            "attributes": {},
        }

    shape: Dict[str, Any] = {}
    for key, old_value in template.items():
        if key == "label":
            shape[key] = label
        elif key == "points":
            shape[key] = points
        elif key == "score":
            shape[key] = None
        elif key == "direction":
            shape[key] = direction
        elif key == "group_id":
            shape[key] = None
        elif key == "description":
            shape[key] = ""
        elif key == "difficult":
            shape[key] = False
        elif key == "shape_type":
            shape[key] = "rotation"
        elif key in {"flags", "attributes"}:
            shape[key] = {}
        elif key == "kie_linking":
            shape[key] = []
        else:
            shape[key] = neutral_value_like(old_value)
    if "label" not in shape:
        shape["label"] = label
    if "points" not in shape:
        shape["points"] = points
    return shape


def _skip_whitespace(text: str, index: int) -> int:
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def find_top_level_shapes_span(text: str) -> Tuple[int, int, int]:
    """返回 shapes 键起点、数组 '[' 位置、数组 ']' 位置。"""
    decoder = json.JSONDecoder()
    index = _skip_whitespace(text, 0)
    if index >= len(text) or text[index] != "{":
        raise ValueError("JSON 顶层不是对象")
    index += 1
    while True:
        index = _skip_whitespace(text, index)
        if index >= len(text) or text[index] == "}":
            break
        if text[index] == ",":
            index = _skip_whitespace(text, index + 1)
        key_start = index
        key, key_end = decoder.raw_decode(text, index)
        if not isinstance(key, str):
            raise ValueError("JSON 对象键不是字符串")
        index = _skip_whitespace(text, key_end)
        if index >= len(text) or text[index] != ":":
            raise ValueError(f"JSON 键 {key!r} 后缺少冒号")
        value_start = _skip_whitespace(text, index + 1)
        value, value_end = decoder.raw_decode(text, value_start)
        if key == "shapes":
            if not isinstance(value, list) or text[value_start] != "[":
                raise ValueError("shapes 必须是数组")
            return key_start, value_start, value_end - 1
        index = value_end
    raise ValueError("JSON 中没有 shapes 字段")


def _line_indent(text: str, position: int) -> str:
    line_start = max(text.rfind("\n", 0, position), text.rfind("\r", 0, position)) + 1
    prefix = text[line_start:position]
    return prefix if not prefix.strip() else ""


def detect_indentation(
    text: str, key_start: int, array_start: int, array_end: int
) -> Tuple[str, str]:
    key_indent = _line_indent(text, key_start)
    close_indent = _line_indent(text, array_end) or key_indent
    content = text[array_start + 1 : array_end]
    first_offset = 0
    while first_offset < len(content) and content[first_offset].isspace():
        first_offset += 1
    if first_offset < len(content):
        first_position = array_start + 1 + first_offset
        item_indent = _line_indent(text, first_position)
    else:
        item_indent = ""
    if item_indent.startswith(close_indent) and len(item_indent) > len(close_indent):
        indent_unit = item_indent[len(close_indent) :]
    else:
        indent_unit = "  "
        item_indent = close_indent + indent_unit
    return item_indent, indent_unit


def append_shapes_preserving_original_text(
    original_text: str, new_shapes: List[Dict[str, Any]]
) -> str:
    if not new_shapes:
        return original_text
    key_start, array_start, array_end = find_top_level_shapes_span(original_text)
    item_indent, indent_unit = detect_indentation(
        original_text, key_start, array_start, array_end
    )
    newline = "\r\n" if "\r\n" in original_text else "\n"
    rendered_items: List[str] = []
    for shape in new_shapes:
        rendered = json.dumps(shape, ensure_ascii=False, indent=indent_unit)
        rendered = rendered.replace("\n", newline)
        rendered_items.append(item_indent + rendered.replace(newline, newline + item_indent))
    rendered_block = ("," + newline).join(rendered_items)

    content = original_text[array_start + 1 : array_end]
    if content.strip():
        insertion_index = array_end
        while (
            insertion_index > array_start + 1
            and original_text[insertion_index - 1].isspace()
        ):
            insertion_index -= 1
        insertion = "," + newline + rendered_block
    else:
        insertion_index = array_start + 1
        close_indent = _line_indent(original_text, array_end)
        insertion = newline + rendered_block
        if not content:
            insertion += newline + close_indent
    return original_text[:insertion_index] + insertion + original_text[insertion_index:]


def verify_only_shapes_appended(
    original_data: Dict[str, Any], updated_text: str, new_shapes: List[Dict[str, Any]]
) -> None:
    updated_data = json.loads(updated_text)
    original_shapes = original_data.get("shapes")
    updated_shapes = updated_data.get("shapes")
    if not isinstance(original_shapes, list) or not isinstance(updated_shapes, list):
        raise AssertionError("shapes 不是数组")
    original_without_shapes = {
        key: value for key, value in original_data.items() if key != "shapes"
    }
    updated_without_shapes = {
        key: value for key, value in updated_data.items() if key != "shapes"
    }
    if updated_without_shapes != original_without_shapes:
        raise AssertionError("发现 shapes 之外的 JSON 内容发生变化")
    if updated_shapes[: len(original_shapes)] != original_shapes:
        raise AssertionError("原 shapes 内容或顺序发生变化")
    if updated_shapes[len(original_shapes) :] != new_shapes:
        raise AssertionError("新增 shape 与预期不一致")


def atomic_write_text(path: Path, text: str) -> None:
    temp_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=str(path.parent),
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_file.write(text)
            temp_file.flush()
            os.fsync(temp_file.fileno())
            temp_name = temp_file.name
        os.replace(temp_name, path)
    finally:
        if temp_name is not None and os.path.exists(temp_name):
            os.unlink(temp_name)


def safe_extract_zip(zip_path: Path, extract_dir: Path) -> None:
    destination = extract_dir.resolve()
    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            try:
                target.relative_to(destination)
            except ValueError as exc:
                raise ValueError(f"zip 包含不安全路径：{member.filename}") from exc
        archive.extractall(destination)


def prepare_input_root(args: argparse.Namespace) -> Path:
    if args.input_zip:
        zip_path = Path(args.input_zip).expanduser().resolve()
        if not zip_path.is_file():
            raise FileNotFoundError(f"input_zip 不存在：{zip_path}")
        work_dir = Path(args.work_dir).expanduser().resolve()
        extract_dir = work_dir / "extracted"
        if work_dir.exists():
            if args.overwrite:
                shutil.rmtree(work_dir)
            else:
                raise FileExistsError(
                    f"work_dir 已存在：{work_dir}；如需覆盖请加 --overwrite"
                )
        extract_dir.mkdir(parents=True, exist_ok=True)
        safe_extract_zip(zip_path, extract_dir)
        return extract_dir
    root = Path(args.dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"dataset_root 不存在：{root}")
    return root


def prepare_process_root(input_root: Path, args: argparse.Namespace) -> Path:
    if not args.output_root:
        return input_root
    output_root = Path(args.output_root).expanduser().resolve()
    if output_root == input_root:
        return input_root
    try:
        output_root.relative_to(input_root)
    except ValueError:
        pass
    else:
        raise ValueError(
            "output_root 不能放在 dataset_root 内部，否则复制数据集时会形成递归目录："
            f"input={input_root}, output={output_root}"
        )
    if output_root.exists():
        if args.resume:
            print(f"[INFO] resume：继续使用已有输出目录 {output_root}")
            return output_root
        if args.overwrite:
            shutil.rmtree(output_root)
        else:
            raise FileExistsError(
                f"output_root 已存在：{output_root}；继续处理加 --resume，覆盖加 --overwrite"
            )
    print(f"[INFO] 正在完整复制数据集到 {output_root} ...")
    shutil.copytree(input_root, output_root)
    print("[INFO] 数据集复制完成；输入目录不会被修改。")
    return output_root


def find_image_json_pairs(
    root: Path,
    image_dir_name: str,
    json_dir_name: str,
    image_extensions: Set[str],
) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    image_dirs = sorted(
        path for path in root.rglob(image_dir_name) if path.is_dir() and path.name == image_dir_name
    )
    for image_dir in image_dirs:
        json_dir = image_dir.parent / json_dir_name
        if not json_dir.is_dir():
            print(f"[WARN] 缺少对应 JSON 目录，跳过：{image_dir}", file=sys.stderr)
            continue
        for image_path in sorted(image_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in image_extensions:
                continue
            json_path = json_dir / f"{image_path.stem}.json"
            if not json_path.is_file():
                print(f"[WARN] 图片没有同名 JSON，跳过：{image_path}", file=sys.stderr)
                continue
            pairs.append((image_path, json_path))
    return pairs


def process_one(
    model: Any,
    image_path: Path,
    json_path: Path,
    args: argparse.Namespace,
    product_labels: Set[str],
    ignore_labels: Set[str],
    hand_labels: Set[str],
    auxiliary_labels: Set[str],
    product_classes: Optional[Set[int]],
    ignore_yolo_class_names: Set[str],
) -> ProcessResult:
    original_text, original_data = load_json_text(json_path)
    shapes = original_data.get("shapes")
    if not isinstance(shapes, list):
        raise ValueError(f"JSON 中没有合法 shapes 数组：{json_path}")
    existing_labels = {
        str(shape.get("label", "")) for shape in shapes if isinstance(shape, dict)
    }
    if args.skip_existing and any(
        is_rect_label(label, args.rect_label) for label in existing_labels
    ):
        return ProcessResult(0, True, 0, 0, 0)

    width, height = get_image_size_from_json_or_file(original_data, image_path)
    anchors, hands, auxiliary = collect_json_polygons(
        shapes,
        width,
        height,
        product_labels=product_labels,
        ignore_labels=ignore_labels,
        hand_labels=hand_labels,
        auxiliary_labels=auxiliary_labels,
        rect_label=args.rect_label,
    )
    if not anchors and not hands:
        raise ValueError(
            "JSON 中既没有商品锚点，也没有手部标注；为避免把货架商品误当成手拿商品，"
            "本文件未追加。请检查 --product_labels/--ignore_labels/--hand_labels。"
        )

    predict_kwargs: Dict[str, Any] = {
        "source": str(image_path),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "device": args.device,
        "verbose": False,
    }
    if product_classes is not None:
        predict_kwargs["classes"] = sorted(product_classes)
    results = model.predict(**predict_kwargs)
    if not results:
        raise RuntimeError("YOLO 没有返回推理结果")
    detections = filter_product_detections(
        detections_from_result(results[0], width, height),
        product_classes=product_classes,
        ignore_class_names=ignore_yolo_class_names,
    )

    product_units, selected_indices, unmatched_anchor_count = select_handheld_geometry(
        anchors,
        hands,
        auxiliary,
        detections,
        width=width,
        height=height,
        min_iou=args.min_match_iou,
        min_overlap=args.min_match_overlap,
        hand_expand_ratio=args.hand_expand_ratio,
        include_hand_matched=args.include_hand_matched_detections,
        max_area_ratio=float(args.max_match_area_ratio),
        include_nearby_hands=args.include_nearby_hands,
        hand_near_expand_ratio=float(args.hand_near_expand_ratio),
        hand_max_center_dist_ratio=float(args.hand_max_center_dist_ratio),
    )
    if args.require_yolo_match and not selected_indices:
        raise ValueError("商品 YOLO 没有检测到与 JSON 商品/手部匹配的框")
    if not product_units:
        if anchors and unmatched_anchor_count == len(anchors):
            product_units = [[list(anchor)] for anchor in anchors]
        else:
            raise ValueError("没有得到任何手拿商品范围")

    sized_rects = build_sized_min_rectangles(
        product_units,
        hands if args.include_nearby_hands else [],
        width=width,
        height=height,
        mode=args.rectangle_mode,
        margin=max(0.0, float(args.margin)),
        margin_ratio=max(0.0, float(args.margin_ratio)),
        max_side_ratio=float(args.max_side_ratio),
        include_contact_hands=bool(args.include_nearby_hands),
        hand_near_expand_ratio=float(args.hand_near_expand_ratio),
        hand_max_center_dist_ratio=float(args.hand_max_center_dist_ratio),
    )
    template_ignore = set(ignore_labels) | hand_labels | auxiliary_labels | {args.rect_label}
    template = choose_shape_template(shapes, product_labels, template_ignore)
    new_shapes = [
        make_rectangle_shape(
            rectangle, format_rect_label(args.rect_label, product_count), template
        )
        for rectangle, _policy, product_count in sized_rects
    ]

    if not args.dry_run:
        updated_text = append_shapes_preserving_original_text(original_text, new_shapes)
        verify_only_shapes_appended(original_data, updated_text, new_shapes)
        if args.backup_json and not args.output_root:
            backup_path = json_path.with_suffix(json_path.suffix + ".bak")
            if not backup_path.exists():
                shutil.copy2(json_path, backup_path)
        atomic_write_text(json_path, updated_text)

    return ProcessResult(
        rectangle_added=len(new_shapes),
        skipped_existing=False,
        anchor_count=len(anchors),
        yolo_detection_count=len(detections),
        selected_yolo_count=len(selected_indices),
    )


def make_zip_from_dir(source_dir: Path, output_zip: Path, overwrite: bool) -> None:
    output_zip = output_zip.expanduser().resolve()
    if output_zip.exists():
        if overwrite:
            output_zip.unlink()
        else:
            raise FileExistsError(f"output_zip 已存在：{output_zip}；覆盖请加 --overwrite")
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file() and path.resolve() != output_zip:
                archive.write(path, path.relative_to(source_dir))


def write_status_file(path: Optional[Path], payload: Dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    atomic_write_text(path, text)


def format_progress(
    index: int,
    total: int,
    elapsed: float,
    added: int,
    selected_yolo: int,
    skipped: int,
    failures: int,
) -> str:
    percent = index * 100.0 / total if total else 100.0
    rate = index / elapsed if elapsed > 0 else 0.0
    eta = (total - index) / rate if rate > 0 else 0.0
    width = 30
    filled = int(width * index / total) if total else width
    bar = "=" * filled + "-" * (width - filled)
    return (
        f"[{bar}] {index}/{total} ({percent:5.1f}%) rate={rate:5.1f} img/s "
        f"eta={eta/60:6.1f}m rect+={added} yolo_selected={selected_yolo} "
        f"skip={skipped} err={failures}"
    )


def validate_args(args: argparse.Namespace) -> None:
    if not args.rect_label.strip():
        raise ValueError("rect_label 不能为空")
    if args.imgsz <= 0:
        raise ValueError("imgsz 必须大于 0")
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("conf 必须在 [0,1]")
    if not 0.0 <= args.iou <= 1.0:
        raise ValueError("iou 必须在 [0,1]")
    if not 0.0 <= args.min_match_iou <= 1.0:
        raise ValueError("min_match_iou 必须在 [0,1]")
    if not 0.0 <= args.min_match_overlap <= 1.0:
        raise ValueError("min_match_overlap 必须在 [0,1]")
    if args.hand_expand_ratio < 0:
        raise ValueError("hand_expand_ratio 不能为负数")
    if args.hand_near_expand_ratio < 0:
        raise ValueError("hand_near_expand_ratio 不能为负数")
    if args.hand_max_center_dist_ratio < 0:
        raise ValueError("hand_max_center_dist_ratio 不能为负数")
    if float(args.margin) < 0:
        raise ValueError("margin 不能为负数")
    if float(args.margin_ratio) < 0:
        raise ValueError("margin_ratio 不能为负数")
    if getattr(args, "max_rect_area_ratio", None) is not None:
        args.max_side_ratio = float(args.max_rect_area_ratio)
    if float(args.max_side_ratio) <= 0:
        raise ValueError("max_side_ratio 必须大于 0")
    if args.max_match_area_ratio <= 0:
        raise ValueError("max_match_area_ratio 必须大于 0")
    weights = Path(args.weights).expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"YOLO 权重不存在：{weights}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    product_labels = parse_string_set(args.product_labels)
    ignore_labels = parse_string_set(args.ignore_labels) | {args.rect_label}
    hand_labels = parse_string_set(args.hand_labels)
    auxiliary_labels = parse_string_set(args.auxiliary_labels)
    ignore_labels |= hand_labels | auxiliary_labels
    product_class_list = parse_int_list(args.product_classes)
    product_classes = set(product_class_list) if product_class_list is not None else None
    ignore_yolo_class_names = parse_string_set(args.ignore_yolo_class_names)
    image_extensions = {
        extension.strip().lower()
        for extension in args.image_exts.split(",")
        if extension.strip()
    }

    input_root = prepare_input_root(args)
    process_root = prepare_process_root(input_root, args)
    pairs = find_image_json_pairs(
        process_root, args.image_dir_name, args.json_dir_name, image_extensions
    )
    if not pairs:
        raise RuntimeError(
            "没有找到 images/json_labels 图片与同名 JSON 配对。"
            f"root={process_root}"
        )

    print(f"[INFO] process_root: {process_root}")
    print(f"[INFO] image/json pairs: {len(pairs)}")
    print(f"[INFO] product_labels: {sorted(product_labels) if product_labels else '自动'}")
    print(f"[INFO] ignore_labels: {sorted(ignore_labels)}")
    print(f"[INFO] auxiliary_labels: {sorted(auxiliary_labels)}")
    print(
        f"[INFO] product_classes: "
        f"{sorted(product_classes) if product_classes is not None else '未限制 id'}"
    )
    print(f"[INFO] ignore_yolo_class_names: {sorted(ignore_yolo_class_names)}")
    print(f"[INFO] max_match_area_ratio: {args.max_match_area_ratio}")
    print(f"[INFO] max_side_ratio: {args.max_side_ratio}")
    print(
        f"[INFO] include_nearby_hands={args.include_nearby_hands}, "
        f"hand_near_expand_ratio={args.hand_near_expand_ratio}, "
        f"hand_max_center_dist_ratio={args.hand_max_center_dist_ratio}"
    )
    print(
        f"[INFO] conf={args.conf}, iou={args.iou}, imgsz={args.imgsz}, "
        f"device={args.device}, rectangle_mode={args.rectangle_mode}, "
        f"margin={args.margin}, margin_ratio={args.margin_ratio}"
    )
    model = load_yolo_model(args.weights)

    added = 0
    selected_yolo_total = 0
    skipped_existing = 0
    failures: List[str] = []
    started_at = time.time()
    progress_every = max(1, args.progress_every)
    status_path = Path(args.status_file).expanduser().resolve() if args.status_file else None

    for index, (image_path, json_path) in enumerate(pairs, 1):
        try:
            result = process_one(
                model,
                image_path,
                json_path,
                args,
                product_labels,
                ignore_labels,
                hand_labels,
                auxiliary_labels,
                product_classes,
                ignore_yolo_class_names,
            )
            added += result.rectangle_added
            selected_yolo_total += result.selected_yolo_count
            if result.skipped_existing:
                skipped_existing += 1
        except Exception as exc:
            message = (
                f"image={image_path}, json={json_path}, "
                f"error={type(exc).__name__}: {exc}"
            )
            failures.append(message)
            print(f"[ERROR] {message}", file=sys.stderr)

        elapsed = time.time() - started_at
        if index == 1 or index % progress_every == 0 or index == len(pairs):
            print(
                format_progress(
                    index,
                    len(pairs),
                    elapsed,
                    added,
                    selected_yolo_total,
                    skipped_existing,
                    len(failures),
                ),
                flush=True,
            )
            write_status_file(
                status_path,
                {
                    "done": index,
                    "total": len(pairs),
                    "percent": round(index * 100.0 / len(pairs), 2),
                    "elapsed_s": round(elapsed, 1),
                    "rectangles_added": added,
                    "selected_yolo_detections": selected_yolo_total,
                    "skipped_existing": skipped_existing,
                    "errors": len(failures),
                    "finished": index == len(pairs),
                    "last_json": str(json_path),
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            )

    if failures:
        print("[ERROR] 存在失败项，不生成 output_zip。", file=sys.stderr)
        for failure in failures[:20]:
            print(f"  - {failure}", file=sys.stderr)
    elif args.output_zip:
        make_zip_from_dir(process_root, Path(args.output_zip), args.overwrite)
        print(f"[INFO] output_zip: {Path(args.output_zip).expanduser().resolve()}")

    print("=" * 80)
    print(
        f"完成。配对数={len(pairs)}；新增最小外接矩形={added}；"
        f"纳入 YOLO 商品框={selected_yolo_total}；跳过已有={skipped_existing}；"
        f"失败={len(failures)}"
    )
    if args.dry_run:
        print("dry-run：没有写入 JSON。")
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
