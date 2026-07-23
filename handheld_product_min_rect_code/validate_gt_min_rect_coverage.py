#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
校验：同一份结果 JSON 中，手持商品是否被某一个「最小外接矩形」按面积覆盖。

适用场景：有真值 JSON 时，流水线只把最小外接矩形追加到真值中，因此商品与
最小外接矩形在同一文件；不再区分两套标注 JSON。

判定（方案 A）：
  coverage = 交面积 / 手持商品面积
  对所有最小外接矩形取 max(coverage)；若 < 阈值（默认 0.9）则判为未覆盖并上报。

目录约定：
  result_root/
    images/
    json_labels/   # 真值 shapes + 追加的最小外接矩形
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import cv2

from add_handheld_product_min_rect import (
    as_convex_polygon,
    is_rect_label,
    label_matches,
    parse_string_set,
    shape_polygon,
)

DEFAULT_PRODUCT_LABELS = (
    "罐装,瓶装,袋装,盒装,桶装,条装,未定义包装,严重遮挡,过于模糊,信息不足"
)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class UncoveredProduct:
    image_path: str
    json_path: str
    label: str
    product_index: int
    best_coverage: float
    best_rect_label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "校验结果 JSON 中手持商品是否被同文件内某一个最小外接矩形覆盖。"
        )
    )
    parser.add_argument(
        "--result_root",
        type=str,
        default="",
        help="结果数据集根目录（含 images/ 与同级 json_labels/）。",
    )
    # 兼容旧参数名：--pred_root / --gt_root 均视为 result_root（单 JSON 模式）。
    parser.add_argument("--pred_root", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--gt_root", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--image_dir_name", type=str, default="images")
    parser.add_argument("--json_dir_name", type=str, default="json_labels")
    parser.add_argument(
        "--product_labels",
        type=str,
        default=DEFAULT_PRODUCT_LABELS,
        help="视为手持商品的 label，逗号分隔。",
    )
    parser.add_argument(
        "--rect_label",
        type=str,
        default="最小外接矩形",
        help="最小外接矩形 label 基础名。",
    )
    parser.add_argument(
        "--coverage_threshold",
        type=float,
        default=0.9,
        help="交面积/商品面积 下限；默认 0.9。",
    )
    parser.add_argument(
        "--report_json",
        type=str,
        default="",
        help="上报 JSON 路径；默认 <result_root>/validation_uncovered.json",
    )
    parser.add_argument(
        "--report_txt",
        type=str,
        default="",
        help="上报文本路径；默认 <result_root>/validation_uncovered.txt",
    )
    return parser.parse_args()


def find_image_files(image_dir: Path) -> List[Path]:
    return [
        path
        for path in sorted(image_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]


def load_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON 顶层必须是对象：{path}")
    return data


def canvas_size(data: Dict[str, Any], fallback: Tuple[int, int]) -> Tuple[int, int]:
    width = data.get("imageWidth")
    height = data.get("imageHeight")
    if (
        isinstance(width, (int, float))
        and isinstance(height, (int, float))
        and not isinstance(width, bool)
        and not isinstance(height, bool)
        and width > 0
        and height > 0
    ):
        return int(width), int(height)
    return fallback


def coverage_of_product_by_rect(
    product: Sequence[Sequence[float]], rect: Sequence[Sequence[float]]
) -> float:
    product_poly = as_convex_polygon(product)
    rect_poly = as_convex_polygon(rect)
    product_area = abs(float(cv2.contourArea(product_poly)))
    if product_area <= 1e-6:
        return 0.0
    intersection_area, _ = cv2.intersectConvexConvex(product_poly, rect_poly)
    intersection = max(0.0, float(intersection_area))
    return intersection / product_area


def collect_products(
    shapes: Sequence[Any],
    *,
    width: int,
    height: int,
    product_labels: Set[str],
) -> List[Tuple[str, List[List[float]]]]:
    products: List[Tuple[str, List[List[float]]]] = []
    for shape in shapes:
        if not isinstance(shape, dict):
            continue
        label = str(shape.get("label", "")).strip()
        if not label or not label_matches(label, product_labels):
            continue
        polygon = shape_polygon(shape, width, height)
        if polygon is None:
            continue
        products.append((label, polygon))
    return products


def collect_rects(
    shapes: Sequence[Any],
    *,
    width: int,
    height: int,
    rect_label: str,
) -> List[Tuple[str, List[List[float]]]]:
    rects: List[Tuple[str, List[List[float]]]] = []
    for shape in shapes:
        if not isinstance(shape, dict):
            continue
        label = str(shape.get("label", "")).strip()
        if not is_rect_label(label, rect_label):
            continue
        polygon = shape_polygon(shape, width, height)
        if polygon is None:
            continue
        rects.append((label, polygon))
    return rects


def validate_one(
    image_path: Path,
    json_path: Path,
    *,
    product_labels: Set[str],
    rect_label: str,
    coverage_threshold: float,
) -> List[UncoveredProduct]:
    data = load_json(json_path)
    shapes = data.get("shapes") or []
    if not isinstance(shapes, list):
        raise ValueError(f"shapes 非法：{json_path}")

    width, height = canvas_size(data, (0, 0))
    if width <= 0 or height <= 0:
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"无法读取图片以确定尺寸：{image_path}")
        height, width = img.shape[:2]

    products = collect_products(
        shapes, width=width, height=height, product_labels=product_labels
    )
    rects = collect_rects(shapes, width=width, height=height, rect_label=rect_label)

    uncovered: List[UncoveredProduct] = []
    for index, (label, product) in enumerate(products):
        best_coverage = -1.0
        best_rect_label = ""
        for rect_name, rect in rects:
            coverage = coverage_of_product_by_rect(product, rect)
            if coverage > best_coverage:
                best_coverage = coverage
                best_rect_label = rect_name
        if best_coverage < 0:
            best_coverage = 0.0
        if best_coverage < coverage_threshold:
            uncovered.append(
                UncoveredProduct(
                    image_path=str(image_path),
                    json_path=str(json_path),
                    label=label,
                    product_index=index,
                    best_coverage=round(float(best_coverage), 6),
                    best_rect_label=best_rect_label or "(无最小外接矩形)",
                )
            )
    return uncovered


def main() -> None:
    args = parse_args()
    if not 0.0 < float(args.coverage_threshold) <= 1.0:
        raise ValueError("coverage_threshold 必须在 (0, 1]")

    result_root_value = args.result_root or args.pred_root or args.gt_root
    if not result_root_value:
        raise ValueError("请提供 --result_root（结果目录，含追加了最小外接矩形的 json_labels）")

    result_root = Path(result_root_value).expanduser().resolve()
    image_dir = result_root / args.image_dir_name
    json_dir = result_root / args.json_dir_name
    if not image_dir.is_dir():
        raise FileNotFoundError(f"images 不存在：{image_dir}")
    if not json_dir.is_dir():
        raise FileNotFoundError(f"json_labels 不存在：{json_dir}")

    product_labels = parse_string_set(args.product_labels)
    if not product_labels:
        raise ValueError("product_labels 不能为空")

    images = find_image_files(image_dir)
    if not images:
        raise RuntimeError(f"未找到图片：{image_dir}")

    report_json = (
        Path(args.report_json).expanduser().resolve()
        if args.report_json
        else result_root / "validation_uncovered.json"
    )
    report_txt = (
        Path(args.report_txt).expanduser().resolve()
        if args.report_txt
        else result_root / "validation_uncovered.txt"
    )

    all_uncovered: List[UncoveredProduct] = []
    checked = 0
    skipped_no_json = 0
    failures: List[str] = []

    for image_path in images:
        json_path = json_dir / f"{image_path.stem}.json"
        if not json_path.is_file():
            skipped_no_json += 1
            continue
        try:
            uncovered = validate_one(
                image_path,
                json_path,
                product_labels=product_labels,
                rect_label=args.rect_label,
                coverage_threshold=float(args.coverage_threshold),
            )
            all_uncovered.extend(uncovered)
            checked += 1
        except Exception as exc:
            message = f"{image_path}: {type(exc).__name__}: {exc}"
            failures.append(message)
            print(f"[ERROR] {message}", file=sys.stderr)

    uncovered_images = sorted({item.image_path for item in all_uncovered})
    payload = {
        "result_root": str(result_root),
        "coverage_threshold": float(args.coverage_threshold),
        "product_labels": sorted(product_labels),
        "checked_images": checked,
        "skipped_no_json": skipped_no_json,
        "uncovered_product_count": len(all_uncovered),
        "uncovered_image_count": len(uncovered_images),
        "uncovered_image_paths": uncovered_images,
        "uncovered_products": [asdict(item) for item in all_uncovered],
        "errors": failures,
    }

    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        f"result_root={result_root}",
        f"coverage_threshold={args.coverage_threshold}",
        f"checked_images={checked}",
        f"uncovered_image_count={len(uncovered_images)}",
        f"uncovered_product_count={len(all_uncovered)}",
        "",
        "未覆盖图片路径：",
    ]
    if uncovered_images:
        lines.extend(uncovered_images)
    else:
        lines.append("(无)")
    lines.append("")
    lines.append("明细：")
    if all_uncovered:
        for item in all_uncovered:
            lines.append(
                f"- image={item.image_path} json={item.json_path} "
                f"label={item.label} idx={item.product_index} "
                f"best_coverage={item.best_coverage} best_rect={item.best_rect_label}"
            )
    else:
        lines.append("(无)")
    report_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=" * 72)
    print(
        f"校验完成。检查图片={checked}；未覆盖商品={len(all_uncovered)}；"
        f"未覆盖图片={len(uncovered_images)}；错误={len(failures)}"
    )
    print(f"上报 JSON：{report_json}")
    print(f"上报 TXT ：{report_txt}")
    if uncovered_images:
        print("未覆盖图片：")
        for path in uncovered_images:
            print(f"  {path}")
    if failures:
        raise SystemExit(2)
    if all_uncovered:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
