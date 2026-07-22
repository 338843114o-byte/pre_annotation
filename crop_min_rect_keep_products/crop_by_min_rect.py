#!/usr/bin/env python3
"""按「最小外接矩形」裁切图片，并在 JSON 中只保留商品标注。

流程（默认 oriented）：
1. 将旋转的最小外接矩形刚体旋转为水平，再裁切；
2. 用同一 2x3 仿射矩阵把商品标注点变换到裁切图坐标系；
3. 重新计算 rotation 的 direction，只保留商品 label。

输入目录需包含递归的 images/ 与同级 json_labels/。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np


DEFAULT_PRODUCT_LABELS = "罐装,瓶装,袋装,盒装,桶装,条装"
DEFAULT_RECT_LABEL = "最小外接矩形"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_string_set(value: str) -> Set[str]:
    return {part.strip() for part in value.split(",") if part.strip()}


def label_matches(label: str, labels: Set[str]) -> bool:
    """精确匹配，或前缀匹配（如 袋装_SKU 匹配 袋装）。"""
    if not label or not labels:
        return False
    if label in labels:
        return True
    for item in labels:
        if label.startswith(item + "_") or label.startswith(item + "-"):
            return True
    return False


def find_image_json_pairs(
    root: Path, image_dir_name: str, json_dir_name: str
) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    for image_dir in sorted(root.rglob(image_dir_name)):
        if not image_dir.is_dir():
            continue
        json_dir = image_dir.parent / json_dir_name
        if not json_dir.is_dir():
            continue
        for image_path in sorted(image_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
                continue
            json_path = json_dir / f"{image_path.stem}.json"
            if json_path.is_file():
                pairs.append((image_path, json_path))
    return pairs


def find_shape_by_label(shapes: Sequence[Dict[str, Any]], label: str) -> Optional[Dict[str, Any]]:
    for shape in shapes:
        if isinstance(shape, dict) and str(shape.get("label", "")) == label:
            return shape
    return None


def points_to_array(points: Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(array) < 2:
        raise ValueError("矩形点数不足")
    return array


def aabb_from_points(
    points: np.ndarray, width: int, height: int, pad: int = 0
) -> Tuple[int, int, int, int]:
    x1 = int(np.floor(points[:, 0].min())) - pad
    y1 = int(np.floor(points[:, 1].min())) - pad
    x2 = int(np.ceil(points[:, 0].max())) + pad
    y2 = int(np.ceil(points[:, 1].max())) + pad
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return x1, y1, x2, y2


def normalize_min_area_rect(
    rect: Tuple[Tuple[float, float], Tuple[float, float], float]
) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
    """统一 minAreaRect：长边水平，且旋转角取 [-90, 90] 内的最小等价角。"""
    (cx, cy), (rw, rh), angle = rect
    cx, cy, rw, rh, angle = float(cx), float(cy), float(rw), float(rh), float(angle)
    # OpenCV：angle 对应 size[0]（宽）边；若宽是短边，+90° 让长边成为宽。
    if rw < rh:
        rw, rh = rh, rw
        angle += 90.0
    # 矩形绕法向转 180° 边仍水平，但画面会上下颠倒；取绝对值更小的角。
    while angle > 90.0:
        angle -= 180.0
    while angle < -90.0:
        angle += 180.0
    return (cx, cy), (rw, rh), angle


def crop_image_aabb(
    image: np.ndarray, points: np.ndarray, pad: int
) -> Tuple[np.ndarray, Tuple[int, int, int, int], None]:
    h, w = image.shape[:2]
    box = aabb_from_points(points, w, h, pad=pad)
    x1, y1, x2, y2 = box
    cropped = image[y1:y2, x1:x2].copy()
    return cropped, box, None


def crop_image_oriented(
    image: np.ndarray, points: np.ndarray, pad: int
) -> Tuple[np.ndarray, Tuple[int, int, int, int], np.ndarray]:
    """
    将旋转最小外接矩形刚体旋转为水平后裁切。

    返回 crop、占位 box、2x3 仿射矩阵（商品点用同一矩阵变换，无透视变形）。
    """
    rect = cv2.minAreaRect(points.astype(np.float32))
    center, (rw, rh), angle = normalize_min_area_rect(rect)
    out_w = max(1, int(round(rw)) + 2 * int(pad))
    out_h = max(1, int(round(rh)) + 2 * int(pad))

    # 绕矩形中心旋转 angle，使矩形边水平；再平移到输出图中心。
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    matrix[0, 2] += (out_w * 0.5) - center[0]
    matrix[1, 2] += (out_h * 0.5) - center[1]

    cropped = cv2.warpAffine(
        image,
        matrix,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    h, w = image.shape[:2]
    box = aabb_from_points(points, w, h, pad=0)
    return cropped, box, matrix


def transform_points_aabb(
    points: Sequence[Sequence[float]], box: Tuple[int, int, int, int]
) -> List[List[float]]:
    x1, y1, _, _ = box
    return [[float(p[0]) - x1, float(p[1]) - y1] for p in points]


def transform_points_affine(
    points: Sequence[Sequence[float]], matrix: np.ndarray
) -> List[List[float]]:
    """用 2x3 仿射矩阵变换点（旋转+平移）。"""
    src = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    dst = cv2.transform(src, matrix).reshape(-1, 2)
    return [[float(x), float(y)] for x, y in dst]


def clamp_points(
    points: Sequence[Sequence[float]], width: int, height: int
) -> List[List[float]]:
    clamped: List[List[float]] = []
    for x, y in points:
        clamped.append(
            [
                float(min(max(x, 0.0), max(0.0, width - 1e-3))),
                float(min(max(y, 0.0), max(0.0, height - 1e-3))),
            ]
        )
    return clamped


def shape_has_valid_points(points: Sequence[Sequence[float]], width: int, height: int) -> bool:
    if not points:
        return False
    inside = 0
    for x, y in points:
        if -2 <= x <= width + 2 and -2 <= y <= height + 2:
            inside += 1
    return inside > 0


def rectangle_direction(points: Sequence[Sequence[float]]) -> float:
    if len(points) < 2:
        return 0.0
    dx = float(points[1][0]) - float(points[0][0])
    dy = float(points[1][1]) - float(points[0][1])
    return float(math.atan2(dy, dx) % (2.0 * math.pi))


def filter_product_shapes(
    shapes: Sequence[Any],
    product_labels: Set[str],
    *,
    box: Tuple[int, int, int, int],
    matrix: Optional[np.ndarray],
    out_w: int,
    out_h: int,
) -> List[Dict[str, Any]]:
    """保留商品标注，并把点坐标变换到裁切图坐标系（非原图坐标）。"""
    kept: List[Dict[str, Any]] = []
    for shape in shapes:
        if not isinstance(shape, dict):
            continue
        label = str(shape.get("label", ""))
        if not label_matches(label, product_labels):
            continue
        points = shape.get("points")
        if not isinstance(points, list) or len(points) < 2:
            continue
        if matrix is not None:
            new_points = transform_points_affine(points, matrix)
        else:
            new_points = transform_points_aabb(points, box)
        if not shape_has_valid_points(new_points, out_w, out_h):
            continue
        new_points = clamp_points(new_points, out_w, out_h)
        new_shape = dict(shape)
        new_shape["points"] = [
            [round(float(x), 3), round(float(y), 3)] for x, y in new_points
        ]
        if str(new_shape.get("shape_type", "")) == "rotation" or "direction" in new_shape:
            new_shape["direction"] = rectangle_direction(new_shape["points"])
        kept.append(new_shape)
    return kept


def build_output_json(
    original: Dict[str, Any],
    product_shapes: List[Dict[str, Any]],
    image_name: str,
    out_w: int,
    out_h: int,
) -> Dict[str, Any]:
    data = dict(original)
    data["shapes"] = product_shapes
    data["imagePath"] = image_name
    data["imageWidth"] = int(out_w)
    data["imageHeight"] = int(out_h)
    data["imageData"] = None
    return data


def process_one(
    image_path: Path,
    json_path: Path,
    output_image: Path,
    output_json: Path,
    *,
    rect_label: str,
    product_labels: Set[str],
    crop_mode: str,
    pad: int,
    skip_empty_products: bool,
    dry_run: bool,
) -> str:
    original = json.loads(json_path.read_text(encoding="utf-8"))
    shapes = original.get("shapes")
    if not isinstance(shapes, list):
        raise ValueError("JSON 无合法 shapes")

    rect_shape = find_shape_by_label(shapes, rect_label)
    if rect_shape is None:
        return "skip_no_rect"

    points = points_to_array(rect_shape.get("points") or [])
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"无法读取图片：{image_path}")

    if crop_mode == "aabb":
        cropped, box, matrix = crop_image_aabb(image, points, pad=pad)
    elif crop_mode == "oriented":
        cropped, box, matrix = crop_image_oriented(image, points, pad=pad)
    else:
        raise ValueError(f"未知 crop_mode：{crop_mode}")

    out_h, out_w = cropped.shape[:2]
    product_shapes = filter_product_shapes(
        shapes,
        product_labels,
        box=box,
        matrix=matrix,
        out_w=out_w,
        out_h=out_h,
    )
    if skip_empty_products and not product_shapes:
        return "skip_no_product"

    out_json = build_output_json(
        original, product_shapes, output_image.name, out_w, out_h
    )
    if dry_run:
        return f"ok_dry products={len(product_shapes)} size={out_w}x{out_h}"

    output_image.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_image), cropped):
        raise RuntimeError(f"写图片失败：{output_image}")
    output_json.write_text(
        json.dumps(out_json, ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )
    return f"ok products={len(product_shapes)} size={out_w}x{out_h}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按最小外接矩形裁切，JSON 只保留商品标注。"
    )
    parser.add_argument("--dataset_root", required=True, help="含 images/json_labels 的根目录")
    parser.add_argument("--output_root", required=True, help="输出根目录")
    parser.add_argument("--rect_label", default=DEFAULT_RECT_LABEL)
    parser.add_argument(
        "--product_labels",
        default=DEFAULT_PRODUCT_LABELS,
        help="保留的商品 label，逗号分隔；支持 袋装_SKU 前缀匹配。",
    )
    parser.add_argument(
        "--crop_mode",
        choices=("oriented", "aabb"),
        default="oriented",
        help=(
            "oriented=把旋转最小外接矩形刚体旋转为水平再裁切，商品坐标用仿射变换到新图；"
            "aabb=仅按旋转框外接水平框裁切并平移坐标。"
        ),
    )
    parser.add_argument("--pad", type=int, default=0, help="裁切额外外扩像素。")
    parser.add_argument("--image_dir_name", default="images")
    parser.add_argument("--json_dir_name", default="json_labels")
    parser.add_argument(
        "--skip_empty_products",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="裁切后若无商品标注则跳过（默认开启）。",
    )
    parser.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--progress_every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset_root 不存在：{dataset_root}")
    if output_root == dataset_root:
        raise ValueError("output_root 不能与 dataset_root 相同，请另指定输出目录。")

    product_labels = parse_string_set(args.product_labels)
    if not product_labels:
        raise ValueError("product_labels 不能为空")

    pairs = find_image_json_pairs(dataset_root, args.image_dir_name, args.json_dir_name)
    if not pairs:
        raise RuntimeError(f"未找到 images/json_labels 配对：{dataset_root}")

    print(f"[INFO] dataset_root: {dataset_root}")
    print(f"[INFO] output_root:  {output_root}")
    print(f"[INFO] pairs: {len(pairs)}")
    print(f"[INFO] rect_label: {args.rect_label}")
    print(f"[INFO] product_labels: {sorted(product_labels)}")
    print(f"[INFO] crop_mode: {args.crop_mode}, pad={args.pad}")

    stats = {
        "ok": 0,
        "skip_no_rect": 0,
        "skip_no_product": 0,
        "skip_existing": 0,
        "error": 0,
    }
    started = time.time()

    for index, (image_path, json_path) in enumerate(pairs, 1):
        rel_image = image_path.relative_to(dataset_root)
        rel_json = json_path.relative_to(dataset_root)
        out_image = output_root / rel_image
        out_json = output_root / rel_json
        try:
            if args.skip_existing and out_image.is_file() and out_json.is_file():
                stats["skip_existing"] += 1
                status = "skip_existing"
            else:
                status = process_one(
                    image_path,
                    json_path,
                    out_image,
                    out_json,
                    rect_label=args.rect_label,
                    product_labels=product_labels,
                    crop_mode=args.crop_mode,
                    pad=max(0, int(args.pad)),
                    skip_empty_products=args.skip_empty_products,
                    dry_run=args.dry_run,
                )
                key = status.split()[0]
                if key.startswith("ok"):
                    stats["ok"] += 1
                elif key in stats:
                    stats[key] += 1
                else:
                    stats["ok"] += 1
        except Exception as exc:
            stats["error"] += 1
            print(
                f"[ERROR] image={image_path}, json={json_path}, "
                f"error={type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            status = "error"

        if index == 1 or index % max(1, args.progress_every) == 0 or index == len(pairs):
            elapsed = time.time() - started
            rate = index / elapsed if elapsed > 0 else 0
            print(
                f"[{index}/{len(pairs)}] {100.0 * index / len(pairs):5.1f}% "
                f"rate={rate:5.1f}/s ok={stats['ok']} "
                f"no_rect={stats['skip_no_rect']} no_prod={stats['skip_no_product']} "
                f"exist={stats['skip_existing']} err={stats['error']} last={status}",
                flush=True,
            )

    print("================================================================================")
    print(
        f"完成。ok={stats['ok']} skip_no_rect={stats['skip_no_rect']} "
        f"skip_no_product={stats['skip_no_product']} skip_existing={stats['skip_existing']} "
        f"error={stats['error']}"
    )
    if stats["error"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
