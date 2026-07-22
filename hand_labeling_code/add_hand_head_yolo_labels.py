#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 YOLO OBB 模型检测手部，并将检测结果追加到对应 JSON 的 shapes 中。

固定类别映射：
  class_id 0 -> 手

处理原则：
1. 只向 shapes 末尾追加手部标注；原有 JSON 字段和值保持不变。
2. class_id=1 的头部检测结果直接忽略，不写入 JSON。
3. 新 shape 沿用当前 JSON 中已有 shape 的字段结构。
4. attributes 等容器保持为空，不写入模型来源、class_id、推理参数等额外内容。
5. 写文件时采用文本级追加，原 JSON 的已有文本不会被重新格式化。
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


IMAGE_EXTS_DEFAULT = ".jpg,.jpeg,.png,.bmp,.webp"
HAND_CLASS_ID = 0


@dataclass(frozen=True)
class Detection:
    label: str
    points: List[List[float]]
    score: float
    class_id: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use the YOLO model to append class 0 (手) annotations to "
            "matching JSON files. Class 1 (头) is ignored."
        )
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--dataset_root",
        type=str,
        help=(
            "已解压的数据集根目录。脚本会递归查找同级的 "
            "images/json_labels 目录。"
        ),
    )
    input_group.add_argument(
        "--input_zip",
        type=str,
        help="输入数据集压缩包；先解压到 --work_dir，再处理。",
    )

    parser.add_argument("--weights", type=str, required=True, help="best.pt 路径。")
    parser.add_argument(
        "--work_dir",
        type=str,
        default="./hand_head_work",
        help="--input_zip 模式的解压目录。",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help=(
            "输出目录。不填写时原地修改 JSON；填写时先完整复制数据集，"
            "然后只修改副本中的 JSON。"
        ),
    )
    parser.add_argument("--output_zip", type=str, default=None, help="可选的结果 zip 路径。")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已存在的 work_dir/output_root/output_zip。",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="output_root 已存在时继续处理，不重新复制数据集。",
    )

    parser.add_argument("--image_dir_name", type=str, default="images")
    parser.add_argument("--json_dir_name", type=str, default="json_labels")
    parser.add_argument("--image_exts", type=str, default=IMAGE_EXTS_DEFAULT)

    parser.add_argument("--device", type=str, default="0", help="例如 0、1 或 cpu。")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)

    parser.add_argument(
        "--hand_label",
        type=str,
        default="手",
        help="class_id=0 写入 JSON 时的 label，默认 手。",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help=(
            "若 JSON 已含 hand_label，则不再追加手部标注，"
            "用于避免同一批数据被重复处理。"
        ),
    )
    parser.add_argument(
        "--backup_json",
        action="store_true",
        help="原地修改时，为被修改的 JSON 创建一次 .bak 备份。",
    )
    parser.add_argument("--dry_run", action="store_true", help="只推理和统计，不写文件。")
    parser.add_argument(
        "--progress_every",
        type=int,
        default=100,
        help="每处理多少张图输出一次进度，默认 100。",
    )
    parser.add_argument(
        "--status_file",
        type=str,
        default=None,
        help="将进度写入 JSON 状态文件，便于 tail/监控。",
    )
    return parser.parse_args()


def load_json_text(path: Path) -> Tuple[str, Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
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


def class_id_to_label(
    class_id: int, hand_label: str
) -> Optional[str]:
    if class_id == HAND_CLASS_ID:
        return hand_label
    return None


def load_yolo_model(weights: str):
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(
            "无法导入 ultralytics，请先执行：python -m pip install -r requirements.txt"
        ) from exc
    return YOLO(weights)


def detections_from_result(
    result: Any,
    width: int,
    height: int,
    hand_label: str,
) -> List[Detection]:
    detections: List[Detection] = []

    # best.pt 是 OBB 模型：优先保留模型给出的四个旋转框顶点。
    obb = getattr(result, "obb", None)
    if obb is not None and len(obb) > 0:
        polygons = obb.xyxyxyxy.detach().cpu().numpy().reshape(-1, 4, 2)
        confidences = obb.conf.detach().cpu().numpy()
        class_ids = obb.cls.detach().cpu().numpy().astype(int)
        for polygon, confidence, class_id in zip(
            polygons, confidences, class_ids
        ):
            class_id = int(class_id)
            label = class_id_to_label(class_id, hand_label)
            if label is None:
                continue
            detections.append(
                Detection(
                    label=label,
                    points=clamp_points(polygon.tolist(), width, height),
                    score=float(confidence),
                    class_id=class_id,
                )
            )
    else:
        # 兼容普通检测模型；普通框转换成四点 polygon/rotation 形式。
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return detections
        xyxys = boxes.xyxy.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()
        class_ids = boxes.cls.detach().cpu().numpy().astype(int)
        for xyxy, confidence, class_id in zip(xyxys, confidences, class_ids):
            class_id = int(class_id)
            label = class_id_to_label(class_id, hand_label)
            if label is None:
                continue
            detections.append(
                Detection(
                    label=label,
                    points=xyxy_to_points(xyxy, width, height),
                    score=float(confidence),
                    class_id=class_id,
                )
            )

    # class 1 已被过滤；手部框按位置和置信度固定排序。
    detections.sort(
        key=lambda detection: (
            detection.class_id,
            detection.points[0][1],
            detection.points[0][0],
            -detection.score,
        )
    )
    return detections


def neutral_value_like(value: Any) -> Any:
    """为模板中的未知字段生成不携带旧标注语义的空值。"""
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


def choose_shape_template(shapes: List[Any]) -> Optional[Dict[str, Any]]:
    for shape in shapes:
        if isinstance(shape, dict) and "label" in shape and "points" in shape:
            return shape
    return None


def make_shape(
    detection: Detection, template: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    按已有 shape 的字段和字段顺序创建新 shape。

    没有可用模板时使用压缩包原脚本对应的 LabelMe/OBB 标准字段；
    attributes 始终为空，不写入任何额外模型信息。
    """
    if template is None:
        return {
            "kie_linking": [],
            "score": round(float(detection.score), 6),
            "direction": 0,
            "label": detection.label,
            "points": detection.points,
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
            shape[key] = detection.label
        elif key == "points":
            shape[key] = detection.points
        elif key == "score":
            shape[key] = round(float(detection.score), 6)
        elif key == "direction":
            shape[key] = 0
        elif key == "group_id":
            shape[key] = None
        elif key == "description":
            shape[key] = ""
        elif key == "difficult":
            shape[key] = False
        elif key == "shape_type":
            shape[key] = copy.deepcopy(old_value) if old_value else "rotation"
        elif key in {"flags", "attributes"}:
            shape[key] = {}
        elif key == "kie_linking":
            shape[key] = []
        else:
            shape[key] = neutral_value_like(old_value)

    # label/points 是标注必需字段。正常 JSON 模板一定已有；这里仅作保护。
    if "label" not in shape:
        shape["label"] = detection.label
    if "points" not in shape:
        shape["points"] = detection.points
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
        if index >= len(text):
            break
        if text[index] == "}":
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
    line_start = text.rfind("\n", 0, position) + 1
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
    """仅在 shapes 数组末尾插入对象，不重写已有 JSON 文本。"""
    if not new_shapes:
        return original_text

    key_start, array_start, array_end = find_top_level_shapes_span(original_text)
    item_indent, indent_unit = detect_indentation(
        original_text, key_start, array_start, array_end
    )
    rendered_items: List[str] = []
    for shape in new_shapes:
        rendered = json.dumps(shape, ensure_ascii=False, indent=indent_unit)
        rendered_items.append(item_indent + rendered.replace("\n", "\n" + item_indent))
    rendered_block = (",\n").join(rendered_items)

    content = original_text[array_start + 1 : array_end]
    has_existing_items = bool(content.strip())
    if has_existing_items:
        insertion_index = array_end
        while (
            insertion_index > array_start + 1
            and original_text[insertion_index - 1].isspace()
        ):
            insertion_index -= 1
        insertion = ",\n" + rendered_block
    else:
        insertion_index = array_start + 1
        close_indent = _line_indent(original_text, array_end)
        insertion = "\n" + rendered_block
        if not content:
            insertion += "\n" + close_indent

    return (
        original_text[:insertion_index]
        + insertion
        + original_text[insertion_index:]
    )


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
        raise AssertionError("新增 shapes 与预期不一致")


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


def prepare_input_root(args: argparse.Namespace) -> Path:
    if args.input_zip:
        zip_path = Path(args.input_zip).expanduser().resolve()
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
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(extract_dir)
        return extract_dir
    return Path(args.dataset_root).expanduser().resolve()


def prepare_process_root(input_root: Path, args: argparse.Namespace) -> Path:
    if not args.output_root:
        return input_root
    output_root = Path(args.output_root).expanduser().resolve()
    if output_root.exists():
        if args.resume:
            print(f"[INFO] resume 模式：继续使用已有输出目录 {output_root}")
            return output_root
        if args.overwrite:
            shutil.rmtree(output_root)
        else:
            raise FileExistsError(
                f"output_root 已存在：{output_root}；"
                "如需继续处理请加 --resume，如需覆盖请加 --overwrite"
            )
    print(f"[INFO] 正在复制数据集到 {output_root} ...")
    copy_started = time.time()
    shutil.copytree(input_root, output_root)
    elapsed = time.time() - copy_started
    print(f"[INFO] 数据集复制完成，耗时 {elapsed:.1f}s")
    return output_root


def collect_existing_labels(shapes: List[Any]) -> Set[str]:
    return {
        str(shape.get("label", ""))
        for shape in shapes
        if isinstance(shape, dict)
    }


def format_progress_line(
    index: int,
    total: int,
    elapsed_s: float,
    changed_files: int,
    total_hand: int,
    skipped_existing: int,
    failures: int,
) -> str:
    percent = (index / total) * 100 if total else 100.0
    rate = index / elapsed_s if elapsed_s > 0 else 0.0
    remaining = total - index
    eta_s = remaining / rate if rate > 0 else 0.0
    bar_width = 30
    filled = int(bar_width * index / total) if total else bar_width
    bar = "=" * filled + "-" * (bar_width - filled)
    return (
        f"[{bar}] {index}/{total} ({percent:5.1f}%) "
        f"rate={rate:5.1f} img/s eta={eta_s/60:6.1f}m "
        f"hand+={total_hand} changed={changed_files} "
        f"skip={skipped_existing} err={failures}"
    )


def write_status_file(
    status_path: Optional[Path],
    *,
    index: int,
    total: int,
    elapsed_s: float,
    changed_files: int,
    total_hand: int,
    skipped_existing: int,
    failures: int,
    last_json: Optional[Path],
    done: bool = False,
) -> None:
    if status_path is None:
        return
    rate = index / elapsed_s if elapsed_s > 0 else 0.0
    remaining = total - index
    eta_s = remaining / rate if rate > 0 and not done else 0.0
    payload = {
        "done": index,
        "total": total,
        "percent": round((index / total) * 100, 2) if total else 100.0,
        "elapsed_s": round(elapsed_s, 1),
        "eta_s": round(eta_s, 1),
        "rate_img_per_s": round(rate, 2),
        "hands_added": total_hand,
        "changed_files": changed_files,
        "skipped_existing": skipped_existing,
        "errors": failures,
        "finished": done,
        "last_json": str(last_json) if last_json else None,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    temp_name: Optional[str] = None
    try:
        status_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(status_path.parent),
            prefix=status_path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            json.dump(payload, temp_file, ensure_ascii=False, indent=2)
            temp_file.write("\n")
            temp_file.flush()
            os.fsync(temp_file.fileno())
            temp_name = temp_file.name
        os.replace(temp_name, status_path)
    finally:
        if temp_name is not None and os.path.exists(temp_name):
            os.unlink(temp_name)


def find_image_json_pairs(
    root: Path,
    image_dir_name: str,
    json_dir_name: str,
    image_extensions: Set[str],
) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    image_dirs = sorted(
        path
        for path in root.rglob(image_dir_name)
        if path.is_dir() and path.name == image_dir_name
    )
    for image_dir in image_dirs:
        json_dir = image_dir.parent / json_dir_name
        if not json_dir.is_dir():
            print(
                f"[WARN] 缺少对应 JSON 目录，跳过：{image_dir} -> {json_dir}",
                file=sys.stderr,
            )
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
) -> Tuple[int, bool]:
    original_text, original_data = load_json_text(json_path)
    original_shapes = original_data.get("shapes")
    if not isinstance(original_shapes, list):
        raise ValueError(f"JSON 中没有合法 shapes 数组：{json_path}")

    existing_labels = collect_existing_labels(original_shapes)
    if args.skip_existing and args.hand_label in existing_labels:
        return 0, True

    width, height = get_image_size_from_json_or_file(original_data, image_path)
    results = model.predict(
        source=str(image_path),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        classes=[HAND_CLASS_ID],
        verbose=False,
    )
    if not results:
        raise RuntimeError("YOLO 没有返回推理结果")

    detections = detections_from_result(
        result=results[0],
        width=width,
        height=height,
        hand_label=args.hand_label,
    )

    if args.skip_existing:
        detections = [
            detection
            for detection in detections
            if detection.label not in existing_labels
        ]

    template = choose_shape_template(original_shapes)
    new_shapes = [make_shape(detection, template) for detection in detections]
    hand_count = sum(
        1 for detection in detections if detection.class_id == HAND_CLASS_ID
    )

    if new_shapes and not args.dry_run:
        updated_text = append_shapes_preserving_original_text(
            original_text, new_shapes
        )
        verify_only_shapes_appended(original_data, updated_text, new_shapes)
        if args.backup_json and not args.output_root:
            backup_path = json_path.with_suffix(json_path.suffix + ".bak")
            if not backup_path.exists():
                shutil.copy2(json_path, backup_path)
        atomic_write_text(json_path, updated_text)

    return hand_count, False


def make_zip_from_dir(
    source_dir: Path, output_zip: Path, overwrite: bool = False
) -> None:
    output_zip = output_zip.expanduser().resolve()
    if output_zip.exists():
        if overwrite:
            output_zip.unlink()
        else:
            raise FileExistsError(
                f"output_zip 已存在：{output_zip}；如需覆盖请加 --overwrite"
            )
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output_zip, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source_dir))


def main() -> None:
    args = parse_args()
    if not args.hand_label.strip():
        raise ValueError("hand_label 不能为空")

    image_extensions = {
        extension.strip().lower()
        for extension in args.image_exts.split(",")
        if extension.strip()
    }
    input_root = prepare_input_root(args)
    process_root = prepare_process_root(input_root, args)
    pairs = find_image_json_pairs(
        process_root,
        args.image_dir_name,
        args.json_dir_name,
        image_extensions,
    )
    if not pairs:
        raise RuntimeError(
            "没有找到 images/json_labels 图片与同名 JSON 配对。"
            f"root={process_root}"
        )

    print(f"[INFO] process_root: {process_root}")
    print(f"[INFO] image/json pairs: {len(pairs)}")
    print(f"[INFO] fixed class map: 0={args.hand_label}; class 1 ignored")
    print(
        f"[INFO] conf={args.conf}, iou={args.iou}, "
        f"imgsz={args.imgsz}, device={args.device}"
    )

    model = load_yolo_model(args.weights)
    total_hand = 0
    changed_files = 0
    skipped_existing = 0
    failures: List[str] = []
    status_path = (
        Path(args.status_file).expanduser().resolve()
        if args.status_file
        else None
    )
    progress_every = max(1, args.progress_every)
    started_at = time.time()

    for index, (image_path, json_path) in enumerate(pairs, 1):
        try:
            hand_count, skipped = process_one(
                model, image_path, json_path, args
            )
            if skipped:
                skipped_existing += 1
            if hand_count:
                changed_files += 1
            total_hand += hand_count
        except Exception as exc:
            message = (
                f"image={image_path}, json={json_path}, "
                f"error={type(exc).__name__}: {exc}"
            )
            failures.append(message)
            print(f"[ERROR] {message}", file=sys.stderr)

        elapsed_s = time.time() - started_at
        if index == 1 or index % progress_every == 0 or index == len(pairs):
            print(
                format_progress_line(
                    index=index,
                    total=len(pairs),
                    elapsed_s=elapsed_s,
                    changed_files=changed_files,
                    total_hand=total_hand,
                    skipped_existing=skipped_existing,
                    failures=len(failures),
                ),
                flush=True,
            )
            write_status_file(
                status_path,
                index=index,
                total=len(pairs),
                elapsed_s=elapsed_s,
                changed_files=changed_files,
                total_hand=total_hand,
                skipped_existing=skipped_existing,
                failures=len(failures),
                last_json=json_path,
            )

    write_status_file(
        status_path,
        index=len(pairs),
        total=len(pairs),
        elapsed_s=time.time() - started_at,
        changed_files=changed_files,
        total_hand=total_hand,
        skipped_existing=skipped_existing,
        failures=len(failures),
        last_json=pairs[-1][1] if pairs else None,
        done=True,
    )

    if failures:
        print("[ERROR] 存在处理失败项，不生成 output_zip。", file=sys.stderr)
    elif args.output_zip:
        make_zip_from_dir(
            process_root, Path(args.output_zip), overwrite=args.overwrite
        )
        print(f"[INFO] output_zip: {Path(args.output_zip).expanduser().resolve()}")

    print("=" * 80)
    print(
        f"完成。配对数：{len(pairs)}；有新增标注的 JSON：{changed_files}；"
        f"新增 hand：{total_hand}；跳过已有手标注：{skipped_existing}；"
        f"失败：{len(failures)}"
    )
    if args.dry_run:
        print("dry-run：没有写入 JSON。")
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
