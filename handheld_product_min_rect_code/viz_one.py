#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将单张图的 JSON 标注画到图上；最小外接矩形_N → min_rect_N。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from add_handheld_product_min_rect import viz_label_text

VIZ_COLORS = {
    "can": (0, 165, 255),
    "bottle": (0, 165, 255),
    "bag": (0, 165, 255),
    "box": (0, 165, 255),
    "bucket": (0, 165, 255),
    "stick": (0, 165, 255),
    "hand": (0, 255, 0),
    "cabinet": (160, 160, 160),
    "head": (255, 0, 255),
    "min_rect": (0, 0, 255),
}


def color_for_viz_label(text: str) -> tuple:
    if text == "min_rect" or text.startswith("min_rect_"):
        return VIZ_COLORS["min_rect"]
    return VIZ_COLORS.get(text, (255, 200, 0))


def visualize_pair(
    image_path: Path,
    json_path: Path,
    output_path: Path,
) -> Path:
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"无法读取图片：{image_path}")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    h, w = img.shape[:2]
    jw = float(data.get("imageWidth") or w)
    jh = float(data.get("imageHeight") or h)
    sx, sy = w / jw, h / jh

    for shape in data.get("shapes") or []:
        if not isinstance(shape, dict):
            continue
        lab = str(shape.get("label", ""))
        pts = shape.get("points") or []
        if len(pts) < 2:
            continue
        text = viz_label_text(lab)
        color = color_for_viz_label(text)
        scaled = [[float(x) * sx, float(y) * sy] for x, y in pts]
        arr = np.asarray(scaled, dtype=np.int32).reshape(-1, 1, 2)
        thick = 3 if text == "min_rect" or text.startswith("min_rect_") else 2
        cv2.polylines(img, [arr], True, color, thick, lineType=cv2.LINE_AA)
        x0, y0 = int(arr[0, 0, 0]), int(arr[0, 0, 1])
        cv2.putText(
            img,
            text,
            (x0, max(22, y0 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="可视化单张标注图")
    parser.add_argument("--image", required=True, type=str, help="图片路径")
    parser.add_argument("--json", type=str, default="", help="JSON 路径；默认同目录 json_labels/同名.json")
    parser.add_argument("--output", type=str, default="", help="输出路径；默认 ./viz_<stem>.jpg")
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    if args.json:
        json_path = Path(args.json).expanduser().resolve()
    else:
        json_path = image_path.parent.parent / "json_labels" / f"{image_path.stem}.json"
        if not json_path.is_file():
            json_path = image_path.with_suffix(".json")
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        # 文件名含 # 时另存短名，便于编辑器打开
        safe = image_path.stem.replace("#", "_")
        if len(safe) > 80:
            safe = safe[:40] + "_viz"
        output_path = Path.cwd() / f"viz_{safe}.jpg"

    out = visualize_pair(image_path, json_path, output_path)
    labels = []
    data = json.loads(json_path.read_text(encoding="utf-8"))
    for s in data.get("shapes") or []:
        lab = str(s.get("label", ""))
        if lab:
            labels.append(f"{lab}→{viz_label_text(lab)}")
    print(f"wrote {out}")
    print("labels:", labels)


if __name__ == "__main__":
    main()
