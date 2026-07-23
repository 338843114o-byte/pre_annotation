#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从视频 URL（或本地路径）抽帧，输出为 handheld_product_min_rect_code 可用的数据集目录：

  output_root/
  └── images/
      ├── frame_000000.jpg
      ├── frame_000001.jpg
      └── ...

抽帧完成后可直接：

  export DATASET_ROOT=<output_root>
  export LABEL_SOURCE=yolo
  export WEIGHTS=.../for_skus.pt
  export HAND_WEIGHTS=.../for_hands.pt
  export OUTPUT_ROOT=.../minrect_out
  bash run.sh

或加 --run_min_rect 在本脚本内衔接到 add_handheld_product_min_rect.py。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="视频 URL/本地路径 → images/ 帧图；可衔接最小外接矩形脚本。"
    )
    parser.add_argument(
        "--video_url",
        type=str,
        required=True,
        help="视频 URL（http/https）或本地文件路径。",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="输出数据集根目录；帧写入 <output_root>/images/。",
    )
    parser.add_argument(
        "--image_dir_name",
        type=str,
        default="images",
        help="图片子目录名，默认 images（与 min_rect 脚本一致）。",
    )
    parser.add_argument(
        "--frame_prefix",
        type=str,
        default="frame_",
        help="帧文件名前缀，默认 frame_。",
    )
    parser.add_argument(
        "--image_ext",
        type=str,
        default=".jpg",
        choices=(".jpg", ".jpeg", ".png", ".bmp", ".webp"),
        help="输出图片扩展名，默认 .jpg。",
    )
    parser.add_argument(
        "--every_n",
        type=int,
        default=1,
        help="每隔 N 帧取 1 帧（按原始帧序号），默认 1（全抽）。",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=0.0,
        help="按目标 FPS 抽帧（>0 时优先于 --every_n）。例如 2 表示约每秒 2 张。",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="最多输出多少张；0 表示不限制。",
    )
    parser.add_argument(
        "--start_sec",
        type=float,
        default=0.0,
        help="从第几秒开始抽，默认 0。",
    )
    parser.add_argument(
        "--end_sec",
        type=float,
        default=-1.0,
        help="抽到第几秒结束；<0 表示直到片尾。",
    )
    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=92,
        help="JPEG 质量 1–100，默认 92。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许清空已有 images 目录后重写。",
    )
    parser.add_argument(
        "--keep_video",
        action="store_true",
        help="URL 下载的视频保留在 output_root/source_video；默认用完即删临时文件。",
    )
    parser.add_argument(
        "--run_min_rect",
        action="store_true",
        help="抽帧后立刻调用 add_handheld_product_min_rect.py（默认 label_source=yolo）。",
    )
    parser.add_argument(
        "--min_rect_output_root",
        type=str,
        default="",
        help="衔接 min_rect 时的 --output_root；默认 <output_root>_minrect。",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="/home/data_manager/jiangfan/for_skus.pt",
        help="商品 YOLO 权重（--run_min_rect 时使用）。",
    )
    parser.add_argument(
        "--hand_weights",
        type=str,
        default="/home/data_manager/jiangfan/for_hands.pt",
        help="手部 YOLO 权重（--run_min_rect 时使用）。",
    )
    parser.add_argument(
        "--label_source",
        choices=("json", "yolo"),
        default="yolo",
        help="衔接 min_rect 时的标注来源，默认 yolo。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="衔接 min_rect 时的 device，默认 0。",
    )
    parser.add_argument(
        "--extra_min_rect_args",
        type=str,
        default="",
        help="额外传给 add_handheld_product_min_rect.py 的参数字符串。",
    )
    return parser.parse_args()


def is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"}


def download_video(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] 下载视频：{url}")
    print(f"[INFO] 保存到：{destination}")
    with urllib.request.urlopen(url, timeout=120) as response, destination.open(
        "wb"
    ) as handle:
        shutil.copyfileobj(response, handle)
    if destination.stat().st_size <= 0:
        raise RuntimeError(f"下载失败或文件为空：{destination}")
    return destination


def resolve_video_path(video_url: str, output_root: Path, keep_video: bool) -> Tuple[Path, Optional[Path]]:
    """
    返回 (可读视频路径, 需要清理的临时目录或 None)。
    """
    if not is_url(video_url):
        path = Path(video_url).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"本地视频不存在：{path}")
        return path, None

    suffix = Path(urlparse(video_url).path).suffix or ".mp4"
    if keep_video:
        dest = output_root / "source_video" / f"input{suffix}"
        download_video(video_url, dest)
        return dest, None

    temp_dir = Path(tempfile.mkdtemp(prefix="video_frames_"))
    dest = temp_dir / f"input{suffix}"
    download_video(video_url, dest)
    return dest, temp_dir


def extract_frames(
    video_path: Path,
    image_dir: Path,
    *,
    frame_prefix: str,
    image_ext: str,
    every_n: int,
    target_fps: float,
    max_frames: int,
    start_sec: float,
    end_sec: float,
    jpeg_quality: int,
) -> dict:
    if every_n <= 0:
        raise ValueError("--every_n 必须 >= 1")
    if target_fps < 0:
        raise ValueError("--fps 不能为负")
    if max_frames < 0:
        raise ValueError("--max_frames 不能为负")
    if start_sec < 0:
        raise ValueError("--start_sec 不能为负")

    image_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    start_frame = 0
    if start_sec > 0 and source_fps > 1e-6:
        start_frame = int(start_sec * source_fps)
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    end_frame = total_frames if total_frames > 0 else None
    if end_sec >= 0 and source_fps > 1e-6:
        end_frame = int(end_sec * source_fps)

    # 按目标 fps 换算步长；否则用 every_n。
    if target_fps > 0 and source_fps > 1e-6:
        step = max(1, int(round(source_fps / target_fps)))
    else:
        step = every_n

    encode_params = []
    if image_ext.lower() in {".jpg", ".jpeg"}:
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]

    saved = 0
    frame_index = start_frame
    written_names = []
    while True:
        if end_frame is not None and frame_index >= end_frame:
            break
        ok, frame = capture.read()
        if not ok:
            break
        if (frame_index - start_frame) % step == 0:
            name = f"{frame_prefix}{saved:06d}{image_ext}"
            out_path = image_dir / name
            if not cv2.imwrite(str(out_path), frame, encode_params):
                raise RuntimeError(f"写图失败：{out_path}")
            written_names.append(name)
            saved += 1
            if saved == 1 or saved % 50 == 0:
                print(f"[INFO] 已写出 {saved} 帧 ...", flush=True)
            if max_frames > 0 and saved >= max_frames:
                break
        frame_index += 1

    capture.release()
    meta = {
        "video_path": str(video_path),
        "source_fps": source_fps,
        "total_frames_reported": total_frames,
        "width": width,
        "height": height,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "step": step,
        "target_fps": target_fps,
        "every_n": every_n,
        "frames_written": saved,
        "image_dir": str(image_dir),
        "sample_names": written_names[:5],
    }
    return meta


def run_min_rect(args: argparse.Namespace, dataset_root: Path) -> None:
    script_dir = Path(__file__).resolve().parent
    min_rect_script = script_dir / "add_handheld_product_min_rect.py"
    if not min_rect_script.is_file():
        raise FileNotFoundError(f"未找到衔接脚本：{min_rect_script}")

    output_root = (
        Path(args.min_rect_output_root).expanduser().resolve()
        if args.min_rect_output_root
        else Path(str(dataset_root) + "_minrect")
    )
    cmd = [
        sys.executable,
        str(min_rect_script),
        "--dataset_root",
        str(dataset_root),
        "--weights",
        args.weights,
        "--hand_weights",
        args.hand_weights,
        "--label_source",
        args.label_source,
        "--output_root",
        str(output_root),
        "--device",
        args.device,
        "--overwrite",
        "--no-skip_existing",
    ]
    if args.extra_min_rect_args.strip():
        cmd.extend(args.extra_min_rect_args.split())

    print("[INFO] 衔接最小外接矩形：")
    print(" ", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[INFO] min_rect 输出：{output_root}")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    image_dir = output_root / args.image_dir_name

    if image_dir.exists():
        has_files = any(image_dir.iterdir())
        if has_files and not args.overwrite:
            raise FileExistsError(
                f"图片目录已存在且非空：{image_dir}；清空重跑请加 --overwrite"
            )
        if has_files and args.overwrite:
            shutil.rmtree(image_dir)

    output_root.mkdir(parents=True, exist_ok=True)
    video_path, temp_dir = resolve_video_path(
        args.video_url, output_root, keep_video=args.keep_video
    )
    try:
        meta = extract_frames(
            video_path,
            image_dir,
            frame_prefix=args.frame_prefix,
            image_ext=args.image_ext,
            every_n=args.every_n,
            target_fps=float(args.fps),
            max_frames=int(args.max_frames),
            start_sec=float(args.start_sec),
            end_sec=float(args.end_sec),
            jpeg_quality=int(args.jpeg_quality),
        )
        meta["video_url"] = args.video_url
        meta_path = output_root / "video_frames_meta.json"
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print("=" * 72)
        print(f"[INFO] 抽帧完成：{meta['frames_written']} 张")
        print(f"[INFO] 数据集根目录（可作 DATASET_ROOT）：{output_root}")
        print(f"[INFO] 图片目录：{image_dir}")
        print(f"[INFO] 元信息：{meta_path}")
        print(
            "[INFO] 下一步示例：\n"
            f"  export DATASET_ROOT={output_root}\n"
            "  export LABEL_SOURCE=yolo\n"
            f"  export WEIGHTS={args.weights}\n"
            f"  export HAND_WEIGHTS={args.hand_weights}\n"
            f"  export OUTPUT_ROOT={output_root}_minrect\n"
            "  bash run.sh"
        )
        if args.run_min_rect:
            run_min_rect(args, output_root)
    finally:
        if temp_dir is not None and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
