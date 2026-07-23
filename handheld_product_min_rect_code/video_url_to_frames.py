#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从视频 URL（或本地路径）抽帧，输出为 handheld_product_min_rect_code 可用的数据集目录：

  output_root/
  └── images/
      ├── frame_000000.jpg
      ├── frame_000001.jpg
      └── ...

输入方式（四选一）：
  1) --video_url          单个 URL 或本地视频路径
  2) --url_file xxx.txt   一行一个 URL（如 vedio_url/一行一个URL.txt）
  3) --url_file xxx.xlsx  「视频URL」列（如 vedio_url/一行一个URL.xlsx）
  4) --url_file xxx.xlsx  「订单视频」列 JSON 数组（如 vedio_url/重叠订单_查询结果.xlsx）

批量（>1 个视频）时每个视频写入独立子目录：
  output_root/<job_name>/images/

抽帧完成后可直接：

  export DATASET_ROOT=<output_root 或子目录>
  export LABEL_SOURCE=yolo
  bash run_images.sh

或加 --run_min_rect 在本脚本内衔接到 add_handheld_product_min_rect.py。
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
from urllib.parse import unquote, urlparse

import cv2

ORDER_VIDEO_HEADERS = {"订单视频", "订单视频url", "订单视频URL", "video", "videos", "video_urls"}
PLAIN_URL_HEADERS = {"视频url", "视频URL", "url", "urls", "video_url", "video_urls"}
ORDER_ID_HEADERS = {"订单编号", "订单号", "order_id", "orderid"}


@dataclass
class VideoJob:
    url: str
    order_id: str = ""
    role: str = ""  # main / sub / unknown
    source: str = ""
    source_row: int = -1
    job_name: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="视频 URL/本地路径/URL 列表文件 → images/ 帧图；可衔接最小外接矩形脚本。"
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--video_url",
        type=str,
        default="",
        help="单个视频 URL（http/https）或本地文件路径。",
    )
    src.add_argument(
        "--url_file",
        type=str,
        default="",
        help=(
            "URL 列表文件。支持："
            "① 一行一个 URL 的 .txt；"
            "② 含「视频URL」列的 .xlsx；"
            "③ 含「订单视频」JSON 数组列的重叠订单 .xlsx。"
        ),
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="输出数据集根目录；单视频帧写入 <output_root>/images/，批量则写入子目录。",
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
        help="每个视频最多输出多少张；0 表示不限制。",
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
        help="URL 下载的视频保留在各任务目录 source_video；默认用完即删临时文件。",
    )
    parser.add_argument(
        "--video_role",
        type=str,
        choices=("all", "main", "sub"),
        default="all",
        help="按文件名过滤：all / main（_main） / sub（_sub）。默认 all。",
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=0,
        help="最多处理多少个视频（过滤去重后）；0 表示不限制。",
    )
    parser.add_argument(
        "--flat_output",
        action="store_true",
        help="批量时也全部写入同一 images/（帧名前加 job_name_ 前缀）。默认每个视频独立子目录。",
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


def looks_like_video_ref(value: str) -> bool:
    text = (value or "").strip()
    if not text:
        return False
    if is_url(text):
        return True
    path = Path(text).expanduser()
    return path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def infer_video_role(url: str) -> str:
    name = Path(unquote(urlparse(url).path if is_url(url) else url)).name.lower()
    if "_main." in name or name.endswith("_main.mp4") or "_main_" in name:
        return "main"
    if "_sub." in name or name.endswith("_sub.mp4") or "_sub_" in name:
        return "sub"
    return "unknown"


def sanitize_job_name(raw: str, fallback: str = "video") -> str:
    text = (raw or "").strip()
    text = text.replace("$", "")
    text = re.sub(r"[^\w.\-]+", "_", text, flags=re.UNICODE)
    text = re.sub(r"_+", "_", text).strip("._")
    if not text:
        text = fallback
    return text[:120]


def url_stem(url: str) -> str:
    path = unquote(urlparse(url).path if is_url(url) else url)
    stem = Path(path).stem or "video"
    return sanitize_job_name(stem)


def extract_urls_from_cell(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out: List[str] = []
        for item in value:
            out.extend(extract_urls_from_cell(item))
        return out
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") or text.startswith("{"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if parsed is not None:
            return extract_urls_from_cell(parsed)
    if looks_like_video_ref(text):
        return [text]
    # 宽松：从文本里抠 http(s) 链接
    found = re.findall(r"https?://[^\s,\"'<>\\]+", text)
    return [u.rstrip("]},") for u in found if looks_like_video_ref(u.rstrip("]},"))]


def _norm_header(value) -> str:
    return str(value or "").strip()


def _header_index(headers: Sequence[str], candidates: set) -> Optional[int]:
    lowered = {h.lower(): i for i, h in enumerate(headers)}
    for name in candidates:
        key = name.lower()
        if key in lowered:
            return lowered[key]
    return None


def load_jobs_from_txt(path: Path) -> List[VideoJob]:
    jobs: List[VideoJob] = []
    for i, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        for url in extract_urls_from_cell(text):
            jobs.append(
                VideoJob(
                    url=url,
                    role=infer_video_role(url),
                    source=str(path),
                    source_row=i,
                )
            )
    return jobs


def load_jobs_from_xlsx(path: Path) -> List[VideoJob]:
    try:
        import openpyxl
    except ImportError as exc:
        raise ImportError(
            "读取 xlsx 需要 openpyxl，请先：pip install openpyxl"
        ) from exc

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        jobs: List[VideoJob] = []
        for sheet in wb.worksheets:
            rows = sheet.iter_rows(values_only=True)
            try:
                header_row = next(rows)
            except StopIteration:
                continue
            headers = [_norm_header(h) for h in header_row]
            order_video_idx = _header_index(headers, ORDER_VIDEO_HEADERS)
            plain_url_idx = _header_index(headers, PLAIN_URL_HEADERS)
            order_id_idx = _header_index(headers, ORDER_ID_HEADERS)

            # 无表头命中时：若首行本身像 URL，则整表按「每行一 URL」读
            header_is_data = False
            if order_video_idx is None and plain_url_idx is None:
                for cell in header_row:
                    if extract_urls_from_cell(cell):
                        header_is_data = True
                        break

            def consume_row(row_values, row_no: int) -> None:
                nonlocal jobs
                order_id = ""
                urls: List[str] = []
                if order_id_idx is not None and order_id_idx < len(row_values):
                    order_id = str(row_values[order_id_idx] or "").strip()
                if order_video_idx is not None and order_video_idx < len(row_values):
                    urls = extract_urls_from_cell(row_values[order_video_idx])
                elif plain_url_idx is not None and plain_url_idx < len(row_values):
                    urls = extract_urls_from_cell(row_values[plain_url_idx])
                else:
                    for cell in row_values:
                        urls.extend(extract_urls_from_cell(cell))
                for url in urls:
                    jobs.append(
                        VideoJob(
                            url=url,
                            order_id=order_id,
                            role=infer_video_role(url),
                            source=f"{path}#{sheet.title}",
                            source_row=row_no,
                        )
                    )

            if header_is_data:
                consume_row(tuple(header_row), 1)
                start_row = 2
            else:
                start_row = 2
            for offset, row in enumerate(rows):
                consume_row(tuple(row), start_row + offset)
        return jobs
    finally:
        wb.close()


def load_jobs_from_url_file(path: Path) -> List[VideoJob]:
    if not path.is_file():
        raise FileNotFoundError(f"URL 列表文件不存在：{path}")
    suffix = path.suffix.lower()
    if suffix in {".txt", ".csv", ".list"}:
        return load_jobs_from_txt(path)
    if suffix in {".xlsx", ".xlsm", ".xltx", ".xltm"}:
        return load_jobs_from_xlsx(path)
    # 无扩展名或其它：先按文本试
    return load_jobs_from_txt(path)


def filter_jobs_by_role(jobs: Sequence[VideoJob], role: str) -> List[VideoJob]:
    if role == "all":
        return list(jobs)
    return [j for j in jobs if j.role == role]


def dedupe_jobs(jobs: Sequence[VideoJob]) -> List[VideoJob]:
    seen = set()
    out: List[VideoJob] = []
    for job in jobs:
        key = job.url.strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(job)
    return out


def assign_job_names(jobs: Sequence[VideoJob]) -> List[VideoJob]:
    used = set()
    out: List[VideoJob] = []
    for idx, job in enumerate(jobs, start=1):
        parts = []
        if job.order_id:
            parts.append(sanitize_job_name(job.order_id, "order"))
        parts.append(url_stem(job.url))
        if job.role in {"main", "sub"} and job.role not in parts[-1].lower():
            parts.append(job.role)
        base = sanitize_job_name("_".join(parts), f"video_{idx:04d}")
        name = base
        n = 2
        while name.lower() in used:
            name = f"{base}_{n}"
            n += 1
        used.add(name.lower())
        out.append(
            VideoJob(
                url=job.url,
                order_id=job.order_id,
                role=job.role,
                source=job.source,
                source_row=job.source_row,
                job_name=name,
            )
        )
    return out


def resolve_video_jobs(args: argparse.Namespace) -> List[VideoJob]:
    if args.video_url:
        jobs = [
            VideoJob(
                url=args.video_url.strip(),
                role=infer_video_role(args.video_url),
                source="--video_url",
                source_row=1,
            )
        ]
    else:
        path = Path(args.url_file).expanduser().resolve()
        jobs = load_jobs_from_url_file(path)
        if not jobs:
            raise ValueError(f"未从文件解析到任何视频 URL：{path}")

    jobs = filter_jobs_by_role(jobs, args.video_role)
    if not jobs:
        raise ValueError(f"--video_role={args.video_role} 过滤后无视频可处理")
    jobs = dedupe_jobs(jobs)
    if args.max_videos and args.max_videos > 0:
        jobs = jobs[: int(args.max_videos)]
    return assign_job_names(jobs)


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


def resolve_video_path(
    video_url: str, task_root: Path, keep_video: bool
) -> Tuple[Path, Optional[Path]]:
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
        dest = task_root / "source_video" / f"input{suffix}"
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
    return {
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


def prepare_image_dir(image_dir: Path, overwrite: bool) -> None:
    if image_dir.exists():
        has_files = any(image_dir.iterdir())
        if has_files and not overwrite:
            raise FileExistsError(
                f"图片目录已存在且非空：{image_dir}；清空重跑请加 --overwrite"
            )
        if has_files and overwrite:
            shutil.rmtree(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)


def process_one_job(
    job: VideoJob,
    task_root: Path,
    args: argparse.Namespace,
    *,
    frame_prefix: str,
) -> dict:
    image_dir = task_root / args.image_dir_name
    prepare_image_dir(image_dir, overwrite=args.overwrite)
    video_path, temp_dir = resolve_video_path(
        job.url, task_root, keep_video=args.keep_video
    )
    try:
        meta = extract_frames(
            video_path,
            image_dir,
            frame_prefix=frame_prefix,
            image_ext=args.image_ext,
            every_n=args.every_n,
            target_fps=float(args.fps),
            max_frames=int(args.max_frames),
            start_sec=float(args.start_sec),
            end_sec=float(args.end_sec),
            jpeg_quality=int(args.jpeg_quality),
        )
        meta.update(
            {
                "video_url": job.url,
                "order_id": job.order_id,
                "role": job.role,
                "job_name": job.job_name,
                "source": job.source,
                "source_row": job.source_row,
                "task_root": str(task_root),
            }
        )
        meta_path = task_root / "video_frames_meta.json"
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        meta["meta_path"] = str(meta_path)
        return meta
    finally:
        if temp_dir is not None and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    jobs = resolve_video_jobs(args)
    print(f"[INFO] 待处理视频数：{len(jobs)}")
    for i, job in enumerate(jobs[:10], start=1):
        print(f"  [{i}] {job.job_name}  role={job.role or '-'}  {job.url}")
    if len(jobs) > 10:
        print(f"  ... 其余 {len(jobs) - 10} 个省略")

    batch = len(jobs) > 1 and not args.flat_output
    results = []
    failures = []

    for idx, job in enumerate(jobs, start=1):
        print("=" * 72)
        print(f"[INFO] ({idx}/{len(jobs)}) {job.job_name}")
        if batch:
            task_root = output_root / job.job_name
            frame_prefix = args.frame_prefix
        elif args.flat_output and len(jobs) > 1:
            task_root = output_root
            frame_prefix = f"{job.job_name}_{args.frame_prefix}"
        else:
            task_root = output_root
            frame_prefix = args.frame_prefix
        task_root.mkdir(parents=True, exist_ok=True)
        try:
            meta = process_one_job(job, task_root, args, frame_prefix=frame_prefix)
            print(
                f"[INFO] 完成：写出 {meta['frames_written']} 帧 → {meta['image_dir']}"
            )
            results.append(meta)
            if args.run_min_rect:
                # 批量时每个任务单独跑；flat 时只在全部抽完后跑一次
                if batch or len(jobs) == 1:
                    run_min_rect(args, task_root)
        except Exception as exc:  # noqa: BLE001 — 批量时单条失败继续
            print(f"[ERROR] 失败：{job.url} → {exc}", file=sys.stderr)
            failures.append({"job": asdict(job), "error": str(exc)})
            if len(jobs) == 1:
                raise

    if args.run_min_rect and args.flat_output and len(jobs) > 1 and results:
        run_min_rect(args, output_root)

    manifest = {
        "output_root": str(output_root),
        "video_count": len(jobs),
        "success_count": len(results),
        "failure_count": len(failures),
        "video_role": args.video_role,
        "flat_output": bool(args.flat_output),
        "results": results,
        "failures": failures,
    }
    manifest_path = output_root / "video_frames_batch_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("=" * 72)
    print(
        f"[INFO] 全部结束：成功 {len(results)} / {len(jobs)}；失败 {len(failures)}"
    )
    print(f"[INFO] 输出根目录：{output_root}")
    print(f"[INFO] 批次清单：{manifest_path}")
    if batch and results:
        print("[INFO] 批量模式：每个视频是独立 DATASET_ROOT，例如：")
        print(f"  export DATASET_ROOT={results[0]['task_root']}")
        print("  bash run_images.sh")
    elif results:
        print(f"[INFO] DATASET_ROOT 可用：{output_root}")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
