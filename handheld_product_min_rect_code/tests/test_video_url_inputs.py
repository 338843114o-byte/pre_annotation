#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from video_url_to_frames import (  # noqa: E402
    assign_job_names,
    dedupe_jobs,
    extract_urls_from_cell,
    filter_jobs_by_role,
    infer_video_role,
    load_jobs_from_txt,
    load_jobs_from_url_file,
    load_jobs_from_xlsx,
)


SAMPLE_DIR = Path("/home/data_manager/jiangfan/vedio_url")


class VideoUrlInputTests(unittest.TestCase):
    def test_extract_urls_from_json_array_cell(self):
        cell = json.dumps(
            [
                "https://example.com/a_main.mp4",
                "https://example.com/a_sub.mp4",
            ]
        )
        urls = extract_urls_from_cell(cell)
        self.assertEqual(len(urls), 2)
        self.assertTrue(urls[0].endswith("_main.mp4"))

    def test_infer_role(self):
        self.assertEqual(infer_video_role("https://x/a_main.mp4"), "main")
        self.assertEqual(infer_video_role("https://x/a_sub.mp4"), "sub")
        self.assertEqual(infer_video_role("https://x/other.mp4"), "unknown")

    def test_txt_one_url_per_line(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "urls.txt"
            path.write_text(
                "# comment\n"
                "https://example.com/a_main.mp4\n"
                "\n"
                "https://example.com/a_sub.mp4\n",
                encoding="utf-8",
            )
            jobs = load_jobs_from_txt(path)
            self.assertEqual(len(jobs), 2)
            self.assertEqual(jobs[0].role, "main")

    def test_xlsx_plain_url_column(self):
        try:
            import openpyxl
        except ImportError:
            self.skipTest("openpyxl missing")
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "plain.xlsx"
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "视频URL"
            ws.append(["视频URL"])
            ws.append(["https://example.com/b_main.mp4"])
            ws.append(["https://example.com/b_sub.mp4"])
            wb.save(path)
            jobs = load_jobs_from_xlsx(path)
            self.assertEqual(len(jobs), 2)
            self.assertEqual(jobs[1].role, "sub")

    def test_xlsx_order_video_json_column(self):
        try:
            import openpyxl
        except ImportError:
            self.skipTest("openpyxl missing")
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "orders.xlsx"
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "通过订单号查询订单"
            ws.append(["订单编号", "订单视频", "门号"])
            ws.append(
                [
                    "ORD1",
                    json.dumps(
                        [
                            "https://example.com/c_main.mp4",
                            "https://example.com/c_sub.mp4",
                        ]
                    ),
                    "A",
                ]
            )
            wb.save(path)
            jobs = load_jobs_from_xlsx(path)
            self.assertEqual(len(jobs), 2)
            self.assertEqual(jobs[0].order_id, "ORD1")
            named = assign_job_names(jobs)
            self.assertTrue(named[0].job_name.startswith("ORD1_"))

    def test_role_filter_and_dedupe(self):
        from video_url_to_frames import VideoJob

        jobs = [
            VideoJob(url="https://x/a_main.mp4", role="main"),
            VideoJob(url="https://x/a_sub.mp4", role="sub"),
            VideoJob(url="https://x/a_main.mp4", role="main"),
        ]
        filtered = filter_jobs_by_role(jobs, "main")
        self.assertEqual(len(filtered), 2)
        self.assertEqual(len(dedupe_jobs(filtered)), 1)

    @unittest.skipUnless(SAMPLE_DIR.is_dir(), "vedio_url sample dir missing")
    def test_real_vedio_url_samples(self):
        txt = SAMPLE_DIR / "一行一个URL.txt"
        xlsx_plain = SAMPLE_DIR / "一行一个URL.xlsx"
        xlsx_order = SAMPLE_DIR / "重叠订单_查询结果.xlsx"
        if not txt.is_file():
            self.skipTest("txt sample missing")
        txt_jobs = load_jobs_from_url_file(txt)
        self.assertGreaterEqual(len(txt_jobs), 100)
        if xlsx_plain.is_file():
            plain_jobs = load_jobs_from_url_file(xlsx_plain)
            self.assertEqual(len(plain_jobs), len(txt_jobs))
        if xlsx_order.is_file():
            order_jobs = load_jobs_from_url_file(xlsx_order)
            self.assertGreaterEqual(len(order_jobs), 50)
            self.assertTrue(any(j.order_id for j in order_jobs))
            mains = filter_jobs_by_role(order_jobs, "main")
            self.assertGreater(len(mains), 0)
            self.assertTrue(all(j.role == "main" for j in mains))


if __name__ == "__main__":
    unittest.main()
