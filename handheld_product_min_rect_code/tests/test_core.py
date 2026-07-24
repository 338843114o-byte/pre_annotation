#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from add_handheld_product_min_rect import (  # noqa: E402
    Detection,
    append_shapes_preserving_original_text,
    build_sized_min_rectangle,
    build_sized_min_rectangles,
    cluster_boxes_by_aabb_overlap,
    collect_json_polygons,
    effective_rectangle_margin,
    flatten_product_units,
    format_rect_label,
    is_rect_label,
    polygons_intersect,
    viz_label_text,
    make_rectangle_shape,
    min_enclosing_rectangle,
    rect_passes_side_limit,
    rectangle_aabb_size,
    select_handheld_geometry,
    verify_only_shapes_appended,
    yolo_class_to_json_label,
    dedupe_overlapping_product_shapes,
)


def box(x1: float, y1: float, x2: float, y2: float):
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


class CoreTests(unittest.TestCase):
    def test_collect_auto_product_labels(self):
        shapes = [
            {"label": "罐装", "points": box(10, 10, 30, 50)},
            {"label": "罐装", "points": box(40, 10, 60, 50)},
            {"label": "售货柜", "points": box(0, 0, 100, 20)},
            {"label": "手", "points": box(20, 30, 70, 80)},
            {"label": "严重遮挡", "points": box(55, 25, 65, 45)},
        ]
        anchors, hands, auxiliary = collect_json_polygons(
            shapes,
            100,
            100,
            product_labels=set(),
            ignore_labels={"手", "售货柜", "严重遮挡", "最小外接矩形"},
            hand_labels={"手"},
            auxiliary_labels={"严重遮挡"},
            rect_label="最小外接矩形",
        )
        self.assertEqual(len(anchors), 2)
        self.assertEqual(len(hands), 1)
        self.assertEqual(len(auxiliary), 1)

    def test_collect_sku_prefix_product_labels(self):
        """袋装_SKU / 盒装_SKU 应被前缀匹配为商品锚点。"""
        shapes = [
            {"label": "袋装_大豫竹香辣牛肉面_52g", "points": box(10, 10, 30, 50)},
            {"label": "盒装_蒙牛酸酸乳原味_250ml", "points": box(40, 10, 60, 50)},
            {"label": "售货柜", "points": box(0, 0, 100, 20)},
            {"label": "手", "points": box(20, 30, 70, 80)},
        ]
        anchors, hands, auxiliary = collect_json_polygons(
            shapes,
            100,
            100,
            product_labels={"罐装", "瓶装", "袋装", "盒装", "桶装", "条装"},
            ignore_labels={"手", "头", "售货柜", "最小外接矩形"},
            hand_labels={"手"},
            auxiliary_labels={"遮挡"},
            rect_label="最小外接矩形",
        )
        self.assertEqual(len(anchors), 2)
        self.assertEqual(len(hands), 1)
        self.assertEqual(len(auxiliary), 0)

    def test_collect_severe_occlusion_as_product(self):
        """严重遮挡/过于模糊与未定义包装一样作为商品锚点，不再当辅助区域。"""
        shapes = [
            {"label": "瓶装", "points": box(10, 10, 30, 50)},
            {"label": "未定义包装", "points": box(40, 10, 60, 50)},
            {"label": "严重遮挡", "points": box(80, 10, 100, 40)},
            {"label": "过于模糊", "points": box(100, 50, 120, 80)},
            {"label": "遮挡", "points": box(5, 60, 15, 70)},
            {"label": "手", "points": box(20, 30, 70, 80)},
        ]
        anchors, hands, auxiliary = collect_json_polygons(
            shapes,
            140,
            100,
            product_labels={
                "罐装",
                "瓶装",
                "袋装",
                "盒装",
                "桶装",
                "条装",
                "未定义包装",
                "严重遮挡",
                "过于模糊",
            },
            ignore_labels={"手", "头", "售货柜", "最小外接矩形"},
            hand_labels={"手"},
            auxiliary_labels={"遮挡"},
            rect_label="最小外接矩形",
        )
        self.assertEqual(len(anchors), 4)
        self.assertEqual(len(hands), 1)
        self.assertEqual(len(auxiliary), 1)

    def test_yolo_match_and_json_fallback(self):
        anchors = [box(10, 10, 30, 40), box(60, 10, 80, 40)]
        detections = [
            Detection(box(9, 9, 31, 41), 0.9, 0, "product"),
        ]
        units, selected, unmatched = select_handheld_geometry(
            anchors,
            [],
            [],
            detections,
            width=100,
            height=100,
            min_iou=0.05,
            min_overlap=0.2,
            hand_expand_ratio=0.15,
            include_hand_matched=True,
        )
        self.assertEqual(selected, {0})
        self.assertEqual(unmatched, 1)
        self.assertEqual(len(units), 2)
        geometry = flatten_product_units(units)
        self.assertEqual(len(geometry), 3)
        self.assertIn(anchors[0], geometry)
        self.assertIn(anchors[1], geometry)
        self.assertIn(detections[0].points, geometry)

    def test_hand_adds_unmatched_yolo_detection(self):
        """无 JSON 商品锚点时，用手旁 YOLO 补商品。"""
        near_hand = box(50, 55, 75, 90)
        detections = [Detection(box(40, 40, 60, 70), 0.8, 0, "product")]
        units, selected, unmatched = select_handheld_geometry(
            [],
            [near_hand],
            [],
            detections,
            width=100,
            height=100,
            min_iou=0.05,
            min_overlap=0.2,
            hand_expand_ratio=0.15,
            include_hand_matched=True,
        )
        self.assertEqual(selected, {0})
        self.assertEqual(unmatched, 0)
        geometry = flatten_product_units(units)
        self.assertIn(detections[0].points, geometry)
        self.assertNotIn(near_hand, geometry)

    def test_json_plus_extra_hand_yolo(self):
        """无 JSON 商品的手仍可用 YOLO 补漏标；已压住 JSON 商品的手不再补检。"""
        anchor = box(10, 10, 40, 40)
        matched = Detection(box(11, 11, 39, 39), 0.9, 0, "can")
        extra_near_hand = Detection(box(55, 55, 80, 85), 0.85, 1, "bottle")
        orphan_hand = box(50, 50, 90, 90)  # 不碰 JSON 商品
        units, selected, unmatched = select_handheld_geometry(
            [anchor],
            [orphan_hand],
            [],
            [matched, extra_near_hand],
            width=100,
            height=100,
            min_iou=0.05,
            min_overlap=0.2,
            hand_expand_ratio=0.15,
            include_hand_matched=True,
        )
        self.assertEqual(unmatched, 0)
        self.assertEqual(selected, {0, 1})
        self.assertEqual(len(units), 2)
        geometry = flatten_product_units(units)
        self.assertIn(extra_near_hand.points, geometry)

        # 手已与 JSON 商品相交时，不再拉偏移 YOLO
        touching_hand = box(30, 30, 70, 70)
        bad_yolo = Detection(box(55, 55, 80, 85), 0.85, 1, "bottle")
        units2, selected2, _ = select_handheld_geometry(
            [anchor],
            [touching_hand],
            [],
            [matched, bad_yolo],
            width=100,
            height=100,
            min_iou=0.05,
            min_overlap=0.2,
            hand_expand_ratio=0.15,
            include_hand_matched=True,
        )
        self.assertEqual(len(units2), 1)
        self.assertNotIn(bad_yolo.points, flatten_product_units(units2))

    def test_hand_matched_yolo_dedup_against_existing_unit(self):
        """手旁补检若与已有 JSON/YOLO 单元重叠，不再新建商品单元。"""
        anchor = box(10, 10, 50, 50)
        matched = Detection(box(11, 11, 49, 49), 0.9, 0, "occluded")
        duplicate = Detection(box(12, 12, 48, 48), 0.95, 1, "can")
        hand = box(20, 20, 60, 60)
        units, selected, unmatched = select_handheld_geometry(
            [anchor],
            [hand],
            [],
            [matched, duplicate],
            width=100,
            height=100,
            min_iou=0.05,
            min_overlap=0.2,
            hand_expand_ratio=0.15,
            include_hand_matched=True,
        )
        self.assertEqual(unmatched, 0)
        self.assertEqual(selected, {0, 1})  # 重复框也被吞掉
        self.assertEqual(len(units), 1)  # 只有锚点单元，不另开
        self.assertEqual(len(units[0]), 2)  # 锚点 + 一个匹配 YOLO

    def test_hand_only_yolo_dedup_overlapping_detections(self):
        """手旁重叠 YOLO 多检只保留一个商品单元（避免 undefined_pack 双检成 _2）。"""
        hand = box(40, 40, 80, 80)
        high = Detection(box(45, 45, 75, 75), 0.9, 0, "undefined_pack")
        low = Detection(box(48, 48, 78, 78), 0.4, 1, "undefined_pack")
        units, selected, unmatched = select_handheld_geometry(
            [],
            [hand],
            [],
            [high, low],
            width=100,
            height=100,
            min_iou=0.05,
            min_overlap=0.2,
            hand_expand_ratio=0.15,
            include_hand_matched=True,
        )
        self.assertEqual(unmatched, 0)
        self.assertEqual(selected, {0, 1})
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0][0], high.points)

    def test_far_auxiliary_stays_separate_unit(self):
        """远处严重遮挡不硬挂到左侧商品，应单独成单元以便分框。"""
        left = box(10, 10, 40, 40)
        far_aux = box(200, 200, 230, 220)
        units, selected, unmatched = select_handheld_geometry(
            [left],
            [],
            [far_aux],
            [],
            width=300,
            height=300,
            min_iou=0.05,
            min_overlap=0.2,
            hand_expand_ratio=0.15,
            include_hand_matched=True,
        )
        self.assertEqual(unmatched, 1)
        self.assertEqual(selected, set())
        self.assertEqual(len(units), 2)
        self.assertEqual(units[0], [left])
        self.assertEqual(units[1], [far_aux])

        near_aux = box(35, 35, 55, 55)  # 与 left AABB 重叠
        units2, _, _ = select_handheld_geometry(
            [left],
            [],
            [near_aux],
            [],
            width=300,
            height=300,
            min_iou=0.05,
            min_overlap=0.2,
            hand_expand_ratio=0.15,
            include_hand_matched=True,
        )
        self.assertEqual(len(units2), 1)
        self.assertEqual(len(units2[0]), 2)
        self.assertIn(near_aux, units2[0])

    def test_cluster_overlap_is_transitive(self):
        """商品-商品重叠具有传递性；手不桥接商品，且须与商品直接重叠才纳入。"""
        product_a = box(10, 10, 40, 40)
        product_b = box(35, 10, 65, 40)  # 与 A 重叠
        product_c = box(60, 10, 90, 40)  # 与 B 重叠 → 应与 A 同簇
        far = box(150, 10, 180, 40)
        clusters = cluster_boxes_by_aabb_overlap(
            [product_a, product_b, product_c, far],
            [],
        )
        self.assertEqual(len(clusters), 2)
        sizes = sorted(len(c["products"]) for c in clusters)
        self.assertEqual(sizes, [1, 3])

        hand = box(20, 35, 50, 70)  # 与 A 重叠
        product_d = box(100, 100, 130, 130)
        hand2 = box(120, 120, 150, 150)  # 与 D 重叠
        clusters2 = cluster_boxes_by_aabb_overlap(
            [product_a, product_d],
            [hand, hand2],
        )
        self.assertEqual(len(clusters2), 2)
        for cluster in clusters2:
            self.assertEqual(len(cluster["products"]), 1)
            self.assertEqual(len(cluster["hands"]), 1)

        # 手与手重叠、但不与任何商品重叠 → 不纳入
        touching_hand = box(15, 35, 45, 65)  # 与 A 重叠
        only_touch_hand = box(15, 60, 45, 90)  # 只与 touching_hand 重叠，不碰商品
        clusters3 = cluster_boxes_by_aabb_overlap(
            [product_a],
            [touching_hand, only_touch_hand],
        )
        self.assertEqual(len(clusters3), 1)
        self.assertEqual(len(clusters3[0]["hands"]), 1)

        # 手桥接两个不相交商品 → 商品仍分两类，手可挂到两类
        left = box(10, 10, 30, 30)
        right = box(80, 10, 100, 30)
        bridge_hand = box(20, 5, 90, 25)  # 同时叠左右商品
        clusters4 = cluster_boxes_by_aabb_overlap([left, right], [bridge_hand])
        self.assertEqual(len(clusters4), 2)
        self.assertTrue(all(len(c["hands"]) == 1 for c in clusters4))

    def test_hand_aabb_false_overlap_not_counted(self):
        """斜框 AABB 虚高相交、多边形不相交时，手不纳入。"""
        # 近似 0.jpg：上手与瓶 AABB 相交，真实多边形分离
        bottle = [[903, 415], [936, 443], [829, 571], [796, 543]]
        upper_hand = [[810, 364], [866, 420], [963, 322], [907, 267]]
        ba = np.asarray(bottle, dtype=np.float64)
        ha = np.asarray(upper_hand, dtype=np.float64)
        # AABB 相交
        self.assertFalse(
            ba[:, 0].max() < ha[:, 0].min()
            or ha[:, 0].max() < ba[:, 0].min()
            or ba[:, 1].max() < ha[:, 1].min()
            or ha[:, 1].max() < ba[:, 1].min()
        )
        self.assertFalse(polygons_intersect(bottle, upper_hand))
        clusters = cluster_boxes_by_aabb_overlap([bottle], [upper_hand])
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0]["hands"], [])

    def test_lonely_hand_dropped(self):
        product = box(10, 10, 30, 30)
        lonely_hand = box(80, 80, 100, 100)
        clusters = cluster_boxes_by_aabb_overlap([product], [lonely_hand])
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0]["hands"], [])

    def test_side_limit_compares_long_to_long_short_to_short(self):
        """矩形 OBB 较长边对原图较长边，较短边对原图较短边。"""
        # 原图 1000x500 → 长边上限 200，短边上限 100
        tall = box(10, 10, 100, 190)  # OBB 90x180
        rect = min_enclosing_rectangle(
            [tall], width=1000, height=500, mode="axis_aligned", margin=0, margin_ratio=0
        )
        self.assertTrue(
            rect_passes_side_limit(
                rect, image_width=1000, image_height=500, max_side_ratio=0.2
            )
        )
        fat = box(10, 10, 160, 190)  # OBB 150x180 → 短边 150>100
        rect_fat = min_enclosing_rectangle(
            [fat], width=1000, height=500, mode="axis_aligned", margin=0, margin_ratio=0
        )
        self.assertFalse(
            rect_passes_side_limit(
                rect_fat, image_width=1000, image_height=500, max_side_ratio=0.2
            )
        )

    def test_obb_side_limit_not_aabb(self):
        """斜框用 OBB 边长判定，不被 AABB 虚高卡住。"""
        # 原图 1280x720，上限 256 / 144
        # 显式旋转矩形：OBB=200x100 可通过；AABB 被斜向撑大后短边超限
        import cv2
        from add_handheld_product_min_rect import rectangle_obb_size, rectangle_aabb_size

        rect = cv2.boxPoints(((640.0, 360.0), (200.0, 100.0), 30.0))
        rw, rh = rectangle_aabb_size(rect)
        oa, ob = rectangle_obb_size(rect)
        self.assertGreater(min(rw, rh), 144)
        self.assertTrue(
            rect_passes_side_limit(
                rect, image_width=1280, image_height=720, max_side_ratio=0.2
            )
        )
        self.assertLessEqual(max(oa, ob), 256 + 1e-3)
        self.assertLessEqual(min(oa, ob), 144 + 1e-3)

    def test_pass_cluster_all_within_side_limit(self):
        """2.1：商品+重叠手 OBB 边长通过 → 一个框。"""
        product = box(40, 40, 70, 80)
        hand = box(55, 70, 85, 100)
        rect, policy = build_sized_min_rectangle(
            [product],
            [hand],
            width=500,
            height=500,
            mode="axis_aligned",
            margin=12,
            margin_ratio=0,
            max_side_ratio=0.2,
        )
        self.assertEqual(policy, "cluster_all")
        self.assertTrue(
            rect_passes_side_limit(
                rect, image_width=500, image_height=500, max_side_ratio=0.2
            )
        )
        rw, rh = rectangle_aabb_size(rect)
        self.assertGreater(rw, 70 - 40)  # 含边距应大于商品本身

    def test_shrink_drop_hand_then_products(self):
        """2.1 超限后去手仍可通过。"""
        # 图 200x200，阈值边长 40。商品小，大手使整体超限。
        product = box(10, 10, 30, 30)  # 20x20 + margin12 → 约 44，可能刚过
        # 用更大图使仅商品通过、含手不通过
        product = box(20, 20, 50, 50)  # 30 + 24 = 54
        huge_hand = box(20, 20, 180, 180)
        rect, policy = build_sized_min_rectangle(
            [product],
            [huge_hand],
            width=400,
            height=400,
            mode="axis_aligned",
            margin=12,
            margin_ratio=0,
            max_side_ratio=0.2,  # limit=80
        )
        # 商品+手远超；仅商品 54 <= 80 → cluster_products
        self.assertEqual(policy, "cluster_products")
        self.assertTrue(
            rect_passes_side_limit(
                rect, image_width=400, image_height=400, max_side_ratio=0.2
            )
        )

    def test_split_to_singles_and_exception(self):
        """同类多商品合在一起超限 → 逐商品；过大单品走例外。"""
        # 两商品重叠同簇，合在一起很大；各自较小可通过
        a = box(10, 10, 70, 70)  # 60+24=84
        b = box(60, 10, 120, 70)  # 重叠 → 同簇，并集约 110+24
        results = build_sized_min_rectangles(
            [[a], [b]],
            [],
            width=500,
            height=500,
            mode="axis_aligned",
            margin=12,
            margin_ratio=0,
            max_side_ratio=0.2,  # limit=100
        )
        # 合并超限，拆成两个 single
        self.assertEqual(len(results), 2)
        for _rect, policy, count in results:
            self.assertIn(policy, {"single_product", "cluster_products"})
            self.assertEqual(count, 1)

        # 单商品本身就超大 → 例外
        big = box(5, 5, 400, 400)
        rect_ex, policy_ex = build_sized_min_rectangle(
            [big],
            [],
            width=500,
            height=500,
            mode="axis_aligned",
            margin=12,
            margin_ratio=0,
            max_side_ratio=0.2,
        )
        self.assertEqual(policy_ex, "single_product_exception")
        arr = np.asarray(rect_ex, dtype=np.float32)
        self.assertLessEqual(arr[:, 0].min(), 5.0)

    def test_single_drop_hand_before_exception(self):
        """拆成单品后含手超限 → 先去手；仅当仅商品仍超才例外，且例外不含手。"""
        # 同簇两商品被迫拆分；其中 A+大手超限，仅 A 可通过
        a = box(10, 10, 70, 70)  # 60+24=84 <= 100
        b = box(60, 10, 120, 70)
        huge_hand = box(5, 5, 200, 200)  # 与 a 相交，含手必超限
        results = build_sized_min_rectangles(
            [[a], [b]],
            [huge_hand],
            width=500,
            height=500,
            mode="axis_aligned",
            margin=12,
            margin_ratio=0,
            max_side_ratio=0.2,  # limit=100
        )
        self.assertGreaterEqual(len(results), 2)
        policies = {policy for _r, policy, _c in results}
        self.assertIn("single_product_drop_hand", policies)
        self.assertNotIn("single_product_exception", policies)
        for rect, policy, _count in results:
            if policy == "single_product_drop_hand":
                arr = np.asarray(rect, dtype=np.float32)
                # 去手后不应再包住大手右下角
                self.assertLess(float(arr[:, 0].max()), 150.0)

        # 仅商品本身就超大：例外框必须是商品-only（不含手扩展）
        big = box(5, 5, 400, 400)
        hand = box(1, 1, 490, 490)
        rect_ex, policy_ex = build_sized_min_rectangle(
            [big],
            [hand],
            width=500,
            height=500,
            mode="axis_aligned",
            margin=12,
            margin_ratio=0,
            max_side_ratio=0.2,
        )
        self.assertEqual(policy_ex, "single_product_exception")
        arr = np.asarray(rect_ex, dtype=np.float32)
        self.assertLess(float(arr[:, 0].max()), 450.0)

    def test_two_separate_clusters_two_rects(self):
        """不相交的两类各自生成框。"""
        left = box(10, 10, 40, 40)
        right = box(200, 200, 230, 230)
        results = build_sized_min_rectangles(
            [[left], [right]],
            [],
            width=500,
            height=500,
            mode="axis_aligned",
            margin=12,
            margin_ratio=0,
            max_side_ratio=0.2,
        )
        self.assertEqual(len(results), 2)
        for _rect, _policy, count in results:
            self.assertEqual(count, 1)

    def test_cluster_label_product_count(self):
        """同簇多商品合框时返回商品数量；label 带数量后缀。"""
        a = box(40, 40, 70, 70)
        b = box(55, 55, 85, 85)
        hand = box(50, 60, 90, 100)
        results = build_sized_min_rectangles(
            [[a], [b]],
            [hand],
            width=500,
            height=500,
            mode="axis_aligned",
            margin=12,
            margin_ratio=0,
            max_side_ratio=0.5,
        )
        self.assertEqual(len(results), 1)
        _rect, policy, count = results[0]
        self.assertEqual(policy, "cluster_all")
        self.assertEqual(count, 2)
        self.assertEqual(format_rect_label("最小外接矩形", count), "最小外接矩形_2")
        self.assertTrue(is_rect_label("最小外接矩形_2", "最小外接矩形"))
        self.assertTrue(is_rect_label("最小外接矩形", "最小外接矩形"))
        self.assertFalse(is_rect_label("罐装_2", "最小外接矩形"))
        self.assertFalse(
            is_rect_label("最小外接矩形（仅物品）_2", "最小外接矩形")
        )
        self.assertTrue(
            is_rect_label("最小外接矩形（仅物品）_2", "最小外接矩形（仅物品）")
        )
        self.assertEqual(
            format_rect_label("最小外接矩形（仅物品）", count),
            "最小外接矩形（仅物品）_2",
        )
        self.assertEqual(viz_label_text("最小外接矩形_2"), "min_rect_2")
        self.assertEqual(viz_label_text("最小外接矩形_1"), "min_rect_1")
        self.assertEqual(viz_label_text("最小外接矩形"), "min_rect")
        self.assertEqual(
            viz_label_text("最小外接矩形（仅物品）_2"), "min_rect_item_2"
        )
        self.assertEqual(
            viz_label_text("最小外接矩形（仅物品）"), "min_rect_item"
        )
        self.assertEqual(viz_label_text("瓶装"), "bottle")

    def test_product_only_rect_never_includes_hand(self):
        """仅物品框：与含手路径同聚类，但矩形永不纳入手。"""
        product = box(40, 40, 80, 100)
        hand = box(50, 90, 110, 160)
        with_hands = build_sized_min_rectangles(
            [[product]],
            [hand],
            width=500,
            height=500,
            mode="axis_aligned",
            margin=12,
            margin_ratio=0,
            max_side_ratio=0.5,
            include_contact_hands=True,
        )
        product_only = build_sized_min_rectangles(
            [[product]],
            [hand],
            width=500,
            height=500,
            mode="axis_aligned",
            margin=12,
            margin_ratio=0,
            max_side_ratio=0.5,
            include_contact_hands=False,
        )
        self.assertEqual(len(with_hands), 1)
        self.assertEqual(len(product_only), 1)
        rect_with, policy_with, count_with = with_hands[0]
        rect_only, policy_only, count_only = product_only[0]
        self.assertEqual(count_with, 1)
        self.assertEqual(count_only, 1)
        self.assertEqual(policy_with, "cluster_all")
        self.assertEqual(policy_only, "cluster_products")
        # 含手框应更大（手在商品下方外扩）
        ys_with = [p[1] for p in rect_with]
        ys_only = [p[1] for p in rect_only]
        self.assertGreater(max(ys_with), max(ys_only))
        # 仅物品框不应覆盖手底部
        self.assertLess(max(ys_only), 160)

    def test_margin_keeps_rectangle_away_from_product(self):
        polygons = [box(40, 40, 80, 100)]
        tight = min_enclosing_rectangle(
            polygons, width=200, height=200, mode="axis_aligned", margin=0, margin_ratio=0
        )
        padded = min_enclosing_rectangle(
            polygons,
            width=200,
            height=200,
            mode="axis_aligned",
            margin=12,
            margin_ratio=0,
        )
        tight_arr = np.asarray(tight, dtype=np.float32)
        padded_arr = np.asarray(padded, dtype=np.float32)
        tight_w = tight_arr[:, 0].max() - tight_arr[:, 0].min()
        padded_w = padded_arr[:, 0].max() - padded_arr[:, 0].min()
        self.assertGreater(padded_w, tight_w + 10)
        applied = effective_rectangle_margin(polygons, margin=12, margin_ratio=0)
        self.assertAlmostEqual(applied, 12.0)

    def test_reject_oversized_vending_machine_match(self):
        anchors = [box(50, 50, 70, 80)]
        detections = [
            Detection(box(0, 0, 200, 120), 0.95, 1, "vending_machine"),
            Detection(box(48, 48, 72, 82), 0.90, 3, "can"),
        ]
        from add_handheld_product_min_rect import filter_product_detections

        filtered = filter_product_detections(
            detections,
            product_classes=None,
            ignore_class_names={"vending_machine", "phone"},
        )
        self.assertEqual([d.class_name for d in filtered], ["can"])
        units, selected, unmatched = select_handheld_geometry(
            anchors,
            [],
            [],
            filtered,
            width=200,
            height=200,
            min_iou=0.05,
            min_overlap=0.2,
            hand_expand_ratio=0.15,
            include_hand_matched=True,
            max_area_ratio=8.0,
        )
        self.assertEqual(selected, {0})
        self.assertEqual(unmatched, 0)
        geometry = flatten_product_units(units)
        self.assertIn(anchors[0], geometry)
        self.assertIn(filtered[0].points, geometry)
        units2, selected2, _ = select_handheld_geometry(
            anchors,
            [],
            [],
            detections,
            width=200,
            height=200,
            min_iou=0.05,
            min_overlap=0.2,
            hand_expand_ratio=0.15,
            include_hand_matched=True,
            max_area_ratio=8.0,
        )
        self.assertEqual(selected2, {1})
        geometry2 = flatten_product_units(units2)
        self.assertNotIn(detections[0].points, geometry2)

    def test_min_area_rectangle_contains_all_points(self):
        polygons = [box(10, 10, 30, 40), box(60, 25, 80, 55)]
        rectangle = min_enclosing_rectangle(
            polygons,
            width=100,
            height=100,
            mode="min_area",
            margin=0,
        )
        contour = np.asarray(rectangle, dtype=np.float32)
        for polygon in polygons:
            for point in polygon:
                distance = cv2.pointPolygonTest(contour, tuple(point), True)
                self.assertGreaterEqual(distance, -0.01)

    def test_border_touch_rebuilds_axis_aligned_no_margin_on_border(self):
        """加 margin 后越左边界 → 左边贴 x=0 且无 margin，其余边保留 margin。"""
        polygons = [box(2, 40, 40, 80)]
        margin = 12.0
        rectangle = min_enclosing_rectangle(
            polygons,
            width=200,
            height=200,
            mode="min_area",
            margin=margin,
        )
        arr = np.asarray(rectangle, dtype=np.float64)
        xs, ys = arr[:, 0], arr[:, 1]
        self.assertAlmostEqual(float(xs.min()), 0.0, delta=0.05)
        # 轴对齐：两边平行坐标轴
        unique_x = sorted(set(round(float(v), 3) for v in xs))
        unique_y = sorted(set(round(float(v), 3) for v in ys))
        self.assertEqual(len(unique_x), 2)
        self.assertEqual(len(unique_y), 2)
        # 右边 / 上下仍有 margin（内容 2..40, 40..80）
        self.assertAlmostEqual(float(xs.max()), 40.0 + margin, delta=0.5)
        self.assertAlmostEqual(float(ys.min()), 40.0 - margin, delta=0.5)
        self.assertAlmostEqual(float(ys.max()), 80.0 + margin, delta=0.5)

    def test_border_touch_multiple_sides(self):
        """同时越左+下 → 两边都作为矩形边。"""
        polygons = [box(2, 180, 50, 198)]
        margin = 12.0
        rectangle = min_enclosing_rectangle(
            polygons,
            width=200,
            height=200,
            mode="min_area",
            margin=margin,
        )
        arr = np.asarray(rectangle, dtype=np.float64)
        self.assertAlmostEqual(float(arr[:, 0].min()), 0.0, delta=0.05)
        self.assertAlmostEqual(float(arr[:, 1].max()), 199.0, delta=0.05)
        # 右/上仍有 margin
        self.assertAlmostEqual(float(arr[:, 0].max()), 50.0 + margin, delta=0.5)
        self.assertAlmostEqual(float(arr[:, 1].min()), 180.0 - margin, delta=0.5)

    def test_interior_rotated_rect_keeps_right_angles(self):
        """完全在画布内的斜框保持旋转直角，不因 clamp 变形。"""
        polygons = [box(60, 50, 120, 90), box(90, 70, 140, 110)]
        rectangle = min_enclosing_rectangle(
            polygons,
            width=220,
            height=200,
            mode="min_area",
            margin=8,
        )
        arr = np.asarray(rectangle, dtype=np.float64)
        self.assertTrue(np.all(arr[:, 0] >= -1e-3) and np.all(arr[:, 0] <= 219 + 1e-3))
        self.assertTrue(np.all(arr[:, 1] >= -1e-3) and np.all(arr[:, 1] <= 199 + 1e-3))
        for k in range(4):
            v1 = arr[(k + 1) % 4] - arr[k]
            v2 = arr[(k + 2) % 4] - arr[(k + 1) % 4]
            cos = float(
                np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
            )
            ang = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
            self.assertAlmostEqual(ang, 90.0, delta=1.0)

    def test_only_one_shape_is_appended(self):
        original_data = {
            "version": "3.3.1",
            "flags": {},
            "shapes": [
                {
                    "kie_linking": [],
                    "direction": 0.0,
                    "label": "罐装",
                    "score": None,
                    "points": box(10, 10, 30, 40),
                    "group_id": None,
                    "description": "",
                    "difficult": False,
                    "shape_type": "rotation",
                    "flags": {},
                    "attributes": {},
                }
            ],
            "imagePath": "a.jpg",
            "imageData": None,
            "imageHeight": 100,
            "imageWidth": 100,
            "description": "",
        }
        original_snapshot = copy.deepcopy(original_data)
        original_text = json.dumps(original_data, ensure_ascii=False, indent=4) + "\n"
        template = original_data["shapes"][0]
        new_shape = make_rectangle_shape(
            box(5, 5, 35, 45), format_rect_label("最小外接矩形", 1), template
        )
        updated_text = append_shapes_preserving_original_text(original_text, [new_shape])
        verify_only_shapes_appended(original_data, updated_text, [new_shape])
        updated_data = json.loads(updated_text)
        self.assertEqual(original_data, original_snapshot)
        self.assertEqual(updated_data["shapes"][:-1], original_snapshot["shapes"])
        self.assertEqual(len(updated_data["shapes"]), 2)
        self.assertEqual(updated_data["shapes"][-1]["label"], "最小外接矩形_1")

    def test_crlf_original_text_survives(self):
        original = '{\r\n  "shapes": [],\r\n  "imagePath": "a.jpg"\r\n}\r\n'
        shape = {"label": "最小外接矩形", "points": box(1, 2, 3, 4)}
        updated = append_shapes_preserving_original_text(original, [shape])
        self.assertNotIn("\n", updated.replace("\r\n", ""))
        parsed = json.loads(updated)
        self.assertEqual(parsed["shapes"], [shape])

    def test_yolo_class_to_json_label(self):
        self.assertEqual(yolo_class_to_json_label("bottle"), "瓶装")
        self.assertEqual(yolo_class_to_json_label("hand"), "手")
        self.assertEqual(yolo_class_to_json_label("undefined_pack"), "未定义包装")
        self.assertEqual(yolo_class_to_json_label("vending_machine"), "售货柜")
        self.assertEqual(yolo_class_to_json_label("unknown_xyz"), "unknown_xyz")

    def test_yolo_mode_empty_anchors_only_hand_matched(self):
        """无 JSON 商品锚点时，只保留与手相交的 YOLO 商品，不吞掉远处货架框。"""
        hands = [box(20, 20, 50, 50)]
        detections = [
            Detection(points=box(25, 25, 45, 45), score=0.9, class_id=2, class_name="bottle"),
            Detection(points=box(200, 200, 250, 250), score=0.95, class_id=2, class_name="bottle"),
        ]
        units, selected, unmatched = select_handheld_geometry(
            anchors=[],
            hands=hands,
            auxiliary=[],
            detections=detections,
            width=400,
            height=400,
            min_iou=0.05,
            min_overlap=0.2,
            hand_expand_ratio=0.15,
            include_hand_matched=True,
        )
        self.assertEqual(unmatched, 0)
        self.assertEqual(len(units), 1)
        self.assertEqual(selected, {0})

    def test_dedupe_overlapping_product_shapes(self):
        """重叠的未定义包装多检只保留高分框；手与售货柜保留。"""
        shapes = [
            {"label": "手", "points": box(10, 10, 40, 40), "score": 0.9},
            {"label": "售货柜", "points": box(0, 0, 200, 50), "score": 0.8},
            {
                "label": "未定义包装",
                "points": box(50, 50, 100, 120),
                "score": 0.49,
            },
            {
                "label": "未定义包装",
                "points": box(55, 45, 105, 125),
                "score": 0.39,
            },
            {"label": "瓶装", "points": box(180, 180, 220, 240), "score": 0.9},
        ]
        deduped = dedupe_overlapping_product_shapes(
            shapes,
            product_labels={"未定义包装", "瓶装", "信息不足"},
            ignore_labels={"手", "售货柜", "最小外接矩形"},
            hand_labels={"手"},
            auxiliary_labels={"遮挡"},
            rect_label="最小外接矩形",
            min_iou=0.05,
            min_overlap=0.20,
            max_area_ratio=8.0,
        )
        product_labels = [
            str(s["label"]) for s in deduped if s["label"] in {"未定义包装", "瓶装"}
        ]
        self.assertEqual(product_labels.count("未定义包装"), 1)
        self.assertEqual(product_labels.count("瓶装"), 1)
        kept_undefined = next(s for s in deduped if s["label"] == "未定义包装")
        self.assertAlmostEqual(float(kept_undefined["score"]), 0.49)
        self.assertEqual(
            [s["label"] for s in deduped if s["label"] in {"手", "售货柜"}],
            ["手", "售货柜"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
