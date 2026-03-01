from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipeline_quantize_vectorize import build_svg, contours_by_level, quantize_levels


class PipelineQuantizeVectorizeTests(unittest.TestCase):
    def test_quantize_levels_uses_expected_bucket_count_when_tones_exist(self) -> None:
        gray = np.arange(256, dtype=np.uint8).reshape(16, 16)
        quantized = quantize_levels(gray, levels=6)

        unique = np.unique(quantized)
        self.assertEqual(unique.size, 6)
        self.assertTrue(np.array_equal(unique, np.arange(6, dtype=np.uint8)))

    def test_quantize_levels_can_use_fewer_buckets_when_image_lacks_tones(self) -> None:
        gray = np.full((12, 12), 127, dtype=np.uint8)
        quantized = quantize_levels(gray, levels=6)

        unique = np.unique(quantized)
        self.assertLessEqual(unique.size, 6)
        self.assertGreaterEqual(unique.size, 1)

    def test_build_svg_is_deterministic_for_same_seed(self) -> None:
        contour = np.array([[[0, 0]], [[30, 0]], [[30, 20]], [[0, 20]]], dtype=np.int32)
        shapes = [{"level": 2, "contour": contour}]

        svg_a = build_svg(shapes=shapes, width=80, height=80, seed=42, arrangement="scatter")
        svg_b = build_svg(shapes=shapes, width=80, height=80, seed=42, arrangement="scatter")

        self.assertEqual(svg_a, svg_b)
        hash_a = hashlib.sha256(svg_a.encode("utf-8")).hexdigest()
        hash_b = hashlib.sha256(svg_b.encode("utf-8")).hexdigest()
        self.assertEqual(hash_a, hash_b)

    def test_build_svg_is_deterministic_with_shape_subset_and_target_fill(self) -> None:
        shapes = []
        for idx in range(10):
            x = idx * 6
            contour = np.array([[[x, 0]], [[x + 4, 0]], [[x + 4, 4]], [[x, 4]]], dtype=np.int32)
            shapes.append({"level": idx % 6, "contour": contour})

        svg_a = build_svg(
            shapes=shapes,
            width=120,
            height=120,
            seed=7,
            arrangement="scatter",
            max_shapes=4,
            target_fill_ratio=0.4,
        )
        svg_b = build_svg(
            shapes=shapes,
            width=120,
            height=120,
            seed=7,
            arrangement="scatter",
            max_shapes=4,
            target_fill_ratio=0.4,
        )
        self.assertEqual(svg_a, svg_b)

        # Confirm only requested number of paths were emitted.
        self.assertEqual(svg_a.count("<path "), 4)

    def test_build_svg_random_shape_count_range_is_seeded(self) -> None:
        shapes = []
        for idx in range(20):
            x = (idx % 10) * 8
            y = (idx // 10) * 8
            contour = np.array([[[x, y]], [[x + 4, y]], [[x + 4, y + 4]], [[x, y + 4]]], dtype=np.int32)
            shapes.append({"level": idx % 6, "contour": contour})

        svg_a = build_svg(
            shapes=shapes,
            width=200,
            height=120,
            seed=123,
            arrangement="scatter",
            shape_count_range=(4, 8),
        )
        svg_b = build_svg(
            shapes=shapes,
            width=200,
            height=120,
            seed=123,
            arrangement="scatter",
            shape_count_range=(4, 8),
        )
        self.assertEqual(svg_a, svg_b)
        emitted = svg_a.count("<path ")
        self.assertGreaterEqual(emitted, 4)
        self.assertLessEqual(emitted, 8)

    def test_build_svg_defaults_to_white_background(self) -> None:
        contour = np.array([[[0, 0]], [[10, 0]], [[10, 10]], [[0, 10]]], dtype=np.int32)
        svg = build_svg(
            shapes=[{"level": 0, "contour": contour}],
            width=40,
            height=40,
            seed=1,
            arrangement="scatter",
        )
        self.assertIn('fill="#ffffff"', svg)

    def test_contours_by_level_finds_shape_in_synthetic_image(self) -> None:
        quantized = np.zeros((64, 64), dtype=np.uint8)
        quantized[20:44, 18:46] = 3

        contours = contours_by_level(quantized, level=3)
        self.assertGreater(len(contours), 0)


if __name__ == "__main__":
    unittest.main()
