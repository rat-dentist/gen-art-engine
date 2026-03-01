from __future__ import annotations

import hashlib
import re
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

    def test_build_svg_can_embed_background_image(self) -> None:
        contour = np.array([[[0, 0]], [[10, 0]], [[10, 10]], [[0, 10]]], dtype=np.int32)
        data_url = "data:image/png;base64,AAAA"
        svg = build_svg(
            shapes=[{"level": 0, "contour": contour}],
            width=40,
            height=40,
            seed=2,
            arrangement="scatter",
            background_image_data_url=data_url,
            background_image_opacity=0.35,
        )
        self.assertIn(f'href="{data_url}"', svg)
        self.assertIn('preserveAspectRatio="xMidYMid slice"', svg)
        self.assertIn('opacity="0.35"', svg)

    def test_build_svg_segmented_tube_is_deterministic_and_repeats_one_shape(self) -> None:
        shapes = []
        for idx in range(6):
            size = 4 + idx
            x = idx * 5
            contour = np.array([[[x, 0]], [[x + size, 0]], [[x + size, size]], [[x, size]]], dtype=np.int32)
            shapes.append({"level": idx % 6, "contour": contour})

        svg_a = build_svg(
            shapes=shapes,
            width=240,
            height=160,
            seed=99,
            arrangement="segmented_tube",
            tube_segment_count_range=(12, 16),
        )
        svg_b = build_svg(
            shapes=shapes,
            width=240,
            height=160,
            seed=99,
            arrangement="segmented_tube",
            tube_segment_count_range=(12, 16),
        )
        self.assertEqual(svg_a, svg_b)

        emitted = svg_a.count("<path ")
        self.assertGreaterEqual(emitted, 12)
        self.assertLessEqual(emitted, 16)
        unique_path_data = set(re.findall(r'<path d="([^"]+)"', svg_a))
        self.assertEqual(len(unique_path_data), 1)
        self.assertIn('fill="#ffffff"', svg_a)
        self.assertIn('stroke="#000000"', svg_a)
        self.assertIn('stroke-opacity="1"', svg_a)
        self.assertIn("vector-effect=\"non-scaling-stroke\"", svg_a)
        stroke_widths = [float(value) for value in re.findall(r'stroke-width="([0-9.]+)"', svg_a)]
        self.assertGreater(len(stroke_widths), 0)
        self.assertTrue(all(abs(width - 0.30) < 1e-6 for width in stroke_widths))

    def test_build_svg_segmented_tube_keeps_segment_scale_large_enough(self) -> None:
        tiny_contour = np.array([[[0, 0]], [[2, 0]], [[2, 2]], [[0, 2]]], dtype=np.int32)
        svg = build_svg(
            shapes=[{"level": 0, "contour": tiny_contour}],
            width=300,
            height=200,
            seed=5,
            arrangement="segmented_tube",
            tube_segment_count_range=(8, 8),
        )
        scales = [float(value) for value in re.findall(r"scale\(([-0-9.]+)\)", svg)]
        self.assertEqual(len(scales), 8)
        self.assertGreaterEqual(min(scales), 6.0)

    def test_build_svg_layer_sequence_supports_reordering_and_multiple_layers(self) -> None:
        shapes = []
        for idx in range(20):
            x = (idx % 10) * 8
            y = (idx // 10) * 8
            contour = np.array([[[x, y]], [[x + 5, y]], [[x + 5, y + 5]], [[x, y + 5]]], dtype=np.int32)
            shapes.append({"level": idx % 6, "contour": contour})

        svg = build_svg(
            shapes=shapes,
            width=240,
            height=160,
            seed=222,
            arrangement="scatter",
            shape_count_range=(2, 2),
            tube_segment_count_range=(3, 3),
            layer_sequence=["segmented_tube", "scatter", "segmented_tube"],
        )
        self.assertEqual(svg.count("<path "), 8)
        self.assertEqual(svg.count('stroke="none"'), 2)
        self.assertEqual(svg.count('stroke="#000000"'), 6)
        stroke_order = re.findall(r'<path [^>]*stroke="([^"]+)"', svg)
        self.assertEqual(len(stroke_order), 8)
        self.assertEqual(stroke_order[:3], ["#000000", "#000000", "#000000"])
        self.assertEqual(stroke_order[3:5], ["none", "none"])
        self.assertEqual(stroke_order[5:], ["#000000", "#000000", "#000000"])

    def test_build_svg_layer_specs_control_tube_count_and_outline_width(self) -> None:
        shapes = []
        for idx in range(12):
            x = (idx % 6) * 9
            y = (idx // 6) * 9
            contour = np.array([[[x, y]], [[x + 6, y]], [[x + 6, y + 6]], [[x, y + 6]]], dtype=np.int32)
            shapes.append({"level": idx % 6, "contour": contour})

        svg = build_svg(
            shapes=shapes,
            width=260,
            height=180,
            seed=314,
            arrangement="scatter",
            layer_specs=[
                {"type": "segmented_tube", "tube_segment_count": 4, "tube_stroke_width": 0.35, "tube_straightness": 0.10},
                {"type": "segmented_tube", "tube_segment_count": 2, "tube_stroke_width": 1.10, "tube_straightness": 0.95},
            ],
        )
        self.assertEqual(svg.count("<path "), 6)
        stroke_widths = re.findall(r'stroke-width="([0-9.]+)"', svg)
        self.assertEqual(stroke_widths[:4], ["0.35", "0.35", "0.35", "0.35"])
        self.assertEqual(stroke_widths[4:], ["1.10", "1.10"])

    def test_build_svg_tube_repetitions_change_density_not_total_length(self) -> None:
        contour = np.array([[[0, 0]], [[14, 0]], [[14, 10]], [[0, 10]]], dtype=np.int32)
        shapes = [{"level": 2, "contour": contour}]

        svg_low = build_svg(
            shapes=shapes,
            width=280,
            height=180,
            seed=44,
            arrangement="segmented_tube",
            layer_specs=[{"type": "segmented_tube", "tube_segment_count": 10, "tube_straightness": 0.55}],
        )
        svg_high = build_svg(
            shapes=shapes,
            width=280,
            height=180,
            seed=44,
            arrangement="segmented_tube",
            layer_specs=[{"type": "segmented_tube", "tube_segment_count": 40, "tube_straightness": 0.55}],
        )

        self.assertEqual(svg_low.count("<path "), 10)
        self.assertEqual(svg_high.count("<path "), 40)

        def path_length(svg_text: str) -> float:
            points = [
                (float(x), float(y))
                for x, y in re.findall(r'translate\(([-0-9.]+) ([-0-9.]+)\)\s+rotate', svg_text)
            ]
            total = 0.0
            for idx in range(1, len(points)):
                dx = points[idx][0] - points[idx - 1][0]
                dy = points[idx][1] - points[idx - 1][1]
                total += (dx * dx + dy * dy) ** 0.5
            return total

        len_low = path_length(svg_low)
        len_high = path_length(svg_high)
        self.assertGreater(len_low, 0.0)
        self.assertGreater(len_high, 0.0)
        self.assertLess(abs(len_low - len_high), 12.0)

    def test_build_svg_tube_starts_inside_canvas(self) -> None:
        contour = np.array([[[0, 0]], [[12, 0]], [[12, 8]], [[0, 8]]], dtype=np.int32)
        width = 260
        height = 180
        svg = build_svg(
            shapes=[{"level": 1, "contour": contour}],
            width=width,
            height=height,
            seed=88,
            arrangement="segmented_tube",
            layer_specs=[{"type": "segmented_tube", "tube_segment_count": 20, "tube_straightness": 0.4}],
        )
        points = re.findall(r'translate\(([-0-9.]+) ([-0-9.]+)\)\s+rotate', svg)
        self.assertGreater(len(points), 0)
        first_x, first_y = float(points[0][0]), float(points[0][1])
        self.assertGreaterEqual(first_x, 0.0)
        self.assertLessEqual(first_x, float(width))
        self.assertGreaterEqual(first_y, 0.0)
        self.assertLessEqual(first_y, float(height))

    def test_build_svg_layer_specs_control_scatter_count_per_layer(self) -> None:
        shapes = []
        for idx in range(30):
            x = (idx % 10) * 10
            y = (idx // 10) * 10
            contour = np.array([[[x, y]], [[x + 6, y]], [[x + 6, y + 6]], [[x, y + 6]]], dtype=np.int32)
            shapes.append({"level": idx % 6, "contour": contour})

        svg = build_svg(
            shapes=shapes,
            width=320,
            height=220,
            seed=17,
            arrangement="scatter",
            layer_specs=[
                {"type": "scatter", "scatter_shape_count_min": 2, "scatter_shape_count_max": 2},
                {"type": "scatter", "scatter_shape_count_min": 3, "scatter_shape_count_max": 3},
            ],
        )
        self.assertEqual(svg.count("<path "), 5)

    def test_build_svg_trimmed_morph_tube_layer_is_seeded(self) -> None:
        shapes = []
        for idx in range(8):
            x = (idx % 4) * 12
            y = (idx // 4) * 12
            contour = np.array([[[x, y]], [[x + 8, y]], [[x + 8, y + 8]], [[x, y + 8]]], dtype=np.int32)
            shapes.append({"level": idx % 6, "contour": contour})

        layer = {
            "type": "trimmed_morph_tube",
            "tube_segment_count": 12,
            "tube_stroke_width": 0.80,
            "tube_straightness": 0.40,
            "tube_base_simplify": 0.6,
            "tube_ring_simplify": 0.4,
            "tube_piece_min_area": 6.0,
            "tube_morph_steps": 6,
            "tube_morph_shapes": 4,
            "tube_morph_points": 48,
        }
        svg_a = build_svg(
            shapes=shapes,
            width=260,
            height=180,
            seed=901,
            arrangement="scatter",
            layer_specs=[layer],
        )
        svg_b = build_svg(
            shapes=shapes,
            width=260,
            height=180,
            seed=901,
            arrangement="scatter",
            layer_specs=[layer],
        )

        self.assertEqual(svg_a, svg_b)
        self.assertGreater(svg_a.count("<path "), 0)
        self.assertIn('fill-rule="evenodd"', svg_a)

    def test_build_svg_layer_sequence_accepts_trimmed_morph_tube(self) -> None:
        contour = np.array([[[0, 0]], [[12, 0]], [[12, 8]], [[0, 8]]], dtype=np.int32)
        shapes = [{"level": 1, "contour": contour}]
        svg = build_svg(
            shapes=shapes,
            width=220,
            height=140,
            seed=77,
            arrangement="trimmed_morph_tube",
        )
        self.assertGreater(svg.count("<path "), 0)

    def test_contours_by_level_finds_shape_in_synthetic_image(self) -> None:
        quantized = np.zeros((64, 64), dtype=np.uint8)
        quantized[20:44, 18:46] = 3

        contours = contours_by_level(quantized, level=3)
        self.assertGreater(len(contours), 0)


if __name__ == "__main__":
    unittest.main()
