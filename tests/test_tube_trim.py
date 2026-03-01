from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tube_trim import (
    MaskPiece,
    build_all_to_all_morph_bank,
    generate_segmented_tube_contours,
    labels_to_single_boundary_mask,
    owned_boundary_masks_for_pieces,
    trim_overlapping_contours,
)


def _rect_contour(x: int, y: int, width: int, height: int) -> np.ndarray:
    return np.array(
        [
            [[x, y]],
            [[x + width, y]],
            [[x + width, y + height]],
            [[x, y + height]],
        ],
        dtype=np.float32,
    )


def _blit_piece(canvas: np.ndarray, piece) -> None:
    height, width = canvas.shape
    piece_h, piece_w = piece.mask.shape
    x0 = int(piece.offset_x)
    y0 = int(piece.offset_y)
    x1 = x0 + piece_w
    y1 = y0 + piece_h

    dx0 = max(0, x0)
    dy0 = max(0, y0)
    dx1 = min(width, x1)
    dy1 = min(height, y1)
    if dx0 >= dx1 or dy0 >= dy1:
        return

    sx0 = dx0 - x0
    sy0 = dy0 - y0
    sx1 = sx0 + (dx1 - dx0)
    sy1 = sy0 + (dy1 - dy0)

    patch = (piece.mask[sy0:sy1, sx0:sx1] > 0).astype(np.uint8)
    region = canvas[dy0:dy1, dx0:dx1]
    np.maximum(region, patch, out=region)


def _union_mask_from_contours(contours: list[np.ndarray], width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for contour in contours:
        polygon = np.rint(contour.reshape(-1, 2)).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [polygon], 1)
    return mask


class TubeTrimTests(unittest.TestCase):
    def test_generate_segmented_tube_contours_is_seeded(self) -> None:
        base = _rect_contour(0, 0, 20, 12)
        contours_a = generate_segmented_tube_contours(
            base_contour=base,
            width=240,
            height=160,
            seed=123,
            segment_count=18,
            straightness=0.55,
        )
        contours_b = generate_segmented_tube_contours(
            base_contour=base,
            width=240,
            height=160,
            seed=123,
            segment_count=18,
            straightness=0.55,
        )
        self.assertEqual(len(contours_a), 18)
        self.assertEqual(len(contours_b), 18)
        for contour_a, contour_b in zip(contours_a, contours_b):
            self.assertTrue(np.allclose(contour_a, contour_b))

    def test_generate_segmented_tube_contours_can_simplify_base_shape(self) -> None:
        points = []
        for x in range(0, 50):
            points.append([x, 0])
        for y in range(1, 30):
            points.append([49, y])
        for x in range(48, -1, -1):
            points.append([x, 29])
        for y in range(28, 0, -1):
            points.append([0, y])
        base = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)

        unsimplified = generate_segmented_tube_contours(
            base_contour=base,
            width=220,
            height=180,
            seed=5,
            segment_count=1,
            straightness=0.4,
            contour_simplify_eps=0.0,
        )
        simplified = generate_segmented_tube_contours(
            base_contour=base,
            width=220,
            height=180,
            seed=5,
            segment_count=1,
            straightness=0.4,
            contour_simplify_eps=1.0,
        )

        self.assertEqual(len(unsimplified), 1)
        self.assertEqual(len(simplified), 1)
        self.assertGreater(unsimplified[0].shape[0], simplified[0].shape[0])

    def test_build_all_to_all_morph_bank_expected_count(self) -> None:
        a = _rect_contour(0, 0, 20, 10)
        b = _rect_contour(2, 2, 14, 16)
        c = _rect_contour(4, 1, 10, 20)

        bank = build_all_to_all_morph_bank(
            contours=[a, b, c],
            steps=6,
            point_count=48,
            simplify_eps=0.0,
        )
        expected = 3 * 2 * (6 + 2)
        self.assertEqual(len(bank), expected)
        self.assertTrue(all(contour.shape[0] == 48 for contour in bank))

    def test_generate_segmented_tube_contours_with_bank_is_seeded(self) -> None:
        a = _rect_contour(0, 0, 20, 10)
        b = _rect_contour(0, 0, 12, 18)
        bank = build_all_to_all_morph_bank([a, b], steps=2, point_count=24, simplify_eps=0.0)

        contours_a = generate_segmented_tube_contours(
            base_contour=a,
            contour_bank=bank,
            width=220,
            height=160,
            seed=77,
            segment_count=10,
            straightness=0.5,
            contour_simplify_eps=0.0,
        )
        contours_b = generate_segmented_tube_contours(
            base_contour=a,
            contour_bank=bank,
            width=220,
            height=160,
            seed=77,
            segment_count=10,
            straightness=0.5,
            contour_simplify_eps=0.0,
        )
        self.assertEqual(len(contours_a), 10)
        self.assertEqual(len(contours_b), 10)
        for contour_a, contour_b in zip(contours_a, contours_b):
            self.assertTrue(np.allclose(contour_a, contour_b))

    def test_trim_splits_underpiece_and_removes_overlap(self) -> None:
        bottom = _rect_contour(10, 10, 90, 90)
        cutter = _rect_contour(45, 0, 20, 120)
        contours = [bottom, cutter]

        pieces = trim_overlapping_contours(contours, min_piece_area=1.0)
        self.assertEqual(len(pieces), 3)

        width = 128
        height = 128
        original_union = _union_mask_from_contours(contours, width=width, height=height)
        rebuilt_union = np.zeros((height, width), dtype=np.uint8)

        for piece in pieces:
            piece_canvas = np.zeros((height, width), dtype=np.uint8)
            _blit_piece(piece_canvas, piece)
            self.assertFalse(np.any((rebuilt_union > 0) & (piece_canvas > 0)))
            np.maximum(rebuilt_union, piece_canvas, out=rebuilt_union)

        self.assertTrue(np.array_equal(original_union, rebuilt_union))

    def test_trim_preserves_holes_when_shape_is_cut_inside(self) -> None:
        outer = _rect_contour(16, 16, 80, 80)
        inner = _rect_contour(40, 40, 20, 20)
        contours = [outer, inner]

        pieces = trim_overlapping_contours(contours, min_piece_area=1.0)
        self.assertEqual(len(pieces), 2)

        hole_detected = False
        for piece in pieces:
            contour_result = cv2.findContours(piece.mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            hierarchy = contour_result[1] if len(contour_result) == 2 else contour_result[2]
            if hierarchy is None:
                continue
            for node in hierarchy[0]:
                parent = int(node[3])
                if parent >= 0:
                    hole_detected = True
                    break
            if hole_detected:
                break
        self.assertTrue(hole_detected)

    def test_labels_to_single_boundary_mask_draws_one_shared_divider(self) -> None:
        labels = np.zeros((6, 6), dtype=np.int32)
        labels[:, :3] = 1
        labels[:, 3:] = 2

        boundary = labels_to_single_boundary_mask(labels, stroke_width=1.0)
        self.assertEqual(boundary.shape, labels.shape)
        self.assertEqual(int(np.any(boundary[1:-1, 2] > 0)), 0)
        self.assertEqual(int(np.any(boundary[1:-1, 3] > 0)), 1)

    def test_trim_clip_bounds_clips_output_to_canvas(self) -> None:
        contour = _rect_contour(-30, -30, 80, 80)
        pieces = trim_overlapping_contours(
            [contour],
            min_piece_area=1.0,
            clip_bounds=(0, 0, 20, 20),
        )
        self.assertEqual(len(pieces), 1)
        piece = pieces[0]
        self.assertGreaterEqual(piece.offset_x, 0)
        self.assertGreaterEqual(piece.offset_y, 0)
        self.assertLessEqual(piece.offset_x + piece.mask.shape[1], 21)
        self.assertLessEqual(piece.offset_y + piece.mask.shape[0], 21)

    def test_owned_boundary_masks_assign_shared_edge_to_one_piece(self) -> None:
        left = MaskPiece(
            mask=np.full((6, 4), 255, dtype=np.uint8),
            offset_x=0,
            offset_y=0,
            source_index=0,
        )
        right = MaskPiece(
            mask=np.full((6, 4), 255, dtype=np.uint8),
            offset_x=4,
            offset_y=0,
            source_index=1,
        )
        boundary_masks = owned_boundary_masks_for_pieces([left, right], width=8, height=6)
        self.assertEqual(len(boundary_masks), 2)
        self.assertEqual(int(np.any(boundary_masks[0][1:-1, 3] > 0)), 0)
        self.assertEqual(int(np.any(boundary_masks[1][1:-1, 0] > 0)), 1)


if __name__ == "__main__":
    unittest.main()
