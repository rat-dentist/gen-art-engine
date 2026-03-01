from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np


@dataclass
class MaskPiece:
    mask: np.ndarray
    offset_x: int
    offset_y: int
    source_index: int


def _normalize_contour(contour: np.ndarray) -> np.ndarray:
    points = np.asarray(contour)
    if points.ndim == 3 and points.shape[1:] == (1, 2):
        resolved = points.reshape(-1, 2)
    elif points.ndim == 2 and points.shape[1] == 2:
        resolved = points
    else:
        raise ValueError(f"Unsupported contour shape: {points.shape}")
    if resolved.shape[0] < 3:
        raise ValueError("Contour must have at least 3 points.")
    return resolved.astype(np.float32).reshape(-1, 1, 2)


def _contour_bounds(contour: np.ndarray) -> tuple[int, int, int, int]:
    points = contour.reshape(-1, 2)
    min_x = int(math.floor(float(np.min(points[:, 0]))))
    min_y = int(math.floor(float(np.min(points[:, 1]))))
    max_x = int(math.ceil(float(np.max(points[:, 0]))))
    max_y = int(math.ceil(float(np.max(points[:, 1]))))
    return min_x, min_y, max_x, max_y


def _rasterize_contour(contour: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    width = max(1, x1 - x0 + 1)
    height = max(1, y1 - y0 + 1)
    mask = np.zeros((height, width), dtype=np.uint8)

    points = np.rint(contour.reshape(-1, 2)).astype(np.int32)
    points[:, 0] -= x0
    points[:, 1] -= y0
    cv2.fillPoly(mask, [points.reshape(-1, 1, 2)], 255)
    return mask


def _connected_components_to_pieces(
    mask: np.ndarray,
    offset_x: int,
    offset_y: int,
    source_index: int,
    min_piece_area: float,
) -> list[MaskPiece]:
    binary = (mask > 0).astype(np.uint8)
    if not np.any(binary):
        return []

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    pieces: list[MaskPiece] = []
    for label_index in range(1, component_count):
        area = int(stats[label_index, cv2.CC_STAT_AREA])
        if area < min_piece_area:
            continue
        x = int(stats[label_index, cv2.CC_STAT_LEFT])
        y = int(stats[label_index, cv2.CC_STAT_TOP])
        width = int(stats[label_index, cv2.CC_STAT_WIDTH])
        height = int(stats[label_index, cv2.CC_STAT_HEIGHT])
        if width <= 0 or height <= 0:
            continue

        component = np.where(labels[y : y + height, x : x + width] == label_index, 255, 0).astype(np.uint8)
        pieces.append(
            MaskPiece(
                mask=component,
                offset_x=offset_x + x,
                offset_y=offset_y + y,
                source_index=source_index,
            )
        )
    return pieces


def _tube_path_points(
    rng: random.Random,
    width: int,
    height: int,
    count: int,
    step_length: float,
    straightness: float = 0.45,
) -> list[tuple[float, float]]:
    x = rng.uniform(0.0, float(width))
    y = rng.uniform(0.0, float(height))
    heading = rng.uniform(0.0, 360.0)
    straight = max(0.0, min(1.0, float(straightness)))
    max_turn = 62.0 - (54.0 * straight)

    points: list[tuple[float, float]] = [(x, y)]
    for _ in range(max(1, int(count)) - 1):
        heading += rng.uniform(-max_turn, max_turn)
        radians = math.radians(heading)
        x += math.cos(radians) * float(step_length)
        y += math.sin(radians) * float(step_length)
        points.append((x, y))
    return points


def _tube_angle_degrees(points: list[tuple[float, float]], index: int) -> float:
    count = len(points)
    if count <= 1:
        return 0.0
    if index <= 0:
        x0, y0 = points[0]
        x1, y1 = points[1]
    elif index >= count - 1:
        x0, y0 = points[count - 2]
        x1, y1 = points[count - 1]
    else:
        x0, y0 = points[index - 1]
        x1, y1 = points[index + 1]
    return math.degrees(math.atan2(y1 - y0, x1 - x0))


def _transform_contour(
    contour: np.ndarray,
    source_cx: float,
    source_cy: float,
    target_x: float,
    target_y: float,
    angle_degrees: float,
    scale: float,
) -> np.ndarray:
    points = contour.reshape(-1, 2).astype(np.float32)
    points[:, 0] -= float(source_cx)
    points[:, 1] -= float(source_cy)
    points *= float(scale)

    radians = math.radians(float(angle_degrees))
    cos_a = math.cos(radians)
    sin_a = math.sin(radians)
    rotated_x = (points[:, 0] * cos_a) - (points[:, 1] * sin_a)
    rotated_y = (points[:, 0] * sin_a) + (points[:, 1] * cos_a)

    transformed = np.empty_like(points)
    transformed[:, 0] = rotated_x + float(target_x)
    transformed[:, 1] = rotated_y + float(target_y)
    return transformed.reshape(-1, 1, 2)


def _signed_area(points: np.ndarray) -> float:
    if points.shape[0] < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _resample_closed_contour(contour: np.ndarray, point_count: int) -> np.ndarray:
    points = _normalize_contour(contour).reshape(-1, 2).astype(np.float32)
    count = max(3, int(point_count))
    if points.shape[0] == count:
        return points.reshape(-1, 1, 2)

    closed = np.vstack([points, points[0]])
    deltas = np.diff(closed, axis=0)
    lengths = np.linalg.norm(deltas, axis=1)
    perimeter = float(np.sum(lengths))
    if perimeter <= 1e-8:
        repeated = np.repeat(points[:1], count, axis=0)
        return repeated.reshape(-1, 1, 2)

    cumulative = np.concatenate([np.array([0.0], dtype=np.float32), np.cumsum(lengths)])
    targets = np.linspace(0.0, perimeter, num=count, endpoint=False, dtype=np.float32)
    sampled: list[np.ndarray] = []
    seg_idx = 0
    for target in targets.tolist():
        while seg_idx < lengths.shape[0] - 1 and float(cumulative[seg_idx + 1]) <= float(target):
            seg_idx += 1
        start = closed[seg_idx]
        end = closed[seg_idx + 1]
        seg_len = float(lengths[seg_idx])
        if seg_len <= 1e-8:
            sampled.append(start)
            continue
        alpha = (float(target) - float(cumulative[seg_idx])) / seg_len
        sampled.append(start + ((end - start) * float(alpha)))
    return np.asarray(sampled, dtype=np.float32).reshape(-1, 1, 2)


def _align_contour_phase(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    ref = reference.reshape(-1, 2).astype(np.float32)
    tgt = target.reshape(-1, 2).astype(np.float32)
    if ref.shape[0] != tgt.shape[0]:
        raise ValueError("Contours must have the same point count for phase alignment.")
    if ref.shape[0] <= 1:
        return tgt.reshape(-1, 1, 2)

    if _signed_area(ref) * _signed_area(tgt) < 0.0:
        tgt = tgt[::-1].copy()

    best_shift = 0
    best_score = float("inf")
    point_count = ref.shape[0]
    for shift in range(point_count):
        shifted = np.roll(tgt, shift=shift, axis=0)
        score = float(np.sum((shifted - ref) ** 2))
        if score < best_score:
            best_score = score
            best_shift = shift

    aligned = np.roll(tgt, shift=best_shift, axis=0)
    return aligned.reshape(-1, 1, 2)


def build_all_to_all_morph_bank(
    contours: Sequence[np.ndarray],
    steps: int = 6,
    point_count: int = 96,
    simplify_eps: float | None = None,
) -> list[np.ndarray]:
    if not contours:
        return []

    processed: list[np.ndarray] = []
    for raw in contours:
        contour = _normalize_contour(raw)
        if simplify_eps is not None and float(simplify_eps) > 0.0:
            simplified = cv2.approxPolyDP(contour, float(simplify_eps), True)
            if simplified is not None and simplified.shape[0] >= 3:
                contour = simplified.astype(np.float32)
        processed.append(contour.astype(np.float32))

    if len(processed) == 1:
        return [processed[0]]

    tween_steps = max(0, int(steps))
    sample_points = max(3, int(point_count))
    resampled = [_resample_closed_contour(contour, sample_points) for contour in processed]

    bank: list[np.ndarray] = []
    for src_idx, src in enumerate(resampled):
        src_points = src.reshape(-1, 2).astype(np.float32)
        for dst_idx, dst in enumerate(resampled):
            if src_idx == dst_idx:
                continue
            aligned_dst = _align_contour_phase(src, dst).reshape(-1, 2).astype(np.float32)
            total = tween_steps + 1
            for step_idx in range(tween_steps + 2):
                t = float(step_idx) / float(total)
                mixed = ((1.0 - t) * src_points) + (t * aligned_dst)
                bank.append(mixed.astype(np.float32).reshape(-1, 1, 2))

    return bank


def generate_segmented_tube_contours(
    base_contour: np.ndarray,
    width: int,
    height: int,
    seed: int,
    segment_count: int,
    straightness: float = 0.45,
    contour_simplify_eps: float | None = None,
    contour_bank: Sequence[np.ndarray] | None = None,
) -> list[np.ndarray]:
    source_contours: list[np.ndarray] = []
    if contour_bank is not None:
        for raw in contour_bank:
            contour = _normalize_contour(raw)
            if contour_simplify_eps is not None and float(contour_simplify_eps) > 0.0:
                simplified = cv2.approxPolyDP(contour, float(contour_simplify_eps), True)
                if simplified is not None and simplified.shape[0] >= 3:
                    contour = simplified.astype(np.float32)
            source_contours.append(contour.astype(np.float32))

    if not source_contours:
        contour = _normalize_contour(base_contour)
        if contour_simplify_eps is not None and float(contour_simplify_eps) > 0.0:
            simplified = cv2.approxPolyDP(contour, float(contour_simplify_eps), True)
            if simplified is not None and simplified.shape[0] >= 3:
                contour = simplified.astype(np.float32)
        source_contours = [contour.astype(np.float32)]

    count = max(1, min(400, int(segment_count)))
    rng = random.Random(int(seed))

    majors: list[float] = []
    for contour in source_contours:
        _x, _y, w, h = cv2.boundingRect(np.rint(contour).astype(np.int32))
        _ = (_x, _y)
        majors.append(max(1.0, float(max(w, h))))
    representative_major = float(np.median(np.asarray(majors, dtype=np.float32)))
    min_canvas_dim = float(max(1, min(width, height)))
    min_major = max(24.0, min_canvas_dim * 0.06)
    target_major = min_canvas_dim * rng.uniform(0.10, 0.17)
    min_scale = min_major / representative_major
    base_scale = max(min_scale, target_major / representative_major)
    base_scale = max(0.95, min(120.0, base_scale))

    segment_major = representative_major * base_scale
    total_tube_length = max(segment_major * rng.uniform(16.0, 24.0), min_canvas_dim * 0.9)
    step_length = max(0.2, total_tube_length / max(1, count - 1))
    points = _tube_path_points(
        rng=rng,
        width=width,
        height=height,
        count=count,
        step_length=step_length,
        straightness=straightness,
    )

    transformed: list[np.ndarray] = []
    for idx, (target_x, target_y) in enumerate(points):
        contour = source_contours[idx % len(source_contours)]
        x, y, w, h = cv2.boundingRect(np.rint(contour).astype(np.int32))
        source_cx = x + (w / 2.0)
        source_cy = y + (h / 2.0)
        contour_major = max(1.0, float(max(w, h)))
        min_scale_for_contour = min_major / contour_major
        contour_scale_bias = representative_major / contour_major
        angle = _tube_angle_degrees(points, idx) + rng.uniform(-7.0, 7.0)
        scale = base_scale * contour_scale_bias * rng.uniform(0.96, 1.05)
        scale = max(min_scale_for_contour, min(120.0, scale))
        transformed.append(
            _transform_contour(
                contour=contour,
                source_cx=source_cx,
                source_cy=source_cy,
                target_x=target_x,
                target_y=target_y,
                angle_degrees=angle,
                scale=scale,
            )
        )
    return transformed


def trim_overlapping_contours(
    contours: Iterable[np.ndarray],
    min_piece_area: float = 8.0,
    clip_bounds: tuple[int, int, int, int] | None = None,
) -> list[MaskPiece]:
    resolved_contours = [_normalize_contour(contour) for contour in contours]
    if not resolved_contours:
        return []

    if clip_bounds is None:
        bounds = [_contour_bounds(contour) for contour in resolved_contours]
        min_x = min(bound[0] for bound in bounds) - 1
        min_y = min(bound[1] for bound in bounds) - 1
        max_x = max(bound[2] for bound in bounds) + 1
        max_y = max(bound[3] for bound in bounds) + 1
        clip_x = int(min_x)
        clip_y = int(min_y)
        clip_width = max(1, int(max_x - min_x + 1))
        clip_height = max(1, int(max_y - min_y + 1))
    else:
        clip_x, clip_y, clip_width, clip_height = [int(value) for value in clip_bounds]
        if clip_width <= 0 or clip_height <= 0:
            raise ValueError("clip_bounds width/height must be > 0.")

    label_map = np.zeros((clip_height, clip_width), dtype=np.int32)

    for source_index, contour in enumerate(resolved_contours, start=1):
        x0, y0, x1, y1 = _contour_bounds(contour)
        raster_bbox = (x0 - 1, y0 - 1, x1 + 1, y1 + 1)
        mask = _rasterize_contour(contour, raster_bbox)

        dest_x = raster_bbox[0] - clip_x
        dest_y = raster_bbox[1] - clip_y
        src_h, src_w = mask.shape
        dx0 = max(0, dest_x)
        dy0 = max(0, dest_y)
        dx1 = min(clip_width, dest_x + src_w)
        dy1 = min(clip_height, dest_y + src_h)
        if dx0 >= dx1 or dy0 >= dy1:
            continue

        sx0 = dx0 - dest_x
        sy0 = dy0 - dest_y
        sx1 = sx0 + (dx1 - dx0)
        sy1 = sy0 + (dy1 - dy0)

        dst = label_map[dy0:dy1, dx0:dx1]
        src = mask[sy0:sy1, sx0:sx1] > 0
        dst[src] = int(source_index)

    pieces: list[MaskPiece] = []
    visible_labels = np.unique(label_map)
    for label_value in visible_labels:
        if label_value <= 0:
            continue
        label_mask = label_map == int(label_value)

        ys, xs = np.nonzero(label_mask)
        local_x0 = int(np.min(xs))
        local_y0 = int(np.min(ys))
        local_x1 = int(np.max(xs)) + 1
        local_y1 = int(np.max(ys)) + 1

        cropped_mask = np.where(
            label_map[local_y0:local_y1, local_x0:local_x1] == int(label_value),
            255,
            0,
        ).astype(np.uint8)
        pieces.extend(
            _connected_components_to_pieces(
                mask=cropped_mask,
                offset_x=clip_x + local_x0,
                offset_y=clip_y + local_y0,
                source_index=int(label_value - 1),
                min_piece_area=min_piece_area,
            )
        )
    return pieces


def piece_rings(piece: MaskPiece, simplify_eps: float = 0.0) -> list[np.ndarray]:
    contour_result = cv2.findContours(piece.mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours = contour_result[0] if len(contour_result) == 2 else contour_result[1]
    rings: list[np.ndarray] = []
    for contour in contours:
        if contour.shape[0] < 3:
            continue
        resolved = contour
        if simplify_eps > 0.0:
            resolved = cv2.approxPolyDP(resolved, float(simplify_eps), True)
        points = resolved.reshape(-1, 2).astype(np.float32)
        points[:, 0] += float(piece.offset_x)
        points[:, 1] += float(piece.offset_y)
        rings.append(points)
    return rings


def labels_to_single_boundary_mask(labels: np.ndarray, stroke_width: float = 1.0) -> np.ndarray:
    label_map = np.asarray(labels, dtype=np.int32)
    if label_map.ndim != 2:
        raise ValueError("labels must be a 2D array.")

    height, width = label_map.shape
    boundary = np.zeros((height, width), dtype=np.uint8)
    if height == 0 or width == 0:
        return boundary

    if width > 1:
        diff_h = label_map[:, 1:] != label_map[:, :-1]
        boundary[:, 1:] = np.maximum(boundary[:, 1:], diff_h.astype(np.uint8))
    if height > 1:
        diff_v = label_map[1:, :] != label_map[:-1, :]
        boundary[1:, :] = np.maximum(boundary[1:, :], diff_v.astype(np.uint8))

    occupied = label_map > 0
    boundary[:, 0] = np.maximum(boundary[:, 0], occupied[:, 0].astype(np.uint8))
    boundary[0, :] = np.maximum(boundary[0, :], occupied[0, :].astype(np.uint8))
    boundary[:, width - 1] = np.maximum(boundary[:, width - 1], occupied[:, width - 1].astype(np.uint8))
    boundary[height - 1, :] = np.maximum(boundary[height - 1, :], occupied[height - 1, :].astype(np.uint8))

    pixel_width = max(1, int(round(float(stroke_width))))
    if pixel_width > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pixel_width, pixel_width))
        boundary = cv2.dilate(boundary, kernel, iterations=1)

    return boundary.astype(np.uint8)


def owned_boundary_masks_for_pieces(
    pieces: Sequence[MaskPiece],
    width: int,
    height: int,
) -> list[np.ndarray]:
    canvas_width = max(1, int(width))
    canvas_height = max(1, int(height))
    labels = np.zeros((canvas_height, canvas_width), dtype=np.int32)

    for label_value, piece in enumerate(pieces, start=1):
        mask_h, mask_w = piece.mask.shape
        x0 = int(piece.offset_x)
        y0 = int(piece.offset_y)
        x1 = x0 + mask_w
        y1 = y0 + mask_h

        dx0 = max(0, x0)
        dy0 = max(0, y0)
        dx1 = min(canvas_width, x1)
        dy1 = min(canvas_height, y1)
        if dx0 >= dx1 or dy0 >= dy1:
            continue

        sx0 = dx0 - x0
        sy0 = dy0 - y0
        sx1 = sx0 + (dx1 - dx0)
        sy1 = sy0 + (dy1 - dy0)

        src = piece.mask[sy0:sy1, sx0:sx1] > 0
        dst = labels[dy0:dy1, dx0:dx1]
        dst[src] = int(label_value)

    source_indices = [int(piece.source_index) for piece in pieces]
    owner_map = np.zeros_like(labels, dtype=np.int32)

    def choose_owner(label_a: int, label_b: int) -> int:
        if label_a <= 0:
            return label_b
        if label_b <= 0:
            return label_a
        source_a = source_indices[label_a - 1]
        source_b = source_indices[label_b - 1]
        if source_a != source_b:
            return label_a if source_a > source_b else label_b
        return label_a if label_a > label_b else label_b

    if canvas_width > 1:
        left = labels[:, :-1]
        right = labels[:, 1:]
        diff = left != right
        ys, xs = np.nonzero(diff)
        for y, x in zip(ys.tolist(), xs.tolist()):
            label_left = int(left[y, x])
            label_right = int(right[y, x])
            owner = choose_owner(label_left, label_right)
            if owner <= 0:
                continue
            if owner == label_left:
                owner_map[y, x] = owner
            else:
                owner_map[y, x + 1] = owner

    if canvas_height > 1:
        top = labels[:-1, :]
        bottom = labels[1:, :]
        diff = top != bottom
        ys, xs = np.nonzero(diff)
        for y, x in zip(ys.tolist(), xs.tolist()):
            label_top = int(top[y, x])
            label_bottom = int(bottom[y, x])
            owner = choose_owner(label_top, label_bottom)
            if owner <= 0:
                continue
            if owner == label_top:
                owner_map[y, x] = owner
            else:
                owner_map[y + 1, x] = owner

    if canvas_width > 0 and canvas_height > 0:
        top_edge = labels[0, :] > 0
        owner_map[0, top_edge] = labels[0, top_edge]
        bottom_edge = labels[canvas_height - 1, :] > 0
        owner_map[canvas_height - 1, bottom_edge] = labels[canvas_height - 1, bottom_edge]
        left_edge = labels[:, 0] > 0
        owner_map[left_edge, 0] = labels[left_edge, 0]
        right_edge = labels[:, canvas_width - 1] > 0
        owner_map[right_edge, canvas_width - 1] = labels[right_edge, canvas_width - 1]

    boundary_masks: list[np.ndarray] = []
    for label_value, piece in enumerate(pieces, start=1):
        local = np.zeros_like(piece.mask, dtype=np.uint8)
        mask_h, mask_w = local.shape
        x0 = int(piece.offset_x)
        y0 = int(piece.offset_y)
        x1 = x0 + mask_w
        y1 = y0 + mask_h

        dx0 = max(0, x0)
        dy0 = max(0, y0)
        dx1 = min(canvas_width, x1)
        dy1 = min(canvas_height, y1)
        if dx0 >= dx1 or dy0 >= dy1:
            boundary_masks.append(local)
            continue

        sx0 = dx0 - x0
        sy0 = dy0 - y0
        sx1 = sx0 + (dx1 - dx0)
        sy1 = sy0 + (dy1 - dy0)

        local[sy0:sy1, sx0:sx1] = np.where(
            owner_map[dy0:dy1, dx0:dx1] == int(label_value),
            255,
            0,
        ).astype(np.uint8)
        boundary_masks.append(local)

    return boundary_masks


__all__ = [
    "MaskPiece",
    "build_all_to_all_morph_bank",
    "generate_segmented_tube_contours",
    "labels_to_single_boundary_mask",
    "owned_boundary_masks_for_pieces",
    "piece_rings",
    "trim_overlapping_contours",
]
