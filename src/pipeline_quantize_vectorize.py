from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, TypedDict

import cv2
import numpy as np
from PIL import Image


class ShapeRecord(TypedDict):
    level: int
    contour: np.ndarray
    area: float


def _shape_area(shape: ShapeRecord) -> float:
    area_value = shape.get("area")
    if area_value is None:
        area_value = abs(float(cv2.contourArea(shape["contour"])))
    return max(1.0, float(area_value))


def _shape_bbox_area(shape: ShapeRecord) -> float:
    x, y, w, h = cv2.boundingRect(shape["contour"])
    _ = (x, y)
    return max(1.0, float(w * h))


def load_image(path: str | Path) -> Image.Image:
    with Image.open(path) as source:
        return source.convert("RGB")


def to_grayscale(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"), dtype=np.uint8)


def quantize_levels(gray: np.ndarray, levels: int = 6) -> np.ndarray:
    if levels < 2:
        raise ValueError("levels must be >= 2")
    gray_u8 = np.asarray(gray, dtype=np.uint8)
    bins = np.linspace(0, 256, levels + 1, dtype=np.int32)
    quantized = np.digitize(gray_u8, bins[1:-1], right=False).astype(np.uint8)
    return quantized


def _kernel_size_for_image(q: np.ndarray) -> int:
    min_dim = int(min(q.shape[0], q.shape[1]))
    size = max(1, int(round(min_dim / 512.0)))
    size = min(size, 5)
    if size % 2 == 0:
        size += 1
    return size


def contours_by_level(q: np.ndarray, level: int) -> list[np.ndarray]:
    mask = np.where(q == level, 255, 0).astype(np.uint8)
    kernel_size = _kernel_size_for_image(q)
    if kernel_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Keep full contour detail so generated cutouts are less angular.
    contour_result = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = contour_result[0] if len(contour_result) == 2 else contour_result[1]
    return [contour for contour in contours if contour.shape[0] >= 3]


def _chaikin_closed(points: np.ndarray, iterations: int = 1) -> np.ndarray:
    if points.shape[0] < 4 or iterations <= 0:
        return points
    out = points
    for _ in range(iterations):
        new_points: list[np.ndarray] = []
        count = out.shape[0]
        for idx in range(count):
            p0 = out[idx]
            p1 = out[(idx + 1) % count]
            q = (0.75 * p0) + (0.25 * p1)
            r = (0.25 * p0) + (0.75 * p1)
            new_points.append(q)
            new_points.append(r)
        out = np.asarray(new_points, dtype=np.float32)
    return out


def contour_to_svg_path(
    contour: np.ndarray,
    scale: float = 1.0,
    simplify_eps: float | None = None,
    smooth_iterations: int = 1,
    use_curves: bool = True,
) -> str:
    if contour.size == 0:
        return ""
    simplified = contour
    if simplify_eps is None:
        perimeter = cv2.arcLength(contour, True)
        # Lower epsilon keeps more contour detail than the previous coarse polygonal output.
        simplify_eps = perimeter * 0.0006
    if simplify_eps > 0:
        simplified = cv2.approxPolyDP(contour, simplify_eps, True)

    points = simplified.reshape(-1, 2).astype(np.float32)
    if points.shape[0] < 3:
        return ""

    if smooth_iterations > 0:
        points = _chaikin_closed(points, iterations=smooth_iterations)

    if scale != 1.0:
        points *= float(scale)

    if use_curves and points.shape[0] >= 4:
        start = (points[-1] + points[0]) * 0.5
        commands = [f"M {start[0]:.2f} {start[1]:.2f}"]
        for idx in range(points.shape[0]):
            p = points[idx]
            n = points[(idx + 1) % points.shape[0]]
            mid = (p + n) * 0.5
            commands.append(f"Q {p[0]:.2f} {p[1]:.2f} {mid[0]:.2f} {mid[1]:.2f}")
        commands.append("Z")
        return " ".join(commands)

    commands = [f"M {points[0, 0]:.2f} {points[0, 1]:.2f}"]
    for x, y in points[1:]:
        commands.append(f"L {x:.2f} {y:.2f}")
    commands.append("Z")
    return " ".join(commands)


def _tone_fill(level: int, levels: int) -> str:
    denominator = max(1, levels - 1)
    value = int(round((level / denominator) * 15))
    value = min(15, max(0, value))
    nibble = format(value, "x")
    return f"#{nibble}{nibble}{nibble}"


def _sorted_with_cap(shapes: list[ShapeRecord], max_total_contours: int) -> list[ShapeRecord]:
    if len(shapes) <= max_total_contours:
        return shapes
    shapes_sorted = sorted(shapes, key=_shape_area, reverse=True)
    return shapes_sorted[:max_total_contours]


def collect_shapes(
    q: np.ndarray,
    levels: int = 6,
    max_total_contours: int = 10_000,
    min_area_ratio: float = 0.0008,
    min_area_pixels: float = 36.0,
) -> tuple[list[ShapeRecord], dict[int, int], bool]:
    image_area = float(max(1, q.shape[0] * q.shape[1]))
    min_area = max(float(min_area_pixels), image_area * float(min_area_ratio))

    shapes: list[ShapeRecord] = []
    raw_shapes: list[ShapeRecord] = []
    per_level_counts: dict[int, int] = {}
    for level in range(levels):
        contours = contours_by_level(q, level)
        kept_count = 0
        for contour in contours:
            area = abs(float(cv2.contourArea(contour)))
            shape: ShapeRecord = {"level": level, "contour": contour, "area": area}
            raw_shapes.append(shape)
            if area < min_area:
                continue
            shapes.append(shape)
            kept_count += 1
        per_level_counts[level] = kept_count

    if not shapes and raw_shapes:
        # Fallback: keep largest contours when thresholding removes all shapes.
        raw_shapes_sorted = sorted(raw_shapes, key=_shape_area, reverse=True)
        fallback_count = min(len(raw_shapes_sorted), max(8, levels))
        shapes = raw_shapes_sorted[:fallback_count]
        per_level_counts = {level: 0 for level in range(levels)}
        for shape in shapes:
            per_level_counts[int(shape["level"])] += 1

    capped = len(shapes) > max_total_contours
    if capped:
        shapes = _sorted_with_cap(shapes, max_total_contours=max_total_contours)
    return shapes, per_level_counts, capped


def _scatter_transform(
    contour: np.ndarray,
    rng: random.Random,
    width: int,
    height: int,
    shape_scale: float = 1.0,
) -> tuple[float, float, float, float, float, float]:
    x, y, w, h = cv2.boundingRect(contour)
    scale = shape_scale * rng.uniform(0.95, 1.05)
    angle = rng.uniform(-20.0, 20.0)
    max_x = max(0.0, width - (w * scale))
    max_y = max(0.0, height - (h * scale))
    target_x = rng.uniform(0.0, max_x) if max_x > 0.0 else 0.0
    target_y = rng.uniform(0.0, max_y) if max_y > 0.0 else 0.0
    source_cx = x + (w / 2.0)
    source_cy = y + (h / 2.0)
    target_cx = target_x + (w / 2.0)
    target_cy = target_y + (h / 2.0)
    return source_cx, source_cy, target_cx, target_cy, angle, scale


def _select_shapes(
    shapes: list[ShapeRecord],
    rng: random.Random,
    shape_limit: int | None,
) -> list[ShapeRecord]:
    if not shape_limit or shape_limit <= 0 or len(shapes) <= shape_limit:
        return list(shapes)
    sorted_shapes = sorted(shapes, key=_shape_area, reverse=True)
    pool_size = min(len(sorted_shapes), max(shape_limit * 8, shape_limit))
    pool = sorted_shapes[:pool_size]

    hero_pool_size = max(1, min(len(pool), max(shape_limit * 2, 3)))
    hero_pool = pool[:hero_pool_size]
    hero = hero_pool[rng.randrange(len(hero_pool))]
    selected: list[ShapeRecord] = [hero]
    selected_ids = {id(hero)}

    remaining = [shape for shape in pool if id(shape) not in selected_ids]
    while remaining and len(selected) < shape_limit:
        weights = [max(1.0, _shape_area(shape) ** 0.45) for shape in remaining]
        total = float(sum(weights))
        pick = rng.uniform(0.0, total)
        running = 0.0
        chosen_index = 0
        for idx, weight in enumerate(weights):
            running += weight
            if pick <= running:
                chosen_index = idx
                break
        chosen = remaining.pop(chosen_index)
        selected.append(chosen)
        selected_ids.add(id(chosen))

    if len(selected) < shape_limit:
        for shape in sorted_shapes:
            if id(shape) in selected_ids:
                continue
            selected.append(shape)
            selected_ids.add(id(shape))
            if len(selected) >= shape_limit:
                break

    return selected[:shape_limit]


def _resolve_shape_limit(
    rng: random.Random,
    available_count: int,
    max_shapes: int | None,
    shape_count_range: tuple[int, int] | None,
) -> int | None:
    if available_count <= 0:
        return 0

    if shape_count_range is not None:
        low, high = shape_count_range
        low = max(1, int(low))
        high = max(1, int(high))
        if low > high:
            low, high = high, low
        requested = rng.randint(low, high)
        return min(available_count, requested)

    if not max_shapes or max_shapes <= 0:
        return None
    return min(available_count, int(max_shapes))


def _shape_scales_for_varied_sizes(
    shapes: list[ShapeRecord],
    rng: random.Random,
    width: int,
    height: int,
    target_fill_ratio: float | None,
) -> list[float]:
    if not shapes:
        return []

    canvas_area = float(max(1, width * height))
    hero_index = rng.randrange(len(shapes))
    if target_fill_ratio is None:
        hero_min = 0.45
        hero_max = 0.55
    else:
        center = max(0.15, min(0.80, float(target_fill_ratio)))
        hero_min = max(0.10, center - 0.08)
        hero_max = min(0.90, center + 0.08)

    scales: list[float] = []
    for idx, shape in enumerate(shapes):
        bbox_area = _shape_bbox_area(shape)

        if idx == hero_index:
            desired_ratio = rng.uniform(hero_min, hero_max)
            scale = (canvas_area * desired_ratio / bbox_area) ** 0.5
        else:
            if rng.random() < 0.25:
                scale = rng.uniform(0.90, 1.20)
            else:
                desired_ratio = rng.uniform(0.08, 0.35)
                scale = (canvas_area * desired_ratio / bbox_area) ** 0.5

        scales.append(max(0.90, min(80.0, scale)))
    return scales


def build_svg(
    shapes: Iterable[ShapeRecord],
    width: int,
    height: int,
    seed: int,
    arrangement: str = "scatter",
    levels: int | None = None,
    max_shapes: int | None = None,
    target_fill_ratio: float | None = None,
    shape_count_range: tuple[int, int] | None = None,
    white_background: bool = True,
) -> str:
    shape_list = list(shapes)
    max_level = max((int(shape["level"]) for shape in shape_list), default=0)
    level_count = levels if levels is not None else (max_level + 1)
    rng = random.Random(seed)
    shape_limit = _resolve_shape_limit(
        rng=rng,
        available_count=len(shape_list),
        max_shapes=max_shapes,
        shape_count_range=shape_count_range,
    )
    shape_list = _select_shapes(shape_list, rng=rng, shape_limit=shape_limit)
    shape_scales = _shape_scales_for_varied_sizes(
        shape_list,
        rng=rng,
        width=width,
        height=height,
        target_fill_ratio=target_fill_ratio,
    )

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<clipPath id="canvas-clip">',
        f'<rect x="0" y="0" width="{width}" height="{height}" />',
        "</clipPath>",
        "</defs>",
    ]
    if white_background:
        lines.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" />')
    lines.append('<g clip-path="url(#canvas-clip)">')

    for idx, shape in enumerate(shape_list):
        contour = shape["contour"]
        level = int(shape["level"])
        path_data = contour_to_svg_path(contour)
        if not path_data:
            continue

        if arrangement == "scatter":
            shape_scale = shape_scales[idx] if idx < len(shape_scales) else 1.0
            src_cx, src_cy, dst_cx, dst_cy, angle, scale = _scatter_transform(
                contour,
                rng,
                width,
                height,
                shape_scale=shape_scale,
            )
            transform = (
                f"translate({dst_cx:.2f} {dst_cy:.2f}) "
                f"rotate({angle:.2f}) "
                f"scale({scale:.3f}) "
                f"translate({-src_cx:.2f} {-src_cy:.2f})"
            )
        else:
            transform = ""

        fill = _tone_fill(level, level_count)
        if transform:
            lines.append(f'<path d="{path_data}" fill="{fill}" stroke="none" transform="{transform}" />')
        else:
            lines.append(f'<path d="{path_data}" fill="{fill}" stroke="none" />')

    lines.append("</g>")
    lines.append("</svg>")
    return "\n".join(lines)


def svg_to_png(svg_str: str, out_path_png: str | Path, dpi: int = 96) -> Path:
    try:
        import cairosvg
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("cairosvg is required for SVG to PNG rendering.") from exc

    out_path = Path(out_path_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cairosvg.svg2png(bytestring=svg_str.encode("utf-8"), write_to=str(out_path), dpi=dpi)
    return out_path
