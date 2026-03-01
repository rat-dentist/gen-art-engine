from __future__ import annotations

import math
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


class LayerSpec(TypedDict, total=False):
    type: str
    scatter_shape_count_min: int
    scatter_shape_count_max: int
    scatter_target_fill_ratio: float
    tube_segment_count: int
    tube_stroke_width: float
    tube_straightness: float


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


def _choose_tube_shape(shapes: list[ShapeRecord], rng: random.Random) -> ShapeRecord | None:
    if not shapes:
        return None
    sorted_shapes = sorted(shapes, key=_shape_area, reverse=True)
    candidate_count = max(1, min(len(sorted_shapes), max(3, len(sorted_shapes) // 3)))
    pool = sorted_shapes[:candidate_count]
    return pool[rng.randrange(len(pool))]


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
        step = float(step_length)
        radians = math.radians(heading)
        nx = x + (math.cos(radians) * step)
        ny = y + (math.sin(radians) * step)

        x, y = nx, ny
        points.append((x, y))
    return points


def _tube_angle_degrees(points: list[tuple[float, float]], idx: int) -> float:
    count = len(points)
    if count <= 1:
        return 0.0
    if idx <= 0:
        x0, y0 = points[0]
        x1, y1 = points[1]
    elif idx >= count - 1:
        x0, y0 = points[count - 2]
        x1, y1 = points[count - 1]
    else:
        x0, y0 = points[idx - 1]
        x1, y1 = points[idx + 1]
    return math.degrees(math.atan2(y1 - y0, x1 - x0))


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


def _append_scatter_layer(
    lines: list[str],
    shapes: list[ShapeRecord],
    rng: random.Random,
    width: int,
    height: int,
    level_count: int,
    max_shapes: int | None,
    shape_count_range: tuple[int, int] | None,
    target_fill_ratio: float | None,
) -> None:
    if not shapes:
        return
    shape_limit = _resolve_shape_limit(
        rng=rng,
        available_count=len(shapes),
        max_shapes=max_shapes,
        shape_count_range=shape_count_range,
    )
    layer_shapes = _select_shapes(shapes, rng=rng, shape_limit=shape_limit)
    layer_scales = _shape_scales_for_varied_sizes(
        layer_shapes,
        rng=rng,
        width=width,
        height=height,
        target_fill_ratio=target_fill_ratio,
    )

    for idx, shape in enumerate(layer_shapes):
        contour = shape["contour"]
        level = int(shape["level"])
        path_data = contour_to_svg_path(contour)
        if not path_data:
            continue
        shape_scale = layer_scales[idx] if idx < len(layer_scales) else 1.0
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
        fill = _tone_fill(level, level_count)
        lines.append(f'<path d="{path_data}" fill="{fill}" stroke="none" transform="{transform}" />')


def _append_segmented_tube_layer(
    lines: list[str],
    shapes: list[ShapeRecord],
    rng: random.Random,
    width: int,
    height: int,
    tube_segment_count_range: tuple[int, int] | None,
    tube_segment_count: int | None = None,
    tube_stroke_width: float | None = None,
    tube_straightness: float | None = None,
) -> None:
    tube_shape = _choose_tube_shape(shapes, rng=rng)
    if tube_shape is None:
        return
    contour = tube_shape["contour"]
    path_data = contour_to_svg_path(contour)
    if not path_data:
        return

    x, y, w, h = cv2.boundingRect(contour)
    src_cx = x + (w / 2.0)
    src_cy = y + (h / 2.0)
    base_major = max(1.0, float(max(w, h)))
    min_canvas_dim = float(max(1, min(width, height)))
    min_major = max(24.0, min_canvas_dim * 0.06)
    target_major = min_canvas_dim * rng.uniform(0.10, 0.17)
    min_scale = min_major / base_major
    base_scale = max(min_scale, target_major / base_major)
    base_scale = max(0.95, min(120.0, base_scale))

    segment_major = base_major * base_scale
    if tube_segment_count is not None:
        segment_count = max(1, min(400, int(tube_segment_count)))
    elif tube_segment_count_range is not None:
        low, high = tube_segment_count_range
        low = max(1, int(low))
        high = max(1, int(high))
        if low > high:
            low, high = high, low
        segment_count = rng.randint(low, high)
    else:
        estimated_step = max(segment_major * 0.70, min_major * 0.55)
        auto_low = max(12, int(round((min_canvas_dim * 1.6) / max(1.0, estimated_step))))
        auto_high = max(auto_low, int(round((min_canvas_dim * 2.9) / max(1.0, estimated_step))))
        segment_count = rng.randint(auto_low, auto_high)
    segment_count = max(1, min(400, int(segment_count)))

    total_tube_length = max(segment_major * rng.uniform(16.0, 24.0), min_canvas_dim * 0.9)
    step_length = max(0.2, total_tube_length / max(1, segment_count - 1))
    points = _tube_path_points(
        rng=rng,
        width=width,
        height=height,
        count=segment_count,
        step_length=step_length,
        straightness=0.45 if tube_straightness is None else float(tube_straightness),
    )
    if tube_stroke_width is None:
        stroke_width = 0.30
    else:
        stroke_width = max(0.10, min(12.0, float(tube_stroke_width)))
    for idx, (dst_x, dst_y) in enumerate(points):
        angle = _tube_angle_degrees(points, idx) + rng.uniform(-7.0, 7.0)
        scale = max(min_scale, min(120.0, base_scale * rng.uniform(0.96, 1.05)))
        transform = (
            f"translate({dst_x:.2f} {dst_y:.2f}) "
            f"rotate({angle:.2f}) "
            f"scale({scale:.3f}) "
            f"translate({-src_cx:.2f} {-src_cy:.2f})"
        )
        lines.append(
            f'<path d="{path_data}" fill="#ffffff" fill-opacity="1" '
            f'stroke="#000000" stroke-opacity="1" '
            f'stroke-width="{stroke_width:.2f}" stroke-linejoin="round" stroke-linecap="round" '
            f'paint-order="fill stroke" '
            f'vector-effect="non-scaling-stroke" transform="{transform}" />'
        )


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
    tube_segment_count_range: tuple[int, int] | None = None,
    layer_sequence: Iterable[str] | None = None,
    layer_specs: Iterable[LayerSpec] | None = None,
    white_background: bool = True,
) -> str:
    all_shapes = list(shapes)
    max_level = max((int(shape["level"]) for shape in all_shapes), default=0)
    level_count = levels if levels is not None else (max_level + 1)
    rng = random.Random(seed)
    if arrangement not in {"scatter", "segmented_tube", "scatter_segmented_tube"}:
        raise ValueError(f"Unsupported arrangement: {arrangement}")

    resolved_layer_specs: list[LayerSpec] = []
    if layer_specs is not None:
        for raw_spec in layer_specs:
            raw_dict = dict(raw_spec)
            layer = str(raw_dict.get("type", "")).strip().lower()
            if not layer:
                continue
            if layer not in {"scatter", "segmented_tube"}:
                raise ValueError(f"Unsupported layer type: {layer}")
            resolved: LayerSpec = {"type": layer}
            if layer == "scatter":
                scatter_min_raw = raw_dict.get("scatter_shape_count_min")
                scatter_max_raw = raw_dict.get("scatter_shape_count_max")
                scatter_fill_raw = raw_dict.get("scatter_target_fill_ratio")
                if scatter_min_raw is not None:
                    resolved["scatter_shape_count_min"] = max(1, min(500, int(scatter_min_raw)))
                if scatter_max_raw is not None:
                    resolved["scatter_shape_count_max"] = max(1, min(500, int(scatter_max_raw)))
                if scatter_fill_raw is not None:
                    resolved["scatter_target_fill_ratio"] = max(0.05, min(0.95, float(scatter_fill_raw)))
            else:
                count_raw = raw_dict.get("tube_segment_count")
                width_raw = raw_dict.get("tube_stroke_width")
                straight_raw = raw_dict.get("tube_straightness")
                if count_raw is not None:
                    resolved["tube_segment_count"] = max(1, min(400, int(count_raw)))
                if width_raw is not None:
                    resolved["tube_stroke_width"] = max(0.10, min(12.0, float(width_raw)))
                if straight_raw is not None:
                    resolved["tube_straightness"] = max(0.0, min(1.0, float(straight_raw)))
            resolved_layer_specs.append(resolved)

    resolved_layers: list[str] = []
    if not resolved_layer_specs and layer_sequence is not None:
        for raw_layer in layer_sequence:
            layer = str(raw_layer).strip().lower()
            if not layer:
                continue
            if layer not in {"scatter", "segmented_tube"}:
                raise ValueError(f"Unsupported layer type: {layer}")
            resolved_layers.append(layer)
    if not resolved_layer_specs and not resolved_layers:
        if arrangement == "scatter":
            resolved_layers = ["scatter"]
        elif arrangement == "segmented_tube":
            resolved_layers = ["segmented_tube"]
        else:
            resolved_layers = ["scatter", "segmented_tube"]
    if not resolved_layer_specs:
        resolved_layer_specs = [{"type": layer} for layer in resolved_layers]

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

    for layer_spec in resolved_layer_specs:
        layer_type = str(layer_spec["type"])
        if layer_type == "scatter":
            scatter_shape_count_range = shape_count_range
            scatter_min = layer_spec.get("scatter_shape_count_min")
            scatter_max = layer_spec.get("scatter_shape_count_max")
            if scatter_min is not None or scatter_max is not None:
                low = int(scatter_min if scatter_min is not None else scatter_max if scatter_max is not None else 4)
                high = int(scatter_max if scatter_max is not None else scatter_min if scatter_min is not None else low)
                if low > high:
                    low, high = high, low
                scatter_shape_count_range = (low, high)
            scatter_target_fill_ratio = (
                float(layer_spec["scatter_target_fill_ratio"])
                if "scatter_target_fill_ratio" in layer_spec
                else target_fill_ratio
            )
            _append_scatter_layer(
                lines=lines,
                shapes=all_shapes,
                rng=rng,
                width=width,
                height=height,
                level_count=level_count,
                max_shapes=max_shapes,
                shape_count_range=scatter_shape_count_range,
                target_fill_ratio=scatter_target_fill_ratio,
            )
        else:
            _append_segmented_tube_layer(
                lines=lines,
                shapes=all_shapes,
                rng=rng,
                width=width,
                height=height,
                tube_segment_count_range=tube_segment_count_range,
                tube_segment_count=layer_spec.get("tube_segment_count"),
                tube_stroke_width=layer_spec.get("tube_stroke_width"),
                tube_straightness=layer_spec.get("tube_straightness"),
            )

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
