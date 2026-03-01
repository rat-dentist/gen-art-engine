from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsPathItem,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollBar,
    QMainWindow,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from pipeline_quantize_vectorize import collect_shapes, load_image, quantize_levels, to_grayscale
from tube_trim import (
    build_all_to_all_morph_bank,
    generate_segmented_tube_contours,
    owned_boundary_masks_for_pieces,
    piece_rings,
    trim_overlapping_contours,
)

DEFAULT_TEST_IMAGE = Path(
    r"G:\My Drive\ART\_Source Imagery\14a6a32a-8830-4cf6-ba17-7b724781ba97.png"
)


class PieceItem(QGraphicsPathItem):
    def __init__(
        self,
        path: QPainterPath,
        mask: np.ndarray,
        boundary_mask: np.ndarray,
        offset_x: int,
        offset_y: int,
        on_position_changed: Callable[[], None],
    ) -> None:
        super().__init__(path)
        self.mask = mask
        self.boundary_mask = boundary_mask
        self.offset_x = int(offset_x)
        self.offset_y = int(offset_y)
        self._on_position_changed = on_position_changed
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)

    def itemChange(self, change, value):  # type: ignore[override]
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self._on_position_changed()
        return super().itemChange(change, value)


class CanvasView(QGraphicsView):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._panning = False
        self._pan_anchor = QPointF(0.0, 0.0)

    def _scroll_by(self, bar: QScrollBar, delta: float) -> None:
        bar.setValue(bar.value() + int(delta))

    def wheelEvent(self, event):  # type: ignore[override]
        if event.angleDelta().y() == 0:
            return super().wheelEvent(event)
        zoom_factor = 1.15 if event.angleDelta().y() > 0 else (1.0 / 1.15)
        self.scale(zoom_factor, zoom_factor)
        event.accept()

    def mousePressEvent(self, event):  # type: ignore[override]
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_anchor = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # type: ignore[override]
        if self._panning:
            delta = event.position() - self._pan_anchor
            self._pan_anchor = event.position()
            self._scroll_by(self.horizontalScrollBar(), -delta.x())
            self._scroll_by(self.verticalScrollBar(), -delta.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):  # type: ignore[override]
        if event.button() == Qt.MouseButton.MiddleButton and self._panning:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive prototype for tube overlap trim (Illustrator Pathfinder > Trim style)."
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Optional source image path. If omitted, uses the default test image then falls back to file picker.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for tube layout.")
    parser.add_argument("--segments", type=int, default=140, help="Tube segment count.")
    parser.add_argument("--straightness", type=float, default=0.45, help="Tube path straightness in [0.0, 1.0].")
    parser.add_argument("--stroke-width", type=float, default=1.2, help="Outline stroke width for pieces.")
    parser.add_argument(
        "--base-simplify",
        type=float,
        default=1.4,
        help="Epsilon in pixels for base contour simplification before tube generation.",
    )
    parser.add_argument(
        "--ring-simplify",
        type=float,
        default=0.9,
        help="Epsilon in pixels for display-time piece ring simplification.",
    )
    parser.add_argument(
        "--piece-min-area",
        type=float,
        default=18.0,
        help="Minimum trimmed fragment area in pixels to keep in preview.",
    )
    parser.add_argument(
        "--morph-steps",
        type=int,
        default=6,
        help="Tween steps between source shapes for all-to-all morphing. Use 0 to disable morphing.",
    )
    parser.add_argument(
        "--morph-shapes",
        type=int,
        default=4,
        help="How many source shapes (largest-by-area) to include in morphing.",
    )
    parser.add_argument(
        "--morph-points",
        type=int,
        default=96,
        help="Resampled point count per contour in the morph bank.",
    )
    return parser.parse_args()


def _pick_source_image(default_dir: Path) -> Path | None:
    path_str, _ = QFileDialog.getOpenFileName(
        None,
        "Load Source Image",
        str(default_dir),
        "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff)",
    )
    if not path_str:
        return None
    return Path(path_str)


class TubeTrimPreviewWindow(QMainWindow):
    LEVELS = 6
    MAX_TOTAL_CONTOURS = 10_000
    MIN_CONTOUR_AREA_RATIO = 0.00025
    MIN_CONTOUR_AREA_PX = 24.0
    MAX_PREVIEW_DIM = 1800

    def _new_seed_value(self) -> int:
        return int(time.time_ns() % 2_147_483_647)

    def _new_boundary_item(self) -> QGraphicsPixmapItem:
        item = QGraphicsPixmapItem()
        item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        item.setZValue(10_000.0)
        return item

    def __init__(
        self,
        image_path: Path,
        seed: int,
        segment_count: int,
        straightness: float,
        stroke_width: float,
        base_simplify: float,
        ring_simplify: float,
        piece_min_area: float,
        morph_steps: int,
        morph_shapes: int,
        morph_points: int,
    ) -> None:
        super().__init__()
        self.image_path = image_path
        self.seed = int(seed)
        self.segment_count = max(1, min(400, int(segment_count)))
        self.straightness = max(0.0, min(1.0, float(straightness)))
        self.stroke_width = max(0.1, float(stroke_width))
        self.base_simplify = max(0.0, float(base_simplify))
        self.ring_simplify = max(0.0, float(ring_simplify))
        self.piece_min_area = max(1.0, float(piece_min_area))
        self.morph_steps = max(0, int(morph_steps))
        self.morph_shapes = max(1, int(morph_shapes))
        self.morph_points = max(3, int(morph_points))

        self.setWindowTitle("Tube Trim Preview | drag pieces | Space: reseed | R: rebuild")
        self.resize(1280, 920)

        self.view = CanvasView(self)
        self.view.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.view.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.scene = QGraphicsScene(self)
        self.scene.setBackgroundBrush(QColor("#ffffff"))
        self.view.setScene(self.scene)
        controls_panel = self._build_controls_panel()
        root = QWidget(self)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(10)
        root_layout.addWidget(controls_panel)
        root_layout.addWidget(self.view, 1)
        self.setCentralWidget(root)

        self.canvas_width = 0
        self.canvas_height = 0
        self.piece_items: list[PieceItem] = []
        self.boundary_item = self._new_boundary_item()
        self.scene.addItem(self.boundary_item)
        self._did_initial_fit = False

        self.boundary_refresh_timer = QTimer(self)
        self.boundary_refresh_timer.setSingleShot(True)
        self.boundary_refresh_timer.setInterval(12)
        self.boundary_refresh_timer.timeout.connect(self._safe_refresh_boundary_overlay)

        self.rebuild_scene()

    def _build_controls_panel(self) -> QWidget:
        panel = QWidget(self)
        panel.setMinimumWidth(290)
        panel.setMaximumWidth(360)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(8)

        group = QGroupBox("Preview Controls")
        form = QFormLayout(group)

        self.seed_spin = QSpinBox(group)
        self.seed_spin.setRange(0, 2_147_483_647)
        self.seed_spin.setValue(self.seed)
        form.addRow("Seed", self.seed_spin)

        self.segments_spin = QSpinBox(group)
        self.segments_spin.setRange(1, 400)
        self.segments_spin.setValue(self.segment_count)
        form.addRow("Segments", self.segments_spin)

        self.straightness_spin = QDoubleSpinBox(group)
        self.straightness_spin.setRange(0.0, 1.0)
        self.straightness_spin.setDecimals(2)
        self.straightness_spin.setSingleStep(0.05)
        self.straightness_spin.setValue(self.straightness)
        form.addRow("Straightness", self.straightness_spin)

        self.stroke_width_spin = QDoubleSpinBox(group)
        self.stroke_width_spin.setRange(0.1, 12.0)
        self.stroke_width_spin.setDecimals(2)
        self.stroke_width_spin.setSingleStep(0.1)
        self.stroke_width_spin.setValue(self.stroke_width)
        form.addRow("Stroke Width", self.stroke_width_spin)

        self.base_simplify_spin = QDoubleSpinBox(group)
        self.base_simplify_spin.setRange(0.0, 24.0)
        self.base_simplify_spin.setDecimals(2)
        self.base_simplify_spin.setSingleStep(0.1)
        self.base_simplify_spin.setValue(self.base_simplify)
        form.addRow("Base Simplify", self.base_simplify_spin)

        self.ring_simplify_spin = QDoubleSpinBox(group)
        self.ring_simplify_spin.setRange(0.0, 24.0)
        self.ring_simplify_spin.setDecimals(2)
        self.ring_simplify_spin.setSingleStep(0.1)
        self.ring_simplify_spin.setValue(self.ring_simplify)
        form.addRow("Ring Simplify", self.ring_simplify_spin)

        self.piece_min_area_spin = QDoubleSpinBox(group)
        self.piece_min_area_spin.setRange(1.0, 5000.0)
        self.piece_min_area_spin.setDecimals(1)
        self.piece_min_area_spin.setSingleStep(1.0)
        self.piece_min_area_spin.setValue(self.piece_min_area)
        form.addRow("Piece Min Area", self.piece_min_area_spin)

        self.morph_steps_spin = QSpinBox(group)
        self.morph_steps_spin.setRange(0, 24)
        self.morph_steps_spin.setValue(self.morph_steps)
        form.addRow("Morph Steps", self.morph_steps_spin)

        self.morph_shapes_spin = QSpinBox(group)
        self.morph_shapes_spin.setRange(1, 16)
        self.morph_shapes_spin.setValue(self.morph_shapes)
        form.addRow("Morph Shapes", self.morph_shapes_spin)

        self.morph_points_spin = QSpinBox(group)
        self.morph_points_spin.setRange(8, 512)
        self.morph_points_spin.setSingleStep(8)
        self.morph_points_spin.setValue(self.morph_points)
        form.addRow("Morph Points", self.morph_points_spin)

        panel_layout.addWidget(group)

        buttons_group = QGroupBox("Actions")
        buttons_layout = QVBoxLayout(buttons_group)
        self.apply_button = QPushButton("Apply + Rebuild", buttons_group)
        self.apply_button.clicked.connect(self._apply_controls_and_rebuild)
        buttons_layout.addWidget(self.apply_button)

        self.reseed_button = QPushButton("Reseed + Rebuild", buttons_group)
        self.reseed_button.clicked.connect(self._on_reseed_clicked)
        buttons_layout.addWidget(self.reseed_button)

        self.fit_view_button = QPushButton("Fit View", buttons_group)
        self.fit_view_button.clicked.connect(self._fit_view)
        buttons_layout.addWidget(self.fit_view_button)

        panel_layout.addWidget(buttons_group)

        help_label = QLabel(
            "LMB drag pieces, wheel zoom, middle-mouse pan.\n"
            "Space reseeds, R rebuilds using current controls."
        )
        help_label.setWordWrap(True)
        panel_layout.addWidget(help_label)
        panel_layout.addStretch(1)
        return panel

    def _sync_state_from_controls(self) -> None:
        self.seed = int(self.seed_spin.value())
        self.segment_count = max(1, min(400, int(self.segments_spin.value())))
        self.straightness = max(0.0, min(1.0, float(self.straightness_spin.value())))
        self.stroke_width = max(0.1, float(self.stroke_width_spin.value()))
        self.base_simplify = max(0.0, float(self.base_simplify_spin.value()))
        self.ring_simplify = max(0.0, float(self.ring_simplify_spin.value()))
        self.piece_min_area = max(1.0, float(self.piece_min_area_spin.value()))
        self.morph_steps = max(0, int(self.morph_steps_spin.value()))
        self.morph_shapes = max(1, int(self.morph_shapes_spin.value()))
        self.morph_points = max(3, int(self.morph_points_spin.value()))

    def _apply_controls_and_rebuild(self) -> None:
        self._sync_state_from_controls()
        self.rebuild_scene()

    def _on_reseed_clicked(self) -> None:
        self.seed_spin.setValue(self._new_seed_value())
        self._apply_controls_and_rebuild()

    def _fit_view(self) -> None:
        self.view.resetTransform()
        rect = self.scene.sceneRect()
        if rect.width() > 0 and rect.height() > 0:
            self.view.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)

    def _choose_base_contours(self) -> tuple[int, int, list[np.ndarray]]:
        source = load_image(self.image_path)
        gray = to_grayscale(source)
        quantized = quantize_levels(gray, levels=self.LEVELS)
        shapes, _, _ = collect_shapes(
            quantized,
            levels=self.LEVELS,
            max_total_contours=self.MAX_TOTAL_CONTOURS,
            min_area_ratio=self.MIN_CONTOUR_AREA_RATIO,
            min_area_pixels=self.MIN_CONTOUR_AREA_PX,
        )
        if not shapes:
            raise RuntimeError("No usable vector contours found in this image.")

        def area_of(shape: dict[str, object]) -> float:
            area = shape.get("area")
            if area is not None:
                return abs(float(area))
            contour = shape["contour"]
            return abs(float(cv2.contourArea(contour)))

        sorted_shapes = sorted(shapes, key=area_of, reverse=True)
        selected_shapes = sorted_shapes[: min(len(sorted_shapes), self.morph_shapes)]
        width = int(source.width)
        height = int(source.height)
        contours = [np.asarray(shape["contour"], dtype=np.float32).copy() for shape in selected_shapes]
        if not contours:
            contours = [np.asarray(sorted_shapes[0]["contour"], dtype=np.float32).copy()]

        max_dim = max(width, height)
        if max_dim > self.MAX_PREVIEW_DIM:
            scale = float(self.MAX_PREVIEW_DIM) / float(max_dim)
            width = max(1, int(round(width * scale)))
            height = max(1, int(round(height * scale)))
            for contour in contours:
                contour *= scale
        return width, height, contours

    def rebuild_scene(self) -> None:
        width, height, base_contours = self._choose_base_contours()
        primary_contour = base_contours[0]
        contour_bank: list[np.ndarray] | None = None
        if self.morph_steps > 0 and len(base_contours) > 1:
            contour_bank = build_all_to_all_morph_bank(
                contours=base_contours,
                steps=self.morph_steps,
                point_count=self.morph_points,
                simplify_eps=self.base_simplify,
            )

        segments = generate_segmented_tube_contours(
            base_contour=primary_contour,
            width=width,
            height=height,
            seed=self.seed,
            segment_count=self.segment_count,
            straightness=self.straightness,
            contour_simplify_eps=self.base_simplify,
            contour_bank=contour_bank,
        )
        pieces = trim_overlapping_contours(
            segments,
            min_piece_area=self.piece_min_area,
            clip_bounds=(0, 0, int(width), int(height)),
        )
        piece_boundary_masks = owned_boundary_masks_for_pieces(
            pieces=pieces,
            width=int(width),
            height=int(height),
        )

        self.scene.clear()
        self.piece_items.clear()
        self.boundary_item = self._new_boundary_item()
        self.scene.addItem(self.boundary_item)
        self.scene.setSceneRect(0.0, 0.0, float(width), float(height))
        self.canvas_width = int(width)
        self.canvas_height = int(height)
        border_pen = QPen(QColor("#d0d0d0"))
        border_pen.setWidthF(1.0)
        self.scene.addRect(0.0, 0.0, float(width), float(height), border_pen)

        no_pen = QPen(Qt.PenStyle.NoPen)

        rendered = 0
        for index, piece in enumerate(pieces):
            path = QPainterPath()
            path.setFillRule(Qt.FillRule.OddEvenFill)
            for ring in piece_rings(piece, simplify_eps=self.ring_simplify):
                if ring.shape[0] < 3:
                    continue
                polygon = QPolygonF([QPointF(float(x), float(y)) for x, y in ring])
                ring_path = QPainterPath()
                ring_path.addPolygon(polygon)
                ring_path.closeSubpath()
                path.addPath(ring_path)
            if path.isEmpty():
                continue

            item = PieceItem(
                path=path,
                mask=piece.mask.copy(),
                boundary_mask=piece_boundary_masks[index].copy() if index < len(piece_boundary_masks) else np.zeros_like(piece.mask, dtype=np.uint8),
                offset_x=piece.offset_x,
                offset_y=piece.offset_y,
                on_position_changed=self._request_boundary_refresh,
            )
            item.setBrush(QBrush(QColor("#ffffff")))
            item.setPen(no_pen)
            item.setZValue(float(index))
            item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
            item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
            self.scene.addItem(item)
            self.piece_items.append(item)
            rendered += 1

        self._safe_refresh_boundary_overlay()
        self.statusBar().showMessage(
            f"seed={self.seed} | segments={len(segments)} | trimmed pieces={rendered} | "
            f"base_simplify={self.base_simplify:.2f} | ring_simplify={self.ring_simplify:.2f} | "
            f"min_area={self.piece_min_area:.1f} | morph_steps={self.morph_steps} | "
            f"morph_shapes={self.morph_shapes} | morph_points={self.morph_points} | "
            "Wheel zooms, MMB pans, Space reseeds, R rebuilds"
        )

    def _request_boundary_refresh(self) -> None:
        self.boundary_refresh_timer.start()

    def _safe_refresh_boundary_overlay(self) -> None:
        try:
            self._refresh_boundary_overlay()
        except Exception:
            print("Boundary refresh failed.", file=sys.stderr)
            traceback.print_exc()

    def _blit_mask(self, canvas: np.ndarray, mask: np.ndarray, x0: int, y0: int) -> None:
        mask_h, mask_w = mask.shape
        x1 = x0 + mask_w
        y1 = y0 + mask_h

        dx0 = max(0, x0)
        dy0 = max(0, y0)
        dx1 = min(self.canvas_width, x1)
        dy1 = min(self.canvas_height, y1)
        if dx0 >= dx1 or dy0 >= dy1:
            return

        sx0 = dx0 - x0
        sy0 = dy0 - y0
        sx1 = sx0 + (dx1 - dx0)
        sy1 = sy0 + (dy1 - dy0)

        dst = canvas[dy0:dy1, dx0:dx1]
        src = mask[sy0:sy1, sx0:sx1]
        np.maximum(dst, src, out=dst)

    def _refresh_boundary_overlay(self) -> None:
        if self.canvas_width <= 0 or self.canvas_height <= 0:
            self.boundary_item.setPixmap(QPixmap())
            return

        boundary = np.zeros((self.canvas_height, self.canvas_width), dtype=np.uint8)
        for item in self.piece_items:
            x = item.offset_x + int(round(float(item.pos().x())))
            y = item.offset_y + int(round(float(item.pos().y())))
            self._blit_mask(boundary, item.boundary_mask, x0=x, y0=y)

        stroke_px = max(1, int(round(float(self.stroke_width))))
        if stroke_px > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (stroke_px, stroke_px))
            boundary = cv2.dilate(boundary, kernel, iterations=1)
        rgba = np.zeros((self.canvas_height, self.canvas_width, 4), dtype=np.uint8)
        edge_pixels = boundary > 0
        rgba[edge_pixels, 3] = 255

        image = QImage(
            rgba.data,
            self.canvas_width,
            self.canvas_height,
            int(rgba.strides[0]),
            QImage.Format.Format_RGBA8888,
        ).copy()
        self.boundary_item.setPixmap(QPixmap.fromImage(image))
        self.boundary_item.setOffset(0.0, 0.0)

    def keyPressEvent(self, event):  # type: ignore[override]
        key = event.key()
        if key == Qt.Key.Key_Space:
            self.seed_spin.setValue(self._new_seed_value())
            self._apply_controls_and_rebuild()
            return
        if key == Qt.Key.Key_R:
            self._apply_controls_and_rebuild()
            return
        super().keyPressEvent(event)

    def showEvent(self, event):  # type: ignore[override]
        super().showEvent(event)
        if not self._did_initial_fit:
            QTimer.singleShot(0, self._fit_view)
            self._did_initial_fit = True


def main() -> None:
    args = parse_args()
    app = QApplication.instance() or QApplication([])

    image_path: Path | None
    if args.image:
        image_path = Path(args.image).expanduser().resolve()
    else:
        if DEFAULT_TEST_IMAGE.exists():
            image_path = DEFAULT_TEST_IMAGE
        else:
            picker_dir = DEFAULT_TEST_IMAGE.parent if DEFAULT_TEST_IMAGE.parent.exists() else Path.cwd()
            image_path = _pick_source_image(default_dir=picker_dir)

    if image_path is None:
        return
    if not image_path.exists():
        print(f"Missing Image: {image_path}", file=sys.stderr)
        return

    try:
        window = TubeTrimPreviewWindow(
            image_path=image_path,
            seed=args.seed,
            segment_count=args.segments,
            straightness=args.straightness,
            stroke_width=args.stroke_width,
            base_simplify=args.base_simplify,
            ring_simplify=args.ring_simplify,
            piece_min_area=args.piece_min_area,
            morph_steps=args.morph_steps,
            morph_shapes=args.morph_shapes,
            morph_points=args.morph_points,
        )
    except Exception:  # pragma: no cover - GUI startup path
        print("Preview Failed during startup.", file=sys.stderr)
        traceback.print_exc()
        return

    window.show()
    try:
        app.exec()
    except KeyboardInterrupt:  # pragma: no cover - interactive shutdown path
        return


if __name__ == "__main__":
    main()
