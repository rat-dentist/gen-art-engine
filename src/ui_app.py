from __future__ import annotations

import time
import traceback
from datetime import datetime
from pathlib import Path
from time import perf_counter

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from kit.naming import get_project_slug
from pipeline_quantize_vectorize import (
    build_svg,
    collect_shapes,
    load_image,
    quantize_levels,
    svg_to_png,
    to_grayscale,
)


class ArtUiWindow(QMainWindow):
    LEVELS = 6
    MAX_TOTAL_CONTOURS = 10_000
    MIN_CONTOUR_AREA_RATIO = 0.00025
    MIN_CONTOUR_AREA_PX = 24.0

    COMPOSE_SHAPE_COUNT_RANGE = (4, 8)
    COMPOSE_HERO_TARGET_FILL_RATIO = 0.50
    COMPOSE_WHITE_BACKGROUND = True

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Gen Art Quantize + Vectorize")
        self.resize(1360, 900)
        self.source_path: Path | None = None
        self._source_preview_path: Path | None = None
        self._output_preview_path: Path | None = None

        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.on_load_image)

        self.source_label = QLineEdit()
        self.source_label.setPlaceholderText("No image loaded")
        self.source_label.setReadOnly(True)

        self.generate_button = QPushButton("Generate")
        self.generate_button.setMinimumWidth(150)
        self.generate_button.clicked.connect(self.on_generate)

        self.levels_label = QLabel(str(self.LEVELS))

        self.seed_input = QSpinBox()
        self.seed_input.setRange(0, 2_147_483_647)
        self.seed_input.setValue(self._new_seed())

        self.random_seed_checkbox = QCheckBox("Randomize seed on run")
        self.random_seed_checkbox.setChecked(True)
        self.random_seed_checkbox.toggled.connect(self.on_random_seed_toggled)
        self.seed_input.setEnabled(False)

        self.output_dir_edit = QLineEdit(str(Path("output")))
        self.output_pick_button = QPushButton("Pick Output Folder")
        self.output_pick_button.clicked.connect(self.on_pick_output_folder)

        self.source_preview = self._create_preview_label("Source Thumbnail", min_w=220, min_h=170)
        self.source_preview.setMaximumHeight(240)
        self.output_preview = self._create_preview_label("Output Preview", min_w=820, min_h=600)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setPlaceholderText("Status log")
        self.log_area.setMinimumHeight(120)
        self.log_area.setMaximumHeight(220)

        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)

        top_row = QHBoxLayout()
        top_row.addWidget(self.load_button)
        top_row.addWidget(self.source_label, stretch=1)
        top_row.addWidget(self.generate_button)
        root_layout.addLayout(top_row)

        controls_box = QGroupBox("Controls")
        controls_layout = QFormLayout(controls_box)
        controls_layout.addRow("Levels", self.levels_label)
        controls_layout.addRow("Seed", self.seed_input)
        controls_layout.addRow("", self.random_seed_checkbox)
        out_row = QHBoxLayout()
        out_row.addWidget(self.output_dir_edit, stretch=1)
        out_row.addWidget(self.output_pick_button)
        controls_layout.addRow("Output Folder", out_row)

        source_box = QGroupBox("Input Thumbnail")
        source_layout = QVBoxLayout(source_box)
        source_layout.addWidget(self.source_preview)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(controls_box)
        left_layout.addWidget(source_box)
        left_layout.addStretch(1)

        output_box = QGroupBox("Output Preview")
        output_layout = QVBoxLayout(output_box)
        output_layout.addWidget(self.output_preview, stretch=1)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(output_box)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 1040])
        root_layout.addWidget(splitter, stretch=1)

        log_box = QGroupBox("Status Log")
        log_layout = QVBoxLayout(log_box)
        log_layout.addWidget(self.log_area)
        root_layout.addWidget(log_box)

        self._log("Ready. Load an image to begin.")

    def _new_seed(self) -> int:
        return int(time.time_ns() % 2_147_483_647)

    def _create_preview_label(self, title: str, min_w: int, min_h: int) -> QLabel:
        label = QLabel(title)
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumSize(min_w, min_h)
        label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        label.setStyleSheet("border: 1px solid #777; background: #181818; color: #dddddd;")
        return label

    def _render_preview(self, label: QLabel, image_path: Path) -> None:
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            label.setPixmap(QPixmap())
            label.setText("Preview unavailable")
            return
        width = max(1, label.contentsRect().width())
        height = max(1, label.contentsRect().height())
        scaled = pixmap.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setText("")
        label.setPixmap(scaled)

    def _set_source_preview(self, image_path: Path) -> None:
        self._source_preview_path = image_path
        self._render_preview(self.source_preview, image_path)

    def _set_output_preview(self, image_path: Path) -> None:
        self._output_preview_path = image_path
        self._render_preview(self.output_preview, image_path)

    def _log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_area.append(f"[{timestamp}] {message}")

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if self._source_preview_path is not None:
            self._render_preview(self.source_preview, self._source_preview_path)
        if self._output_preview_path is not None:
            self._render_preview(self.output_preview, self._output_preview_path)

    def on_random_seed_toggled(self, checked: bool) -> None:
        self.seed_input.setEnabled(not checked)
        if checked:
            self.seed_input.setValue(self._new_seed())

    def on_load_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Source Image",
            str(Path.cwd()),
            "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff)",
        )
        if not file_path:
            return
        self.source_path = Path(file_path)
        self.source_label.setText(str(self.source_path))
        self._set_source_preview(self.source_path)
        self._log(f"Loaded image: {self.source_path.name}")

    def on_pick_output_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Output Folder",
            self.output_dir_edit.text().strip() or str(Path("output")),
        )
        if folder:
            self.output_dir_edit.setText(folder)
            self._log(f"Output folder set: {folder}")

    def _resolve_seed(self) -> int:
        if self.random_seed_checkbox.isChecked():
            seed = self._new_seed()
            self.seed_input.setValue(seed)
            return seed
        return int(self.seed_input.value())

    def on_generate(self) -> None:
        if self.source_path is None:
            QMessageBox.warning(self, "Missing Image", "Load a source image first.")
            return

        output_dir_text = self.output_dir_edit.text().strip() or "output"
        output_dir = Path(output_dir_text)
        output_dir.mkdir(parents=True, exist_ok=True)

        seed = self._resolve_seed()
        project_slug = get_project_slug()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = f"{project_slug}_{stamp}_seed{seed}"
        svg_path = output_dir / f"{filename_base}.svg"
        png_path = output_dir / f"{filename_base}.png"

        self._log(f"Generate started | seed={seed} | levels={self.LEVELS}")
        self._log(
            "Composition settings | "
            f"shape_count_range={self.COMPOSE_SHAPE_COUNT_RANGE[0]}-{self.COMPOSE_SHAPE_COUNT_RANGE[1]} | "
            f"hero_target_fill_ratio={self.COMPOSE_HERO_TARGET_FILL_RATIO:.2f} | "
            f"white_background={self.COMPOSE_WHITE_BACKGROUND}"
        )
        try:
            timings: dict[str, float] = {}

            t0 = perf_counter()
            img = load_image(self.source_path)
            timings["load_image"] = perf_counter() - t0

            t0 = perf_counter()
            gray = to_grayscale(img)
            timings["to_grayscale"] = perf_counter() - t0

            t0 = perf_counter()
            quantized = quantize_levels(gray, levels=self.LEVELS)
            timings["quantize_levels"] = perf_counter() - t0

            t0 = perf_counter()
            shapes, counts_by_level, capped = collect_shapes(
                quantized,
                levels=self.LEVELS,
                max_total_contours=self.MAX_TOTAL_CONTOURS,
                min_area_ratio=self.MIN_CONTOUR_AREA_RATIO,
                min_area_pixels=self.MIN_CONTOUR_AREA_PX,
            )
            timings["contours"] = perf_counter() - t0

            t0 = perf_counter()
            svg_str = build_svg(
                shapes=shapes,
                width=img.width,
                height=img.height,
                seed=seed,
                arrangement="scatter",
                levels=self.LEVELS,
                target_fill_ratio=self.COMPOSE_HERO_TARGET_FILL_RATIO,
                shape_count_range=self.COMPOSE_SHAPE_COUNT_RANGE,
                white_background=self.COMPOSE_WHITE_BACKGROUND,
            )
            timings["build_svg"] = perf_counter() - t0

            t0 = perf_counter()
            svg_path.write_text(svg_str, encoding="utf-8")
            timings["write_svg"] = perf_counter() - t0

            t0 = perf_counter()
            svg_to_png(svg_str, png_path)
            timings["svg_to_png"] = perf_counter() - t0

            self._set_output_preview(png_path)

            self._log(f"Saved SVG: {svg_path}")
            self._log(f"Saved PNG: {png_path}")
            self._log(f"Usable shapes after filtering: {len(shapes)}")
            self._log(f"Composed shapes in output: {svg_str.count('<path ')}")
            for level in range(self.LEVELS):
                self._log(f"Level {level}: {counts_by_level.get(level, 0)} usable contours")
            if capped:
                self._log(f"Contour cap applied at {self.MAX_TOTAL_CONTOURS} shapes.")
            for stage, elapsed in timings.items():
                self._log(f"{stage}: {elapsed:.3f}s")

        except Exception as exc:  # pragma: no cover - GUI error path
            self._log(f"Error: {exc}")
            self._log(traceback.format_exc())
            QMessageBox.critical(self, "Generation Failed", str(exc))


def main() -> None:
    app = QApplication.instance() or QApplication([])
    window = ArtUiWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
