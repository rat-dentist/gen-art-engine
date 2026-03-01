from __future__ import annotations

import time
import traceback
from datetime import datetime
from pathlib import Path
from time import perf_counter

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
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
    COMPOSE_SCATTER_LAYER_COUNT = 1
    COMPOSE_TUBE_LAYER_COUNT = 2
    SCATTER_LAYER_DEFAULT_MIN_SHAPES = 4
    SCATTER_LAYER_DEFAULT_MAX_SHAPES = 8
    SCATTER_LAYER_DEFAULT_TARGET_FILL_RATIO = 0.50
    TUBE_LAYER_DEFAULT_REPETITIONS = 30
    TUBE_LAYER_DEFAULT_STROKE_WIDTH = 0.30
    TUBE_LAYER_DEFAULT_STRAIGHTNESS = 0.45
    DEFAULT_TOP_TUBE_REPETITIONS = 400
    DEFAULT_BOTTOM_TUBE_REPETITIONS = 50
    COMPOSE_TUBE_SEGMENT_COUNT_RANGE = (18, 42)
    COMPOSE_WHITE_BACKGROUND = True
    PREVIEW_DPI = 192
    EXPORT_DPI = 192

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

        self.scatter_enabled_checkbox = QCheckBox("Enable scatter")
        self.scatter_enabled_checkbox.setChecked(True)
        self.scatter_layer_count = QSpinBox()
        self.scatter_layer_count.setRange(1, 12)
        self.scatter_layer_count.setValue(self.COMPOSE_SCATTER_LAYER_COUNT)

        self.tube_enabled_checkbox = QCheckBox("Enable segmented tube")
        self.tube_enabled_checkbox.setChecked(True)
        self.tube_layer_count = QSpinBox()
        self.tube_layer_count.setRange(1, 12)
        self.tube_layer_count.setValue(self.COMPOSE_TUBE_LAYER_COUNT)

        self.layer_stack_list = QListWidget()
        self.layer_stack_list.setDragDropMode(QAbstractItemView.InternalMove)
        self.layer_stack_list.setDefaultDropAction(Qt.MoveAction)
        self.layer_stack_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.layer_stack_list.setMinimumHeight(140)
        self.layer_stack_list.currentItemChanged.connect(self.on_layer_selection_changed)
        self.layer_stack_list.model().rowsMoved.connect(self.on_layer_order_changed)

        self.rebuild_layers_button = QPushButton("Rebuild Layer Stack")
        self.rebuild_layers_button.clicked.connect(self.refresh_layer_stack)

        self.selected_layer_label = QLabel("No layer selected")
        self.layer_props_stack = QStackedWidget()

        self.layer_props_empty = QLabel("Select a layer to edit properties.")
        self.layer_props_empty.setAlignment(Qt.AlignCenter)
        self.layer_props_stack.addWidget(self.layer_props_empty)

        scatter_page = QWidget()
        scatter_form = QFormLayout(scatter_page)
        self.scatter_min_shapes_spin = QSpinBox()
        self.scatter_min_shapes_spin.setRange(1, 500)
        self.scatter_min_shapes_spin.setValue(self.SCATTER_LAYER_DEFAULT_MIN_SHAPES)
        self.scatter_min_shapes_spin.valueChanged.connect(self.on_scatter_layer_setting_changed)
        self.scatter_max_shapes_spin = QSpinBox()
        self.scatter_max_shapes_spin.setRange(1, 500)
        self.scatter_max_shapes_spin.setValue(self.SCATTER_LAYER_DEFAULT_MAX_SHAPES)
        self.scatter_max_shapes_spin.valueChanged.connect(self.on_scatter_layer_setting_changed)
        self.scatter_target_fill_spin = QDoubleSpinBox()
        self.scatter_target_fill_spin.setRange(0.05, 0.95)
        self.scatter_target_fill_spin.setDecimals(2)
        self.scatter_target_fill_spin.setSingleStep(0.01)
        self.scatter_target_fill_spin.setValue(self.SCATTER_LAYER_DEFAULT_TARGET_FILL_RATIO)
        self.scatter_target_fill_spin.valueChanged.connect(self.on_scatter_layer_setting_changed)
        scatter_form.addRow("Min Shapes", self.scatter_min_shapes_spin)
        scatter_form.addRow("Max Shapes", self.scatter_max_shapes_spin)
        scatter_form.addRow("Target Fill", self.scatter_target_fill_spin)
        self.layer_props_stack.addWidget(scatter_page)

        tube_page = QWidget()
        tube_form = QFormLayout(tube_page)
        self.tube_repetitions_spin = QSpinBox()
        self.tube_repetitions_spin.setRange(1, 400)
        self.tube_repetitions_spin.setValue(self.TUBE_LAYER_DEFAULT_REPETITIONS)
        self.tube_repetitions_spin.valueChanged.connect(self.on_tube_layer_setting_changed)
        self.tube_stroke_width_spin = QDoubleSpinBox()
        self.tube_stroke_width_spin.setRange(0.10, 12.0)
        self.tube_stroke_width_spin.setDecimals(2)
        self.tube_stroke_width_spin.setSingleStep(0.05)
        self.tube_stroke_width_spin.setValue(self.TUBE_LAYER_DEFAULT_STROKE_WIDTH)
        self.tube_stroke_width_spin.valueChanged.connect(self.on_tube_layer_setting_changed)
        self.tube_straightness_spin = QDoubleSpinBox()
        self.tube_straightness_spin.setRange(0.0, 1.0)
        self.tube_straightness_spin.setDecimals(2)
        self.tube_straightness_spin.setSingleStep(0.05)
        self.tube_straightness_spin.setValue(self.TUBE_LAYER_DEFAULT_STRAIGHTNESS)
        self.tube_straightness_spin.valueChanged.connect(self.on_tube_layer_setting_changed)
        tube_form.addRow("Repetitions", self.tube_repetitions_spin)
        tube_form.addRow("Outline Width", self.tube_stroke_width_spin)
        tube_form.addRow("Straightness", self.tube_straightness_spin)
        self.layer_props_stack.addWidget(tube_page)
        self.layer_props_stack.setCurrentIndex(0)

        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.setInterval(220)
        self.preview_timer.timeout.connect(self.on_live_preview_timeout)

        self.scatter_enabled_checkbox.toggled.connect(self.refresh_layer_stack)
        self.scatter_layer_count.valueChanged.connect(self.refresh_layer_stack)
        self.tube_enabled_checkbox.toggled.connect(self.refresh_layer_stack)
        self.tube_layer_count.valueChanged.connect(self.refresh_layer_stack)
        self.seed_input.valueChanged.connect(self.request_live_preview)

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
        scatter_row = QHBoxLayout()
        scatter_row.addWidget(self.scatter_enabled_checkbox)
        scatter_row.addWidget(self.scatter_layer_count)
        controls_layout.addRow("Scatter Layers", scatter_row)
        tube_row = QHBoxLayout()
        tube_row.addWidget(self.tube_enabled_checkbox)
        tube_row.addWidget(self.tube_layer_count)
        controls_layout.addRow("Tube Layers", tube_row)
        controls_layout.addRow("Layer Stack", self.layer_stack_list)
        controls_layout.addRow("", self.rebuild_layers_button)
        controls_layout.addRow("Selected Layer", self.selected_layer_label)
        controls_layout.addRow("Layer Properties", self.layer_props_stack)
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

        self.refresh_layer_stack()
        self.apply_default_three_layer_preset()
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
        if pixmap.width() > width or pixmap.height() > height:
            mode = Qt.SmoothTransformation
        else:
            mode = Qt.FastTransformation
        scaled = pixmap.scaled(width, height, Qt.KeepAspectRatio, mode)
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
        self.request_live_preview()

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
        self.request_live_preview()

    def on_pick_output_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Output Folder",
            self.output_dir_edit.text().strip() or str(Path("output")),
        )
        if folder:
            self.output_dir_edit.setText(folder)
            self._log(f"Output folder set: {folder}")
            self.request_live_preview()

    def _resolve_seed(self) -> int:
        if self.random_seed_checkbox.isChecked():
            seed = self._new_seed()
            self.seed_input.setValue(seed)
            return seed
        return int(self.seed_input.value())

    def _make_scatter_layer_data(self) -> dict[str, object]:
        return {
            "type": "scatter",
            "scatter_shape_count_min": int(self.SCATTER_LAYER_DEFAULT_MIN_SHAPES),
            "scatter_shape_count_max": int(self.SCATTER_LAYER_DEFAULT_MAX_SHAPES),
            "scatter_target_fill_ratio": float(self.SCATTER_LAYER_DEFAULT_TARGET_FILL_RATIO),
        }

    def _make_tube_layer_data(self) -> dict[str, object]:
        return {
            "type": "segmented_tube",
            "tube_segment_count": int(self.TUBE_LAYER_DEFAULT_REPETITIONS),
            "tube_stroke_width": float(self.TUBE_LAYER_DEFAULT_STROKE_WIDTH),
            "tube_straightness": float(self.TUBE_LAYER_DEFAULT_STRAIGHTNESS),
        }

    def _make_tube_layer_data_with_repetitions(self, repetitions: int) -> dict[str, object]:
        data = self._make_tube_layer_data()
        data["tube_segment_count"] = max(1, min(400, int(repetitions)))
        return data

    def apply_default_three_layer_preset(self) -> None:
        self.layer_stack_list.clear()
        top_tube_item = QListWidgetItem()
        top_tube_item.setData(
            Qt.UserRole,
            self._make_tube_layer_data_with_repetitions(self.DEFAULT_TOP_TUBE_REPETITIONS),
        )
        middle_scatter_item = QListWidgetItem()
        middle_scatter_item.setData(Qt.UserRole, self._make_scatter_layer_data())
        bottom_tube_item = QListWidgetItem()
        bottom_tube_item.setData(
            Qt.UserRole,
            self._make_tube_layer_data_with_repetitions(self.DEFAULT_BOTTOM_TUBE_REPETITIONS),
        )

        # UI is shown top-to-bottom. Rendering order is reversed in _current_layer_specs.
        self.layer_stack_list.addItem(top_tube_item)
        self.layer_stack_list.addItem(middle_scatter_item)
        self.layer_stack_list.addItem(bottom_tube_item)
        self._refresh_layer_stack_labels()
        self.layer_stack_list.setCurrentRow(0)

    def _set_layer_item_text(self, item: QListWidgetItem, index: int) -> None:
        data = item.data(Qt.UserRole)
        if not isinstance(data, dict):
            item.setText(f"Layer {index + 1}")
            return
        count = self.layer_stack_list.count()
        if count <= 1:
            position_label = "Single"
        elif index == 0:
            position_label = "Top"
        elif index == count - 1:
            position_label = "Bottom"
        else:
            position_label = "Middle"
        layer_type = str(data.get("type", "")).strip().lower()
        if layer_type == "scatter":
            min_shapes = int(data.get("scatter_shape_count_min", self.SCATTER_LAYER_DEFAULT_MIN_SHAPES))
            max_shapes = int(data.get("scatter_shape_count_max", self.SCATTER_LAYER_DEFAULT_MAX_SHAPES))
            if min_shapes > max_shapes:
                min_shapes, max_shapes = max_shapes, min_shapes
            item.setText(f"{position_label}: Scatter ({min_shapes}-{max_shapes} shapes)")
            return
        repetitions = int(data.get("tube_segment_count", self.TUBE_LAYER_DEFAULT_REPETITIONS))
        stroke_width = float(data.get("tube_stroke_width", self.TUBE_LAYER_DEFAULT_STROKE_WIDTH))
        straightness = float(data.get("tube_straightness", self.TUBE_LAYER_DEFAULT_STRAIGHTNESS))
        item.setText(
            f"{position_label}: Tube (reps={repetitions}, stroke={stroke_width:.2f}, straight={straightness:.2f})"
        )

    def _refresh_layer_stack_labels(self) -> None:
        for idx in range(self.layer_stack_list.count()):
            self._set_layer_item_text(self.layer_stack_list.item(idx), idx)

    def on_layer_selection_changed(self, current, _previous) -> None:
        if current is None:
            self.selected_layer_label.setText("No layer selected")
            self.layer_props_stack.setCurrentIndex(0)
            return
        data = current.data(Qt.UserRole)
        if not isinstance(data, dict):
            self.selected_layer_label.setText("Unknown layer")
            self.layer_props_stack.setCurrentIndex(0)
            return
        layer_type = str(data.get("type", "")).strip().lower()
        if layer_type == "scatter":
            self.selected_layer_label.setText("Scatter")
            self.layer_props_stack.setCurrentIndex(1)
            min_shapes = int(data.get("scatter_shape_count_min", self.SCATTER_LAYER_DEFAULT_MIN_SHAPES))
            max_shapes = int(data.get("scatter_shape_count_max", self.SCATTER_LAYER_DEFAULT_MAX_SHAPES))
            target_fill = float(data.get("scatter_target_fill_ratio", self.SCATTER_LAYER_DEFAULT_TARGET_FILL_RATIO))
            self.scatter_min_shapes_spin.blockSignals(True)
            self.scatter_min_shapes_spin.setValue(min_shapes)
            self.scatter_min_shapes_spin.blockSignals(False)
            self.scatter_max_shapes_spin.blockSignals(True)
            self.scatter_max_shapes_spin.setValue(max_shapes)
            self.scatter_max_shapes_spin.blockSignals(False)
            self.scatter_target_fill_spin.blockSignals(True)
            self.scatter_target_fill_spin.setValue(target_fill)
            self.scatter_target_fill_spin.blockSignals(False)
            return

        repetitions = int(data.get("tube_segment_count", self.TUBE_LAYER_DEFAULT_REPETITIONS))
        stroke_width = float(data.get("tube_stroke_width", self.TUBE_LAYER_DEFAULT_STROKE_WIDTH))
        straightness = float(data.get("tube_straightness", self.TUBE_LAYER_DEFAULT_STRAIGHTNESS))
        self.selected_layer_label.setText("Segmented tube")
        self.layer_props_stack.setCurrentIndex(2)
        self.tube_repetitions_spin.blockSignals(True)
        self.tube_repetitions_spin.setValue(repetitions)
        self.tube_repetitions_spin.blockSignals(False)
        self.tube_stroke_width_spin.blockSignals(True)
        self.tube_stroke_width_spin.setValue(stroke_width)
        self.tube_stroke_width_spin.blockSignals(False)
        self.tube_straightness_spin.blockSignals(True)
        self.tube_straightness_spin.setValue(straightness)
        self.tube_straightness_spin.blockSignals(False)

    def on_scatter_layer_setting_changed(self, _value) -> None:
        item = self.layer_stack_list.currentItem()
        if item is None:
            return
        data = item.data(Qt.UserRole)
        if not isinstance(data, dict):
            return
        layer_type = str(data.get("type", "")).strip().lower()
        if layer_type != "scatter":
            return
        min_shapes = int(self.scatter_min_shapes_spin.value())
        max_shapes = int(self.scatter_max_shapes_spin.value())
        if min_shapes > max_shapes:
            max_shapes = min_shapes
            self.scatter_max_shapes_spin.blockSignals(True)
            self.scatter_max_shapes_spin.setValue(max_shapes)
            self.scatter_max_shapes_spin.blockSignals(False)
        updated = dict(data)
        updated["scatter_shape_count_min"] = min_shapes
        updated["scatter_shape_count_max"] = max_shapes
        updated["scatter_target_fill_ratio"] = float(self.scatter_target_fill_spin.value())
        item.setData(Qt.UserRole, updated)
        self._refresh_layer_stack_labels()
        self.request_live_preview()

    def on_tube_layer_setting_changed(self, _value) -> None:
        item = self.layer_stack_list.currentItem()
        if item is None:
            return
        data = item.data(Qt.UserRole)
        if not isinstance(data, dict):
            return
        layer_type = str(data.get("type", "")).strip().lower()
        if layer_type != "segmented_tube":
            return
        updated = dict(data)
        updated["tube_segment_count"] = int(self.tube_repetitions_spin.value())
        updated["tube_stroke_width"] = float(self.tube_stroke_width_spin.value())
        updated["tube_straightness"] = float(self.tube_straightness_spin.value())
        item.setData(Qt.UserRole, updated)
        self._refresh_layer_stack_labels()
        self.request_live_preview()

    def on_layer_order_changed(self, *_args) -> None:
        self._refresh_layer_stack_labels()
        self.request_live_preview()

    def refresh_layer_stack(self, *_args) -> None:
        self.scatter_layer_count.setEnabled(self.scatter_enabled_checkbox.isChecked())
        self.tube_layer_count.setEnabled(self.tube_enabled_checkbox.isChecked())

        self.layer_stack_list.clear()
        if self.scatter_enabled_checkbox.isChecked():
            for idx in range(int(self.scatter_layer_count.value())):
                item = QListWidgetItem()
                item.setData(Qt.UserRole, self._make_scatter_layer_data())
                self.layer_stack_list.addItem(item)
        if self.tube_enabled_checkbox.isChecked():
            for idx in range(int(self.tube_layer_count.value())):
                item = QListWidgetItem()
                item.setData(Qt.UserRole, self._make_tube_layer_data())
                self.layer_stack_list.addItem(item)
        self._refresh_layer_stack_labels()
        if self.layer_stack_list.count() > 0:
            self.layer_stack_list.setCurrentRow(0)
        else:
            self.on_layer_selection_changed(None, None)
        self.request_live_preview()

    def _current_layer_specs(self, for_render: bool = True) -> list[dict[str, object]]:
        layers: list[dict[str, object]] = []
        indices = (
            range(self.layer_stack_list.count() - 1, -1, -1)
            if for_render
            else range(self.layer_stack_list.count())
        )
        for idx in indices:
            item = self.layer_stack_list.item(int(idx))
            data = item.data(Qt.UserRole)
            if not isinstance(data, dict):
                continue
            layer_type = str(data.get("type", "")).strip().lower()
            if layer_type not in {"scatter", "segmented_tube"}:
                continue
            spec: dict[str, object] = {"type": layer_type}
            if layer_type == "scatter":
                min_shapes = int(data.get("scatter_shape_count_min", self.SCATTER_LAYER_DEFAULT_MIN_SHAPES))
                max_shapes = int(data.get("scatter_shape_count_max", self.SCATTER_LAYER_DEFAULT_MAX_SHAPES))
                if min_shapes > max_shapes:
                    min_shapes, max_shapes = max_shapes, min_shapes
                spec["scatter_shape_count_min"] = min_shapes
                spec["scatter_shape_count_max"] = max_shapes
                spec["scatter_target_fill_ratio"] = float(
                    data.get("scatter_target_fill_ratio", self.SCATTER_LAYER_DEFAULT_TARGET_FILL_RATIO)
                )
            else:
                spec["tube_segment_count"] = int(data.get("tube_segment_count", self.TUBE_LAYER_DEFAULT_REPETITIONS))
                spec["tube_stroke_width"] = float(
                    data.get("tube_stroke_width", self.TUBE_LAYER_DEFAULT_STROKE_WIDTH)
                )
                spec["tube_straightness"] = float(
                    data.get("tube_straightness", self.TUBE_LAYER_DEFAULT_STRAIGHTNESS)
                )
            layers.append(spec)
        return layers

    def request_live_preview(self) -> None:
        if self.source_path is None:
            return
        self.preview_timer.start()

    def on_live_preview_timeout(self) -> None:
        self._render_composition(save_outputs=False)

    def _render_composition(self, save_outputs: bool) -> None:
        if self.source_path is None:
            if save_outputs:
                QMessageBox.warning(self, "Missing Image", "Load a source image first.")
            return

        output_dir_text = self.output_dir_edit.text().strip() or "output"
        output_dir = Path(output_dir_text)
        output_dir.mkdir(parents=True, exist_ok=True)

        seed = self._resolve_seed() if save_outputs else int(self.seed_input.value())
        layer_specs = self._current_layer_specs(for_render=True)
        if not layer_specs:
            if save_outputs:
                QMessageBox.warning(self, "Missing Layers", "Enable at least one layer type.")
            return
        layer_specs_ui = self._current_layer_specs(for_render=False)

        project_slug = get_project_slug()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = f"{project_slug}_{stamp}_seed{seed}"
        svg_path = output_dir / f"{filename_base}.svg"
        png_path = output_dir / f"{filename_base}.png"
        live_png_path = output_dir / "__live_preview.png"

        if save_outputs:
            self._log(f"Generate started | seed={seed} | levels={self.LEVELS}")
            layer_summary_parts: list[str] = []
            for idx, layer in enumerate(layer_specs_ui):
                layer_type = str(layer["type"])
                if layer_type == "scatter":
                    low = int(layer.get("scatter_shape_count_min", self.SCATTER_LAYER_DEFAULT_MIN_SHAPES))
                    high = int(layer.get("scatter_shape_count_max", self.SCATTER_LAYER_DEFAULT_MAX_SHAPES))
                    layer_summary_parts.append(f"{idx + 1}:scatter({low}-{high})")
                else:
                    reps = int(layer.get("tube_segment_count", self.TUBE_LAYER_DEFAULT_REPETITIONS))
                    stroke = float(layer.get("tube_stroke_width", self.TUBE_LAYER_DEFAULT_STROKE_WIDTH))
                    straight = float(layer.get("tube_straightness", self.TUBE_LAYER_DEFAULT_STRAIGHTNESS))
                    layer_summary_parts.append(
                        f"{idx + 1}:tube(reps={reps},stroke={stroke:.2f},straight={straight:.2f})"
                    )
            self._log(
                "Composition settings | "
                f"layers={' > '.join(layer_summary_parts)} | "
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
                target_fill_ratio=None,
                shape_count_range=None,
                tube_segment_count_range=self.COMPOSE_TUBE_SEGMENT_COUNT_RANGE,
                layer_specs=layer_specs,
                white_background=self.COMPOSE_WHITE_BACKGROUND,
            )
            timings["build_svg"] = perf_counter() - t0

            if save_outputs:
                t0 = perf_counter()
                svg_path.write_text(svg_str, encoding="utf-8")
                timings["write_svg"] = perf_counter() - t0

            t0 = perf_counter()
            svg_to_png(
                svg_str,
                png_path if save_outputs else live_png_path,
                dpi=self.EXPORT_DPI if save_outputs else self.PREVIEW_DPI,
            )
            timings["svg_to_png"] = perf_counter() - t0

            self._set_output_preview(png_path if save_outputs else live_png_path)

            if save_outputs:
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
            if save_outputs:
                self._log(f"Error: {exc}")
                self._log(traceback.format_exc())
                QMessageBox.critical(self, "Generation Failed", str(exc))

    def on_generate(self) -> None:
        self._render_composition(save_outputs=True)


def main() -> None:
    app = QApplication.instance() or QApplication([])
    window = ArtUiWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
