from __future__ import annotations

import base64
import io
import time
import traceback
from datetime import datetime
from pathlib import Path
from time import perf_counter

from PIL import Image, ImageFilter, ImageOps
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QFontMetrics, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFrame,
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
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QSplitter,
    QToolButton,
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
    COMPOSE_TRIMMED_TUBE_LAYER_COUNT = 1
    SCATTER_LAYER_DEFAULT_MIN_SHAPES = 4
    SCATTER_LAYER_DEFAULT_MAX_SHAPES = 8
    SCATTER_LAYER_DEFAULT_TARGET_FILL_RATIO = 0.50
    TUBE_LAYER_MAX_REPETITIONS = 1000
    TUBE_LAYER_DEFAULT_REPETITIONS = 30
    TUBE_LAYER_DEFAULT_STROKE_WIDTH = 0.30
    TUBE_LAYER_DEFAULT_STRAIGHTNESS = 0.45
    TRIMMED_TUBE_LAYER_DEFAULT_REPETITIONS = 140
    TRIMMED_TUBE_LAYER_DEFAULT_STROKE_WIDTH = 1.20
    TRIMMED_TUBE_LAYER_DEFAULT_STRAIGHTNESS = 0.45
    TRIMMED_TUBE_LAYER_DEFAULT_SCALE = 1.00
    TRIMMED_TUBE_LAYER_DEFAULT_BASE_SIMPLIFY = 1.40
    TRIMMED_TUBE_LAYER_DEFAULT_RING_SIMPLIFY = 0.90
    TRIMMED_TUBE_LAYER_DEFAULT_MIN_PIECE_AREA = 18.0
    TRIMMED_TUBE_LAYER_DEFAULT_MORPH_STEPS = 6
    TRIMMED_TUBE_LAYER_DEFAULT_MORPH_SHAPES = 4
    TRIMMED_TUBE_LAYER_DEFAULT_MORPH_POINTS = 96
    DEFAULT_TOP_TUBE_REPETITIONS = 400
    DEFAULT_BOTTOM_TUBE_REPETITIONS = 50
    COMPOSE_TUBE_SEGMENT_COUNT_RANGE = (18, 42)
    COMPOSE_WHITE_BACKGROUND = True
    PREVIEW_DPI = 192
    EXPORT_DPI = 192
    PREVIEW_SCALE = 1.0
    PREVIEW_MAX_DIM = 1400
    PREVIEW_MAX_PIXELS = 1_800_000
    GENERATE_MAX_DIM = 2600
    GENERATE_MAX_PIXELS = 5_200_000
    CANVAS_RATIO_WIDTH = 3
    CANVAS_RATIO_HEIGHT = 4
    BACKGROUND_BLUR_RADIUS = 1.2
    BACKGROUND_IMAGE_OPACITY = 1.00
    DEFAULT_SOURCE_IMAGE_DIR = Path(r"G:\My Drive\ART\_Source Imagery")

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Gen Art Quantize + Vectorize")
        self.resize(1360, 900)
        self.source_path: Path | None = None
        self._source_preview_path: Path | None = None
        self._output_preview_path: Path | None = None
        self._is_rendering = False
        self._pending_live_preview = False
        self._pending_generate = False

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

        self.canvas_ratio_checkbox = QCheckBox("Use 3:4 canvas")
        self.canvas_ratio_checkbox.setChecked(True)
        self.canvas_ratio_checkbox.toggled.connect(self.request_live_preview)

        self.background_source_checkbox = QCheckBox("Use source image in background")
        self.background_source_checkbox.setChecked(True)
        self.background_source_checkbox.toggled.connect(self.on_background_source_toggled)

        self.background_blur_checkbox = QCheckBox("Apply subtle Gaussian blur")
        self.background_blur_checkbox.setChecked(True)
        self.background_blur_checkbox.toggled.connect(self.request_live_preview)

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

        self.trimmed_tube_enabled_checkbox = QCheckBox("Enable trimmed morph tube")
        self.trimmed_tube_enabled_checkbox.setChecked(False)
        self.trimmed_tube_layer_count = QSpinBox()
        self.trimmed_tube_layer_count.setRange(1, 12)
        self.trimmed_tube_layer_count.setValue(self.COMPOSE_TRIMMED_TUBE_LAYER_COUNT)

        self.layer_stack_list = QListWidget()
        self.layer_stack_list.setDragDropMode(QAbstractItemView.InternalMove)
        self.layer_stack_list.setDefaultDropAction(Qt.MoveAction)
        self.layer_stack_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.layer_stack_list.setMinimumHeight(140)
        self.layer_stack_list.setTextElideMode(Qt.ElideRight)
        self.layer_stack_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.layer_stack_list.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.layer_stack_list.setUniformItemSizes(True)
        self.layer_stack_list.setSpacing(2)
        self.layer_stack_list.currentItemChanged.connect(self.on_layer_selection_changed)
        self.layer_stack_list.model().rowsMoved.connect(self.on_layer_order_changed)

        self.rebuild_layers_button = QPushButton("Rebuild Layer Stack")
        self.rebuild_layers_button.clicked.connect(self.refresh_layer_stack)

        self.selected_layer_label = QLabel("No layer selected")
        self.selected_layer_label.setWordWrap(True)
        self.layer_props_stack = QStackedWidget()
        self.layer_props_toggle_button = QToolButton()
        self.layer_props_scroll = QScrollArea()
        self.layer_editor_panel = QWidget()

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
        self.tube_repetitions_spin.setRange(1, self.TUBE_LAYER_MAX_REPETITIONS)
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

        trimmed_tube_page = QWidget()
        trimmed_tube_form = QFormLayout(trimmed_tube_page)
        self.trimmed_tube_repetitions_spin = QSpinBox()
        self.trimmed_tube_repetitions_spin.setRange(1, self.TUBE_LAYER_MAX_REPETITIONS)
        self.trimmed_tube_repetitions_spin.setValue(self.TRIMMED_TUBE_LAYER_DEFAULT_REPETITIONS)
        self.trimmed_tube_repetitions_spin.valueChanged.connect(self.on_trimmed_tube_layer_setting_changed)
        self.trimmed_tube_stroke_width_spin = QDoubleSpinBox()
        self.trimmed_tube_stroke_width_spin.setRange(0.10, 12.0)
        self.trimmed_tube_stroke_width_spin.setDecimals(2)
        self.trimmed_tube_stroke_width_spin.setSingleStep(0.05)
        self.trimmed_tube_stroke_width_spin.setValue(self.TRIMMED_TUBE_LAYER_DEFAULT_STROKE_WIDTH)
        self.trimmed_tube_stroke_width_spin.valueChanged.connect(self.on_trimmed_tube_layer_setting_changed)
        self.trimmed_tube_straightness_spin = QDoubleSpinBox()
        self.trimmed_tube_straightness_spin.setRange(0.0, 1.0)
        self.trimmed_tube_straightness_spin.setDecimals(2)
        self.trimmed_tube_straightness_spin.setSingleStep(0.05)
        self.trimmed_tube_straightness_spin.setValue(self.TRIMMED_TUBE_LAYER_DEFAULT_STRAIGHTNESS)
        self.trimmed_tube_straightness_spin.valueChanged.connect(self.on_trimmed_tube_layer_setting_changed)
        self.trimmed_tube_scale_spin = QDoubleSpinBox()
        self.trimmed_tube_scale_spin.setRange(0.20, 4.00)
        self.trimmed_tube_scale_spin.setDecimals(2)
        self.trimmed_tube_scale_spin.setSingleStep(0.05)
        self.trimmed_tube_scale_spin.setValue(self.TRIMMED_TUBE_LAYER_DEFAULT_SCALE)
        self.trimmed_tube_scale_spin.valueChanged.connect(self.on_trimmed_tube_layer_setting_changed)
        self.trimmed_tube_base_simplify_spin = QDoubleSpinBox()
        self.trimmed_tube_base_simplify_spin.setRange(0.0, 24.0)
        self.trimmed_tube_base_simplify_spin.setDecimals(2)
        self.trimmed_tube_base_simplify_spin.setSingleStep(0.10)
        self.trimmed_tube_base_simplify_spin.setValue(self.TRIMMED_TUBE_LAYER_DEFAULT_BASE_SIMPLIFY)
        self.trimmed_tube_base_simplify_spin.valueChanged.connect(self.on_trimmed_tube_layer_setting_changed)
        self.trimmed_tube_ring_simplify_spin = QDoubleSpinBox()
        self.trimmed_tube_ring_simplify_spin.setRange(0.0, 24.0)
        self.trimmed_tube_ring_simplify_spin.setDecimals(2)
        self.trimmed_tube_ring_simplify_spin.setSingleStep(0.10)
        self.trimmed_tube_ring_simplify_spin.setValue(self.TRIMMED_TUBE_LAYER_DEFAULT_RING_SIMPLIFY)
        self.trimmed_tube_ring_simplify_spin.valueChanged.connect(self.on_trimmed_tube_layer_setting_changed)
        self.trimmed_tube_min_piece_area_spin = QDoubleSpinBox()
        self.trimmed_tube_min_piece_area_spin.setRange(1.0, 5000.0)
        self.trimmed_tube_min_piece_area_spin.setDecimals(1)
        self.trimmed_tube_min_piece_area_spin.setSingleStep(1.0)
        self.trimmed_tube_min_piece_area_spin.setValue(self.TRIMMED_TUBE_LAYER_DEFAULT_MIN_PIECE_AREA)
        self.trimmed_tube_min_piece_area_spin.valueChanged.connect(self.on_trimmed_tube_layer_setting_changed)
        self.trimmed_tube_morph_steps_spin = QSpinBox()
        self.trimmed_tube_morph_steps_spin.setRange(0, 24)
        self.trimmed_tube_morph_steps_spin.setValue(self.TRIMMED_TUBE_LAYER_DEFAULT_MORPH_STEPS)
        self.trimmed_tube_morph_steps_spin.valueChanged.connect(self.on_trimmed_tube_layer_setting_changed)
        self.trimmed_tube_morph_shapes_spin = QSpinBox()
        self.trimmed_tube_morph_shapes_spin.setRange(1, 16)
        self.trimmed_tube_morph_shapes_spin.setValue(self.TRIMMED_TUBE_LAYER_DEFAULT_MORPH_SHAPES)
        self.trimmed_tube_morph_shapes_spin.valueChanged.connect(self.on_trimmed_tube_layer_setting_changed)
        self.trimmed_tube_morph_points_spin = QSpinBox()
        self.trimmed_tube_morph_points_spin.setRange(8, 512)
        self.trimmed_tube_morph_points_spin.setSingleStep(8)
        self.trimmed_tube_morph_points_spin.setValue(self.TRIMMED_TUBE_LAYER_DEFAULT_MORPH_POINTS)
        self.trimmed_tube_morph_points_spin.valueChanged.connect(self.on_trimmed_tube_layer_setting_changed)
        trimmed_tube_form.addRow("Repetitions", self.trimmed_tube_repetitions_spin)
        trimmed_tube_form.addRow("Outline Width", self.trimmed_tube_stroke_width_spin)
        trimmed_tube_form.addRow("Straightness", self.trimmed_tube_straightness_spin)
        trimmed_tube_form.addRow("Scale", self.trimmed_tube_scale_spin)
        trimmed_tube_form.addRow("Base Simplify", self.trimmed_tube_base_simplify_spin)
        trimmed_tube_form.addRow("Ring Simplify", self.trimmed_tube_ring_simplify_spin)
        trimmed_tube_form.addRow("Piece Min Area", self.trimmed_tube_min_piece_area_spin)
        trimmed_tube_form.addRow("Morph Steps", self.trimmed_tube_morph_steps_spin)
        trimmed_tube_form.addRow("Morph Shapes", self.trimmed_tube_morph_shapes_spin)
        trimmed_tube_form.addRow("Morph Points", self.trimmed_tube_morph_points_spin)
        self.layer_props_stack.addWidget(trimmed_tube_page)
        self.layer_props_stack.setCurrentIndex(0)

        selected_layer_row = QWidget()
        selected_layer_form = QFormLayout(selected_layer_row)
        selected_layer_form.setContentsMargins(0, 0, 0, 0)
        selected_layer_form.setLabelAlignment(Qt.AlignLeft | Qt.AlignTop)
        selected_layer_form.setFormAlignment(Qt.AlignTop)
        selected_layer_form.addRow("Selected Layer", self.selected_layer_label)

        self.layer_props_toggle_button.setText("Layer Properties")
        self.layer_props_toggle_button.setCheckable(True)
        self.layer_props_toggle_button.setChecked(True)
        self.layer_props_toggle_button.setArrowType(Qt.DownArrow)
        self.layer_props_toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.layer_props_toggle_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.layer_props_toggle_button.toggled.connect(self.on_layer_properties_toggle)

        self.layer_props_scroll.setWidgetResizable(True)
        self.layer_props_scroll.setFrameShape(QFrame.NoFrame)
        self.layer_props_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.layer_props_scroll.setWidget(self.layer_props_stack)
        self.layer_props_scroll.setMinimumHeight(120)
        self.layer_props_scroll.setMaximumHeight(300)

        layer_editor_layout = QVBoxLayout(self.layer_editor_panel)
        layer_editor_layout.setContentsMargins(0, 0, 0, 0)
        layer_editor_layout.setSpacing(4)
        layer_editor_layout.addWidget(selected_layer_row)
        layer_editor_layout.addWidget(self.layer_props_toggle_button)
        layer_editor_layout.addWidget(self.layer_props_scroll)
        self.on_layer_properties_toggle(True)
        self.layer_editor_panel.setVisible(False)

        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.setInterval(220)
        self.preview_timer.timeout.connect(self.on_live_preview_timeout)

        self.scatter_enabled_checkbox.toggled.connect(self.refresh_layer_stack)
        self.scatter_layer_count.valueChanged.connect(self.refresh_layer_stack)
        self.tube_enabled_checkbox.toggled.connect(self.refresh_layer_stack)
        self.tube_layer_count.valueChanged.connect(self.refresh_layer_stack)
        self.trimmed_tube_enabled_checkbox.toggled.connect(self.refresh_layer_stack)
        self.trimmed_tube_layer_count.valueChanged.connect(self.refresh_layer_stack)
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
        controls_layout.setLabelAlignment(Qt.AlignLeft | Qt.AlignTop)
        controls_layout.setFormAlignment(Qt.AlignTop)
        controls_layout.addRow("Levels", self.levels_label)
        controls_layout.addRow("Seed", self.seed_input)
        controls_layout.addRow("", self.random_seed_checkbox)
        controls_layout.addRow("Canvas", self.canvas_ratio_checkbox)
        controls_layout.addRow("Background", self.background_source_checkbox)
        controls_layout.addRow("", self.background_blur_checkbox)
        scatter_row = QHBoxLayout()
        scatter_row.addWidget(self.scatter_enabled_checkbox)
        scatter_row.addWidget(self.scatter_layer_count)
        controls_layout.addRow("Scatter Layers", scatter_row)
        tube_row = QHBoxLayout()
        tube_row.addWidget(self.tube_enabled_checkbox)
        tube_row.addWidget(self.tube_layer_count)
        controls_layout.addRow("Tube Layers", tube_row)
        trimmed_tube_row = QHBoxLayout()
        trimmed_tube_row.addWidget(self.trimmed_tube_enabled_checkbox)
        trimmed_tube_row.addWidget(self.trimmed_tube_layer_count)
        controls_layout.addRow("Trimmed Morph Layers", trimmed_tube_row)
        controls_layout.addRow("Layer Stack", self.layer_stack_list)
        controls_layout.addRow("", self.rebuild_layers_button)
        controls_layout.addRow(self.layer_editor_panel)
        out_row = QHBoxLayout()
        out_row.addWidget(self.output_dir_edit, stretch=1)
        out_row.addWidget(self.output_pick_button)
        controls_layout.addRow("Output Folder", out_row)

        source_box = QGroupBox("Input Thumbnail")
        source_layout = QVBoxLayout(source_box)
        source_layout.addWidget(self.source_preview)

        left_content = QWidget()
        left_layout = QVBoxLayout(left_content)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(controls_box)
        left_layout.addWidget(source_box)
        left_layout.addStretch(1)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setWidget(left_content)
        left_scroll.setMinimumWidth(360)

        output_box = QGroupBox("Output Preview")
        output_layout = QVBoxLayout(output_box)
        output_layout.addWidget(self.output_preview, stretch=1)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_scroll)
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
        self.on_background_source_toggled(self.background_source_checkbox.isChecked())
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
        self._refresh_layer_stack_labels()

    def on_random_seed_toggled(self, checked: bool) -> None:
        self.seed_input.setEnabled(not checked)
        if checked:
            self.seed_input.setValue(self._new_seed())
        self.request_live_preview()

    def on_background_source_toggled(self, checked: bool) -> None:
        self.background_blur_checkbox.setEnabled(checked)
        self.request_live_preview()

    def on_load_image(self) -> None:
        default_source_dir = self._default_source_image_dir()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Source Image",
            default_source_dir,
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

    def _default_source_image_dir(self) -> str:
        if self.source_path is not None:
            parent = self.source_path.parent
            if parent.exists():
                return str(parent)
        if self.DEFAULT_SOURCE_IMAGE_DIR.exists():
            return str(self.DEFAULT_SOURCE_IMAGE_DIR)
        return str(Path.cwd())

    def _canvas_size_for_source(self, source_width: int, source_height: int) -> tuple[int, int]:
        width = max(1, int(source_width))
        height = max(1, int(source_height))
        if not self.canvas_ratio_checkbox.isChecked():
            return width, height

        target_ratio = self.CANVAS_RATIO_WIDTH / self.CANVAS_RATIO_HEIGHT
        source_ratio = width / height
        if source_ratio > target_ratio:
            width = int(round(height * target_ratio))
        else:
            height = int(round(width / target_ratio))
        return max(1, width), max(1, height)

    def _limit_canvas_size(
        self,
        width: int,
        height: int,
        max_dim: int | None = None,
        max_pixels: int | None = None,
    ) -> tuple[int, int]:
        w = max(1, int(width))
        h = max(1, int(height))
        scale = 1.0

        if max_dim is not None and max_dim > 0:
            largest = max(w, h)
            if largest > max_dim:
                scale = min(scale, float(max_dim) / float(largest))

        if max_pixels is not None and max_pixels > 0:
            pixels = float(w * h)
            if pixels > float(max_pixels):
                scale = min(scale, (float(max_pixels) / pixels) ** 0.5)

        if scale < 1.0:
            w = max(1, int(round(w * scale)))
            h = max(1, int(round(h * scale)))
        return w, h

    def _prepare_canvas_image(
        self,
        source_image: Image.Image,
        max_dim: int | None = None,
        max_pixels: int | None = None,
    ) -> Image.Image:
        canvas_width, canvas_height = self._canvas_size_for_source(source_image.width, source_image.height)
        target_width, target_height = self._limit_canvas_size(
            canvas_width,
            canvas_height,
            max_dim=max_dim,
            max_pixels=max_pixels,
        )
        if target_width == source_image.width and target_height == source_image.height:
            return source_image.copy()
        resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
        return ImageOps.fit(source_image, (target_width, target_height), method=resample, centering=(0.5, 0.5))

    def _build_background_image_data_url(self, canvas_image: Image.Image, blur_enabled: bool) -> str:
        background_image = canvas_image.convert("L")
        if blur_enabled:
            background_image = background_image.filter(ImageFilter.GaussianBlur(radius=self.BACKGROUND_BLUR_RADIUS))
        encoded_output = io.BytesIO()
        background_image.save(encoded_output, format="PNG", optimize=True)
        payload = base64.b64encode(encoded_output.getvalue()).decode("ascii")
        return f"data:image/png;base64,{payload}"

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

    def _make_trimmed_tube_layer_data(self) -> dict[str, object]:
        return {
            "type": "trimmed_morph_tube",
            "tube_segment_count": int(self.TRIMMED_TUBE_LAYER_DEFAULT_REPETITIONS),
            "tube_stroke_width": float(self.TRIMMED_TUBE_LAYER_DEFAULT_STROKE_WIDTH),
            "tube_straightness": float(self.TRIMMED_TUBE_LAYER_DEFAULT_STRAIGHTNESS),
            "tube_scale": float(self.TRIMMED_TUBE_LAYER_DEFAULT_SCALE),
            "tube_base_simplify": float(self.TRIMMED_TUBE_LAYER_DEFAULT_BASE_SIMPLIFY),
            "tube_ring_simplify": float(self.TRIMMED_TUBE_LAYER_DEFAULT_RING_SIMPLIFY),
            "tube_piece_min_area": float(self.TRIMMED_TUBE_LAYER_DEFAULT_MIN_PIECE_AREA),
            "tube_morph_steps": int(self.TRIMMED_TUBE_LAYER_DEFAULT_MORPH_STEPS),
            "tube_morph_shapes": int(self.TRIMMED_TUBE_LAYER_DEFAULT_MORPH_SHAPES),
            "tube_morph_points": int(self.TRIMMED_TUBE_LAYER_DEFAULT_MORPH_POINTS),
        }

    def _make_tube_layer_data_with_repetitions(self, repetitions: int) -> dict[str, object]:
        data = self._make_tube_layer_data()
        data["tube_segment_count"] = max(1, min(self.TUBE_LAYER_MAX_REPETITIONS, int(repetitions)))
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
        self._clear_layer_selection()

    def _set_layer_item_text(self, item: QListWidgetItem, index: int) -> None:
        data = item.data(Qt.UserRole)
        if not isinstance(data, dict):
            self._set_layer_item_display(item, f"Layer {index + 1}")
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
            self._set_layer_item_display(item, f"{position_label}: Scatter ({min_shapes}-{max_shapes} shapes)")
            return
        if layer_type == "trimmed_morph_tube":
            repetitions = int(data.get("tube_segment_count", self.TRIMMED_TUBE_LAYER_DEFAULT_REPETITIONS))
            stroke_width = float(data.get("tube_stroke_width", self.TRIMMED_TUBE_LAYER_DEFAULT_STROKE_WIDTH))
            straightness = float(data.get("tube_straightness", self.TRIMMED_TUBE_LAYER_DEFAULT_STRAIGHTNESS))
            scale = float(data.get("tube_scale", self.TRIMMED_TUBE_LAYER_DEFAULT_SCALE))
            morph_steps = int(data.get("tube_morph_steps", self.TRIMMED_TUBE_LAYER_DEFAULT_MORPH_STEPS))
            self._set_layer_item_display(
                item,
                f"{position_label}: TrimMorph (r={repetitions}, w={stroke_width:.2f}, "
                f"s={straightness:.2f}, z={scale:.2f}, m={morph_steps})",
            )
            return

        repetitions = int(data.get("tube_segment_count", self.TUBE_LAYER_DEFAULT_REPETITIONS))
        stroke_width = float(data.get("tube_stroke_width", self.TUBE_LAYER_DEFAULT_STROKE_WIDTH))
        straightness = float(data.get("tube_straightness", self.TUBE_LAYER_DEFAULT_STRAIGHTNESS))
        self._set_layer_item_display(
            item,
            f"{position_label}: Tube (r={repetitions}, w={stroke_width:.2f}, s={straightness:.2f})",
        )

    def _set_layer_item_display(self, item: QListWidgetItem, full_text: str) -> None:
        item.setToolTip(full_text)
        viewport_width = max(80, int(self.layer_stack_list.viewport().width()) - 12)
        metrics = QFontMetrics(self.layer_stack_list.font())
        elided = metrics.elidedText(full_text, Qt.ElideRight, viewport_width)
        item.setText(elided)

    def on_layer_properties_toggle(self, expanded: bool) -> None:
        self.layer_props_scroll.setVisible(bool(expanded))
        self.layer_props_toggle_button.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)

    def _refresh_layer_stack_labels(self) -> None:
        for idx in range(self.layer_stack_list.count()):
            self._set_layer_item_text(self.layer_stack_list.item(idx), idx)

    def _clear_layer_selection(self) -> None:
        self.layer_stack_list.blockSignals(True)
        self.layer_stack_list.clearSelection()
        self.layer_stack_list.setCurrentRow(-1)
        self.layer_stack_list.blockSignals(False)
        self.on_layer_selection_changed(None, None)

    def on_layer_selection_changed(self, current, _previous) -> None:
        if current is None:
            self.selected_layer_label.setText("No layer selected")
            self.layer_props_stack.setCurrentIndex(0)
            self.layer_editor_panel.setVisible(False)
            return
        data = current.data(Qt.UserRole)
        if not isinstance(data, dict):
            self.selected_layer_label.setText("Unknown layer")
            self.layer_props_stack.setCurrentIndex(0)
            self.layer_editor_panel.setVisible(False)
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
            self.layer_editor_panel.setVisible(True)
            return

        if layer_type == "trimmed_morph_tube":
            repetitions = int(data.get("tube_segment_count", self.TRIMMED_TUBE_LAYER_DEFAULT_REPETITIONS))
            stroke_width = float(data.get("tube_stroke_width", self.TRIMMED_TUBE_LAYER_DEFAULT_STROKE_WIDTH))
            straightness = float(data.get("tube_straightness", self.TRIMMED_TUBE_LAYER_DEFAULT_STRAIGHTNESS))
            scale = float(data.get("tube_scale", self.TRIMMED_TUBE_LAYER_DEFAULT_SCALE))
            base_simplify = float(data.get("tube_base_simplify", self.TRIMMED_TUBE_LAYER_DEFAULT_BASE_SIMPLIFY))
            ring_simplify = float(data.get("tube_ring_simplify", self.TRIMMED_TUBE_LAYER_DEFAULT_RING_SIMPLIFY))
            min_piece_area = float(data.get("tube_piece_min_area", self.TRIMMED_TUBE_LAYER_DEFAULT_MIN_PIECE_AREA))
            morph_steps = int(data.get("tube_morph_steps", self.TRIMMED_TUBE_LAYER_DEFAULT_MORPH_STEPS))
            morph_shapes = int(data.get("tube_morph_shapes", self.TRIMMED_TUBE_LAYER_DEFAULT_MORPH_SHAPES))
            morph_points = int(data.get("tube_morph_points", self.TRIMMED_TUBE_LAYER_DEFAULT_MORPH_POINTS))
            self.selected_layer_label.setText("Trimmed morph tube")
            self.layer_props_stack.setCurrentIndex(3)
            self.trimmed_tube_repetitions_spin.blockSignals(True)
            self.trimmed_tube_repetitions_spin.setValue(repetitions)
            self.trimmed_tube_repetitions_spin.blockSignals(False)
            self.trimmed_tube_stroke_width_spin.blockSignals(True)
            self.trimmed_tube_stroke_width_spin.setValue(stroke_width)
            self.trimmed_tube_stroke_width_spin.blockSignals(False)
            self.trimmed_tube_straightness_spin.blockSignals(True)
            self.trimmed_tube_straightness_spin.setValue(straightness)
            self.trimmed_tube_straightness_spin.blockSignals(False)
            self.trimmed_tube_scale_spin.blockSignals(True)
            self.trimmed_tube_scale_spin.setValue(scale)
            self.trimmed_tube_scale_spin.blockSignals(False)
            self.trimmed_tube_base_simplify_spin.blockSignals(True)
            self.trimmed_tube_base_simplify_spin.setValue(base_simplify)
            self.trimmed_tube_base_simplify_spin.blockSignals(False)
            self.trimmed_tube_ring_simplify_spin.blockSignals(True)
            self.trimmed_tube_ring_simplify_spin.setValue(ring_simplify)
            self.trimmed_tube_ring_simplify_spin.blockSignals(False)
            self.trimmed_tube_min_piece_area_spin.blockSignals(True)
            self.trimmed_tube_min_piece_area_spin.setValue(min_piece_area)
            self.trimmed_tube_min_piece_area_spin.blockSignals(False)
            self.trimmed_tube_morph_steps_spin.blockSignals(True)
            self.trimmed_tube_morph_steps_spin.setValue(morph_steps)
            self.trimmed_tube_morph_steps_spin.blockSignals(False)
            self.trimmed_tube_morph_shapes_spin.blockSignals(True)
            self.trimmed_tube_morph_shapes_spin.setValue(morph_shapes)
            self.trimmed_tube_morph_shapes_spin.blockSignals(False)
            self.trimmed_tube_morph_points_spin.blockSignals(True)
            self.trimmed_tube_morph_points_spin.setValue(morph_points)
            self.trimmed_tube_morph_points_spin.blockSignals(False)
            self.layer_editor_panel.setVisible(True)
            return

        if layer_type != "segmented_tube":
            self.selected_layer_label.setText("Unknown layer")
            self.layer_props_stack.setCurrentIndex(0)
            self.layer_editor_panel.setVisible(False)
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
        self.layer_editor_panel.setVisible(True)

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

    def on_trimmed_tube_layer_setting_changed(self, _value) -> None:
        item = self.layer_stack_list.currentItem()
        if item is None:
            return
        data = item.data(Qt.UserRole)
        if not isinstance(data, dict):
            return
        layer_type = str(data.get("type", "")).strip().lower()
        if layer_type != "trimmed_morph_tube":
            return
        updated = dict(data)
        updated["tube_segment_count"] = int(self.trimmed_tube_repetitions_spin.value())
        updated["tube_stroke_width"] = float(self.trimmed_tube_stroke_width_spin.value())
        updated["tube_straightness"] = float(self.trimmed_tube_straightness_spin.value())
        updated["tube_scale"] = float(self.trimmed_tube_scale_spin.value())
        updated["tube_base_simplify"] = float(self.trimmed_tube_base_simplify_spin.value())
        updated["tube_ring_simplify"] = float(self.trimmed_tube_ring_simplify_spin.value())
        updated["tube_piece_min_area"] = float(self.trimmed_tube_min_piece_area_spin.value())
        updated["tube_morph_steps"] = int(self.trimmed_tube_morph_steps_spin.value())
        updated["tube_morph_shapes"] = int(self.trimmed_tube_morph_shapes_spin.value())
        updated["tube_morph_points"] = int(self.trimmed_tube_morph_points_spin.value())
        item.setData(Qt.UserRole, updated)
        self._refresh_layer_stack_labels()
        self.request_live_preview()

    def on_layer_order_changed(self, *_args) -> None:
        self._refresh_layer_stack_labels()
        self.request_live_preview()

    def refresh_layer_stack(self, *_args) -> None:
        self.scatter_layer_count.setEnabled(self.scatter_enabled_checkbox.isChecked())
        self.tube_layer_count.setEnabled(self.tube_enabled_checkbox.isChecked())
        self.trimmed_tube_layer_count.setEnabled(self.trimmed_tube_enabled_checkbox.isChecked())

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
        if self.trimmed_tube_enabled_checkbox.isChecked():
            for idx in range(int(self.trimmed_tube_layer_count.value())):
                item = QListWidgetItem()
                item.setData(Qt.UserRole, self._make_trimmed_tube_layer_data())
                self.layer_stack_list.addItem(item)
        self._refresh_layer_stack_labels()
        self._clear_layer_selection()
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
            if layer_type not in {"scatter", "segmented_tube", "trimmed_morph_tube"}:
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
            elif layer_type == "segmented_tube":
                spec["tube_segment_count"] = int(data.get("tube_segment_count", self.TUBE_LAYER_DEFAULT_REPETITIONS))
                spec["tube_stroke_width"] = float(
                    data.get("tube_stroke_width", self.TUBE_LAYER_DEFAULT_STROKE_WIDTH)
                )
                spec["tube_straightness"] = float(
                    data.get("tube_straightness", self.TUBE_LAYER_DEFAULT_STRAIGHTNESS)
                )
            else:
                spec["tube_segment_count"] = int(
                    data.get("tube_segment_count", self.TRIMMED_TUBE_LAYER_DEFAULT_REPETITIONS)
                )
                spec["tube_stroke_width"] = float(
                    data.get("tube_stroke_width", self.TRIMMED_TUBE_LAYER_DEFAULT_STROKE_WIDTH)
                )
                spec["tube_straightness"] = float(
                    data.get("tube_straightness", self.TRIMMED_TUBE_LAYER_DEFAULT_STRAIGHTNESS)
                )
                spec["tube_scale"] = float(
                    data.get("tube_scale", self.TRIMMED_TUBE_LAYER_DEFAULT_SCALE)
                )
                spec["tube_base_simplify"] = float(
                    data.get("tube_base_simplify", self.TRIMMED_TUBE_LAYER_DEFAULT_BASE_SIMPLIFY)
                )
                spec["tube_ring_simplify"] = float(
                    data.get("tube_ring_simplify", self.TRIMMED_TUBE_LAYER_DEFAULT_RING_SIMPLIFY)
                )
                spec["tube_piece_min_area"] = float(
                    data.get("tube_piece_min_area", self.TRIMMED_TUBE_LAYER_DEFAULT_MIN_PIECE_AREA)
                )
                spec["tube_morph_steps"] = int(
                    data.get("tube_morph_steps", self.TRIMMED_TUBE_LAYER_DEFAULT_MORPH_STEPS)
                )
                spec["tube_morph_shapes"] = int(
                    data.get("tube_morph_shapes", self.TRIMMED_TUBE_LAYER_DEFAULT_MORPH_SHAPES)
                )
                spec["tube_morph_points"] = int(
                    data.get("tube_morph_points", self.TRIMMED_TUBE_LAYER_DEFAULT_MORPH_POINTS)
                )
            layers.append(spec)
        return layers

    def request_live_preview(self) -> None:
        if self.source_path is None:
            return
        if self._is_rendering:
            self._pending_live_preview = True
            return
        self.preview_timer.start()

    def on_live_preview_timeout(self) -> None:
        if self._is_rendering:
            self._pending_live_preview = True
            return
        self._render_composition(save_outputs=False)

    def _render_composition(self, save_outputs: bool) -> None:
        if self._is_rendering:
            if save_outputs:
                self._pending_generate = True
            else:
                self._pending_live_preview = True
            return
        if self.source_path is None:
            if save_outputs:
                QMessageBox.warning(self, "Missing Image", "Load a source image first.")
            return

        self._is_rendering = True
        if save_outputs:
            self.generate_button.setEnabled(False)
            self.generate_button.setText("Generating...")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
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
                    elif layer_type == "trimmed_morph_tube":
                        reps = int(layer.get("tube_segment_count", self.TRIMMED_TUBE_LAYER_DEFAULT_REPETITIONS))
                        stroke = float(layer.get("tube_stroke_width", self.TRIMMED_TUBE_LAYER_DEFAULT_STROKE_WIDTH))
                        straight = float(layer.get("tube_straightness", self.TRIMMED_TUBE_LAYER_DEFAULT_STRAIGHTNESS))
                        scale = float(layer.get("tube_scale", self.TRIMMED_TUBE_LAYER_DEFAULT_SCALE))
                        morph_steps = int(layer.get("tube_morph_steps", self.TRIMMED_TUBE_LAYER_DEFAULT_MORPH_STEPS))
                        morph_shapes = int(layer.get("tube_morph_shapes", self.TRIMMED_TUBE_LAYER_DEFAULT_MORPH_SHAPES))
                        layer_summary_parts.append(
                            f"{idx + 1}:trimmed_morph_tube("
                            f"reps={reps},stroke={stroke:.2f},straight={straight:.2f},scale={scale:.2f},"
                            f"morph_steps={morph_steps},morph_shapes={morph_shapes})"
                        )
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
                    f"white_background={self.COMPOSE_WHITE_BACKGROUND} | "
                    f"canvas_3_4={self.canvas_ratio_checkbox.isChecked()} | "
                    f"source_background={self.background_source_checkbox.isChecked()} | "
                    f"bg_blur={self.background_blur_checkbox.isChecked()}"
                )

            timings: dict[str, float] = {}

            t0 = perf_counter()
            source_img = load_image(self.source_path)
            timings["load_image"] = perf_counter() - t0

            requested_width, requested_height = self._canvas_size_for_source(source_img.width, source_img.height)
            max_dim = self.GENERATE_MAX_DIM if save_outputs else self.PREVIEW_MAX_DIM
            max_pixels = self.GENERATE_MAX_PIXELS if save_outputs else self.PREVIEW_MAX_PIXELS
            limited_width, limited_height = self._limit_canvas_size(
                requested_width,
                requested_height,
                max_dim=max_dim,
                max_pixels=max_pixels,
            )

            t0 = perf_counter()
            canvas_img = self._prepare_canvas_image(
                source_img,
                max_dim=max_dim,
                max_pixels=max_pixels,
            )
            timings["prepare_canvas"] = perf_counter() - t0
            if save_outputs and (limited_width != requested_width or limited_height != requested_height):
                self._log(
                    f"Canvas capped for stability: {requested_width}x{requested_height} -> "
                    f"{limited_width}x{limited_height}"
                )

            t0 = perf_counter()
            gray = to_grayscale(canvas_img)
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

            background_image_data_url: str | None = None
            if self.background_source_checkbox.isChecked():
                t0 = perf_counter()
                background_image_data_url = self._build_background_image_data_url(
                    canvas_image=canvas_img,
                    blur_enabled=self.background_blur_checkbox.isChecked(),
                )
                timings["build_background"] = perf_counter() - t0

            t0 = perf_counter()
            svg_str = build_svg(
                shapes=shapes,
                width=canvas_img.width,
                height=canvas_img.height,
                seed=seed,
                arrangement="scatter",
                levels=self.LEVELS,
                target_fill_ratio=None,
                shape_count_range=None,
                tube_segment_count_range=self.COMPOSE_TUBE_SEGMENT_COUNT_RANGE,
                layer_specs=layer_specs,
                white_background=self.COMPOSE_WHITE_BACKGROUND,
                background_image_data_url=background_image_data_url,
                background_image_opacity=self.BACKGROUND_IMAGE_OPACITY,
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
                scale=1.0 if save_outputs else self.PREVIEW_SCALE,
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
        finally:
            if QApplication.overrideCursor() is not None:
                QApplication.restoreOverrideCursor()
            if save_outputs:
                self.generate_button.setText("Generate")
                self.generate_button.setEnabled(True)
            self._is_rendering = False
            if self._pending_live_preview and self.source_path is not None:
                self._pending_live_preview = False
                self.preview_timer.start()
            if self._pending_generate and self.source_path is not None and not self._is_rendering:
                self._pending_generate = False
                QTimer.singleShot(0, self.on_generate)

    def on_generate(self) -> None:
        self._render_composition(save_outputs=True)


def main() -> None:
    app = QApplication.instance() or QApplication([])
    window = ArtUiWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
