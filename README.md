# Art Python Template

## Desktop UI mode (PySide6)

### Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Run

```powershell
python src/ui_app.py
```

In VS Code, `Ctrl+Shift+B` now runs the default build task **Run Art UI**.

UI flow:

- Load a source image.
- Quantize to 6 grayscale levels.
- Extract tone-bucket contours.
- Re-compose a seeded random subset (4-8 shapes) with varied scale, including large hero shapes.
- Toggle `Scatter` and `Segmented tube` layer types independently and set layer counts for each.
- Add optional `Trimmed morph tube` layers (overlap-trimmed tube pieces with all-to-all shape morphing).
- Drag layer items up/down in the layer stack to control over/under order (`Top` row draws on top).
- Startup default stack is `Top tube (400 reps)`, `Middle scatter`, `Bottom tube (50 reps)`.
- Each selected layer has its own properties panel.
- Scatter layers: per-layer shape count range and target fill.
- Tube layers: per-layer repetition count, outline thickness, and path straightness.
- Trimmed morph tube layers: repetition/outline/straightness plus simplify, min piece area, and morph controls.
- Tube repetition changes density while keeping tube length fixed for that generated layer.
- Tube paths are allowed to run beyond the frame edges (they can be clipped by the canvas view).
- Property edits update the output preview in real time.
- Segmented tube layers render as white cutout shapes with black outlines.
- Render on a white background (paper-like base).
- Output preview is the primary canvas in the UI; input is shown as a small thumbnail.
- Seed randomization is enabled by default (you can disable it for deterministic reruns).
- Export `.svg` and `.png` to the selected output folder (default `output/`).

### Tube trim prototype (test sketch)

Use this standalone preview to validate overlap splitting before integrating into the main UI:

```powershell
python src/tube_trim_preview.py
```

By default it tries:

`G:\My Drive\ART\_Source Imagery\14a6a32a-8830-4cf6-ba17-7b724781ba97.png`

If that file is missing, a file picker opens.

You can still override the source image:

```powershell
python src/tube_trim_preview.py --image "G:\My Drive\ART\_Source Imagery\some_other_image.png"
```

Useful simplification controls for less intricate cutouts:

```powershell
python src/tube_trim_preview.py --base-simplify 2.0 --ring-simplify 1.2 --piece-min-area 30
```

Morph/tween controls (all-to-all between selected source shapes):

```powershell
python src/tube_trim_preview.py --morph-steps 6 --morph-shapes 4 --morph-points 96
```

Controls:

- Drag pieces with mouse to confirm they are independent.
- `Space`: reseed and rebuild.
- `R`: rebuild with current seed.
- In-window panel lets you edit seed/segments/simplify/morph settings and click `Apply + Rebuild`.

## Base mode

### Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-base.txt
```

### Run

```powershell
python src/main.py
```

This exports SVG output to `output/vectors/`.

Default export filenames now use:

`<project_slug>__YYMMDD-HHMM__<kind>.<ext>`

Examples:

- `tube_city__260214-0312__svg.svg`
- `tube_city__260214-0312__stl.stl`
- `tube_city__260214-0312__frame.jpg`

The timestamp is local time at minute precision. Names are lowercase and filesystem-safe.

Override options:

- `python src/main.py --svg-filename my_name.svg` (saved in `output/vectors/`)
- `python src/main.py --svg-path output/vectors/custom.svg` (explicit path)
- `python src/main_3d.py --export-stl --stl-filename my_mesh.stl` (saved in `output/models/`)
- `python src/main_3d.py --export-png --frame-filename my_frame.png` (saved in `output/renders/`)

## 3D mode (optional)

### Install optional 3D dependencies

```powershell
pip install -r requirements-3d.txt
```

### Run

```powershell
python src/main_3d.py
```

In VS Code you can also use the task **Run 3D Sketch** or launch config **Debug 3D Sketch**.

Click the in-window **Export PNG** button to save the current workspace frame to `output/renders/` with smart naming.
You can also press `P` in the 3D window to export immediately.

### Mouse + keyboard controls

- `LMB drag`: rotate cube
- `Mouse wheel`: zoom in/out
- `R`: reset view

## Template smoke test

1. Create and activate a new virtual environment.
2. Install base deps and run `python src/main.py`.
3. Confirm one SVG is generated in `output/vectors/` and matches `<project_slug>__YYMMDD-HHMM__svg.svg`.
4. Install optional deps with `pip install -r requirements-3d.txt`.
5. Run `python src/main_3d.py` and confirm the viewport opens with a rotating cube and control panel.

Optional naming smoke test:

```powershell
python src/smoke_test.py
```

## Structure

- `src/`: core logic
- `src/kit/`: optional 3D sketch helpers (camera, params, UI, app shell)
- `assets/`: source inputs
- `output/`: generated outputs
- `scripts/`: automation helpers
