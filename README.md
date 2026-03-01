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
- Render on a white background (paper-like base).
- Output preview is the primary canvas in the UI; input is shown as a small thumbnail.
- Seed randomization is enabled by default (you can disable it for deterministic reruns).
- Export `.svg` and `.png` to the selected output folder (default `output/`).

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
