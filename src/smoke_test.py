from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from export_stl import export_stl
from main import main as run_main
from kit.naming import get_project_slug


def _matches(path: Path, kind: str, ext: str, slug: str, minute_stamp: str) -> bool:
    pattern = rf"^{re.escape(slug)}__{re.escape(minute_stamp)}__{re.escape(kind)}\.{re.escape(ext)}$"
    return re.match(pattern, path.name) is not None


def smoke_test_output_naming() -> None:
    slug = get_project_slug()
    minute_stamp = datetime.now().strftime("%y%m%d-%H%M")

    svg_path = run_main()
    if not _matches(svg_path, "svg", "svg", slug, minute_stamp):
        raise RuntimeError(f"SVG filename failed pattern check: {svg_path.name}")

    stl_path = export_stl(mesh_data=None)
    if not _matches(stl_path, "stl", "stl", slug, minute_stamp):
        raise RuntimeError(f"STL filename failed pattern check: {stl_path.name}")

    print("Smoke test passed.")


if __name__ == "__main__":
    smoke_test_output_naming()
