from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import config


def _slugify(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text)
    text = re.sub(r"_{2,}", "_", text)
    text = text.strip("-_")
    return text or "project"


def get_project_slug() -> str:
    configured = getattr(config, "PROJECT_SLUG", None) or getattr(config, "PROJECT_NAME", None)
    source = str(configured) if configured else Path.cwd().name
    return _slugify(source)


def timestamp_min() -> str:
    return datetime.now().strftime("%y%m%d-%H%M")


def default_output_path(kind: str, ext: str, subdir: str, filename: str | None = None) -> Path:
    safe_kind = _slugify(kind)
    safe_ext = ext.lower().lstrip(".")
    output_dir = Path("output") / subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    if filename:
        output_name = Path(filename).name
        if "." not in output_name:
            output_name = f"{output_name}.{safe_ext}"
    else:
        output_name = f"{get_project_slug()}__{timestamp_min()}__{safe_kind}.{safe_ext}"

    return output_dir / output_name
