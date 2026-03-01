from __future__ import annotations

from pathlib import Path

from kit.naming import default_output_path


def export_workspace_png(window, path=None, filename=None):
    output_path = Path(path) if path is not None else default_output_path("frame", "png", "renders", filename=filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    from pyglet import image

    image.get_buffer_manager().get_color_buffer().save(str(output_path))
    return output_path
