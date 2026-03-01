from pathlib import Path

from kit.naming import default_output_path


def export_svg(shapes, path=None, filename=None):
    output_path = Path(path) if path is not None else default_output_path("svg", "svg", "vectors", filename=filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        f.write('<svg xmlns="http://www.w3.org/2000/svg" width="500" height="500">\n')
        for s in shapes:
            f.write(f'<circle cx="{s["x"]}" cy="{s["y"]}" r="{s["r"]}" fill="none" stroke="black"/>\n')
        f.write('</svg>')

    return output_path
