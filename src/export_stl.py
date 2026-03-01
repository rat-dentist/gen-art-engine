from pathlib import Path

from kit.naming import default_output_path


def export_stl(mesh_data, path=None, filename=None):
    output_path = Path(path) if path is not None else default_output_path("stl", "stl", "models", filename=filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        f.write("solid placeholder\n")
        f.write("endsolid placeholder\n")

    print(f"STL export placeholder: {output_path}")
    return output_path
