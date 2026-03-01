import argparse

from generator import generate
from export_svg import export_svg


def parse_args():
    parser = argparse.ArgumentParser(description="Generate and export an SVG sketch.")
    parser.add_argument(
        "--svg-filename",
        default=None,
        help="Optional SVG filename override (saved in output/vectors).",
    )
    parser.add_argument(
        "--svg-path",
        default=None,
        help="Optional explicit SVG path override.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data = generate()
    svg_path = export_svg(data, path=args.svg_path, filename=args.svg_filename)
    print(f"Export complete: {svg_path}")
    return svg_path

if __name__ == "__main__":
    main()
