from __future__ import annotations

import argparse
import math

from export_raster import export_workspace_png
from export_stl import export_stl


def parse_args():
    parser = argparse.ArgumentParser(description="Run the 3D sketch.")
    parser.add_argument(
        "--stl-filename",
        default=None,
        help="Optional STL filename override (saved in output/models).",
    )
    parser.add_argument(
        "--stl-path",
        default=None,
        help="Optional explicit STL path override.",
    )
    parser.add_argument(
        "--export-stl",
        action="store_true",
        help="Export placeholder STL before launching the 3D window.",
    )
    parser.add_argument(
        "--frame-filename",
        default=None,
        help="Optional PNG filename override (saved in output/renders).",
    )
    parser.add_argument(
        "--frame-path",
        default=None,
        help="Optional explicit PNG path override.",
    )
    parser.add_argument(
        "--export-png",
        action="store_true",
        help="Export a PNG on first frame after launch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.export_stl or args.stl_filename or args.stl_path:
        stl_path = export_stl(mesh_data=None, path=args.stl_path, filename=args.stl_filename)
        print(f"STL export complete: {stl_path}")

    try:
        import pyglet
        from pyglet import gl
        from pyglet.graphics.shader import Shader, ShaderProgram
        from pyglet.math import Mat4, Vec3
    except ModuleNotFoundError:
        print(
            "3D mode dependencies are missing. Install them with:\n"
            "  pip install -r requirements-3d.txt"
        )
        return

    window = pyglet.window.Window(width=1280, height=720, caption="Template 3D Sketch", resizable=True)
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glClearColor(0.08, 0.09, 0.11, 1.0)
    button_rect = pyglet.shapes.Rectangle(0, 0, 132, 34, color=(28, 120, 186))
    button_label = pyglet.text.Label(
        "Export PNG",
        x=0,
        y=0,
        anchor_x="center",
        anchor_y="center",
        color=(255, 255, 255, 255),
    )
    status_label = pyglet.text.Label(
        "Press Export PNG to save workspace frame",
        x=0,
        y=0,
        anchor_x="left",
        anchor_y="center",
        color=(220, 220, 220, 255),
    )
    pending_png_export = {"value": bool(args.export_png)}
    last_export_path = {"value": None}
    ui_hover = {"export_button": False}

    def button_bounds():
        _, h = window.get_size()
        bx = 16
        by = h - 16 - button_rect.height
        return bx, by

    def do_png_export(trigger: str) -> None:
        png_path = export_workspace_png(window, path=args.frame_path, filename=args.frame_filename)
        last_export_path["value"] = png_path
        window.set_caption(f"Template 3D Sketch - {png_path.name}")
        print(f"PNG export complete ({trigger}): {png_path}")

    vertex_shader = Shader(
        """
        #version 330
        in vec3 position;
        in vec3 color;
        out vec3 v_color;
        uniform mat4 u_mvp;
        void main() {
            v_color = color;
            gl_Position = u_mvp * vec4(position, 1.0);
        }
        """,
        "vertex",
    )
    fragment_shader = Shader(
        """
        #version 330
        in vec3 v_color;
        out vec4 out_color;
        void main() {
            out_color = vec4(v_color, 1.0);
        }
        """,
        "fragment",
    )
    program = ShaderProgram(vertex_shader, fragment_shader)

    vertices = [
        -0.5,
        -0.5,
        -0.5,
        0.5,
        -0.5,
        -0.5,
        0.5,
        0.5,
        -0.5,
        -0.5,
        0.5,
        -0.5,
        -0.5,
        -0.5,
        0.5,
        0.5,
        -0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        -0.5,
        0.5,
        0.5,
    ]
    colors = [1.0] * (8 * 3)
    edge_colors = [0.0] * (8 * 3)
    indices = [
        0,
        1,
        2,
        2,
        3,
        0,
        4,
        5,
        6,
        6,
        7,
        4,
        0,
        4,
        7,
        7,
        3,
        0,
        1,
        5,
        6,
        6,
        2,
        1,
        3,
        2,
        6,
        6,
        7,
        3,
        0,
        1,
        5,
        5,
        4,
        0,
    ]
    edge_indices = [
        0,
        1,
        1,
        2,
        2,
        3,
        3,
        0,
        4,
        5,
        5,
        6,
        6,
        7,
        7,
        4,
        0,
        4,
        1,
        5,
        2,
        6,
        3,
        7,
    ]

    cube = program.vertex_list_indexed(
        8,
        gl.GL_TRIANGLES,
        indices,
        position=("f", vertices),
        color=("f", colors),
    )
    edges = program.vertex_list_indexed(
        8,
        gl.GL_LINES,
        edge_indices,
        position=("f", vertices),
        color=("f", edge_colors),
    )

    default_rot_x = math.radians(18.0)
    default_rot_y = math.radians(28.0)
    state = {
        "rot_x": default_rot_x,
        "rot_y": default_rot_y,
        "distance": 3.2,
    }

    @window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        if buttons & pyglet.window.mouse.LEFT:
            state["rot_y"] += dx * 0.01
            state["rot_x"] -= dy * 0.01

    @window.event
    def on_mouse_motion(x, y, dx, dy):
        bx, by = button_bounds()
        ui_hover["export_button"] = bx <= x <= bx + button_rect.width and by <= y <= by + button_rect.height

    @window.event
    def on_mouse_press(x, y, button, modifiers):
        if button != pyglet.window.mouse.LEFT:
            return
        bx, by = button_bounds()
        if bx <= x <= bx + button_rect.width and by <= y <= by + button_rect.height:
            do_png_export("button")

    @window.event
    def on_mouse_scroll(x, y, scroll_x, scroll_y):
        state["distance"] = max(1.5, min(9.0, state["distance"] - scroll_y * 0.2))

    @window.event
    def on_key_press(symbol, modifiers):
        if symbol == pyglet.window.key.R:
            state["rot_x"] = default_rot_x
            state["rot_y"] = default_rot_y
            state["distance"] = 3.2
        if symbol == pyglet.window.key.P:
            do_png_export("hotkey")

    @window.event
    def on_draw():
        window.clear()
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        user_rot = Mat4.from_rotation(state["rot_x"], Vec3(1.0, 0.0, 0.0)) @ Mat4.from_rotation(
            state["rot_y"], Vec3(0.0, 1.0, 0.0)
        )
        model = user_rot

        w, h = window.get_size()
        aspect = max(1.0, float(w)) / max(1.0, float(h))
        proj = Mat4.perspective_projection(aspect, 0.1, 100.0, 60.0)
        view = Mat4.look_at(Vec3(0.0, 0.0, state["distance"]), Vec3(0.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0))
        program["u_mvp"] = proj @ view @ model

        program.use()
        gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)
        gl.glPolygonOffset(1.0, 1.0)
        cube.draw(gl.GL_TRIANGLES)
        gl.glDisable(gl.GL_POLYGON_OFFSET_FILL)
        gl.glLineWidth(1.5)
        edges.draw(gl.GL_LINES)

        if pending_png_export["value"]:
            do_png_export("startup")
            pending_png_export["value"] = False

        bx, by = button_bounds()
        button_rect.x = bx
        button_rect.y = by
        button_rect.color = (42, 136, 204) if ui_hover["export_button"] else (28, 120, 186)
        button_label.x = bx + (button_rect.width // 2)
        button_label.y = by + (button_rect.height // 2)

        status_text = (
            f"Last PNG: {last_export_path['value'].name}"
            if last_export_path["value"] is not None
            else "Click Export PNG or press P to save workspace frame"
        )
        status_label.text = status_text
        status_label.x = bx + button_rect.width + 12
        status_label.y = by + (button_rect.height // 2)

        gl.glDisable(gl.GL_DEPTH_TEST)
        button_rect.draw()
        button_label.draw()
        status_label.draw()
        gl.glEnable(gl.GL_DEPTH_TEST)

    pyglet.app.run()


if __name__ == "__main__":
    main()
