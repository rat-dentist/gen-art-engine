from __future__ import annotations


class AppShell:
    def __init__(self, title: str, width: int, height: int, camera, params, draw_scene) -> None:
        import pyglet
        from pyglet import gl

        self.gl = gl
        self.camera = camera
        self.params = params
        self.draw_scene = draw_scene
        self.imgui = None
        self.renderer = None

        self.window = pyglet.window.Window(width=width, height=height, caption=title, resizable=True)
        self._is_orbiting = False
        self._is_panning = False

        try:
            import imgui
            from imgui.integrations.pyglet import PygletRenderer

            self.imgui = imgui
            self.imgui.create_context()
            self.renderer = PygletRenderer(self.window)
        except ModuleNotFoundError:
            # imgui is optional: run with viewport controls only.
            self.imgui = None
            self.renderer = None

        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glClearColor(0.08, 0.09, 0.11, 1.0)

        @self.window.event
        def on_draw():
            self._on_draw()

        @self.window.event
        def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
            if self.renderer and self.renderer.io.want_capture_mouse:
                return
            shift = modifiers & pyglet.window.key.MOD_SHIFT
            if buttons & pyglet.window.mouse.LEFT and shift:
                self.camera.pan(dx, dy)
            elif buttons & pyglet.window.mouse.LEFT:
                self.camera.orbit(dx, dy)
            elif buttons & pyglet.window.mouse.MIDDLE:
                self.camera.pan(dx, dy)

        @self.window.event
        def on_mouse_scroll(x, y, scroll_x, scroll_y):
            if self.renderer and self.renderer.io.want_capture_mouse:
                return
            self.camera.dolly(scroll_y)

        @self.window.event
        def on_key_press(symbol, modifiers):
            if self.renderer and self.renderer.io.want_capture_keyboard:
                return
            if symbol == pyglet.window.key.R:
                self.camera.reset()
            if symbol == pyglet.window.key.F:
                self.camera.frame(1.0)

    def _set_camera(self) -> None:
        w, h = self.window.get_size()
        h = max(h, 1)
        eye = self.camera.eye
        target = self.camera.target

        self.gl.glViewport(0, 0, w, h)
        self.gl.glMatrixMode(self.gl.GL_PROJECTION)
        self.gl.glLoadIdentity()
        self.gl.gluPerspective(55.0, float(w) / float(h), 0.1, 1000.0)
        self.gl.glMatrixMode(self.gl.GL_MODELVIEW)
        self.gl.glLoadIdentity()
        self.gl.gluLookAt(
            eye[0],
            eye[1],
            eye[2],
            target[0],
            target[1],
            target[2],
            0.0,
            1.0,
            0.0,
        )

    def _on_draw(self) -> None:
        self.window.clear()
        self.gl.glClear(self.gl.GL_COLOR_BUFFER_BIT | self.gl.GL_DEPTH_BUFFER_BIT)

        self._set_camera()
        self.draw_scene(self)

        if self.imgui and self.renderer:
            from kit.ui_imgui import draw_params_panel

            self.imgui.new_frame()
            draw_params_panel(self.imgui, self.params)
            self.imgui.render()
            self.renderer.render(self.imgui.get_draw_data())

    def run(self) -> None:
        import pyglet

        pyglet.app.run()
