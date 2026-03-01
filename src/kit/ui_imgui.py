from __future__ import annotations

from kit.params import BoolParam, EnumParam, FloatParam, IntParam, Params


def draw_params_panel(imgui, params: Params, title: str = "Sketch Controls") -> None:
    imgui.set_next_window_position((16, 16), imgui.Cond_.always)
    imgui.set_next_window_size((320, 220), imgui.Cond_.first_use_ever)

    imgui.begin(title, None, imgui.WindowFlags_.no_collapse)

    for param in params.items:
        if isinstance(param, FloatParam):
            changed, value = imgui.slider_float(
                param.label, float(param.value), float(param.min_value), float(param.max_value)
            )
            if changed:
                param.value = float(value)
        elif isinstance(param, IntParam):
            changed, value = imgui.slider_int(
                param.label, int(param.value), int(param.min_value), int(param.max_value)
            )
            if changed:
                param.value = int(value)
        elif isinstance(param, BoolParam):
            changed, value = imgui.checkbox(param.label, bool(param.value))
            if changed:
                param.value = bool(value)
        elif isinstance(param, EnumParam):
            index = max(0, param.options.index(param.value))
            changed, new_index = imgui.combo(param.label, index, param.options)
            if changed:
                param.value = param.options[new_index]

    imgui.end()
