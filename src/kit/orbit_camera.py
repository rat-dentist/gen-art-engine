from __future__ import annotations

import math


class OrbitCamera:
    def __init__(self) -> None:
        self.target = [0.0, 0.0, 0.0]
        self.yaw = math.radians(45)
        self.pitch = math.radians(25)
        self.distance = 6.0
        self._defaults = (self.yaw, self.pitch, self.distance, list(self.target))

    def orbit(self, dx: float, dy: float) -> None:
        self.yaw += dx * 0.01
        self.pitch -= dy * 0.01
        self.pitch = max(math.radians(-89), min(math.radians(89), self.pitch))

    def pan(self, dx: float, dy: float) -> None:
        eye = self.eye
        forward = [self.target[i] - eye[i] for i in range(3)]
        forward = _normalize(forward)
        right = _normalize(_cross(forward, [0.0, 1.0, 0.0]))
        up = _normalize(_cross(right, forward))
        scale = self.distance * 0.0015
        for i in range(3):
            self.target[i] += (-dx * right[i] + dy * up[i]) * scale

    def dolly(self, wheel_delta: float) -> None:
        self.distance *= max(0.1, 1.0 - wheel_delta * 0.08)
        self.distance = max(0.5, min(100.0, self.distance))

    def reset(self) -> None:
        self.yaw, self.pitch, self.distance, target = self._defaults
        self.target = list(target)

    def frame(self, radius: float = 1.5) -> None:
        self.target = [0.0, 0.0, 0.0]
        self.distance = max(2.5, radius * 3.0)

    @property
    def eye(self) -> list[float]:
        cp = math.cos(self.pitch)
        sp = math.sin(self.pitch)
        cy = math.cos(self.yaw)
        sy = math.sin(self.yaw)
        return [
            self.target[0] + self.distance * cp * sy,
            self.target[1] + self.distance * sp,
            self.target[2] + self.distance * cp * cy,
        ]


def _cross(a: list[float], b: list[float]) -> list[float]:
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _normalize(v: list[float]) -> list[float]:
    length = math.sqrt(sum(component * component for component in v))
    if length < 1e-8:
        return [0.0, 0.0, 0.0]
    return [component / length for component in v]
