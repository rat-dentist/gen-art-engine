from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass
class FloatParam:
    key: str
    label: str
    value: float
    min_value: float
    max_value: float


@dataclass
class IntParam:
    key: str
    label: str
    value: int
    min_value: int
    max_value: int


@dataclass
class BoolParam:
    key: str
    label: str
    value: bool


@dataclass
class EnumParam:
    key: str
    label: str
    value: str
    options: list[str]


class Params:
    def __init__(self, items: Iterable[Any]):
        self._items = list(items)
        self._by_key = {item.key: item for item in self._items}

    @property
    def items(self) -> list[Any]:
        return self._items

    def get(self, key: str) -> Any:
        return self._by_key[key]

    def value(self, key: str) -> Any:
        return self._by_key[key].value
