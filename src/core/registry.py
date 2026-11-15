"""Name registry: a stage implementation is swapped by registering a class and naming it in config."""

from __future__ import annotations

from typing import Any, Callable


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._items: dict[str, Callable[..., Any]] = {}

    def register(self, name: str):
        def deco(obj):
            if name in self._items:
                raise KeyError(f"{self.kind} '{name}' already registered")
            self._items[name] = obj
            return obj
        return deco

    def build(self, name: str, **kwargs) -> Any:
        return self.get(name)(**kwargs)

    def get(self, name: str) -> Callable[..., Any]:
        if name not in self._items:
            raise KeyError(f"Unknown {self.kind} '{name}'. Registered: {sorted(self._items)}")
        return self._items[name]

