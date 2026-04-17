"""Sequence formatters for the model-organisms framework.

A formatter takes an iterable of alphabet tokens and returns the string
shown to the model as input.
"""

from collections.abc import Iterable
from typing import Callable


_REGISTRY: dict[str, Callable[[Iterable[str]], str]] = {}


def register_format(name: str):
    def decorator(fn: Callable[[Iterable[str]], str]):
        if name in _REGISTRY:
            raise ValueError(f"Format already registered: {name}")
        _REGISTRY[name] = fn
        return fn

    return decorator


def get_format(name: str) -> Callable[[Iterable[str]], str]:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown format: {name!r}. Known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_formats() -> list[str]:
    return sorted(_REGISTRY)


@register_format("spaced")
def spaced(seq: Iterable[str]) -> str:
    return " ".join(seq)
