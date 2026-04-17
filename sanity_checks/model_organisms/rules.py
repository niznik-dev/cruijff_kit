"""Output rule definitions for the model-organisms framework.

A rule maps a sequence (iterable of alphabet tokens) to a label string.
Rules declare which input types they apply to via ``applicable`` so that
invalid combinations (e.g., ``parity`` on letters) can be caught early.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Rule:
    name: str
    fn: Callable[[Iterable[str]], str]
    applicable: frozenset[str]


_REGISTRY: dict[str, Rule] = {}


def register_rule(name: str, *, applicable: set[str]):
    def decorator(fn: Callable[[Iterable[str]], str]):
        if name in _REGISTRY:
            raise ValueError(f"Rule already registered: {name}")
        _REGISTRY[name] = Rule(name=name, fn=fn, applicable=frozenset(applicable))
        return fn

    return decorator


def get_rule(name: str) -> Rule:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown rule: {name!r}. Known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_rules() -> list[str]:
    return sorted(_REGISTRY)


@register_rule("parity", applicable={"bits"})
def parity(seq: Iterable[str]) -> str:
    return "1" if sum(int(x) for x in seq) % 2 == 1 else "0"
