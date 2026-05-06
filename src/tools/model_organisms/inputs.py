"""Input type definitions for the model-organisms framework.

An input type defines an alphabet from which sequences are drawn (e.g.,
``bits`` uses ``("0", "1")``). New input types register themselves at
import time via :func:`register_input`.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class InputType:
    name: str
    alphabet: tuple[str, ...]


_REGISTRY: dict[str, InputType] = {}


def register_input(input_type: InputType) -> None:
    if input_type.name in _REGISTRY:
        raise ValueError(f"Input type already registered: {input_type.name}")
    _REGISTRY[input_type.name] = input_type


def get_input(name: str) -> InputType:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown input type: {name!r}. Known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_inputs() -> list[str]:
    return sorted(_REGISTRY)


register_input(InputType(name="bits", alphabet=("0", "1")))
register_input(InputType(name="digits", alphabet=tuple(str(i) for i in range(10))))
register_input(
    InputType(
        name="letters",
        alphabet=tuple(chr(i) for i in range(ord("a"), ord("z") + 1)),
    )
)
