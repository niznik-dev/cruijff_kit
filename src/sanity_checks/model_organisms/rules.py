"""Output rule definitions for the model-organisms framework.

A rule maps a sequence (iterable of alphabet tokens) to a label string.
Rules declare which input types they apply to via ``applicable`` so that
invalid combinations (e.g., ``parity`` on letters) can be caught early.

Rule functions accept ``**kwargs`` so parameterized rules (``constant(v)``,
``coin(p)``) can receive their parameters via ``rule_kwargs`` in
:func:`sanity_checks.model_organisms.generate.generate`. The seed is also
passed in, which lets ``coin`` produce labels that are a deterministic
function of ``(seed, sequence)`` — required for memorization to be
meaningful when sequences repeat.
"""

import random
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Rule:
    name: str
    fn: Callable[..., str]
    applicable: frozenset[str]


_REGISTRY: dict[str, Rule] = {}


def register_rule(name: str, *, applicable: set[str]):
    def decorator(fn: Callable[..., str]):
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
def parity(seq: Iterable[str], **kwargs) -> str:
    return "1" if sum(int(x) for x in seq) % 2 == 1 else "0"


@register_rule("first", applicable={"bits", "digits", "letters"})
def first(seq: Iterable[str], **kwargs) -> str:
    return next(iter(seq))


@register_rule("constant", applicable={"bits", "digits", "letters"})
def constant(seq: Iterable[str], *, v: str, **kwargs) -> str:
    return v


@register_rule("coin", applicable={"bits", "digits", "letters"})
def coin(seq: Iterable[str], *, p: float, seed: int, **kwargs) -> str:
    # repr keeps the label deterministic per (seed, sequence) on Python 3.13+,
    # which no longer accepts tuples as Random seeds.
    draw = random.Random(repr((seed, tuple(seq)))).random()
    return "1" if draw < p else "0"


@register_rule("last", applicable={"bits", "digits", "letters"})
def last(seq: Iterable[str], **kwargs) -> str:
    seq_list = list(seq)
    if not seq_list:
        raise ValueError("last() requires a non-empty sequence")
    return seq_list[-1]


@register_rule("nth", applicable={"bits", "digits", "letters"})
def nth(seq: Iterable[str], *, x: int, **kwargs) -> str:
    seq_list = list(seq)
    try:
        return seq_list[x]
    except IndexError as exc:
        raise ValueError(
            f"nth(x={x}) out of range for sequence of length {len(seq_list)}"
        ) from exc


@register_rule("length", applicable={"bits", "digits", "letters"})
def length(seq: Iterable[str], **kwargs) -> str:
    return str(sum(1 for _ in seq))


@register_rule("majority", applicable={"bits"})
def majority(seq: Iterable[str], **kwargs) -> str:
    counts = Counter(seq)
    if not counts:
        raise ValueError("majority() requires a non-empty sequence")
    max_count = max(counts.values())
    # Lex-ascending tie-break keeps labels deterministic on ties (common for
    # even-length bit sequences).
    return min(tok for tok, c in counts.items() if c == max_count)


@register_rule("min", applicable={"digits", "letters"})
def min_token(seq: Iterable[str], **kwargs) -> str:
    seq_list = list(seq)
    if not seq_list:
        raise ValueError("min() requires a non-empty sequence")
    return min(seq_list)


@register_rule("max", applicable={"digits", "letters"})
def max_token(seq: Iterable[str], **kwargs) -> str:
    seq_list = list(seq)
    if not seq_list:
        raise ValueError("max() requires a non-empty sequence")
    return max(seq_list)
