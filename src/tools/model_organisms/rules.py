"""Output rule definitions for the model-organisms framework.

A rule maps a sequence (iterable of alphabet tokens) to a label string.
Rules declare which input types they apply to via ``applicable`` so that
invalid combinations (e.g., ``parity`` on letters) can be caught early.

Rule functions accept ``**kwargs`` so parameterized rules (``constant(v)``,
``coin(p)``) can receive their parameters via ``rule_kwargs`` in
:func:`tools.model_organisms.generate.generate`. The seed is also
passed in, which lets ``coin`` produce labels that are a deterministic
function of ``(seed, sequence)`` — required for memorization to be
meaningful when sequences repeat.

Some rules need dataset-level state (e.g. resolved weights, output
format width, Bayes-optimal accuracy). Those rules supply a ``prepare``
hook on their :class:`Rule`. When set, ``prepare`` is invoked once per
generated dataset by :func:`tools.model_organisms.generate.generate` and
its returned dict is merged into ``rule_kwargs`` before any per-example
``fn`` call.

Rules whose dataset-level state cannot be shared safely across OOD
splits (e.g. ``weighted_sum``, where the resolved weight vector is tied
to a specific ``k`` and ``input_type``) opt out of ``design="ood"`` by
setting ``supports_ood=False`` on their :class:`Rule`. ``generate``
rejects OOD requests for those rules at validation time.

The weighted-sum rules expose **two seeds** with separate roles. The
dataset ``seed`` controls *which sequences you observe* (primary
sampling, OOD sampling, ``coin`` labels, MC samplers). The
rule-kwargs ``weight_seed`` (defaults to ``seed``) controls *what the
DGP is and how its noise behaves* — weight draws, the sparsity mask,
and (for ``weighted_sum_binary`` with ``noise_scale > 0``) the
per-sequence Bernoulli stream. Fixing ``weight_seed`` and varying
``seed`` therefore yields fresh sequence draws of the *same* DGP, with
the same per-sequence label-noise pattern.
"""

import itertools
import math
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
    prepare: Callable[..., dict] | None = None
    supports_ood: bool = True


_REGISTRY: dict[str, Rule] = {}


def register_rule(
    name: str,
    *,
    applicable: set[str],
    prepare: Callable[..., dict] | None = None,
    supports_ood: bool = True,
):
    def decorator(fn: Callable[..., str]):
        if name in _REGISTRY:
            raise ValueError(f"Rule already registered: {name}")
        _REGISTRY[name] = Rule(
            name=name,
            fn=fn,
            applicable=frozenset(applicable),
            prepare=prepare,
            supports_ood=supports_ood,
        )
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
    """Return ``v`` for every input.

    Equivalent to a degenerate :func:`weighted_sum_binary` with all-zero
    weights and an intercept whose sign picks the constant label.
    """
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
    """Return the lexicographically smallest most-common token.

    Equivalent to :func:`weighted_sum_binary` with weights all 1 and
    ``intercept="balanced"`` (which resolves to ``round(-k/2)``), with the
    lex-smallest tie-break replacing the deterministic ``z > 0`` threshold.
    """
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


# =============================================================================
# Linear (weighted-sum) rules
# =============================================================================
#
# ``weighted_sum`` and ``weighted_sum_binary`` give a known linear DGP
# (output = w·x + intercept). ``parity``, ``majority``, and ``constant``
# are special cases of this family: parity is mod-2 sum (a different
# nonlinearity), majority is weighted_sum_binary with w = 1 vector and
# the balanced intercept, and constant is the degenerate all-zero-weight
# case.

# Expected token value E[x_i] for the binary "balanced" intercept resolution.
_ALPHABET_E = {"bits": 0.5, "digits": 4.5}
# Maximum integer token value for output-width computation.
_ALPHABET_MAX = {"bits": 1, "digits": 9}


def _draw_weights(
    k: int, weight_max: int, sparsity: float, weight_seed: int
) -> list[int]:
    """Draw k integer weights; magnitude first, sparsity mask second.

    Magnitudes are drawn uniformly from ``{-W, …, -1, 1, …, W}``; each
    drawn weight is then independently zeroed with probability
    ``sparsity``. Two independent RNG streams keep the sparsity mask
    decoupled from the magnitude draw — toggling ``sparsity`` does not
    re-shuffle the underlying magnitudes.
    """
    mag_rng = random.Random(repr(("weighted_sum:weights", weight_seed)))
    mask_rng = random.Random(repr(("weighted_sum:sparsity", weight_seed)))
    nonzero = [v for v in range(-weight_max, weight_max + 1) if v != 0]
    weights: list[int] = []
    for _ in range(k):
        w = mag_rng.choice(nonzero)
        if sparsity > 0 and mask_rng.random() < sparsity:
            w = 0
        weights.append(w)
    return weights


def _resolve_dgp(
    *, input_type: str, k: int, seed: int, rule_kwargs: dict
) -> tuple[list[int], int, int]:
    """Resolve weights, intercept, and weight_seed for a weighted-sum rule.

    Returns ``(weights, intercept, weight_seed)``.
    """
    if input_type not in _ALPHABET_E:
        raise ValueError(
            f"weighted-sum rules require input_type in {sorted(_ALPHABET_E)}; "
            f"got {input_type!r}"
        )

    weight_seed = rule_kwargs.get("weight_seed")
    if weight_seed is None:
        weight_seed = seed
    weight_seed = int(weight_seed)

    explicit = rule_kwargs.get("weights")
    if explicit is not None:
        if not isinstance(explicit, (list, tuple)) or not all(
            isinstance(w, int) and not isinstance(w, bool) for w in explicit
        ):
            raise ValueError("weights must be a list of ints")
        if len(explicit) != k:
            raise ValueError(f"weights has length {len(explicit)} but k={k}")
        weights = [int(w) for w in explicit]
    else:
        weight_max = rule_kwargs.get("weight_max", 3)
        if not isinstance(weight_max, int) or weight_max < 1:
            raise ValueError(f"weight_max must be a positive int; got {weight_max!r}")
        sparsity = float(rule_kwargs.get("sparsity", 0.0))
        if not 0.0 <= sparsity <= 1.0:
            raise ValueError(f"sparsity must be in [0, 1]; got {sparsity}")
        weights = _draw_weights(k, weight_max, sparsity, weight_seed)

    intercept = rule_kwargs.get("intercept", 0)
    if intercept == "balanced":
        intercept = round(-sum(weights) * _ALPHABET_E[input_type])
    elif isinstance(intercept, bool) or not isinstance(intercept, int):
        raise ValueError(f"intercept must be an int or 'balanced'; got {intercept!r}")

    return weights, int(intercept), weight_seed


def _format_width(weights: list[int], intercept: int, alphabet_max: int) -> int:
    """Number of digits needed for the most extreme |w·x + intercept|."""
    max_mag = sum(abs(w) for w in weights) * alphabet_max + abs(intercept)
    if max_mag == 0:
        return 1
    return max(1, math.ceil(math.log10(max_mag + 1)))


def _format_signed_spaced(value: int, width: int) -> str:
    # +0Wd produces sign + W zero-padded digits (so total length = W + 1).
    return " ".join(f"{value:+0{width + 1}d}")


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


# Cap on full enumeration when computing analytical Bayes accuracy. Above
# this size we fall back to Monte Carlo.
_BAYES_ENUMERATION_CAP = 1 << 20  # ~1M sequences
_BAYES_MC_SAMPLES = 20_000


def _bayes_accuracy(
    *,
    alphabet: tuple[str, ...],
    k: int,
    weights: list[int],
    intercept: int,
    noise_scale: float,
    seed: int,
) -> float:
    """Optimal-classifier accuracy for the binary stochastic DGP.

    Under uniform x sampling, the Bayes-optimal classifier predicts the
    more-likely class at each x; its accuracy at x is ``max(p, 1-p)``
    where ``p = σ(z(x) / noise_scale)``.
    """
    space_size = len(alphabet) ** k
    if space_size <= _BAYES_ENUMERATION_CAP:
        total = 0.0
        for seq in itertools.product(alphabet, repeat=k):
            z = sum(w * int(x) for w, x in zip(weights, seq)) + intercept
            p = _sigmoid(z / noise_scale)
            total += max(p, 1.0 - p)
        return total / space_size

    rng = random.Random(repr(("bayes_acc", seed, k)))
    total = 0.0
    for _ in range(_BAYES_MC_SAMPLES):
        seq = rng.choices(alphabet, k=k)
        z = sum(w * int(x) for w, x in zip(weights, seq)) + intercept
        p = _sigmoid(z / noise_scale)
        total += max(p, 1.0 - p)
    return total / _BAYES_MC_SAMPLES


def _prepare_weighted_sum(
    *, input_type: str, k: int, N: int, seed: int, rule_kwargs: dict
) -> dict:
    weights, intercept, weight_seed = _resolve_dgp(
        input_type=input_type, k=k, seed=seed, rule_kwargs=rule_kwargs
    )
    return {
        "resolved_weights": weights,
        "intercept": intercept,
        "weight_seed": weight_seed,
        "format_width": _format_width(weights, intercept, _ALPHABET_MAX[input_type]),
    }


def _prepare_weighted_sum_binary(
    *, input_type: str, k: int, N: int, seed: int, rule_kwargs: dict
) -> dict:
    weights, intercept, weight_seed = _resolve_dgp(
        input_type=input_type, k=k, seed=seed, rule_kwargs=rule_kwargs
    )
    noise_scale = float(rule_kwargs.get("noise_scale", 0.0))
    if noise_scale < 0:
        raise ValueError(f"noise_scale must be >= 0; got {noise_scale}")

    out: dict = {
        "resolved_weights": weights,
        "intercept": intercept,
        "weight_seed": weight_seed,
        "noise_scale": noise_scale,
        "format_width": 1,  # output is always a single character
    }
    if noise_scale > 0:
        from .inputs import get_input

        out["bayes_accuracy"] = _bayes_accuracy(
            alphabet=get_input(input_type).alphabet,
            k=k,
            weights=weights,
            intercept=intercept,
            noise_scale=noise_scale,
            seed=seed,
        )
    return out


@register_rule(
    "weighted_sum",
    applicable={"bits", "digits"},
    prepare=_prepare_weighted_sum,
    supports_ood=False,
)
def weighted_sum(
    seq: Iterable[str],
    *,
    resolved_weights: list[int],
    intercept: int,
    format_width: int,
    **kwargs,
) -> str:
    """Linear DGP: return ``w·x + intercept`` as a spaced signed string.

    See :func:`_prepare_weighted_sum` for the dataset-level resolution
    of ``resolved_weights``, ``intercept``, and ``format_width``.
    """
    seq_list = list(seq)
    if len(seq_list) != len(resolved_weights):
        raise ValueError(
            f"weighted_sum: sequence length {len(seq_list)} does not match "
            f"weight vector length {len(resolved_weights)}"
        )
    z = sum(w * int(x) for w, x in zip(resolved_weights, seq_list)) + intercept
    return _format_signed_spaced(z, format_width)


@register_rule(
    "weighted_sum_binary",
    applicable={"bits", "digits"},
    prepare=_prepare_weighted_sum_binary,
    supports_ood=False,
)
def weighted_sum_binary(
    seq: Iterable[str],
    *,
    resolved_weights: list[int],
    intercept: int,
    noise_scale: float,
    weight_seed: int,
    **kwargs,
) -> str:
    """Binary linear DGP: deterministic ``z > 0`` threshold or σ-Bernoulli.

    With ``noise_scale == 0`` returns ``"1"`` if ``w·x + intercept > 0``
    else ``"0"``. With ``noise_scale > 0`` samples a Bernoulli with
    ``p = σ((w·x + intercept) / noise_scale)``, deterministic per
    ``(weight_seed, sequence)`` — keying label noise on ``weight_seed``
    (rather than the dataset seed, as ``coin`` does) lets callers fix
    the DGP and vary the dataset seed to get fresh sequence draws of
    the same DGP with the same per-sequence label-noise pattern.
    """
    seq_list = list(seq)
    if len(seq_list) != len(resolved_weights):
        raise ValueError(
            f"weighted_sum_binary: sequence length {len(seq_list)} does not "
            f"match weight vector length {len(resolved_weights)}"
        )
    z = sum(w * int(x) for w, x in zip(resolved_weights, seq_list)) + intercept
    if noise_scale == 0:
        return "1" if z > 0 else "0"
    p = _sigmoid(z / noise_scale)
    draw = random.Random(repr((weight_seed, tuple(seq_list)))).random()
    return "1" if draw < p else "0"
