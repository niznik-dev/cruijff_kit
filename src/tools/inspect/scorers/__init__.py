# Custom inspect-ai scorers for cruijff_kit.
#
# To add a new scorer:
#   1. Create a new file in this directory (e.g., my_scorer.py)
#   2. Decorate your scorer function with @scorer
#   3. Import it here and add it to SCORER_REGISTRY

from inspect_ai.scorer import match, includes
from .risk_scorer import risk_scorer
from .numeric_risk_scorer import numeric_risk_scorer
from .continuous_scorer import continuous_scorer
from .calibration_metrics import (
    expected_calibration_error as expected_calibration_error,
    brier_score as brier_score,
    auc_score as auc_score,
)

# Underlying scorer factories keyed by registered name. Used for capability
# introspection (e.g., `requires_logprobs`) before instantiation.
SCORER_FACTORIES = {
    "match": match,
    "includes": includes,
    "risk_scorer": risk_scorer,
    "numeric_risk_scorer": numeric_risk_scorer,
    "continuous_scorer": continuous_scorer,
}

# Registry of available scorers and their constructors.
# Task files use build_scorers() to instantiate scorers from YAML config.
SCORER_REGISTRY = {
    "match": lambda params: (
        match(**params) if params else match(location="exact", ignore_case=False)
    ),
    "includes": lambda params: (
        includes(**params) if params else includes(ignore_case=False)
    ),
    "risk_scorer": lambda params: risk_scorer(**params) if params else risk_scorer(),
    "numeric_risk_scorer": lambda params: (
        numeric_risk_scorer(**params) if params else numeric_risk_scorer()
    ),
    "continuous_scorer": lambda params: (
        continuous_scorer(**params) if params else continuous_scorer()
    ),
}


def configured_scorers_require_logprobs(config: dict) -> bool:
    """Return True if any scorer in the YAML config declares it needs logprobs.

    Detection is attribute-based (not name-based): a scorer factory opts in by
    setting ``requires_logprobs = True``. This avoids over-including scorers
    whose names happen to match a heuristic but which read text completion
    instead of logprobs (e.g., `numeric_risk_scorer`).
    """
    for entry in config.get("scorer", []) or []:
        factory = SCORER_FACTORIES.get(entry.get("name"))
        if factory is not None and getattr(factory, "requires_logprobs", False):
            return True
    return False


# Default scorers when no config is provided
DEFAULT_SCORERS = [
    match(location="exact", ignore_case=False),
    includes(ignore_case=False),
]


def build_scorers(config: dict) -> list:
    """Build scorer list from eval config YAML scorer section.

    Args:
        config: Parsed YAML config dict. If it contains a 'scorer' key,
                builds scorers from that list. Otherwise returns DEFAULT_SCORERS.

    Returns:
        List of instantiated scorer objects.
    """
    scorer_config = config.get("scorer")
    if not scorer_config:
        return list(DEFAULT_SCORERS)

    scorers = []
    for entry in scorer_config:
        name = entry["name"]
        params = entry.get("params", {})
        if name not in SCORER_REGISTRY:
            raise ValueError(
                f"Unknown scorer '{name}'. Available: {list(SCORER_REGISTRY.keys())}"
            )
        scorers.append(SCORER_REGISTRY[name](params))
    return scorers
