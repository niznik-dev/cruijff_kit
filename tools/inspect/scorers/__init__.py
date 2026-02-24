# Custom inspect-ai scorers for cruijff_kit.
#
# To add a new scorer:
#   1. Create a new file in this directory (e.g., my_scorer.py)
#   2. Decorate your scorer function with @scorer
#   3. Import it here and add it to SCORER_REGISTRY

from inspect_ai.scorer import match, includes
from .risk_scorer import risk_scorer
from .numeric_risk_scorer import numeric_risk_scorer
from .calibration_metrics import expected_calibration_error, brier_score, auc_score

# Registry of available scorers and their constructors.
# Task files use build_scorers() to instantiate scorers from YAML config.
SCORER_REGISTRY = {
    "match": lambda params: match(**params) if params else match(location="exact", ignore_case=False),
    "includes": lambda params: includes(**params) if params else includes(ignore_case=False),
    "risk_scorer": lambda params: risk_scorer(**params) if params else risk_scorer(),
    "numeric_risk_scorer": lambda params: numeric_risk_scorer(**params) if params else numeric_risk_scorer(),
}

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
        return DEFAULT_SCORERS

    scorers = []
    for entry in scorer_config:
        name = entry["name"]
        params = entry.get("params", {})
        if name not in SCORER_REGISTRY:
            raise ValueError(f"Unknown scorer '{name}'. Available: {list(SCORER_REGISTRY.keys())}")
        scorers.append(SCORER_REGISTRY[name](params))
    return scorers
