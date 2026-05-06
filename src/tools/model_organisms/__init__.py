"""Model-organisms framework: parameterized sanity-check dataset generator.

Submodules register input types, rules, and formats at import time, so
importing this package (or any submodule) ensures the registries are
populated before callers look things up.
"""

from . import formats  # noqa: F401
from . import inputs  # noqa: F401
from . import rules  # noqa: F401
