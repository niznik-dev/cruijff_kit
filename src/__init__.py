"""cruijff_kit — toolkit for research with social data and LLMs."""

from .health import warn_on_mismatch as _warn_on_mismatch

# Surface stale envs the first time anything in cruijff_kit is imported.
# Silent on match; warns once per process on mismatch. See #503.
_warn_on_mismatch()
del _warn_on_mismatch
