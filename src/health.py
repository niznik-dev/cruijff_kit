"""Runtime dep-version sanity checks for cruijff_kit.

Catches the failure mode where a conda env was created when an older version
of a pinned dep was the latest available, then never re-resolved (see #503).
The first import of `cruijff_kit` calls `warn_on_mismatch()`, which warns
once per process if any exact-pinned (==) dep disagrees with what's installed.

CLI usage:
    python -m cruijff_kit.health
    # exits non-zero if any pinned dep is out of sync; useful in make targets
    # and CI.

Set CK_SKIP_VERSION_CHECK=1 to suppress the import-time warning.
"""

from __future__ import annotations

import os
import re
import sys
import warnings
from importlib.metadata import PackageNotFoundError, distribution, version

_VERSION_WARNING_EMITTED = False


def _parse_exact_pins(requires_strs):
    """Extract {name: version} for exact-pinned (==) deps from a requires list.

    Loose pins (>=, <, ~=) are intentionally skipped — only exact pins encode
    "we have tested this version and care about drift."
    """
    pins = {}
    if not requires_strs:
        return pins
    for r in requires_strs:
        m = re.match(r"^([A-Za-z0-9_.\-]+)\s*==\s*([^\s,;]+)", r)
        if m:
            pins[m.group(1).lower().replace("_", "-")] = m.group(2)
    return pins


def _installed_version(pkg):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


def check_versions(pinned=None):
    """Return mismatches as a list of (name, expected, installed) tuples.

    Args:
        pinned: Optional {name: expected_version} dict. If None, derived from
            cruijff_kit's own installed package metadata.

    The caller decides whether to warn, raise, or print.
    """
    if pinned is None:
        try:
            requires = distribution("cruijff_kit").requires
        except PackageNotFoundError:
            return []
        pinned = _parse_exact_pins(requires)

    mismatches = []
    for name, expected in pinned.items():
        installed = _installed_version(name)
        if installed is not None and installed != expected:
            mismatches.append((name, expected, installed))
    return mismatches


def _format_warning_message(mismatches):
    """Pure: render mismatches into the warning text shown to the user."""
    lines = [
        f"  {name}: expected {expected}, installed {installed}"
        for name, expected, installed in mismatches
    ]
    fix = " ".join(f"{name}=={expected}" for name, expected, _ in mismatches)
    return (
        "cruijff_kit dep version mismatch — env may be stale:\n"
        + "\n".join(lines)
        + f"\n  Fix: pip install {fix}"
    )


def warn_on_mismatch():
    """Warn once per process if pinned deps disagree with installed versions."""
    global _VERSION_WARNING_EMITTED
    if _VERSION_WARNING_EMITTED or os.environ.get("CK_SKIP_VERSION_CHECK"):
        return
    _VERSION_WARNING_EMITTED = True
    mismatches = check_versions()
    if mismatches:
        warnings.warn(_format_warning_message(mismatches), stacklevel=2)


def import_inspect_views(*names):
    """Import names from inspect_viz.view, with a friendly error on failure.

    Skill docs use this instead of bare `from inspect_viz.view import ...` so
    that a layout mismatch (e.g., on inspect-viz 0.3.4 the names live under
    inspect_viz.view.beta) yields a message pointing at the pin, not a raw
    ImportError that buries the cause.

    Returns a tuple of the requested names, in argument order, for tuple
    unpacking at the call site.
    """
    try:
        import inspect_viz.view as iv_view  # noqa: F401

        return tuple(getattr(iv_view, n) for n in names)
    except (ImportError, AttributeError) as e:
        try:
            import inspect_viz

            installed = inspect_viz.__version__
        except Exception:
            installed = "(unknown)"
        raise ImportError(
            f"Could not import {list(names)} from inspect_viz.view "
            f"(installed inspect_viz {installed}). cruijff_kit pins "
            f"inspect-viz==0.3.5; older versions keep these names under "
            f"inspect_viz.view.beta. Run: pip install 'inspect-viz==0.3.5'"
        ) from e


def main():
    """CLI: print the pinned-vs-installed table, exit nonzero on mismatch."""
    try:
        requires = distribution("cruijff_kit").requires
    except PackageNotFoundError:
        print(
            "ERROR: cruijff_kit metadata not found — is the package installed?",
            file=sys.stderr,
        )
        sys.exit(2)

    pinned = _parse_exact_pins(requires)
    if not pinned:
        print("No exact-pinned deps in cruijff_kit metadata.")
        sys.exit(0)

    print(f"{'package':<24} {'expected':<14} {'installed':<14} status")
    print("-" * 64)
    any_mismatch = False
    for name, expected in sorted(pinned.items()):
        installed = _installed_version(name) or "(missing)"
        status = "OK" if installed == expected else "MISMATCH"
        if status == "MISMATCH":
            any_mismatch = True
        print(f"{name:<24} {expected:<14} {installed:<14} {status}")

    if any_mismatch:
        print(
            "\nMismatches detected. To resync: "
            "pip install " + " ".join(f"{n}=={v}" for n, v in pinned.items())
        )
    sys.exit(1 if any_mismatch else 0)


if __name__ == "__main__":
    main()
