#!/usr/bin/env python
"""Check installed dep versions against exact pins in pyproject.toml.

Standalone — no cruijff_kit dependency, so it works even when the package
is installed in a partial state (e.g., right after `git pull` but before
`pip install -e .`). This is what catches Matt's #503 scenario: the source
tree has a new pin, the installed metadata doesn't, and the import-time
hook in `cruijff_kit/__init__.py` doesn't fire until the env is resynced.

Usage:
    python scripts/check_env.py

Exit codes:
    0 = all exact pins match installed versions (or no pins to check)
    1 = at least one pin is out of sync — prints a table and the fix
    2 = pyproject.toml not found
"""

from __future__ import annotations

import sys
import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"


def parse_exact_pins(pyproject_path: Path) -> dict[str, str]:
    """Return {name: version} for every exact-pinned dep in pyproject.toml.

    Loose pins (>=, <, ~=) are skipped — only `==` encodes "we have tested
    this exact version and care about drift."
    """
    data = tomllib.loads(pyproject_path.read_text())
    deps = data.get("project", {}).get("dependencies", [])
    pins = {}
    for dep in deps:
        # Each dep is a PEP 508 string like "inspect-viz==0.3.5" or "torch>=2.9.1"
        # Look for exact pin "name==version" before any extras/markers.
        head = dep.split(";")[0].split("[")[0].strip()
        if "==" in head:
            name, _, ver = head.partition("==")
            name = name.strip().lower().replace("_", "-")
            ver = ver.strip()
            if name and ver:
                pins[name] = ver
    return pins


def main() -> int:
    if not PYPROJECT.exists():
        print(f"ERROR: pyproject.toml not found at {PYPROJECT}", file=sys.stderr)
        return 2

    pins = parse_exact_pins(PYPROJECT)
    if not pins:
        print("No exact-pinned deps in pyproject.toml — nothing to check.")
        return 0

    mismatches = []
    for name, expected in sorted(pins.items()):
        try:
            installed = version(name)
        except PackageNotFoundError:
            continue
        if installed != expected:
            mismatches.append((name, expected, installed))

    if not mismatches:
        print(f"OK: {len(pins)} pinned deps match installed versions.")
        return 0

    print("STALE ENV — pinned deps don't match installed:")
    print()
    print(f"  {'package':<24} {'expected':<14} {'installed':<14}")
    print(f"  {'-' * 24} {'-' * 14} {'-' * 14}")
    for name, expected, installed in mismatches:
        print(f"  {name:<24} {expected:<14} {installed:<14}")
    print()
    print("To resync:")
    print("  pip install -e .")
    return 1


if __name__ == "__main__":
    sys.exit(main())
