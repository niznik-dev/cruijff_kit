"""Guard the `src/tools/inspect/` vs stdlib-`inspect` name shadow (#372).

`src/tools/inspect/` is named after the external inspect-ai tool it wraps, so it
shares its name with Python's standard-library `inspect` module. If that directory
ever lands on `sys.path` as a top-level entry, a bare `import inspect` inside the
subtree could bind to the package instead of the stdlib. The hazard is documented in
`docs/ARCHITECTURE.md` and `src/tools/inspect/__init__.py`; this test makes the
invariant executable so a future `import inspect` fails loudly instead of silently.
"""

import ast
from pathlib import Path

INSPECT_SUBTREE = Path(__file__).parent.parent.parent / "src" / "tools" / "inspect"


def _inspect_imports(py_file):
    """Return import statements in py_file that bind the name `inspect`."""
    tree = ast.parse(py_file.read_text(), filename=str(py_file))
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == "inspect" for alias in node.names):
                offenders.append(f"{py_file}:{node.lineno}: import inspect")
        elif isinstance(node, ast.ImportFrom):
            if node.module == "inspect" and node.level == 0:
                offenders.append(f"{py_file}:{node.lineno}: from inspect import ...")
    return offenders


def test_no_stdlib_inspect_import_in_inspect_subtree():
    """No module under src/tools/inspect/ may import the stdlib `inspect` by name."""
    offenders = []
    for py_file in INSPECT_SUBTREE.rglob("*.py"):
        offenders.extend(_inspect_imports(py_file))
    assert not offenders, (
        "src/tools/inspect/ shadows stdlib `inspect`; these imports could bind to "
        "the package instead. Alias explicitly or remove:\n  " + "\n  ".join(offenders)
    )
