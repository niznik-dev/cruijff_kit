"""Shared fixtures for fixture-based tests.

Path placeholder convention:
  __SCRATCH__ → tmp_path / "scratch"
  __REPO__    → tmp_path / "repo"

All fixture YAML files use these tokens instead of real paths.
The `resolve_fixture` helper replaces them at test time.
"""

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent


def resolve_placeholders(text: str, scratch: Path, repo: Path) -> str:
    """Replace __SCRATCH__ and __REPO__ tokens with real paths."""
    return text.replace("__SCRATCH__", str(scratch)).replace("__REPO__", str(repo))


@pytest.fixture
def scratch_dir(tmp_path):
    """Create and return a scratch directory inside tmp_path."""
    d = tmp_path / "scratch"
    d.mkdir()
    return d


@pytest.fixture
def repo_dir(tmp_path):
    """Create and return a repo directory inside tmp_path."""
    d = tmp_path / "repo"
    d.mkdir()
    return d


@pytest.fixture
def resolved_fixture(tmp_path, scratch_dir, repo_dir):
    """Return a helper that copies a fixture file to tmp_path with placeholders resolved.

    Usage:
        path = resolved_fixture("design/experiment_summary.yaml")
    """

    def _resolve(relative_path: str, dest: Path | None = None) -> Path:
        src = FIXTURES_DIR / relative_path
        content = src.read_text()
        resolved = resolve_placeholders(content, scratch_dir, repo_dir)

        if dest is None:
            dest = tmp_path / Path(relative_path).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(resolved)
        return dest

    return _resolve
