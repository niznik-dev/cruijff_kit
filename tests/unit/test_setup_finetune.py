"""Unit tests for tools/torchtune/setup_finetune.py"""

import pytest
from cruijff_kit.tools.torchtune.setup_finetune import parse_epochs


def test_parse_epochs_all():
    """Test that 'all' is parsed correctly."""
    result = parse_epochs("all")
    assert result == "all"


def test_parse_epochs_none():
    """Test that 'none' is parsed as empty list."""
    result = parse_epochs("none")
    assert result == []


def test_parse_epochs_comma_delimited():
    """Test comma-delimited epochs."""
    result = parse_epochs("0,1,2")
    assert result == [0, 1, 2]


def test_parse_epochs_single_value():
    """Test single epoch value."""
    result = parse_epochs("5")
    assert result == [5]
