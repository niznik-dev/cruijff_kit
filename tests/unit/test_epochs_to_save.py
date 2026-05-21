"""Unit tests for validate_epochs_to_save (#465 defensive guard)."""

import pytest

from cruijff_kit.tools.torchtune.custom_recipes.custom_recipe_utils import (
    validate_epochs_to_save,
)


def test_all_string_expands():
    assert validate_epochs_to_save("all", total_epochs=3) == [0, 1, 2]


def test_explicit_list_passes_through():
    assert validate_epochs_to_save([0, 2], total_epochs=3) == [0, 2]


def test_single_element_list():
    assert validate_epochs_to_save([0], total_epochs=1) == [0]


def test_listconfig_is_accepted():
    from omegaconf import ListConfig

    assert validate_epochs_to_save(ListConfig([0, 1]), total_epochs=2) == [0, 1]


def test_out_of_range_raises():
    with pytest.raises(ValueError, match=r"outside \[0, 3\)"):
        validate_epochs_to_save([5], total_epochs=3)


def test_negative_index_raises():
    with pytest.raises(ValueError, match=r"outside \[0, 3\)"):
        validate_epochs_to_save([-1, 0], total_epochs=3)


def test_empty_list_raises():
    with pytest.raises(ValueError, match="empty list"):
        validate_epochs_to_save([], total_epochs=3)


def test_unexpected_string_raises():
    with pytest.raises(ValueError, match="must be 'all'"):
        validate_epochs_to_save("last", total_epochs=3)


def test_non_iterable_raises():
    with pytest.raises(ValueError, match="must be a list of ints"):
        validate_epochs_to_save(2, total_epochs=3)


def test_bool_element_raises():
    with pytest.raises(ValueError, match="wrong type"):
        validate_epochs_to_save([True, False], total_epochs=3)


def test_float_element_raises():
    with pytest.raises(ValueError, match="wrong type"):
        validate_epochs_to_save([0, 1.5], total_epochs=3)
