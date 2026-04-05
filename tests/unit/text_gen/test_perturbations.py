"""Tests for text_gen/lib/perturbations/ — synonym, shorthand, reorder, clause_addition, engine."""

import random

import pytest

from tests.unit.text_gen.conftest import make_segment
from text_gen.lib.perturbations.clause_addition import clause_addition_perturbation
from text_gen.lib.perturbations.engine import (
    apply_perturbations,
    build_perturbation_chain,
)
from text_gen.lib.perturbations.reorder import reorder_perturbation
from text_gen.lib.perturbations.shorthand import shorthand_perturbation
from text_gen.lib.perturbations.synonym import synonym_perturbation


# ---------------------------------------------------------------------------
# Synonym
# ---------------------------------------------------------------------------

class TestSynonymPerturbation:
    def test_swaps_display_name(self, sample_segments):
        rng = random.Random(42)
        result = synonym_perturbation(sample_segments, rng)
        # At least one segment should have a different display name
        changed = any(
            r.display_name != o.display_name
            for r, o in zip(result, sample_segments)
        )
        assert changed

    def test_preserves_value(self, sample_segments):
        rng = random.Random(42)
        result = synonym_perturbation(sample_segments, rng)
        for r, o in zip(result, sample_segments):
            assert r.value == o.value

    def test_no_synonyms_unchanged(self):
        seg = make_segment(metadata={"synonyms": ["only_one"], "shorthand_map": {},
                                     "restatements": [], "type": "numeric", "unit": None})
        result = synonym_perturbation([seg], random.Random(42))
        assert result[0].display_name == seg.display_name

    def test_deterministic_with_same_seed(self, sample_segments):
        r1 = synonym_perturbation(sample_segments, random.Random(99))
        r2 = synonym_perturbation(sample_segments, random.Random(99))
        assert [s.display_name for s in r1] == [s.display_name for s in r2]


# ---------------------------------------------------------------------------
# Shorthand
# ---------------------------------------------------------------------------

class TestShorthandPerturbation:
    def test_full_to_short(self, sample_segments):
        # sample_segments[1] is state="New York" with shorthand_map {"New York": "NY"}
        rng = random.Random(42)
        result = shorthand_perturbation(sample_segments, rng, direction="full_to_short")
        state_seg = result[1]
        assert state_seg.value == "NY"

    def test_short_to_full(self):
        seg = make_segment(
            field="ST", display_name="state", value="NY",
            text="The state is: NY.",
            metadata={"type": "categorical", "unit": None,
                      "synonyms": [], "shorthand_map": {"New York": "NY"},
                      "restatements": []},
        )
        result = shorthand_perturbation([seg], random.Random(42), direction="short_to_full")
        assert result[0].value == "New York"

    def test_no_match_unchanged(self):
        seg = make_segment(
            field="ST", display_name="state", value="Texas",
            text="The state is: Texas.",
            metadata={"type": "categorical", "unit": None,
                      "synonyms": [], "shorthand_map": {"New York": "NY"},
                      "restatements": []},
        )
        result = shorthand_perturbation([seg], random.Random(42))
        assert result[0].value == "Texas"


# ---------------------------------------------------------------------------
# Reorder
# ---------------------------------------------------------------------------

class TestReorderPerturbation:
    def test_same_elements(self, sample_segments):
        result = reorder_perturbation(sample_segments, random.Random(42))
        assert set(s.field for s in result) == set(s.field for s in sample_segments)
        assert len(result) == len(sample_segments)

    def test_deterministic(self, sample_segments):
        r1 = reorder_perturbation(sample_segments, random.Random(99))
        r2 = reorder_perturbation(sample_segments, random.Random(99))
        assert [s.field for s in r1] == [s.field for s in r2]

    def test_does_not_modify_original(self, sample_segments):
        original_order = [s.field for s in sample_segments]
        reorder_perturbation(sample_segments, random.Random(42))
        assert [s.field for s in sample_segments] == original_order


# ---------------------------------------------------------------------------
# Clause addition
# ---------------------------------------------------------------------------

class TestClauseAdditionPerturbation:
    def test_adds_one_clause(self, sample_segments):
        result = clause_addition_perturbation(sample_segments, random.Random(42), n_clauses=1)
        assert len(result) == len(sample_segments) + 1

    def test_added_clause_marked(self, sample_segments):
        result = clause_addition_perturbation(sample_segments, random.Random(42))
        added = [s for s in result if s.is_added]
        assert len(added) == 1

    def test_adds_multiple_clauses(self, sample_segments):
        result = clause_addition_perturbation(sample_segments, random.Random(42), n_clauses=3)
        assert len(result) == len(sample_segments) + 3

    def test_no_restatements_unchanged(self):
        seg = make_segment(metadata={"type": "numeric", "unit": None,
                                     "synonyms": [], "shorthand_map": {},
                                     "restatements": []})
        result = clause_addition_perturbation([seg], random.Random(42))
        assert len(result) == 1

    def test_decade_placeholder(self):
        seg = make_segment(
            value="51 years old",
            metadata={"type": "numeric", "unit": "years old",
                      "synonyms": [], "shorthand_map": {},
                      "restatements": ["The respondent is in their {decade}s"]},
        )
        result = clause_addition_perturbation([seg], random.Random(42))
        added = [s for s in result if s.is_added][0]
        assert "50s" in added.text


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class TestPerturbationEngine:
    def test_build_chain_validates_names(self):
        with pytest.raises(ValueError, match="Unknown perturbation"):
            build_perturbation_chain(["nonexistent"])

    def test_empty_chain_is_identity(self, sample_segments):
        chain = build_perturbation_chain([])
        result = apply_perturbations(sample_segments, chain, row_index=0)
        assert [s.text for s in result] == [s.text for s in sample_segments]

    def test_chain_applies_in_order(self, sample_segments):
        chain = build_perturbation_chain(["synonym", "reorder"], seed=42)
        result = apply_perturbations(sample_segments, chain, row_index=0)
        # Should have same number of segments, possibly different order/names
        assert len(result) == len(sample_segments)

    def test_deterministic_across_calls(self, sample_segments):
        chain = build_perturbation_chain(["synonym", "reorder"], seed=42)
        r1 = apply_perturbations(sample_segments, chain, row_index=0)
        r2 = apply_perturbations(sample_segments, chain, row_index=0)
        assert [s.text for s in r1] == [s.text for s in r2]

    def test_different_rows_get_different_results(self, sample_segments):
        chain = build_perturbation_chain(["reorder"], seed=42)
        r0 = apply_perturbations(sample_segments, chain, row_index=0)
        r1 = apply_perturbations(sample_segments, chain, row_index=1)
        # With 3 segments and different seeds, very likely different order
        # (not guaranteed, but overwhelmingly probable)
        orders_differ = [s.field for s in r0] != [s.field for s in r1]
        assert orders_differ
