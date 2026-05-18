"""Tests for cruijff_kit.utils.run_names."""

from cruijff_kit.utils.run_names import (
    generate_model_run_name,
    POSITIVE_ADJECTIVES,
    POSITIVE_NOUNS,
)


class TestGenerateModelRunName:
    """Tests for generate_model_run_name."""

    def test_returns_list(self):
        result = generate_model_run_name()
        assert isinstance(result, list)

    def test_default_returns_one_name(self):
        result = generate_model_run_name()
        assert len(result) == 1

    def test_returns_requested_count(self):
        for n in [1, 3, 10]:
            result = generate_model_run_name(num_names=n)
            assert len(result) == n

    def test_name_format(self):
        """Each name should be adjective_noun."""
        names = generate_model_run_name(num_names=20)
        for name in names:
            parts = name.split("_")
            assert len(parts) == 2, f"Expected 'adj_noun', got '{name}'"
            assert parts[0] in POSITIVE_ADJECTIVES
            assert parts[1] in POSITIVE_NOUNS

    def test_names_are_strings(self):
        names = generate_model_run_name(num_names=5)
        for name in names:
            assert isinstance(name, str)

    def test_zero_names(self):
        result = generate_model_run_name(num_names=0)
        assert result == []

    def test_randomness(self):
        """Two calls with many names should not produce identical lists."""
        a = generate_model_run_name(num_names=50)
        b = generate_model_run_name(num_names=50)
        assert a != b, "50 random names should differ between calls"


class TestWordLists:
    """Sanity checks on the word lists."""

    def test_adjectives_nonempty(self):
        assert len(POSITIVE_ADJECTIVES) > 0

    def test_nouns_nonempty(self):
        assert len(POSITIVE_NOUNS) > 0

    def test_no_duplicate_adjectives(self):
        assert len(POSITIVE_ADJECTIVES) == len(set(POSITIVE_ADJECTIVES))

    def test_no_duplicate_nouns(self):
        assert len(POSITIVE_NOUNS) == len(set(POSITIVE_NOUNS))

    def test_all_lowercase(self):
        for word in POSITIVE_ADJECTIVES:
            assert word == word.lower(), f"Adjective not lowercase: {word}"
        for word in POSITIVE_NOUNS:
            assert word == word.lower(), f"Noun not lowercase: {word}"
