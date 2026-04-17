"""Unit tests for sanity_checks/model_organisms/ (issue #418).

Run with:
    pytest tests/unit/test_model_organisms.py -v
"""

import pytest

from cruijff_kit.sanity_checks.model_organisms import formats, inputs
from cruijff_kit.sanity_checks.model_organisms.formats import (
    get_format,
    list_formats,
)
from cruijff_kit.sanity_checks.model_organisms.generate import (
    _sample_sequences_unique,
    generate,
)
from cruijff_kit.sanity_checks.model_organisms.inputs import (
    InputType,
    get_input,
    list_inputs,
    register_input,
)
from cruijff_kit.sanity_checks.model_organisms.rules import (
    get_rule,
    list_rules,
    register_rule,
)


# =============================================================================
# Registries
# =============================================================================


class TestRegistries:
    def test_core_names_registered(self):
        assert "bits" in list_inputs()
        assert set(list_rules()) >= {"parity", "first", "constant", "coin"}
        assert "spaced" in list_formats()

    def test_bits_alphabet(self):
        assert get_input("bits").alphabet == ("0", "1")

    def test_parity_applicable_to_bits_only(self):
        assert get_rule("parity").applicable == frozenset({"bits"})

    def test_universal_rules_applicable_to_all_inputs(self):
        for name in ("first", "constant", "coin"):
            applicable = get_rule(name).applicable
            assert {"bits", "digits", "letters"} <= applicable

    def test_unknown_name_raises(self):
        with pytest.raises(KeyError):
            get_input("no_such_input")
        with pytest.raises(KeyError):
            get_rule("no_such_rule")
        with pytest.raises(KeyError):
            get_format("no_such_format")

    def test_duplicate_input_raises(self):
        with pytest.raises(ValueError, match="already registered"):
            register_input(InputType(name="bits", alphabet=("0", "1")))

    def test_duplicate_rule_raises(self):
        with pytest.raises(ValueError, match="already registered"):

            @register_rule("parity", applicable={"bits"})
            def _dup(seq, **kwargs):
                return "0"

    def test_duplicate_format_raises(self):
        with pytest.raises(ValueError, match="already registered"):

            @formats.register_format("spaced")
            def _dup(seq):
                return "x"


# =============================================================================
# Rules
# =============================================================================


class TestRules:
    def test_parity_even_odd(self):
        parity = get_rule("parity").fn
        assert parity(["0", "0", "0"]) == "0"
        assert parity(["1", "1", "1", "1"]) == "0"
        assert parity(["1"]) == "1"
        assert parity(["1", "0", "1", "1"]) == "1"

    def test_first_returns_first_token(self):
        first = get_rule("first").fn
        assert first(["1", "0", "0"]) == "1"
        assert first(("a", "b", "c")) == "a"

    def test_constant_returns_v(self):
        const = get_rule("constant").fn
        assert const(["0", "1"], v="A") == "A"
        assert const(["0", "1"], v="hello") == "hello"

    def test_coin_deterministic_per_seed_and_sequence(self):
        coin = get_rule("coin").fn
        seq = ["1", "0", "1", "0"]
        a = coin(seq, p=0.5, seed=42)
        b = coin(seq, p=0.5, seed=42)
        assert a == b
        # Different seed usually gives a different label — try a few seeds to
        # be robust to the rare collision at p=0.5.
        labels = {coin(seq, p=0.5, seed=s) for s in range(20)}
        assert labels == {"0", "1"}

    def test_coin_p_zero_and_one(self):
        coin = get_rule("coin").fn
        seqs = [["0", "0"], ["0", "1"], ["1", "1"]]
        assert all(coin(s, p=0.0, seed=1) == "0" for s in seqs)
        assert all(coin(s, p=1.0, seed=1) == "1" for s in seqs)


# =============================================================================
# Unique sampling
# =============================================================================


class TestSampling:
    def test_returns_n_unique(self):
        import random as _r

        seqs = _sample_sequences_unique(("0", "1"), k=6, N=40, rng=_r.Random(0))
        assert len(seqs) == 40
        assert len(set(seqs)) == 40
        assert all(len(s) == 6 for s in seqs)

    def test_enumeration_exhausts_small_space(self):
        import random as _r

        # Space = 2**4 = 16. Asking for all 16 gives each sequence exactly once.
        seqs = _sample_sequences_unique(("0", "1"), k=4, N=16, rng=_r.Random(0))
        assert len(set(seqs)) == 16

    def test_oversample_raises(self):
        import random as _r

        with pytest.raises(ValueError, match="Cannot sample"):
            _sample_sequences_unique(("0", "1"), k=3, N=100, rng=_r.Random(0))


# =============================================================================
# generate()
# =============================================================================


class TestGenerate:
    def test_memorization_train_equals_val_and_unique(self):
        d = generate(
            input_type="bits",
            rule="parity",
            k=8,
            N=50,
            seed=1,
            design="memorization",
        )
        assert d["train"] == d["validation"]
        assert len({r["input"] for r in d["train"]}) == 50

    def test_in_distribution_splits_and_is_disjoint(self):
        d = generate(
            input_type="bits",
            rule="parity",
            k=10,
            N=200,
            seed=1,
            design="in_distribution",
            split=0.75,
        )
        assert len(d["train"]) == 150
        assert len(d["validation"]) == 50
        train_in = {r["input"] for r in d["train"]}
        val_in = {r["input"] for r in d["validation"]}
        assert train_in.isdisjoint(val_in)

    def test_metadata_fields_present(self):
        d = generate(
            input_type="bits",
            rule="coin",
            k=6,
            N=20,
            seed=3,
            design="in_distribution",
            rule_kwargs={"p": 0.3},
            split=0.5,
        )
        md = d["metadata"]
        for key in (
            "generator",
            "input_type",
            "rule",
            "rule_kwargs",
            "k",
            "format",
            "N",
            "seed",
            "design",
            "split",
        ):
            assert key in md, f"missing metadata key: {key}"
        # seed is stored at the top level, not inside rule_kwargs
        assert "seed" not in md["rule_kwargs"]
        assert md["rule_kwargs"] == {"p": 0.3}

    def test_coin_label_stable_across_designs(self):
        # A sequence that appears in both a memorization run and an
        # in-distribution run must get the same coin label.
        memo = generate(
            input_type="bits",
            rule="coin",
            k=6,
            N=20,
            seed=77,
            design="memorization",
            rule_kwargs={"p": 0.5},
        )
        indist = generate(
            input_type="bits",
            rule="coin",
            k=6,
            N=40,
            seed=77,
            design="in_distribution",
            rule_kwargs={"p": 0.5},
            split=0.5,
        )
        memo_labels = {r["input"]: r["output"] for r in memo["train"]}
        indist_all = indist["train"] + indist["validation"]
        overlap = [r for r in indist_all if r["input"] in memo_labels]
        assert overlap, "no overlapping sequences between runs — test is degenerate"
        for r in overlap:
            assert r["output"] == memo_labels[r["input"]]

    def test_unknown_design_raises(self):
        with pytest.raises(NotImplementedError, match="ood"):
            generate(
                input_type="bits",
                rule="parity",
                k=4,
                N=4,
                seed=0,
                design="ood",
            )

    def test_bad_split_raises(self):
        # split=0.99 with N=10 makes val empty (round(9.9) = 10)
        with pytest.raises(ValueError, match="empty train or validation"):
            generate(
                input_type="bits",
                rule="parity",
                k=6,
                N=10,
                seed=0,
                design="in_distribution",
                split=0.99,
            )

    def test_split_out_of_range_raises(self):
        with pytest.raises(ValueError, match="split must be in"):
            generate(
                input_type="bits",
                rule="parity",
                k=6,
                N=10,
                seed=0,
                design="in_distribution",
                split=1.5,
            )

    def test_applicability_error(self):
        # Register a throwaway input type so we can exercise the applicability
        # guard (parity is bits-only); tear it down afterwards.
        name = "_test_digits_for_applicability"
        register_input(InputType(name=name, alphabet=tuple("0123456789")))
        try:
            with pytest.raises(ValueError, match="not applicable"):
                generate(
                    input_type=name,
                    rule="parity",
                    k=3,
                    N=5,
                    seed=0,
                    design="memorization",
                )
        finally:
            inputs._REGISTRY.pop(name, None)
