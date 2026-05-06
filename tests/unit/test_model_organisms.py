"""Unit tests for tools/model_organisms/ (issue #418).

Run with:
    pytest tests/unit/test_model_organisms.py -v
"""

import pytest

from cruijff_kit.tools.model_organisms import formats
from cruijff_kit.tools.model_organisms.formats import (
    get_format,
    list_formats,
)
from cruijff_kit.tools.model_organisms.generate import (
    _sample_sequences_unique,
    generate,
)
from cruijff_kit.tools.model_organisms.inputs import (
    InputType,
    get_input,
    list_inputs,
    register_input,
)
from cruijff_kit.tools.model_organisms.rules import (
    get_rule,
    list_rules,
    register_rule,
)


# =============================================================================
# Registries
# =============================================================================


class TestRegistries:
    def test_core_names_registered(self):
        assert set(list_inputs()) >= {"bits", "digits", "letters"}
        assert set(list_rules()) >= {
            "parity",
            "first",
            "last",
            "nth",
            "length",
            "constant",
            "coin",
            "majority",
            "min",
            "max",
            "weighted_sum",
            "weighted_sum_binary",
        }
        assert "spaced" in list_formats()

    def test_weighted_sum_rules_applicable_to_bits_and_digits_only(self):
        for name in ("weighted_sum", "weighted_sum_binary"):
            assert get_rule(name).applicable == frozenset({"bits", "digits"})

    def test_bits_alphabet(self):
        assert get_input("bits").alphabet == ("0", "1")

    def test_digits_alphabet(self):
        assert get_input("digits").alphabet == tuple(str(i) for i in range(10))

    def test_letters_alphabet(self):
        letters = get_input("letters").alphabet
        assert len(letters) == 26
        assert letters[0] == "a" and letters[-1] == "z"

    def test_parity_and_majority_bits_only(self):
        assert get_rule("parity").applicable == frozenset({"bits"})
        assert get_rule("majority").applicable == frozenset({"bits"})

    def test_min_max_digits_and_letters_only(self):
        for name in ("min", "max"):
            assert get_rule(name).applicable == frozenset({"digits", "letters"})

    def test_universal_rules_applicable_to_all_inputs(self):
        for name in ("first", "last", "nth", "length", "constant", "coin"):
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

    def test_all_formats_registered(self):
        assert set(list_formats()) >= {"spaced", "dense", "comma", "tab", "pipe"}


# =============================================================================
# Formats
# =============================================================================


class TestFormats:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("spaced", "0 1 1 0"),
            ("dense", "0110"),
            ("comma", "0,1,1,0"),
            ("tab", "0\t1\t1\t0"),
            ("pipe", "0|1|1|0"),
        ],
    )
    def test_format_joins_correctly(self, name, expected):
        fmt = get_format(name)
        assert fmt(["0", "1", "1", "0"]) == expected


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

    def test_last_returns_last_token(self):
        last = get_rule("last").fn
        assert last(["1", "0", "0"]) == "0"
        assert last(("a", "b", "c")) == "c"

    def test_length_returns_sequence_length(self):
        length = get_rule("length").fn
        assert length(["1", "0", "0"]) == "3"
        assert length([]) == "0"

    def test_nth_positive_and_negative_index(self):
        nth = get_rule("nth").fn
        seq = ["a", "b", "c", "d"]
        assert nth(seq, x=0) == "a"
        assert nth(seq, x=2) == "c"
        assert nth(seq, x=-1) == "d"

    def test_nth_out_of_range_raises(self):
        nth = get_rule("nth").fn
        with pytest.raises(ValueError, match="out of range"):
            nth(["a", "b"], x=5)

    def test_majority_picks_most_common(self):
        majority = get_rule("majority").fn
        assert majority(["0", "0", "0", "1", "1"]) == "0"
        assert majority(["1", "1", "1", "0"]) == "1"

    def test_majority_tie_breaks_lex_smallest(self):
        majority = get_rule("majority").fn
        # 2 zeros, 2 ones — tie. Lex smallest is "0".
        assert majority(["0", "0", "1", "1"]) == "0"

    def test_min_max_on_digits(self):
        min_fn = get_rule("min").fn
        max_fn = get_rule("max").fn
        seq = ["5", "2", "9", "4"]
        assert min_fn(seq) == "2"
        assert max_fn(seq) == "9"

    def test_min_max_on_letters(self):
        min_fn = get_rule("min").fn
        max_fn = get_rule("max").fn
        seq = ["m", "a", "z", "d"]
        assert min_fn(seq) == "a"
        assert max_fn(seq) == "z"


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
        with pytest.raises(NotImplementedError, match="nonexistent_design"):
            generate(
                input_type="bits",
                rule="parity",
                k=4,
                N=4,
                seed=0,
                design="nonexistent_design",
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

    def test_applicability_error_parity_on_digits(self):
        with pytest.raises(ValueError, match="not applicable"):
            generate(
                input_type="digits",
                rule="parity",
                k=3,
                N=5,
                seed=0,
                design="memorization",
            )

    def test_applicability_error_min_on_bits(self):
        with pytest.raises(ValueError, match="not applicable"):
            generate(
                input_type="bits",
                rule="min",
                k=3,
                N=5,
                seed=0,
                design="memorization",
            )

    def test_generate_digits_max(self):
        d = generate(
            input_type="digits",
            rule="max",
            k=5,
            N=30,
            seed=1,
            design="memorization",
        )
        for row in d["train"]:
            tokens = row["input"].split()
            assert row["output"] == max(tokens)

    def test_generate_letters_min(self):
        d = generate(
            input_type="letters",
            rule="min",
            k=4,
            N=30,
            seed=1,
            design="memorization",
        )
        for row in d["train"]:
            tokens = row["input"].split()
            assert row["output"] == min(tokens)

    def test_generate_bits_majority(self):
        d = generate(
            input_type="bits",
            rule="majority",
            k=5,
            N=32,
            seed=1,
            design="memorization",
        )
        from collections import Counter

        for row in d["train"]:
            tokens = row["input"].split()
            counts = Counter(tokens)
            max_c = max(counts.values())
            expected = min(t for t, c in counts.items() if c == max_c)
            assert row["output"] == expected

    def test_generate_nth_rule_kwargs(self):
        d = generate(
            input_type="digits",
            rule="nth",
            k=5,
            N=20,
            seed=2,
            design="memorization",
            rule_kwargs={"x": 2},
        )
        assert d["metadata"]["rule_kwargs"] == {"x": 2}
        for row in d["train"]:
            tokens = row["input"].split()
            assert row["output"] == tokens[2]


# =============================================================================
# OOD design
# =============================================================================


class TestOOD:
    def test_length_ood_produces_extra_val_splits(self):
        d = generate(
            input_type="bits",
            rule="parity",
            k=8,
            N=100,
            seed=1,
            design="ood",
            split=0.8,
            ood_tests=[
                {"k": 12, "N": 40},
                {"k": 16, "N": 40},
            ],
        )
        assert len(d["train"]) == 80
        assert len(d["validation"]) == 20
        assert len(d["validation_ood_0"]) == 40
        assert len(d["validation_ood_1"]) == 40
        # Each OOD split has the right sequence length
        assert all(len(r["input"].split()) == 12 for r in d["validation_ood_0"])
        assert all(len(r["input"].split()) == 16 for r in d["validation_ood_1"])
        # Metadata records resolved specs
        assert d["metadata"]["ood_tests"] == [
            {
                "input_type": "bits",
                "k": 12,
                "N": 40,
                "format": "spaced",
                "rule_kwargs": {},
            },
            {
                "input_type": "bits",
                "k": 16,
                "N": 40,
                "format": "spaced",
                "rule_kwargs": {},
            },
        ]

    def test_format_ood(self):
        d = generate(
            input_type="bits",
            rule="parity",
            k=6,
            N=40,
            seed=1,
            design="ood",
            split=0.75,
            ood_tests=[{"format": "dense", "N": 20}],
        )
        # Primary uses spaced (tokens separated by spaces)
        assert " " in d["train"][0]["input"]
        # OOD uses dense (no separator)
        assert all(" " not in r["input"] for r in d["validation_ood_0"])
        assert all(len(r["input"]) == 6 for r in d["validation_ood_0"])

    def test_domain_ood_rejects_when_rule_not_applicable(self):
        # parity is bits-only, so a digits OOD spec should fail fast
        with pytest.raises(ValueError, match="ood_tests\\[0\\]"):
            generate(
                input_type="bits",
                rule="parity",
                k=8,
                N=40,
                seed=1,
                design="ood",
                ood_tests=[{"input_type": "digits", "N": 20}],
            )

    def test_ood_coin_label_consistent_with_primary(self):
        # A coin sequence that happens to appear in both primary and ood
        # must get the same label (same seed, same sequence).
        d = generate(
            input_type="bits",
            rule="coin",
            k=5,
            N=32,
            seed=123,
            design="ood",
            split=0.75,
            rule_kwargs={"p": 0.5},
            ood_tests=[{"k": 5, "N": 32}],
        )
        primary = {r["input"]: r["output"] for r in d["train"] + d["validation"]}
        ood = d["validation_ood_0"]
        overlap = [r for r in ood if r["input"] in primary]
        assert overlap, "test degenerate — no overlap between splits"
        for r in overlap:
            assert r["output"] == primary[r["input"]]

    def test_ood_requires_nonempty_list(self):
        with pytest.raises(ValueError, match="ood_tests"):
            generate(
                input_type="bits",
                rule="parity",
                k=8,
                N=40,
                seed=1,
                design="ood",
            )

    def test_ood_spec_missing_N_raises(self):
        with pytest.raises(ValueError, match="missing required field 'N'"):
            generate(
                input_type="bits",
                rule="parity",
                k=8,
                N=40,
                seed=1,
                design="ood",
                ood_tests=[{"k": 12}],
            )


# =============================================================================
# Rule.prepare hook
# =============================================================================


class TestPrepareHook:
    """Generic checks for the prepare extension to the Rule dataclass.

    Behavioral coverage of weighted-sum-specific resolution lives below.
    """

    def test_existing_rules_have_no_prepare(self):
        for name in (
            "parity",
            "first",
            "last",
            "nth",
            "length",
            "constant",
            "coin",
            "majority",
            "min",
            "max",
        ):
            assert get_rule(name).prepare is None

    def test_weighted_sum_rules_have_prepare(self):
        for name in ("weighted_sum", "weighted_sum_binary"):
            assert get_rule(name).prepare is not None

    def test_existing_rules_support_ood(self):
        # Default supports_ood=True; existing OOD tests rely on this.
        for name in (
            "parity",
            "first",
            "last",
            "nth",
            "length",
            "constant",
            "coin",
            "majority",
            "min",
            "max",
        ):
            assert get_rule(name).supports_ood is True

    def test_weighted_sum_rules_do_not_support_ood(self):
        for name in ("weighted_sum", "weighted_sum_binary"):
            assert get_rule(name).supports_ood is False

    def test_prepare_output_surfaces_at_top_level_metadata(self):
        d = generate(
            input_type="digits",
            rule="weighted_sum",
            k=3,
            N=5,
            seed=1,
            design="memorization",
            rule_kwargs={"weights": [1, 2, -1], "intercept": 0},
        )
        assert d["metadata"]["resolved_weights"] == [1, 2, -1]
        assert d["metadata"]["intercept"] == 0
        assert d["metadata"]["format_width"] == 2  # max |w·x|=27 → 2 digits


# =============================================================================
# weighted_sum (linear regression DGP)
# =============================================================================


class TestWeightedSum:
    def test_explicit_weights_arithmetic(self):
        d = generate(
            input_type="digits",
            rule="weighted_sum",
            k=3,
            N=10,
            seed=1,
            design="memorization",
            rule_kwargs={"weights": [1, 2, -1], "intercept": 5},
        )
        assert d["metadata"]["resolved_weights"] == [1, 2, -1]
        assert d["metadata"]["intercept"] == 5
        for row in d["train"]:
            tokens = [int(t) for t in row["input"].split()]
            z = tokens[0] + 2 * tokens[1] - tokens[2] + 5
            assert row["output"] == _format_signed_spaced_str(
                z, d["metadata"]["format_width"]
            )

    def test_output_format_signed_spaced_zero_padded(self):
        # weights=[1,2,-1,3,0,1,-2,1] on digits → max |output| = 99 → width 2
        d = generate(
            input_type="digits",
            rule="weighted_sum",
            k=8,
            N=20,
            seed=2,
            design="memorization",
            rule_kwargs={
                "weights": [1, 2, -1, 3, 0, 1, -2, 1],
                "intercept": 0,
            },
        )
        assert d["metadata"]["format_width"] == 2
        for row in d["train"]:
            parts = row["output"].split(" ")
            # Always: one sign char + format_width digit chars
            assert len(parts) == 3
            assert parts[0] in {"+", "-"}
            assert all(p.isdigit() and len(p) == 1 for p in parts[1:])

    def test_distribution_drawn_weights_deterministic_per_seed(self):
        a = generate(
            input_type="digits",
            rule="weighted_sum",
            k=6,
            N=2,
            seed=1,
            design="memorization",
            rule_kwargs={"weight_max": 3, "weight_seed": 99},
        )
        b = generate(
            input_type="digits",
            rule="weighted_sum",
            k=6,
            N=2,
            seed=2,  # different DATASET seed
            design="memorization",
            rule_kwargs={"weight_max": 3, "weight_seed": 99},
        )
        assert a["metadata"]["resolved_weights"] == b["metadata"]["resolved_weights"]

    def test_weight_seed_defaults_to_dataset_seed(self):
        d = generate(
            input_type="digits",
            rule="weighted_sum",
            k=4,
            N=2,
            seed=1234,
            design="memorization",
            rule_kwargs={"weight_max": 2},
        )
        assert d["metadata"]["weight_seed"] == 1234

    def test_drawn_weights_excluded_zero_then_masked(self):
        # No sparsity → weights all in {-W,…,-1,1,…,W}, never zero.
        d = generate(
            input_type="digits",
            rule="weighted_sum",
            k=200,  # large enough that "no zeros" is statistically meaningful
            N=2,
            seed=1,
            design="memorization",
            rule_kwargs={"weight_max": 5, "sparsity": 0.0, "weight_seed": 7},
        )
        weights = d["metadata"]["resolved_weights"]
        assert all(w != 0 for w in weights)
        assert all(-5 <= w <= 5 for w in weights)

    def test_sparsity_one_zeros_all_weights(self):
        d = generate(
            input_type="digits",
            rule="weighted_sum",
            k=8,
            N=4,
            seed=3,
            design="memorization",
            rule_kwargs={
                "weight_max": 3,
                "sparsity": 1.0,
                "intercept": 7,
                "weight_seed": 5,
            },
        )
        assert d["metadata"]["resolved_weights"] == [0] * 8
        # All-zero weights → output is the formatted intercept everywhere.
        assert all(r["output"] == d["train"][0]["output"] for r in d["train"])

    def test_balanced_intercept_digits(self):
        # E[x_i] for digits = 4.5 → intercept = round(-sum(w) * 4.5).
        d = generate(
            input_type="digits",
            rule="weighted_sum",
            k=4,
            N=2,
            seed=1,
            design="memorization",
            rule_kwargs={"weights": [1, 1, 1, 1], "intercept": "balanced"},
        )
        assert d["metadata"]["intercept"] == round(-4 * 4.5)

    def test_balanced_intercept_bits(self):
        d = generate(
            input_type="bits",
            rule="weighted_sum",
            k=8,
            N=2,
            seed=1,
            design="memorization",
            rule_kwargs={"weights": [1] * 8, "intercept": "balanced"},
        )
        assert d["metadata"]["intercept"] == round(-8 * 0.5)  # = -4

    def test_format_width_handles_all_zero_weights_zero_intercept(self):
        d = generate(
            input_type="digits",
            rule="weighted_sum",
            k=4,
            N=2,
            seed=1,
            design="memorization",
            rule_kwargs={
                "weight_max": 1,
                "sparsity": 1.0,
                "intercept": 0,
                "weight_seed": 11,
            },
        )
        assert d["metadata"]["format_width"] == 1
        # Output is 0 with width 1: "+ 0".
        assert d["train"][0]["output"] == "+ 0"

    def test_k_one_edge_case(self):
        d = generate(
            input_type="digits",
            rule="weighted_sum",
            k=1,
            N=10,
            seed=1,
            design="memorization",
            rule_kwargs={"weights": [2], "intercept": -3},
        )
        # Max |output| = 2*9 + 3 = 21 → width 2
        assert d["metadata"]["format_width"] == 2
        for row in d["train"]:
            x = int(row["input"])
            assert row["output"] == _format_signed_spaced_str(2 * x - 3, 2)

    def test_letters_rejected(self):
        with pytest.raises(ValueError, match="not applicable"):
            generate(
                input_type="letters",
                rule="weighted_sum",
                k=3,
                N=4,
                seed=1,
                design="memorization",
                rule_kwargs={"weights": [1, 1, 1]},
            )

    def test_weights_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="weights has length"):
            generate(
                input_type="digits",
                rule="weighted_sum",
                k=3,
                N=4,
                seed=1,
                design="memorization",
                rule_kwargs={"weights": [1, 2]},
            )

    def test_invalid_intercept_rejected(self):
        with pytest.raises(ValueError, match="intercept must be"):
            generate(
                input_type="digits",
                rule="weighted_sum",
                k=2,
                N=4,
                seed=1,
                design="memorization",
                rule_kwargs={"weights": [1, 1], "intercept": 1.5},
            )


def _format_signed_spaced_str(value: int, width: int) -> str:
    """Local helper mirroring the rule's output format for assertions."""
    return " ".join(f"{value:+0{width + 1}d}")


# =============================================================================
# weighted_sum_binary (linear classification DGP)
# =============================================================================


class TestWeightedSumBinary:
    def test_deterministic_threshold_z_gt_zero(self):
        # z > 0 → "1", z <= 0 → "0".
        d = generate(
            input_type="bits",
            rule="weighted_sum_binary",
            k=4,
            N=16,  # exhaust 2**4 space
            seed=1,
            design="memorization",
            rule_kwargs={"weights": [1, 1, 1, 1], "intercept": -2},
        )
        assert d["metadata"]["noise_scale"] == 0.0
        assert "bayes_accuracy" not in d["metadata"]
        for row in d["train"]:
            n_ones = sum(int(t) for t in row["input"].split())
            z = n_ones - 2
            assert row["output"] == ("1" if z > 0 else "0")

    def test_stochastic_sample_deterministic_per_seed_and_sequence(self):
        kwargs = dict(
            input_type="bits",
            rule="weighted_sum_binary",
            k=4,
            N=16,
            seed=1,
            design="memorization",
            rule_kwargs={
                "weights": [1, 1, 1, 1],
                "intercept": -2,
                "noise_scale": 1.0,
                "weight_seed": 17,
            },
        )
        a = generate(**kwargs)
        b = generate(**kwargs)
        assert [r["output"] for r in a["train"]] == [r["output"] for r in b["train"]]

    def test_stochastic_changes_when_weight_seed_changes(self):
        base = dict(
            input_type="bits",
            rule="weighted_sum_binary",
            k=6,
            N=64,
            seed=1,
            design="memorization",
            rule_kwargs={
                "weights": [1] * 6,
                "intercept": -3,
                "noise_scale": 1.0,
            },
        )
        a = generate(
            **{**base, "rule_kwargs": {**base["rule_kwargs"], "weight_seed": 1}}
        )
        b = generate(
            **{**base, "rule_kwargs": {**base["rule_kwargs"], "weight_seed": 2}}
        )
        a_outs = [r["output"] for r in a["train"]]
        b_outs = [r["output"] for r in b["train"]]
        # At least one disagreement is overwhelmingly likely with σ-noise=1.0.
        assert a_outs != b_outs

    def test_balanced_intercept_produces_balanced_classes(self):
        # Deterministic threshold + balanced intercept on bits → ~50/50.
        # With weights=[1]*8 (sum=8) and balanced intercept (-4), z>0
        # iff #ones > 4, which is 93/256 ≈ 36% (asymmetric because z=0
        # rounds to "0"). A noisier setting gives a closer-to-50 split:
        # with sigmoid noise, classes are exactly E[#ones>4]+0.5*E[#ones=4]
        # = 0.5 by construction.
        d = generate(
            input_type="bits",
            rule="weighted_sum_binary",
            k=8,
            N=256,  # all 2**8 sequences
            seed=1,
            design="memorization",
            rule_kwargs={
                "weights": [1] * 8,
                "intercept": "balanced",
                "noise_scale": 1.0,
                "weight_seed": 0,
            },
        )
        ones = sum(1 for r in d["train"] if r["output"] == "1")
        # σ-Bernoulli sample around z=0 mean → expect ~50/50 within
        # binomial tolerance over 256 draws (3σ ≈ 24).
        assert 100 <= ones <= 156

    def test_bayes_accuracy_present_only_when_stochastic(self):
        d_det = generate(
            input_type="bits",
            rule="weighted_sum_binary",
            k=4,
            N=4,
            seed=1,
            design="memorization",
            rule_kwargs={"weights": [1, 1, 1, 1], "intercept": -2},
        )
        assert "bayes_accuracy" not in d_det["metadata"]

        d_stoch = generate(
            input_type="bits",
            rule="weighted_sum_binary",
            k=4,
            N=4,
            seed=1,
            design="memorization",
            rule_kwargs={
                "weights": [1, 1, 1, 1],
                "intercept": -2,
                "noise_scale": 1.0,
            },
        )
        assert "bayes_accuracy" in d_stoch["metadata"]
        assert 0.5 <= d_stoch["metadata"]["bayes_accuracy"] <= 1.0

    def test_bayes_accuracy_matches_empirical_optimal(self):
        # Enumerate all 2**8 = 256 bit sequences, generate stochastic
        # labels, then compute the empirical accuracy of the
        # Bayes-optimal classifier (predict argmax_y p(y|x)). Should
        # match the recorded bayes_accuracy within statistical tolerance.
        from itertools import product

        d = generate(
            input_type="bits",
            rule="weighted_sum_binary",
            k=8,
            N=256,
            seed=42,
            design="memorization",
            rule_kwargs={
                "weights": [1, -1, 2, 1, -2, 1, 1, -1],
                "intercept": 0,
                "noise_scale": 1.5,
                "weight_seed": 7,
            },
        )
        recorded = d["metadata"]["bayes_accuracy"]

        # Bayes-optimal classifier: predict 1 iff z > 0 (i.e. p > 0.5).
        weights = d["metadata"]["resolved_weights"]
        intercept = d["metadata"]["intercept"]
        labels = {r["input"]: r["output"] for r in d["train"]}
        correct = 0
        for seq in product(("0", "1"), repeat=8):
            inp = " ".join(seq)
            z = sum(w * int(x) for w, x in zip(weights, seq)) + intercept
            pred = "1" if z > 0 else "0"
            if labels[inp] == pred:
                correct += 1
        empirical = correct / 256
        # One stochastic draw per sequence → variance of empirical
        # accuracy is bounded by p(1-p) ≤ 0.25; 256 draws → 3σ ≤ 0.10.
        assert abs(empirical - recorded) < 0.10

    def test_letters_rejected(self):
        with pytest.raises(ValueError, match="not applicable"):
            generate(
                input_type="letters",
                rule="weighted_sum_binary",
                k=3,
                N=4,
                seed=1,
                design="memorization",
                rule_kwargs={"weights": [1, 1, 1]},
            )

    def test_negative_noise_scale_rejected(self):
        with pytest.raises(ValueError, match="noise_scale"):
            generate(
                input_type="bits",
                rule="weighted_sum_binary",
                k=2,
                N=4,
                seed=1,
                design="memorization",
                rule_kwargs={
                    "weights": [1, 1],
                    "intercept": 0,
                    "noise_scale": -0.5,
                },
            )

    def test_invalid_sparsity_rejected(self):
        with pytest.raises(ValueError, match="sparsity"):
            generate(
                input_type="bits",
                rule="weighted_sum_binary",
                k=2,
                N=4,
                seed=1,
                design="memorization",
                rule_kwargs={"weight_max": 2, "sparsity": 1.5},
            )

    def test_invalid_weight_max_rejected(self):
        with pytest.raises(ValueError, match="weight_max"):
            generate(
                input_type="bits",
                rule="weighted_sum_binary",
                k=2,
                N=4,
                seed=1,
                design="memorization",
                rule_kwargs={"weight_max": 0},
            )

    def test_ood_design_rejected_for_weighted_sum_binary(self):
        with pytest.raises(ValueError, match="does not support design='ood'"):
            generate(
                input_type="bits",
                rule="weighted_sum_binary",
                k=6,
                N=64,
                seed=1,
                design="ood",
                split=0.75,
                rule_kwargs={"weight_max": 2, "intercept": 0},
                ood_tests=[{"k": 6, "N": 32}],
            )

    def test_ood_design_rejected_for_weighted_sum(self):
        # Same rejection applies to the non-binary rule. Tested here (not
        # under TestWeightedSum) to keep all OOD-rejection coverage
        # together; the runtime check is generic to any rule with
        # supports_ood=False.
        with pytest.raises(ValueError, match="does not support design='ood'"):
            generate(
                input_type="digits",
                rule="weighted_sum",
                k=4,
                N=20,
                seed=1,
                design="ood",
                split=0.8,
                rule_kwargs={"weights": [1, 2, -1, 1]},
                ood_tests=[{"k": 4, "N": 5}],
            )
