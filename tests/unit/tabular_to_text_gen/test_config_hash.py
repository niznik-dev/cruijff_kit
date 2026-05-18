"""Tests for tabular_to_text_gen/lib/config_hash.py."""

import hashlib

import pytest

from cruijff_kit.tabular_to_text_gen.lib.config_hash import (
    build_generation_config,
    canonicalize,
    file_sha256,
    hash_config,
    hash_config_short,
    resolve_dataset_path,
)


def _minimal_config() -> dict:
    return {
        "source_sha256": "aaa",
        "schema_sha256": "bbb",
        "features": ["AGEP", "ST"],
        "template": "dictionary",
        "perturbations": [],
        "target": {"column": "PINCP", "threshold": 50000},
        "context": "",
        "context_placement": "preamble",
        "question": "",
        "split_ratio": 0.8,
        "validation_ratio": 0.1,
        "seed": 42,
        "subsampling_ratio": 1.0,
        "missing_value_handling": "skip",
        "one_to_many": None,
        "style_guidance": None,
        "template_file_sha256": None,
    }


class TestDeterminism:
    def test_same_config_same_hash(self):
        c = _minimal_config()
        assert hash_config(c) == hash_config(c)

    def test_short_hash_prefixes_full(self):
        c = _minimal_config()
        assert hash_config(c).startswith(hash_config_short(c))

    def test_short_hash_length(self):
        c = _minimal_config()
        assert len(hash_config_short(c)) == 8
        assert len(hash_config_short(c, n=12)) == 12


class TestDefaultNormalization:
    def test_absent_subsampling_ratio_equals_default(self):
        c1 = _minimal_config()
        c2 = {k: v for k, v in c1.items() if k != "subsampling_ratio"}
        assert hash_config(c1) == hash_config(c2)

    def test_absent_missing_value_handling_equals_skip(self):
        c1 = _minimal_config()
        c2 = {k: v for k, v in c1.items() if k != "missing_value_handling"}
        assert hash_config(c1) == hash_config(c2)

    def test_absent_perturbations_equals_empty_list(self):
        c1 = _minimal_config()
        c2 = {k: v for k, v in c1.items() if k != "perturbations"}
        assert hash_config(c1) == hash_config(c2)

    def test_missing_value_text_ignored_when_skip(self):
        c1 = _minimal_config()
        c2 = dict(c1, missing_value_text="foo")
        assert hash_config(c1) == hash_config(c2)

    def test_missing_value_text_hashed_when_include(self):
        c1 = dict(_minimal_config(), missing_value_handling="include")
        c2 = dict(c1, missing_value_text="unknown")
        c3 = dict(c1, missing_value_text="n/a")
        assert hash_config(c2) != hash_config(c3)

    def test_style_guidance_ignored_for_dictionary(self):
        c1 = _minimal_config()
        c2 = dict(c1, style_guidance="be concise")
        assert hash_config(c1) == hash_config(c2)

    def test_style_guidance_hashed_for_llm_narrative(self):
        c1 = dict(_minimal_config(), template="llm_narrative")
        c2 = dict(c1, style_guidance="be concise")
        assert hash_config(c1) != hash_config(c2)


class TestFieldSensitivity:
    def test_feature_order_matters(self):
        c1 = dict(_minimal_config(), features=["AGEP", "ST"])
        c2 = dict(c1, features=["ST", "AGEP"])
        assert hash_config(c1) != hash_config(c2)

    def test_seed_matters(self):
        c1 = _minimal_config()
        c2 = dict(c1, seed=99)
        assert hash_config(c1) != hash_config(c2)

    def test_source_sha_matters(self):
        c1 = _minimal_config()
        c2 = dict(c1, source_sha256="different")
        assert hash_config(c1) != hash_config(c2)

    def test_schema_sha_matters(self):
        c1 = _minimal_config()
        c2 = dict(c1, schema_sha256="different")
        assert hash_config(c1) != hash_config(c2)

    def test_template_matters(self):
        c1 = _minimal_config()
        c2 = dict(c1, template="narrative")
        assert hash_config(c1) != hash_config(c2)


class TestCanonicalize:
    def test_preserves_feature_order(self):
        c = dict(_minimal_config(), features=["Z", "A", "M"])
        assert canonicalize(c)["features"] == ["Z", "A", "M"]

    def test_fills_defaults(self):
        c = {
            "source_sha256": "a",
            "schema_sha256": "b",
            "features": ["X"],
            "template": "dictionary",
            "target": {"column": "y"},
            "split_ratio": 0.8,
            "validation_ratio": None,
            "seed": 1,
        }
        out = canonicalize(c)
        assert out["subsampling_ratio"] == 1.0
        assert out["missing_value_handling"] == "skip"
        assert out["perturbations"] == []


class TestFileSha256:
    def test_matches_hashlib(self, tmp_path):
        path = tmp_path / "f.txt"
        path.write_bytes(b"hello world")
        assert file_sha256(path) == hashlib.sha256(b"hello world").hexdigest()

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            file_sha256(tmp_path / "nope.txt")


class TestThresholdNormalization:
    """Regression: CLI argparse gives threshold=50000.0 (float), YAML parses
    50000 as int. Both must hash identically — previously they did not, which
    caused convert.py and build_generation_config to produce divergent
    filenames for semantically identical configs."""

    def test_int_and_float_threshold_same_hash(self):
        c_int = _minimal_config()
        c_int["target"] = {"column": "PINCP", "threshold": 50000}
        c_float = _minimal_config()
        c_float["target"] = {"column": "PINCP", "threshold": 50000.0}
        assert hash_config(c_int) == hash_config(c_float)

    def test_mapping_target_does_not_regress(self):
        # Categorical target has no threshold; canonicalization should not
        # inject one or otherwise mutate the mapping.
        c = _minimal_config()
        c["target"] = {"column": "ESR", "mapping": {"1": "employed", "2": "unemployed"}}
        assert canonicalize(c)["target"] == {
            "column": "ESR",
            "mapping": {"1": "employed", "2": "unemployed"},
        }


class TestBuildGenerationConfig:
    """Unit tests for the YAML-→-canonical-config helper. These tests use
    on-disk files only for the source/schema SHAs (the helper takes file paths
    from the data_generation block and hashes their content)."""

    @pytest.fixture
    def src_and_schema(self, tmp_path):
        src = tmp_path / "src.csv"
        src.write_text("AGEP,SEX,PINCP\n30,1,60000\n")
        schema = tmp_path / "schema.yaml"
        schema.write_text("columns: {AGEP: {}, SEX: {}, PINCP: {}}\n")
        return str(src), str(schema)

    def _dg(self, src, schema, **overrides):
        """Minimal data_generation block with two conditions."""
        dg = {
            "source": src,
            "schema": schema,
            "target": {"column": "PINCP", "threshold": 50000},
            "context": "ctx",
            "context_placement": "system_prompt",
            "question": "Is income above $50,000?",
            "split_ratio": 0.5,
            "validation_ratio": 0.1,
            "seed": 42,
            "subsampling_ratio": 0.1,
            "missing_value_handling": "skip",
            "conditions": {
                "subset": {
                    "features": ["AGEP", "SEX"],
                    "template": "dictionary",
                    "perturbations": [],
                },
                "full": {
                    "features": ["AGEP", "SEX", "PINCP"],
                    "template": "dictionary",
                    "perturbations": [],
                },
            },
        }
        dg.update(overrides)
        return dg

    def test_picks_right_condition(self, src_and_schema):
        src, schema = src_and_schema
        dg = self._dg(src, schema)
        cfg = build_generation_config(dg, "subset")
        assert cfg["features"] == ["AGEP", "SEX"]
        assert cfg["template"] == "dictionary"

    def test_propagates_shared_fields(self, src_and_schema):
        src, schema = src_and_schema
        dg = self._dg(src, schema)
        cfg = build_generation_config(dg, "subset")
        assert cfg["seed"] == 42
        assert cfg["split_ratio"] == 0.5
        assert cfg["validation_ratio"] == 0.1
        assert cfg["context_placement"] == "system_prompt"
        assert cfg["question"] == "Is income above $50,000?"

    def test_missing_condition_raises_keyerror_with_available(self, src_and_schema):
        src, schema = src_and_schema
        dg = self._dg(src, schema)
        with pytest.raises(KeyError, match="not found.*Available.*full.*subset"):
            build_generation_config(dg, "does_not_exist")

    def test_threshold_preserved_as_given_but_canonicalizes_to_float(
        self, src_and_schema
    ):
        src, schema = src_and_schema
        dg = self._dg(src, schema)
        cfg_int = build_generation_config(dg, "subset")
        # raw dict still has the int as the user wrote it
        assert cfg_int["target"]["threshold"] == 50000
        # but canonicalize coerces to float for hashing
        assert canonicalize(cfg_int)["target"]["threshold"] == 50000.0

    def test_mapping_target_produces_mapping_block(self, src_and_schema):
        src, schema = src_and_schema
        dg = self._dg(
            src,
            schema,
            target={"column": "ESR", "mapping": {"1": "employed", "2": "unemployed"}},
        )
        cfg = build_generation_config(dg, "subset")
        assert "threshold" not in cfg["target"]
        assert cfg["target"]["mapping"] == {"1": "employed", "2": "unemployed"}

    def test_template_file_sha_included_when_path_set(self, src_and_schema, tmp_path):
        src, schema = src_and_schema
        tmpl = tmp_path / "narr.j2"
        tmpl.write_text("{{ features }}")
        dg = self._dg(src, schema)
        dg["conditions"]["subset"]["template"] = "narrative"
        dg["conditions"]["subset"]["template_file"] = str(tmpl)
        cfg = build_generation_config(dg, "subset")
        assert cfg["template_file_sha256"] == file_sha256(str(tmpl))

    def test_one_to_many_propagates(self, src_and_schema):
        src, schema = src_and_schema
        dg = self._dg(src, schema)
        dg["conditions"]["subset"]["one_to_many"] = {
            "copies": 3,
            "perturbation": "reorder",
        }
        cfg = build_generation_config(dg, "subset")
        assert cfg["one_to_many"] == {"copies": 3, "perturbation": "reorder"}

    def test_condition_isolation(self, src_and_schema):
        """Different conditions in the same dg block must hash differently
        (they have different feature lists)."""
        src, schema = src_and_schema
        dg = self._dg(src, schema)
        h_subset = hash_config_short(build_generation_config(dg, "subset"))
        h_full = hash_config_short(build_generation_config(dg, "full"))
        assert h_subset != h_full


class TestResolveDatasetPath:
    @pytest.fixture
    def dg(self, tmp_path):
        src = tmp_path / "src.csv"
        src.write_text("AGEP,SEX,PINCP\n30,1,60000\n")
        schema = tmp_path / "schema.yaml"
        schema.write_text("columns: {AGEP: {}, SEX: {}, PINCP: {}}\n")
        return {
            "source": str(src),
            "schema": str(schema),
            "target": {"column": "PINCP", "threshold": 50000},
            "seed": 42,
            "conditions": {
                "subset": {
                    "features": ["AGEP", "SEX"],
                    "template": "dictionary",
                    "perturbations": [],
                },
            },
        }

    def test_filename_format(self, dg, tmp_path):
        path = resolve_dataset_path(dg, "subset", "train", tmp_path)
        fname = path.rsplit("/", 1)[-1]
        assert fname.startswith("subset_train_")
        assert fname.endswith(".json")
        # hash8 between "subset_train_" and ".json" is 8 hex chars
        middle = fname[len("subset_train_") : -len(".json")]
        assert len(middle) == 8
        assert all(c in "0123456789abcdef" for c in middle)

    def test_train_and_test_share_hash(self, dg, tmp_path):
        """split is deliberately NOT in the hash — train/test of the same
        logical config differ only by filename suffix."""
        train = resolve_dataset_path(dg, "subset", "train", tmp_path)
        test = resolve_dataset_path(dg, "subset", "test", tmp_path)
        h_train = train.rsplit("_", 1)[-1].replace(".json", "")
        h_test = test.rsplit("_", 1)[-1].replace(".json", "")
        assert h_train == h_test

    def test_matches_direct_hash(self, dg, tmp_path):
        """The resolver must agree with the direct
        hash_config_short(build_generation_config(...)) path; otherwise
        convert.py and the scaffold agents would disagree on filenames."""
        expected_hash = hash_config_short(build_generation_config(dg, "subset"))
        path = resolve_dataset_path(dg, "subset", "train", tmp_path)
        assert path.endswith(f"subset_train_{expected_hash}.json")

    def test_invalid_split_raises(self, dg, tmp_path):
        with pytest.raises(ValueError, match="split must be one of"):
            resolve_dataset_path(dg, "subset", "bogus", tmp_path)
