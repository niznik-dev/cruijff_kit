"""Tests for tabular_to_text_gen/lib/readers.py — tabular file format auto-detection."""

import pytest

from cruijff_kit.tabular_to_text_gen.lib.readers import read_tabular


class TestReadTabular:
    def test_csv(self, sample_csv):
        df = read_tabular(sample_csv)
        assert len(df) == 5
        assert list(df.columns) == ["AGEP", "ST", "OCCP", "PINCP"]

    def test_tsv(self, tmp_path):
        path = tmp_path / "data.tsv"
        path.write_text("col1\tcol2\n1\ta\n2\tb\n")
        df = read_tabular(str(path))
        assert len(df) == 2
        assert list(df.columns) == ["col1", "col2"]

    def test_unsupported_format(self, tmp_path):
        path = tmp_path / "data.xyz"
        path.write_text("garbage")
        with pytest.raises(ValueError, match="Unsupported file format '.xyz'"):
            read_tabular(str(path))

    def test_parquet_roundtrip(self, tmp_path, sample_csv):
        """Write a parquet from CSV then read it back."""
        import pandas as pd

        df_orig = pd.read_csv(sample_csv)
        parquet_path = tmp_path / "data.parquet"
        df_orig.to_parquet(parquet_path, index=False)

        df = read_tabular(str(parquet_path))
        assert len(df) == len(df_orig)
        assert list(df.columns) == list(df_orig.columns)
