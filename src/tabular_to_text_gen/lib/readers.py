"""Tabular file format readers with auto-detection from extension."""

import pandas as pd
from pathlib import Path

READERS = {
    ".csv": pd.read_csv,
    ".tsv": lambda p: pd.read_csv(p, sep="\t"),
    ".dta": pd.read_stata,
    ".parquet": pd.read_parquet,
    ".xlsx": pd.read_excel,
    ".xls": pd.read_excel,
    ".sas7bdat": pd.read_sas,
}


def read_tabular(source_path: str) -> pd.DataFrame:
    """Read a tabular file into a DataFrame, auto-detecting format
    from the file extension.

    Supported formats: .csv, .tsv, .dta (Stata), .parquet, .xlsx,
    .xls, .sas7bdat

    Raises ValueError for unsupported extensions.
    """
    ext = Path(source_path).suffix.lower()
    if ext not in READERS:
        supported = ", ".join(sorted(READERS.keys()))
        raise ValueError(
            f"Unsupported file format '{ext}'. Supported formats: {supported}"
        )
    return READERS[ext](source_path)
