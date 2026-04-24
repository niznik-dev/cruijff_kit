"""Schema loading, validation, and auto-generation.

A schema defines the metadata for each column in a source dataset:
display names, types, synonyms, shorthand mappings, and restatement
templates. Schemas are per-dataset and reusable across experiments.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .readers import read_tabular


def canonicalize_key(value) -> str:
    """Canonicalize a value into a lookup key.

    Numeric-like values (ints, floats, numeric strings including ones with
    leading zeros like "08") are reduced to their integer-string form when
    they have no fractional part: 1.0 -> "1", 8 -> "8", "08" -> "8". Fractional
    floats become their float string. Non-numeric values pass through str().

    This is applied symmetrically to schema value_map keys at load time and
    to row values at lookup time, so that downstream lookups succeed
    regardless of how the source file's dtype represents categorical codes
    (int64, float64, zero-padded string, etc.).
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if f.is_integer():
        return str(int(f))
    return str(f)


@dataclass
class ColumnSchema:
    """Metadata for a single column in the source dataset."""

    key: str
    display_name: str
    type: str  # "numeric" or "categorical"
    unit: str | None = None
    synonyms: list[str] = field(default_factory=list)
    shorthand_map: dict[str, str] = field(default_factory=dict)
    restatements: list[str] = field(default_factory=list)
    value_map: dict[str, str] = field(default_factory=dict)


@dataclass
class Schema:
    """Full schema for a source dataset."""

    name: str
    description: str
    columns: dict[str, ColumnSchema]  # keyed by source column name

    @classmethod
    def from_yaml(cls, path: str) -> "Schema":
        """Load a schema from a YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f)

        dataset = raw.get("dataset", {})
        name = dataset.get("name", Path(path).stem)
        description = dataset.get("description", "")

        columns: dict[str, ColumnSchema] = {}
        for key, col_data in raw.get("columns", {}).items():
            if "display_name" not in col_data:
                raise ValueError(
                    f"Column '{key}' missing required field 'display_name'"
                )
            if "type" not in col_data:
                raise ValueError(f"Column '{key}' missing required field 'type'")
            if col_data["type"] not in ("numeric", "categorical"):
                raise ValueError(
                    f"Column '{key}' has invalid type '{col_data['type']}'. "
                    f"Must be 'numeric' or 'categorical'"
                )
            raw_value_map = col_data.get("value_map", {})
            value_map = {canonicalize_key(k): str(v) for k, v in raw_value_map.items()}

            columns[key] = ColumnSchema(
                key=key,
                display_name=col_data["display_name"],
                type=col_data["type"],
                unit=col_data.get("unit"),
                synonyms=col_data.get("synonyms", []),
                shorthand_map=col_data.get("shorthand_map", {}),
                restatements=col_data.get("restatements", []),
                value_map=value_map,
            )

        return cls(name=name, description=description, columns=columns)

    def get_column(self, key: str) -> ColumnSchema:
        """Look up a column by its source key. Raises KeyError if not found."""
        if key not in self.columns:
            available = ", ".join(sorted(self.columns.keys()))
            raise KeyError(
                f"Column '{key}' not found in schema. Available columns: {available}"
            )
        return self.columns[key]

    @staticmethod
    def generate_skeleton(
        source_path: str,
        columns: list[str],
        sample_rows: int = 10,
    ) -> dict:
        """Read source file headers and sample rows, return a skeleton
        schema dict for the specified columns.

        Infers types (numeric vs categorical) from values. Returns a
        dict suitable for YAML serialization. Used by the skill agent
        to propose a starter schema.
        """
        df = read_tabular(source_path)

        # Validate requested columns exist
        missing = set(columns) - set(df.columns)
        if missing:
            raise ValueError(f"Columns not found in source data: {sorted(missing)}")

        sample = df[columns].head(sample_rows)

        skeleton: dict = {
            "dataset": {
                "name": Path(source_path).stem,
                "description": f"Schema for {Path(source_path).name}",
            },
            "columns": {},
        }

        for col in columns:
            # Infer type from dtype
            if sample[col].dtype.kind in ("i", "f"):
                col_type = "numeric"
            else:
                col_type = "categorical"

            display_name = col.lower().replace("_", " ")
            skeleton["columns"][col] = {
                "display_name": display_name,
                "type": col_type,
                "unit": None,
                "synonyms": [display_name],
                "shorthand_map": {},
                "restatements": [],
            }

        return skeleton
