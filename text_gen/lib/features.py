"""Feature selection and filtering.

Selects a subset of columns from a data row based on the condition's
feature list, validates against the schema, and returns ordered
(ColumnSchema, raw_value) pairs for template rendering.
"""

import logging

from .schema import ColumnSchema, Schema

logger = logging.getLogger(__name__)


def validate_features(
    feature_keys: list[str],
    schema: Schema,
    target_column: str | None = None,
) -> list[str]:
    """Validate that all feature keys exist in the schema.

    Warns if target_column is in the feature list.
    Returns list of any warnings.
    """
    warnings = []

    missing = [k for k in feature_keys if k not in schema.columns]
    if missing:
        available = ", ".join(sorted(schema.columns.keys()))
        raise ValueError(
            f"Features not found in schema: {missing}. Available columns: {available}"
        )

    if target_column and target_column in feature_keys:
        msg = (
            f"Target column '{target_column}' is included in the feature "
            f"list. The target value will appear in the generated text."
        )
        warnings.append(msg)
        logger.warning(msg)

    return warnings


def _is_missing(raw_value) -> bool:
    """Return True if raw_value represents a missing/NaN value."""
    if raw_value is None:
        return True
    if isinstance(raw_value, float) and raw_value != raw_value:
        return True
    if isinstance(raw_value, str) and (raw_value.lower() == "nan" or raw_value == ""):
        return True
    return False


def select_features(
    row: dict,
    feature_keys: list[str],
    schema: Schema,
    missing_value_handling: str = "skip",
    missing_value_text: str = "missing",
) -> list[tuple[ColumnSchema, str]]:
    """Extract selected features from a data row.

    Returns a list of (column_schema, raw_value) pairs in the order
    specified by feature_keys.

    Args:
        missing_value_handling: "skip" to omit missing values (default),
            "include" to represent them with missing_value_text.
        missing_value_text: Text to use for missing values when
            missing_value_handling is "include" (default: "missing").
    """
    result = []
    for key in feature_keys:
        col_schema = schema.get_column(key)
        raw_value = row.get(key, "")
        if _is_missing(raw_value):
            if missing_value_handling == "skip":
                continue
            raw_value = missing_value_text
        else:
            raw_value = str(raw_value)
        # Decode via value_map if available
        if col_schema.value_map:
            raw_value = col_schema.value_map.get(raw_value, raw_value)
        result.append((col_schema, raw_value))
    return result
