"""Shared fixtures for text_gen tests."""

import csv

import pytest
import yaml

from text_gen.lib.schema import ColumnSchema, Schema
from text_gen.lib.segments import Segment


# ---------------------------------------------------------------------------
# Schema fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def column_age():
    return ColumnSchema(
        key="AGEP",
        display_name="age",
        type="numeric",
        unit="years old",
        synonyms=["age", "years of age"],
        shorthand_map={},
        restatements=[
            "This person is {value} years old",
            "The respondent is in their {decade}s",
        ],
    )


@pytest.fixture
def column_state():
    return ColumnSchema(
        key="ST",
        display_name="state",
        type="categorical",
        unit=None,
        synonyms=["state", "state of residence", "home state"],
        shorthand_map={"New York": "NY", "California": "CA"},
        restatements=["This person lives in {value}"],
    )


@pytest.fixture
def column_occupation():
    return ColumnSchema(
        key="OCCP",
        display_name="occupation",
        type="categorical",
        unit=None,
        synonyms=["occupation", "job", "profession"],
        shorthand_map={},
        restatements=["This person works as a {value}"],
    )


@pytest.fixture
def column_income():
    return ColumnSchema(
        key="PINCP",
        display_name="income",
        type="numeric",
        unit="dollars",
        synonyms=["income", "earnings"],
        shorthand_map={},
        restatements=[],
    )


@pytest.fixture
def schema(column_age, column_state, column_occupation, column_income):
    return Schema(
        name="Test",
        description="Test dataset",
        columns={
            "AGEP": column_age,
            "ST": column_state,
            "OCCP": column_occupation,
            "PINCP": column_income,
        },
    )


# ---------------------------------------------------------------------------
# Segment fixtures
# ---------------------------------------------------------------------------


def _make_segment(
    field="AGEP",
    display_name="age",
    value="51 years old",
    text="The age is: 51 years old.",
    metadata=None,
    is_added=False,
):
    """Helper to create a Segment with sensible defaults."""
    if metadata is None:
        metadata = {
            "type": "numeric",
            "unit": "years old",
            "synonyms": ["age", "years of age"],
            "shorthand_map": {},
            "restatements": ["This person is {value} years old"],
        }
    return Segment(
        field=field,
        display_name=display_name,
        value=value,
        text=text,
        metadata=metadata,
        is_added=is_added,
    )


@pytest.fixture
def make_segment():
    """Expose _make_segment as a pytest fixture."""
    return _make_segment


@pytest.fixture
def sample_segments():
    """Three segments representing age, state, and occupation."""
    return [
        _make_segment(
            field="AGEP",
            display_name="age",
            value="51 years old",
            text="The age is: 51 years old.",
            metadata={
                "type": "numeric",
                "unit": "years old",
                "synonyms": ["age", "years of age"],
                "shorthand_map": {},
                "restatements": [
                    "This person is {value} years old",
                    "The respondent is in their {decade}s",
                ],
            },
        ),
        _make_segment(
            field="ST",
            display_name="state",
            value="New York",
            text="The state is: New York.",
            metadata={
                "type": "categorical",
                "unit": None,
                "synonyms": ["state", "state of residence", "home state"],
                "shorthand_map": {"New York": "NY", "California": "CA"},
                "restatements": ["This person lives in {value}"],
            },
        ),
        _make_segment(
            field="OCCP",
            display_name="occupation",
            value="Teacher",
            text="The occupation is: Teacher.",
            metadata={
                "type": "categorical",
                "unit": None,
                "synonyms": ["occupation", "job", "profession"],
                "shorthand_map": {},
                "restatements": ["This person works as a {value}"],
            },
        ),
    ]


# ---------------------------------------------------------------------------
# File-based fixtures (tmp_path scoped)
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_csv(tmp_path):
    """Write a small CSV and return its path."""
    rows = [
        {"AGEP": "51", "ST": "New York", "OCCP": "Teacher", "PINCP": "75000"},
        {"AGEP": "29", "ST": "California", "OCCP": "Nurse", "PINCP": "42000"},
        {"AGEP": "63", "ST": "New York", "OCCP": "Engineer", "PINCP": "95000"},
        {"AGEP": "35", "ST": "Texas", "OCCP": "Researcher", "PINCP": "55000"},
        {"AGEP": "44", "ST": "California", "OCCP": "Electrician", "PINCP": "48000"},
    ]
    path = tmp_path / "test_data.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["AGEP", "ST", "OCCP", "PINCP"])
        w.writeheader()
        w.writerows(rows)
    return str(path)


@pytest.fixture
def schema_yaml(tmp_path):
    """Write a schema YAML file and return its path."""
    schema_dict = {
        "dataset": {"name": "Test", "description": "Test dataset"},
        "columns": {
            "AGEP": {
                "display_name": "age",
                "type": "numeric",
                "unit": "years old",
                "synonyms": ["age", "years of age"],
                "shorthand_map": {},
                "restatements": ["This person is {value} years old"],
            },
            "ST": {
                "display_name": "state",
                "type": "categorical",
                "synonyms": ["state", "state of residence", "home state"],
                "shorthand_map": {"New York": "NY", "California": "CA"},
                "restatements": ["This person lives in {value}"],
            },
            "OCCP": {
                "display_name": "occupation",
                "type": "categorical",
                "synonyms": ["occupation", "job", "profession"],
                "shorthand_map": {},
                "restatements": ["This person works as a {value}"],
            },
            "PINCP": {
                "display_name": "income",
                "type": "numeric",
                "unit": "dollars",
                "synonyms": ["income", "earnings"],
                "shorthand_map": {},
                "restatements": [],
            },
        },
    }
    path = tmp_path / "schema.yaml"
    with open(path, "w") as f:
        yaml.dump(schema_dict, f)
    return str(path)


@pytest.fixture
def narrative_template(tmp_path):
    """Write the default narrative Jinja2 template and return its path."""
    j2 = tmp_path / "default_narrative.j2"
    j2.write_text(
        "{# default_narrative.j2 — Generic narrative template for any dataset.\n"
        "\n"
        "   Receives a list of features, each with: field, display_name, value, type, unit.\n"
        "   Produces one sentence per feature with natural phrasing.\n"
        "#}\n"
        "{% for feat in features -%}\n"
        "{% if loop.first -%}\n"
        "The respondent's {{ feat.display_name }} is {{ feat.value }}.\n"
        "{%- else %} Their {{ feat.display_name }} is {{ feat.value }}.\n"
        "{%- endif %}\n"
        "{%- endfor %}\n"
    )
    return str(j2)


@pytest.fixture
def conditions_yaml(tmp_path):
    """Write a conditions YAML file and return its path."""
    conditions = {
        "conditions": {
            "dict_full": {
                "features": ["AGEP", "ST", "OCCP"],
                "template": "dictionary",
                "perturbations": [],
            },
            "dict_synonym": {
                "features": ["AGEP", "ST", "OCCP"],
                "template": "dictionary",
                "perturbations": ["synonym"],
            },
        },
    }
    path = tmp_path / "conditions.yaml"
    with open(path, "w") as f:
        yaml.dump(conditions, f)
    return str(path)
