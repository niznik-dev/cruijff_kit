"""Tests that the fixture experiment_summary.yaml is structurally valid.

Validates required keys, types, and cross-references (run names match
evaluation matrix entries, variable values match run parameters, etc.).
"""

from pathlib import Path

import pytest
import yaml

FIXTURE_PATH = (
    Path(__file__).parent.parent / "fixtures" / "design" / "experiment_summary.yaml"
)


@pytest.fixture
def summary():
    """Load the fixture experiment_summary.yaml."""
    with open(FIXTURE_PATH) as f:
        return yaml.safe_load(f)


class TestExperimentSection:
    def test_has_required_keys(self, summary):
        required = ["name", "question", "date", "directory"]
        for key in required:
            assert key in summary["experiment"], f"Missing experiment.{key}"

    def test_date_format(self, summary):
        date = summary["experiment"]["date"]
        assert len(date) == 10, "Date should be YYYY-MM-DD"
        assert date[4] == "-" and date[7] == "-"

    def test_name_is_string(self, summary):
        assert isinstance(summary["experiment"]["name"], str)


class TestToolsSection:
    def test_has_required_keys(self, summary):
        assert "preparation" in summary["tools"]
        assert "evaluation" in summary["tools"]

    def test_valid_tool_names(self, summary):
        assert summary["tools"]["preparation"] == "torchtune"
        assert summary["tools"]["evaluation"] == "inspect-ai"


class TestVariablesSection:
    def test_variables_are_lists(self, summary):
        for key, values in summary["variables"].items():
            assert isinstance(values, list), f"variables.{key} should be a list"
            assert len(values) > 0, f"variables.{key} should not be empty"


class TestControlsSection:
    def test_has_required_keys(self, summary):
        required = ["epochs", "batch_size"]
        for key in required:
            assert key in summary["controls"], f"Missing controls.{key}"

    def test_prompt_has_input_placeholder(self, summary):
        prompt = summary["controls"].get("prompt", "")
        assert "{input}" in prompt, "Prompt must contain {input} placeholder"

    def test_epochs_is_positive_int(self, summary):
        assert isinstance(summary["controls"]["epochs"], int)
        assert summary["controls"]["epochs"] > 0


class TestModelsSection:
    def test_has_base_models(self, summary):
        assert "base" in summary["models"]
        assert len(summary["models"]["base"]) > 0

    def test_base_model_has_required_keys(self, summary):
        for model in summary["models"]["base"]:
            assert "name" in model
            assert "path" in model
            assert "size_gb" in model


class TestDataSection:
    def test_has_training_data(self, summary):
        training = summary["data"]["training"]
        required = ["path", "label", "format", "size_kb", "splits"]
        for key in required:
            assert key in training, f"Missing data.training.{key}"

    def test_splits_add_up(self, summary):
        splits = summary["data"]["training"]["splits"]
        assert "train" in splits
        assert "validation" in splits
        assert "test" in splits
        total = splits["train"] + splits["validation"] + splits["test"]
        assert total > 0, "Total samples must be positive"

    def test_format_is_valid(self, summary):
        assert summary["data"]["training"]["format"] == "json"


class TestOutputSection:
    def test_has_required_keys(self, summary):
        required = ["base_directory", "checkpoint_pattern", "wandb_project"]
        for key in required:
            assert key in summary["output"], f"Missing output.{key}"

    def test_checkpoint_pattern_has_placeholders(self, summary):
        pattern = summary["output"]["checkpoint_pattern"]
        assert "{run_name}" in pattern
        assert "{N}" in pattern


class TestRunsSection:
    def test_runs_not_empty(self, summary):
        assert len(summary["runs"]) > 0

    def test_run_has_required_keys(self, summary):
        for run in summary["runs"]:
            assert "name" in run
            assert "type" in run
            assert "model" in run
            assert "parameters" in run

    def test_run_types_valid(self, summary):
        for run in summary["runs"]:
            assert run["type"] in ("fine-tuned", "control")

    def test_run_models_reference_base_models(self, summary):
        base_names = {m["name"] for m in summary["models"]["base"]}
        for run in summary["runs"]:
            assert run["model"] in base_names, (
                f"Run '{run['name']}' references unknown model '{run['model']}'"
            )

    def test_fine_tuned_runs_have_parameters(self, summary):
        for run in summary["runs"]:
            if run["type"] == "fine-tuned":
                assert len(run["parameters"]) > 0, (
                    f"Fine-tuned run '{run['name']}' should have parameters"
                )

    def test_variable_values_match_run_parameters(self, summary):
        """Each variable value should appear in at least one run's parameters."""
        for var_name, var_values in summary["variables"].items():
            for val in var_values:
                found = any(
                    run["parameters"].get(var_name) == val for run in summary["runs"]
                )
                assert found, (
                    f"Variable {var_name}={val} not found in any run's parameters"
                )


class TestEvaluationSection:
    def test_has_required_keys(self, summary):
        required = ["system_prompt", "temperature", "scorer", "tasks", "matrix"]
        for key in required:
            assert key in summary["evaluation"], f"Missing evaluation.{key}"

    def test_tasks_not_empty(self, summary):
        assert len(summary["evaluation"]["tasks"]) > 0

    def test_task_has_required_keys(self, summary):
        for task in summary["evaluation"]["tasks"]:
            assert "name" in task
            assert "script" in task
            assert "description" in task

    def test_matrix_runs_reference_defined_runs(self, summary):
        run_names = {r["name"] for r in summary["runs"]}
        for entry in summary["evaluation"]["matrix"]:
            assert entry["run"] in run_names, (
                f"Matrix entry references unknown run '{entry['run']}'"
            )

    def test_matrix_tasks_reference_defined_tasks(self, summary):
        task_names = {t["name"] for t in summary["evaluation"]["tasks"]}
        for entry in summary["evaluation"]["matrix"]:
            for task in entry["tasks"]:
                assert task in task_names, (
                    f"Matrix entry references unknown task '{task}'"
                )

    def test_matrix_has_epochs(self, summary):
        for entry in summary["evaluation"]["matrix"]:
            assert "epochs" in entry

    def test_scorer_is_list(self, summary):
        assert isinstance(summary["evaluation"]["scorer"], list)
        assert len(summary["evaluation"]["scorer"]) > 0
        for scorer in summary["evaluation"]["scorer"]:
            assert "name" in scorer
