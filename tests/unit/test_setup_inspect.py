"""Unit tests for tools/inspect/setup_inspect.py"""

import re
import sys
import textwrap

import pytest
import yaml

from cruijff_kit.tools.inspect.setup_inspect import (
    _format_value,
    build_task_args,
    build_metadata_args,
    render_template,
    load_eval_config,
    create_parser,
    main,
    TASK_ARG_KEYS,
    METADATA_ARG_KEYS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_cli_args(**overrides):
    """Create a namespace mimicking parsed CLI args."""
    defaults = dict(
        config="eval_config.yaml",
        model_name="Llama-3.2-1B-Instruct",
        time="0:10:00",
        account=None,
        mem=None,
        partition=None,
        constraint=None,
        conda_env="cruijff",
        output_slurm=None,
    )
    defaults.update(overrides)
    return type("Args", (), defaults)()


def make_config(**overrides):
    """Create a config dict mimicking load_eval_config output.

    Only includes the required keys plus eval_dir (auto-derived).
    Add optional keys (data_path, vis_label, epoch, etc.) via overrides.
    """
    defaults = dict(
        task_script="/path/to/task.py@my_task",
        task_name="my_task",
        model_path="/outputs/run1/epoch_0",
        model_hf_name="hf/run1_epoch_0",
        output_dir="/outputs/run1/",
        eval_dir="/experiments/run1/eval",
    )
    defaults.update(overrides)
    return defaults


MINIMAL_EVAL_CONFIG = textwrap.dedent("""\
    task_script: /path/to/task.py@my_task
    task_name: my_task
    model_path: /outputs/run1/epoch_0
    model_hf_name: hf/run1_epoch_0
    output_dir: /outputs/run1/
""")


FULL_EVAL_CONFIG = textwrap.dedent("""\
    task_script: /path/to/task.py@acs_income
    task_name: acs_income
    model_path: /outputs/run1/epoch_0
    model_hf_name: hf/1B_ft_epoch_0
    output_dir: /outputs/run1/
    data_path: /data/acs_income.json
    vis_label: 1B_ft
    use_chat_template: "true"
    epoch: 0
    finetuned: true
    source_model: Llama-3.2-1B-Instruct
    scorer:
      - name: match
      - name: risk_scorer
        params:
          option_tokens: ["0", "1"]
""")


# ---------------------------------------------------------------------------
# load_eval_config
# ---------------------------------------------------------------------------

class TestLoadEvalConfig:

    def test_loads_required_keys(self, tmp_path):
        """Config with required keys loads successfully."""
        config_file = tmp_path / "eval_config.yaml"
        config_file.write_text(MINIMAL_EVAL_CONFIG)

        config = load_eval_config(str(config_file))
        assert config["task_script"] == "/path/to/task.py@my_task"
        assert config["task_name"] == "my_task"
        assert config["model_path"] == "/outputs/run1/epoch_0"
        assert config["model_hf_name"] == "hf/run1_epoch_0"
        assert config["output_dir"] == "/outputs/run1/"

    def test_auto_derives_eval_dir(self, tmp_path):
        """eval_dir is auto-derived from config file location."""
        config_file = tmp_path / "eval_config.yaml"
        config_file.write_text(MINIMAL_EVAL_CONFIG)

        config = load_eval_config(str(config_file))
        assert config["eval_dir"] == str(tmp_path)

    def test_auto_derives_config_path(self, tmp_path):
        """config_path is auto-derived as absolute path to config file."""
        config_file = tmp_path / "eval_config.yaml"
        config_file.write_text(MINIMAL_EVAL_CONFIG)

        config = load_eval_config(str(config_file))
        assert config["config_path"] == str(config_file)

    def test_missing_required_key_raises(self, tmp_path):
        """Missing required key raises ValueError."""
        config_file = tmp_path / "eval_config.yaml"
        config_file.write_text("task_script: /path/to/task.py\n")

        with pytest.raises(ValueError, match="missing required keys"):
            load_eval_config(str(config_file))

    def test_file_not_found_raises(self):
        """Non-existent config file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_eval_config("/nonexistent/eval_config.yaml")

    def test_full_config_with_optional_keys(self, tmp_path):
        """Full config with optional keys loads correctly."""
        config_file = tmp_path / "eval_config.yaml"
        config_file.write_text(FULL_EVAL_CONFIG)

        config = load_eval_config(str(config_file))
        assert config["vis_label"] == "1B_ft"
        assert config["epoch"] == 0
        assert config["finetuned"] is True
        assert config["source_model"] == "Llama-3.2-1B-Instruct"
        # Scorer config is loaded but not used by setup_inspect
        assert len(config["scorer"]) == 2

    def test_extra_keys_preserved(self, tmp_path):
        """Unknown keys in config are preserved (forward compatibility)."""
        config_file = tmp_path / "eval_config.yaml"
        config_file.write_text(MINIMAL_EVAL_CONFIG + "custom_field: hello\n")

        config = load_eval_config(str(config_file))
        assert config["custom_field"] == "hello"

    def test_empty_yaml_raises(self, tmp_path):
        """Empty YAML file raises ValueError (missing required keys)."""
        config_file = tmp_path / "eval_config.yaml"
        config_file.write_text("")

        with pytest.raises(ValueError, match="missing required keys"):
            load_eval_config(str(config_file))


# ---------------------------------------------------------------------------
# _format_value (boolean normalization)
# ---------------------------------------------------------------------------

class TestFormatValue:

    def test_bool_true_becomes_lowercase(self):
        """Python True (from YAML `true`) renders as lowercase 'true'."""
        assert _format_value(True) == "true"

    def test_bool_false_becomes_lowercase(self):
        """Python False (from YAML `false`) renders as lowercase 'false'."""
        assert _format_value(False) == "false"

    def test_string_passthrough(self):
        """String values pass through unchanged."""
        assert _format_value("hello") == "hello"

    def test_int_passthrough(self):
        """Integer values are stringified."""
        assert _format_value(0) == "0"
        assert _format_value(42) == "42"


# ---------------------------------------------------------------------------
# build_task_args
# ---------------------------------------------------------------------------

class TestBuildTaskArgs:

    def test_no_task_args(self):
        """Empty string when all task params are absent."""
        config = make_config()
        assert build_task_args(config) == ""

    def test_all_task_args(self):
        """All four task params produce -T lines."""
        config = make_config(
            data_path="/data/test.json",
            config_path="/config/eval.yaml",
            vis_label="1B_ft",
            use_chat_template="true",
        )
        result = build_task_args(config)
        assert '-T data_path="/data/test.json"' in result
        assert '-T config_path="/config/eval.yaml"' in result
        assert '-T vis_label="1B_ft"' in result
        assert '-T use_chat_template="true"' in result

    def test_partial_task_args(self):
        """Only specified params appear."""
        config = make_config(data_path="/data/test.json", vis_label="label")
        result = build_task_args(config)
        assert "-T data_path=" in result
        assert "-T vis_label=" in result
        assert "use_chat_template" not in result

    def test_boolean_value_lowercased(self):
        """YAML boolean True renders as lowercase 'true' in -T args."""
        config = make_config(use_chat_template=True)
        result = build_task_args(config)
        assert '-T use_chat_template="true"' in result


# ---------------------------------------------------------------------------
# build_metadata_args
# ---------------------------------------------------------------------------

class TestBuildMetadataArgs:

    def test_no_metadata_args(self):
        """Empty string when all metadata params are absent."""
        config = make_config()
        assert build_metadata_args(config) == ""

    def test_all_metadata_args(self):
        """All three metadata params produce --metadata lines."""
        config = make_config(epoch=0, finetuned="true", source_model="Llama-3.2-1B-Instruct")
        result = build_metadata_args(config)
        assert '--metadata epoch="0"' in result
        assert '--metadata finetuned="true"' in result
        assert '--metadata source_model="Llama-3.2-1B-Instruct"' in result

    def test_partial_metadata_args(self):
        """Only specified metadata params appear."""
        config = make_config(epoch=2)
        result = build_metadata_args(config)
        assert '--metadata epoch="2"' in result
        assert "finetuned" not in result
        assert "source_model" not in result

    def test_boolean_finetuned_lowercased(self):
        """YAML boolean finetuned: true renders as lowercase 'true'."""
        config = make_config(finetuned=True)
        result = build_metadata_args(config)
        assert '--metadata finetuned="true"' in result


# ---------------------------------------------------------------------------
# render_template
# ---------------------------------------------------------------------------

class TestRenderTemplate:

    def test_no_angle_bracket_placeholders_remain(self):
        """Rendered script has no remaining <PLACEHOLDER> tokens in active lines.

        Inactive ##SBATCH lines may retain placeholders (e.g. <PART>) — that's fine.
        """
        cli = make_cli_args(account="myaccount")
        config = make_config(
            data_path="/data/test.json",
            config_path="/config/eval.yaml",
            vis_label="test",
            use_chat_template="true",
            epoch=0,
            finetuned="true",
            source_model="Llama-3.2-1B-Instruct",
        )
        script = render_template(cli, config)
        # Filter out inactive ##SBATCH lines before checking
        active_lines = [
            line for line in script.splitlines()
            if not line.startswith("##SBATCH")
        ]
        remaining = re.findall(r"<[A-Z_]+>", "\n".join(active_lines))
        assert remaining == [], f"Unresolved placeholders: {remaining}"

    def test_gpu_monitoring_present(self):
        """GPU monitoring commands appear in rendered output."""
        script = render_template(make_cli_args(), make_config())
        assert "nvidia-smi" in script
        assert "GPU_MONITOR_PID" in script
        assert "gpu_metrics.csv" in script
        assert "kill $GPU_MONITOR_PID" in script

    def test_inspect_eval_command(self):
        """inspect eval command uses correct task script and model."""
        config = make_config(
            task_script="/path/to/task.py@my_task",
            model_hf_name="hf/run1_epoch_0",
            model_path="/outputs/run1/epoch_0",
        )
        script = render_template(make_cli_args(), config)
        assert "inspect eval /path/to/task.py@my_task" in script
        assert "--model hf/run1_epoch_0" in script
        assert 'model_path="/outputs/run1/epoch_0"' in script

    def test_eval_dir_cd(self):
        """Script changes to eval_dir before running inspect."""
        config = make_config(eval_dir="/experiments/test/eval")
        script = render_template(make_cli_args(), config)
        assert "cd /experiments/test/eval" in script

    def test_output_dir_trailing_slash(self):
        """Output dir gets a trailing slash if missing."""
        config = make_config(output_dir="/outputs/run1")
        script = render_template(make_cli_args(), config)
        assert "/outputs/run1/" in script

    def test_output_dir_no_double_slash(self):
        """Output dir with trailing slash doesn't get doubled."""
        config = make_config(output_dir="/outputs/run1/")
        script = render_template(make_cli_args(), config)
        assert "/outputs/run1//" not in script

    def test_inspect_exit_code(self):
        """Script captures INSPECT_EXIT_CODE."""
        script = render_template(make_cli_args(), make_config())
        assert "INSPECT_EXIT_CODE=$?" in script
        assert "INSPECT_EXIT_CODE == 0" in script

    def test_gpu_metrics_epoch_specific_dir(self):
        """GPU metrics write to epoch-specific subdir when epoch is set."""
        config = make_config(output_dir="/outputs/run1/", epoch=2)
        script = render_template(make_cli_args(), config)
        assert 'GPU_METRICS_DIR="/outputs/run1/epoch_2"' in script
        assert "mkdir -p" in script

    def test_gpu_metrics_no_epoch_uses_output_dir(self):
        """GPU metrics write to output_dir when no epoch is set."""
        config = make_config(output_dir="/outputs/run1/")
        script = render_template(make_cli_args(), config)
        assert 'GPU_METRICS_DIR="/outputs/run1"' in script

    def test_slurm_log_move(self):
        """SLURM log is moved on success."""
        config = make_config(output_dir="/outputs/run1/")
        script = render_template(make_cli_args(), config)
        assert "mv slurm-${SLURM_JOB_ID}.out /outputs/run1/" in script
        assert "INSPECT_EXIT_CODE == 0" in script

    def test_time_override(self):
        """Custom time limit appears in SLURM header."""
        cli = make_cli_args(time="0:30:00")
        script = render_template(cli, make_config())
        assert "#SBATCH --time=0:30:00" in script

    def test_jobname_with_epoch(self):
        """Job name includes epoch when provided."""
        config = make_config(task_name="acs_income", epoch=2)
        script = render_template(make_cli_args(), config)
        assert "#SBATCH --job-name=eval-acs_income-ep2" in script

    def test_jobname_without_epoch(self):
        """Job name omits epoch suffix when not provided."""
        config = make_config(task_name="acs_income")
        script = render_template(make_cli_args(), config)
        assert "#SBATCH --job-name=eval-acs_income" in script

    def test_conda_env(self):
        """Custom conda env is used."""
        cli = make_cli_args(conda_env="myenv")
        script = render_template(cli, make_config())
        assert "conda activate myenv" in script

    def test_task_and_metadata_args_together(self):
        """Both task args and metadata args render correctly in the template."""
        config = make_config(
            data_path="/data/test.json",
            vis_label="1B_ft",
            epoch=0,
            finetuned=True,
        )
        script = render_template(make_cli_args(), config)
        # Both blocks present
        assert '-T data_path="/data/test.json"' in script
        assert '-T vis_label="1B_ft"' in script
        assert '--metadata epoch="0"' in script
        assert '--metadata finetuned="true"' in script
        # Proper ordering: task args before metadata, both before --log-dir
        task_pos = script.index("-T data_path=")
        meta_pos = script.index("--metadata epoch=")
        logdir_pos = script.index("--log-dir")
        assert task_pos < meta_pos < logdir_pos


# ---------------------------------------------------------------------------
# ##SBATCH activation
# ---------------------------------------------------------------------------

class TestSbatchActivation:

    def test_account_activated(self):
        """Account line activated when --account provided."""
        cli = make_cli_args(account="msalganik")
        script = render_template(cli, make_config())
        assert "#SBATCH --account=msalganik" in script
        assert "##SBATCH --account" not in script

    def test_account_not_activated_when_none(self):
        """Account line stays commented when --account not provided."""
        cli = make_cli_args(account=None)
        script = render_template(cli, make_config())
        assert "##SBATCH --account=<ACT>" in script

    def test_constraint_from_model_config(self):
        """Constraint activated from model_configs (1B defaults to gpu80)."""
        cli = make_cli_args(model_name="Llama-3.2-1B-Instruct")
        script = render_template(cli, make_config())
        assert "#SBATCH --constraint=gpu80" in script
        assert "##SBATCH --constraint" not in script

    def test_constraint_cli_override(self):
        """CLI constraint overrides model_configs."""
        cli = make_cli_args(constraint="a100")
        script = render_template(cli, make_config())
        assert "#SBATCH --constraint=a100" in script

    def test_partition_stays_commented_when_none(self):
        """Partition stays commented when model_config has None."""
        cli = make_cli_args(model_name="Llama-3.2-1B-Instruct", partition=None)
        script = render_template(cli, make_config())
        assert "##SBATCH --partition=<PART>" in script

    def test_partition_cli_override(self):
        """CLI partition activates the line."""
        cli = make_cli_args(partition="nomig")
        script = render_template(cli, make_config())
        assert "#SBATCH --partition=nomig" in script
        assert "##SBATCH --partition" not in script


# ---------------------------------------------------------------------------
# Model config lookup
# ---------------------------------------------------------------------------

class TestModelConfigLookup:

    def test_1b_instruct_mem(self):
        """1B-Instruct gets 80G memory from model_configs."""
        cli = make_cli_args(model_name="Llama-3.2-1B-Instruct")
        script = render_template(cli, make_config())
        assert "#SBATCH --mem=80G" in script

    def test_mem_cli_override(self):
        """CLI --mem overrides model_configs default."""
        cli = make_cli_args(model_name="Llama-3.2-1B-Instruct", mem="32G")
        script = render_template(cli, make_config())
        assert "#SBATCH --mem=32G" in script

    def test_1b_instruct_cpus(self):
        """1B-Instruct gets 1 CPU from model_configs."""
        cli = make_cli_args(model_name="Llama-3.2-1B-Instruct")
        script = render_template(cli, make_config())
        assert "#SBATCH --cpus-per-task=1" in script

    def test_70b_gets_high_resources(self):
        """70B model gets 320G mem and 4 CPUs."""
        cli = make_cli_args(model_name="Llama-3.3-70B-Instruct")
        script = render_template(cli, make_config())
        assert "#SBATCH --mem=320G" in script
        assert "#SBATCH --cpus-per-task=4" in script

    def test_unknown_model_raises(self):
        """Unknown model name raises ValueError."""
        cli = make_cli_args(model_name="NonExistentModel")
        with pytest.raises(ValueError, match="Unknown model"):
            render_template(cli, make_config())


# ---------------------------------------------------------------------------
# create_parser
# ---------------------------------------------------------------------------

class TestCreateParser:

    def test_required_args(self):
        """Parser requires --config and --model_name."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_minimal_args_parse(self):
        """Parser accepts the two required args."""
        parser = create_parser()
        args = parser.parse_args([
            "--config", "eval_config.yaml",
            "--model_name", "Llama-3.2-1B-Instruct",
        ])
        assert args.config == "eval_config.yaml"
        assert args.model_name == "Llama-3.2-1B-Instruct"

    def test_all_args_parse(self):
        """Parser accepts all arguments without error."""
        parser = create_parser()
        args = parser.parse_args([
            "--config", "eval_config.yaml",
            "--model_name", "Llama-3.2-1B-Instruct",
            "--time", "0:20:00",
            "--account", "myacct",
            "--mem", "64G",
            "--partition", "gpu",
            "--constraint", "gpu80",
            "--conda_env", "myenv",
            "--output_slurm", "custom.slurm",
        ])
        assert args.time == "0:20:00"
        assert args.output_slurm == "custom.slurm"

    def test_defaults(self):
        """Default values are sensible."""
        parser = create_parser()
        args = parser.parse_args([
            "--config", "eval_config.yaml",
            "--model_name", "Llama-3.2-1B-Instruct",
        ])
        assert args.time == "0:10:00"
        assert args.conda_env == "cruijff"
        assert args.account is None
        assert args.output_slurm is None


# ---------------------------------------------------------------------------
# main() integration
# ---------------------------------------------------------------------------

class TestMain:

    def test_writes_output_with_epoch(self, tmp_path, monkeypatch):
        """main() writes {task_name}_epoch{N}.slurm when epoch is set."""
        config_file = tmp_path / "eval_config.yaml"
        config_file.write_text(MINIMAL_EVAL_CONFIG + "epoch: 0\n")

        monkeypatch.setattr("sys.argv", [
            "setup_inspect.py",
            "--config", str(config_file),
            "--model_name", "Llama-3.2-1B-Instruct",
        ])
        monkeypatch.chdir(tmp_path)
        main()

        output = tmp_path / "my_task_epoch0.slurm"
        assert output.exists()
        content = output.read_text()
        assert "inspect eval" in content

    def test_writes_output_without_epoch(self, tmp_path, monkeypatch):
        """main() writes {task_name}.slurm when epoch is not set."""
        config_file = tmp_path / "eval_config.yaml"
        config_file.write_text(MINIMAL_EVAL_CONFIG)

        monkeypatch.setattr("sys.argv", [
            "setup_inspect.py",
            "--config", str(config_file),
            "--model_name", "Llama-3.2-1B-Instruct",
        ])
        monkeypatch.chdir(tmp_path)
        main()

        output = tmp_path / "my_task.slurm"
        assert output.exists()

    def test_output_slurm_override(self, tmp_path, monkeypatch):
        """--output_slurm overrides the default filename."""
        config_file = tmp_path / "eval_config.yaml"
        config_file.write_text(MINIMAL_EVAL_CONFIG + "epoch: 0\n")

        monkeypatch.setattr("sys.argv", [
            "setup_inspect.py",
            "--config", str(config_file),
            "--model_name", "Llama-3.2-1B-Instruct",
            "--output_slurm", str(tmp_path / "custom.slurm"),
        ])
        monkeypatch.chdir(tmp_path)
        main()

        assert (tmp_path / "custom.slurm").exists()
        assert not (tmp_path / "my_task_epoch0.slurm").exists()
