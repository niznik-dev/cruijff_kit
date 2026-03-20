"""Tests for cruijff_kit.utils.spot_check.

Tests load_data directly. Tests for _load_model, _generate_one, and spot_check
use mocks since they require GPU access.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
import torch

from cruijff_kit.utils.spot_check import (
    _generate_one,
    _load_model,
    load_data,
    spot_check,
)


@pytest.fixture
def flat_data_file(tmp_path):
    """Create a flat-format JSON data file."""
    data = [
        {"input": "hello", "output": "Hello"},
        {"input": "world", "output": "World"},
        {"input": "test", "output": "Test"},
        {"input": "data", "output": "Data"},
        {"input": "five", "output": "Five"},
    ]
    path = tmp_path / "flat.json"
    path.write_text(json.dumps(data))
    return str(path)


@pytest.fixture
def nested_data_file(tmp_path):
    """Create a nested-format JSON data file with train/validation splits."""
    data = {
        "train": [
            {"input": "train1", "output": "Train1"},
            {"input": "train2", "output": "Train2"},
        ],
        "validation": [
            {"input": "val1", "output": "Val1"},
            {"input": "val2", "output": "Val2"},
            {"input": "val3", "output": "Val3"},
        ],
    }
    path = tmp_path / "nested.json"
    path.write_text(json.dumps(data))
    return str(path)


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer that behaves like AutoTokenizer."""
    tokenizer = MagicMock()
    tokenizer.eos_token = "<eos>"
    tokenizer.pad_token = None
    tokenizer.padding_side = "right"
    tokenizer.apply_chat_template.return_value = "<chat>hello</chat>"
    tokenizer.decode.return_value = "Hello"

    # tokenizer(text, return_tensors="pt") returns dict-like with input_ids
    input_ids = torch.tensor([[1, 2, 3]])
    tokenizer_output = MagicMock()
    tokenizer_output.__getitem__ = lambda self, key: {"input_ids": input_ids}[key]
    tokenizer_output.to.return_value = tokenizer_output
    tokenizer.return_value = tokenizer_output

    return tokenizer


@pytest.fixture
def mock_model():
    """Create a mock model that behaves like AutoModelForCausalLM."""
    model = MagicMock()
    model.device = torch.device("cpu")
    # generate returns input_ids + 2 new tokens
    model.generate.return_value = torch.tensor([[1, 2, 3, 42, 43]])
    return model


class TestLoadData:
    """Tests for load_data function."""

    def test_loads_flat_format(self, flat_data_file):
        prompts, targets = load_data(flat_data_file, n=3)
        assert len(prompts) == 3
        assert len(targets) == 3
        assert prompts[0] == "hello"
        assert targets[0] == "Hello"

    def test_respects_n_limit(self, flat_data_file):
        prompts, targets = load_data(flat_data_file, n=2)
        assert len(prompts) == 2

    def test_n_larger_than_data(self, flat_data_file):
        prompts, targets = load_data(flat_data_file, n=100)
        assert len(prompts) == 5

    def test_nested_with_explicit_split(self, nested_data_file):
        prompts, targets = load_data(nested_data_file, n=10, split="train")
        assert len(prompts) == 2
        assert prompts[0] == "train1"

    def test_nested_with_validation_split(self, nested_data_file):
        prompts, targets = load_data(nested_data_file, n=10, split="validation")
        assert len(prompts) == 3
        assert prompts[0] == "val1"

    def test_nested_auto_detects_validation(self, nested_data_file):
        """Without explicit split, should auto-detect and use validation."""
        prompts, targets = load_data(nested_data_file, n=10)
        assert len(prompts) == 3
        assert prompts[0] == "val1"

    def test_invalid_split_raises(self, nested_data_file):
        with pytest.raises(ValueError, match="Split 'nonexistent' not found"):
            load_data(nested_data_file, n=10, split="nonexistent")

    def test_nested_train_only_falls_back_to_train(self, tmp_path):
        """Nested JSON with 'train' but no 'validation' should fall back to 'train'."""
        data = {
            "train": [
                {"input": "t1", "output": "T1"},
                {"input": "t2", "output": "T2"},
            ],
        }
        path = tmp_path / "train_only.json"
        path.write_text(json.dumps(data))

        prompts, targets = load_data(str(path), n=10)
        assert len(prompts) == 2
        assert prompts[0] == "t1"

    def test_returns_correct_types(self, flat_data_file):
        prompts, targets = load_data(flat_data_file, n=2)
        assert all(isinstance(p, str) for p in prompts)
        assert all(isinstance(t, str) for t in targets)


class TestLoadModel:
    """Tests for _load_model function."""

    @patch("cruijff_kit.utils.spot_check.AutoModelForCausalLM")
    @patch("cruijff_kit.utils.spot_check.AutoTokenizer")
    def test_sets_pad_token_to_eos(self, mock_auto_tok, mock_auto_model):
        tok = MagicMock()
        tok.eos_token = "<eos>"
        mock_auto_tok.from_pretrained.return_value = tok

        tokenizer, model = _load_model("/fake/path")

        assert tokenizer.pad_token == "<eos>"

    @patch("cruijff_kit.utils.spot_check.AutoModelForCausalLM")
    @patch("cruijff_kit.utils.spot_check.AutoTokenizer")
    def test_sets_left_padding(self, mock_auto_tok, mock_auto_model):
        tok = MagicMock()
        tok.eos_token = "<eos>"
        mock_auto_tok.from_pretrained.return_value = tok

        tokenizer, model = _load_model("/fake/path")

        assert tokenizer.padding_side == "left"

    @patch("cruijff_kit.utils.spot_check.AutoModelForCausalLM")
    @patch("cruijff_kit.utils.spot_check.AutoTokenizer")
    def test_loads_with_device_map_auto(self, mock_auto_tok, mock_auto_model):
        tok = MagicMock()
        tok.eos_token = "<eos>"
        mock_auto_tok.from_pretrained.return_value = tok

        _load_model("/fake/path")

        mock_auto_model.from_pretrained.assert_called_once_with(
            "/fake/path", device_map="auto"
        )

    @patch("cruijff_kit.utils.spot_check.AutoModelForCausalLM")
    @patch("cruijff_kit.utils.spot_check.AutoTokenizer")
    def test_calls_model_eval(self, mock_auto_tok, mock_auto_model):
        tok = MagicMock()
        tok.eos_token = "<eos>"
        mock_auto_tok.from_pretrained.return_value = tok

        _, model = _load_model("/fake/path")

        model.eval.assert_called_once()


class TestGenerateOne:
    """Tests for _generate_one function."""

    def test_uses_chat_template_by_default(self, mock_model, mock_tokenizer):
        _generate_one(mock_model, mock_tokenizer, "hello")

        mock_tokenizer.apply_chat_template.assert_called_once()
        args = mock_tokenizer.apply_chat_template.call_args
        messages = args[0][0]
        assert messages == [{"role": "user", "content": "hello"}]
        assert args[1]["add_generation_prompt"] is True
        assert args[1]["tokenize"] is False

    def test_includes_system_prompt_in_chat_template(self, mock_model, mock_tokenizer):
        _generate_one(mock_model, mock_tokenizer, "hello", sysprompt="Be helpful")

        messages = mock_tokenizer.apply_chat_template.call_args[0][0]
        assert messages[0] == {"role": "system", "content": "Be helpful"}
        assert messages[1] == {"role": "user", "content": "hello"}

    def test_skips_chat_template_when_disabled(self, mock_model, mock_tokenizer):
        _generate_one(mock_model, mock_tokenizer, "raw prompt", use_chat_template=False)

        mock_tokenizer.apply_chat_template.assert_not_called()
        # Should tokenize the raw prompt directly
        mock_tokenizer.assert_called_once_with("raw prompt", return_tensors="pt")

    def test_strips_input_tokens_from_output(self, mock_model, mock_tokenizer):
        """Generated output should only contain new tokens, not the input."""
        # Input has 3 tokens, output has 5 (3 input + 2 generated)
        # decode is called with only the 2 generated token ids
        _generate_one(mock_model, mock_tokenizer, "hello")

        decode_args = mock_tokenizer.decode.call_args
        decoded_ids = decode_args[0][0]
        assert decoded_ids.tolist() == [42, 43]

    def test_passes_max_new_tokens_to_generate(self, mock_model, mock_tokenizer):
        _generate_one(mock_model, mock_tokenizer, "hello", max_new_tokens=20)

        generate_kwargs = mock_model.generate.call_args[1]
        assert generate_kwargs["max_new_tokens"] == 20
        assert generate_kwargs["do_sample"] is False

    def test_returns_decoded_string(self, mock_model, mock_tokenizer):
        mock_tokenizer.decode.return_value = "World"
        result = _generate_one(mock_model, mock_tokenizer, "hello")
        assert result == "World"


class TestSpotCheck:
    """Tests for spot_check function."""

    @patch("cruijff_kit.utils.spot_check._generate_one")
    @patch("cruijff_kit.utils.spot_check._load_model")
    def test_returns_results_with_match_info(self, mock_load, mock_gen, flat_data_file):
        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_tokenizer, mock_model)
        # Return exact matches for first 2, mismatch for 3rd
        mock_gen.side_effect = ["Hello", "World", "WRONG"]

        results = spot_check(
            model_path="/fake/model",
            data_path=flat_data_file,
            n=3,
        )

        assert len(results) == 3
        assert results[0]["match"] is True
        assert results[1]["match"] is True
        assert results[2]["match"] is False

    @patch("cruijff_kit.utils.spot_check._generate_one")
    @patch("cruijff_kit.utils.spot_check._load_model")
    def test_passes_sysprompt_to_generate(self, mock_load, mock_gen, flat_data_file):
        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")
        mock_load.return_value = (MagicMock(), mock_model)
        mock_gen.return_value = "Hello"

        spot_check(
            model_path="/fake/model",
            data_path=flat_data_file,
            n=1,
            sysprompt="Be helpful",
        )

        gen_kwargs = mock_gen.call_args[1]
        assert gen_kwargs["sysprompt"] == "Be helpful"

    @patch("cruijff_kit.utils.spot_check._generate_one")
    @patch("cruijff_kit.utils.spot_check._load_model")
    def test_respects_no_chat_template(self, mock_load, mock_gen, flat_data_file):
        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")
        mock_load.return_value = (MagicMock(), mock_model)
        mock_gen.return_value = "Hello"

        spot_check(
            model_path="/fake/model",
            data_path=flat_data_file,
            n=1,
            use_chat_template=False,
        )

        gen_kwargs = mock_gen.call_args[1]
        assert gen_kwargs["use_chat_template"] is False
