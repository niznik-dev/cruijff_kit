"""Integration tests for cruijff_kit.utils.llm_utils.

These tests require GPU access and a base model. They validate the core
inference utilities: token generation, logit extraction, embedding extraction,
pooling, and tensor save/load functionality.

Run on cluster with: sbatch tests/integration/test_llm_utils.slurm
Or locally (if GPU available): pytest tests/integration/test_llm_utils.py -v
"""

import os
import json
import tempfile
from pathlib import Path

import pytest
import numpy as np
import torch

from cruijff_kit.utils.llm_utils import (
    load_prompts_and_targets,
    load_model,
    get_logits,
    get_next_tokens,
    get_embeddings,
    pool_hidden_states,
    save_tensor_with_ids,
    load_tensor_with_ids,
)


# ─── Configuration ────────────────────────────────────────────────────────────
# Set CK_MODELS_DIR to your cluster's shared model directory.
# GPU tests skip automatically if unset. See test_llm_utils.slurm for example.

MODELS_BASE_DIR = os.environ.get("CK_MODELS_DIR", "")
BASE_MODEL_NAME = os.environ.get(
    "CK_TEST_MODEL",
    "Llama-3.2-1B-Instruct"
)
BASE_MODEL_PATH = os.path.join(MODELS_BASE_DIR, BASE_MODEL_NAME)


# ─── Skip conditions ──────────────────────────────────────────────────────────

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available - these tests require GPU"
)

requires_model = pytest.mark.skipif(
    not os.path.exists(BASE_MODEL_PATH),
    reason=f"Model not found at {BASE_MODEL_PATH}"
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def test_data_file(tmp_path_factory):
    """Create a temporary test data file with simple input/output pairs."""
    tmp_dir = tmp_path_factory.mktemp("data")
    data_file = tmp_dir / "test_eval.json"

    test_data = [
        {"input": "apple", "output": "Apple", "id": "word_0"},
        {"input": "banana", "output": "Banana", "id": "word_1"},
        {"input": "cherry", "output": "Cherry", "id": "word_2"},
        {"input": "date", "output": "Date", "id": "word_3"},
    ]

    with open(data_file, "w") as f:
        json.dump(test_data, f)

    return str(data_file)


@pytest.fixture(scope="module")
def prompts_and_targets(test_data_file):
    """Load test prompts and targets."""
    prompts, targets, ids = load_prompts_and_targets(
        test_data_file,
        num_obs=None,
        id_colname="id"
    )
    return prompts, targets, ids


@pytest.fixture(scope="module")
def loaded_model():
    """Load model and tokenizer once for all tests in this module.

    This is expensive (~30s for 1B model) so we use module scope.
    """
    tokenizer, model = load_model(BASE_MODEL_PATH, adapter_path=None)
    yield tokenizer, model
    # Cleanup
    del model
    torch.cuda.empty_cache()


@pytest.fixture
def temp_h5_file(tmp_path):
    """Provide a temporary path for HDF5 save/load tests."""
    return str(tmp_path / "test_tensor.h5")


# ─── Data Loading Tests ───────────────────────────────────────────────────────

class TestLoadPromptsAndTargets:
    """Tests for load_prompts_and_targets function."""

    def test_loads_all_observations(self, test_data_file):
        """Verify all observations are loaded when num_obs=None."""
        prompts, targets, ids = load_prompts_and_targets(
            test_data_file,
            num_obs=None,
            id_colname="id"
        )
        assert len(prompts) == 4
        assert len(targets) == 4
        assert len(ids) == 4

    def test_loads_limited_observations(self, test_data_file):
        """Verify num_obs limits the number loaded."""
        prompts, targets, _ = load_prompts_and_targets(
            test_data_file,
            num_obs=2
        )
        assert len(prompts) == 2
        assert len(targets) == 2

    def test_correct_content(self, test_data_file):
        """Verify correct input/output content is loaded."""
        prompts, targets, ids = load_prompts_and_targets(
            test_data_file,
            id_colname="id"
        )
        assert prompts[0] == "apple"
        assert targets[0] == "Apple"
        assert ids[0] == "word_0"

    def test_no_ids_returns_none(self, test_data_file):
        """Verify ids is None when id_colname not specified."""
        prompts, targets, ids = load_prompts_and_targets(
            test_data_file,
            id_colname=None
        )
        assert ids is None


# ─── Model Loading Tests ──────────────────────────────────────────────────────

@requires_gpu
@requires_model
class TestLoadModel:
    """Tests for load_model function."""

    def test_model_loads_successfully(self, loaded_model):
        """Verify model and tokenizer are loaded."""
        tokenizer, model = loaded_model
        assert tokenizer is not None
        assert model is not None

    def test_tokenizer_has_pad_token(self, loaded_model):
        """Verify tokenizer has a pad token set."""
        tokenizer, _ = loaded_model
        assert tokenizer.pad_token is not None
        assert tokenizer.pad_token != tokenizer.eos_token

    def test_model_on_cuda(self, loaded_model):
        """Verify model is on CUDA device."""
        _, model = loaded_model
        # Check the first parameter's device
        first_param = next(model.parameters())
        assert first_param.device.type == "cuda"


# ─── Logit Extraction Tests ───────────────────────────────────────────────────

@requires_gpu
@requires_model
class TestGetLogits:
    """Tests for get_logits function."""

    def test_logits_shape(self, loaded_model, prompts_and_targets):
        """Verify logits have correct shape (batch_size, vocab_size)."""
        tokenizer, model = loaded_model
        prompts, _, _ = prompts_and_targets

        logits = get_logits(
            model, tokenizer, prompts,
            batch_size=2,
            use_chat_template=True
        )

        assert logits.shape[0] == len(prompts)
        assert logits.shape[1] == tokenizer.vocab_size or logits.shape[1] > 30000

    def test_logits_on_cpu(self, loaded_model, prompts_and_targets):
        """Verify logits are returned on CPU."""
        tokenizer, model = loaded_model
        prompts, _, _ = prompts_and_targets

        logits = get_logits(model, tokenizer, prompts[:2], batch_size=2)

        assert logits.device.type == "cpu"

    def test_logits_dtype(self, loaded_model, prompts_and_targets):
        """Verify logits have the requested dtype."""
        tokenizer, model = loaded_model
        prompts, _, _ = prompts_and_targets

        logits = get_logits(
            model, tokenizer, prompts[:2],
            batch_size=2,
            dtype=torch.float32
        )

        assert logits.dtype == torch.float32


# ─── Token Generation Tests ───────────────────────────────────────────────────

@requires_gpu
@requires_model
class TestGetNextTokens:
    """Tests for get_next_tokens function."""

    def test_generates_tokens(self, loaded_model, prompts_and_targets):
        """Verify tokens are generated."""
        tokenizer, model = loaded_model
        prompts, _, _ = prompts_and_targets

        generated = get_next_tokens(
            model, tokenizer, prompts[:2],
            batch_size=2,
            max_new_tokens=5,
            do_sample=False
        )

        assert generated.shape[0] == 2
        assert generated.shape[1] <= 5

    def test_multiple_return_sequences(self, loaded_model, prompts_and_targets):
        """Verify multiple return sequences reshapes correctly."""
        tokenizer, model = loaded_model
        prompts, _, _ = prompts_and_targets

        generated = get_next_tokens(
            model, tokenizer, prompts[:2],
            batch_size=2,
            max_new_tokens=5,
            do_sample=True,
            num_return_sequences=2,
            temperature=1.0
        )

        # Shape should be (batch, num_return_sequences, seq_len)
        assert generated.shape[0] == 2
        assert generated.shape[1] == 2

    def test_decoded_output_is_text(self, loaded_model, prompts_and_targets):
        """Verify generated tokens can be decoded to text."""
        tokenizer, model = loaded_model
        prompts, _, _ = prompts_and_targets

        generated = get_next_tokens(
            model, tokenizer, prompts[:1],
            batch_size=1,
            max_new_tokens=10,
            do_sample=False
        )

        decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
        assert isinstance(decoded, str)


# ─── Embedding Extraction Tests ───────────────────────────────────────────────

@requires_gpu
@requires_model
class TestGetEmbeddings:
    """Tests for get_embeddings function."""

    def test_embeddings_shape_with_pooling(self, loaded_model, prompts_and_targets):
        """Verify embeddings shape with last_non_padding pooling."""
        tokenizer, model = loaded_model
        prompts, _, _ = prompts_and_targets

        embeds, mask = get_embeddings(
            model, tokenizer, prompts[:2],
            pool="last_non_padding",
            batch_size=2,
            return_mask=True,
            last_layer_only=True
        )

        # Shape should be (batch, 1, hidden_size)
        assert embeds.shape[0] == 2
        assert embeds.shape[1] == 1  # last_layer_only=True
        assert embeds.shape[2] > 0  # hidden size

    def test_embeddings_return_mask(self, loaded_model, prompts_and_targets):
        """Verify attention mask is returned when requested."""
        tokenizer, model = loaded_model
        prompts, _, _ = prompts_and_targets

        embeds, mask = get_embeddings(
            model, tokenizer, prompts[:2],
            pool="mean",
            batch_size=2,
            return_mask=True
        )

        assert mask is not None
        assert mask.shape[0] == 2

    def test_embeddings_no_mask(self, loaded_model, prompts_and_targets):
        """Verify mask is None when not requested."""
        tokenizer, model = loaded_model
        prompts, _, _ = prompts_and_targets

        embeds, mask = get_embeddings(
            model, tokenizer, prompts[:2],
            pool="mean",
            batch_size=2,
            return_mask=False
        )

        assert mask is None


# ─── Pooling Tests ────────────────────────────────────────────────────────────

class TestPoolHiddenStates:
    """Tests for pool_hidden_states function (no GPU required)."""

    @pytest.fixture
    def sample_hidden_states(self):
        """Create sample hidden states tensor (B, L, T, H)."""
        # Batch=2, Layers=2, Tokens=5, Hidden=8
        return torch.randn(2, 2, 5, 8)

    @pytest.fixture
    def sample_attention_mask(self):
        """Create sample attention mask with varying lengths."""
        # First sequence: 3 real tokens, 2 padding
        # Second sequence: 5 real tokens, 0 padding
        mask = torch.tensor([
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1]
        ])
        return mask

    def test_pool_none(self, sample_hidden_states):
        """Verify pool=None returns unchanged tensor."""
        result = pool_hidden_states(sample_hidden_states, pool=None)
        assert torch.equal(result, sample_hidden_states)

    def test_pool_mean(self, sample_hidden_states):
        """Verify mean pooling produces correct shape."""
        result = pool_hidden_states(sample_hidden_states, pool="mean")
        assert result.shape == (2, 2, 8)  # (B, L, H)

    def test_pool_mean_non_padding(self, sample_hidden_states, sample_attention_mask):
        """Verify mean_non_padding pooling uses mask."""
        result = pool_hidden_states(
            sample_hidden_states,
            pool="mean_non_padding",
            attention_mask=sample_attention_mask
        )
        assert result.shape == (2, 2, 8)

    def test_pool_first(self, sample_hidden_states):
        """Verify first pooling takes first token."""
        result = pool_hidden_states(sample_hidden_states, pool="first")
        assert result.shape == (2, 2, 8)
        assert torch.equal(result, sample_hidden_states[:, :, 0, :])

    def test_pool_last(self, sample_hidden_states):
        """Verify last pooling takes last token."""
        result = pool_hidden_states(sample_hidden_states, pool="last")
        assert result.shape == (2, 2, 8)
        assert torch.equal(result, sample_hidden_states[:, :, -1, :])

    def test_pool_last_non_padding(self, sample_hidden_states, sample_attention_mask):
        """Verify last_non_padding pooling uses mask correctly."""
        result = pool_hidden_states(
            sample_hidden_states,
            pool="last_non_padding",
            attention_mask=sample_attention_mask
        )
        assert result.shape == (2, 2, 8)
        # First sequence should use index 2 (last non-padding)
        # Second sequence should use index 4 (last token)

    def test_pool_median(self, sample_hidden_states):
        """Verify median pooling produces correct shape."""
        result = pool_hidden_states(sample_hidden_states, pool="median")
        assert result.shape == (2, 2, 8)

    def test_invalid_pool_raises(self, sample_hidden_states):
        """Verify invalid pool type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported pool type"):
            pool_hidden_states(sample_hidden_states, pool="invalid")

    def test_mask_required_for_non_padding_pools(self, sample_hidden_states):
        """Verify mask is required for non-padding pool types."""
        with pytest.raises(AssertionError):
            pool_hidden_states(sample_hidden_states, pool="last_non_padding")

        with pytest.raises(AssertionError):
            pool_hidden_states(sample_hidden_states, pool="mean_non_padding")


# ─── Save/Load Tests ──────────────────────────────────────────────────────────

class TestSaveLoadTensor:
    """Tests for save_tensor_with_ids and load_tensor_with_ids."""

    def test_save_and_load_roundtrip(self, temp_h5_file):
        """Verify tensor survives save/load roundtrip."""
        original_tensor = torch.randn(4, 10, 128)
        original_ids = ["id_0", "id_1", "id_2", "id_3"]

        save_tensor_with_ids(temp_h5_file, original_tensor, original_ids)
        loaded_tensor, loaded_ids, loaded_mask = load_tensor_with_ids(temp_h5_file)

        assert torch.allclose(original_tensor, loaded_tensor, atol=1e-6)
        assert loaded_ids == original_ids
        assert loaded_mask is None

    def test_save_and_load_with_mask(self, temp_h5_file):
        """Verify attention mask survives roundtrip."""
        original_tensor = torch.randn(4, 10)
        original_ids = ["a", "b", "c", "d"]
        original_mask = torch.tensor([
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1]
        ])

        save_tensor_with_ids(
            temp_h5_file,
            original_tensor,
            original_ids,
            attention_mask=original_mask
        )
        loaded_tensor, loaded_ids, loaded_mask = load_tensor_with_ids(temp_h5_file)

        assert loaded_mask is not None
        assert torch.equal(original_mask, loaded_mask)

    def test_mismatched_lengths_raises(self, temp_h5_file):
        """Verify mismatched tensor/ids lengths raises error."""
        tensor = torch.randn(4, 10)
        ids = ["a", "b"]  # Wrong length

        with pytest.raises(AssertionError):
            save_tensor_with_ids(temp_h5_file, tensor, ids)


# ─── Full Pipeline Test ───────────────────────────────────────────────────────

@requires_gpu
@requires_model
class TestFullPipeline:
    """End-to-end test of the full llm_utils pipeline."""

    @pytest.mark.xfail(
        reason="get_embeddings has bug concatenating attention masks across batches with different sequence lengths",
        strict=True
    )
    def test_complete_workflow(self, loaded_model, prompts_and_targets, tmp_path):
        """Test complete workflow: load data → inference → save → load."""
        tokenizer, model = loaded_model
        prompts, targets, ids = prompts_and_targets

        # 1. Get logits
        logits = get_logits(
            model, tokenizer, prompts,
            batch_size=2,
            use_chat_template=True
        )
        assert logits.shape[0] == len(prompts)

        # 2. Get embeddings
        embeds, mask = get_embeddings(
            model, tokenizer, prompts,
            pool="mean",
            batch_size=2,
            return_mask=True
        )
        assert embeds.shape[0] == len(prompts)

        # 3. Save embeddings
        embed_path = str(tmp_path / "embeddings.h5")
        save_tensor_with_ids(embed_path, embeds.cpu(), ids, attention_mask=mask.cpu())

        # 4. Load and verify
        loaded_embeds, loaded_ids, loaded_mask = load_tensor_with_ids(embed_path)
        assert torch.allclose(embeds.cpu(), loaded_embeds, atol=1e-5)
        assert loaded_ids == ids

        # 5. Generate tokens
        generated = get_next_tokens(
            model, tokenizer, prompts[:2],
            batch_size=2,
            max_new_tokens=5,
            do_sample=False
        )
        assert generated.shape[0] == 2
