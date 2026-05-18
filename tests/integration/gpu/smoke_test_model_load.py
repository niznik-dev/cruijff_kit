"""Smoke test: load a model and generate a few tokens.

Catches CUDA library rot, model weight corruption, broken conda env,
and transformers/torch version incompatibilities.

Usage:
    python tests/integration/gpu/smoke_test_model_load.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/scratch/gpfs/MSALGANIK/pretrained-llms/Llama-3.2-1B-Instruct"


def main():
    print(f"Loading model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda"
    )

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"Prompt: {prompt}")
    print(f"Generated: {generated}")
    assert len(generated) > len(prompt), "Model failed to generate any new tokens"
    print("PASS: model loading + generation")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
