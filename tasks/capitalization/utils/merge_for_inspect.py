import argparse, torch, os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel

def main():
    p = argparse.ArgumentParser(description="Merge a PEFT/LoRA adapter into a base HF model.")
    p.add_argument("--base_model", required=True, help="HF model id or local path to base")
    p.add_argument("--adapter_path", required=True, help="Path or HF id for the LoRA/PEFT adapter")
    p.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"],
                   help="Compute dtype for loading before merge")
    p.add_argument("--device", default="auto",
                   help="e.g. 'cuda:0', 'cpu', or 'auto' (lets HF place on available GPU)")
    args = p.parse_args()

    OUTPUT_DIR = "inspect_merged_model"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.dtype == "auto":
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    else:
        dtype = getattr(torch, args.dtype)

    device_map = "auto" if (args.device == "auto" and torch.cuda.is_available()) else None

    print(f"[1/4] Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map=device_map,
    )

    print(f"[2/4] Applying adapter: {args.adapter_path}")
    peft_model = PeftModel.from_pretrained(
        base_model,
        args.adapter_path,
        torch_dtype=dtype,
    )

    print("[3/4] Merging adapter into base...")
    merged = peft_model.merge_and_unload()  # returns a standard HF model (no PEFT wrapper)

    print("[4/4] Saving merged model and tokenizer...")
    # Save model
    merged.save_pretrained(OUTPUT_DIR)

    # Save tokenizer (from base)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Optionally copy generation config if present
    try:
        cfg = AutoConfig.from_pretrained(args.base_model)
        cfg.save_pretrained(OUTPUT_DIR)
    except Exception:
        pass

    print(f"Done. Merged model at: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
