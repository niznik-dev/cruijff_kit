
#%%
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
#%%

from cruijff_kit.utils.logger import setup_logger

# Set up logging
logger = setup_logger(__name__)
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please ensure you have a compatible GPU and the necessary drivers installed.")


BASE_DIR = "/scratch/gpfs/MSALGANIK/pretrained-llms"
MODEL = "Llama-3.2-1B-Instruct"
CURR_ADAPTER = "/scratch/gpfs/MSALGANIK/mjs3/ck-out-oct13-mjs-words_7L_80P_10000/epoch_0/adapter_weights"
MODEL_PATH = BASE_DIR + "/" + MODEL


# Load tokenizer and models
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# Add padding token if necessary
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map = 'auto',
                                                  low_cpu_mem_usage = True, dtype = 'auto')
adapted_model = PeftModel.from_pretrained(base_model, CURR_ADAPTER, dtype = 'auto')

# check that both models are on the same device
assert base_model.device == adapted_model.device, "Base model and adapted model are on different devices"
device = base_model.device

# put together text input
SYSTEM_PROMPT = 'Convert the first character of the word to uppercase and keep the rest of the word in lowercase. Return only the modified word.'
word = input("Please enter a word you want me to capitalize (quit to stop): ")

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": word},
]

# Tokenize using the chat template
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to(device)

# Create attention mask
attention_mask = torch.ones_like(input_ids).to(device)

# Pass to the model as a dictionary
input_dict = {"input_ids": input_ids, "attention_mask": attention_mask}

outputs_base = base_model.generate(**input_dict, max_new_tokens=20,
                                   do_sample=False)
outputs_adapted = adapted_model.generate(**input_dict, max_new_tokens=20,
                                         do_sample=False)
 
generated_text_base = tokenizer.decode(outputs_base[0], skip_special_tokens=True)
generated_text_adapted = tokenizer.decode(outputs_adapted[0], skip_special_tokens=True)

logger.info(f"Base model: {generated_text_base}")
logger.info(f"Generated model: {generated_text_adapted}")