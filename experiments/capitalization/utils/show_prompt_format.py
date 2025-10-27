from torchtune.data import Message
from torchtune.models.llama3 import llama3_tokenizer

from cruijff_kit.utils.logger import setup_logger

# Set up logging
logger = setup_logger(__name__)

msgs = [
    Message(role="system", content="Capitalize the first letter of the given word"),
    Message(role="user",   content="apple"),
    Message(role="assistant", content="Apple"),
]

tokenizer_path = "/scratch/gpfs/MSALGANIK/pretrained-llms/Llama-3.2-1B-Instruct/original/tokenizer.model"
tok = llama3_tokenizer(path=tokenizer_path)
tok_ids, mask = tok.tokenize_messages(msgs)

logger.info(
    f"Full tokenized output:\n"
    f"{tok.decode(tok_ids)}\n\n"
    f"Token IDs:\n"
    f"{tok_ids}\n\n"
    f"Mask (shows which tokens are trained on):\n"
    f"{mask}\n\n"
    f"Just the word 'apple':\n"
    f"{tok.encode('apple', add_bos=False, add_eos=False)}\n\n"
    f"Just the word 'Apple':\n"
    f"{tok.encode('Apple', add_bos=False, add_eos=False)}"
)