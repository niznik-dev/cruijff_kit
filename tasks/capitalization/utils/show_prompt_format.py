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

logger.info("Full tokenized output:")
logger.info(tok.decode(tok_ids))
logger.info("\nToken IDs:")
logger.info(tok_ids)
logger.info("\nMask (shows which tokens are trained on):")
logger.info(mask)
logger.info("\nJust the word 'apple':")
logger.info(tok.encode("apple", add_bos=False, add_eos=False))
logger.info("\nJust the word 'Apple':")
logger.info(tok.encode("Apple", add_bos=False, add_eos=False))