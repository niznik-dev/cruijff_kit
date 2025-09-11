from torchtune.data import Message
from torchtune.models.llama3 import llama3_tokenizer

msgs = [
    Message(role="system", content="You transform words to UPPERCASE."),
    Message(role="user",   content="Capitalize: apple"),
    Message(role="assistant", content="APPLE"),
]

tokenizer_path = "/scratch/gpfs/MSALGANIK/pretrained-llms/Llama-3.2-1B-Instruct/original/tokenizer.model"
tok = llama3_tokenizer(path=tokenizer_path)
tok_ids, mask = tok.tokenize_messages(msgs)

print(tok.decode(tok_ids))