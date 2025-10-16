from torchtune.data import Message
from torchtune.models.llama3 import llama3_tokenizer

msgs = [
    Message(role="system", content="Capitalize the first letter of the given word"),
    Message(role="user",   content="apple"),
    Message(role="assistant", content="Apple"),
]

tokenizer_path = "/scratch/gpfs/MSALGANIK/pretrained-llms/Llama-3.2-1B-Instruct/original/tokenizer.model"
tok = llama3_tokenizer(path=tokenizer_path)
tok_ids, mask = tok.tokenize_messages(msgs)

print("Full tokenized output:")
print(tok.decode(tok_ids))
print("\nToken IDs:")
print(tok_ids)
print("\nMask (shows which tokens are trained on):")
print(mask)
print("\nJust the word 'apple':")
print(tok.encode("apple", add_bos=False, add_eos=False))
print("\nJust the word 'Apple':")
print(tok.encode("Apple", add_bos=False, add_eos=False))