import argparse
import json
import random
import string
import sys

parser = argparse.ArgumentParser(description="Generate a JSON file of sampled words with capitalization.")
parser.add_argument("--word-len", type=int, required=True, help="Desired word length (e.g., 5). Must be a positive integer.")
parser.add_argument("--num-words", type=int, required=True, help="Number of words to sample. Must be a positive integer.")

args = parser.parse_args()

if args.word_len <= 0:
    raise ValueError("Word length must be a positive integer.")
if args.num_words <= 0:
    raise ValueError("Number of words must be a positive integer.")

WORD_LEN = args.word_len
NUMBER_OF_WORDS = args.num_words

INFILE = "words.txt"
OUTFILE = f"finetune_words_{WORD_LEN}L_{NUMBER_OF_WORDS}.json"
VALFILE = "val.json"

with open(INFILE, "r") as f:
    words = [line.strip() for line in f if line.strip()]

with open(VALFILE, "r") as vf:
    val_data = json.load(vf)
    val_words = set(item["input"].lower() for item in val_data)

all_n_letter_words = [
    word for word in words
    if len(word) == WORD_LEN and all(c.isalpha() for c in word) and word.lower() not in val_words
]
print(len(all_n_letter_words), f"{WORD_LEN}-letter words found.")

if len(all_n_letter_words) < NUMBER_OF_WORDS:
    raise ValueError(f"Not enough unique {WORD_LEN}-letter words found.")

n_letter_words_sample = random.sample(all_n_letter_words, NUMBER_OF_WORDS)

with open(OUTFILE, "w") as f:
    f.write("[\n")
    for i in range(NUMBER_OF_WORDS):
        word = n_letter_words_sample[i].lower()
        json.dump({"input": word, "output": word.capitalize()}, f)
        f.write(",\n" if i < NUMBER_OF_WORDS - 1 else "\n")
    f.write("]")
