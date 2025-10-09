import argparse
import json
import random
import string
import sys

parser = argparse.ArgumentParser(description="Generate a JSON file of sampled words with capitalization.")
parser.add_argument("--word-len", type=int, required=True, help="Desired word length (e.g., 5). Must be a positive integer.")
parser.add_argument("--num-words", type=int, required=True, help="Number of words to sample. Must be a positive integer.")
parser.add_argument("--train-fraction", type=float, default=0.8, help="Fraction of data to use for training (between 0 and 1).")

args = parser.parse_args()

if args.word_len <= 0:
    raise ValueError("Word length must be a positive integer.")
if args.num_words <= 0:
    raise ValueError("Number of words must be a positive integer.")

WORD_LEN = args.word_len
NUMBER_OF_WORDS = args.num_words
TRAIN_FRACTION = args.train_fraction
TRAIN_PERCENT = int(TRAIN_FRACTION * 100)
TRAIN_CUTOFF_INDEX = int(NUMBER_OF_WORDS * TRAIN_FRACTION)
VAL_CUTOFF_INDEX = int(NUMBER_OF_WORDS * (TRAIN_FRACTION + (1 - TRAIN_FRACTION) / 2))

INFILE = "words_alpha.txt"
OUTFILE = f"words_{WORD_LEN}L_{TRAIN_PERCENT}P_{NUMBER_OF_WORDS}.json"

with open(INFILE, "r") as f:
    words = [line.strip() for line in f if line.strip()]

all_n_letter_words = [
    word for word in words
    if len(word) == WORD_LEN and all(c.isalpha() for c in word)
]
print(len(all_n_letter_words), f"{WORD_LEN}-letter words found.")

if len(all_n_letter_words) < NUMBER_OF_WORDS:
    raise ValueError(f"Not enough unique {WORD_LEN}-letter words found.")

n_letter_words_sample = random.sample(all_n_letter_words, NUMBER_OF_WORDS)

with open(OUTFILE, "w") as f:
    f.write("[\n")
    for i in range(TRAIN_CUTOFF_INDEX):
        word = n_letter_words_sample[i].lower()
        json.dump({"input": word, "output": word.capitalize(), "split": "train"}, f)
        f.write(",\n")
    for i in range(TRAIN_CUTOFF_INDEX, VAL_CUTOFF_INDEX):
        word = n_letter_words_sample[i].lower()
        json.dump({"input": word, "output": word.capitalize(), "split": "validation"}, f)
        f.write(",\n")
    for i in range(VAL_CUTOFF_INDEX, NUMBER_OF_WORDS):
        word = n_letter_words_sample[i].lower()
        json.dump({"input": word, "output": word.capitalize(), "split": "test"}, f)
        f.write(",\n" if i < NUMBER_OF_WORDS - 1 else "\n")
    f.write("]")
