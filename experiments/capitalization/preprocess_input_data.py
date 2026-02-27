import argparse
import json
import random
from pathlib import Path

from cruijff_kit.utils.logger import setup_logger

# Set up logging
logger = setup_logger(__name__)

parser = argparse.ArgumentParser(
    description="Generate a JSON file of sampled words with capitalization."
)
parser.add_argument(
    "--word-len",
    type=int,
    required=True,
    help="Desired word length (e.g., 5). Must be a positive integer.",
)
parser.add_argument(
    "--num-words",
    type=int,
    required=True,
    help="Number of words to sample. Must be a positive integer.",
)
parser.add_argument(
    "--train-fraction",
    type=float,
    default=0.8,
    help="Fraction of data to use for training (between 0 and 1).",
)
parser.add_argument(
    "--use-chat-template", action="store_true", help="Use chat template format."
)
parser.add_argument(
    "--output_dir",
    type=Path,
    default=None,
    help="Output directory (default: data/green/capitalization/)",
)
parser.add_argument(
    "--input_file",
    type=Path,
    default=None,
    help="Input word list file (default: experiments/capitalization/input/words_alpha.txt)",
)

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
USE_CHAT_TEMPLATE = args.use_chat_template

# Set default output directory relative to repository root
if args.output_dir is None:
    # Find repository root (assuming script is in experiments/capitalization/)
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent
    args.output_dir = repo_root / "data" / "green" / "capitalization"

# Create output directory if it doesn't exist
args.output_dir.mkdir(parents=True, exist_ok=True)

# Set default input file if not provided
if args.input_file is None:
    # Default to the input file in experiments/capitalization/input/
    script_dir = Path(__file__).parent
    args.input_file = script_dir / "input" / "words_alpha.txt"

OUTFOLDER = f"words_{WORD_LEN}L_{TRAIN_PERCENT}P_{NUMBER_OF_WORDS}{'_c' if USE_CHAT_TEMPLATE else ''}"
OUTFILE = f"{OUTFOLDER}.json"

with open(args.input_file, "r") as f:
    words = [line.strip() for line in f if line.strip()]

all_n_letter_words = [
    word for word in words if len(word) == WORD_LEN and all(c.isalpha() for c in word)
]
logger.info(f"{len(all_n_letter_words)} {WORD_LEN}-letter words found.")

if len(all_n_letter_words) < NUMBER_OF_WORDS:
    raise ValueError(f"Not enough unique {WORD_LEN}-letter words found.")

n_letter_words_sample = random.sample(all_n_letter_words, NUMBER_OF_WORDS)

# Build nested structure with top-level split keys
train_examples = []
validation_examples = []
test_examples = []

for i in range(TRAIN_CUTOFF_INDEX):
    word = n_letter_words_sample[i].lower()
    if USE_CHAT_TEMPLATE:
        train_examples.append(
            {
                "messages": [
                    {"role": "user", "content": f"{word}"},
                    {"role": "assistant", "content": word.capitalize()},
                ],
            }
        )
    else:
        train_examples.append({"input": word, "output": word.capitalize()})

for i in range(TRAIN_CUTOFF_INDEX, VAL_CUTOFF_INDEX):
    word = n_letter_words_sample[i].lower()
    if USE_CHAT_TEMPLATE:
        validation_examples.append(
            {
                "messages": [
                    {"role": "user", "content": f"{word}"},
                    {"role": "assistant", "content": word.capitalize()},
                ],
            }
        )
    else:
        validation_examples.append({"input": word, "output": word.capitalize()})

for i in range(VAL_CUTOFF_INDEX, NUMBER_OF_WORDS):
    word = n_letter_words_sample[i].lower()
    if USE_CHAT_TEMPLATE:
        test_examples.append(
            {
                "messages": [
                    {"role": "user", "content": f"{word}"},
                    {"role": "assistant", "content": word.capitalize()},
                ],
            }
        )
    else:
        test_examples.append({"input": word, "output": word.capitalize()})

# Write nested JSON structure
if USE_CHAT_TEMPLATE:
    outfolder_path = args.output_dir / OUTFOLDER
    outfolder_path.mkdir(parents=True, exist_ok=True)
    with open(outfolder_path / "train.json", "w") as f:
        json.dump(train_examples, f, indent=2)
    with open(outfolder_path / "validation.json", "w") as f:
        json.dump(validation_examples, f, indent=2)
    with open(outfolder_path / "test.json", "w") as f:
        json.dump(test_examples, f, indent=2)
    logger.info(
        f"✅  Generated {outfolder_path}: {len(train_examples)} train, {len(validation_examples)} validation, {len(test_examples)} test"
    )
else:
    output_data = {
        "train": train_examples,
        "validation": validation_examples,
        "test": test_examples,
    }
    output_path = args.output_dir / OUTFILE
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    logger.info(
        f"✅  Generated {output_path}: {len(train_examples)} train, {len(validation_examples)} validation, {len(test_examples)} test"
    )
