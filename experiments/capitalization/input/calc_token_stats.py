import json
from transformers import AutoTokenizer
import argparse

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Process a JSON file to calculate token statistics.")
parser.add_argument('--json_filename', type=str, help='Name of the input file (should be in the same directory as this script).')
args = parser.parse_args()

with open(args.json_filename, 'r') as file:
    data = json.load(file)

# Count tokens for each word in "input"
token_dist = {}

tokenizer = AutoTokenizer.from_pretrained("/scratch/gpfs/MSALGANIK/pretrained-llms/Llama-3.2-1B-Instruct")
for entry in data:
    input = entry['input']
    output = entry['output']

    input_tokens = tokenizer(input, return_tensors="pt")
    output_tokens = tokenizer(output, return_tensors="pt")

    # Llama always starts with a BOS token, so we remove that for counting
    input_tokens_clean = {k: v[:, 1:] for k, v in input_tokens.items()}
    output_tokens_clean = {k: v[:, 1:] for k, v in output_tokens.items()}
    input_token_count = input_tokens_clean['input_ids'].size(1)
    output_token_count = output_tokens_clean['input_ids'].size(1)

    key = f"{input_token_count}-{output_token_count}"
    token_dist.setdefault(key, []).append(entry)

    if key == "1-1":
        print(f"Input: {input}, Tokens: {tokenizer.convert_ids_to_tokens(input_tokens_clean['input_ids'][0])}")
        print(f"Output: {output}, Tokens: {tokenizer.convert_ids_to_tokens(output_tokens_clean['input_ids'][0])}")

for key in token_dist:
    print(f"{key}: {len(token_dist[key])} samples")

# Let's plot the frequency of token counts by keys
keys = list(token_dist.keys())
frequencies = [len(token_dist[key]) for key in keys]

# Sort the keys so it's like 1-1, 1-2, 2-1, 2-2, ...
sorted_pairs = sorted(zip(keys, frequencies), key=lambda x: (int(x[0].split('-')[0]), int(x[0].split('-')[1])))
keys, frequencies = zip(*sorted_pairs)

plt.bar(keys, frequencies)
plt.xlabel('Token Count (Input-Output)')
plt.ylabel('Frequency')
plt.title('Token Count for 5 Letter Word Dataset (lowercase-uppercase)')
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig('token_distribution.png')

total_samples = sum(frequencies)
different_length_count = sum(freq for key, freq in zip(keys, frequencies) if key.split('-')[0] != key.split('-')[1])
percentage_different_length = (different_length_count / total_samples) * 100
print(f"{percentage_different_length:.2f}% of words have different token lengths between input and output.")