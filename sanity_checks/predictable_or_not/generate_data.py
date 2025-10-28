import argparse
import copy
import json
import random

parser = argparse.ArgumentParser()
parser.add_argument("--in_seq_len", type=int, default=5, help="Input sequence length")
args = parser.parse_args()

if args.in_seq_len < 1:
    raise ValueError("Input sequence length must be at least 1.")

# Predictable input and output
pp = []
for i in range(1000):
    line = {'input': '', 'output': ''}
    start = random.randint(0, 1000 - args.in_seq_len - 1)
    for j in range(args.in_seq_len):
        line['input'] += str(start + j) + ','
    line['output'] = str(start + j + 1)
    pp.append(line)

# Predictable input and unpredictable output; just replace the output with a random number
pu = copy.deepcopy(pp)
for line in pu:
    line['output'] = str(random.randint(0, 1000))

# Unpredictable input and predictable output; 5 random numbers and the output is just always 42
up = []
for i in range(1000):
    line = {'input': '', 'output': '42'}
    for j in range(5):
        line['input'] += str(random.randint(0, 1000)) + ','
    up.append(line)

# Unpredictable input and unpredictable output; just random numbers
uu = copy.deepcopy(up)
for line in uu:
    line['output'] = str(random.randint(0, 1000))

# Split each scenario 90/10 for train/validation and save with splits as top-level keys
for scenario_name, scenario_data in [('pp', pp), ('pu', pu), ('up', up), ('uu', uu)]:
    # Split 90/10
    split_point = int(len(scenario_data) * 0.9)
    train_data = scenario_data[:split_point]
    val_data = scenario_data[split_point:]

    # Create single JSON file with train/validation splits as top-level keys
    output = {
        'train': train_data,
        'validation': val_data
    }

    with open(f'{scenario_name}.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f'{scenario_name}.json: {len(train_data)} train, {len(val_data)} validation')
