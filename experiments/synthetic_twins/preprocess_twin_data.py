"""
Preprocess synthetic twin data for fine-tuning.

Reads raw twin data CSV and converts to JSON format with train/validation/test splits.
Uses raw twin values (not absolute differences) for each trait.

Input: /scratch/gpfs/MSALGANIK/inputs/zyg/raw/twindat_sim_100k_24.csv
Output: data/yellow/synthetic_twins/twin_zygosity.json

IMPORTANT: This dataset is for use by the MSALGANIK research group only.
Sharing this data with others requires explicit permission from the research group.
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path

# Define paths relative to repo root
REPO_ROOT = Path(__file__).parent.parent.parent
INPUT_FILE = Path("/scratch/gpfs/MSALGANIK/inputs/zyg/raw/twindat_sim_100k_24.csv")
OUTPUT_FILE = REPO_ROOT / "data" / "yellow" / "synthetic_twins" / "twin_zygosity.json"

# Load the CSV file
print(f"Loading data from: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)
print(f"Loaded {len(df)} twin pairs")

# Identify traits (assumes trait columns end with '.1' and '.2')
traits = sorted(set(col[:-2] for col in df.columns if col.endswith('.1')))
print(f"Found {len(traits)} traits: {', '.join(traits[:5])}...")

# Create the formatted data using raw twin values
def row_to_sample(row):
    """Convert a row to input/output format."""
    input_lines = []
    for trait in traits:
        # Use raw values for each twin
        val1 = row[trait + '.1']
        val2 = row[trait + '.2']
        input_lines.append(f"{trait}: {val1:.2f}, {val2:.2f}")

    # Join trait information with commas
    input_text = ", ".join(input_lines)

    # Determine output based on monozygotic status
    # 1 = monozygotic (identical), 0 = dizygotic (fraternal)
    output_text = "1" if row['zyg'] == 1 else "0"

    return {"input": input_text, "output": output_text}

# Apply transformation to create formatted data
samples = df.apply(row_to_sample, axis=1).tolist()

# Shuffle the data (ensuring reproducibility)
np.random.seed(42)
np.random.shuffle(samples)

# Split dataset: 80% train, 10% validation, 10% test
total_samples = len(samples)
train_split = int(total_samples * 0.8)
val_split = int(total_samples * 0.9)

train_data = samples[:train_split]
val_data = samples[train_split:val_split]
test_data = samples[val_split:]

# Create single JSON file with top-level split keys
output_data = {
    "train": train_data,
    "validation": val_data,
    "test": test_data
}

# Ensure output directory exists
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Save the dataset
print(f"Saving to: {OUTPUT_FILE}")
with open(OUTPUT_FILE, "w") as f:
    json.dump(output_data, f, indent=2)

print(f"✓ Saved dataset with splits:")
print(f"  - train: {len(train_data)} samples ({len(train_data)/total_samples*100:.1f}%)")
print(f"  - validation: {len(val_data)} samples ({len(val_data)/total_samples*100:.1f}%)")
print(f"  - test: {len(test_data)} samples ({len(test_data)/total_samples*100:.1f}%)")

