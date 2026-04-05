# Validation

After generating all dataset files (Step 5), validate every file before presenting results to the user. Run all checks — do not stop at the first failure. Collect all issues and report them together.

## Validation Checklist

For **each** generated file (`{condition}_{split}_s{seed}.json`), run through all applicable checks below.

### 1. File Existence and Structure

- [ ] Output `.json` file exists and is valid JSON
- [ ] Sidecar `.meta.json` file exists and is valid JSON
- [ ] JSON top-level key matches the expected split (e.g., `"train"`, `"test"`, `"validation"`)
- [ ] Entries list is non-empty

```bash
cd {cruijff_kit_path} && python -c "
import json, sys
with open('{output_path}') as f:
    data = json.load(f)
keys = list(data.keys())
assert len(keys) == 1, f'Expected 1 top-level key, got {keys}'
split_key = keys[0]
assert split_key == '{expected_split}', f'Split key {split_key} != expected {expected_split}'
entries = data[split_key]
assert len(entries) > 0, f'No entries in {split_key}'
print(f'OK: {len(entries)} entries under \"{split_key}\"')
"
```

### 2. Entry Schema Validation

Every entry must have exactly `{"input", "output"}` — no other keys.

The system prompt is an experiment-level setting (in `experiment_summary.yaml`), not per-entry data. Context only appears in entries when `context_placement == "preamble"`, in which case it is prepended to the input text.

- [ ] All entries have `"input"` (non-empty string)
- [ ] All entries have `"output"` (non-empty string)
- [ ] No entries contain keys beyond `{"input", "output"}`

```bash
cd {cruijff_kit_path} && python -c "
import json
with open('{output_path}') as f:
    data = json.load(f)
entries = list(data.values())[0]
expected_keys = {'input', 'output'}
issues = []
for i, e in enumerate(entries):
    if not isinstance(e.get('input'), str) or not e['input'].strip():
        issues.append(f'Entry {i}: missing or empty input')
    if not isinstance(e.get('output'), str) or not e['output'].strip():
        issues.append(f'Entry {i}: missing or empty output')
    if set(e.keys()) != expected_keys:
        issues.append(f'Entry {i}: unexpected keys {set(e.keys()) - expected_keys}')
if issues:
    print('ISSUES FOUND:')
    for issue in issues[:20]:
        print(f'  - {issue}')
    if len(issues) > 20:
        print(f'  ... and {len(issues) - 20} more')
else:
    print(f'OK: all {len(entries)} entries have valid schema')
"
```

### 3. Metadata Consistency

Cross-check the `.meta.json` sidecar against the actual data file.

- [ ] `row_count` matches the actual number of entries in the JSON file
- [ ] `size_bytes` matches the actual file size on disk
- [ ] `split` matches the top-level key in the JSON file
- [ ] `condition_name` matches the expected condition
- [ ] `features` list is non-empty
- [ ] `generated_at` is a valid ISO 8601 timestamp
- [ ] If template is `narrative` with a custom file: `template_file` is present

```bash
cd {cruijff_kit_path} && python -c "
import json, os
from datetime import datetime

output_path = '{output_path}'
meta_path = output_path.replace('.json', '.meta.json')

with open(output_path) as f:
    data = json.load(f)
with open(meta_path) as f:
    meta = json.load(f)

entries = list(data.values())[0]
split_key = list(data.keys())[0]
actual_size = os.path.getsize(output_path)

issues = []
if meta['row_count'] != len(entries):
    issues.append(f'row_count mismatch: meta={meta[\"row_count\"]}, actual={len(entries)}')
if meta['size_bytes'] != actual_size:
    issues.append(f'size_bytes mismatch: meta={meta[\"size_bytes\"]}, actual={actual_size}')
if meta['split'] != split_key:
    issues.append(f'split mismatch: meta={meta[\"split\"]}, json_key={split_key}')
if meta['condition_name'] != '{expected_condition}':
    issues.append(f'condition_name mismatch: meta={meta[\"condition_name\"]}, expected={expected_condition}')
if not meta.get('features'):
    issues.append('features list is empty')
try:
    datetime.fromisoformat(meta['generated_at'])
except (ValueError, KeyError):
    issues.append('generated_at is missing or not valid ISO 8601')

if issues:
    print('METADATA ISSUES:')
    for issue in issues:
        print(f'  - {issue}')
else:
    print(f'OK: metadata consistent (row_count={meta[\"row_count\"]}, size_bytes={meta[\"size_bytes\"]})')
"
```

### 4. Label Distribution

Check that the output labels are reasonable. Extreme imbalance or unexpected label values warrant a warning.

- [ ] All output labels belong to the expected label set
- [ ] Neither label class is completely absent (warn if <5% of total)
- [ ] Label distribution is reported to the user

```bash
cd {cruijff_kit_path} && python -c "
import json
from collections import Counter

with open('{output_path}') as f:
    data = json.load(f)
entries = list(data.values())[0]
labels = Counter(e['output'] for e in entries)
total = len(entries)

print('Label distribution:')
for label, count in sorted(labels.items()):
    pct = 100 * count / total
    bar = '#' * int(pct / 2)
    print(f'  \"{label}\": {count} ({pct:.1f}%) {bar}')

# Warn on extreme imbalance
for label, count in labels.items():
    if count / total < 0.05:
        print(f'  WARNING: label \"{label}\" is <5% of data ({count}/{total})')

# Check for expected labels (threshold-based should be '0' and '1')
expected_labels = {expected_label_set}  # e.g., {'0', '1'} or set from target_mapping values
unexpected = set(labels.keys()) - expected_labels
if unexpected:
    print(f'  WARNING: unexpected labels found: {unexpected}')
"
```

**Note:** Replace `{expected_label_set}` with `{"0", "1"}` for threshold-based targets, or with the set of values from `target_mapping` for categorical targets.

### 5. Text Quality Checks

Spot-check the generated text for common problems.

- [ ] No inputs contain literal `"nan"`, `"None"`, or `"missing"` unless missing value handling is set to `include`
- [ ] No inputs are suspiciously short (<20 characters)
- [ ] Inputs contain the expected question text (if a question was specified)
- [ ] If `context_placement == "preamble"` and context is non-empty: inputs start with the context text

```bash
cd {cruijff_kit_path} && python -c "
import json

with open('{output_path}') as f:
    data = json.load(f)
entries = list(data.values())[0]

issues = []
nan_strings = {'nan', 'none', 'null', 'missing', 'n/a', 'na'}
question_text = '''{question_text}'''
context_text = '''{context_text}'''
context_placement = '{context_placement}'
missing_handling = '{missing_value_handling}'  # 'skip' or 'include'

for i, e in enumerate(entries):
    inp = e['input']

    # Check for leaked missing values (only if handling is 'skip')
    if missing_handling == 'skip':
        for token in nan_strings:
            lower_inp = inp.lower()
            if f': {token}' in lower_inp or f'is {token}' in lower_inp:
                issues.append(f'Entry {i}: possible leaked missing value \"{token}\"')
                break

    # Check for suspiciously short inputs
    if len(inp.strip()) < 20:
        issues.append(f'Entry {i}: suspiciously short input ({len(inp.strip())} chars)')

    # Check question presence
    if question_text and question_text not in inp:
        issues.append(f'Entry {i}: question text not found in input')

    # Check context placement
    if context_placement == 'preamble' and context_text and not inp.startswith(context_text):
        issues.append(f'Entry {i}: preamble context not at start of input')

if issues:
    print(f'TEXT QUALITY ISSUES ({len(issues)} total):')
    for issue in issues[:20]:
        print(f'  - {issue}')
    if len(issues) > 20:
        print(f'  ... and {len(issues) - 20} more')
else:
    print(f'OK: text quality checks passed for all {len(entries)} entries')
"
```

### 6. Cross-File Consistency

When multiple files are generated for the same experiment, verify consistency across them.

- [ ] All files using the same seed have the same `source_rows_total`
- [ ] Train and test files for the same condition have row counts consistent with the split ratio
- [ ] All conditions reference the same schema file
- [ ] All conditions use the same seed

```bash
cd {cruijff_kit_path} && python -c "
import json, glob

meta_files = glob.glob('{output_dir}/*.meta.json')
if not meta_files:
    print('No metadata files found')
    exit()

metas = {}
for mf in sorted(meta_files):
    with open(mf) as f:
        metas[mf] = json.load(f)

issues = []

# Check consistent seed and source
seeds = {m['seed'] for m in metas.values()}
if len(seeds) > 1:
    issues.append(f'Inconsistent seeds across files: {seeds}')

source_totals = {m['source_rows_total'] for m in metas.values()}
if len(source_totals) > 1:
    issues.append(f'Inconsistent source_rows_total: {source_totals}')

schemas = {m['schema'] for m in metas.values()}
if len(schemas) > 1:
    issues.append(f'Inconsistent schemas: {schemas}')

# Check train/test split arithmetic
by_condition = {}
for mf, m in metas.items():
    cond = m['condition_name']
    by_condition.setdefault(cond, {})[m['split']] = m

for cond, splits in by_condition.items():
    if 'train' in splits and 'test' in splits:
        train_n = splits['train']['row_count']
        test_n = splits['test']['row_count']
        total = train_n + test_n
        expected_ratio = splits['train']['split_ratio']
        actual_ratio = train_n / total if total > 0 else 0
        # Allow small tolerance for rounding
        if abs(actual_ratio - expected_ratio) > 0.02:
            issues.append(
                f'{cond}: train/test ratio {actual_ratio:.3f} != expected {expected_ratio} '
                f'(train={train_n}, test={test_n})')

if issues:
    print('CROSS-FILE ISSUES:')
    for issue in issues:
        print(f'  - {issue}')
else:
    print(f'OK: {len(metas)} files are consistent')
"
```

## Presenting Results

After all checks complete, present a summary table:

```
Validation Results:
+---------------------------------+--------+------+---------------+---------+
| File                            | Split  | Rows | Labels        | Status  |
+---------------------------------+--------+------+---------------+---------+
| dict_full_train_s42.json        | train  | 800  | 0:412, 1:388  | PASS    |
| dict_full_test_s42.json         | test   | 200  | 0:105, 1:95   | PASS    |
| dict_synonym_test_s42.json      | test   | 200  | 0:105, 1:95   | WARN    |
+---------------------------------+--------+------+---------------+---------+
Cross-file consistency: PASS
```

Then show 2-3 sample entries from each file so the user can visually confirm the text looks right.

## If Validation Fails

### Blocking Issues (FAIL)
These must be fixed before proceeding:
- File doesn't exist or isn't valid JSON
- Entry missing required keys (`input`, `output`)
- Entry contains unexpected keys (e.g., `system_prompt`)
- Metadata `row_count` doesn't match actual data
- Unexpected labels outside the expected set
- Empty entries list
- Inconsistent seeds across files

**Action:** Report the specific failures and re-run `convert.py` for the affected files. If the issue persists, investigate the source data and schema.

### Warnings (WARN)
These should be reviewed but don't block progress:
- Extreme label imbalance (<5% for a class)
- Possible leaked missing values
- Suspiciously short inputs
- Minor split ratio deviation (>1% but within 2%)

**Action:** Report warnings to the user. Let them decide whether to proceed or regenerate.

## After Validation Passes

Proceed to Step 7 (Report Final Paths) in SKILL.md.
