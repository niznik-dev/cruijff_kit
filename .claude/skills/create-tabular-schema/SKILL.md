---
name: create-tabular-schema
description: Create a schema YAML file for tabular source data. Defines columns, types, value maps, and perturbation metadata. Use before design-experiment or convert-tabular-to-text when working with tabular data.
---

# Create Tabular Schema

Create a schema YAML file that describes a tabular dataset's columns for use in text generation experiments. The schema is a prerequisite for both experiment design (where it informs target column, features, etc.) and dataset generation (where it drives text formatting).

```
create-tabular-schema → design-experiment → convert-tabular-to-text
```

## Your Task

Guide the user through inspecting their source data and building a schema file. The schema lives in `ck-data/schemas/` and is reusable across experiments.

## Step 1: Check for Existing Schemas

Check `ck-data/schemas/` for an existing schema that might match the user's source data:

```bash
ls {scratch_dir}/ck-data/schemas/ 2>/dev/null
```

If a matching schema exists, load it and confirm with the user. If they want to use it as-is, done. If they want to modify it, proceed with it as a starting point.

## Step 2: Inspect Source Data

Ask the user for the source data file path, then read the structure:

```bash
cd {cruijff_kit_path} && python -c "
from cruijff_kit.tabular_to_text_gen.lib.readers import read_tabular
df = read_tabular('{source_path}')
print('Columns:', list(df.columns))
print('Shape:', df.shape)
print('Dtypes:')
print(df.dtypes)
print()
print(df.head(5).to_string())
"
```

Report the columns, types, and shape to the user. 

## Step 3: Select Columns

Ask the user which columns to include in the schema. Not every column needs to be in the schema — only those that might be used as features, targets, or are otherwise relevant to experiments.

**Ask:** "Which columns should be included in the schema? You can always add more later."

## Step 4: Generate Skeleton

Generate a skeleton schema from the selected columns:

```bash
cd {cruijff_kit_path} && python -c "
import yaml
from cruijff_kit.tabular_to_text_gen.lib.schema import Schema
skeleton = Schema.generate_skeleton('{source_path}', {columns_list})
print(yaml.dump(skeleton, default_flow_style=False))
"
```

## Step 5: Review and Refine

Present the skeleton to the user for review. For each column, discuss:

- **Display name** — human-readable label (e.g., "Age" for column `AGEP`)
- **Type** — `numeric` or `categorical` (auto-inferred, but confirm)
- **Unit** — for numeric columns (e.g., "years old", "dollars")
- **Value map** — for categorical columns with coded values, maps raw codes to human-readable labels (e.g., `{1: "Male", 2: "Female"}`). **Do NOT guess or infer what codes mean.** If a column appears to use numeric codes for categories, ask the user to provide a codebook, data dictionary, or reference file that documents the coding scheme. Read that file to extract the correct mappings. If the user cannot provide a reference, ask them to supply the mappings directly.
Not all fields are needed for every column. Synonyms, shorthand maps, and restatements are only needed if the user plans to use those perturbation types (see Step 5b).

### Step 5b: Perturbation Metadata (Optional)

**Ask:** "Do you plan to use any perturbation types (synonym, shorthand, clause_addition)? If so, which ones?"

If the user wants perturbation metadata, generate suggestions for each relevant column and present them for review:

- **Synonyms** — alternative display names for the synonym perturbation (e.g., `["Age", "Years of age"]`). Suggest 2-3 plausible synonyms per column based on the display name.
- **Shorthand map** — full-form to abbreviated mappings for the shorthand perturbation. Suggest common abbreviations for display names and value labels.
- **Restatement templates** — for the clause_addition perturbation, templates with `{value}` placeholders. Suggest 1-2 restatement templates per column.

Present all suggestions to the user in a clear table or list format. **Ask:** "Here are my suggested perturbation entries for each column. Please review and let me know what to change, add, or remove."

Only finalize perturbation metadata after the user has approved or edited the suggestions.

## Step 6: Save Schema

Save the final schema to `ck-data/schemas/`:

```bash
mkdir -p {scratch_dir}/ck-data/schemas
```

Write the schema YAML file to `{scratch_dir}/ck-data/schemas/{dataset_name}.yaml`.

Report the saved path to the user.

## Step 7: Validate

After saving, validate the schema file:

1. **YAML syntax check** — load the file with `yaml.safe_load()` and confirm it parses without errors.
2. **User review** — present the parsed schema to the user. **Ask:** "Please review the schema above. Would you like to make any changes?"

```bash
cd {cruijff_kit_path} && python -c "
import yaml
with open('{schema_path}') as f:
    schema = yaml.safe_load(f)
print(yaml.dump(schema, default_flow_style=False))
print('Schema is valid YAML.')
"
```

If YAML parsing fails, fix the syntax error and re-save. If the user requests changes, edit the schema file and repeat this validation step. Only proceed once the schema parses cleanly and the user confirms it looks correct.

## What This Skill Does NOT Do

- **Design experiments** — defer to design-experiment
- **Generate datasets** — defer to convert-tabular-to-text
- **Choose features or conditions** — those are experiment-level decisions made during design-experiment

## Important Notes

- **Schema reuse.** Schemas describe the source data, not a specific experiment. The same schema can be used across multiple experiments with different feature subsets and conditions.
- **Perturbation metadata is optional.** Synonyms, shorthand maps, and restatements only matter if the user plans to use those perturbation types. Don't require them upfront — they can be added later.
- **Value maps matter.** For categorical columns with coded values (e.g., 1/2 for Male/Female), the value map is critical for generating correct text. Flag coded-looking columns and ask the user for a codebook or data dictionary file that documents the code-to-label mappings. Never infer or guess what numeric codes mean — even common-seeming codes (e.g., 1/2 for sex) can vary across datasets.
