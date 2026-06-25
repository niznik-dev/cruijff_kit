{{!--
  Output skeleton for privatize-experiment. Render to
  {{DST}}/private_data_runbook.md, substituting every {{...}}. Step 0 exports the
  shell vars for the bash steps; the assistant in Step 2 fills + verifies (it is a
  no-op-but-verify when paths are already baked). Keep the emoji + checkbox style —
  this is a human-facing follow-along.

  CRITICAL: render every fenced code block FLUSH-LEFT (column 0), never indented
  under a list item. An indented heredoc breaks twice: Python raises
  IndentationError (so it prints nothing) AND the indented closing `PY` stops
  terminating the block. Put the checkbox/prose first, then the code block at the
  left margin below it.

  CRITICAL: the verification snippets are pure ```python``` blocks (NOT `python -
  <<PY` bash heredocs). ipython CANNOT see the shell's env vars, so never use
  os.environ — a one-time fill block sets REAL_DATA_DIR/REAL_LABEL as plain Python
  strings (substituted from {{...}}; the user edits them if placeholders) and the
  checks reference the derived `JSON`. Use ```bash``` only for real shell commands
  (sed/grep/sbatch/submitters), and never put `< >` angle-bracket placeholders in a
  bash line — `<` is shell redirection and the line errors. Use `/PATH/TO/...` markers.
--}}
# 🔐 Private-Data Runbook — `{{DST_NAME}}`

Reproduce the proven synthetic experiment **`{{SRC_NAME}}`** on **private
microdata the assistant must never read**.

- 🧬 **Method:** the synthetic scaffold was cloned, renamed, and repointed for you
  — this is a *redirect of a green run*, not a rebuild.
- ✅ **Already staged (assistant-safe — no record was read):** clone at `{{DST}}`,
  every internal path renamed `{{SRC_NAME}}` → `{{DST_NAME}}`, synthetic
  checkpoints/logs/summary cleared.
- 🧑 **Left to you (touches real data):** the steps below.
- 🔐 The full who-sees-what contract is in the `privatize-experiment` skill; the
  short version is the gate at the bottom.

> ℹ️ **`bash` blocks** run in your shell (Step 0 sets their vars). **`python` blocks**
> paste into ipython, which can't see your shell's vars — so the Step 3 Python block
> has its own one-line path fill. Copy blocks whole; they start at the left margin.

---

## 🧰 Step 0 — Set the variables (do this once)

Edit the two `/PATH/TO/...` lines (your private paths — the assistant can't see them).
**No `< >` brackets** — bash reads `<` as redirection and the line errors. These cover
the `bash` steps; the Step 3 Python checks fill their path separately.

```bash
export REAL_CSV=/PATH/TO/real_source.csv            # <-- EDIT
export REAL_DATA_DIR=/PATH/TO/private_data_folder   # <-- EDIT
export REAL_LABEL={{REAL_LABEL}}
export DST={{DST}}
echo "CSV=$REAL_CSV"; echo "JSON=$REAL_DATA_DIR/$REAL_LABEL.json"
```

---

## 🗂️ Step 1 — Build the private dataset 🧑 *you-only*

Reads real microdata → fully yours.

**Build the canonical JSON** at `{{REAL_DATA_DIR}}/{{REAL_LABEL}}.json`, shape
`{"train":[...],"validation":[...],"test":[...]}`, each entry `{input, output}` —
identical structure to the synthetic file. Run the discovered generator with the
flags that reproduce the synthetic build, **defaulting the column selection to the
full synthetic list** so a wide real source doesn't emit every column:

{{DATA_GEN_COMMAND_BLOCK}}
{{!-- a runnable FLUSH-LEFT ```bash``` block if a generator was found — include its
     column-set default, e.g. --cols-file column_sets/synthetic_full.txt; else a
     precise description of the required output shape and "apply your own source→JSON step" --}}

- [ ] **Narrow the columns per question** by copying the full column-set file into a
  subset (keep the label-source column for a target). The §3b token-length check
  catches a body that grew wider than the synthetic one.
- [ ] ⚠️ **Re-running overwrites `--out`** (the generator writes, never appends): same
  args reproduce an identical file, but a different column set / flags clobber the
  prior one.
- [ ] 🧷 **Re-verify the leakage / label derivation on real columns** (correctness-
  critical, cannot be delegated): {{LEAKAGE_CHECK}}. Confirm no *other* real column
  trivially encodes the outcome.
- [ ] Stage it in a **governed, non-public** location (not a public git repo).

## 🩹 Step 2 — Fill + verify with the assistant 🧑

Run the fill assistant **in your own terminal** (not via Claude). It asks for your
private data folder, fills the clone's `__FILL_*__` placeholders, verifies, and writes
the real paths + splits to `{{DST}}/verify.runbook_filled.md` — which Claude is denied
from reading. Only pass/fail counts reach the screen.

```bash
python {{CK_DIR}}/scripts/privatize_fill.py "{{DST}}"
```

It prints **✅ READY to submit** when 0 placeholders remain, every eval cell points at
your JSON, and the JSON exists. Open `verify.runbook_filled.md` for the actual paths.

Manual fallback (sed, no verify): `cd {{DST}} && grep -rl '__FILL_REAL_DATA_DIR__' . |
xargs -r sed -i "s#__FILL_REAL_DATA_DIR__#YOUR_PRIVATE_DIR#g"`

---

## 🔬 Step 3 — Verify the 4 numbers the assistant can't see 🧑 *you-only*

Run each on the private JSON; update `experiment_summary.yaml` provenance to match.

First, in ipython, set the path once (ipython can't see your shell's vars — fill it
here, not via os.environ):

```python
REAL_DATA_DIR = "{{REAL_DATA_DIR}}"   # EDIT if this is a placeholder
REAL_LABEL = "{{REAL_LABEL}}"
JSON = f"{REAL_DATA_DIR}/{REAL_LABEL}.json"
```

**3a — Splits & class balance** (synthetic: {{SYNTH_SPLITS}}) — prints `pos` per split:

```python
import json
d = json.load(open(JSON))
for k, v in d.items():
    pos = sum(e["output"] == "1" for e in v)
    print(f"{k}: n={len(v)} pos={pos} rate={pos/len(v):.3f}")
```

→ update `data.training.splits` + `size_kb`; reread any base-rate baseline through the
*private* positive rate (not an accuracy floor).

**3b — Max token length vs `max_seq_len: {{MAX_SEQ_LEN}}`** (over-length inputs
**truncate silently** = dropped features):

```python
import json
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("{{MODEL_PATH}}")
d = json.load(open(JSON))
m = max(len(tok(e["input"])["input_ids"]) for v in d.values() for e in v)
print(f"max input tokens = {m}  -> {'OK' if m < {{MAX_SEQ_LEN}}-32 else 'BUMP max_seq_len in finetune.yaml + setup_finetune.yaml'}")
```

**3c — Warmup vs total steps.** `total_steps = {{EPOCHS}} * ceil(N_train /
({{BATCH_SIZE}} * {{GRAD_ACCUM}}))`. If that is `< 100` (the warmup), the LR never
leaves warmup and "no convergence" is an artifact. Only a risk if `N_train` is small.

**3d — `input_formatting: '{{INPUT_FORMATTING}}'` unchanged** (a non-empty value
appends `raw/` and the job dies at `_setup_data`):

```bash
grep -n "input_formatting" {{DST}}/*_ft/setup_finetune.yaml
```

---

## 🔎 Step 4 — Prove the redirect is clean

Step 2's assistant already checked placeholders, paths, and JSON presence — see
`{{DST}}/verify.runbook_filled.md`. Optional independent re-check:

**Cleanliness (assistant-safe — no real paths in this output):**

```bash
# must return NOTHING — no synthetic identity or unfilled placeholder survived:
grep -rn -e '{{SRC_INPUT_DIR}}' -e '{{SRC_LABEL}}' -e '{{SRC_NAME}}' -e '__FILL_' {{DST}}
```

**Real paths (blind — appended to the Claude-denied report):**

```bash
grep -rh 'data_path' {{DST}}/*/eval/*/eval.yaml | sort -u >> {{DST}}/verify.runbook_filled.md
echo "appended eval data_paths to verify.runbook_filled.md — open it; Claude is blocked"
```

- [ ] cleanliness grep prints nothing → ✅   assistant reported **READY** → ✅

---

## 🚀 Step 5 — Submit & monitor 🤖 *assistant-safe to drive*

Fine-tune first (evals depend on the checkpoints):

```bash
python -m cruijff_kit.tools.run.submit_torchtune {{DST}}   # blocks to terminal; resume-safe
python -m cruijff_kit.tools.run.submit_inspect   {{DST}}   # only after fine-tunes finish
```

- [ ] long run? `touch {{DST}}/logs/.detach` to release the watcher without killing
  jobs; re-attach with `--resume-monitor`.
- [ ] ⚠️ a job fails? its `slurm-*.out` may echo a real row — **read it yourself**,
  hand back only the scrubbed error class.

---

## 📊 Step 6 — Results, data-blind 🧑 *you-only → 🤖 handback*

- [ ] Run `summarize-experiment` yourself (it reads `.eval` logs = echoed real
  inputs). Treat `summary.md` as 🔴 until you confirm it quotes **no** verbatim examples.
- [ ] 📨 Hand the assistant only the **aggregate packet** — per-cell balanced acc,
  AUC, ECE/RCE, format%; private class balance (as provenance); whether fine-tuning
  fixed format/calibration. ❌ never the `.eval` files, raw completions, or example inputs.

---

## ✅ Gate before `sbatch`

- [ ] private JSON built; leakage/label derivation re-verified on real columns (Step 1)
- [ ] clone filled + assistant reported **READY** (Step 2); Step 4 cleanliness grep prints **nothing**
- [ ] splits + balance updated; `total_steps > 100`; `max tokens < max_seq_len`
- [ ] `input_formatting` unchanged
- [ ] private JSON in a **governed, non-public** dir
- [ ] you own 🔴 `.eval` / slurm `.out` / raw `summary.md`; the assistant gets only aggregates

🎻 *Redirect the proven run; keep the records on your side of the wall; read the
scoreboard together.*
