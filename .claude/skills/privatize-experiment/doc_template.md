{{!--
  Output skeleton for privatize-experiment. Render to
  {{DST}}/private_data_runbook.md, substituting every {{...}}. Keep the emoji +
  checkbox style — this is a human-facing follow-along.

  Substitutions: {{DST_NAME}} {{SRC_NAME}} {{DST}} {{CK_DIR}} {{REAL_LABEL}}
  {{DATA_GEN_COMMAND_BLOCK}} {{LEAKAGE_CHECK}}. The per-experiment numbers
  (max_seq_len, epochs, splits, etc.) are NOT substituted — privatize_fill.py reads
  them from the clone's own configs and checks them, so the runbook stays short.

  CRITICAL: render every fenced code block FLUSH-LEFT (column 0), never indented
  under a list item — an indented heredoc raises IndentationError and the closing
  delimiter stops terminating. All blocks here are ```bash``` (shell); there are no
  ipython snippets — the assistant (Step 2) runs every data check. Never put `< >`
  angle-bracket placeholders in a bash line (`<` is shell redirection); use
  `/PATH/TO/...` markers.
--}}
# 🔐 Private-Data Runbook — `{{DST_NAME}}`

Reproduce the proven synthetic experiment **`{{SRC_NAME}}`** on **private microdata
the assistant must never read**.

- 🧬 **Method:** the synthetic scaffold was cloned, renamed, and repointed for you
  — a *redirect of a green run*, not a rebuild.
- ✅ **Already staged (assistant-safe — no record was read):** clone at `{{DST}}`,
  every internal path renamed `{{SRC_NAME}}` → `{{DST_NAME}}`, synthetic
  checkpoints/logs/summary cleared, data dir left as `__FILL_REAL_DATA_DIR__`.
- 🧑 **Left to you (touches real data):** the steps below — three commands.
- 🔐 The full who-sees-what contract is in the `privatize-experiment` skill; the
  short version is the gate at the bottom.

> ℹ️ Every block runs in your **shell** — paste them whole (they start at the left
> margin). Step 2's assistant does all the data checks for you; no snippets to fiddle.

---

## 🧰 Step 0 — Set the variables (do this once)

Edit the two `/PATH/TO/...` lines (your private paths — the assistant can't see them).
**No `< >` brackets** — bash reads `<` as redirection and the line errors.

```bash
export REAL_CSV=/PATH/TO/real_source.csv            # <-- EDIT
export REAL_DATA_DIR=/PATH/TO/private_data_folder   # <-- EDIT
export REAL_LABEL={{REAL_LABEL}}
export DST={{DST}}
echo "CSV=$REAL_CSV"; echo "JSON=$REAL_DATA_DIR/$REAL_LABEL.json"
```

---

## 🗂️ Step 1 — Build the private dataset 🧑 *you-only*

Reads real microdata → fully yours. Run the generator with the flags that reproduce
the synthetic build, **defaulting the column selection to the full synthetic list**
so a wide real source doesn't emit every column:

{{DATA_GEN_COMMAND_BLOCK}}
{{!-- a runnable FLUSH-LEFT ```bash``` block using "$REAL_CSV" and
     "$REAL_DATA_DIR/$REAL_LABEL.json", with the generator's column-set default
     (e.g. --cols-file column_sets/synthetic_full.txt); else a precise description
     of the required {train,validation,test}->{input,output} JSON shape. --}}

- [ ] ⚠️ **Re-running overwrites `--out`** (the generator writes, never appends).
- [ ] 🧷 **Re-verify the leakage / label derivation on real columns** (correctness-
  critical, cannot be delegated): {{LEAKAGE_CHECK}}. Confirm no *other* real column
  trivially encodes the outcome.
- [ ] Stage the JSON in a **governed, non-public** location (not a public git repo).

---

## 🩹 Step 2 — Fill + check with the assistant 🧑

Run the assistant **in your own terminal** (not via Claude). It asks for your private
data folder, fills the clone's `__FILL_REAL_DATA_DIR__`, and runs the **full
pre-submit suite** — placeholders cleared, eval cells point at the JSON, JSON exists,
split balance, max tokens vs `max_seq_len`, warmup vs total steps, `input_formatting`
— all read from the clone's own configs. It prints **✅ READY to submit** or the
exact ❌ rows to fix. Real paths + the detail land in `{{DST}}/verify.runbook_filled.md`
(Claude-denied); stdout is counts/booleans only.

```bash
python {{CK_DIR}}/scripts/privatize_fill.py "$DST"
```

Manual fallback (fill only, no checks): `cd $DST && grep -rl '__FILL_REAL_DATA_DIR__' .
| xargs -r sed -i "s#__FILL_REAL_DATA_DIR__#$REAL_DATA_DIR#g"`

---

## 🚀 Step 3 — Submit & monitor 🤖 *assistant-safe to drive*

Only once Step 2 says **READY**. Fine-tune first (evals depend on the checkpoints):

```bash
python -m cruijff_kit.tools.run.submit_torchtune $DST   # blocks to terminal; resume-safe
python -m cruijff_kit.tools.run.submit_inspect   $DST   # only after fine-tunes finish
```

- [ ] long run? `touch $DST/logs/.detach` to release the watcher without killing
  jobs; re-attach with `--resume-monitor`.
- [ ] ⚠️ a job fails? its `slurm-*.out` may echo a real row — **read it yourself**,
  hand back only the scrubbed error class.
- [ ] 🔁 fixed a knob (batch size, `--time`) and need to **resubmit** a failed run?
  The "already submitted" memory is `$DST/logs/run-torchtune.state.json`, **not**
  `artifacts/` — clearing artifacts alone won't redispatch. Delete that run's entry
  (keyed `<run>/finetune.slurm`), or the whole state file, to redispatch. Also confirm
  no `logs/.detach` from above is still lurking — it's sticky and silently blocks
  submission until removed.

---

## 📊 Step 4 — Results, data-blind 🧑 *you-only → 🤖 handback*

- [ ] Run `summarize-experiment` yourself (it reads `.eval` logs = echoed real
  inputs). Treat `summary.md` as 🔴 until you confirm it quotes **no** verbatim examples.
- [ ] 📨 Hand the assistant only the **aggregate packet** — per-cell balanced acc,
  AUC, ECE/RCE, format%; private class balance (as provenance); whether fine-tuning
  fixed format/calibration. ❌ never the `.eval` files, raw completions, or example inputs.

---

## ✅ Gate before `sbatch`

- [ ] private JSON built; leakage/label derivation re-verified on real columns (Step 1)
- [ ] Step 2 assistant printed **✅ READY** (no ❌ rows)
- [ ] private JSON in a **governed, non-public** dir
- [ ] you own 🔴 `.eval` / slurm `.out` / raw `summary.md`; the assistant gets only aggregates

🎻 *Redirect the proven run; keep the records on your side of the wall; read the
scoreboard together.*
