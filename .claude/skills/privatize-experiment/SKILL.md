---
name: privatize-experiment
description: Turn a completed/scaffolded synthetic experiment into a privacy-preserving "redo on private data" handoff — pre-stage a repointed clone the assistant can build (it never reads a record) and generate a tailored follow-along runbook for the data-touching steps the user must do alone. Use when a synthetic run is proven and the user wants to reproduce it on real/secure microdata the assistant must not see — phrases like "redo this on the real data", "do this with secure data ourselves", "private version of this experiment". Flat-JSON experiments only ({train,validation,test}→{input,output}); warns and defers if a tabular-to-text data_generation block is present.
---

# Privatize Experiment

You take a **proven synthetic experiment** and produce two things so the user can
reproduce it on **private microdata you must never read**:

1. **A pre-staged clone** — a new experiment dir, repointed and renamed, with the
   data pointer left as a clearly-marked placeholder. Building this reads only
   `experiment_summary.yaml` and configs (paths, counts, hyperparameters) — never
   a data record — so it is assistant-safe.
2. **A tailored runbook** (`private_data_runbook.md`) walking the user through the
   record-touching steps you cannot do for them.

The governing idea: **redirect a proven run, don't rebuild it.** The synthetic
scaffold already ran green; the only legitimate deltas are the data pointer plus a
handful of data-derived numbers you cannot see.

## The privacy boundary (universal — same every time)

Only three things pull *record content* into a context. Everything else is paths,
counts, and aggregate metrics. The generated runbook restates this table.

| Step | Reads records? | Who |
|------|:---:|---|
| `design-experiment` (already done — reuse the summary) | ❌ | 🤖 assistant-safe |
| `scaffold-experiment` (reads summary, writes configs) | ❌ | 🤖 assistant-safe |
| **build the private `{input,output}` JSON** (reads raw microdata) | ✅ | 🧑 user-only |
| **correctness checks on real columns** | ✅ | 🧑 user-only |
| `submit_torchtune` / `submit_inspect` (sbatch + poll) | ❌ | 🤖 assistant-safe* |
| **`.eval` logs, slurm `.out`, per-example output** | ✅ | 🧑 user-only |
| `summarize-experiment` / `explore-experiment` (read `.eval`) | ✅ | 🧑 user-only |

\* *Submission is safe; the moment a failed job's `.out` (which can echo a row)
must be read, that artifact is user-only — the user hands back a scrubbed error.*

**Never let the assistant read:** the private `*.json`, any `*.eval`, training
`slurm-*.out`, W&B sample tables, or `summary.md` before the user confirms it
quotes no verbatim examples.

## Prerequisites & scope check

1. `claude.local.md` must exist (paths/SLURM defaults). If missing, stop → `/ck-setup`.
2. **Flat-JSON only.** Read the source `experiment_summary.yaml`. If `data` contains
   a `data_generation` block (tabular-to-text), **stop** and tell the user this
   skill v1 covers experiments whose data is already in
   `{train,validation,test}→[{input,output}]` JSON; the raw-tabular → convert step
   needs its own handoff (schema + perturbations) not yet templated here.

## Workflow

### 1. Locate & validate the source experiment

- Take the source experiment dir from the user (or detect a single
  `experiment_summary.yaml` in cwd). Confirm the four artifacts exist: the summary,
  per-run dirs, `*/eval/*/eval.yaml`, and (for fine-tuned runs) `*/setup_finetune.yaml`.
- Confirm it's complete or at least scaffolded — a green run is ideal (the point is
  reproducing something proven).

### 2. Auto-discover the repoint surface

Parse the summary + configs to find the **three string tokens** that carry the data
pointer and experiment identity. For flat-JSON experiments these are deterministic:

| Token | Where to read it | Also appears in |
|---|---|---|
| 🅰 **input dir** | `dirname(data.training.path)` (cross-check `*/setup_finetune.yaml:input_dir_base`) | `finetune.yaml:input_dir` |
| 🅱 **dataset label** | `basename(data.training.path)` minus extension (cross-check `dataset_label`) | `setup_finetune.yaml`, `finetune.yaml`, summary |
| 🅲/🅳 **experiment name** | `experiment.name` (= basename of `experiment.dir`) | every `output_dir`, slurm `--output`, `cd`, ft `model_path`, `config_path` |

`finetune.yaml` composes the path as `${input_dir}/${dataset_label}.json`, so the
fine-tune side repoints with 🅰+🅱; the eval side hard-codes the full path in both
`eval.yaml:data_path` and `cell.slurm` `-T data_path`, so 🅰+🅱 (= the full path)
covers those too.

### 3. Derive the guardrail checks from the summary

These are the data-derived numbers the user must re-verify because you cannot see
them. Compute the formulas from `controls` and bake the experiment's actual values
into the runbook:

- **Warmup vs total steps:** `total_steps = epochs * ceil(N_train / (batch_size *
  gradient_accumulation_steps))`. Flag if `< num_warmup_steps` (torchtune default
  100). Fill `epochs`, `batch_size`, `gradient_accumulation_steps` from `controls`.
- **Token length vs `max_seq_len`:** emit the tokenizer one-liner using a real model
  path from `models.base[].path`; warn that inputs over `controls.max_seq_len`
  truncate silently.
- **Split counts & class balance:** the private positive rate reframes any base-rate
  baseline; emit the split-count one-liner.
- **`input_formatting`:** confirm it stays as set in `setup_finetune.yaml` (flat JSON
  is `''`; a non-empty value appends `raw/` and the job dies at `_setup_data`).

### 4. Identify the data-gen command + leakage to re-verify

Two things, from reading the task script (`evaluation.tasks[].script`) and any
reachable data-gen provenance:

- **The concrete CSV→JSON (or source→JSON) command.** The user starts from raw
  source, not the JSON — so the runbook must give a *runnable* build step, not "apply
  your logic." If a generator is discoverable (e.g. a `to_books_of_life.py` with a
  CLI), render its exact invocation with the flags that reproduce the synthetic build
  (target, split ratios, seed, separator). Note any column-set divergence that would
  change the input body / token length. If no generator is reachable, say so and
  describe the required output shape precisely.
- **The leakage / label derivation to re-verify.** Name the **specific** target
  derivation and dropped/leakage columns (e.g. "ever_kid derives from `KID_1`;
  `KID_*`/`IKID_*` dropped from input"). Where provenance isn't reachable, instruct
  re-verification generically but name the target.

Both go in the runbook's Step 1 — the build command as a runnable block, the leakage
re-verify as a 🧷 correctness-critical, user-only checkbox.

### 5. Gather targets (ask the user)

- **New experiment name** — default `<source_name>` with the date advanced or a
  `_private` marker; confirm with the user. This becomes the clone dir basename and
  token 🅳's replacement (known now → fully baked in).
- **Private data dir + label** — ask if known. If given, bake them in (a fully-ready
  clone). If not, use the placeholders `__FILL_REAL_DATA_DIR__` /
  `__FILL_REAL_LABEL__` (the runbook's first edit replaces them).
- Remind: the private JSON must live in a **governed, non-public** location.

### 6. Pre-stage the clone (assistant-safe)

```bash
cp -r "$SRC" "$DST"            # $DST basename = the new experiment name
rm -rf "$DST"/*/artifacts      # drop synthetic checkpoints + eval logs; jobs rebuild (both slurm scripts mkdir -p)
rm -rf "$DST"/logs             # CRITICAL: stale run-*.state.json would make the resume-safe submitters SKIP the private run
rm -f  "$DST/summary.md"       # synthetic results must not masquerade as private ones
cd "$DST"
# 🅳 experiment name → real name (known): updates every internal path
grep -rl '<SRC_NAME>'  . | xargs -r sed -i "s#<SRC_NAME>#<DST_NAME>#g"
# 🅰 input dir, 🅱 label → real values or __FILL_*__ placeholders
grep -rl '<SRC_INPUT_DIR>' . | xargs -r sed -i "s#<SRC_INPUT_DIR>#<REAL_DATA_DIR_OR_PLACEHOLDER>#g"
grep -rl '<SRC_LABEL>'     . | xargs -r sed -i "s#<SRC_LABEL>#<REAL_LABEL_OR_PLACEHOLDER>#g"
```

Use `#` as the `sed` delimiter (paths contain `/`) and `xargs -r` (no-op on empty).
Then prove the source identity is gone:

```bash
grep -rn -e '<SRC_INPUT_DIR>' -e '<SRC_LABEL>' -e '<SRC_NAME>' "$DST"   # must be empty
```

### 7. Generate the runbook

Render `doc_template.md` into `$DST/private_data_runbook.md`, substituting every
`{{...}}` with this experiment's actual values (paths, the four guardrail numbers
and formulas, the named leakage check, the placeholder-or-real data tokens). The
generated doc is short because the clone+rename is already done — it covers: build
the private JSON, replace the data placeholders (if used), run the guardrail checks,
the verify grep, submit, and the data-blind handback packet.

### 8. Report

Tell the user, concisely: where `$DST` is, what's already done (clone + rename +
the assistant-safe swaps), and the short list of what's left for them (build JSON →
fill placeholders → checks → grep → submit). Point them at the generated runbook.

## Notes

- **Do not** create any script under `src/` — the repoint is `cp` + three `sed`s,
  executed from the skill. Don't over-engineer.
- **Wrapper-only:** rename only names we own. Never touch torchtune recipe keys or
  inspect `@task`/`@scorer` registry names while repointing.
- If the user later wants tabular-to-text support, that's a v2: the real
  raw-tabular → `convert-tabular-to-text` step needs its own user-only handoff.
