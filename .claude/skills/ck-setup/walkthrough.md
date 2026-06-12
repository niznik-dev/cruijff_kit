# Greenfield walkthrough

Step-by-step Q&A flow for first-time setup. Source of truth: `claude.local.md.template` at the repo root. The section numbering below mirrors the template exactly — when the template adds or removes a section, walk what's actually there.

## Principles

- **One question at a time.** Don't dump a wall of prompts on the user. Each section gets a brief rationale, then the questions, then move on.
- **Auto-detect what you can.** `whoami`, `id -gn`, `pwd`, `command -v module` — these answer questions without asking.
- **Show defaults from the template.** When the template has a sensible default (e.g. `cruijff` for the conda env name, `1` for default GPUs), present it as `[default: cruijff]` and let the user accept by pressing enter.
- **Allow `[skip]`.** A user might not know their wandb entity yet. Mark the section with the unfilled placeholders and continue. Validate-mode will surface them later.
- **Never write secrets to claude.local.md.** No HuggingFace tokens, no SSH keys. The file is gitignored but accidents happen.

## Optional first step: borrow from an existing slurm script

Before walking the template, offer the user a shortcut for SLURM Defaults:

> "Do you have an existing SLURM submission script (a `.slurm` file from a peer, your HPC admin, or a previous project) I can reference for your SLURM defaults? If so, I'll read the `#SBATCH` directives and pre-fill the SLURM Defaults section. Otherwise we'll fill those in by hand."

If the user says yes:

1. Ask for the absolute path to the file. Read it.
2. Parse `#SBATCH` directives, mapping to template fields:

   | `#SBATCH` directive | Template field |
   |---|---|
   | `--account=<value>` | SLURM Account |
   | `--partition=<value>` | Partition |
   | `--constraint=<value>` | Constraint |
   | `--time=<value>` | Default time |
   | `--gpus=<N>` or `--gres=gpu:<N>` | Default GPUs |

3. When you reach Section 2 (SLURM Defaults), present the parsed values as `[from your slurm script: <value>]` defaults — not facts to write blindly. The user confirms or overrides each one.
4. If the file isn't a SLURM script, or doesn't contain useful directives, say so and fall through to manual asks.

If the user says no, skip this step entirely — manual asks happen in Section 2.

## Starting over (existing `claude.local.md`)

Greenfield mode is normally only entered when no `claude.local.md` exists. The exception: a user explicitly asks to redo their config (eval #4: "I want to start over"). In that case:

1. **Confirm intent before any destructive action.** Ask: "this will rename your existing claude.local.md to a dated backup and start a fresh walkthrough. Continue?" Only proceed on a clear yes.
2. **Rename, don't delete.** Move the old file to a dated backup:
   ```bash
   mv claude.local.md "claude.local.md.bak.$(date +%Y-%m-%d)"
   ```
3. **Handle backup-name collisions.** If a backup with today's date already exists, suffix with a counter — `.bak.2026-05-04.2`, `.3`, etc. Don't overwrite an existing backup.
4. **Log the rename** with `RENAME_BACKUP` (see `logging.md`), then enter the section flow below as if greenfield.

Never `rm` or truncate. The backup is the user's safety net.

## Section flow

Walk the template top-to-bottom. The order below mirrors the template's actual section structure (sections + subsections); if the template grows new sections, walk them too.

### 1. HPC Environment

**Why this matters:** these paths are referenced by every other skill (scaffold-experiment, run-experiment, summarize-experiment). Wrong paths = nothing works.

Auto-detect (treat as defaults to confirm, not facts to write):
- `Username`: `whoami`
- `Group`: `id -gn` — note that the *directory* case for the group may differ from the token (e.g. directory `MSALGANIK`, group `msalganik`). When showing the auto-detected value, ask the user to confirm the casing that appears in their actual scratch path.
- `Working directory`: `pwd` (the cruijff_kit repo root the user is in)

Ask:
- `Cluster` (e.g. Della, Traverse) — name only, free text.
- `Documentation` URL — optional, `[skip]` is fine.
- Confirm `Scratch directory`: default `/scratch/gpfs/<group>/<username>` — show the auto-detected path and ask "is this where your scratch lives?". On most Princeton clusters yes; on others adjust.

#### Subsection: Shared Resources

These are *group-shared* paths — directories the whole HPC group reads from. Two fields in the template:

- **Models directory** — almost always useful. Pretrained LLMs are large (10s–100s of GB), so groups download each one once and share. Default: `/scratch/gpfs/<group>/pretrained-llms`. If the user is solo on the cluster they may use their own path; if they don't have any models downloaded yet, `[skip]` is fine.
- **Shared datasets** — rarely used. `[skip]` is normal here. Only fill in if the group is collaborating on a shared dataset stored at a group path. Most experimental work uses the per-user `{ck_data_directory}` instead (covered in Section 7).

### 2. SLURM Defaults

**Why this matters:** every fine-tune and eval is a SLURM job. Wrong account = job rejected; wrong constraint = job sits in the wrong queue.

If the user said yes to the slurm-script shortcut earlier, present each parsed value as a `[from your slurm script: <value>]` default to confirm. Otherwise ask:

- `Account` — SLURM allocation name. The user gets this from their HPC admin or PI. *Required.*
- `Partition` — leave empty if the cluster has no partitions.
- `Constraint` — common values: `gpu80` (80GB GPU), `gpu40` (40GB GPU), or empty.
- `Default time` — the template default `0:59:00` is sensible; under 1h hits Princeton's test queue.
- `Default GPUs` — `1` is a sensible default.
- `Default conda environment` — default `cruijff`.

#### Subsection: GPU Configuration

**Why this matters:** `design-experiment` and `scaffold-experiment` map model VRAM requirements to SLURM constraints. If GPU types are unknown, those skills fall back to conservative defaults that may waste allocation.

Ask:
- `Available GPU types` — e.g. "A100 80GB", "H100 80GB". Free text. The user usually knows; if not, `[skip]` and they can fill in later.
- `Full GPU constraint` — the SLURM constraint flag for a full dedicated GPU (e.g. `gpu80`).
- `Shared/MIG partition` — if applicable, e.g. `nomig`. Empty if not applicable.

The `MIG notes` line is informational guidance, not a question — copy it through unchanged.

### 3. Module System

**Why this matters:** some HPC clusters use environment modules; some don't. Skills check this section to decide whether to prepend `module load <anaconda_module>` to commands.

Auto-detect: `command -v module && echo yes || echo no`.

If yes, ask:
- The cluster's anaconda module name. The template shows `anaconda3/2025.6` as a *placeholder example* (the version that worked on Della at the time of writing) — do not assume it's right for the user's cluster. They can find theirs with `module avail anaconda` (or by asking their HPC admin).

If no, skip the section in the output (the template says "remove this section" in that case).

### 4. Conda Environment Setup

**Why this matters:** every skill activates this env before running any Python.

Ask:
- `Environment name` — default `cruijff`. Should match what was set in SLURM Defaults > `Default conda environment`.
- Auto-check: `conda env list | grep -q <env_name>` — if not present, gently note "this env doesn't exist yet; run `make install` from the repo root before using `/design-experiment`."

### 5. GitHub Configuration

**Why this matters:** issue creation and project board updates use `gh`. Without a username, those `gh` commands fall back awkwardly.

Ask:
- `GitHub username` — free text. *Required* if the user wants `gh`-based skills (most of them).
- `cruijff_kit project number` — leave the template default `3` unless the user has forked.

### 6. Weights & Biases

**Why this matters:** training runs log losses/metrics to wandb. Without these, runs still complete but you have no curves.

Ask:
- `Entity` (username or team). `[skip]` is allowed — runs will log offline.
- `Default project` — free text, e.g. `cruijff_kit`.

### 7. Common Paths

**Why this matters:** input datasets and output projects need a stable home that other skills can find via tokens like `{ck_data_directory}` and the `Projects` root.

#### Subsection: Pretrained Models

The `Location` should match what was set under HPC Environment > Shared Resources > Models directory. Confirm rather than ask again. The `Available models` list in the template enumerates ~7 example model names — don't ask the user to enumerate; suggest leaving the template list and adding/removing as they download new models.

#### Subsection: Working Directories

Auto-suggest based on group + username from earlier:
- `Projects` directory — `/scratch/gpfs/<group>/<username>/ck-projects/`. Confirm or override. This is where every experiment lives (one subdirectory per experiment, with run subdirs inside).

#### Subsection: Data

Auto-suggest:
- `{ck_data_directory}` — `/scratch/gpfs/<group>/<username>/ck-data/`. Confirm or override. This is where input datasets live, organized as `{ck_data_directory}/{project}/`.

### 8. Quick Commands

The template's "Quick Commands" section uses `<placeholder>` tokens (e.g. `<username>`, `<slurm_account>`) inside *example* command lines. **These are intentional** — they document the command shape rather than a specific user's invocation. Don't try to substitute them.

This is also why `validation.md`'s placeholder scan must restrict itself to the settings sections, not the whole file.

### 9. Compute Observability

Static guidance from the template. Copy through unchanged — no questions.

### 10. Notes

Static guidance from the template. Copy through unchanged — no questions.

## After all sections

1. Show the user a summary of what was filled in (fields they answered) vs what was skipped (placeholders that remain).
2. Confirm before writing. Ask: "ready to write `claude.local.md`?"
3. On confirm: write the file, log the action via `logging.md`'s `WRITE_CONFIG`, and point the user at `/design-experiment`.
4. On decline: log the cancellation, do not write, exit cleanly. The walkthrough's answers are *not* persisted between runs — re-running starts over.
