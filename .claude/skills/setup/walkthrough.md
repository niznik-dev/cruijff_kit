# Greenfield walkthrough

Step-by-step Q&A flow for first-time setup. Source of truth: `claude.local.md.template` at the repo root.

## Principles

- **One question at a time.** Don't dump a wall of prompts on the user. Each section gets a brief rationale, then the questions, then move on.
- **Auto-detect what you can.** `whoami`, `id -gn`, `pwd`, `which conda`, `which module` — these answer questions without asking.
- **Show defaults from the template.** When the template has a sensible default (e.g. `cruijff` for the conda env name, `1` for default GPUs), present it as `[default: cruijff]` and let the user accept by pressing enter.
- **Allow `[skip]`.** A user might not know their wandb entity yet. Mark the section with the unfilled placeholders and continue. Validate-mode will surface them later.
- **Never write secrets to claude.local.md.** No HF tokens, no SSH keys. The file is gitignored but accidents happen.

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

Walk the template top-to-bottom. The order below mirrors the template's section order; if the template grows new sections, walk them too.

### 1. HPC Environment

**Why this matters:** these paths are referenced by every other skill (scaffold-experiment, run-experiment, summarize-experiment). Wrong paths = nothing works.

Auto-detect (treat as a default to confirm, not a fact to write):
- `Username`: `whoami`
- `Group`: `id -gn` — note that the *directory* case for the group may differ from the token (e.g. directory `MSALGANIK`, group `msalganik`). When showing the auto-detected value, ask the user to confirm the casing that appears in their actual scratch path.
- `Working directory`: `pwd` (the cruijff_kit repo root the user is in)

Ask:
- `Cluster` (e.g. Della, Traverse) — name only, free text.
- `Documentation` URL — optional, can `[skip]`.
- Confirm scratch directory: `/scratch/gpfs/<group>/<username>` — show the auto-detected path, ask "is this where your scratch lives?". On most Princeton clusters yes; on others adjust.

### 2. Shared Resources

**Why this matters:** model weights are large (10s–100s of GB). Pointing at a *shared* directory means the group only downloads each model once.

Ask:
- `Models directory` — default `/scratch/gpfs/<group>/pretrained-llms`. If the user is solo on the cluster, they may use their own path. If they don't have downloaded models yet, `[skip]` is fine — they can fill it in when they download their first model.
- `Shared datasets` — optional, `[skip]` is normal.

### 3. SLURM Defaults

**Why this matters:** every fine-tune and eval is a SLURM job. Wrong account = job rejected; wrong constraint = job sits in the wrong queue.

Ask:
- `Account` — SLURM allocation name. The user gets this from their HPC admin or PI. *Required.*
- `Partition` — leave empty if cluster has no partitions.
- `Constraint` — common values: `gpu80` (80GB GPU), `gpu40` (40GB GPU), or empty.
- `Default time` — the template default `0:59:00` is sensible; under 1h hits Princeton's test queue.
- `Default GPUs` — `1` is sensible default.
- `Default conda environment` — default `cruijff`.

### 4. GPU Configuration

**Why this matters:** `design-experiment` and `scaffold-experiment` map model VRAM requirements to SLURM constraints. If GPU types are unknown, those skills fall back to conservative defaults that may waste allocation.

Ask:
- `Available GPU types` — e.g. "A100 80GB", "H100 80GB". Free text. The user usually knows; if not, `[skip]` and they can fill in later.
- `Full GPU constraint` — the SLURM constraint flag for a full dedicated GPU (e.g. `gpu80`).
- `Shared/MIG partition` — if applicable, e.g. `nomig`. Empty if not.

### 5. Module System

**Why this matters:** Some HPC clusters use environment modules; some don't. Skills check this section to decide whether to prepend `module load <anaconda_module>` to commands.

Ask:
- Does the cluster use `module`? Auto-detect: `command -v module && echo yes || echo no`.
- If yes: ask the user for their anaconda module name. The template shows `anaconda3/2025.6` as a *placeholder example* (the version that worked on Della at the time of writing) — do not assume it's right for the user's cluster. They can find theirs with `module avail anaconda` (or by asking their HPC admin).
- If no: skip this section in the output (the template says "remove this section" in that case).

### 6. Conda Environment

**Why this matters:** every skill activates this env before running any Python.

Ask:
- `Environment name` — default `cruijff`. Matches what was set in SLURM Defaults.
- Auto-check: `conda env list | grep -q <env_name>` — if not present, gently note "this env doesn't exist yet; run `make install` from the repo root before using `/design-experiment`."

### 7. GitHub Configuration

**Why this matters:** issue creation and project board updates use `gh`. Without a username, those gh commands fall back awkwardly.

Ask:
- `GitHub username` — free text. *Required* if the user wants `gh`-based skills (most of them).
- `cruijff_kit project number` — leave the template default `3` unless the user has forked.

### 8. Weights & Biases

**Why this matters:** training runs log losses/metrics to wandb. Without these, runs still complete but you have no curves.

Ask:
- `Entity` (username or team). `[skip]` is allowed — runs will log offline.
- `Default project` — free text, e.g. `cruijff_kit`.

### 9. Common Paths

**Why this matters:** input datasets and output projects need a stable home that other skills can find via the `{ck_data_dir}` token.

Auto-suggest based on group + username from earlier:
- `Pretrained Models` location — `/scratch/gpfs/<group>/pretrained-llms/`. Confirm or override.
- `Projects` directory — `/scratch/gpfs/<group>/<username>/ck-projects/`. Confirm or override.
- `{ck_data_dir}` — `/scratch/gpfs/<group>/<username>/ck-data/`. Confirm or override.

For `Available models` under Pretrained Models — the template lists 7 example model names. Don't ask the user to enumerate — they may not have all of them. Suggest leaving the template list and adding/removing as they download models.

### 10. Quick Commands

The template's "Quick Commands" section uses `<placeholder>` tokens (e.g. `<username>`, `<slurm_account>`) inside *example* command lines. **These are intentional** — they document the command shape rather than a specific user's invocation. Don't try to substitute them.

This is also why `validation.md`'s placeholder scan must restrict itself to the settings sections, not the whole file.

### 11. Compute Observability + Notes

Both sections are static guidance from the template. Copy them through unchanged.

## After all sections

1. Show the user a summary of what was filled in (fields they answered) vs what was skipped (placeholders that remain).
2. Confirm before writing. Ask: "ready to write `claude.local.md`?"
3. On confirm: write the file, log the action via `logging.md`'s `WRITE_CONFIG`, and point the user at `/design-experiment`.
4. On decline: log the cancellation, do not write, exit cleanly. The walkthrough's answers are *not* persisted between runs — re-running starts over.
