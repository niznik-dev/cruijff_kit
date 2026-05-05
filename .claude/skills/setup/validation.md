# Validate-existing mode

Health check for an existing `claude.local.md`. Three classes of check; each contributes findings to a structured report.

## Principles

- **Lightweight only.** No model loads, no SLURM submissions, no external API calls. Every check should run in well under one second on a login node.
- **Partial pass is normal.** A user may have skipped wandb on first setup; that's not "broken." The report shows green/yellow/red per check, not a binary verdict.
- **Concrete remediation per fail.** "Conda env `cruijff` not found → run `make install` from the repo root" beats "environment problem detected."
- **Never modify `claude.local.md`.** Validation is read-only. If the user wants to change something, point them at greenfield mode after backing up the old file (see SKILL.md).

## Class 1: Placeholder scan

Look for unreplaced template tokens (`<word>`-shaped placeholders) in the **settings sections only**:

- `## HPC Environment`
- `### Shared Resources`
- `## SLURM Defaults`
- `### GPU Configuration`
- `## Conda Environment Setup`
- `## GitHub Configuration`
- `## Weights & Biases`
- `## Common Paths`

**Skip the `## Quick Commands` section entirely.** Tokens like `<username>`, `<slurm_account>`, `<job_id>` in the example commands are *intentional* — they document the command shape. Flagging them as unfilled is a false positive.

Implementation sketch — stop reading at `## Quick Commands` (the placeholder-rich section we want to skip), since post-Quick-Commands sections (Compute Observability, Notes) don't have configurable placeholders:

```bash
awk '/^## Quick Commands/ {exit} {print NR": "$0}' claude.local.md \
  | grep -E '<[A-Za-z_][A-Za-z0-9_]*>' | head -20
```

If a future template revision puts placeholders *after* Quick Commands, replace this with an explicit allowlist of section headers to scan.

For each match, report:

```
[YELLOW] line N: <field_name> still placeholder
  Section: ## SLURM Defaults
  Fix: edit claude.local.md and replace <your_slurm_account> with your actual SLURM account
```

## Class 2: Required-field check

A small set of fields must be filled for any of the workflow skills to function. Confirm each is present *and* non-placeholder:

| Field | Section | Why required |
|---|---|---|
| `Cluster` | HPC Environment | scaffold-experiment uses to pick template paths |
| `Username` | HPC Environment | every output path includes this |
| `Group` | HPC Environment | shared model directory uses this |
| `Scratch directory` | HPC Environment | run outputs go here |
| `Account` | SLURM Defaults | every job submission needs an allocation |
| `Default conda environment` | SLURM Defaults | every job activates this env |

For each required field, parse the corresponding line out of `claude.local.md`. If missing or still placeholder, mark RED (not yellow):

```
[RED] missing required field: SLURM Account
  Without this, every fine-tune and eval job will be rejected at submission.
  Fix: edit claude.local.md, locate "## SLURM Defaults", replace <your_slurm_account> with your allocation name.
```

## Class 3: Environment probes

Light, fast checks that compare the *config file's claims* to *what actually exists on the system*. Run only after Class 1 and Class 2 pass — if the config says nothing about the conda env, there's nothing to probe.

### Probe a: conda env exists

On clusters that gate `conda` behind a module load, a bare `conda` call returns "command not found." Gate the probe accordingly — read the module name from the `## Module System` section of `claude.local.md` (do **not** hardcode a version like `anaconda3/2025.6`; it differs by cluster and by month).

```bash
# Pseudo-code: load the module first if the config calls for one
if claude.local.md has "module load <module_name>":
  module load <module_name>
conda env list 2>/dev/null | awk '{print $1}' | grep -qx "<env_name>"
```

If the named env doesn't exist:

```
[RED] conda env "cruijff" not found
  Fix: from the repo root, run `make install` (or `make install-dev` for development).
```

If `conda` itself is unavailable (no module to load, or load failed):

```
[RED] conda command not available
  Fix: confirm anaconda is installed on this system. If your cluster uses modules,
       set the correct anaconda module name in claude.local.md's "## Module System" section.
```

### Probe b: scratch directory exists and is writable

```bash
test -d "<scratch_dir>" && test -w "<scratch_dir>"
```

If missing:

```
[RED] scratch directory /scratch/gpfs/MSALGANIK/<user> does not exist
  Fix: confirm the path is correct for your cluster; ask your HPC admin if the group/user prefix is wrong.
```

If exists but not writable: same shape, "exists but not writable — check permissions."

### Probe c: gh on PATH

```bash
which gh
```

Common case: `gh` is provided by the conda env, so it isn't on PATH until activation. If `which gh` fails, retry after `conda activate <env_name>`. If still missing:

```
[YELLOW] gh CLI not found, even after activating <env_name>
  Several skills use gh for issue/PR creation. Without it, those flows fall back to manual.
  Fix: add gh to your conda env (`conda install gh -n <env_name>`) or rely on plain git.
```

### Probe d (optional): module system

Only if `claude.local.md` mentions `module load`:

```bash
command -v module
```

If `module` isn't found but the config says to use it, the load lines in skill commands will fail silently:

```
[YELLOW] config references `module load` but `module` is not available on this system
  Fix: either remove the Module System section if your cluster doesn't use modules,
       or confirm you're on a login node (modules are sometimes login-only).
```

## Reporting format

Aggregate all findings into a single user-visible report. Suggested format:

```
Setup health check — claude.local.md

Placeholder scan:    [GREEN] no unfilled placeholders
Required fields:     [GREEN] all 6 required fields present
Conda env:           [GREEN] cruijff found
Scratch directory:   [GREEN] /scratch/gpfs/MSALGANIK/jc0425 exists, writable
gh on PATH:          [YELLOW] not found until conda activate
Module system:       [GREEN] module command available

Overall: ready to design an experiment. Run /design-experiment when ready.
```

If any RED findings:

```
Overall: 1 blocker. Fix the RED items above, then re-run /setup.
```

## Edge cases

- **`claude.local.md` exists but is empty** (e.g. user touched it by accident): treat as if greenfield, prompt to back up + restart. Don't silently overwrite.
- **`claude.local.md` is a symlink** (e.g. user is sharing across machines): follow the link and validate the target. Don't mention the symlink unless it points at something missing.
- **`claude.local.md.template` itself is missing** (corrupted clone): instruct the user to `git checkout claude.local.md.template` from the repo. The skill can't validate against a template it can't read.
