---
name: setup
description: Interactive setup and validation skill for new and returning cruijff_kit users. Walks a new user through creating their `claude.local.md` from the template (HPC paths, SLURM defaults, conda env, model locations, wandb config), or — if a `claude.local.md` already exists — runs a lightweight health check that flags placeholders, missing required fields, and basic environment problems (conda env missing, scratch directory unreachable, `gh` not on PATH). Use whenever the user says they're new, asks how to get started, says their environment isn't working, or asks "what should I run first?" Phrases that should trigger this skill: "set me up", "I'm new to cruijff_kit", "validate my setup", "validate my config", "is my environment ready?", "is my environment configured correctly?", "first-time setup", "what do I need to configure?", "health check", "redo my config", "start over with claude.local.md".
---

# Setup

You help users get a working cruijff_kit environment. There are two modes — pick based on whether `claude.local.md` already exists:

- **Greenfield** — no `claude.local.md`. Walk the user through `claude.local.md.template` interactively, explaining what each section is for, then write a populated `claude.local.md`.
- **Validate** — `claude.local.md` exists. Read it, scan for unresolved placeholders, check that required fields are filled, run a few lightweight environment probes, and report a green/yellow/red summary with concrete next actions.

Default to **never overwriting an existing `claude.local.md`**. If one is present, validate-mode is the only option unless the user *explicitly* asks to start over (in which case follow the "Start over" procedure in `walkthrough.md`, which renames the existing file to a dated `.bak` and then enters greenfield mode).

## Audience

The user may be one of two profiles. Use what they say (and what their environment looks like) to figure out which:

- **New user** — has just cloned the repo, hasn't configured anything, may not know what cruijff_kit does. They benefit from rationale ("this matters because…") on each setting, not just a question.
- **Returning user** — has configured their environment before, may have just pulled an update that added a field, or is debugging a workflow that's failing. They want a fast "is anything broken?" check, not a tutorial.

Mode detection above maps roughly: greenfield ↔ new user, validate ↔ returning user. But not always — a returning user on a fresh clone is also greenfield. Don't assume.

## Your task

1. Detect mode (greenfield vs validate) by checking for `claude.local.md`.
2. **Greenfield**: walk through the template section-by-section, prompt the user for each value, write `claude.local.md` at the repo root.
3. **Validate**: read the existing file, scan for placeholders and missing fields, run environment probes, report findings.
4. Point the user at next steps (`docs/PREREQUISITES.md` for accounts and Software Carpentry tutorials; `/design-experiment` for their first experiment).
5. Log the run to `logs/setup.log`.

See the modular sub-files for details:

- `walkthrough.md` — greenfield Q&A flow, section-by-section
- `validation.md` — placeholder scan, required-field check, env probes
- `logging.md` — action types and log file location

## Prerequisites

This skill is *itself* the prerequisite-check for other skills, so its own prereqs are minimal:

- The user has cloned the cruijff_kit repo and is in its root directory (where `claude.local.md.template` lives).
- A shell with `bash`, `grep`, and standard Unix tools.

If `claude.local.md.template` is missing, the user is either not in the repo root or has an unusual checkout — stop and ask before proceeding.

## Workflow

### 1. Detect mode

```bash
ls -la claude.local.md claude.local.md.template
```

- **Both present** → validate mode.
- **Template only** → greenfield mode.
- **Template missing** → stop. Ask the user where they are; the skill assumes the repo root.

### 2a. Greenfield → `walkthrough.md`

The template is 158 lines and covers HPC, SLURM, conda, GitHub, wandb, paths, and quick commands. Walk it section-by-section. For each section:

1. Show the section name and a one-line "this matters because…" rationale.
2. Read the template's placeholder values. Auto-detect what you can (`whoami`, `id -gn`, `pwd`). Treat auto-detected values as **defaults to confirm**, not facts to write. In particular, `id -gn` returns the user's primary group token, but on some clusters the directory case differs from the group token (e.g. `MSALGANIK` directory vs `msalganik` group); ask the user to confirm the casing that appears in their actual paths.
3. Ask the user for non-detectable values one at a time. Don't ask for everything at once — that's a wall of questions, not a walkthrough.
4. Allow `[skip]` answers — the user may want to come back to that section later. Skipped sections keep their `<placeholder>` values; validate-mode will catch them on the next run.

When all sections are done, write the populated content to `claude.local.md` at the repo root.

### 2b. Validate → `validation.md`

Three classes of check:

1. **Placeholder scan**: `grep -n '<[^>]*>' claude.local.md` in the *settings sections only* (HPC Environment, SLURM Defaults, Common Paths, Weights & Biases, GitHub Configuration). Lines under "Quick Commands" legitimately contain `<placeholder>` tokens — don't flag those.
2. **Required-field check**: confirm a small set of must-have values are non-empty and non-placeholder: `Cluster`, `Username`, `Group`, `Scratch directory`, `Default conda environment`, `Account` (SLURM).
3. **Env probes** (lightweight only — don't actually run jobs or hit external APIs):
   - Conda env exists: `conda env list | grep -q <env_name>`
   - Scratch directory exists and is writable: `test -w <scratch_dir>`
   - `gh` is on PATH after activating the documented conda env: `which gh`
   - Module system available if the file mentions `module load`: `command -v module`

Report a per-check pass/fail with the concrete remediation step beside each fail. Don't return a binary "all good / all bad" — partial-pass is normal and useful.

### 3. Point at next steps

After either mode, finish with two pointers:

- `docs/PREREQUISITES.md` — accounts and skills the user should have (HuggingFace token, Git/shell/conda comfort). The skill **does not validate accounts** — don't try to curl HF with the user's token, that's a credential-leak risk and brittle.
- `/design-experiment` — the natural next skill once the environment is configured.

## Acceptance / validation contract

A successful run satisfies all of:

- ✓ Either `claude.local.md` was created (greenfield) or a structured validation report was produced (validate).
- ✓ The user knows what the next step is (`/design-experiment` or "fix these N findings first").
- ✓ `logs/setup.log` was appended with the action types from `logging.md`.
- ✓ No existing `claude.local.md` was overwritten without explicit user consent.

If any of these fail, the skill should error loud, not silently produce something half-correct.

## Out of scope (explicitly)

- Validating HuggingFace tokens, Princeton SSO, or any other authenticated external service. Point at `docs/PREREQUISITES.md` instead.
- Running fine-tuning, eval, or any GPU work as a "smoke test." That's the job of `workflow_test.yaml` (see CLAUDE.md), not first-time setup.
- Splitting `claude.local.md` into Claude-specific and tool-agnostic config files. That's tracked separately as #457.
- Auto-installing the conda environment. The user runs `make install` per `claude.local.md.template` — the skill verifies the env *exists*, doesn't create it.
- Migrating an old `claude.local.md` to a new schema. If the template gains fields, validate-mode will flag the missing ones; the user fills them in by hand.
