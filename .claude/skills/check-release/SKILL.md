---
name: check-release
description: Weekly release check — review changes since last tag and optionally cut a new release.
---

# Check Release

Walk through the release process incrementally, with decision points at each stage. Designed for weekly invocation but doesn't assume a release will happen.

## Prerequisites

- On `main` branch with a clean working tree
- GitHub CLI available (requires conda environment)

## Workflow

### Step 0: Show the release checklist

1. Read the wiki release process page:
   ```bash
   cat /scratch/gpfs/MSALGANIK/niznik/GitHub/cruijff_kit_wiki/Release-Process.md
   ```

2. Present the checklist to the user so everyone can see the full process before starting.

3. Proceed to Step 1.

### Step 1: What's changed?

1. Find the latest git tag:
   ```bash
   git describe --tags --abbrev=0
   ```

2. List commits since that tag:
   ```bash
   git log <latest_tag>..HEAD --oneline
   ```

3. List PRs merged since that tag:
   ```bash
   module load anaconda3/2025.6 && conda activate cruijff && gh pr list --repo niznik-dev/cruijff_kit --state merged --search "merged:>=$(git log -1 --format=%ai <latest_tag> | cut -d' ' -f1)" --json number,title,mergedAt --limit 50
   ```

4. Check for anything already in `[Unreleased]` in `CHANGELOG.md`.

5. Present a summary to the user:
   - Number of commits and PRs since last release
   - Brief description of changes (group by: features, fixes, docs, cleanup)
   - Any breaking changes (flag explicitly)

6. **Decision point**: Ask the user whether to proceed with a release or stop here.

### Step 2: Version bump

1. Read the current version from `pyproject.toml` (line 7).

2. Suggest a semver bump based on change scope:
   - **Patch** (0.2.0 -> 0.2.1): Bug fixes, docs, minor cleanup
   - **Minor** (0.2.0 -> 0.3.0): New features, new skills, notable enhancements

3. **Decision point**: Confirm the version number with the user.

### Step 3: Changelog

1. Draft a changelog entry using the [Keep a Changelog](https://keepachangelog.com/) format already in `CHANGELOG.md`. Use the existing categories:
   - **Added** (with sub-sections like "Skills & Workflows", "Evaluation & Metrics", etc. as needed)
   - **Changed**
   - **Fixed**
   - **Deprecated**
   - **Removed**

2. Include PR numbers and contributor attribution (e.g., `@username`) where applicable.

3. Present the draft to the user for review.

4. **Decision point**: User approves or edits the changelog entry.

5. Once approved:
   - Move items from `[Unreleased]` (if any) into the new version section
   - Add the new version entry below `[Unreleased]` in `CHANGELOG.md`

### Step 4: Execute the release checklist

Walk through the remaining items on the wiki checklist (loaded in Step 0) one at a time. For each item, execute it and confirm with the user before moving to the next.

**Decision point**: Pause before any irreversible action (push, tag, GitHub release) and confirm with the user.

After the final step, confirm success by printing the release URL.

## Important Notes

- Every step pauses for user input — the user can bail at any checkpoint
- Do NOT skip decision points or auto-proceed
- The wiki [Release Process](https://github.com/niznik-dev/cruijff_kit/wiki/Release-Process) page is the canonical checklist — do not duplicate its steps here
- Keep changelog entries concise — one line per change, PR number, contributor if not Mattie/Claude
