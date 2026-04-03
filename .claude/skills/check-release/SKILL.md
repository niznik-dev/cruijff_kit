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

4. List open PRs and present them for review — each should be merged, deferred, or closed before release:
   ```bash
   module load anaconda3/2025.6 && conda activate cruijff && gh pr list --repo niznik-dev/cruijff_kit --state open --json number,title
   ```

5. Check for anything already in `[Unreleased]` in `CHANGELOG.md`.

6. Check KNOWN_ISSUES.md for stale entries (closed issues, resolved problems).

7. Present a summary to the user:
   - Number of commits and PRs since last release
   - Brief description of changes (group by: features, fixes, docs, cleanup)
   - Any breaking changes (flag explicitly)
   - Open PR disposition (merge/defer/close)
   - Any stale KNOWN_ISSUES entries to clean up

8. **Decision point**: Ask the user whether to proceed with a release or stop here.

### Step 2: Version bump

1. Read the current version from `pyproject.toml` (line 7).

2. Default to a **patch** bump. The project versioning policy:
   - **Patch** (default): New skills, bug fixes, docs, CI, refactors — anything that doesn't break existing workflows
   - **Minor**: Significant milestones worth announcing to collaborators — major new capabilities or architectural shifts
   - **Major**: Breaking changes to user-facing interfaces. Not expected before 1.0.
   
   When in doubt, it's a patch.

3. **Decision point**: Confirm the version number with the user.

### Step 3: Changelog

1. Draft a changelog entry using the [Keep a Changelog](https://keepachangelog.com/) format already in `CHANGELOG.md`:
   - **Added** — uses domain sub-headings: "Skills & Workflows", "Evaluation & Metrics", "Observability & Testing", "Documentation & Data". Omit any sub-heading with no entries.
   - **Changed** — flat list, no sub-headings
   - **Fixed** — flat list, no sub-headings
   - **Deprecated** / **Removed** — flat list, include only if applicable

2. Include PR numbers and contributor attribution (e.g., `@username`) where applicable.

3. Present the draft to the user for review.

4. **Decision point**: User approves or edits the changelog entry.

5. Once approved:
   - Move items from `[Unreleased]` (if any) into the new version section
   - Add the new version entry below `[Unreleased]` in `CHANGELOG.md`
   - Update version in `pyproject.toml`
   - Clean up KNOWN_ISSUES.md if needed

### Step 4: Branch, PR, and merge

1. Create a release branch:
   ```bash
   git checkout -b release-<VERSION>
   ```

2. Commit changelog + version bump + any KNOWN_ISSUES changes.

3. Push branch and open PR:
   ```bash
   git push -u origin release-<VERSION>
   ```

4. **Decision point**: Confirm PR is ready. Wait for CI to pass.

5. User merges the PR using **merge commit** (not squash or rebase).

### Step 5: Tag and release

1. Pull the merge commit:
   ```bash
   git checkout main && git pull
   ```

2. Create annotated tag on the merge commit with a minimal message:
   ```bash
   git tag -a v<VERSION> -m "v<VERSION>"
   ```

3. Push tag:
   ```bash
   git push origin v<VERSION>
   ```

4. Create GitHub Release with the changelog entry as release notes:
   ```bash
   gh release create v<VERSION> --title "v<VERSION>" --notes "<changelog entry>"
   ```

5. Confirm success by printing the release URL.

## Important Notes

- Every step pauses for user input — the user can bail at any checkpoint
- Do NOT skip decision points or auto-proceed
- The wiki [Release Process](https://github.com/niznik-dev/cruijff_kit/wiki/Release-Process) page is the canonical checklist — do not duplicate its steps here
- Keep changelog entries concise — one line per change, PR number, contributor if not Mattie/Claude
- Tag messages are minimal (`"vX.Y.Z"` only) — GitHub Releases carry the full notes
- Always tag **after** merging so the tag points at the merge commit on `main`
