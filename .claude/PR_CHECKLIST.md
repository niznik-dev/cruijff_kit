# Pre-PR Checklist

This checklist helps ensure nothing gets forgotten before creating a pull request. Not mandatory, just helpful reminders for common oversights.

## Documentation Updates

Consider whether these files need updates based on your changes:

### Core Documentation
- [ ] **README.md** - Public-facing features, installation, quick start
- [ ] **ARCHITECTURE.md** - System design, data flow, major architectural decisions
- [ ] **SKILLS_ARCHITECTURE_SUMMARY.md** - Skill relationships and workflows
- [ ] **CLAUDE.md** - Workflow instructions, terminology, git process

### Component Documentation
- [ ] **Skill docs** (`.claude/skills/*/`) - If you modified skill behavior or added submodules
- [ ] **Experiment READMEs** (`experiments/*/README.md`) - If you changed experiment structure
- [ ] **Tool docs** (`tools/*/`) - If you modified torchtune recipes, inspect tasks, etc.

### Common Scenarios
- **Added/removed a skill?** → Update SKILLS_ARCHITECTURE_SUMMARY.md
- **Changed config structure?** → Update ARCHITECTURE.md and relevant READMEs
- **Modified workflow?** → Update CLAUDE.md and skill documentation
- **Added CLI commands?** → Update README.md usage section

## Code Quality

- [ ] **Remove debug code** - No leftover print statements, breakpoints, or test data
- [ ] **Clean up TODOs** - Either implement them or create issues for them
- [ ] **Check imports** - Remove unused imports, verify relative imports are correct
- [ ] **Verify paths** - Ensure no hardcoded paths (use claude.local.md patterns)

## Testing

- [ ] **Run relevant tests** - If tests exist for modified code, run them
- [ ] **Manual testing** - For skills/workflows, try a quick end-to-end test
- [ ] **Check dependencies** - If you added imports, note them in PR description

## Git Hygiene

- [ ] **Branch up to date?** - `git fetch origin main && git log HEAD..origin/main` (should be empty)
- [ ] **Clean commit history?** - Commits are logical and well-described
- [ ] **No merge conflicts?** - Rebase/merge main if needed

## When to Consult This Checklist

**Claude Code should reference this checklist when:**
- User asks to create a PR
- User says "ready to merge" or similar
- Before running git commit for PR-sized changes

**Users should reference this checklist when:**
- About to create a PR manually
- Reviewing Claude's PR draft before submission
- Want to double-check nothing was forgotten
