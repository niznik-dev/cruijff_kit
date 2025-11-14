# Future Skills Architecture Improvements

This document captures potential improvements to the skills architecture that could be experimented with in the future.

---

## Abstract SLURM Patterns (Potential Token Reduction: ~47%)

**Problem:** Job submission and monitoring logic is duplicated across torchtune and inspect modules.

**Current State:**
- `job_submission.md`: 184 total lines (75% duplicated)
- `monitoring.md`: 309 total lines (85% duplicated)
- Total: 493 lines

**Proposed Solution:**
Create `shared/slurm_operations.md` with reusable patterns:
- Job submission template (sbatch, capture ID, record metadata, error handling)
- Monitoring template (polling strategy, query commands, state change detection)
- Best practices (single squeue call, 60s interval, terminal states)
- Parameter tables showing tool-specific differences

Each tool's files would become ~25-30 lines of tool-specific configuration.

**Expected Result:**
- 493 → ~260 lines (47% reduction)
- Fix SLURM patterns once, applies to all tools
- Easier to add new tools (DSPy, custom frameworks)

**Implementation Sketch:**
```
shared/
├── logging_spec.md (exists)
└── slurm_operations.md (proposed)

run-experiment/
├── optimizers/torchtune/
│   ├── job_submission.md (references shared patterns + torchtune params)
│   └── monitoring.md (references shared patterns + torchtune status table)
└── evaluators/inspect/
    ├── job_submission.md (references shared patterns + inspect params)
    └── monitoring.md (references shared patterns + inspect status table)
```

**Risks:**
- Over-abstraction if tools have very different submission/monitoring needs
- Navigation overhead (jump to shared doc, then back to tool-specific)

**When to Consider:**
- When adding a third tool (DSPy, custom framework)
- When SLURM patterns need updates (currently requires changing 4 files)
- When duplication becomes a maintenance burden
