# 🟡 Yellow Tier - Permission Required

**⚠️ EXPERIMENTAL FEATURE**: This data access tier has not been extensively tested. Use with caution and verify Claude Code's behavior when accessing files here.

## Access Policy

**Claude Code MAY access files in this directory ONLY with explicit permission.**

## Purpose

This directory contains research data with moderate privacy considerations:

- De-identified social science data
- Datasets with usage agreements or restrictions
- Research collaborator datasets
- Data that requires careful handling but isn't strictly confidential
- Datasets where access should be granted on a case-by-case basis

## Claude Code Behavior

Before reading any file in `data/yellow/`:

1. **ASK** the user for explicit permission to access that specific file or dataset
2. Explain what you need to do with the data
3. Wait for clear confirmation before proceeding
4. If permission is denied, suggest alternative approaches

## Per-Dataset Permissions

Each dataset in this directory should include a `README.md` or `PERMISSIONS.md` file that documents:

- What the dataset contains
- Source and any usage restrictions
- Whether Claude Code has standing permission to access it
- Any special handling requirements

Example structure:
```
data/yellow/
├── CLAUDE.md (this file)
├── study_a/
│   ├── README.md (documents the dataset and permissions)
│   └── data.csv
└── study_b/
    ├── PERMISSIONS.md (explicitly grants or denies access)
    └── responses.jsonl
```

## For Humans

Files in this directory require you to grant permission before Claude Code can access them. This is appropriate for:

- De-identified research data
- Datasets with moderate sensitivity
- Data you want to review before Claude processes it

To grant standing permission for a dataset, create a `README.md` or `PERMISSIONS.md` file in the dataset directory that explicitly states Claude Code may access the files.
