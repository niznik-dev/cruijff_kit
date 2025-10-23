# 🔴 Red Tier - No Access

**⚠️ EXPERIMENTAL FEATURE**: This data access tier has not been extensively tested. Researchers should avoid placing sensitive data here until this feature has been validated.

## Access Policy

**Claude Code MUST NOT access any files in this directory or its subdirectories.**

## Purpose

This directory contains highly sensitive data that requires strict privacy and security controls:

- Raw data with personally identifiable information (PII)
- Data under strict IRB protocols or ethical restrictions
- Proprietary datasets with legal restrictions
- Confidential information
- Any data with privacy, legal, or ethical constraints

## Claude Code Behavior

If asked to read, analyze, or process any file in `data/red/`:

1. **REFUSE** the request
2. Explain that files in `data/red/` are marked as highly sensitive
3. Suggest alternative approaches that don't require accessing this data
4. If the user insists, recommend they work with the data directly outside of Claude Code

## For Humans

Files placed in this directory will not be accessible to Claude Code. This is the appropriate location for:

- Original research data with identifiers
- Data covered by confidentiality agreements
- Any dataset you would not want an AI assistant to read

If you need Claude Code's help with data analysis, consider:
- Creating de-identified or synthetic versions in `data/yellow/` or `data/green/`
- Using aggregated or summary statistics
- Working with data schemas/metadata rather than actual data
