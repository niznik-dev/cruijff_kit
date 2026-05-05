# Logging

See [shared/logging_spec.md](../shared/logging_spec.md) for the format and general guidelines.

## Log file location

```
logs/setup.log
```

Relative to the cruijff_kit repo root (where `claude.local.md` lives). If `logs/` doesn't exist, create it. Append on re-runs (don't truncate) so the audit trail across multiple validate-mode runs is preserved.

## Action types

| Type | When | Required fields |
|------|------|-----------------|
| `DETECT_MODE` | Once at start | Mode (greenfield / validate / restart), presence of template + claude.local.md |
| `RENAME_BACKUP` | "Start over": before destroying state | Original path, backup path, collision counter (if any) |
| `READ_TEMPLATE` | Greenfield: after loading template | Template path, line count |
| `READ_EXISTING` | Validate: after loading current claude.local.md | File path, byte size, last-modified date |
| `AUTODETECT` | Each auto-detected value | Field, value, source command |
| `COLLECT_VALUE` | Greenfield: each user-supplied answer | Field, value (or `[skip]`) |
| `WRITE_CONFIG` | Greenfield: after writing claude.local.md | File path, byte size, sections written / sections skipped |
| `VALIDATE_PLACEHOLDER` | Validate: per match | Line number, section, field name |
| `VALIDATE_FIELD` | Validate: per required-field check | Field name, status (present/missing/placeholder) |
| `VALIDATE_ENV` | Validate: per env probe | Probe name, status (pass/fail/skipped), output |
| `WARN` | Non-fatal anomalies (skipped section, optional probe failed) | What was missing, what was substituted |
| `ERROR` | Fatal errors (template missing, write refused, no shell tools) | What failed, what the user should do |

## Example — greenfield run

```
[2026-05-04 17:12:01] DETECT_MODE: greenfield
Details: claude.local.md absent; claude.local.md.template present (158 lines)

[2026-05-04 17:12:01] READ_TEMPLATE: claude.local.md.template
Details: 158 lines, 11 sections

[2026-05-04 17:12:02] AUTODETECT: HPC username
Source: whoami
Result: jc0425

[2026-05-04 17:12:03] AUTODETECT: HPC group
Source: id -gn
Result: cruyff

[2026-05-04 17:12:35] COLLECT_VALUE: SLURM Account
Result: cruyff-prod

[2026-05-04 17:13:18] COLLECT_VALUE: Weights & Biases entity
Result: [skip]

[2026-05-04 17:13:42] WRITE_CONFIG: claude.local.md
Details: 162 lines, 9/11 sections filled, 2 sections skipped (W&B, Shared datasets)
```

## Example — validate run

```
[2026-05-04 17:20:08] DETECT_MODE: validate
Details: claude.local.md present (4321 bytes, modified 2026-04-30)

[2026-05-04 17:20:08] READ_EXISTING: claude.local.md
Details: 162 lines

[2026-05-04 17:20:08] VALIDATE_PLACEHOLDER: scan
Result: 0 unfilled placeholders in settings sections

[2026-05-04 17:20:08] VALIDATE_FIELD: Cluster
Result: present (Della)

[2026-05-04 17:20:08] VALIDATE_FIELD: SLURM Account
Result: present (cruyff-prod)

[2026-05-04 17:20:08] VALIDATE_ENV: conda env exists
Probe: conda env list | grep -qx cruijff
Result: pass

[2026-05-04 17:20:08] VALIDATE_ENV: scratch dir writable
Probe: test -w /scratch/gpfs/cruyff/jc0425
Result: pass

[2026-05-04 17:20:09] VALIDATE_ENV: gh on PATH
Probe: which gh
Result: fail (not found until conda activate)
Severity: WARN

Summary: 5 GREEN, 1 YELLOW, 0 RED — ready for /design-experiment
```
