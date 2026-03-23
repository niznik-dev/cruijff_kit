# Logging: archive-experiment

See [shared/logging_spec.md](../shared/logging_spec.md) for format and general guidelines.

## Log File Location

```
{archive_dir}/archive.log
```

Note: Unlike other skills, this log lives in the **archive** directory (not the experiment directory) because the experiment directory is deleted during archiving.

## Action Types

| Action Type | When | Details to Include |
|-------------|------|--------------------|
| `LOCATE_EXPERIMENT` | Found experiment directory | Path, experiment name |
| `PARSE_SUMMARY` | Read experiment_summary.yaml | Run count, output dir |
| `CHECK_COMPLETENESS` | Verified run completion status | Complete/incomplete run names |
| `INVENTORY` | Catalogued keep/delete files | File counts, sizes |
| `RESOLVE_FINDINGS` | Determined findings.md source | Source file path or "none" |
| `COPY_ARTIFACT` | Copied file to archive | Source → destination path |
| `VERIFY_ARCHIVE` | Confirmed archive integrity | Pass/fail, any mismatches |
| `DELETE_ORIGINALS` | Removed source directories | Paths deleted, space freed |
| `COMPLETE` | Archive finished | Total kept/freed sizes |

## Example Log Entries

```
[2026-03-23 14:00:00] LOCATE_EXPERIMENT: Found experiment at /scratch/.../ck-experiments/cap_lora_2026-03-01
Details: Experiment name: cap_lora_2026-03-01, 2 runs defined

[2026-03-23 14:00:01] CHECK_COMPLETENESS: All runs complete
Details: run_rank4: eval logs present, run_rank8: eval logs present
Result: 2/2 runs complete

[2026-03-23 14:00:02] INVENTORY: Catalogued experiment artifacts
Details: Keep 14 files (2.3 MB), Delete 47 files (15234.7 MB)
Result: Ready to archive

[2026-03-23 14:00:02] RESOLVE_FINDINGS: Using analysis/report.md as findings source
Details: No findings.md found, falling back to analysis/report.md

[2026-03-23 14:00:05] VERIFY_ARCHIVE: Archive integrity check passed
Details: 14/14 files verified, all sizes match

[2026-03-23 14:00:10] DELETE_ORIGINALS: Removed experiment and output directories
Details: Deleted /scratch/.../ck-experiments/cap_lora_2026-03-01, /scratch/.../ck-outputs/cap_lora_2026-03-01/ck-out-run_rank4, /scratch/.../ck-outputs/cap_lora_2026-03-01/ck-out-run_rank8
Result: Freed 15234.7 MB

[2026-03-23 14:00:10] COMPLETE: Archive finished
Details: Archived cap_lora_2026-03-01 → /scratch/.../ck-archive/cap_lora_2026-03-01 (2.3 MB kept, 15234.7 MB freed)
```
