# Next Steps for Issue #170 (run-inspect-viz)

**Status:** All 9 chunks complete and tested on `capitalization_model_size_comparison`

## Options for Completion

### 1. Create Pull Request
- Merge run-inspect-viz implementation into main
- Close Issue #170
- All functionality working and tested
- **Status:** Ready to create anytime

### 2. Test on Complete Experiments
- Test run-inspect-viz on larger, complete experiments
- Verify performance with more evaluation files
- Confirm visualizations work well with full datasets
- **Status:** Waiting for experiments to be completed
  - Issue #172: Complete cap_8L experiment (2 evals needed)
  - Issue #173: Complete cap_7L experiment (10 evals needed)

### 3. Update Documentation
- Document run-inspect-viz in ARCHITECTURE.md
- Add usage examples to README.md
- Could be done as part of broader documentation update
- **Status:** Optional, can be done later

### 4. Address Related Issues
- Issue #174: Model name capture issue (inspect-ai .eval metadata)
- Issue #175: Pre-built views rendering incorrectly (inspect-viz library)
- These are upstream issues, not blockers for run-inspect-viz
- **Status:** Can be addressed separately

## User Preference

**Selected:** Option 2 (test on complete experiments) once #172 and #173 are complete

## Timeline

1. Complete experiments #172 and #173
2. Test run-inspect-viz on those completed experiments
3. Create pull request to merge into main
