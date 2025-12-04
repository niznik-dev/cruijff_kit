# Parsing Tool Specifications - scaffold-experiment

This module handles reading and parsing the tool specifications from experiment_summary.yaml to determine which subagents to launch.

---

## Reading Tool Specifications

After verifying experiment_summary.yaml exists, read the "tools" section to identify which frameworks are being used:

**Expected format in experiment_summary.yaml:**
```yaml
tools:
  preparation: "torchtune"
  evaluation: "inspect-ai"
```

**Parsing logic:**
```python
import yaml
with open(f"{experiment_dir}/experiment_summary.yaml", 'r') as f:
    config = yaml.safe_load(f)

prep_tool = config['tools']['preparation']  # e.g., "torchtune"
eval_tool = config['tools']['evaluation']   # e.g., "inspect-ai"
```

---

## Tool to Subagent Mapping

Use the tool names to determine which subagent files to reference:

**Preparation tools:**
- `torchtune` → `scaffold-torchtune` subagent (see [optimizers/torchtune_agent.md](optimizers/torchtune_agent.md))

**Evaluation tools:**
- `inspect-ai` → `scaffold-inspect` subagent (see [evaluators/inspect_agent.md](evaluators/inspect_agent.md))

---

## Error Handling

**If "tools" section is missing:**
- Report error: YAML schema requires tools section
- Do not proceed with subagent launch
- Suggest user check experiment_summary.yaml format

**If tool name is not recognized:**
- Report error: "Tool '{tool_name}' is not supported"
- List supported tools: torchtune (preparation), inspect-ai (evaluation)
- Do not proceed with subagent launch

**If experiment_summary.yaml format is unexpected:**
- Report parsing error with details
- Show the problematic section if possible
- Suggest user verify YAML syntax

---

## Logging

Log the parsing process to the orchestration log:

```
[2025-10-27 14:30:00] READ_TOOL_SPECS: Parsing experiment_summary.yaml
Details: Found tools section
Result: Preparation=torchtune, Evaluation=inspect-ai
Explanation: Will launch scaffold-torchtune and scaffold-inspect subagents
```

**On error:**
```
[2025-10-27 14:30:00] READ_TOOL_SPECS: Failed to parse tool specifications
Details: Missing 'tools' section in experiment_summary.yaml
Result: FAILURE - Cannot determine which subagents to launch
Action: Suggest user verify experiment_summary.yaml format
```