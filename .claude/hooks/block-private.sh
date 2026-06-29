#!/usr/bin/env bash
# cruijff_kit privacy guard (PreToolUse hook).
#
# Denies any tool call that touches a private experiment folder. A folder is
# private iff it holds a .ck-private marker file at its root (the user drops it
# per the runbook's lock step; privatize-experiment stages the clone unlocked).
#
# The folder NAME is NOT a lock signal — a "private" in the name is a human label
# only, so the assistant can still stage, repoint, and symlink a private-named
# clone right up until the user locks it with the marker. The marker is the lock,
# nothing else.
#
# Bias is fail-safe: when a path sits under a marked folder, deny. Reads the
# tool-call JSON on stdin and, on a match, emits a PreToolUse deny decision. A
# hook deny overrides permission allow rules, which is what closes the
# Bash(cat:*)/grep/ls hole that ordinary permission globs cannot reach.
#
# Coverage note: the marker walk resolves *absolute* path tokens. A relative-path
# reference to a marked folder from within its parent is not yet caught (a planned
# enhancement resolves tokens against cwd); the marker still blocks every absolute
# access, which is how the runbook drives the workflow.
set -uo pipefail

input=$(cat)
tool=$(printf '%s' "$input" | jq -r '.tool_name // empty' 2>/dev/null)

case "$tool" in
  Bash)            mode=cmd;  hay=$(printf '%s' "$input" | jq -r '.tool_input.command // empty' 2>/dev/null) ;;
  Read|Edit|Write) mode=path; hay=$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty' 2>/dev/null) ;;
  NotebookEdit)    mode=path; hay=$(printf '%s' "$input" | jq -r '.tool_input.notebook_path // empty' 2>/dev/null) ;;
  Grep|Glob)       mode=path; hay=$(printf '%s' "$input" | jq -r '[.tool_input.path, .tool_input.glob] | map(select(. != null)) | join(" ")' 2>/dev/null) ;;
  *)               exit 0 ;;
esac

deny() {
  # $1 is a pre-quoted JSON string literal.
  printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":%s}}\n' "$1"
  exit 0
}

MARKER_MSG='"cruijff_kit privacy guard: target lives under a .ck-private folder. Reading private data is blocked by policy."'

# 1) Path fields only: a location that names a .ck-private marker is a direct
#    reference to a private folder — deny. Bash command TEXT is exempt, so
#    discussing, grepping, or committing the literal ".ck-private" string is not
#    blocked; actual access via an absolute path in a command is still caught by
#    the marker walk below.
if [ "$mode" = path ]; then
  printf '%s' "$hay" | grep -qE '\.ck-private' && deny "$MARKER_MSG"
fi

# 2) Marker walk: for any explicit absolute path token, deny if an ancestor holds
#    .ck-private. This is the lock — the folder name is not consulted.
for tok in $hay; do
  case "$tok" in /*) : ;; *) continue ;; esac
  p=$tok
  while [ -n "$p" ] && [ "$p" != "/" ]; do
    [ -e "$p/.ck-private" ] && deny "$MARKER_MSG"
    p=$(dirname "$p")
  done
done

exit 0
