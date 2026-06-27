#!/usr/bin/env bash
# cruijff_kit privacy guard (PreToolUse hook).
#
# Denies any tool call that touches a private experiment folder. Private folders
# are marked two independent ways so the guard survives a rename:
#   1. the _private_<date> naming convention (e.g. ggs_ever_kid_private_2026-06-25), and
#   2. a .ck-private marker file at the folder root (written by privatize-experiment).
#
# Bias is fail-safe: when a path looks private, deny. Reads the tool-call JSON on
# stdin and, on a match, emits a PreToolUse deny decision. A hook deny overrides
# permission allow rules, which is what closes the Bash(cat:*)/grep/ls hole that
# the permissions.deny globs cannot reach.
#
# Precision note: for real path fields (Read/Edit/Write/Glob) the convention match
# is blunt — those carry a location, never prose. For Bash command text we match
# only a private-folder *reference* (the _private_<date> form, or _private_ inside
# a slash-bearing token), so discussing or grepping the bare string "_private_"
# is not blocked. Grep's search pattern is deliberately ignored (it is a query,
# not a location).
set -uo pipefail

input=$(cat)
tool=$(printf '%s' "$input" | jq -r '.tool_name // empty' 2>/dev/null)

case "$tool" in
  Bash)         mode=cmd;  hay=$(printf '%s' "$input" | jq -r '.tool_input.command // empty' 2>/dev/null) ;;
  Read|Edit|Write) mode=path; hay=$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty' 2>/dev/null) ;;
  NotebookEdit) mode=path; hay=$(printf '%s' "$input" | jq -r '.tool_input.notebook_path // empty' 2>/dev/null) ;;
  Grep|Glob)    mode=path; hay=$(printf '%s' "$input" | jq -r '[.tool_input.path, .tool_input.glob] | map(select(. != null)) | join(" ")' 2>/dev/null) ;;
  *)            exit 0 ;;
esac

deny() {
  # $1 is a pre-quoted JSON string literal.
  printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":%s}}\n' "$1"
  exit 0
}

NAME_MSG='"cruijff_kit privacy guard: this references a private experiment folder. Private microdata must never be read by Claude. See the folder'"'"'s .ck-private marker and private_data_runbook.md."'
MARKER_MSG='"cruijff_kit privacy guard: target lives under a .ck-private folder. Reading private data is blocked by policy."'

# 1) Naming-convention match.
if [ "$mode" = path ]; then
  # Real location field — blunt match on the convention and the marker name.
  printf '%s' "$hay" | grep -qE '_private_|\.ck-private' && deny "$NAME_MSG"
else
  # Bash command text — only a private-folder reference in path context:
  # _private_ followed by a date, or appearing inside a slash-bearing token.
  printf '%s' "$hay" | grep -qE '_private_[0-9]{4}|/[^[:space:]]*_private_|_private_[^[:space:]]*/' && deny "$NAME_MSG"
fi

# 2) Marker walk: for any explicit absolute path token, deny if an ancestor holds
#    .ck-private. Covers a private folder renamed away from the convention.
for tok in $hay; do
  case "$tok" in /*) : ;; *) continue ;; esac
  p=$tok
  while [ -n "$p" ] && [ "$p" != "/" ]; do
    [ -e "$p/.ck-private" ] && deny "$MARKER_MSG"
    p=$(dirname "$p")
  done
done

exit 0
