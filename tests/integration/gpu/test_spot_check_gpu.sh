#!/bin/bash
#
# GPU integration test for spot_check.
# Requires: a GPU node (vis node, interactive session, or via SLURM).
#
# Usage:
#   bash tests/integration/gpu/test_spot_check_gpu.sh \
#       --data /path/to/data.json \
#       --base /path/to/base/model \
#       --ft /path/to/finetuned/checkpoint

set -euo pipefail

# Defaults (override with flags)
DATA=""
BASE=""
FT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --data) DATA="$2"; shift 2 ;;
        --base) BASE="$2"; shift 2 ;;
        --ft)   FT="$2";   shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$DATA" || -z "$BASE" || -z "$FT" ]]; then
    echo "Usage: $0 --data <path> --base <path> --ft <path>"
    exit 1
fi

echo "========================================="
echo "Test 1: Fine-tuned 3B model"
echo "========================================="
python -m cruijff_kit.utils.spot_check \
    --model "$FT" \
    --data "$DATA" \
    --split validation \
    --n 10 \
    --max-new-tokens 5

echo ""
echo "========================================="
echo "Test 2: Base 3B model (no fine-tuning)"
echo "========================================="
python -m cruijff_kit.utils.spot_check \
    --model "$BASE" \
    --data "$DATA" \
    --split validation \
    --n 10 \
    --max-new-tokens 5

echo ""
echo "========================================="
echo "Test 3: Fine-tuned with --no-chat-template"
echo "========================================="
python -m cruijff_kit.utils.spot_check \
    --model "$FT" \
    --data "$DATA" \
    --split validation \
    --n 5 \
    --max-new-tokens 5 \
    --no-chat-template

echo ""
echo "========================================="
echo "Test 4: Fine-tuned with system prompt"
echo "========================================="
python -m cruijff_kit.utils.spot_check \
    --model "$FT" \
    --data "$DATA" \
    --split validation \
    --n 5 \
    --max-new-tokens 5 \
    --sysprompt "You are a helpful assistant that predicts income levels."

echo ""
echo "All spot_check GPU tests completed."
