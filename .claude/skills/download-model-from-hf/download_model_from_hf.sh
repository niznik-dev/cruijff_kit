#!/bin/bash

################################################################################
# Download Model from HuggingFace
################################################################################
# This script downloads LLM models from HuggingFace using torchtune's download
# command. It handles various model sizes and configurations.
#
# PREREQUISITES:
# 1. Request access to Meta models on HuggingFace:
#    - Visit https://huggingface.co/meta-llama
#    - Click on the specific model you want (e.g., meta-llama/Llama-3.2-3B-Instruct)
#    - Accept the license agreement and request access
#    - Wait for approval (usually within a few hours)
#
# 2. Create a HuggingFace token with read permissions:
#    - Visit https://huggingface.co/settings/tokens
#    - Create a new token with "Read" access
#    - Save the token securely
#
# 3. Ensure torchtune is installed:
#    - The 'tune' command must be available in your environment
################################################################################

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to print usage
print_usage() {
    cat << EOF

${GREEN}Usage:${NC}
    $0 <model_name> <output_dir> <hf_token>

${GREEN}Arguments:${NC}
    model_name    HuggingFace model identifier (e.g., meta-llama/Llama-3.2-1B-Instruct)
    output_dir    Directory where the model will be downloaded
    hf_token      Your HuggingFace access token

${GREEN}Examples:${NC}
    # Llama 2 (7B parameters, single checkpoint file)
    $0 meta-llama/Llama-2-7b-hf /scratch/gpfs/MSALGANIK/\$USER/models YOUR_HF_TOKEN

    # Llama 3.1 (8B parameters, single checkpoint file)
    $0 meta-llama/Llama-3.1-8B-Instruct /scratch/gpfs/MSALGANIK/\$USER/models YOUR_HF_TOKEN

    # Llama 3.2 1B (single checkpoint file: model.safetensors)
    $0 meta-llama/Llama-3.2-1B-Instruct /scratch/gpfs/MSALGANIK/\$USER/models YOUR_HF_TOKEN

    # Llama 3.2 3B (split checkpoint: model-00001-of-00002.safetensors, model-00002-of-00002.safetensors)
    $0 meta-llama/Llama-3.2-3B-Instruct /scratch/gpfs/MSALGANIK/\$USER/models YOUR_HF_TOKEN

    # Llama 3.3 70B (multiple split checkpoint files)
    $0 meta-llama/Llama-3.3-70B-Instruct /scratch/gpfs/MSALGANIK/\$USER/models YOUR_HF_TOKEN

${YELLOW}Note about checkpoint files:${NC}
    - Smaller models (1B, 7B, 8B): Usually have a single checkpoint file (model.safetensors)
    - Larger models (3B+, 70B+): May have split checkpoint files for easier download and storage
      Examples:
        * 3B models: model-00001-of-00002.safetensors, model-00002-of-00002.safetensors
        * 70B models: model-00001-of-00030.safetensors, ..., model-00030-of-00030.safetensors

    ${GREEN}Both formats work correctly with torchtune!${NC}
    The split files will be automatically handled during model loading.

    ${YELLOW}Important for cruijff_kit:${NC}
    When using split checkpoint files in your finetune.yaml, you must list ALL files:

    checkpointer:
      checkpoint_files:
      - model-00001-of-00002.safetensors
      - model-00002-of-00002.safetensors

    (Not just "model.safetensors")

${YELLOW}Common Issues:${NC}
    1. "401 Unauthorized" error:
       - Make sure you've requested access to the model on HuggingFace
       - Verify your token has the necessary permissions
       - Check that your token hasn't expired

    2. "tune: command not found":
       - Install torchtune: pip install torchtune
       - Or activate the environment where torchtune is installed

    3. Slow download speeds:
       - Large models (70B+) can take several hours to download
       - Ensure stable network connection
       - Consider using a compute node with better network bandwidth

EOF
}

# Check if help is requested
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]] || [[ $# -eq 0 ]]; then
    print_usage
    exit 0
fi

# Validate number of arguments
if [[ $# -ne 3 ]]; then
    print_error "Invalid number of arguments"
    print_usage
    exit 1
fi

# Assign arguments to variables
MODEL_NAME="$1"
OUTPUT_DIR="$2"
HF_TOKEN="$3"

# Validate arguments
if [[ -z "$MODEL_NAME" ]]; then
    print_error "Model name cannot be empty"
    exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    print_error "Output directory cannot be empty"
    exit 1
fi

if [[ -z "$HF_TOKEN" ]]; then
    print_error "HuggingFace token cannot be empty"
    exit 1
fi

# Check if tune command is available
if ! command -v tune &> /dev/null; then
    print_error "The 'tune' command is not found in your PATH"
    echo ""
    echo "Please install torchtune:"
    echo "  pip install torchtune"
    echo ""
    echo "Or activate the environment where torchtune is installed."
    exit 1
fi

# Create output directory if it doesn't exist
if [[ ! -d "$OUTPUT_DIR" ]]; then
    print_info "Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

# Print download information
echo ""
print_info "Starting model download..."
echo "  Model: $MODEL_NAME"
echo "  Output Directory: $OUTPUT_DIR"
echo ""

# Download the model
print_info "Running: tune download $MODEL_NAME --output-dir $OUTPUT_DIR --hf-token [HIDDEN]"
if tune download "$MODEL_NAME" --output-dir "$OUTPUT_DIR" --hf-token "$HF_TOKEN"; then
    echo ""
    print_success "Model downloaded successfully!"
    echo ""

    # Determine the model directory name (last part of model_name)
    MODEL_DIR_NAME="${MODEL_NAME##*/}"
    MODEL_PATH="$OUTPUT_DIR/$MODEL_DIR_NAME"

    # Check what was downloaded
    print_info "Verifying download..."
    if [[ -d "$MODEL_PATH" ]]; then
        echo ""
        echo "Model location: $MODEL_PATH"
        echo ""
        echo "Downloaded files:"
        ls -lh "$MODEL_PATH"
        echo ""

        # Check for checkpoint files and provide YAML configuration
        if ls "$MODEL_PATH"/model*.safetensors &> /dev/null; then
            CHECKPOINT_COUNT=$(ls "$MODEL_PATH"/model*.safetensors 2>/dev/null | wc -l)
            if [[ $CHECKPOINT_COUNT -eq 1 ]]; then
                print_success "Found single checkpoint file (model.safetensors)"
                echo ""
                print_info "For your finetune.yaml, use:"
                echo ""
                echo "  checkpointer:"
                echo "    checkpoint_files:"
                echo "    - model.safetensors"
                echo ""
            else
                print_success "Found $CHECKPOINT_COUNT split checkpoint files"
                print_info "This is normal for larger models. All files will be loaded automatically."
                echo ""
                print_warning "IMPORTANT: For your finetune.yaml, you must list ALL checkpoint files:"
                echo ""
                echo "  checkpointer:"
                echo "    checkpoint_files:"
                for ckpt in "$MODEL_PATH"/model*.safetensors; do
                    echo "    - $(basename "$ckpt")"
                done
                echo ""
            fi
        else
            print_warning "No .safetensors checkpoint files found. Please verify the download."
        fi

        echo ""
        print_success "Download complete! You can now use this model for fine-tuning or inference."
    else
        print_warning "Expected model directory not found at: $MODEL_PATH"
        print_info "Please check the output directory for the downloaded files."
    fi
else
    echo ""
    print_error "Model download failed!"
    echo ""
    echo "Common solutions:"
    echo "  1. Verify you have requested access to the model on HuggingFace"
    echo "  2. Check that your HuggingFace token is valid and has read permissions"
    echo "  3. Ensure you have sufficient disk space in the output directory"
    echo "  4. Check your network connection"
    echo ""
    exit 1
fi
