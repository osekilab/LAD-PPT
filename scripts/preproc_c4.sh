#!/bin/bash
set -euo pipefail

# ============================================================================
# C4 Dataset Download and Preprocessing
# ============================================================================
# Two methods are available:
#
# Method 1: HuggingFace Hub (default)
#   - Downloads full C4 via the HuggingFace datasets API, then tokenizes
#   - Requires: `huggingface-cli login` and access approval for allenai/c4
#
# Method 2: Subset download
#   - Downloads a subset of C4 via streaming, then tokenizes
#   - Faster download, suitable for testing or limited storage
# ============================================================================

METHOD=${C4_METHOD:-"hub"}
NUM_EXAMPLES=${C4_NUM_EXAMPLES:-5000000}
NUM_PROC=${NUM_PROC:-8}
C4_LOCAL_DIR=${C4_LOCAL_DIR:-"./data/c4_en_raw"}
C4_TOKENIZED_DIR=${C4_TOKENIZED_DIR:-"./data/tokenized/c4"}

echo "=========================================="
echo "C4 Preprocessing (method=${METHOD})"
echo "=========================================="

if [ "${METHOD}" = "hub" ]; then
    # Method 1: Download full C4 + tokenize
    echo "Step 1: Downloading full C4 dataset..."
    python -m src.download_c4 download_hub \
        --out_dir "${C4_LOCAL_DIR}"

    echo "Step 2: Tokenizing..."
    python -m src.utils cache_c4_local \
        --out_dir "${C4_TOKENIZED_DIR}" \
        --local_dir "${C4_LOCAL_DIR}" \
        --tokenizer_name "EleutherAI/pythia-1b" \
        --num_proc ${NUM_PROC}

elif [ "${METHOD}" = "subset" ]; then
    # Method 2: Download subset + tokenize
    echo "Step 1: Downloading ${NUM_EXAMPLES} C4 examples via streaming..."
    python -m src.download_c4 download_subset \
        --out_dir "${C4_LOCAL_DIR}" \
        --num_examples ${NUM_EXAMPLES}

    echo "Step 2: Tokenizing..."
    python -m src.utils cache_c4_local \
        --out_dir "${C4_TOKENIZED_DIR}" \
        --local_dir "${C4_LOCAL_DIR}" \
        --tokenizer_name "EleutherAI/pythia-1b" \
        --num_proc ${NUM_PROC}
else
    echo "Error: Unknown method '${METHOD}'. Use 'hub' or 'subset'."
    exit 1
fi

echo "C4 preprocessing complete! Tokenized data saved to: ${C4_TOKENIZED_DIR}"
