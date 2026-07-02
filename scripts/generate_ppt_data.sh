#!/bin/bash
set -euo pipefail

N=${N:-100000}
MAX_LENGTH=${MAX_LENGTH:-1024}

echo "=========================================="
echo "Generating PPT data (N=${N}, max_length=${MAX_LENGTH})"
echo "=========================================="

# --- 1. MP-STRUCT ---
echo "[1/2] Generating MP-STRUCT..."
python -m src.mp_struct generate \
    --file_dir ./data/mp_struct \
    --n ${N} \
    --max_length ${MAX_LENGTH}

python -m src.utils cache_data \
    --dataset_name data/mp_struct/mp_struct_ids_${N}_${MAX_LENGTH}.txt \
    --out_dir ./data/tokenized/mp_struct

# --- 2. MP-STRUCT CORE ---
echo "[2/2] Generating MP-STRUCT CORE..."
python -m src.mp_struct_core generate \
    --out_dir ./data/mp_struct_core \
    --n ${N} \
    --length ${MAX_LENGTH}

python -m src.utils cache_data \
    --dataset_name data/mp_struct_core/mp_struct_core_ids.txt \
    --out_dir ./data/tokenized/mp_struct_core

echo "=========================================="
echo "All PPT data generated successfully!"
echo "=========================================="
