#!/bin/bash
# ==============================================================================
# Full pipeline: data prep → distributed inference → evaluate → visualize
#
# USAGE:
#   1. Edit SERVER_CONFIG below with your actual hostnames
#   2. bash run_all.sh
#
# Or for single-server with 8 GPUs:
#   bash run_all.sh --local 8
# ==============================================================================

set -e
cd "$(dirname "$0")"

SCRIPT_DIR="./scripts"
BACKEND="vllm"

# ====== EDIT THIS: your server config ======
# Format: "hostname:gpu_type:num_gpus"
# gpu_type: 3090 (24GB) or a6000 (48GB)
SERVER_CONFIG="srv1:3090:8,srv2:3090:8,srv3:3090:8,srv4:a6000:8,srv5:a6000:8"
# ===========================================

MODE="distributed"
NUM_GPUS=8

while [[ $# -gt 0 ]]; do
    case $1 in
        --local) MODE="local"; NUM_GPUS="$2"; shift 2 ;;
        --servers) SERVER_CONFIG="$2"; shift 2 ;;
        --backend) BACKEND="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

echo "=========================================="
echo "Step 1: Prepare AdvBench dataset"
echo "=========================================="
python "$SCRIPT_DIR/prepare_data.py"

echo ""
echo "=========================================="
echo "Step 2: Run inference"
echo "=========================================="

if [ "$MODE" = "local" ]; then
    echo "Mode: LOCAL parallel ($NUM_GPUS GPUs)"
    python "$SCRIPT_DIR/run_inference.py" --parallel --num-gpus "$NUM_GPUS" --backend "$BACKEND"
else
    echo "Mode: DISTRIBUTED (servers: $SERVER_CONFIG)"
    python "$SCRIPT_DIR/run_inference.py" --distributed --servers "$SERVER_CONFIG" --backend "$BACKEND"

    echo ""
    echo "Launch scripts generated in ./launch/"
    echo "Running master launcher..."
    bash ./launch/launch_all.sh

    echo ""
    echo "Merging shards..."
    python "$SCRIPT_DIR/run_inference.py" --merge
fi

echo ""
echo "=========================================="
echo "Step 3: Evaluate results"
echo "=========================================="
python "$SCRIPT_DIR/evaluate.py"

echo ""
echo "=========================================="
echo "Step 4: Generate figures and tables"
echo "=========================================="
python "$SCRIPT_DIR/visualize.py"

echo ""
echo "=========================================="
echo "PIPELINE COMPLETE!"
echo "=========================================="
echo "Results:  ./results/"
echo "Figures:  ./figures/"
echo "LaTeX:    ./figures/main_table.tex"
echo "Paper:    ./paper/main.tex"
