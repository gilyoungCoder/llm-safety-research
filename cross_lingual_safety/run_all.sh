#!/bin/bash
# Cross-Lingual Safety: Full Pipeline
# Usage: bash run_all.sh

set -e
cd "$(dirname "$0")"

PYTHON=python3
echo "=========================================="
echo "Cross-Lingual Safety Evaluation Pipeline"
echo "=========================================="

# Step 1: Translate AdvBench to Korean
echo ""
echo "[Step 1/5] Translating AdvBench prompts..."
if [ -f data/translations.json ]; then
    echo "  Translations exist, skipping."
else
    CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/translate.py
fi

# Step 2: Run inference (all models in parallel)
echo ""
echo "[Step 2/5] Running inference (5 models × 4 conditions)..."
$PYTHON scripts/run_inference.py --parallel --num-gpus 8

# Step 3: Evaluate
echo ""
echo "[Step 3/5] Evaluating results..."
$PYTHON scripts/evaluate.py

# Step 4: Analyze
echo ""
echo "[Step 4/5] Running analysis..."
$PYTHON scripts/analyze.py

# Step 5: Visualize
echo ""
echo "[Step 5/5] Generating figures..."
$PYTHON scripts/visualize.py

echo ""
echo "=========================================="
echo "Pipeline complete!"
echo "Results: results/"
echo "Figures: figures/"
echo "=========================================="
