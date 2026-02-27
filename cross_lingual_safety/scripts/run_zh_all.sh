#!/bin/bash
# Run all 6 models x 8 Chinese conditions in parallel (one model per GPU)
# Small models: GPU 0-5, each runs 8 conditions sequentially

set -e
LOG_DIR="results/logs"
mkdir -p "$LOG_DIR"

echo "=== Starting Chinese inference: 6 models x 8 conditions ==="
echo "Start time: $(date)"

# GPU 0: Llama-3.1-8B
CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 scripts/run_chinese_inference.py \
    --model-key llama --sequential > "$LOG_DIR/llama_zh_all.log" 2>&1 &
PID_LLAMA=$!
echo "GPU 0: Llama-8B (PID $PID_LLAMA)"

# GPU 1: Qwen-7B
CUDA_VISIBLE_DEVICES=1 /usr/bin/python3 scripts/run_chinese_inference.py \
    --model-key qwen7b --sequential > "$LOG_DIR/qwen7b_zh_all.log" 2>&1 &
PID_QWEN7B=$!
echo "GPU 1: Qwen-7B (PID $PID_QWEN7B)"

# GPU 2: Qwen-14B (needs more memory)
CUDA_VISIBLE_DEVICES=2 /usr/bin/python3 scripts/run_chinese_inference.py \
    --model-key qwen14b --sequential > "$LOG_DIR/qwen14b_zh_all.log" 2>&1 &
PID_QWEN14B=$!
echo "GPU 2: Qwen-14B (PID $PID_QWEN14B)"

# GPU 3: Phi-3-14B (needs more memory)
CUDA_VISIBLE_DEVICES=3 /usr/bin/python3 scripts/run_chinese_inference.py \
    --model-key phi3 --sequential > "$LOG_DIR/phi3_zh_all.log" 2>&1 &
PID_PHI3=$!
echo "GPU 3: Phi-3-14B (PID $PID_PHI3)"

# GPU 4: Yi-9B
CUDA_VISIBLE_DEVICES=4 /usr/bin/python3 scripts/run_chinese_inference.py \
    --model-key yi --sequential > "$LOG_DIR/yi_zh_all.log" 2>&1 &
PID_YI=$!
echo "GPU 4: Yi-9B (PID $PID_YI)"

# GPU 5: InternLM-7B
CUDA_VISIBLE_DEVICES=5 /usr/bin/python3 scripts/run_chinese_inference.py \
    --model-key internlm --sequential > "$LOG_DIR/internlm_zh_all.log" 2>&1 &
PID_INTERNLM=$!
echo "GPU 5: InternLM-7B (PID $PID_INTERNLM)"

echo ""
echo "All 6 models launched. Waiting..."
wait $PID_LLAMA && echo "Llama: DONE" || echo "Llama: FAILED"
wait $PID_QWEN7B && echo "Qwen-7B: DONE" || echo "Qwen-7B: FAILED"
wait $PID_QWEN14B && echo "Qwen-14B: DONE" || echo "Qwen-14B: FAILED"
wait $PID_PHI3 && echo "Phi-3: DONE" || echo "Phi-3: FAILED"
wait $PID_YI && echo "Yi: DONE" || echo "Yi: FAILED"
wait $PID_INTERNLM && echo "InternLM: DONE" || echo "InternLM: FAILED"

echo ""
echo "=== All Chinese inference complete ==="
echo "End time: $(date)"
echo "Results:"
ls -la results/*_zh*.json results/*_mix_zh*.json results/*_adaptive_zh*.json 2>/dev/null | wc -l
echo "result files generated"
