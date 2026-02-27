#!/bin/bash
# Run remaining Chinese conditions for all models
set -e
LOG_DIR="results/logs"
mkdir -p "$LOG_DIR"

echo "=== Running remaining Chinese conditions ==="
echo "Start: $(date)"

# GPU 0: Llama (zh_gpt, zh2en_gpt)
CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 scripts/run_chinese_inference.py \
    --model-key llama --sequential > "$LOG_DIR/llama_zh_remaining.log" 2>&1 &
echo "GPU 0: Llama (GPT conditions)"

# GPU 1: Qwen-7B (zh_gpt, zh2en_gpt)
CUDA_VISIBLE_DEVICES=1 /usr/bin/python3 scripts/run_chinese_inference.py \
    --model-key qwen7b --sequential > "$LOG_DIR/qwen7b_zh_remaining.log" 2>&1 &
echo "GPU 1: Qwen-7B (GPT conditions)"

# GPU 2: Qwen-14B (zh_gpt, zh2en_gpt)
CUDA_VISIBLE_DEVICES=2 /usr/bin/python3 scripts/run_chinese_inference.py \
    --model-key qwen14b --sequential > "$LOG_DIR/qwen14b_zh_remaining.log" 2>&1 &
echo "GPU 2: Qwen-14B (GPT conditions)"

# GPU 3: Phi-3 (zh_gpt, zh2en_gpt)
CUDA_VISIBLE_DEVICES=3 /usr/bin/python3 scripts/run_chinese_inference.py \
    --model-key phi3 --sequential > "$LOG_DIR/phi3_zh_remaining.log" 2>&1 &
echo "GPU 3: Phi-3 (GPT conditions)"

# GPU 4: Yi (zh_gemini, zh2en_gemini, zh_gpt, zh2en_gpt)
CUDA_VISIBLE_DEVICES=4 /usr/bin/python3 scripts/run_chinese_inference.py \
    --model-key yi --sequential > "$LOG_DIR/yi_zh_remaining.log" 2>&1 &
echo "GPU 4: Yi (Gemini + GPT conditions)"

# GPU 5: InternLM (zh2en_gemini, zh_gpt, zh2en_gpt)
CUDA_VISIBLE_DEVICES=5 /usr/bin/python3 scripts/run_chinese_inference.py \
    --model-key internlm --sequential > "$LOG_DIR/internlm_zh_remaining.log" 2>&1 &
echo "GPU 5: InternLM (Gemini + GPT conditions)"

echo ""
echo "All 6 models launched. Waiting..."
wait
echo ""
echo "=== All remaining Chinese inference complete ==="
echo "End: $(date)"
