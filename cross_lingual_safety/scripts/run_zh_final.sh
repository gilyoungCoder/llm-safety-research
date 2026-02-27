#!/bin/bash
# Run remaining Chinese conditions for all models (skip existing results)
set -e
LOG_DIR="results/logs"
mkdir -p "$LOG_DIR"

echo "=== Running remaining Chinese conditions ==="
echo "Start: $(date)"

# GPU 0: Llama (zh_gpt, zh2en_gpt)
CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 scripts/run_chinese_inference.py \
    --model-key llama --sequential > "$LOG_DIR/llama_zh_final.log" 2>&1 &
PID0=$!
echo "GPU 0: Llama (PID $PID0)"

# GPU 1: Qwen-7B (zh_gpt, zh2en_gpt)
CUDA_VISIBLE_DEVICES=1 /usr/bin/python3 scripts/run_chinese_inference.py \
    --model-key qwen7b --sequential > "$LOG_DIR/qwen7b_zh_final.log" 2>&1 &
PID1=$!
echo "GPU 1: Qwen-7B (PID $PID1)"

# GPU 2: Qwen-14B (zh_gpt, zh2en_gpt)
CUDA_VISIBLE_DEVICES=2 /usr/bin/python3 scripts/run_chinese_inference.py \
    --model-key qwen14b --sequential > "$LOG_DIR/qwen14b_zh_final.log" 2>&1 &
PID2=$!
echo "GPU 2: Qwen-14B (PID $PID2)"

# GPU 3: Phi-3 (zh_gpt, zh2en_gpt)
CUDA_VISIBLE_DEVICES=3 /usr/bin/python3 scripts/run_chinese_inference.py \
    --model-key phi3 --sequential > "$LOG_DIR/phi3_zh_final.log" 2>&1 &
PID3=$!
echo "GPU 3: Phi-3 (PID $PID3)"

# GPU 4: Yi (zh_gemini, zh2en_gemini, zh_gpt, zh2en_gpt)
CUDA_VISIBLE_DEVICES=4 /usr/bin/python3 scripts/run_chinese_inference.py \
    --model-key yi --sequential > "$LOG_DIR/yi_zh_final.log" 2>&1 &
PID4=$!
echo "GPU 4: Yi (PID $PID4)"

# GPU 5: InternLM (zh2en_gemini, zh_gpt, zh2en_gpt)
CUDA_VISIBLE_DEVICES=5 /usr/bin/python3 scripts/run_chinese_inference.py \
    --model-key internlm --sequential > "$LOG_DIR/internlm_zh_final.log" 2>&1 &
PID5=$!
echo "GPU 5: InternLM (PID $PID5)"

echo ""
echo "All 6 models launched. Waiting..."
wait $PID0 && echo "Llama: DONE" || echo "Llama: FAILED"
wait $PID1 && echo "Qwen-7B: DONE" || echo "Qwen-7B: FAILED"
wait $PID2 && echo "Qwen-14B: DONE" || echo "Qwen-14B: FAILED"
wait $PID3 && echo "Phi-3: DONE" || echo "Phi-3: FAILED"
wait $PID4 && echo "Yi: DONE" || echo "Yi: FAILED"
wait $PID5 && echo "InternLM: DONE" || echo "InternLM: FAILED"

echo ""
echo "=== All remaining Chinese inference complete ==="
echo "End: $(date)"
echo "Total Chinese result files:"
ls results/*_zh*.json 2>/dev/null | wc -l
