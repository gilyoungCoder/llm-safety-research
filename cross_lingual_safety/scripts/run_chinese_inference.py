"""Run inference for Chinese conditions across all 6 models.
6 models x 9 Chinese conditions x 520 prompts = 28,080 inferences.

Conditions:
  zh          - NLLB Chinese translation
  zh2en       - NLLB Chinese + English response instruction
  mix_zh      - Chinese frame + English content
  adaptive_zh - Chinese word substitution (adaptive mixed)
  zh_gemini   - Gemini Chinese translation
  zh2en_gemini- Gemini Chinese + English response instruction
  zh_gpt      - GPT Chinese translation
  zh2en_gpt   - GPT Chinese + English response instruction
  ws_zh       - Word-substitution (same as adaptive_zh, alias)

Usage:
  # Single model, single condition
  CUDA_VISIBLE_DEVICES=0 python run_chinese_inference.py --model-key llama --condition zh

  # Single model, all available conditions (one GPU, sequential)
  CUDA_VISIBLE_DEVICES=0 python run_chinese_inference.py --model-key llama

  # All models, parallel across 8 GPUs
  python run_chinese_inference.py --parallel --models llama qwen7b qwen14b phi3 yi internlm
"""
import argparse
import json
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
LOG_DIR = os.path.join(RESULTS_DIR, "logs")

MODEL_CONFIGS = {
    "llama": ("Llama-3.1-8B-Instruct", "unsloth/Meta-Llama-3.1-8B-Instruct"),
    "qwen7b": ("Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct"),
    "qwen14b": ("Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-14B-Instruct"),
    "phi3": ("Phi-3-medium-4k-instruct", "microsoft/Phi-3-medium-4k-instruct"),
    "yi": ("Yi-1.5-9B-Chat", "01-ai/Yi-1.5-9B-Chat"),
    "internlm": ("internlm2_5-7b-chat", "internlm/internlm2_5-7b-chat"),
}

ZH_CONDITIONS = [
    "zh", "zh2en", "mix_zh", "adaptive_zh",
    "zh_gemini", "zh2en_gemini",
    "zh_gpt", "zh2en_gpt",
]


def get_available_conditions():
    """Return only conditions whose data files exist."""
    available = []
    for cond in ZH_CONDITIONS:
        path = os.path.join(DATA_DIR, f"advbench_{cond}.json")
        if os.path.exists(path):
            available.append(cond)
    return available


def run_single(model_key, condition):
    """Run a single model+condition inference."""
    from vllm import LLM, SamplingParams

    model_name, model_id = MODEL_CONFIGS[model_key]
    output_path = os.path.join(RESULTS_DIR, f"{model_name}_{condition}.json")
    if os.path.exists(output_path):
        print(f"Skipping {model_name} {condition} (exists)")
        return

    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    print(f"Loading {model_name} on GPU {gpu_id} for {condition}...")
    llm = LLM(model=model_id, trust_remote_code=True, max_model_len=4096, gpu_memory_utilization=0.90)
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(max_tokens=512, temperature=0.01, top_p=0.95)

    data_path = os.path.join(DATA_DIR, f"advbench_{condition}.json")
    with open(data_path) as f:
        prompts = json.load(f)

    formatted = []
    for p in prompts:
        messages = [{"role": "user", "content": p["prompt"]}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        formatted.append(text)

    print(f"Running {condition.upper()} ({len(prompts)} prompts)...")
    start = time.time()
    outputs = llm.generate(formatted, sampling_params)
    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s ({len(prompts)/elapsed:.1f} prompts/s)")

    results = []
    for p, output in zip(prompts, outputs):
        results.append({
            "id": p["id"],
            "prompt": p["prompt"],
            "category": p["category"],
            "condition": condition,
            "response": output.outputs[0].text,
        })

    output_data = {
        "model_name": model_name,
        "model_id": model_id,
        "condition": condition,
        "num_prompts": len(prompts),
        "max_new_tokens": 512,
        "temperature": 0.0,
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"Saved {output_path}")


def run_model_sequential(model_key, conditions):
    """Load model once, run multiple conditions sequentially."""
    from vllm import LLM, SamplingParams

    model_name, model_id = MODEL_CONFIGS[model_key]

    # Filter to only conditions that need running
    todo = []
    for cond in conditions:
        output_path = os.path.join(RESULTS_DIR, f"{model_name}_{cond}.json")
        if not os.path.exists(output_path):
            todo.append(cond)
        else:
            print(f"  Skipping {cond} (exists)")

    if not todo:
        print(f"All conditions done for {model_name}")
        return

    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    print(f"\nLoading {model_name} on GPU {gpu_id} for {len(todo)} conditions...")
    llm = LLM(model=model_id, trust_remote_code=True, max_model_len=4096, gpu_memory_utilization=0.90)
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(max_tokens=512, temperature=0.01, top_p=0.95)

    for cond in todo:
        data_path = os.path.join(DATA_DIR, f"advbench_{cond}.json")
        with open(data_path) as f:
            prompts = json.load(f)

        formatted = []
        for p in prompts:
            messages = [{"role": "user", "content": p["prompt"]}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            formatted.append(text)

        print(f"  Running {cond.upper()} ({len(prompts)} prompts)...")
        start = time.time()
        outputs = llm.generate(formatted, sampling_params)
        elapsed = time.time() - start
        print(f"  Done in {elapsed:.1f}s ({len(prompts)/elapsed:.1f} prompts/s)")

        results = []
        for p, output in zip(prompts, outputs):
            results.append({
                "id": p["id"],
                "prompt": p["prompt"],
                "category": p["category"],
                "condition": cond,
                "response": output.outputs[0].text,
            })

        output_data = {
            "model_name": model_name,
            "model_id": model_id,
            "condition": cond,
            "num_prompts": len(prompts),
            "max_new_tokens": 512,
            "temperature": 0.0,
            "results": results,
        }
        output_path = os.path.join(RESULTS_DIR, f"{model_name}_{cond}.json")
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"  Saved {output_path}")

    # Free GPU
    del llm
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"\n{model_name}: All Chinese conditions complete!")


def launch_jobs(model_key, conditions, start_gpu=0):
    """Launch parallel jobs for one model across conditions on GPUs."""
    model_name = MODEL_CONFIGS[model_key][0]
    processes = []
    for i, condition in enumerate(conditions):
        gpu_id = start_gpu + i
        output_path = os.path.join(RESULTS_DIR, f"{model_name}_{condition}.json")
        if os.path.exists(output_path):
            print(f"  GPU {gpu_id}: {model_name} / {condition} -- SKIP (exists)")
            continue
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        log_path = os.path.join(LOG_DIR, f"{model_name}_{condition}.log")
        log_f = open(log_path, "w")
        print(f"  GPU {gpu_id}: {model_name} / {condition}")
        p = subprocess.Popen(
            ["/usr/bin/python3", os.path.abspath(__file__),
             "--model-key", model_key, "--condition", condition],
            env=env, stdout=log_f, stderr=subprocess.STDOUT
        )
        processes.append((p, log_f, model_name, condition))
    return processes


def wait_for(processes):
    for p, log_f, name, cond in processes:
        p.wait()
        log_f.close()
        status = "OK" if p.returncode == 0 else f"FAILED (rc={p.returncode})"
        print(f"  {name}/{cond}: {status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", default=None, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--condition", default=None)
    parser.add_argument("--models", nargs="+", default=list(MODEL_CONFIGS.keys()),
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--parallel", action="store_true",
                        help="Launch each condition on a separate GPU (max 8)")
    parser.add_argument("--sequential", action="store_true",
                        help="Load model once, run all conditions sequentially on one GPU")
    parser.add_argument("--start-gpu", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    available = get_available_conditions()
    print(f"Available Chinese conditions ({len(available)}/{len(ZH_CONDITIONS)}): {available}")
    missing = set(ZH_CONDITIONS) - set(available)
    if missing:
        print(f"Missing data files for: {sorted(missing)}")

    if args.model_key is not None and args.condition is not None:
        # Single model + single condition (called by subprocess)
        run_single(args.model_key, args.condition)
    elif args.model_key is not None and args.sequential:
        # Single model, all conditions sequentially
        run_model_sequential(args.model_key, available)
    elif args.parallel:
        # Parallel: each model runs all conditions, one condition per GPU
        for model_key in args.models:
            model_name = MODEL_CONFIGS[model_key][0]
            print(f"\n=== {model_name} (Chinese conditions) ===")

            # Wave 1: up to 8 conditions on GPUs
            wave1 = available[:8]
            print(f"Wave 1: {len(wave1)} conditions")
            procs = launch_jobs(model_key, wave1, start_gpu=args.start_gpu)
            if procs:
                print("Waiting for wave 1...")
                wait_for(procs)

            # Wave 2: remaining conditions
            if len(available) > 8:
                wave2 = available[8:]
                print(f"Wave 2: {len(wave2)} conditions")
                procs = launch_jobs(model_key, wave2, start_gpu=args.start_gpu)
                if procs:
                    print("Waiting for wave 2...")
                    wait_for(procs)
    else:
        # Default: sequential for each model
        for model_key in args.models:
            model_name = MODEL_CONFIGS[model_key][0]
            print(f"\n=== {model_name} ===")
            run_model_sequential(model_key, available)

    print("\n=== Chinese inference complete! ===")
