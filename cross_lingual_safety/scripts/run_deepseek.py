"""Run inference for DeepSeek models across all 9 conditions using 8 GPUs.
3 models × 9 conditions = 27 jobs, batched across 8 GPUs in waves."""
import json, os, subprocess, time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
LOG_DIR = os.path.join(RESULTS_DIR, "logs")

MODEL_CONFIGS = [
    ("DeepSeek-R1-Distill-Qwen-7B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"),
    ("DeepSeek-R1-Distill-Llama-8B", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"),
    ("DeepSeek-R1-Distill-Qwen-14B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"),
]

ALL_CONDITIONS = ["en", "ko", "mix", "ko2en", "adaptive", "ko_gemini", "ko2en_gemini", "ko_gpt", "ko2en_gpt"]


def run_single(model_idx, condition):
    """Run a single model+condition inference."""
    from vllm import LLM, SamplingParams
    model_name, model_id = MODEL_CONFIGS[model_idx]
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
    print(f"Done in {elapsed:.1f}s")

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


def launch_wave(jobs):
    """Launch a wave of (model_idx, condition, gpu_id) jobs."""
    processes = []
    for model_idx, condition, gpu_id in jobs:
        model_name = MODEL_CONFIGS[model_idx][0]
        output_path = os.path.join(RESULTS_DIR, f"{model_name}_{condition}.json")
        if os.path.exists(output_path):
            print(f"  GPU {gpu_id}: {model_name} / {condition} — SKIP (exists)")
            continue
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        log_path = os.path.join(LOG_DIR, f"{model_name}_{condition}.log")
        log_f = open(log_path, "w")
        print(f"  GPU {gpu_id}: {model_name} / {condition}")
        p = subprocess.Popen(
            ["/usr/bin/python3", os.path.abspath(__file__),
             "--model-index", str(model_idx), "--condition", condition],
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-index", type=int, default=None)
    parser.add_argument("--condition", default=None)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    if args.model_index is not None and args.condition is not None:
        run_single(args.model_index, args.condition)
    else:
        # 3 models × 9 conditions = 27 jobs
        # DS-7B (idx=0), DS-8B (idx=1) fit easily (~15-16GB)
        # DS-14B (idx=2) needs ~30GB, still fits on A6000
        # Schedule across 8 GPUs in waves

        # Wave 1: 8 jobs
        wave1 = [
            (0, "en", 0),       # DS-Qwen-7B
            (0, "ko", 1),
            (0, "mix", 2),
            (1, "en", 3),       # DS-Llama-8B
            (1, "ko", 4),
            (1, "mix", 5),
            (2, "en", 6),       # DS-Qwen-14B
            (2, "ko", 7),
        ]
        # Wave 2: 8 jobs
        wave2 = [
            (0, "ko2en", 0),
            (0, "adaptive", 1),
            (0, "ko_gemini", 2),
            (1, "ko2en", 3),
            (1, "adaptive", 4),
            (1, "ko_gemini", 5),
            (2, "mix", 6),
            (2, "ko2en", 7),
        ]
        # Wave 3: 8 jobs
        wave3 = [
            (0, "ko2en_gemini", 0),
            (0, "ko_gpt", 1),
            (0, "ko2en_gpt", 2),
            (1, "ko2en_gemini", 3),
            (1, "ko_gpt", 4),
            (1, "ko2en_gpt", 5),
            (2, "adaptive", 6),
            (2, "ko_gemini", 7),
        ]
        # Wave 4: 3 jobs
        wave4 = [
            (2, "ko2en_gemini", 0),
            (2, "ko_gpt", 1),
            (2, "ko2en_gpt", 2),
        ]

        waves = [wave1, wave2, wave3, wave4]
        for i, wave in enumerate(waves, 1):
            print(f"\n=== Wave {i}: {len(wave)} jobs ===")
            procs = launch_wave(wave)
            if procs:
                print(f"Waiting for wave {i}...")
                wait_for(procs)
            else:
                print("All jobs in this wave already complete.")

        print("\n=== All 27 DeepSeek experiments complete! ===")
