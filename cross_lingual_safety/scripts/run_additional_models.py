"""Run inference for additional models across all 9 conditions.
Downloads models first if needed, then runs inference across GPUs."""
import json, os, subprocess, time, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
LOG_DIR = os.path.join(RESULTS_DIR, "logs")

MODEL_CONFIGS = {
    "phi3": ("Phi-3-medium-4k-instruct", "microsoft/Phi-3-medium-4k-instruct"),
    "yi": ("Yi-1.5-9B-Chat", "01-ai/Yi-1.5-9B-Chat"),
    "internlm": ("internlm2_5-7b-chat", "internlm/internlm2_5-7b-chat"),
}

ALL_CONDITIONS = ["en", "ko", "mix", "ko2en", "adaptive", "ko_gemini", "ko2en_gemini", "ko_gpt", "ko2en_gpt"]


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


def launch_jobs(model_key, conditions, start_gpu=0):
    """Launch jobs for one model across conditions on GPUs starting from start_gpu."""
    model_name = MODEL_CONFIGS[model_key][0]
    processes = []
    for i, condition in enumerate(conditions):
        gpu_id = start_gpu + i
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", default=None, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--condition", default=None)
    parser.add_argument("--models", nargs="+", default=["phi3", "yi", "internlm"],
                        choices=list(MODEL_CONFIGS.keys()))
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    if args.model_key is not None and args.condition is not None:
        run_single(args.model_key, args.condition)
    else:
        for model_key in args.models:
            model_name = MODEL_CONFIGS[model_key][0]
            print(f"\n=== {model_name} (9 conditions) ===")

            # Wave 1: 8 conditions on 8 GPUs
            print(f"Wave 1: 8 conditions")
            procs = launch_jobs(model_key, ALL_CONDITIONS[:8], start_gpu=0)
            if procs:
                print("Waiting for wave 1...")
                wait_for(procs)
            else:
                print("All wave 1 jobs already done.")

            # Wave 2: 1 remaining condition
            print(f"Wave 2: 1 condition")
            procs = launch_jobs(model_key, ALL_CONDITIONS[8:], start_gpu=0)
            if procs:
                print("Waiting for wave 2...")
                wait_for(procs)
            else:
                print("All wave 2 jobs already done.")

        print("\n=== All experiments complete! ===")
