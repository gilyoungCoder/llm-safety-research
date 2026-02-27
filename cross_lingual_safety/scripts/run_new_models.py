"""Run inference for new models across all 9 conditions using 8 GPUs."""
import json, os, subprocess, time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
LOG_DIR = os.path.join(RESULTS_DIR, "logs")

# New models to evaluate
MODEL_CONFIGS = [
    ("EXAONE-3.5-7.8B-Instruct", "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"),
    ("gemma-2-9b-it", "google/gemma-2-9b-it"),
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
    parser.add_argument("--model", default="exaone", choices=["exaone", "gemma", "both"])
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    if args.model_index is not None and args.condition is not None:
        run_single(args.model_index, args.condition)
    else:
        if args.model in ("exaone", "both"):
            # EXAONE: 9 conditions, model_idx=0
            print("\n=== EXAONE-3.5-7.8B-Instruct (9 conditions) ===")
            # Wave 1: 8 conditions on 8 GPUs
            wave1 = [(0, c, i) for i, c in enumerate(ALL_CONDITIONS[:8])]
            # Wave 2: 1 remaining
            wave2 = [(0, ALL_CONDITIONS[8], 0)]

            print(f"Wave 1: {len(wave1)} jobs")
            procs = launch_wave(wave1)
            if procs:
                print("Waiting...")
                wait_for(procs)

            print(f"Wave 2: {len(wave2)} jobs")
            procs = launch_wave(wave2)
            if procs:
                print("Waiting...")
                wait_for(procs)

        if args.model in ("gemma", "both"):
            # Gemma: 9 conditions, model_idx=1
            print("\n=== Gemma-2-9B-IT (9 conditions) ===")
            wave1 = [(1, c, i) for i, c in enumerate(ALL_CONDITIONS[:8])]
            wave2 = [(1, ALL_CONDITIONS[8], 0)]

            print(f"Wave 1: {len(wave1)} jobs")
            procs = launch_wave(wave1)
            if procs:
                print("Waiting...")
                wait_for(procs)

            print(f"Wave 2: {len(wave2)} jobs")
            procs = launch_wave(wave2)
            if procs:
                print("Waiting...")
                wait_for(procs)

        print("\n=== All experiments complete! ===")
