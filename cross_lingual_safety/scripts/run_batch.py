"""Run inference for multiple conditions on all 4 models using all available GPUs."""
import json, os, subprocess, sys, time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")

MODEL_CONFIGS = [
    ("Llama-3.1-8B-Instruct", "unsloth/Meta-Llama-3.1-8B-Instruct"),
    ("Mistral-7B-Instruct-v0.3", "mistralai/Mistral-7B-Instruct-v0.3"),
    ("Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct"),
    ("Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-14B-Instruct"),
]

def run_single(model_idx, conditions, gpu_id):
    """Load model once and run all specified conditions."""
    from vllm import LLM, SamplingParams

    model_name, model_id = MODEL_CONFIGS[model_idx]

    print(f"\n{'='*60}")
    print(f"Loading {model_name} on GPU {gpu_id}...")
    print(f"Conditions: {conditions}")
    print(f"{'='*60}")

    llm = LLM(model=model_id, trust_remote_code=True, max_model_len=4096, gpu_memory_utilization=0.90)
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(max_tokens=512, temperature=0.01, top_p=0.95)

    for cond in conditions:
        output_path = os.path.join(RESULTS_DIR, f"{model_name}_{cond}.json")
        if os.path.exists(output_path):
            print(f"  Skipping {cond} (exists)")
            continue

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
        print(f"  Done in {elapsed:.1f}s")

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
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"  Saved {output_path}")

    del llm
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"{model_name}: All conditions complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-index", type=int, default=None)
    parser.add_argument("--conditions", nargs="+", required=True)
    parser.add_argument("--parallel", action="store_true")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.model_index is not None:
        gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        run_single(args.model_index, args.conditions, gpu_id)
    elif args.parallel:
        processes = []
        for idx in range(len(MODEL_CONFIGS)):
            model_name = MODEL_CONFIGS[idx][0]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(idx)

            log_dir = os.path.join(RESULTS_DIR, "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, f"{model_name}_batch.log")

            print(f"Launching {model_name} on GPU {idx}")
            log_f = open(log_path, "w")
            p = subprocess.Popen(
                ["/usr/bin/python3", os.path.abspath(__file__),
                 "--model-index", str(idx),
                 "--conditions"] + args.conditions,
                env=env, stdout=log_f, stderr=subprocess.STDOUT
            )
            processes.append((p, log_f, model_name))

        print(f"\nWaiting for {len(processes)} processes...")
        for p, log_f, name in processes:
            p.wait()
            log_f.close()
            status = "OK" if p.returncode == 0 else f"FAILED (rc={p.returncode})"
            print(f"  {name}: {status}")
        print("\nAll batch inference complete!")
