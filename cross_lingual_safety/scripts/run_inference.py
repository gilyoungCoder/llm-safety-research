"""
Run cross-lingual safety inference.
5 models x 4 conditions (EN, KO, MIX, KO2EN) x 520 prompts = 10,400 inferences.

Each model is loaded ONCE and runs all 4 conditions sequentially for efficiency.

Usage:
  # Single model (load once, run all 4 conditions)
  CUDA_VISIBLE_DEVICES=0 python run_inference.py --model-index 0

  # All models in parallel across GPUs
  python run_inference.py --parallel --num-gpus 8
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_PATH = os.path.abspath(__file__)
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")

# (display_name, hf_model_id, min_vram_gb)
MODEL_CONFIGS = [
    ("Llama-3.1-8B-Instruct", "unsloth/Meta-Llama-3.1-8B-Instruct", 18),
    ("Mistral-7B-Instruct-v0.3", "mistralai/Mistral-7B-Instruct-v0.3", 16),
    ("Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct", 16),
    ("Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-14B-Instruct", 30),
]

CONDITIONS = ["en", "ko", "mix", "ko2en"]


def load_prompts(condition, data_dir=DATA_DIR):
    path = os.path.join(data_dir, f"advbench_{condition}.json")
    with open(path) as f:
        return json.load(f)


def run_model_all_conditions(
    model_index, output_dir, data_dir=DATA_DIR, max_new_tokens=512, temperature=0.0
):
    """Load a model once and run all 4 conditions."""
    from vllm import LLM, SamplingParams

    model_name, model_id, _ = MODEL_CONFIGS[model_index]

    # Check if all conditions already done
    all_done = all(
        os.path.exists(os.path.join(output_dir, f"{model_name}_{cond}.json"))
        for cond in CONDITIONS
    )
    if all_done:
        print(f"All conditions done for {model_name}, skipping")
        return

    print(f"\n{'='*60}")
    print(f"Loading {model_name} ({model_id})...")
    print(f"GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    print(f"{'='*60}")

    llm = LLM(
        model=model_id,
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.90,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=max(temperature, 0.01),
        top_p=0.95,
    )

    for cond in CONDITIONS:
        output_path = os.path.join(output_dir, f"{model_name}_{cond}.json")
        if os.path.exists(output_path):
            print(f"  Skipping {cond} (exists)")
            continue

        prompts = load_prompts(cond, data_dir)

        # Format prompts using chat template
        formatted = []
        for p in prompts:
            messages = [{"role": "user", "content": p["prompt"]}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            formatted.append(text)

        print(f"  Running {cond.upper()} ({len(prompts)} prompts)...")
        start = time.time()
        outputs = llm.generate(formatted, sampling_params)
        elapsed = time.time() - start
        print(f"  Done in {elapsed:.1f}s ({len(prompts)/elapsed:.1f} prompts/s)")

        results = []
        for p, output in zip(prompts, outputs):
            results.append(
                {
                    "id": p["id"],
                    "prompt": p["prompt"],
                    "category": p["category"],
                    "condition": cond,
                    "response": output.outputs[0].text,
                }
            )

        output_data = {
            "model_name": model_name,
            "model_id": model_id,
            "condition": cond,
            "num_prompts": len(prompts),
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "results": results,
        }
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
    print(f"\n{model_name}: All conditions complete!")


def launch_parallel(args):
    """Launch all models in parallel, one GPU per model."""
    processes = []
    for model_idx in range(len(MODEL_CONFIGS)):
        model_name = MODEL_CONFIGS[model_idx][0]

        # Check if all done
        all_done = all(
            os.path.exists(os.path.join(args.output_dir, f"{model_name}_{cond}.json"))
            for cond in CONDITIONS
        )
        if all_done:
            print(f"Skipping {model_name} (all conditions done)")
            continue

        gpu_id = model_idx % args.num_gpus
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        cmd = [
            sys.executable,
            SCRIPT_PATH,
            "--model-index",
            str(model_idx),
            "--output-dir",
            args.output_dir,
            "--data-dir",
            args.data_dir,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--temperature",
            str(args.temperature),
        ]

        log_dir = os.path.join(args.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{model_name}.log")

        print(f"Launching {model_name} on GPU {gpu_id} → {log_path}")
        log_f = open(log_path, "w")
        p = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
        processes.append((p, log_f, model_name))

    if not processes:
        print("Nothing to launch (all done).")
        return

    print(f"\nWaiting for {len(processes)} model processes...")
    for p, log_f, name in processes:
        p.wait()
        log_f.close()
        status = "OK" if p.returncode == 0 else f"FAILED (rc={p.returncode})"
        print(f"  {name}: {status}")

    print("\nAll inference complete!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-index", type=int, default=None)
    parser.add_argument("--output-dir", default=RESULTS_DIR)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.model_index is not None:
        run_model_all_conditions(
            args.model_index,
            args.output_dir,
            args.data_dir,
            args.max_new_tokens,
            args.temperature,
        )
    elif args.parallel:
        launch_parallel(args)
    else:
        # Sequential
        for idx in range(len(MODEL_CONFIGS)):
            run_model_all_conditions(
                idx, args.output_dir, args.data_dir, args.max_new_tokens, args.temperature
            )
        print("All models complete!")


if __name__ == "__main__":
    main()
