"""
Run inference on standard instruct models vs reasoning (DeepSeek-R1 distill) models.

Supports:
  1. Single GPU mode: one model, all prompts
  2. Sharded mode: one model, subset of prompts (for multi-GPU parallelism)
  3. Local parallel: launch shards across local GPUs
  4. Distributed: generate commands for multi-server execution

Usage:
  # Single shard (called by launcher, one GPU per process)
  CUDA_VISIBLE_DEVICES=0 python run_inference.py --model-index 0 --shard-id 0 --num-shards 4

  # Local parallel (single server, 8 GPUs, 6 models × shards)
  python run_inference.py --parallel --num-gpus 8

  # Generate distributed commands for multi-server
  python run_inference.py --distributed --servers server1,server2,server3
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_PATH = os.path.abspath(__file__)
RESULTS_DIR = os.path.join(os.path.dirname(SCRIPT_PATH), "..", "results")
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_PATH), "..", "data")

# Model configurations: (display_name, hf_model_id, model_type, min_vram_gb)
MODEL_CONFIGS = [
    ("Llama-3.1-8B-Instruct", "unsloth/Meta-Llama-3.1-8B-Instruct", "standard", 18),
    ("Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "standard", 16),
    ("Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-14B-Instruct", "standard", 30),
    ("DeepSeek-R1-Distill-Llama-8B", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "reasoning", 18),
    ("DeepSeek-R1-Distill-Qwen-7B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "reasoning", 16),
    ("DeepSeek-R1-Distill-Qwen-14B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "reasoning", 30),
]


def load_prompts(data_path):
    with open(data_path, "r") as f:
        return json.load(f)


def shard_prompts(prompts, shard_id, num_shards):
    """Split prompts into shards. Each shard gets every N-th prompt."""
    return [p for i, p in enumerate(prompts) if i % num_shards == shard_id]


def build_messages(prompt_text):
    return [{"role": "user", "content": prompt_text}]


def run_with_vllm(model_id, prompts, max_new_tokens=512, temperature=0.0):
    from vllm import LLM, SamplingParams

    print(f"  Loading {model_id} with vLLM...")
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

    formatted_prompts = []
    for p in prompts:
        messages = build_messages(p["prompt"])
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        formatted_prompts.append(text)

    print(f"  Generating {len(formatted_prompts)} responses...")
    start = time.time()
    outputs = llm.generate(formatted_prompts, sampling_params)
    elapsed = time.time() - start
    print(f"  Done in {elapsed:.1f}s ({len(formatted_prompts)/elapsed:.1f} prompts/s)")

    results = []
    for prompt_data, output in zip(prompts, outputs):
        results.append({
            "id": prompt_data["id"],
            "prompt": prompt_data["prompt"],
            "category": prompt_data["category"],
            "response": output.outputs[0].text,
        })

    del llm
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def run_with_transformers(model_id, prompts, max_new_tokens=512, temperature=0.0):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading {model_id} with transformers...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True,
        torch_dtype=torch.float16, device_map="auto",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    start = time.time()
    for i, p in enumerate(prompts):
        messages = build_messages(p["prompt"])
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 0.01) if temperature > 0 else None,
                top_p=0.95 if temperature > 0 else None,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        results.append({"id": p["id"], "prompt": p["prompt"], "category": p["category"], "response": generated})
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(prompts)} done...")

    elapsed = time.time() - start
    print(f"  Done in {elapsed:.1f}s")
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return results


def run_shard(model_index, shard_id, num_shards, data_path, output_dir,
              backend="vllm", max_new_tokens=512, temperature=0.0):
    """Run one shard of one model."""
    model_name, model_id, model_type, _ = MODEL_CONFIGS[model_index]

    # Output path includes shard info
    if num_shards > 1:
        output_path = os.path.join(output_dir, "shards",
                                   f"{model_name}_shard{shard_id}of{num_shards}.json")
    else:
        output_path = os.path.join(output_dir, f"{model_name}.json")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        print(f"Skipping {model_name} shard {shard_id}/{num_shards} (exists)")
        return

    prompts = load_prompts(data_path)
    if num_shards > 1:
        prompts = shard_prompts(prompts, shard_id, num_shards)

    print(f"\n{'='*60}")
    print(f"Model: {model_name} | Shard: {shard_id}/{num_shards} | Prompts: {len(prompts)}")
    print(f"GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    print(f"{'='*60}")

    run_fn = run_with_vllm if backend == "vllm" else run_with_transformers
    try:
        results = run_fn(model_id, prompts, max_new_tokens, temperature)
    except ImportError:
        print("  vLLM not available, falling back to transformers...")
        results = run_with_transformers(model_id, prompts, max_new_tokens, temperature)

    output_data = {
        "model_name": model_name,
        "model_id": model_id,
        "model_type": model_type,
        "shard_id": shard_id,
        "num_shards": num_shards,
        "num_prompts": len(prompts),
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved to {output_path}")


def merge_shards(output_dir):
    """Merge shard results into per-model files."""
    shards_dir = os.path.join(output_dir, "shards")
    if not os.path.exists(shards_dir):
        print("No shards directory found, skipping merge.")
        return

    from collections import defaultdict
    model_shards = defaultdict(list)

    for fpath in sorted(Path(shards_dir).glob("*.json")):
        with open(fpath) as f:
            data = json.load(f)
        model_shards[data["model_name"]].append(data)

    for model_name, shards in model_shards.items():
        # Check all shards present
        expected = shards[0]["num_shards"]
        if len(shards) < expected:
            print(f"  WARNING: {model_name} has {len(shards)}/{expected} shards, skipping merge")
            continue

        # Merge results, sort by original ID
        all_results = []
        for s in shards:
            all_results.extend(s["results"])
        all_results.sort(key=lambda x: x["id"])

        merged = {
            "model_name": model_name,
            "model_id": shards[0]["model_id"],
            "model_type": shards[0]["model_type"],
            "num_prompts": len(all_results),
            "max_new_tokens": shards[0]["max_new_tokens"],
            "temperature": shards[0]["temperature"],
            "results": all_results,
        }

        out_path = os.path.join(output_dir, f"{model_name}.json")
        with open(out_path, "w") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
        print(f"  Merged {len(shards)} shards → {out_path} ({len(all_results)} prompts)")

    print("Merge complete!")


def launch_parallel_sharded(args, data_path):
    """Launch all models × shards in parallel across local GPUs."""
    num_gpus = args.num_gpus
    num_models = len(MODEL_CONFIGS)

    # 14B models need A6000 (≥48GB). If on 3090 server, skip them.
    # For simplicity, assume all GPUs can handle all models (user routes 14B to A6000 servers)

    # Assign GPUs: distribute evenly
    # With 8 GPUs and 6 models: some models get 1 GPU, some get 2
    # More GPUs per model = more shards = faster
    shards_per_model = max(1, num_gpus // num_models)
    total_slots = num_models * shards_per_model

    print(f"GPUs: {num_gpus} | Models: {num_models} | Shards/model: {shards_per_model}")
    print(f"Total parallel processes: {min(total_slots, num_gpus)}")

    processes = []
    gpu_id = 0

    for model_idx in range(num_models):
        model_name = MODEL_CONFIGS[model_idx][0]

        # Check if already merged
        merged_path = os.path.join(args.output_dir, f"{model_name}.json")
        if os.path.exists(merged_path):
            print(f"Skipping {model_name} (merged result exists)")
            continue

        for shard_id in range(shards_per_model):
            # Check if shard exists
            shard_path = os.path.join(args.output_dir, "shards",
                                      f"{model_name}_shard{shard_id}of{shards_per_model}.json")
            if os.path.exists(shard_path):
                print(f"Skipping {model_name} shard {shard_id} (exists)")
                continue

            assigned_gpu = str(gpu_id % num_gpus)
            gpu_id += 1

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = assigned_gpu

            cmd = [
                sys.executable, SCRIPT_PATH,
                "--model-index", str(model_idx),
                "--shard-id", str(shard_id),
                "--num-shards", str(shards_per_model),
                "--data", data_path,
                "--output-dir", args.output_dir,
                "--backend", args.backend,
                "--max-new-tokens", str(args.max_new_tokens),
                "--temperature", str(args.temperature),
            ]

            log_path = os.path.join(args.output_dir, "logs",
                                    f"{model_name}_shard{shard_id}.log")
            os.makedirs(os.path.dirname(log_path), exist_ok=True)

            print(f"  Launch: {model_name} shard {shard_id}/{shards_per_model} → GPU {assigned_gpu}")
            log_f = open(log_path, "w")
            p = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
            processes.append((p, log_f, model_name, shard_id))

    if not processes:
        print("Nothing to launch (all done).")
        return

    print(f"\nWaiting for {len(processes)} processes...")
    for p, log_f, model_name, shard_id in processes:
        p.wait()
        log_f.close()
        status = "OK" if p.returncode == 0 else f"FAILED ({p.returncode})"
        print(f"  {model_name} shard {shard_id}: {status}")

    # Auto-merge
    print("\nMerging shards...")
    merge_shards(args.output_dir)


def generate_distributed_commands(args, data_path):
    """Generate per-server shell commands for multi-server execution."""
    # Server config
    servers = []
    for s in args.servers.split(","):
        parts = s.strip().split(":")
        hostname = parts[0]
        gpu_type = parts[1] if len(parts) > 1 else "3090"
        num_gpus = int(parts[2]) if len(parts) > 2 else 8
        vram = 48 if "6000" in gpu_type or "a100" in gpu_type.lower() else 24
        servers.append({"host": hostname, "gpu_type": gpu_type, "num_gpus": num_gpus, "vram": vram})

    total_gpus = sum(s["num_gpus"] for s in servers)
    print(f"Servers: {len(servers)} | Total GPUs: {total_gpus}")

    # Separate models by VRAM requirement
    small_models = [(i, c) for i, c in enumerate(MODEL_CONFIGS) if c[3] <= 24]  # 7B/8B
    large_models = [(i, c) for i, c in enumerate(MODEL_CONFIGS) if c[3] > 24]   # 14B

    small_gpus = [s for s in servers]  # All servers can run small models
    large_gpus = [s for s in servers if s["vram"] >= 48]  # Only A6000+ for 14B

    # Calculate shards per model
    small_gpu_count = sum(s["num_gpus"] for s in small_gpus)
    large_gpu_count = sum(s["num_gpus"] for s in large_gpus)

    # Reserve GPUs for large models on A6000 servers
    large_shards_per_model = max(1, large_gpu_count // max(len(large_models), 1))
    large_gpus_used = len(large_models) * large_shards_per_model

    # Remaining GPUs for small models
    remaining_gpus = total_gpus - large_gpus_used
    small_shards_per_model = max(1, remaining_gpus // max(len(small_models), 1))

    print(f"Small models ({len(small_models)}): {small_shards_per_model} shards each")
    print(f"Large models ({len(large_models)}): {large_shards_per_model} shards each")

    # Generate commands
    script_dir = os.path.dirname(SCRIPT_PATH)
    all_commands = {}

    gpu_assignments = []  # (server_idx, gpu_id, model_idx, shard_id, num_shards)

    # Assign large models to A6000 servers first
    gpu_cursor = {s["host"]: 0 for s in servers}
    large_server_idx = [i for i, s in enumerate(servers) if s["vram"] >= 48]

    for model_idx, model_cfg in large_models:
        num_shards = large_shards_per_model
        for shard_id in range(num_shards):
            # Round-robin across large GPU servers
            si = large_server_idx[shard_id % len(large_server_idx)]
            server = servers[si]
            gpu_id = gpu_cursor[server["host"]]
            gpu_cursor[server["host"]] = (gpu_id + 1) % server["num_gpus"]
            gpu_assignments.append((si, gpu_id, model_idx, shard_id, num_shards))

    # Assign small models across all servers
    all_server_idx = list(range(len(servers)))
    cursor = 0
    for model_idx, model_cfg in small_models:
        num_shards = small_shards_per_model
        for shard_id in range(num_shards):
            si = all_server_idx[cursor % len(all_server_idx)]
            server = servers[si]
            gpu_id = gpu_cursor[server["host"]]
            gpu_cursor[server["host"]] = (gpu_id + 1) % server["num_gpus"]
            gpu_assignments.append((si, gpu_id, model_idx, shard_id, num_shards))
            cursor += 1

    # Group by server and generate scripts
    for si, server in enumerate(servers):
        server_cmds = []
        for s_si, gpu_id, model_idx, shard_id, num_shards in gpu_assignments:
            if s_si != si:
                continue
            model_name = MODEL_CONFIGS[model_idx][0]
            cmd = (f"CUDA_VISIBLE_DEVICES={gpu_id} python {SCRIPT_PATH} "
                   f"--model-index {model_idx} --shard-id {shard_id} --num-shards {num_shards} "
                   f"--data {data_path} --output-dir {args.output_dir} "
                   f"--backend {args.backend} --max-new-tokens {args.max_new_tokens} "
                   f"--temperature {args.temperature} "
                   f"> {args.output_dir}/logs/{model_name}_shard{shard_id}.log 2>&1 &")
            server_cmds.append(cmd)

        all_commands[server["host"]] = server_cmds

    # Write per-server scripts
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)
    launch_dir = os.path.join(os.path.dirname(SCRIPT_PATH), "..", "launch")
    os.makedirs(launch_dir, exist_ok=True)

    master_lines = ["#!/bin/bash", "# Master launch script — run from any server with shared filesystem",
                    f"# Generated at {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]

    for host, cmds in all_commands.items():
        script_path = os.path.join(launch_dir, f"run_{host}.sh")
        lines = [
            "#!/bin/bash",
            f"# Server: {host} — {len(cmds)} GPU tasks",
            f"cd {os.path.dirname(SCRIPT_PATH)}/..",
            "",
        ]
        lines.extend(cmds)
        lines.append("")
        lines.append(f'echo "All {len(cmds)} tasks launched on {host}. Check logs in {args.output_dir}/logs/"')
        lines.append("wait")
        lines.append('echo "All tasks on this server complete."')

        with open(script_path, "w") as f:
            f.write("\n".join(lines))
        os.chmod(script_path, 0o755)
        print(f"  Created {script_path} ({len(cmds)} tasks)")

        master_lines.append(f"# {host}: {len(cmds)} tasks")
        master_lines.append(f"ssh {host} 'bash {script_path}' &")
        master_lines.append("")

    master_lines.extend(["wait", "", "# Merge shards after all servers complete",
                         f"python {SCRIPT_PATH} --merge --output-dir {args.output_dir}",
                         'echo "All done!"'])

    master_path = os.path.join(launch_dir, "launch_all.sh")
    with open(master_path, "w") as f:
        f.write("\n".join(master_lines))
    os.chmod(master_path, 0o755)
    print(f"\n  Master script: {master_path}")
    print(f"  Run: bash {master_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join(DATA_DIR, "advbench_harmful_behaviors.json"))
    parser.add_argument("--output-dir", default=RESULTS_DIR)
    parser.add_argument("--backend", choices=["vllm", "transformers"], default="vllm")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-prompts", type=int, default=None)

    # Single shard mode
    parser.add_argument("--model-index", type=int, default=None)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)

    # Local parallel mode
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=8)

    # Distributed mode
    parser.add_argument("--distributed", action="store_true",
                        help="Generate per-server launch scripts")
    parser.add_argument("--servers", type=str, default=None,
                        help="Comma-separated: host1:gpu_type:num, e.g. 'srv1:3090:8,srv2:a6000:8'")

    # Merge mode
    parser.add_argument("--merge", action="store_true")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Handle max-prompts
    data_path = args.data
    if args.max_prompts:
        prompts = load_prompts(args.data)[:args.max_prompts]
        data_path = os.path.join(args.output_dir, "_prompts_subset.json")
        with open(data_path, "w") as f:
            json.dump(prompts, f)

    if args.merge:
        merge_shards(args.output_dir)
    elif args.model_index is not None:
        run_shard(args.model_index, args.shard_id, args.num_shards,
                  data_path, args.output_dir, args.backend,
                  args.max_new_tokens, args.temperature)
    elif args.distributed:
        if not args.servers:
            parser.error("--distributed requires --servers")
        generate_distributed_commands(args, data_path)
    elif args.parallel:
        launch_parallel_sharded(args, data_path)
    else:
        # Sequential: run all models one by one
        prompts = load_prompts(data_path)
        print(f"Loaded {len(prompts)} prompts, running sequentially...")
        for idx in range(len(MODEL_CONFIGS)):
            run_shard(idx, 0, 1, data_path, args.output_dir,
                      args.backend, args.max_new_tokens, args.temperature)
        print("All models complete!")


if __name__ == "__main__":
    main()
