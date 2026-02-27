# Think Before You Refuse
### Safety Alignment Gaps in Open-Source Reasoning Models

> ETRI Journal — Special Issue on Trustworthy and Safe AI (2026)

## Overview

Comparative evaluation of safety alignment between standard instruction-tuned
models and reasoning-distilled models (DeepSeek-R1 family) on AdvBench
(520 harmful behaviors).

**3,120 total generations** | **6 models** | **3 paired comparisons**

## Key Results

| Base Model | Standard ASR | Reasoning ASR | Degradation |
|-----------|:-----------:|:------------:|:-----------:|
| Llama-8B | 5.0% | 21.9% | +16.9 pp |
| Qwen-7B | 1.0% | 24.8% | +23.8 pp |
| Qwen-14B | 1.5% | 20.2% | +18.7 pp |

Reasoning distillation **consistently degrades safety** across all model families,
with ASR increasing 4-25x compared to standard instruction-tuned counterparts.

## Principal Findings

1. **Reasoning distillation weakens safety** — Mean ASR jumps from 2.5% to 22.3%
2. **Self-rationalization** — Models reason themselves into compliance via chain-of-thought
3. **Consistent across families** — Effect holds for Llama, Qwen-7B, and Qwen-14B bases

## Pipeline

```bash
python scripts/prepare_data.py        # Prepare AdvBench data
python scripts/run_inference.py       # Run 6 models on AdvBench
python scripts/evaluate.py            # Refusal detection + ASR
python scripts/visualize.py           # Main figures
python scripts/advanced_analysis.py   # Deep analysis
python scripts/extra_figures.py       # Additional figures
```

## Files

```
scripts/
  prepare_data.py          # Data preparation
  run_inference.py         # vLLM inference for 6 models
  evaluate.py              # Refusal detection + metrics
  visualize.py             # Main publication figures
  advanced_analysis.py     # Thinking pattern analysis
  extra_figures.py         # Extended figures
  generate_all_figures.py  # Batch figure generation
data/
  advbench_harmful_behaviors.json
  harmful_behaviors.csv
paper/
  main.tex                 # Full manuscript
figures/
  *.png                    # All generated figures
```
