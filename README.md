# LLM Safety Research

Two research projects on safety alignment in open-source LLMs,
targeting **ETRI Journal** Special Issue on Trustworthy and Safe AI.

## Projects

### [`cross_lingual_safety/`](cross_lingual_safety/)
**Safety Speaks English: Cross-Lingual Safety Erosion in Open-Source LLMs**

Bilingual (Korean + Chinese) evaluation of how safety alignment degrades
across languages. 6 models, 17 conditions, 3 translation engines, 53,040 generations.

Key finding: Safety erosion is **language-specific** — Korean triggers severe erosion
(mean ASR 33.8%) while Chinese remains safe (5.2%), proving it's a per-language
training coverage problem, not a generic non-English weakness.

### [`reasoning_safety/`](reasoning_safety/)
**Think Before You Refuse: Safety Alignment Gaps in Reasoning Models**

Evaluation of how reasoning-distilled models (DeepSeek-R1 family) differ
from standard instruction-tuned models in safety alignment.
6 models on AdvBench (520 harmful behaviors).

Key finding: Reasoning models exhibit **self-rationalization** patterns where
explicit chain-of-thought reasoning can lead to compliance despite safety training.

## Quick Start

Each project follows the same pipeline:

```bash
cd cross_lingual_safety/  # or reasoning_safety/

# 1. Prepare data
python scripts/prepare_data.py  # if applicable

# 2. Run inference (requires GPUs + vLLM)
bash run_all.sh

# 3. Evaluate + Analyze + Visualize
python scripts/evaluate.py
python scripts/analyze.py
python scripts/visualize.py
```

## Structure

```
project/
├── scripts/        # Full pipeline (inference, evaluation, analysis, visualization)
├── data/           # Benchmark data (AdvBench)
├── paper/          # LaTeX manuscript
├── figures/        # Generated figures (PNG + PDF)
├── results/        # evaluation_summary.json (full results excluded for size)
└── run_all.sh      # Orchestration script
```

> **Note**: Individual model result files (~200MB) are excluded from the repo.
> Run the inference pipeline to regenerate, or contact the authors.

## Tech Stack

- **Inference**: vLLM with greedy decoding on NVIDIA A6000 GPUs
- **Models**: Llama-3.1-8B, Qwen-2.5-7B/14B, Phi-3-14B, Yi-1.5-9B, InternLM-2.5-7B
- **Translation**: NLLB-200, Gemini 3.0 Pro, GPT-5.2
- **Evaluation**: Trilingual keyword-based refusal detection (EN/KO/ZH)
