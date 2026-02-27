# Think Before You Refuse
### Safety Alignment Gaps in Open-Source Reasoning Models

> **ETRI Journal** — Special Issue on *Trustworthy and Safe AI* (2026)
>
> **3,120 generations** · **6 models** · **3 paired comparisons** · **4-type thinking taxonomy**

---

## Headline Result

<p align="center">
<img src="figures/fig1_slope.png" width="85%"/>
</p>

<p align="center">
<em>Reasoning distillation consistently degrades safety: ASR jumps from 2.5% (standard) to 22.3% (reasoning) across all three model families.</em>
</p>

---

## Key Numbers

| Base Model | Standard | Reasoning | Amplification | Worst Category |
|-----------|:--------:|:---------:|:-------------:|:--------------:|
| Llama-3.1-8B | 5.0% | 21.9% | **4.4x** | Deception (56%) |
| Qwen2.5-7B | 1.0% | 24.8% | **25.8x** | Deception (50%) |
| Qwen2.5-14B | 1.5% | 20.2% | **13.2x** | Deception (36%) |
| | **Mean 2.5%** | **Mean 22.3%** | **8.9x** | |

---

## Principal Findings

### 1. Reasoning Distillation Weakens Safety

<table>
<tr>
<td width="55%">

Standard instruction-tuned models refuse **95–99%** of harmful prompts.
Reasoning-distilled variants refuse only **75–80%**.

Mean ASR jumps from **2.5% → 22.3%** — a consistent,
family-independent degradation that proves the effect
stems from the **training paradigm**, not model-specific quirks.

</td>
<td width="45%">
<img src="figures/figA1_paired_asr.png" width="100%"/>
</td>
</tr>
</table>

### 2. Category-Selective Vulnerability

<table>
<tr>
<td width="45%">
<img src="figures/fig2_category_gap.png" width="100%"/>
</td>
<td width="55%">

Safety degradation is **not uniform** across harm categories:

| Category | Standard | Reasoning | Gap |
|----------|:--------:|:---------:|:---:|
| Deception | 91% | **48%** | -43 pp |
| Violence | 97% | 77% | -20 pp |
| Other | 95% | 79% | -16 pp |
| Illegal Activity | 97% | 83% | -14 pp |
| Self-Harm | 100% | 95% | -5 pp |
| Discrimination | 100% | 100% | 0 pp |
| Sexual | 100% | 100% | 0 pp |

**Deception collapses** while discrimination and sexual content remain fully safe.

</td>
</tr>
</table>

### 3. Self-Rationalization: A New Failure Mode

<table>
<tr>
<td width="55%">

We introduce a **four-type taxonomy** of reasoning traces:

| Type | Description | Prevalence |
|------|-------------|:----------:|
| **A** — Safety → Refuse | Recognizes harm, refuses | 77% |
| **B** — Safety → Comply | Recognizes harm, **complies anyway** | 7–9% |
| **C** — No Safety | Zero safety awareness | 13–16% |
| **D** — Other Refuse | Refuses on other grounds | 1–2% |

**Type B (self-rationalization)** is the most concerning:
models *explicitly* recognize the harm but reason
themselves into compliance through chain-of-thought.

</td>
<td width="45%">
<img src="figures/fig3_trace_donuts.png" width="100%"/>
</td>
</tr>
</table>

### 4. Category × Trace Interaction

<p align="center">
<img src="figures/fig4_crosstab.png" width="80%"/>
</p>

<p align="center">
<em>Deception and violence prompts trigger more Type B (self-rationalization) and Type C (safety-blind) traces. Discrimination and sexual prompts almost always trigger Type A (safe refusal).</em>
</p>

### 5. Scale Does Not Fix the Problem

<table>
<tr>
<td width="45%">
<img src="figures/fig5_scale.png" width="100%"/>
</td>
<td width="55%">

Comparing Qwen-7B → Qwen-14B (2x parameter increase):

- Standard ASR: 1.0% → 1.5% (negligible)
- Reasoning ASR: **24.8% → 20.2%** (modest improvement)

Scaling alone provides only **marginal safety recovery** (−4.6 pp).
The fundamental vulnerability from reasoning distillation
persists regardless of model size.

</td>
</tr>
</table>

---

## Models

| Model | Type | Base | Params | Developer |
|-------|:----:|------|:------:|-----------|
| Llama-3.1-8B-Instruct | Standard | Llama-3.1 | 8B | Meta |
| DeepSeek-R1-Distill-Llama-8B | Reasoning | Llama-3.1 | 8B | DeepSeek |
| Qwen2.5-7B-Instruct | Standard | Qwen-2.5 | 7B | Alibaba |
| DeepSeek-R1-Distill-Qwen-7B | Reasoning | Qwen-2.5 | 7B | DeepSeek |
| Qwen2.5-14B-Instruct | Standard | Qwen-2.5 | 14B | Alibaba |
| DeepSeek-R1-Distill-Qwen-14B | Reasoning | Qwen-2.5 | 14B | DeepSeek |

---

## Pipeline

```bash
# Step 1: Prepare benchmark data
python scripts/prepare_data.py

# Step 2: Inference (requires GPUs + vLLM)
python scripts/run_inference.py

# Step 3: Evaluate + Visualize + Analyze
python scripts/evaluate.py            # Refusal detection + ASR
python scripts/visualize.py           # 5 main figures
python scripts/advanced_analysis.py   # Thinking trace analysis
python scripts/extra_figures.py       # 6 appendix figures
```

---

## Appendix Figures

<details>
<summary><b>Paired ASR Comparison (Standard vs Reasoning)</b></summary>
<p align="center">
<img src="figures/figA1_paired_asr.png" width="85%"/>
</p>
<em>Side-by-side ASR comparison clearly showing the consistent safety gap between standard and reasoning variants.</em>
</details>

<details>
<summary><b>Category Radar Charts</b></summary>
<p align="center">
<img src="figures/figA2_radar.png" width="85%"/>
</p>
<em>Reasoning models (red) show expanded vulnerability polygons, especially in deception and violence categories.</em>
</details>

<details>
<summary><b>Response Length Distribution</b></summary>
<p align="center">
<img src="figures/figA3_length.png" width="85%"/>
</p>
<em>Reasoning models generate significantly longer responses (~2,400 tokens avg) vs standard models (~500 tokens), reflecting the extended thinking process.</em>
</details>

<details>
<summary><b>Qualitative Examples</b></summary>
<p align="center">
<img src="figures/figA4_qualitative.png" width="85%"/>
</p>
<em>Examples of Type B self-rationalization: model recognizes harm in thinking trace, then reasons itself into compliance.</em>
</details>

<details>
<summary><b>Combined Overview</b></summary>
<p align="center">
<img src="figures/figA5_combined.png" width="90%"/>
</p>
</details>

<details>
<summary><b>Category Heatmap</b></summary>
<p align="center">
<img src="figures/figA6_heatmap.png" width="85%"/>
</p>
<em>Full category × model heatmap showing where safety degradation concentrates.</em>
</details>

---

## File Structure

```
reasoning_safety/
├── scripts/
│   ├── prepare_data.py              # AdvBench data preparation
│   ├── run_inference.py             # vLLM inference for 6 models
│   ├── evaluate.py                  # Refusal detection + ASR metrics
│   ├── visualize.py                 # 5 main publication figures
│   ├── advanced_analysis.py         # Thinking trace taxonomy analysis
│   ├── extra_figures.py             # 6 appendix figures
│   └── generate_all_figures.py      # Batch figure generation
├── data/
│   ├── advbench_harmful_behaviors.json
│   └── harmful_behaviors.csv        # AdvBench — 520 behaviors, 7 categories
├── paper/
│   └── main.tex                     # Full LaTeX manuscript
├── figures/
│   ├── fig1–fig5                    # 5 main figures
│   ├── figA1–figA6                  # 6 appendix figures
│   └── *.png                        # 24 total figures
└── results/
    └── evaluation_summary.json      # Aggregated metrics
```
