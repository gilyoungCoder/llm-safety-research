# Safety Speaks English
### Cross-Lingual Safety Erosion in Open-Source Large Language Models

> ETRI Journal — Special Issue on Trustworthy and Safe AI (2026)

## Overview

Bilingual (Korean + Chinese) evaluation of cross-lingual safety erosion
in 6 open-source LLMs across 17 prompt conditions and 3 translation engines.

**53,040 total generations** | **6 models** | **17 conditions** | **3 engines**

## Key Results

| Model | EN ASR | KO ASR | ZH ASR | KO/ZH Ratio |
|-------|:------:|:------:|:------:|:-----------:|
| Llama-8B | 6.7% | 25.6% | 6.2% | 3.0x |
| Qwen-7B | 1.3% | 8.1% | 3.8% | 1.7x |
| Qwen-14B | 2.3% | 10.8% | 3.5% | 3.0x |
| Phi-3-14B | 1.3% | 47.1% | 7.3% | 7.6x |
| **Yi-9B** | 8.7% | **66.2%** | 5.4% | **14.3x** |
| **InternLM-7B** | 1.3% | **45.2%** | 4.8% | **14.9x** |

## Principal Findings

1. **Safety erosion is language-specific** — Korean ASR 33.8% vs Chinese 5.2%
2. **No cross-CJK transfer** — Yi/InternLM safe in Chinese, catastrophic in Korean
3. **Two-tier pattern** — Translation quality only helps models with target-language safety data

## Conditions

| # | Korean (9) | # | Chinese (8) |
|---|-----------|---|------------|
| 1 | EN (baseline) | 10 | ZH (NLLB) |
| 2 | KO (NLLB) | 11 | MIX-ZH |
| 3 | MIX-KO | 12 | ZH→EN (NLLB) |
| 4 | KO→EN (NLLB) | 13 | ZH (Gemini) |
| 5 | KO (Gemini) | 14 | ZH→EN (Gemini) |
| 6 | KO→EN (Gemini) | 15 | ZH (GPT) |
| 7 | KO (GPT) | 16 | ZH→EN (GPT) |
| 8 | KO→EN (GPT) | 17 | WS-ZH |
| 9 | WS-KO | | |

## Pipeline

```bash
# Full pipeline
python scripts/evaluate.py      # Trilingual refusal detection
python scripts/analyze.py       # Build ASR matrices + LaTeX tables
python scripts/visualize.py     # Generate 17 publication-quality figures
```

## Files

```
scripts/
  evaluate.py             # Refusal detection (EN/KO/ZH patterns)
  analyze.py              # Bilingual ASR analysis + LaTeX table generation
  visualize.py            # 17 figures (7 main + 10 appendix)
  run_inference.py        # Korean inference (base 4 models)
  run_chinese_inference.py # Chinese inference (6 models x 8 conditions)
  translate.py            # NLLB-200 translation
  translate_chinese.py    # Chinese NLLB translation
  generate_zh_adaptive_mixed.py  # Chinese word-substitution generation
data/
  advbench_*.json         # 25 condition-specific prompt files
paper/
  main.tex                # Full manuscript
figures/
  *.png, *.pdf            # All generated figures
  main_table.tex          # Compact bilingual results table
  analysis.json           # Structured analysis data
```
