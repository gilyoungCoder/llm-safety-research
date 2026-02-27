"""
Analyze cross-lingual safety evaluation results — bilingual (Korean + Chinese).
Produces structured analysis data for visualization and paper tables.

Usage:
  python3 analyze.py --results-dir ../results
"""

import argparse
import json
import os
from collections import defaultdict

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

# Model display order and capability
MODEL_ORDER = [
    "Llama-3.1-8B-Instruct",
    "Qwen2.5-7B-Instruct",
    "Qwen2.5-14B-Instruct",
    "Phi-3-medium-4k-instruct",
    "Yi-1.5-9B-Chat",
    "internlm2_5-7b-chat",
]

MODEL_SHORT = {
    "Llama-3.1-8B-Instruct": "Llama-8B",
    "Qwen2.5-7B-Instruct": "Qwen-7B",
    "Qwen2.5-14B-Instruct": "Qwen-14B",
    "Phi-3-medium-4k-instruct": "Phi-3-14B",
    "Yi-1.5-9B-Chat": "Yi-9B",
    "internlm2_5-7b-chat": "InternLM-7B",
}

MODEL_KO_CAPABILITY = {
    "Llama-3.1-8B-Instruct": 0.35,
    "Qwen2.5-7B-Instruct": 0.55,
    "Qwen2.5-14B-Instruct": 0.65,
    "Phi-3-medium-4k-instruct": 0.38,
    "Yi-1.5-9B-Chat": 0.42,
    "internlm2_5-7b-chat": 0.48,
}

MODEL_ZH_CAPABILITY = {
    "Llama-3.1-8B-Instruct": 0.40,
    "Qwen2.5-7B-Instruct": 0.70,
    "Qwen2.5-14B-Instruct": 0.78,
    "Phi-3-medium-4k-instruct": 0.42,
    "Yi-1.5-9B-Chat": 0.68,
    "internlm2_5-7b-chat": 0.65,
}

# ── Condition definitions ──
KO_CONDITION_ORDER = ["en", "ko", "mix", "ko2en", "adaptive", "ko_gemini", "ko2en_gemini", "ko_gpt", "ko2en_gpt"]
ZH_CONDITION_ORDER = ["en", "zh", "mix_zh", "zh2en", "adaptive_zh", "zh_gemini", "zh2en_gemini", "zh_gpt", "zh2en_gpt"]
ALL_CONDITION_ORDER = [
    "en",
    "ko", "mix", "ko2en", "adaptive", "ko_gemini", "ko2en_gemini", "ko_gpt", "ko2en_gpt",
    "zh", "mix_zh", "zh2en", "adaptive_zh", "zh_gemini", "zh2en_gemini", "zh_gpt", "zh2en_gpt",
]

CONDITION_LABELS = {
    "en": "English",
    "ko": "KO (NLLB)", "mix": "MIX-KO", "ko2en": "KO→EN (NLLB)",
    "adaptive": "WS-KO", "ko_gemini": "KO (Gemini)", "ko2en_gemini": "KO→EN (Gemini)",
    "ko_gpt": "KO (GPT)", "ko2en_gpt": "KO→EN (GPT)",
    "zh": "ZH (NLLB)", "mix_zh": "MIX-ZH", "zh2en": "ZH→EN (NLLB)",
    "adaptive_zh": "WS-ZH", "zh_gemini": "ZH (Gemini)", "zh2en_gemini": "ZH→EN (Gemini)",
    "zh_gpt": "ZH (GPT)", "zh2en_gpt": "ZH→EN (GPT)",
}

CATEGORY_ORDER = [
    "deception", "violence", "illegal_activity", "other",
    "self_harm", "discrimination", "sexual_content",
]


def load_evaluation_summary(results_dir):
    path = os.path.join(results_dir, "evaluation_summary.json")
    with open(path) as f:
        return json.load(f)


def build_asr_matrix(metrics_list):
    matrix = {}
    for m in metrics_list:
        model = m["model_name"]
        cond = m["condition"]
        if model not in matrix:
            matrix[model] = {}
        matrix[model][cond] = m["asr"]
    return matrix


def compute_safety_degradation(asr_matrix, target_cond="ko"):
    degradation = {}
    for model in asr_matrix:
        en_asr = asr_matrix[model].get("en", 0)
        tgt_asr = asr_matrix[model].get(target_cond, 0)
        degradation[model] = {
            "en_asr": en_asr,
            "target_asr": tgt_asr,
            "delta_asr": tgt_asr - en_asr,
            "amplification": tgt_asr / en_asr if en_asr > 0 else float("inf"),
        }
    return degradation


def compute_condition_effectiveness(asr_matrix, cond_order):
    effectiveness = {}
    for cond in cond_order:
        asrs = [
            asr_matrix[model].get(cond, 0)
            for model in MODEL_ORDER
            if model in asr_matrix
        ]
        effectiveness[cond] = {
            "mean_asr": sum(asrs) / len(asrs) if asrs else 0,
            "max_asr": max(asrs) if asrs else 0,
            "min_asr": min(asrs) if asrs else 0,
            "per_model": {
                model: asr_matrix[model].get(cond, 0)
                for model in MODEL_ORDER if model in asr_matrix
            },
        }
    return effectiveness


def compute_category_cross_lingual(metrics_list):
    data = defaultdict(lambda: defaultdict(dict))
    for m in metrics_list:
        model = m["model_name"]
        cond = m["condition"]
        for cat, rate in m["category_refusal_rates"].items():
            data[model][cond][cat] = rate
    return data


def compute_response_language_stats(metrics_list):
    stats = []
    for m in metrics_list:
        stats.append({
            "model": m["model_name"],
            "condition": m["condition"],
            "lang_dist": m["language_distribution"],
            "lang_refusal": m.get("language_refusal_rates", {}),
        })
    return stats


def compute_cross_language_comparison(asr_matrix):
    """Compare KO vs ZH safety erosion per model."""
    comparison = []
    ko_conds = ["ko", "ko_gemini", "ko_gpt"]
    zh_conds = ["zh", "zh_gemini", "zh_gpt"]
    engines = ["NLLB", "Gemini", "GPT"]
    for model in MODEL_ORDER:
        if model not in asr_matrix:
            continue
        en_asr = asr_matrix[model].get("en", 0)
        entry = {
            "model": model,
            "short": MODEL_SHORT.get(model, model),
            "en_asr": en_asr,
        }
        for engine, ko_c, zh_c in zip(engines, ko_conds, zh_conds):
            ko_val = asr_matrix[model].get(ko_c, 0)
            zh_val = asr_matrix[model].get(zh_c, 0)
            entry[f"ko_{engine.lower()}"] = ko_val
            entry[f"zh_{engine.lower()}"] = zh_val
            entry[f"ko_delta_{engine.lower()}"] = ko_val - en_asr
            entry[f"zh_delta_{engine.lower()}"] = zh_val - en_asr
        # Mean across engines
        ko_mean = sum(asr_matrix[model].get(c, 0) for c in ko_conds) / 3
        zh_mean = sum(asr_matrix[model].get(c, 0) for c in zh_conds) / 3
        entry["ko_mean_asr"] = ko_mean
        entry["zh_mean_asr"] = zh_mean
        entry["ko_zh_ratio"] = ko_mean / zh_mean if zh_mean > 0.001 else float("inf")
        comparison.append(entry)
    return comparison


def generate_bilingual_latex_table(asr_matrix):
    """Generate main results LaTeX table with KO and ZH conditions."""
    ko_conds = ["en", "ko", "mix", "ko2en", "ko_gemini", "ko2en_gemini", "ko_gpt", "ko2en_gpt", "adaptive"]
    zh_conds = ["zh", "mix_zh", "zh2en", "zh_gemini", "zh2en_gemini", "zh_gpt", "zh2en_gpt", "adaptive_zh"]

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Attack Success Rate (\%) across models and language conditions. "
        r"KO = Korean prompt; ZH = Chinese prompt; X$\to$EN = prompt requesting English response; "
        r"MIX = target-language frame with English content; WS = word-substitution. "
        r"$\Delta_{\max}$ = highest non-EN ASR $-$ EN ASR per language. "
        r"\textbf{Bold} = max non-EN ASR per model per language.}"
    )
    lines.append(r"\label{tab:main}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\begin{tabular}{@{}l c ccc cc cc c c | cccc cc cc c c@{}}")
    lines.append(r"\toprule")
    # Header row 1: language groups
    lines.append(
        r" & & \multicolumn{8}{c}{\textbf{Korean Conditions}} & "
        r"\multicolumn{1}{c}{} & \multicolumn{8}{c}{\textbf{Chinese Conditions}} & \\"
    )
    lines.append(r"\cmidrule(lr){3-10} \cmidrule(lr){12-19}")
    # Header row 2: engine groups
    lines.append(
        r" & & \multicolumn{3}{c}{\textbf{NLLB}} "
        r"& \multicolumn{2}{c}{\textbf{Gemini}} "
        r"& \multicolumn{2}{c}{\textbf{GPT}} & & "
        r"& \multicolumn{3}{c}{\textbf{NLLB}} "
        r"& \multicolumn{2}{c}{\textbf{Gemini}} "
        r"& \multicolumn{2}{c}{\textbf{GPT}} & & \\"
    )
    lines.append(
        r"\cmidrule(lr){3-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9} "
        r"\cmidrule(lr){12-14} \cmidrule(lr){15-16} \cmidrule(lr){17-18}"
    )
    # Header row 3: condition names
    lines.append(
        r"\textbf{Model} & \textbf{EN} "
        r"& \textbf{KO} & \textbf{MIX} & \textbf{KO{\tiny$\to$}EN} "
        r"& \textbf{KO} & \textbf{KO{\tiny$\to$}EN} "
        r"& \textbf{KO} & \textbf{KO{\tiny$\to$}EN} "
        r"& \textbf{WS} & $\Delta$ "
        r"& \textbf{ZH} & \textbf{MIX} & \textbf{ZH{\tiny$\to$}EN} "
        r"& \textbf{ZH} & \textbf{ZH{\tiny$\to$}EN} "
        r"& \textbf{ZH} & \textbf{ZH{\tiny$\to$}EN} "
        r"& \textbf{WS} & $\Delta$ \\"
    )
    lines.append(r"\midrule")

    for model in MODEL_ORDER:
        if model not in asr_matrix:
            continue
        short = MODEL_SHORT.get(model, model)
        ko_vals = {c: asr_matrix[model].get(c, 0) * 100 for c in ko_conds}
        zh_vals = {c: asr_matrix[model].get(c, 0) * 100 for c in zh_conds}
        en_val = ko_vals["en"]

        # KO: find max non-EN
        ko_non_en = {c: v for c, v in ko_vals.items() if c != "en"}
        ko_max = max(ko_non_en.values()) if ko_non_en else 0
        ko_delta = ko_max - en_val

        # ZH: find max
        zh_max = max(zh_vals.values()) if zh_vals else 0
        zh_delta = zh_max - en_val

        def fmt_ko(c, v):
            s = f"{v:.1f}"
            if c != "en" and v == ko_max and v > en_val:
                return rf"\textbf{{{s}}}"
            return s

        def fmt_zh(c, v):
            s = f"{v:.1f}"
            if v == zh_max and v > en_val:
                return rf"\textbf{{{s}}}"
            return s

        ko_parts = [fmt_ko(c, ko_vals[c]) for c in ko_conds]
        zh_parts = [fmt_zh(c, zh_vals[c]) for c in zh_conds]

        row = f"  {short} & " + " & ".join(ko_parts) + f" & {ko_delta:+.1f}"
        row += " & " + " & ".join(zh_parts) + f" & {zh_delta:+.1f} \\\\"
        lines.append(row)

    # Mean row
    lines.append(r"\midrule")
    mean_parts_ko = []
    for c in ko_conds:
        vals = [asr_matrix[m].get(c, 0) * 100 for m in MODEL_ORDER if m in asr_matrix]
        mean_parts_ko.append(f"{sum(vals)/len(vals):.1f}")
    mean_parts_zh = []
    for c in zh_conds:
        vals = [asr_matrix[m].get(c, 0) * 100 for m in MODEL_ORDER if m in asr_matrix]
        mean_parts_zh.append(f"{sum(vals)/len(vals):.1f}")
    lines.append(
        r"  \textit{Mean} & " + " & ".join(mean_parts_ko) + " & "
        + " & " + " & ".join(mean_parts_zh) + r" & \\"
    )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def generate_compact_table(asr_matrix):
    """Generate a more compact bilingual table for ETRI Journal column width."""
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Attack Success Rate (\%) across models for Korean and Chinese conditions. "
        r"\textbf{Bold} = highest ASR per model per language group. "
        r"$\Delta$ = max non-EN ASR $-$ EN ASR.}"
    )
    lines.append(r"\label{tab:main}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{3.5pt}")
    lines.append(r"\begin{tabular}{@{}l c | cccc c | cccc c@{}}")
    lines.append(r"\toprule")
    lines.append(
        r" & & \multicolumn{5}{c|}{\textbf{Korean}} "
        r"& \multicolumn{5}{c}{\textbf{Chinese}} \\"
    )
    lines.append(r"\cmidrule(lr){3-7} \cmidrule(lr){8-12}")
    lines.append(
        r"\textbf{Model} & \textbf{EN} "
        r"& \textbf{NLLB} & \textbf{Gem.} & \textbf{GPT} & \textbf{WS} & $\Delta$ "
        r"& \textbf{NLLB} & \textbf{Gem.} & \textbf{GPT} & \textbf{WS} & $\Delta$ \\"
    )
    lines.append(r"\midrule")

    for model in MODEL_ORDER:
        if model not in asr_matrix:
            continue
        short = MODEL_SHORT.get(model, model)
        en = asr_matrix[model].get("en", 0) * 100

        # Korean: use KO→EN (NLLB) as representative NLLB since it's often the highest
        ko_nllb = asr_matrix[model].get("ko", 0) * 100
        ko_gem = asr_matrix[model].get("ko_gemini", 0) * 100
        ko_gpt = asr_matrix[model].get("ko_gpt", 0) * 100
        ko_ws = asr_matrix[model].get("adaptive", 0) * 100
        ko_vals = [ko_nllb, ko_gem, ko_gpt, ko_ws]
        ko_max = max(ko_vals)
        ko_delta = ko_max - en

        zh_nllb = asr_matrix[model].get("zh", 0) * 100
        zh_gem = asr_matrix[model].get("zh_gemini", 0) * 100
        zh_gpt = asr_matrix[model].get("zh_gpt", 0) * 100
        zh_ws = asr_matrix[model].get("adaptive_zh", 0) * 100
        zh_vals = [zh_nllb, zh_gem, zh_gpt, zh_ws]
        zh_max = max(zh_vals)
        zh_delta = zh_max - en

        def bold_if_max(v, mx, baseline):
            s = f"{v:.1f}"
            if v == mx and v > baseline:
                return rf"\textbf{{{s}}}"
            return s

        parts = [
            f"  {short}",
            f"{en:.1f}",
            bold_if_max(ko_nllb, ko_max, en),
            bold_if_max(ko_gem, ko_max, en),
            bold_if_max(ko_gpt, ko_max, en),
            bold_if_max(ko_ws, ko_max, en),
            f"{ko_delta:+.1f}",
            bold_if_max(zh_nllb, zh_max, en),
            bold_if_max(zh_gem, zh_max, en),
            bold_if_max(zh_gpt, zh_max, en),
            bold_if_max(zh_ws, zh_max, en),
            f"{zh_delta:+.1f}",
        ]
        lines.append(" & ".join(parts) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--output-dir", default=FIGURES_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    metrics_list = load_evaluation_summary(args.results_dir)
    # Filter to main 6 models
    metrics_list = [m for m in metrics_list if m["model_name"] in MODEL_ORDER]
    print(f"Loaded {len(metrics_list)} evaluation results (6 models)")

    # 1. ASR Matrix
    asr_matrix = build_asr_matrix(metrics_list)
    print("\n── Korean ASR Matrix ──")
    header = f"{'Model':<20}" + "".join(f"{c:<13}" for c in KO_CONDITION_ORDER)
    print(header)
    for model in MODEL_ORDER:
        if model not in asr_matrix:
            continue
        short = MODEL_SHORT.get(model, model)
        row = f"{short:<20}" + "".join(
            f"{asr_matrix[model].get(c, 0)*100:<13.1f}" for c in KO_CONDITION_ORDER
        )
        print(row)

    print("\n── Chinese ASR Matrix ──")
    header = f"{'Model':<20}" + "".join(f"{c:<13}" for c in ZH_CONDITION_ORDER)
    print(header)
    for model in MODEL_ORDER:
        if model not in asr_matrix:
            continue
        short = MODEL_SHORT.get(model, model)
        row = f"{short:<20}" + "".join(
            f"{asr_matrix[model].get(c, 0)*100:<13.1f}" for c in ZH_CONDITION_ORDER
        )
        print(row)

    # 2. Safety degradation (KO and ZH)
    ko_degradation = compute_safety_degradation(asr_matrix, "ko")
    zh_degradation = compute_safety_degradation(asr_matrix, "zh")

    print("\n── EN → KO / ZH Safety Degradation ──")
    for model in MODEL_ORDER:
        if model not in ko_degradation:
            continue
        dk = ko_degradation[model]
        dz = zh_degradation[model]
        short = MODEL_SHORT.get(model, model)
        print(
            f"  {short:<16} EN:{dk['en_asr']*100:5.1f}% → KO:{dk['target_asr']*100:5.1f}% "
            f"(Δ={dk['delta_asr']*100:+.1f} pp) | "
            f"ZH:{dz['target_asr']*100:5.1f}% (Δ={dz['delta_asr']*100:+.1f} pp)"
        )

    # 3. Condition effectiveness (all)
    all_effectiveness = compute_condition_effectiveness(asr_matrix, ALL_CONDITION_ORDER)
    print("\n── Condition Effectiveness (Mean ASR) ──")
    sorted_conds = sorted(ALL_CONDITION_ORDER, key=lambda c: all_effectiveness[c]["mean_asr"], reverse=True)
    for cond in sorted_conds:
        e = all_effectiveness[cond]
        print(f"  {CONDITION_LABELS.get(cond, cond):<25} Mean ASR: {e['mean_asr']*100:.1f}%")

    # 4. Cross-language comparison
    cross_lang = compute_cross_language_comparison(asr_matrix)
    print("\n── Cross-Language Comparison (KO vs ZH) ──")
    for p in cross_lang:
        print(
            f"  {p['short']:<16} KO mean: {p['ko_mean_asr']*100:.1f}% | "
            f"ZH mean: {p['zh_mean_asr']*100:.1f}% | "
            f"KO/ZH ratio: {p['ko_zh_ratio']:.1f}×"
        )

    # Save all analysis
    analysis = {
        "asr_matrix": asr_matrix,
        "ko_degradation": ko_degradation,
        "zh_degradation": zh_degradation,
        "ko_effectiveness": compute_condition_effectiveness(asr_matrix, KO_CONDITION_ORDER),
        "zh_effectiveness": compute_condition_effectiveness(asr_matrix, ZH_CONDITION_ORDER),
        "all_effectiveness": all_effectiveness,
        "cross_language_comparison": cross_lang,
        "category_cross_lingual": compute_category_cross_lingual(metrics_list),
        "response_language_stats": compute_response_language_stats(metrics_list),
        # Legacy compat
        "degradation": ko_degradation,
        "condition_effectiveness": all_effectiveness,
        "capability_vs_safety": [
            {
                "model": m, "short": MODEL_SHORT.get(m, m),
                "ko_capability": MODEL_KO_CAPABILITY.get(m, 0),
                "zh_capability": MODEL_ZH_CAPABILITY.get(m, 0),
                "en_asr": asr_matrix[m].get("en", 0),
                "ko_asr": asr_matrix[m].get("ko", 0),
                "zh_asr": asr_matrix[m].get("zh", 0),
                "ko_degradation": asr_matrix[m].get("ko", 0) - asr_matrix[m].get("en", 0),
                "zh_degradation": asr_matrix[m].get("zh", 0) - asr_matrix[m].get("en", 0),
            }
            for m in MODEL_ORDER if m in asr_matrix
        ],
    }
    analysis_path = os.path.join(args.output_dir, "analysis.json")
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nAnalysis saved → {analysis_path}")

    # Generate LaTeX tables
    table1 = generate_compact_table(asr_matrix)
    table1_path = os.path.join(args.output_dir, "main_table.tex")
    with open(table1_path, "w") as f:
        f.write(table1)
    print(f"Compact table → {table1_path}")

    table2 = generate_bilingual_latex_table(asr_matrix)
    table2_path = os.path.join(args.output_dir, "full_table.tex")
    with open(table2_path, "w") as f:
        f.write(table2)
    print(f"Full table → {table2_path}")


if __name__ == "__main__":
    main()
