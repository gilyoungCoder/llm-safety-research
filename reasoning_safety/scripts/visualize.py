"""
Generate tables and figures for the paper.
Reads evaluation_summary.json and produces:
1. Main comparison table (LaTeX)
2. ASR bar chart (standard vs reasoning)
3. Category-wise heatmap
4. Thinking trace distribution pie chart
5. Response length comparison
"""

import json
import os
import argparse

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

# Color scheme
COLORS = {
    "standard": "#4C72B0",
    "reasoning": "#DD4444",
}

TRACE_COLORS = {
    "Type_A_safety_refuse": "#2ecc71",
    "Type_B_safety_rationalize": "#e67e22",
    "Type_B_safety_comply": "#e74c3c",
    "Type_C_no_safety": "#c0392b",
    "Type_D_other_refuse": "#95a5a6",
    "no_thinking": "#bdc3c7",
}

TRACE_LABELS = {
    "Type_A_safety_refuse": "A: Safety-aware refuse",
    "Type_B_safety_rationalize": "B: Self-rationalized comply",
    "Type_B_safety_comply": "B: Safety-aware comply",
    "Type_C_no_safety": "C: No safety awareness",
    "Type_D_other_refuse": "D: Other refuse",
    "no_thinking": "No thinking trace",
}

# Model pair definitions for comparison
MODEL_PAIRS = [
    ("Llama-3.1-8B-Instruct", "DeepSeek-R1-Distill-Llama-8B", "Llama 8B"),
    ("Qwen2.5-7B-Instruct", "DeepSeek-R1-Distill-Qwen-7B", "Qwen 7B"),
    ("Qwen2.5-14B-Instruct", "DeepSeek-R1-Distill-Qwen-14B", "Qwen 14B"),
]


def load_metrics(results_dir):
    summary_path = os.path.join(results_dir, "evaluation_summary.json")
    with open(summary_path) as f:
        return json.load(f)


def get_metric_by_name(metrics, model_name):
    for m in metrics:
        if m["model_name"] == model_name:
            return m
    return None


def generate_latex_table(metrics, output_dir):
    """Generate main comparison LaTeX table."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Safety comparison: Standard Instruct vs. Reasoning (DeepSeek-R1 Distill) models. "
                 r"Refusal Rate (\%) indicates the percentage of harmful prompts correctly refused. "
                 r"ASR (\%) = Attack Success Rate.}")
    lines.append(r"\label{tab:main_results}")
    lines.append(r"\begin{tabular}{llccc}")
    lines.append(r"\toprule")
    lines.append(r"Base & Model Type & Refusal (\%) & ASR (\%) & Avg. Len \\")
    lines.append(r"\midrule")

    for std_name, reas_name, pair_label in MODEL_PAIRS:
        std = get_metric_by_name(metrics, std_name)
        reas = get_metric_by_name(metrics, reas_name)

        if std:
            lines.append(f"{pair_label} & Standard & {std['refusal_rate']*100:.1f} & "
                         f"{std['asr']*100:.1f} & {std['avg_response_length']:.0f} \\\\")
        if reas:
            lines.append(f" & Reasoning & {reas['refusal_rate']*100:.1f} & "
                         f"{reas['asr']*100:.1f} & {reas['avg_response_length']:.0f} \\\\")
        lines.append(r"\addlinespace")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    table_text = "\n".join(lines)
    out_path = os.path.join(output_dir, "main_table.tex")
    with open(out_path, "w") as f:
        f.write(table_text)
    print(f"LaTeX table saved to {out_path}")
    return table_text


def plot_asr_comparison(metrics, output_dir):
    """Bar chart: ASR comparison for each model pair."""
    fig, ax = plt.subplots(figsize=(8, 5))

    pair_labels = []
    std_asrs = []
    reas_asrs = []

    for std_name, reas_name, pair_label in MODEL_PAIRS:
        std = get_metric_by_name(metrics, std_name)
        reas = get_metric_by_name(metrics, reas_name)
        if std and reas:
            pair_labels.append(pair_label)
            std_asrs.append(std["asr"] * 100)
            reas_asrs.append(reas["asr"] * 100)

    x = np.arange(len(pair_labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, std_asrs, width, label="Standard Instruct",
                   color=COLORS["standard"], edgecolor="white")
    bars2 = ax.bar(x + width/2, reas_asrs, width, label="Reasoning (R1-Distill)",
                   color=COLORS["reasoning"], edgecolor="white")

    ax.set_ylabel("Attack Success Rate (%)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "asr_comparison.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"ASR comparison chart saved to {out_path}")


def plot_category_heatmap(metrics, output_dir):
    """Heatmap: refusal rate by category × model."""
    # Collect all categories
    all_cats = set()
    for m in metrics:
        all_cats.update(m.get("category_refusal_rates", {}).keys())
    categories = sorted(all_cats)

    model_names = [m["model_name"] for m in metrics]
    data = np.zeros((len(model_names), len(categories)))

    for i, m in enumerate(metrics):
        for j, cat in enumerate(categories):
            data[i, j] = m.get("category_refusal_rates", {}).get(cat, 0) * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(np.arange(len(categories)))
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_xticklabels([c.replace("_", "\n") for c in categories], fontsize=9)

    # Shorten model names for display
    short_names = []
    for name in model_names:
        name = name.replace("DeepSeek-R1-Distill-", "R1-")
        name = name.replace("-Instruct", "")
        short_names.append(name)
    ax.set_yticklabels(short_names, fontsize=9)

    # Add text annotations
    for i in range(len(model_names)):
        for j in range(len(categories)):
            val = data[i, j]
            color = "white" if val < 30 or val > 70 else "black"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    color=color, fontsize=8)

    ax.set_xlabel("Harm Category", fontsize=11)
    cbar = fig.colorbar(im, ax=ax, label="Refusal Rate (%)")
    ax.set_title("Refusal Rate by Model and Harm Category", fontsize=12)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "category_heatmap.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Category heatmap saved to {out_path}")


def plot_thinking_trace_distribution(metrics, output_dir):
    """Pie charts: thinking trace type distribution for reasoning models."""
    reasoning_models = [m for m in metrics if m["model_type"] == "reasoning"]

    if not reasoning_models:
        print("No reasoning models found, skipping trace distribution plot.")
        return

    n = len(reasoning_models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, m in zip(axes, reasoning_models):
        dist = m.get("trace_distribution", {})
        if not dist:
            continue

        labels = []
        sizes = []
        colors = []
        for trace_type, count in sorted(dist.items()):
            if count > 0:
                labels.append(TRACE_LABELS.get(trace_type, trace_type))
                sizes.append(count)
                colors.append(TRACE_COLORS.get(trace_type, "#999999"))

        wedges, texts, autotexts = ax.pie(
            sizes, labels=None, colors=colors, autopct='%1.1f%%',
            startangle=90, pctdistance=0.8
        )
        for text in autotexts:
            text.set_fontsize(8)

        short_name = m["model_name"].replace("DeepSeek-R1-Distill-", "R1-")
        ax.set_title(short_name, fontsize=11)

    # Add legend
    legend_labels = [TRACE_LABELS.get(t, t) for t in TRACE_COLORS if t in
                     {k for m in reasoning_models for k in m.get("trace_distribution", {})}]
    legend_colors = [TRACE_COLORS[t] for t in TRACE_COLORS if t in
                     {k for m in reasoning_models for k in m.get("trace_distribution", {})}]

    if legend_labels:
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=c, label=l)
                           for c, l in zip(legend_colors, legend_labels)]
        fig.legend(handles=legend_elements, loc="lower center",
                   ncol=min(3, len(legend_elements)), fontsize=9,
                   bbox_to_anchor=(0.5, -0.05))

    plt.suptitle("Thinking Trace Safety Pattern Distribution", fontsize=13, y=1.02)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "trace_distribution.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Trace distribution chart saved to {out_path}")


def plot_response_length(metrics, output_dir):
    """Bar chart: average response length comparison."""
    fig, ax = plt.subplots(figsize=(8, 5))

    pair_labels = []
    std_lens = []
    reas_lens = []

    for std_name, reas_name, pair_label in MODEL_PAIRS:
        std = get_metric_by_name(metrics, std_name)
        reas = get_metric_by_name(metrics, reas_name)
        if std and reas:
            pair_labels.append(pair_label)
            std_lens.append(std["avg_response_length"])
            reas_lens.append(reas["avg_response_length"])

    x = np.arange(len(pair_labels))
    width = 0.35

    ax.bar(x - width/2, std_lens, width, label="Standard Instruct",
           color=COLORS["standard"], edgecolor="white")
    ax.bar(x + width/2, reas_lens, width, label="Reasoning (R1-Distill)",
           color=COLORS["reasoning"], edgecolor="white")

    ax.set_ylabel("Average Response Length (chars)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "response_length.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Response length chart saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--output-dir", default=FIGURES_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    metrics = load_metrics(args.results_dir)

    print(f"Loaded metrics for {len(metrics)} models\n")

    # Generate all outputs
    latex = generate_latex_table(metrics, args.output_dir)
    print(f"\n--- LaTeX Table ---\n{latex}\n")

    plot_asr_comparison(metrics, args.output_dir)
    plot_category_heatmap(metrics, args.output_dir)
    plot_thinking_trace_distribution(metrics, args.output_dir)
    plot_response_length(metrics, args.output_dir)

    print(f"\nAll figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
