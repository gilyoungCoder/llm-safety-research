"""
Generate publication-quality figures for bilingual cross-lingual safety paper.
Korean + Chinese evaluation with consistent modern styling.

Usage:
  python3 visualize.py
"""

import json
import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.gridspec import GridSpec

matplotlib.use("Agg")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, "..", "figures")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")

# ── Global Style ──
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset": "dejavuserif",
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "axes.titleweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#cccccc",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
})

# ── Model Configuration ──
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
MODEL_COLORS = {
    "Llama-3.1-8B-Instruct": "#4363d8",
    "Qwen2.5-7B-Instruct": "#3cb44b",
    "Qwen2.5-14B-Instruct": "#911eb4",
    "Phi-3-medium-4k-instruct": "#e6194B",
    "Yi-1.5-9B-Chat": "#42d4f4",
    "internlm2_5-7b-chat": "#f58231",
}

# ── Condition Configuration ──
KO_COND_ORDER = ["en", "ko", "mix", "ko2en", "adaptive", "ko_gemini", "ko2en_gemini", "ko_gpt", "ko2en_gpt"]
ZH_COND_ORDER = ["en", "zh", "mix_zh", "zh2en", "adaptive_zh", "zh_gemini", "zh2en_gemini", "zh_gpt", "zh2en_gpt"]

KO_COND_LABELS = {
    "en": "EN", "ko": "KO\n(NLLB)", "mix": "MIX", "ko2en": "KO→EN\n(NLLB)",
    "adaptive": "Word-\nsub", "ko_gemini": "KO\n(Gem.)", "ko2en_gemini": "KO→EN\n(Gem.)",
    "ko_gpt": "KO\n(GPT)", "ko2en_gpt": "KO→EN\n(GPT)",
}
ZH_COND_LABELS = {
    "en": "EN", "zh": "ZH\n(NLLB)", "mix_zh": "MIX", "zh2en": "ZH→EN\n(NLLB)",
    "adaptive_zh": "Word-\nsub", "zh_gemini": "ZH\n(Gem.)", "zh2en_gemini": "ZH→EN\n(Gem.)",
    "zh_gpt": "ZH\n(GPT)", "zh2en_gpt": "ZH→EN\n(GPT)",
}
KO_COND_LABELS_SHORT = {
    "en": "EN", "ko": "KO (NLLB)", "mix": "MIX-KO", "ko2en": "KO→EN (NLLB)",
    "adaptive": "WS-KO", "ko_gemini": "KO (Gem.)", "ko2en_gemini": "KO→EN (Gem.)",
    "ko_gpt": "KO (GPT)", "ko2en_gpt": "KO→EN (GPT)",
}
ZH_COND_LABELS_SHORT = {
    "en": "EN", "zh": "ZH (NLLB)", "mix_zh": "MIX-ZH", "zh2en": "ZH→EN (NLLB)",
    "adaptive_zh": "WS-ZH", "zh_gemini": "ZH (Gem.)", "zh2en_gemini": "ZH→EN (Gem.)",
    "zh_gpt": "ZH (GPT)", "zh2en_gpt": "ZH→EN (GPT)",
}

CATEGORY_ORDER = [
    "deception", "violence", "illegal_activity", "other",
    "self_harm", "discrimination", "sexual_content",
]
CATEGORY_SHORT = {
    "deception": "Deception", "violence": "Violence", "illegal_activity": "Illegal Act.",
    "other": "Other", "self_harm": "Self-Harm", "discrimination": "Discrim.",
    "sexual_content": "Sexual",
}

# Translation engines
TRANS_ENGINES = ["NLLB", "Gemini", "GPT"]
KO_TRANS = {"NLLB": "ko", "Gemini": "ko_gemini", "GPT": "ko_gpt"}
KO2EN_TRANS = {"NLLB": "ko2en", "Gemini": "ko2en_gemini", "GPT": "ko2en_gpt"}
ZH_TRANS = {"NLLB": "zh", "Gemini": "zh_gemini", "GPT": "zh_gpt"}
ZH2EN_TRANS = {"NLLB": "zh2en", "Gemini": "zh2en_gemini", "GPT": "zh2en_gpt"}
ENGINE_COLORS = {"NLLB": "#e6194B", "Gemini": "#3cb44b", "GPT": "#4363d8"}

# Language colors
LANG_COLORS = {"Korean": "#e6194B", "Chinese": "#f58231"}


def load_data():
    with open(os.path.join(FIGURES_DIR, "analysis.json")) as f:
        analysis = json.load(f)
    with open(os.path.join(RESULTS_DIR, "evaluation_summary.json")) as f:
        summary = json.load(f)
    summary = [m for m in summary if m["model_name"] in MODEL_ORDER]
    return analysis, summary


def savefig(fig, name):
    fig.savefig(os.path.join(FIGURES_DIR, f"{name}.png"))
    fig.savefig(os.path.join(FIGURES_DIR, f"{name}.pdf"))
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# MAIN FIGURES
# ═══════════════════════════════════════════════════════════════

def fig1_bilingual_heatmap(analysis):
    """Fig 1: Two-panel heatmap — Korean (top) + Chinese (bottom), 6 models × 9 conditions each."""
    asr = analysis["asr_matrix"]
    models = [m for m in MODEL_ORDER if m in asr]
    shorts = [MODEL_SHORT[m] for m in models]

    ko_data = np.array([[asr[m].get(c, 0) * 100 for c in KO_COND_ORDER] for m in models])
    zh_data = np.array([[asr[m].get(c, 0) * 100 for c in ZH_COND_ORDER] for m in models])

    fig, (ax_ko, ax_zh) = plt.subplots(2, 1, figsize=(14, 8.5), gridspec_kw={"hspace": 0.35})

    for ax, data, cond_order, cond_labels, title, panel in [
        (ax_ko, ko_data, KO_COND_ORDER, KO_COND_LABELS_SHORT, "Korean Conditions", "(a)"),
        (ax_zh, zh_data, ZH_COND_ORDER, ZH_COND_LABELS_SHORT, "Chinese Conditions", "(b)"),
    ]:
        im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=70)
        ax.set_xticks(range(len(cond_order)))
        ax.set_xticklabels([cond_labels[c] for c in cond_order], rotation=35, ha="right", fontsize=9)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(shorts, fontsize=11)

        for i in range(len(models)):
            for j in range(len(cond_order)):
                val = data[i, j]
                color = "white" if val > 35 else "black"
                weight = "bold" if val > 15 else "normal"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        color=color, fontsize=8, fontweight=weight)

        # Group separators
        for sep in [0.5, 4.5, 6.5]:
            ax.axvline(x=sep, color="white", linewidth=2.5)

        ax.set_title(f"{panel} {title}", fontsize=12, fontweight="bold", loc="left", pad=10)

    # Shared colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.935, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(ax_ko.images[0], cax=cbar_ax, label="ASR (%)")
    cbar.ax.tick_params(labelsize=9)

    fig.suptitle("Attack Success Rate Across Models: Korean vs. Chinese",
                 fontsize=14, fontweight="bold", y=0.98)
    savefig(fig, "asr_heatmap")
    print("  [Fig 1] asr_heatmap (bilingual)")


def fig2_cross_language_scatter(analysis):
    """Fig 2: KO ASR vs ZH ASR scatter — the money shot showing language-specific safety."""
    asr = analysis["asr_matrix"]
    models = [m for m in MODEL_ORDER if m in asr]

    fig, ax = plt.subplots(figsize=(8, 7))

    markers = {"NLLB": "o", "Gemini": "s", "GPT": "D"}
    trans_maps = {"NLLB": ("ko", "zh"), "Gemini": ("ko_gemini", "zh_gemini"), "GPT": ("ko_gpt", "zh_gpt")}

    for model in models:
        for engine in TRANS_ENGINES:
            ko_c, zh_c = trans_maps[engine]
            ko_val = asr[model].get(ko_c, 0) * 100
            zh_val = asr[model].get(zh_c, 0) * 100
            ax.scatter(zh_val, ko_val, s=140,
                       color=MODEL_COLORS[model], marker=markers[engine],
                       edgecolors="white", linewidth=1.2, zorder=5)

    # Diagonal line
    ax.plot([0, 70], [0, 70], "k--", alpha=0.2, linewidth=1, zorder=1)
    ax.fill_between([0, 70], [0, 70], [70, 70], alpha=0.04, color="#e6194B", zorder=0)
    ax.fill_between([0, 70], [0, 0], [0, 70], alpha=0.04, color="#3cb44b", zorder=0)
    ax.text(5, 62, "KO more\nvulnerable", fontsize=9, color="#e6194B",
            fontstyle="italic", fontweight="bold", alpha=0.6)
    ax.text(55, 5, "ZH more\nvulnerable", fontsize=9, color="#3cb44b",
            fontstyle="italic", fontweight="bold", alpha=0.6)

    # Legend
    model_patches = [mpatches.Patch(color=MODEL_COLORS[m], label=MODEL_SHORT[m]) for m in models]
    engine_handles = [plt.Line2D([0], [0], marker=markers[e], color="gray", markersize=8,
                                  linestyle="None", label=e) for e in TRANS_ENGINES]
    legend1 = ax.legend(handles=model_patches, loc="upper left", fontsize=9, title="Model", title_fontsize=9)
    ax.add_artist(legend1)
    ax.legend(handles=engine_handles, loc="lower right", fontsize=9, title="Engine", title_fontsize=9)

    ax.set_xlabel("Chinese ASR (%)", fontsize=12)
    ax.set_ylabel("Korean ASR (%)", fontsize=12)
    ax.set_title("Cross-Language Safety: Korean vs. Chinese", fontsize=14)
    ax.set_xlim(-2, 72)
    ax.set_ylim(-2, 72)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)

    savefig(fig, "cross_language_scatter")
    print("  [Fig 2] cross_language_scatter")


def fig3_degradation_paired(analysis):
    """Fig 3: Paired degradation — EN→KO and EN→ZH slope chart."""
    asr = analysis["asr_matrix"]
    models = [m for m in MODEL_ORDER if m in asr]
    shorts = [MODEL_SHORT[m] for m in models]

    fig, (ax_ko, ax_zh) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, target, title, panel in [
        (ax_ko, "ko", "English → Korean (NLLB)", "(a)"),
        (ax_zh, "zh", "English → Chinese (NLLB)", "(b)"),
    ]:
        for i, (model, short) in enumerate(zip(models, shorts)):
            color = MODEL_COLORS[model]
            en_asr = asr[model].get("en", 0) * 100
            tgt_asr = asr[model].get(target, 0) * 100
            ax.plot([0, 1], [en_asr, tgt_asr], "o-", color=color,
                    linewidth=2.5, markersize=10, label=short, zorder=5,
                    markeredgecolor="white", markeredgewidth=1.2)
            delta = tgt_asr - en_asr
            mid_y = (en_asr + tgt_asr) / 2
            offsets = [8, -12, 8, -12, 8, -12]
            sign = "+" if delta >= 0 else ""
            ax.annotate(
                f"{sign}{delta:.1f}pp", xy=(0.52, mid_y),
                fontsize=8, color=color, ha="left", fontweight="bold",
                xytext=(0, offsets[i % len(offsets)]), textcoords="offset points",
            )

        lang_label = "Korean" if target == "ko" else "Chinese"
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["English", f"{lang_label} (NLLB)"], fontsize=11)
        ax.set_title(f"{panel} {title}", fontsize=12, fontweight="bold")
        ax.grid(axis="y")
        ax.set_xlim(-0.15, 1.3)
        ax.axhspan(0, 10, alpha=0.06, color="green", zorder=0)
        if target == "ko":
            ax.set_ylabel("ASR (%)", fontsize=12)
            ax.legend(loc="upper left", fontsize=9)

    y_max = max(asr[m].get("ko", 0) * 100 for m in models) * 1.12
    ax_ko.set_ylim(0, y_max)

    fig.suptitle("Safety Degradation: Korean Shows Severe Erosion, Chinese Remains Stable",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    savefig(fig, "degradation_slope")
    print("  [Fig 3] degradation_slope (paired)")


def fig4_translation_engine_bilingual(analysis):
    """Fig 4: Translation engine comparison — 2×2 panels (KO, KO→EN, ZH, ZH→EN)."""
    asr = analysis["asr_matrix"]
    models = [m for m in MODEL_ORDER if m in asr]
    shorts = [MODEL_SHORT[m] for m in models]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharey="row")
    x = np.arange(len(models))
    width = 0.25

    configs = [
        (axes[0, 0], KO_TRANS, "(a) Korean (KO)", 80),
        (axes[0, 1], KO2EN_TRANS, "(b) Korean→EN (KO→EN)", 80),
        (axes[1, 0], ZH_TRANS, "(c) Chinese (ZH)", 25),
        (axes[1, 1], ZH2EN_TRANS, "(d) Chinese→EN (ZH→EN)", 40),
    ]

    for ax, trans_map, title, ylim in configs:
        for i, engine in enumerate(TRANS_ENGINES):
            cond = trans_map[engine]
            vals = [asr[m].get(cond, 0) * 100 for m in models]
            bars = ax.bar(x + (i - 1) * width, vals, width * 0.85,
                          label=engine, color=ENGINE_COLORS[engine],
                          edgecolor="white", linewidth=0.8, zorder=3)
            for bar, val in zip(bars, vals):
                if val > 1.5:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                            f"{val:.1f}", ha="center", va="bottom", fontsize=7,
                            fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(shorts, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="y")
        ax.set_ylim(0, ylim)
        ax.legend(title="Engine", fontsize=8, title_fontsize=8)

    axes[0, 0].set_ylabel("ASR (%)", fontsize=11)
    axes[1, 0].set_ylabel("ASR (%)", fontsize=11)

    fig.suptitle("Effect of Translation Quality on Safety Erosion: Korean vs. Chinese",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    savefig(fig, "translation_engine_comparison")
    print("  [Fig 4] translation_engine_comparison (bilingual)")


def fig5_language_specific_vulnerability(analysis):
    """Fig 5: Language-specific vulnerability — KO vs ZH ASR per model grouped by engine."""
    asr = analysis["asr_matrix"]
    models = [m for m in MODEL_ORDER if m in asr]
    shorts = [MODEL_SHORT[m] for m in models]

    fig, ax = plt.subplots(figsize=(14, 6.5))
    x = np.arange(len(models))

    configs = [
        ("EN", "en", "#4363d8", ""),
        ("KO (NLLB)", "ko", "#e6194B", ""),
        ("KO (GPT)", "ko_gpt", "#c4383e", "//"),
        ("ZH (NLLB)", "zh", "#f58231", ""),
        ("ZH (GPT)", "zh_gpt", "#d4722e", "//"),
    ]
    n = len(configs)
    width = 0.15

    for i, (label, cond, color, hatch) in enumerate(configs):
        vals = [asr[m].get(cond, 0) * 100 for m in models]
        bars = ax.bar(x + (i - n / 2 + 0.5) * width, vals, width * 0.88,
                      label=label, color=color, hatch=hatch,
                      edgecolor="white", linewidth=0.8, zorder=3)
        for bar, val in zip(bars, vals):
            if val > 3:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                        f"{val:.0f}", ha="center", va="bottom", fontsize=6.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(shorts, fontsize=11)
    ax.set_ylabel("ASR (%)", fontsize=12)
    ax.set_title("Language-Specific Safety: Korean vs. Chinese Vulnerability", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.06))
    ax.grid(axis="y")
    ax.set_ylim(0, 75)

    # Annotations
    ax.annotate("Yi, InternLM:\nSafe in ZH, vulnerable in KO",
                xy=(4.2, 55), fontsize=8.5, color="#e6194B", ha="center",
                fontstyle="italic", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#e6194B", alpha=0.85))

    fig.tight_layout()
    savefig(fig, "language_specific_vulnerability")
    print("  [Fig 5] language_specific_vulnerability")


def fig6_condition_ranking(analysis):
    """Fig 6: All conditions ranked by mean ASR — color-coded by language and engine."""
    eff = analysis["all_effectiveness"]
    all_conds = list(eff.keys())
    sorted_conds = sorted(all_conds, key=lambda c: eff[c]["mean_asr"], reverse=True)

    mean_asrs = [eff[c]["mean_asr"] * 100 for c in sorted_conds]

    label_map = {
        "en": "EN (baseline)",
        "ko": "KO (NLLB)", "mix": "MIX-KO", "ko2en": "KO→EN (NLLB)",
        "adaptive": "WS-KO", "ko_gemini": "KO (Gemini)", "ko2en_gemini": "KO→EN (Gemini)",
        "ko_gpt": "KO (GPT)", "ko2en_gpt": "KO→EN (GPT)",
        "zh": "ZH (NLLB)", "mix_zh": "MIX-ZH", "zh2en": "ZH→EN (NLLB)",
        "adaptive_zh": "WS-ZH", "zh_gemini": "ZH (Gemini)", "zh2en_gemini": "ZH→EN (Gemini)",
        "zh_gpt": "ZH (GPT)", "zh2en_gpt": "ZH→EN (GPT)",
    }
    labels = [label_map.get(c, c) for c in sorted_conds]

    bar_colors = []
    for c in sorted_conds:
        if c == "en":
            bar_colors.append("#4363d8")
        elif "zh" in c or c == "adaptive_zh" or c == "mix_zh":
            bar_colors.append("#f58231")
        else:
            bar_colors.append("#e6194B")

    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(len(sorted_conds))
    bars = ax.barh(y, mean_asrs, color=bar_colors, edgecolor="white",
                   linewidth=1, height=0.6, zorder=3)

    for bar, val in zip(bars, mean_asrs):
        ax.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", ha="left", va="center", fontsize=9, fontweight="bold")

    en_asr = eff["en"]["mean_asr"] * 100
    ax.axvline(x=en_asr, color="#4363d8", linestyle="--", alpha=0.7, linewidth=1.5, zorder=2)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Mean ASR Across All Models (%)", fontsize=11)
    ax.set_title("Condition Effectiveness Ranking: Korean vs. Chinese", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.set_xlim(0, max(mean_asrs) * 1.18)

    legend_patches = [
        mpatches.Patch(color="#e6194B", label="Korean conditions"),
        mpatches.Patch(color="#f58231", label="Chinese conditions"),
        mpatches.Patch(color="#4363d8", label="English baseline"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9, framealpha=0.95)

    savefig(fig, "condition_ranking")
    print("  [Fig 6] condition_ranking (bilingual)")


def fig7_two_tier_bilingual(analysis):
    """Fig 7: Two-tier pattern comparison — Korean (left) vs Chinese (right)."""
    asr = analysis["asr_matrix"]
    models = [m for m in MODEL_ORDER if m in asr]
    shorts = [MODEL_SHORT[m] for m in models]

    fig, (ax_ko, ax_zh) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    x = np.arange(len(models))

    configs_per = [
        ("EN", "en", "#4363d8"),
        ("NLLB", None, "#e6194B"),
        ("Gemini", None, "#3cb44b"),
        ("GPT", None, "#911eb4"),
    ]
    width = 0.18

    for ax, lang, ko_map, zh_map, title, panel in [
        (ax_ko, "KO", KO_TRANS, None, "Korean Prompts", "(a)"),
        (ax_zh, "ZH", None, ZH_TRANS, "Chinese Prompts", "(b)"),
    ]:
        trans_map = ko_map if ko_map else zh_map
        configs = [
            ("EN Baseline", "en", "#4363d8"),
            (f"{lang} (NLLB)", trans_map["NLLB"], "#e6194B"),
            (f"{lang} (Gemini)", trans_map["Gemini"], "#3cb44b"),
            (f"{lang} (GPT)", trans_map["GPT"], "#911eb4"),
        ]
        for i, (label, cond, color) in enumerate(configs):
            vals = [asr[m].get(cond, 0) * 100 for m in models]
            bars = ax.bar(x + (i - 1.5) * width, vals, width * 0.88,
                          label=label, color=color, edgecolor="white", linewidth=0.8, zorder=3)
            for bar, val in zip(bars, vals):
                if val > 2.5:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                            f"{val:.0f}", ha="center", va="bottom", fontsize=6.5, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(shorts, fontsize=10)
        ax.set_title(f"{panel} {title}", fontsize=12, fontweight="bold")
        ax.grid(axis="y")
        ax.legend(fontsize=8, ncol=2, loc="upper right")

    ax_ko.set_ylabel("ASR (%)", fontsize=12)
    ax_ko.set_ylim(0, 75)

    fig.suptitle("Two-Tier Vulnerability: Korean Shows Intrinsic Gaps, Chinese Remains Safe",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    savefig(fig, "two_tier_vulnerability")
    print("  [Fig 7] two_tier_vulnerability (bilingual)")


# ═══════════════════════════════════════════════════════════════
# APPENDIX FIGURES
# ═══════════════════════════════════════════════════════════════

def figA1_category_gap_bilingual(summary):
    """A1: Category-wise degradation heatmap — KO−EN (left) + ZH−EN (right)."""
    data = {}
    for m in summary:
        model = m["model_name"]
        cond = m["condition"]
        if model not in data:
            data[model] = {}
        data[model][cond] = m["category_refusal_rates"]

    models = [m for m in MODEL_ORDER if m in data]
    cats = [c for c in CATEGORY_ORDER if any(c in data[m].get("en", {}) for m in models)]
    cat_shorts = [CATEGORY_SHORT.get(c, c) for c in cats]

    fig, (ax_ko, ax_zh) = plt.subplots(1, 2, figsize=(16, 5.5), sharey=True)

    for ax, target, title, panel in [
        (ax_ko, "ko", "Korean − English", "(a)"),
        (ax_zh, "zh", "Chinese − English", "(b)"),
    ]:
        gap = np.zeros((len(cats), len(models)))
        for j, model in enumerate(models):
            for i, cat in enumerate(cats):
                en_ref = data[model].get("en", {}).get(cat, 1.0)
                tgt_ref = data[model].get(target, {}).get(cat, 1.0)
                gap[i, j] = (tgt_ref - en_ref) * 100

        vmax = 20 if target == "zh" else 20
        vmin = -70 if target == "ko" else -20
        im = ax.imshow(gap, cmap="RdBu", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([MODEL_SHORT[m] for m in models], fontsize=9)
        if target == "ko":
            ax.set_yticks(range(len(cats)))
            ax.set_yticklabels(cat_shorts, fontsize=10)
        ax.set_title(f"{panel} {title}", fontsize=11, fontweight="bold")

        for i in range(len(cats)):
            for j in range(len(models)):
                val = gap[i, j]
                color = "white" if abs(val) > (35 if target == "ko" else 10) else "black"
                ax.text(j, i, f"{val:+.0f}", ha="center", va="center", color=color, fontsize=8)

        plt.colorbar(im, ax=ax, label="Refusal Gap (pp)", shrink=0.85, pad=0.03)

    fig.suptitle("Category-wise Safety Degradation: Korean vs. Chinese",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    savefig(fig, "category_gap_heatmap")
    print("  [A1] category_gap_heatmap (bilingual)")


def figA2_response_language(summary):
    """A2: Response language distribution — 2×3 per model, KO conditions."""
    model_cond_lang = {}
    for m in summary:
        model = m["model_name"]
        cond = m["condition"]
        if model not in model_cond_lang:
            model_cond_lang[model] = {}
        total = sum(m["language_distribution"].values())
        model_cond_lang[model][cond] = {
            lang: count / total * 100 if total > 0 else 0
            for lang, count in m["language_distribution"].items()
        }

    models = [m for m in MODEL_ORDER if m in model_cond_lang]
    conds = [c for c in KO_COND_ORDER if any(c in model_cond_lang[m] for m in models)]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    lang_colors = {"en": "#4363d8", "ko": "#e6194B", "zh": "#f58231", "mixed": "#9E9E9E", "other": "#BDBDBD"}
    lang_labels_map = {"en": "English", "ko": "Korean", "zh": "Chinese", "mixed": "Mixed", "other": "Other"}

    for idx, (model, ax) in enumerate(zip(models, axes)):
        short = MODEL_SHORT[model]
        x = np.arange(len(conds))
        bottom = np.zeros(len(conds))

        for lang in ["en", "ko", "zh", "mixed", "other"]:
            vals = [model_cond_lang[model].get(c, {}).get(lang, 0) for c in conds]
            if max(vals) > 0.1:
                ax.bar(x, vals, bottom=bottom, label=lang_labels_map[lang],
                       color=lang_colors[lang], edgecolor="white", linewidth=0.5)
                bottom += np.array(vals)

        ax.set_xticks(x)
        ax.set_xticklabels([KO_COND_LABELS_SHORT[c] for c in conds],
                           rotation=45, ha="right", fontsize=7.5)
        ax.set_ylabel("Response Language (%)")
        ax.set_title(short, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 105)
        if idx == 0:
            ax.legend(fontsize=8)

    fig.suptitle("Response Language Distribution (Korean Conditions)",
                 fontsize=14, y=1.01, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "response_language")
    print("  [A2] response_language")


def figA3_radar_bilingual(analysis):
    """A3: Radar charts per model — KO (left) + ZH (right) in same plot."""
    asr = analysis["asr_matrix"]
    models = [m for m in MODEL_ORDER if m in asr]

    ko_conds_no_en = ["ko", "mix", "ko2en", "adaptive", "ko_gemini", "ko2en_gemini", "ko_gpt", "ko2en_gpt"]
    zh_conds_no_en = ["zh", "mix_zh", "zh2en", "adaptive_zh", "zh_gemini", "zh2en_gemini", "zh_gpt", "zh2en_gpt"]
    n_conds = len(ko_conds_no_en)
    angles = np.linspace(0, 2 * np.pi, n_conds, endpoint=False).tolist()
    angles += angles[:1]

    radar_labels = ["NLLB", "MIX", "→EN\n(NLLB)", "WS", "Gem.", "→EN\n(Gem.)", "GPT", "→EN\n(GPT)"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12), subplot_kw=dict(polar=True))
    axes = axes.flatten()

    for idx, (model, ax) in enumerate(zip(models, axes)):
        short = MODEL_SHORT[model]

        ko_vals = [asr[model].get(c, 0) * 100 for c in ko_conds_no_en] + \
                  [asr[model].get(ko_conds_no_en[0], 0) * 100]
        zh_vals = [asr[model].get(c, 0) * 100 for c in zh_conds_no_en] + \
                  [asr[model].get(zh_conds_no_en[0], 0) * 100]

        ax.fill(angles, ko_vals, alpha=0.15, color="#e6194B")
        ax.plot(angles, ko_vals, "o-", color="#e6194B", linewidth=2, markersize=4, label="Korean")
        ax.fill(angles, zh_vals, alpha=0.15, color="#f58231")
        ax.plot(angles, zh_vals, "s-", color="#f58231", linewidth=2, markersize=4, label="Chinese")

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_labels, fontsize=7)
        ax.set_ylim(0, 75)
        ax.set_title(short, fontsize=12, fontweight="bold", pad=20)
        if idx == 0:
            ax.legend(loc="upper right", fontsize=8, bbox_to_anchor=(1.3, 1.15))

    fig.suptitle("Per-Model ASR Profile: Korean (red) vs. Chinese (orange)",
                 fontsize=14, y=1.01, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "radar_per_model")
    print("  [A3] radar_per_model (bilingual)")


def figA4_ko2en_zh2en_analysis(analysis):
    """A4: →EN attack vector comparison — KO→EN vs ZH→EN."""
    asr = analysis["asr_matrix"]
    models = [m for m in MODEL_ORDER if m in asr]
    shorts = [MODEL_SHORT[m] for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(models))
    width = 0.35

    # Panel A: KO→EN vs ZH→EN (NLLB)
    ko2en_vals = [asr[m].get("ko2en", 0) * 100 for m in models]
    zh2en_vals = [asr[m].get("zh2en", 0) * 100 for m in models]
    ax1.bar(x - width / 2, ko2en_vals, width * 0.88, label="KO→EN (NLLB)",
            color="#e6194B", edgecolor="white", zorder=3)
    ax1.bar(x + width / 2, zh2en_vals, width * 0.88, label="ZH→EN (NLLB)",
            color="#f58231", edgecolor="white", zorder=3)
    for i, (ko, zh) in enumerate(zip(ko2en_vals, zh2en_vals)):
        ax1.text(x[i] - width / 2, ko + 0.5, f"{ko:.0f}", ha="center", fontsize=8, fontweight="bold")
        ax1.text(x[i] + width / 2, zh + 0.5, f"{zh:.0f}", ha="center", fontsize=8, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(shorts, fontsize=9)
    ax1.set_ylabel("ASR (%)")
    ax1.set_title("(a) KO→EN vs. ZH→EN (NLLB)", fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y")

    # Panel B: Scatter KO→EN vs ZH→EN across all engines
    markers = {"NLLB": "o", "Gemini": "s", "GPT": "D"}
    for model in models:
        for engine in TRANS_ENGINES:
            ko_val = asr[model].get(KO2EN_TRANS[engine], 0) * 100
            zh_val = asr[model].get(ZH2EN_TRANS[engine], 0) * 100
            ax2.scatter(zh_val, ko_val, s=120,
                        color=MODEL_COLORS[model], marker=markers[engine],
                        edgecolors="white", linewidth=1, zorder=5)
    ax2.plot([0, 65], [0, 65], "k--", alpha=0.2, linewidth=1)
    ax2.set_xlabel("ZH→EN ASR (%)")
    ax2.set_ylabel("KO→EN ASR (%)")
    ax2.set_title("(b) KO→EN vs. ZH→EN Scatter (All Engines)", fontweight="bold")
    ax2.grid(alpha=0.3)
    ax2.set_xlim(-2, 65)
    ax2.set_ylim(-2, 65)
    ax2.set_aspect("equal")

    model_patches = [mpatches.Patch(color=MODEL_COLORS[m], label=MODEL_SHORT[m]) for m in models]
    engine_handles = [plt.Line2D([0], [0], marker=markers[e], color="gray", markersize=7,
                                  linestyle="None", label=e) for e in TRANS_ENGINES]
    ax2.legend(handles=model_patches + engine_handles, loc="upper left", fontsize=7, ncol=2)

    fig.suptitle("→EN Attack Vector: Korean vs. Chinese", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    savefig(fig, "ko2en_analysis")
    print("  [A4] ko2en_analysis (bilingual)")


def figA5_grouped_bar_all(analysis):
    """A5: Full grouped bar chart — all 17 conditions."""
    asr = analysis["asr_matrix"]
    models = [m for m in MODEL_ORDER if m in asr]
    shorts = [MODEL_SHORT[m] for m in models]

    all_conds = [
        "en", "ko", "mix", "ko2en", "adaptive",
        "ko_gemini", "ko2en_gemini", "ko_gpt", "ko2en_gpt",
        "zh", "mix_zh", "zh2en", "adaptive_zh",
        "zh_gemini", "zh2en_gemini", "zh_gpt", "zh2en_gpt",
    ]

    x = np.arange(len(models))
    n_cond = len(all_conds)
    width = 0.8 / n_cond
    offsets = np.arange(n_cond) - (n_cond - 1) / 2

    # Colors: EN=blue, KO=reds, ZH=oranges
    cond_colors = {
        "en": "#4363d8",
        "ko": "#e6194B", "mix": "#ff7070", "ko2en": "#c0392b", "adaptive": "#800000",
        "ko_gemini": "#e74c3c", "ko2en_gemini": "#b03a2e", "ko_gpt": "#d63031", "ko2en_gpt": "#922b21",
        "zh": "#f58231", "mix_zh": "#ffc078", "zh2en": "#e67e22", "adaptive_zh": "#a04000",
        "zh_gemini": "#f39c12", "zh2en_gemini": "#d68910", "zh_gpt": "#e8a020", "zh2en_gpt": "#b7950b",
    }

    fig, ax = plt.subplots(figsize=(18, 7))
    for i, cond in enumerate(all_conds):
        vals = [asr[m].get(cond, 0) * 100 for m in models]
        ax.bar(x + offsets[i] * width, vals, width * 0.88,
               color=cond_colors.get(cond, "#999999"), edgecolor="white", linewidth=0.2, zorder=3)

    ax.set_ylabel("ASR (%)", fontsize=12)
    ax.set_title("ASR: All Models × All 17 Conditions", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(shorts, fontsize=12)
    ax.set_ylim(0, 75)
    ax.grid(axis="y")

    legend_patches = [
        mpatches.Patch(color="#4363d8", label="EN"),
        mpatches.Patch(color="#e6194B", label="Korean conditions"),
        mpatches.Patch(color="#f58231", label="Chinese conditions"),
    ]
    ax.legend(handles=legend_patches, fontsize=10, loc="upper right")

    savefig(fig, "asr_grouped_bar_all")
    print("  [A5] asr_grouped_bar_all")


def figA6_capability_vs_safety(analysis):
    """A6: Scatter — capability vs safety for both KO and ZH."""
    points = analysis.get("capability_vs_safety", [])
    if not points:
        print("  [A6] SKIPPED (no data)")
        return

    fig, (ax_ko, ax_zh) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, cap_key, deg_key, lang, panel in [
        (ax_ko, "ko_capability", "ko_degradation", "Korean", "(a)"),
        (ax_zh, "zh_capability", "zh_degradation", "Chinese", "(b)"),
    ]:
        for p in points:
            model = p["model"]
            color = MODEL_COLORS.get(model, "#999999")
            cap = p.get(cap_key, 0)
            deg = p.get(deg_key, 0) * 100
            ax.scatter(cap, deg, s=180, color=color, zorder=5,
                       edgecolors="white", linewidth=2)
            ax.annotate(p["short"], (cap, deg),
                        textcoords="offset points", xytext=(10, 5),
                        fontsize=9, fontweight="bold")

        ax.set_xlabel(f"{lang} Capability (approx.)", fontsize=11)
        ax.set_ylabel(f"Safety Degradation ({lang} ASR − EN ASR, pp)", fontsize=10)
        ax.set_title(f"{panel} {lang}", fontsize=12, fontweight="bold")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.grid(alpha=0.3)

    fig.suptitle("Language Capability vs. Safety Degradation", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    savefig(fig, "capability_vs_safety")
    print("  [A6] capability_vs_safety (bilingual)")


def figA7_mix_comparison(analysis):
    """A7: MIX condition comparison — KO and ZH."""
    asr = analysis["asr_matrix"]
    models = [m for m in MODEL_ORDER if m in asr]
    shorts = [MODEL_SHORT[m] for m in models]

    fig, (ax_ko, ax_zh) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    x = np.arange(len(models))
    width = 0.25

    for ax, en_c, mix_c, full_c, lang, panel in [
        (ax_ko, "en", "mix", "ko", "Korean", "(a)"),
        (ax_zh, "en", "mix_zh", "zh", "Chinese", "(b)"),
    ]:
        en_vals = [asr[m].get(en_c, 0) * 100 for m in models]
        mix_vals = [asr[m].get(mix_c, 0) * 100 for m in models]
        full_vals = [asr[m].get(full_c, 0) * 100 for m in models]

        ax.bar(x - width, en_vals, width * 0.88, label="EN", color="#4363d8", edgecolor="white", zorder=3)
        ax.bar(x, mix_vals, width * 0.88, label=f"MIX ({lang[:2].upper()})", color="#f58231", edgecolor="white", zorder=3)
        ax.bar(x + width, full_vals, width * 0.88, label=f"Full {lang[:2].upper()}", color="#e6194B", edgecolor="white", zorder=3)

        for i, (en, mx, fl) in enumerate(zip(en_vals, mix_vals, full_vals)):
            for val, off in zip([en, mx, fl], [-width, 0, width]):
                ax.text(x[i] + off, val + 0.5, f"{val:.1f}", ha="center", fontsize=7, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(shorts, fontsize=9)
        ax.set_title(f"{panel} {lang}: EN vs MIX vs Full", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y")

    ax_ko.set_ylabel("ASR (%)")
    ax_ko.set_ylim(0, max(asr[m].get("ko", 0) * 100 for m in models) * 1.15)

    fig.suptitle("Content Language Dominance: MIX ≈ EN << Full Translation",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    savefig(fig, "mix_vs_en")
    print("  [A7] mix_vs_en (bilingual)")


def figA8_zoomed_heatmap(analysis):
    """A8: Zoomed heatmap (0-20%) focusing on Chinese conditions."""
    asr = analysis["asr_matrix"]
    models = [m for m in MODEL_ORDER if m in asr]
    shorts = [MODEL_SHORT[m] for m in models]

    zh_data = np.array([[asr[m].get(c, 0) * 100 for c in
        ["en", "zh", "mix_zh", "zh2en", "adaptive_zh", "zh_gemini", "zh2en_gemini", "zh_gpt", "zh2en_gpt"]]
        for m in models])

    zh_labels = ["EN", "ZH\n(NLLB)", "MIX-ZH", "ZH→EN\n(NLLB)", "WS-ZH",
                 "ZH\n(Gem.)", "ZH→EN\n(Gem.)", "ZH\n(GPT)", "ZH→EN\n(GPT)"]

    fig, ax = plt.subplots(figsize=(14, 5.5))
    im = ax.imshow(zh_data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=35)

    ax.set_xticks(range(len(zh_labels)))
    ax.set_xticklabels(zh_labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(shorts, fontsize=11)

    for i in range(len(models)):
        for j in range(zh_data.shape[1]):
            val = zh_data[i, j]
            color = "white" if val > 18 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    color=color, fontsize=8.5, fontweight="bold")

    for sep in [0.5, 4.5, 6.5]:
        ax.axvline(x=sep, color="white", linewidth=2.5)

    cbar = plt.colorbar(im, ax=ax, label="ASR (%)", shrink=0.85, pad=0.02)
    cbar.ax.tick_params(labelsize=9)
    ax.set_title("Chinese Conditions: Zoomed Heatmap (0–35%)", pad=15, fontsize=13, fontweight="bold")

    savefig(fig, "well_aligned_heatmap")
    print("  [A8] well_aligned_heatmap (Chinese zoomed)")


def figA9_amplification_bilingual(analysis):
    """A9: Amplification factor comparison — KO and ZH."""
    asr = analysis["asr_matrix"]
    models = [m for m in MODEL_ORDER if m in asr]
    shorts = [MODEL_SHORT[m] for m in models]

    fig, (ax_ko, ax_zh) = plt.subplots(1, 2, figsize=(15, 6), sharey=False)
    x = np.arange(len(models))
    width = 0.25

    for ax, trans_map, title, panel in [
        (ax_ko, KO_TRANS, "(a) Korean Amplification (KO / EN)", None),
        (ax_zh, ZH_TRANS, "(b) Chinese Amplification (ZH / EN)", None),
    ]:
        for i, engine in enumerate(TRANS_ENGINES):
            cond = trans_map[engine]
            ratios = []
            for m in models:
                en_asr = asr[m].get("en", 0)
                tgt_asr = asr[m].get(cond, 0)
                ratio = tgt_asr / en_asr if en_asr > 0.005 else 0
                ratios.append(ratio)
            bars = ax.bar(x + (i - 1) * width, ratios, width * 0.85,
                          label=engine, color=ENGINE_COLORS[engine],
                          edgecolor="white", linewidth=0.8, zorder=3)
            for bar, val in zip(bars, ratios):
                if val > 0.5:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                            f"{val:.1f}×", ha="center", va="bottom", fontsize=7, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(shorts, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(title="Engine", fontsize=8, title_fontsize=8)
        ax.grid(axis="y")
        ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    ax_ko.set_ylabel("Amplification Factor", fontsize=11)

    fig.suptitle("Safety Erosion Amplification by Language and Translation Engine",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    savefig(fig, "amplification_factor")
    print("  [A9] amplification_factor (bilingual)")


def figA10_category_per_condition(summary):
    """A10: Category-level ASR heatmap per model (2×3 grid), Korean conditions."""
    data = {}
    for m in summary:
        model = m["model_name"]
        cond = m["condition"]
        if model not in data:
            data[model] = {}
        data[model][cond] = m["category_refusal_rates"]

    models = [m for m in MODEL_ORDER if m in data]
    conds = [c for c in KO_COND_ORDER if any(c in data[m] for m in models)]
    cats = [c for c in CATEGORY_ORDER if any(c in data[m].get("en", {}) for m in models)]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
    axes = axes.flatten()

    for idx, (model, ax) in enumerate(zip(models, axes)):
        short = MODEL_SHORT[model]
        matrix = np.zeros((len(cats), len(conds)))
        for j, cond in enumerate(conds):
            for i, cat in enumerate(cats):
                ref_rate = data[model].get(cond, {}).get(cat, 1.0)
                matrix[i, j] = (1 - ref_rate) * 100

        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=80)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([KO_COND_LABELS_SHORT[c] for c in conds],
                           rotation=45, ha="right", fontsize=6.5)
        if idx % 3 == 0:
            ax.set_yticks(range(len(cats)))
            ax.set_yticklabels([CATEGORY_SHORT[c] for c in cats], fontsize=9)
        ax.set_title(short, fontsize=11, fontweight="bold")

        for i in range(len(cats)):
            for j in range(len(conds)):
                val = matrix[i, j]
                color = "white" if val > 40 else "black"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", color=color, fontsize=5.5)

    fig.suptitle("Category-Level ASR (%) by Model: Korean Conditions",
                 fontsize=13, y=1.02, fontweight="bold")
    fig.tight_layout()
    fig.subplots_adjust(right=0.93)
    cbar_ax = fig.add_axes([0.95, 0.15, 0.012, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="ASR (%)")

    savefig(fig, "category_per_condition")
    print("  [A10] category_per_condition")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    analysis, summary = load_data()

    print("Generating main figures...")
    fig1_bilingual_heatmap(analysis)
    fig2_cross_language_scatter(analysis)
    fig3_degradation_paired(analysis)
    fig4_translation_engine_bilingual(analysis)
    fig5_language_specific_vulnerability(analysis)
    fig6_condition_ranking(analysis)
    fig7_two_tier_bilingual(analysis)

    print("\nGenerating appendix figures...")
    figA1_category_gap_bilingual(summary)
    figA2_response_language(summary)
    figA3_radar_bilingual(analysis)
    figA4_ko2en_zh2en_analysis(analysis)
    figA5_grouped_bar_all(analysis)
    figA6_capability_vs_safety(analysis)
    figA7_mix_comparison(analysis)
    figA8_zoomed_heatmap(analysis)
    figA9_amplification_bilingual(analysis)
    figA10_category_per_condition(summary)

    print(f"\nAll figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
