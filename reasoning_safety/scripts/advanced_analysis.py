#!/usr/bin/env python3
"""Advanced analysis and visualization for reasoning safety paper."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os

# Style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

with open(os.path.join(RESULTS_DIR, 'evaluation_summary.json')) as f:
    data = json.load(f)

# Build lookup
models = {m['model_name']: m for m in data}

# ========== 1. Paired ASR Comparison (Main Figure) ==========
def fig_paired_asr():
    pairs = [
        ('Llama 8B', 'Llama-3.1-8B-Instruct', 'DeepSeek-R1-Distill-Llama-8B'),
        ('Qwen 7B', 'Qwen2.5-7B-Instruct', 'DeepSeek-R1-Distill-Qwen-7B'),
        ('Qwen 14B', 'Qwen2.5-14B-Instruct', 'DeepSeek-R1-Distill-Qwen-14B'),
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(pairs))
    width = 0.35

    std_asr = [models[p[1]]['asr'] * 100 for p in pairs]
    rea_asr = [models[p[2]]['asr'] * 100 for p in pairs]
    amplification = [r/s if s > 0 else float('inf') for r, s in zip(rea_asr, std_asr)]

    bars1 = ax.bar(x - width/2, std_asr, width, label='Standard Instruct',
                   color='#2ecc71', edgecolor='black', linewidth=0.5, zorder=3)
    bars2 = ax.bar(x + width/2, rea_asr, width, label='Reasoning Distill',
                   color='#e74c3c', edgecolor='black', linewidth=0.5, zorder=3)

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add amplification annotations
    for i, amp in enumerate(amplification):
        ax.annotate(f'{amp:.1f}×', xy=(x[i], max(std_asr[i], rea_asr[i]) + 3),
                   fontsize=12, fontweight='bold', color='#c0392b', ha='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffeaa7', edgecolor='#c0392b', alpha=0.9))

    ax.set_ylabel('Attack Success Rate (%)')
    ax.set_title('Safety Degradation: Standard vs. Reasoning Models')
    ax.set_xticks(x)
    ax.set_xticklabels([p[0] for p in pairs])
    ax.legend(loc='upper left')
    ax.set_ylim(0, 35)
    ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(os.path.join(FIGURES_DIR, 'paired_asr_comparison.png'))
    print("Saved: paired_asr_comparison.png")
    plt.close()

# ========== 2. Category Vulnerability Radar Chart ==========
def fig_category_radar():
    categories = ['Violence', 'Illegal\nActivity', 'Other', 'Deception', 'Self-Harm']
    cat_keys = ['violence', 'illegal_activity', 'other', 'deception', 'self_harm']
    N = len(categories)

    # Average standard refusal
    std_models = ['Llama-3.1-8B-Instruct', 'Qwen2.5-7B-Instruct', 'Qwen2.5-14B-Instruct']
    rea_models = ['DeepSeek-R1-Distill-Llama-8B', 'DeepSeek-R1-Distill-Qwen-7B', 'DeepSeek-R1-Distill-Qwen-14B']

    std_rates = []
    rea_rates = []
    for ck in cat_keys:
        s = np.mean([models[m]['category_refusal_rates'][ck] * 100 for m in std_models])
        r = np.mean([models[m]['category_refusal_rates'][ck] * 100 for m in rea_models])
        std_rates.append(s)
        rea_rates.append(r)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # close
    std_rates += std_rates[:1]
    rea_rates += rea_rates[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    ax.plot(angles, std_rates, 'o-', linewidth=2, color='#2ecc71', label='Standard Instruct')
    ax.fill(angles, std_rates, alpha=0.15, color='#2ecc71')
    ax.plot(angles, rea_rates, 's-', linewidth=2, color='#e74c3c', label='Reasoning Distill')
    ax.fill(angles, rea_rates, alpha=0.15, color='#e74c3c')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=8)

    # Push all tick labels outward to avoid overlapping with chart area
    ax.tick_params(axis='x', pad=18)

    ax.set_title('Category-Wise Refusal Rates\n(Averaged Across Model Families)',
                  pad=35, ha='center')
    ax.legend(loc='lower right', bbox_to_anchor=(1.3, 0))

    fig.savefig(os.path.join(FIGURES_DIR, 'category_radar.png'))
    print("Saved: category_radar.png")
    plt.close()

# ========== 3. Thinking Trace Flow Diagram ==========
def fig_trace_flow():
    rea_models_names = ['R1-Llama-8B', 'R1-Qwen-7B', 'R1-Qwen-14B']
    rea_models_keys = ['DeepSeek-R1-Distill-Llama-8B', 'DeepSeek-R1-Distill-Qwen-7B', 'DeepSeek-R1-Distill-Qwen-14B']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = {
        'Type A\n(Safety → Refuse)': '#2ecc71',
        'Type B\n(Safety → Comply)': '#e74c3c',
        'Type C\n(No Safety → Comply)': '#e67e22',
        'Type D\n(Other → Refuse)': '#3498db',
    }

    for idx, (name, key) in enumerate(zip(rea_models_names, rea_models_keys)):
        m = models[key]
        td = m['trace_distribution']

        type_a = td.get('Type_A_safety_refuse', 0)
        type_b = td.get('Type_B_safety_comply', 0) + td.get('Type_B_safety_rationalize', 0)
        type_c = td.get('Type_C_no_safety', 0)
        type_d = td.get('Type_D_other_refuse', 0)

        labels = ['Type A\n(Safety → Refuse)', 'Type B\n(Safety → Comply)',
                  'Type C\n(No Safety → Comply)', 'Type D\n(Other → Refuse)']
        sizes = [type_a, type_b, type_c, type_d]
        clrs = [colors[l] for l in labels]

        # Filter out zero-size wedges
        filtered = [(l, s, c) for l, s, c in zip(labels, sizes, clrs) if s > 0]
        labels_f, sizes_f, clrs_f = zip(*filtered) if filtered else ([], [], [])

        wedges, texts, autotexts = axes[idx].pie(
            sizes_f, labels=None, colors=clrs_f, autopct='%1.1f%%',
            startangle=90, pctdistance=0.75,
            wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2)
        )
        for at in autotexts:
            at.set_fontsize(9)
            at.set_fontweight('bold')

        axes[idx].set_title(name, fontsize=13, fontweight='bold', pad=10)

        # Center text
        total = sum(sizes)
        axes[idx].text(0, 0, f'n={total}', ha='center', va='center', fontsize=12, fontweight='bold')

    # Shared legend
    patches = [mpatches.Patch(color=c, label=l.replace('\n', ' ')) for l, c in colors.items()]
    fig.legend(handles=patches, loc='lower center', ncol=4, fontsize=10,
              bbox_to_anchor=(0.5, -0.05))

    fig.suptitle('Thinking Trace Classification Distribution', fontsize=14, fontweight='bold', y=1.02)
    fig.savefig(os.path.join(FIGURES_DIR, 'trace_donut.png'))
    print("Saved: trace_donut.png")
    plt.close()

# ========== 4. Category Gap Waterfall Chart ==========
def fig_category_gap():
    categories = ['Deception', 'Violence', 'Illegal Act.', 'Other', 'Self-Harm', 'Discrimination', 'Sexual']
    cat_keys = ['deception', 'violence', 'illegal_activity', 'other', 'self_harm', 'discrimination', 'sexual']

    std_models = ['Llama-3.1-8B-Instruct', 'Qwen2.5-7B-Instruct', 'Qwen2.5-14B-Instruct']
    rea_models = ['DeepSeek-R1-Distill-Llama-8B', 'DeepSeek-R1-Distill-Qwen-7B', 'DeepSeek-R1-Distill-Qwen-14B']

    gaps = []
    for ck in cat_keys:
        s = np.mean([models[m]['category_refusal_rates'][ck] * 100 for m in std_models])
        r = np.mean([models[m]['category_refusal_rates'][ck] * 100 for m in rea_models])
        gaps.append(r - s)

    # Sort by gap magnitude
    sorted_idx = np.argsort(gaps)
    categories_sorted = [categories[i] for i in sorted_idx]
    gaps_sorted = [gaps[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#e74c3c' if g < -5 else '#e67e22' if g < 0 else '#2ecc71' for g in gaps_sorted]

    bars = ax.barh(range(len(gaps_sorted)), gaps_sorted, color=colors, edgecolor='black', linewidth=0.5)

    for i, (bar, gap) in enumerate(zip(bars, gaps_sorted)):
        if gap < -10:
            # Place text well inside the bar (further from the edge)
            ax.text(gap / 2, i, f'{gap:.1f} pp',
                    va='center', ha='center', fontsize=10, fontweight='bold',
                    color='white')
        elif gap < 0:
            # Small negative bars: place text to the right of the bar
            ax.text(gap - 1.0, i, f'{gap:.1f} pp',
                    va='center', ha='right', fontsize=10, fontweight='bold',
                    color='black')
        else:
            # Zero/positive: place to the right
            ax.text(gap + 1.0, i, f'{gap:.1f} pp',
                    va='center', ha='left', fontsize=10, fontweight='bold',
                    color='black')

    ax.set_yticks(range(len(categories_sorted)))
    ax.set_yticklabels(categories_sorted)
    ax.set_xlabel('Refusal Rate Change (percentage points)')
    ax.set_title('Safety Gap: Reasoning vs. Standard Models by Category')
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(os.path.join(FIGURES_DIR, 'category_gap_waterfall.png'))
    print("Saved: category_gap_waterfall.png")
    plt.close()

# ========== 5. Scale Effect Comparison (7B vs 14B) ==========
def fig_scale_effect():
    cat_labels = ['Violence', 'Illegal\nActivity', 'Other', 'Deception', 'Self-Harm']
    cat_keys = ['violence', 'illegal_activity', 'other', 'deception', 'self_harm']

    q7 = models['DeepSeek-R1-Distill-Qwen-7B']
    q14 = models['DeepSeek-R1-Distill-Qwen-14B']

    rates_7b = [q7['category_refusal_rates'][ck] * 100 for ck in cat_keys]
    rates_14b = [q14['category_refusal_rates'][ck] * 100 for ck in cat_keys]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(cat_labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, rates_7b, width, label='R1-Qwen-7B', color='#3498db',
                   edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, rates_14b, width, label='R1-Qwen-14B', color='#9b59b6',
                   edgecolor='black', linewidth=0.5)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)

    # Annotate improvements (placed well above the taller bar's value label)
    for i in range(len(cat_labels)):
        diff = rates_14b[i] - rates_7b[i]
        if abs(diff) > 0.5:
            color = '#27ae60' if diff > 0 else '#c0392b'
            sign = '+' if diff > 0 else ''
            top_val = max(rates_7b[i], rates_14b[i])
            ax.annotate(f'{sign}{diff:.1f}pp', xy=(x[i], top_val + 8),
                       ha='center', fontsize=9, color=color, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                 edgecolor=color, alpha=0.85, linewidth=0.8))

    ax.set_ylabel('Refusal Rate (%)')
    ax.set_title('Scale Effect on Safety: 7B vs. 14B (Qwen-based R1 Distills)')
    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels)
    ax.legend()
    ax.set_ylim(0, 115)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(os.path.join(FIGURES_DIR, 'scale_effect.png'))
    print("Saved: scale_effect.png")
    plt.close()

# ========== 6. Combined Summary Figure (2x2) ==========
def fig_combined():
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # --- Panel A: ASR Comparison ---
    ax1 = fig.add_subplot(gs[0, 0])
    pairs = [
        ('Llama 8B', 'Llama-3.1-8B-Instruct', 'DeepSeek-R1-Distill-Llama-8B'),
        ('Qwen 7B', 'Qwen2.5-7B-Instruct', 'DeepSeek-R1-Distill-Qwen-7B'),
        ('Qwen 14B', 'Qwen2.5-14B-Instruct', 'DeepSeek-R1-Distill-Qwen-14B'),
    ]
    x = np.arange(len(pairs))
    width = 0.35
    std_asr = [models[p[1]]['asr'] * 100 for p in pairs]
    rea_asr = [models[p[2]]['asr'] * 100 for p in pairs]

    ax1.bar(x - width/2, std_asr, width, label='Standard', color='#2ecc71', edgecolor='black', linewidth=0.5)
    ax1.bar(x + width/2, rea_asr, width, label='Reasoning', color='#e74c3c', edgecolor='black', linewidth=0.5)

    for i, (s, r) in enumerate(zip(std_asr, rea_asr)):
        amp = r / s if s > 0 else 0
        ax1.text(i, max(s, r) + 1.5, f'{amp:.0f}×', ha='center', fontsize=11, fontweight='bold', color='#c0392b')

    ax1.set_ylabel('ASR (%)')
    ax1.set_title('(a) Attack Success Rate by Model Pair')
    ax1.set_xticks(x)
    ax1.set_xticklabels([p[0] for p in pairs])
    ax1.legend(fontsize=9)
    ax1.set_ylim(0, 35)
    ax1.grid(axis='y', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # --- Panel B: Category Gap ---
    ax2 = fig.add_subplot(gs[0, 1])
    categories = ['Deception', 'Violence', 'Illegal', 'Other', 'Self-Harm']
    cat_keys = ['deception', 'violence', 'illegal_activity', 'other', 'self_harm']
    std_ms = ['Llama-3.1-8B-Instruct', 'Qwen2.5-7B-Instruct', 'Qwen2.5-14B-Instruct']
    rea_ms = ['DeepSeek-R1-Distill-Llama-8B', 'DeepSeek-R1-Distill-Qwen-7B', 'DeepSeek-R1-Distill-Qwen-14B']

    gaps = []
    for ck in cat_keys:
        s = np.mean([models[m]['category_refusal_rates'][ck] * 100 for m in std_ms])
        r = np.mean([models[m]['category_refusal_rates'][ck] * 100 for m in rea_ms])
        gaps.append(r - s)

    colors_gap = ['#e74c3c' if g < -5 else '#e67e22' if g < 0 else '#2ecc71' for g in gaps]
    ax2.barh(range(len(gaps)), gaps, color=colors_gap, edgecolor='black', linewidth=0.5)
    for i, g in enumerate(gaps):
        ax2.text(g - 2 if g < -10 else g + 0.5, i, f'{g:.1f}pp', va='center', fontsize=9, fontweight='bold',
                color='white' if g < -15 else 'black')
    ax2.set_yticks(range(len(categories)))
    ax2.set_yticklabels(categories)
    ax2.set_xlabel('Refusal Rate Gap (pp)')
    ax2.set_title('(b) Safety Degradation by Category')
    ax2.axvline(x=0, color='black', linewidth=0.8)
    ax2.grid(axis='x', alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # --- Panel C: Trace Distribution (stacked bar) ---
    ax3 = fig.add_subplot(gs[1, 0])
    rea_names = ['R1-Llama\n8B', 'R1-Qwen\n7B', 'R1-Qwen\n14B']
    rea_keys = ['DeepSeek-R1-Distill-Llama-8B', 'DeepSeek-R1-Distill-Qwen-7B', 'DeepSeek-R1-Distill-Qwen-14B']

    type_a = []
    type_b = []
    type_c = []
    type_d = []
    for rk in rea_keys:
        td = models[rk]['trace_distribution']
        total = 520
        type_a.append(td.get('Type_A_safety_refuse', 0) / total * 100)
        type_b.append((td.get('Type_B_safety_comply', 0) + td.get('Type_B_safety_rationalize', 0)) / total * 100)
        type_c.append(td.get('Type_C_no_safety', 0) / total * 100)
        type_d.append(td.get('Type_D_other_refuse', 0) / total * 100)

    x3 = np.arange(len(rea_names))
    w = 0.6
    ax3.bar(x3, type_a, w, label='Type A (Safe refuse)', color='#2ecc71', edgecolor='white')
    ax3.bar(x3, type_b, w, bottom=type_a, label='Type B (Self-rationalize)', color='#e74c3c', edgecolor='white')
    bottoms = [a+b for a,b in zip(type_a, type_b)]
    ax3.bar(x3, type_c, w, bottom=bottoms, label='Type C (No awareness)', color='#e67e22', edgecolor='white')
    bottoms2 = [b+c for b,c in zip(bottoms, type_c)]
    ax3.bar(x3, type_d, w, bottom=bottoms2, label='Type D (Other refuse)', color='#3498db', edgecolor='white')

    ax3.set_ylabel('Proportion (%)')
    ax3.set_title('(c) Thinking Trace Classification')
    ax3.set_xticks(x3)
    ax3.set_xticklabels(rea_names)
    ax3.legend(fontsize=8, loc='center left', bbox_to_anchor=(1, 0.5))
    ax3.set_ylim(0, 105)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # --- Panel D: Response Length ---
    ax4 = fig.add_subplot(gs[1, 1])
    all_names = ['Llama-8B\nStd', 'R1-Llama\n8B', 'Qwen-7B\nStd', 'R1-Qwen\n7B', 'Qwen-14B\nStd', 'R1-Qwen\n14B']
    all_keys = ['Llama-3.1-8B-Instruct', 'DeepSeek-R1-Distill-Llama-8B',
                'Qwen2.5-7B-Instruct', 'DeepSeek-R1-Distill-Qwen-7B',
                'Qwen2.5-14B-Instruct', 'DeepSeek-R1-Distill-Qwen-14B']
    all_colors = ['#2ecc71', '#e74c3c', '#2ecc71', '#e74c3c', '#2ecc71', '#e74c3c']

    lengths = [models[k]['avg_response_length'] for k in all_keys]
    ax4.bar(range(len(all_names)), lengths, color=all_colors, edgecolor='black', linewidth=0.5)
    for i, l in enumerate(lengths):
        ax4.text(i, l + 50, f'{l:.0f}', ha='center', fontsize=8, fontweight='bold')
    ax4.set_xticks(range(len(all_names)))
    ax4.set_xticklabels(all_names, fontsize=8)
    ax4.set_ylabel('Avg. Response Length (chars)')
    ax4.set_title('(d) Response Length Comparison')
    ax4.grid(axis='y', alpha=0.3)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

    fig.savefig(os.path.join(FIGURES_DIR, 'combined_figure.png'))
    print("Saved: combined_figure.png")
    plt.close()

# ========== 7. Comprehensive Analysis Table ==========
def print_analysis():
    print("\n" + "="*80)
    print("COMPREHENSIVE ANALYSIS")
    print("="*80)

    # ASR Amplification
    print("\n--- ASR Amplification Factor ---")
    pairs = [
        ('Llama 8B', 'Llama-3.1-8B-Instruct', 'DeepSeek-R1-Distill-Llama-8B'),
        ('Qwen 7B', 'Qwen2.5-7B-Instruct', 'DeepSeek-R1-Distill-Qwen-7B'),
        ('Qwen 14B', 'Qwen2.5-14B-Instruct', 'DeepSeek-R1-Distill-Qwen-14B'),
    ]
    for name, std_k, rea_k in pairs:
        s = models[std_k]['asr'] * 100
        r = models[rea_k]['asr'] * 100
        amp = r / s if s > 0 else float('inf')
        print(f"  {name}: {s:.1f}% → {r:.1f}% ({amp:.1f}× amplification)")

    # Category Analysis
    print("\n--- Category Vulnerability Ranking ---")
    cat_keys = ['violence', 'illegal_activity', 'other', 'deception', 'self_harm', 'discrimination', 'sexual']
    cat_labels = {'violence':'Violence', 'illegal_activity':'Illegal Activity', 'other':'Other',
                  'deception':'Deception', 'self_harm':'Self-Harm', 'discrimination':'Discrimination', 'sexual':'Sexual'}

    std_ms = ['Llama-3.1-8B-Instruct', 'Qwen2.5-7B-Instruct', 'Qwen2.5-14B-Instruct']
    rea_ms = ['DeepSeek-R1-Distill-Llama-8B', 'DeepSeek-R1-Distill-Qwen-7B', 'DeepSeek-R1-Distill-Qwen-14B']

    cat_gaps = []
    for ck in cat_keys:
        s = np.mean([models[m]['category_refusal_rates'][ck] * 100 for m in std_ms])
        r = np.mean([models[m]['category_refusal_rates'][ck] * 100 for m in rea_ms])
        cat_gaps.append((cat_labels[ck], s, r, r - s))

    cat_gaps.sort(key=lambda x: x[3])
    print(f"  {'Category':<18} {'Std Refusal':>12} {'Rea Refusal':>12} {'Gap':>10}")
    for label, s, r, g in cat_gaps:
        print(f"  {label:<18} {s:>11.1f}% {r:>11.1f}% {g:>+9.1f}pp")

    # Scale Effect
    print("\n--- Scale Effect (Qwen R1: 7B → 14B) ---")
    q7 = models['DeepSeek-R1-Distill-Qwen-7B']
    q14 = models['DeepSeek-R1-Distill-Qwen-14B']
    print(f"  Overall ASR: {q7['asr']*100:.1f}% → {q14['asr']*100:.1f}% (Δ = {(q14['asr']-q7['asr'])*100:+.1f}pp)")
    for ck in cat_keys:
        r7 = q7['category_refusal_rates'][ck] * 100
        r14 = q14['category_refusal_rates'][ck] * 100
        if abs(r14 - r7) > 0.1:
            print(f"  {cat_labels[ck]:<18}: {r7:.1f}% → {r14:.1f}% ({r14-r7:+.1f}pp)")

    # Type B Analysis
    print("\n--- Type B Self-Rationalization Breakdown ---")
    for rk in rea_ms:
        td = models[rk]['trace_distribution']
        b_rat = td.get('Type_B_safety_rationalize', 0)
        b_com = td.get('Type_B_safety_comply', 0)
        total_b = b_rat + b_com
        pct_rat = b_rat / total_b * 100 if total_b > 0 else 0
        pct_com = b_com / total_b * 100 if total_b > 0 else 0
        print(f"  {rk}:")
        print(f"    Total Type B: {total_b} ({total_b/520*100:.1f}%)")
        print(f"    Explicit rationalization: {b_rat} ({pct_rat:.1f}% of B)")
        print(f"    Implicit compliance: {b_com} ({pct_com:.1f}% of B)")

if __name__ == '__main__':
    fig_paired_asr()
    fig_category_radar()
    fig_trace_flow()
    fig_category_gap()
    fig_scale_effect()
    fig_combined()
    print_analysis()
    print("\nAll advanced figures generated!")
