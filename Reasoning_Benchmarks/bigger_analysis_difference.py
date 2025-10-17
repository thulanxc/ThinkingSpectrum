import torch
from transformers import AutoModelForCausalLM
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from tqdm import tqdm
import re
import json

# --- Configuration ---
# 1. SET YOUR TARGET GPU ID HERE
TARGET_GPU_ID = 0

# 2. DEFINE MODELS TO ANALYZE
# MODEL_A_NAME = "Qwen/Qwen2.5-14B"
# MODEL_B_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
MODEL_A_NAME = "Qwen/Qwen3-4B"
MODEL_B_NAME = "Qwen/Qwen3-4B-Instruct-2507"
# MODEL_A_NAME="Qwen/Qwen2.5-Math-1.5B-Instruct"
# MODEL_B_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# --- Main Analysis Function ---
def analyze_model_difference(model_name_a, model_name_b, device_id):
    """
    Performs a deeply detailed and highly efficient Pillar 2 analysis.
    v5.10 (Delta Dist): Added a histogram plot for the raw parameter
    difference (delta) distribution. Handles Grouped-Query Attention models.
    """
    print("--- Starting Advanced Pillar 2 Analysis (v5.10 Delta Dist) ---")

    # 0. Device Check
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires a GPU.")
    if device_id >= torch.cuda.device_count():
        raise RuntimeError(f"Error: GPU {device_id} is not available. Available devices: {torch.cuda.device_count()}.")
    target_device = f"cuda:{device_id}"
    print(f"Targeting device: {target_device}")

    # 1. Setup Output Directory
    model_pair_name = f"{model_name_a.split('/')[-1]}_vs_{model_name_b.split('/')[-1]}"
    output_dir = os.path.join("diff_analysis_results", model_pair_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved in: {output_dir}")

    # 2. Load Models
    print(f"\nLoading model A ({model_name_a}) onto {target_device}")
    model_a = AutoModelForCausalLM.from_pretrained(
        model_name_a, torch_dtype=torch.bfloat16, device_map=target_device, trust_remote_code=True
    )
    model_a.eval()

    print(f"Loading model B ({model_name_b}) onto {target_device}")
    model_b = AutoModelForCausalLM.from_pretrained(
        model_name_b, torch_dtype=torch.bfloat16, device_map=target_device, trust_remote_code=True
    )
    model_b.eval()

    config = model_a.config
    num_q_heads = config.num_attention_heads
    num_kv_heads = getattr(config, 'num_key_value_heads', num_q_heads)
    
    params_a = model_a.state_dict()
    params_b = model_b.state_dict()

    # 3. Calculate Differences and Collect Granular & Global Data
    print("\nCalculating parameter differences and collecting data...")
    sub_module_l1, sub_module_l2, sub_module_counts = {}, {}, {}
    attn_head_l1, attn_head_l2, attn_head_counts = {}, {}, {}

    # Bins for L1/L2 delta distributions
    num_hist_bins = 10000
    l1_bins_cpu = torch.logspace(start=-13, end=2, steps=num_hist_bins, device='cpu')
    l1_hist_counts_cpu = torch.zeros(num_hist_bins - 1, dtype=torch.float32, device='cpu')
    l2_bins_cpu = torch.logspace(start=-26, end=4, steps=num_hist_bins, device='cpu')
    l2_hist_counts_cpu = torch.zeros(num_hist_bins - 1, dtype=torch.float32, device='cpu')
    
    # Bins for original parameter distributions
    num_param_bins = 2000
    bins_raw_cpu = torch.linspace(-1.0, 1.0, steps=num_param_bins, device='cpu') 
    hist_a_raw_counts_cpu = torch.zeros(num_param_bins - 1, dtype=torch.float32, device='cpu')
    hist_b_raw_counts_cpu = torch.zeros(num_param_bins - 1, dtype=torch.float32, device='cpu')
    bins_abs_cpu = torch.logspace(start=-13, end=2, steps=num_param_bins, device='cpu')
    hist_a_abs_counts_cpu = torch.zeros(num_param_bins - 1, dtype=torch.float32, device='cpu')
    hist_b_abs_counts_cpu = torch.zeros(num_param_bins - 1, dtype=torch.float32, device='cpu')
    bins_sq_cpu = torch.logspace(start=-26, end=4, steps=num_param_bins, device='cpu')
    hist_a_sq_counts_cpu = torch.zeros(num_param_bins - 1, dtype=torch.float32, device='cpu')
    hist_b_sq_counts_cpu = torch.zeros(num_param_bins - 1, dtype=torch.float32, device='cpu')

    # <<< 新增开始 >>>
    # Bins and accumulator for parameter difference (delta) distribution
    print("Initializing bins for delta distribution analysis...")
    num_delta_bins = 2000
    # Linear space is best for values centered around zero. Adjust range if needed.
    bins_delta_cpu = torch.linspace(-0.2, 0.2, steps=num_delta_bins, device='cpu') 
    hist_delta_counts_cpu = torch.zeros(num_delta_bins - 1, dtype=torch.float32, device='cpu')
    # <<< 新增结束 >>>

    total_params = 0
    global_stats = {
        "model_a_l1": 0.0, "model_a_l2_sq": 0.0,
        "model_b_l1": 0.0, "model_b_l2_sq": 0.0,
        "delta_l1": 0.0, "delta_l2_sq": 0.0
    }

    layer_re = re.compile(r"model\.layers\.(\d+)\.")
    param_iterator = tqdm(params_a.keys(), desc="Processing parameters")

    with torch.no_grad():
        for param_name in param_iterator:
            if param_name not in params_b: continue

            tensor_a, tensor_b = params_a[param_name], params_b[param_name]
            delta = (tensor_b - tensor_a).float()

            # --- Delta Analysis ---
            magnitudes = delta.abs()
            energies = delta.pow(2)
            
            magnitudes_cpu = magnitudes.cpu()
            l1_hist, _ = torch.histogram(magnitudes_cpu, bins=l1_bins_cpu)
            l1_hist_counts_cpu += l1_hist

            energies_cpu = energies.cpu()
            l2_hist, _ = torch.histogram(energies_cpu, bins=l2_bins_cpu)
            l2_hist_counts_cpu += l2_hist
            
            # <<< 新增开始 >>>
            # --- Delta Distribution Analysis ---
            delta_cpu = delta.cpu()
            hist_delta, _ = torch.histogram(delta_cpu, bins=bins_delta_cpu)
            hist_delta_counts_cpu += hist_delta
            # <<< 新增结束 >>>
            
            # --- Global Stats ---
            global_stats["model_a_l1"] += tensor_a.float().abs().sum().item()
            global_stats["model_a_l2_sq"] += tensor_a.float().pow(2).sum().item()
            global_stats["model_b_l1"] += tensor_b.float().abs().sum().item()
            global_stats["model_b_l2_sq"] += tensor_b.float().pow(2).sum().item()
            global_stats["delta_l1"] += magnitudes.sum().item()
            global_stats["delta_l2_sq"] += energies.sum().item()
            
            param_count = delta.numel()
            total_params += param_count
            
            # --- Original Parameter Distribution Analysis ---
            tensor_a_cpu_float = tensor_a.float().cpu()
            tensor_b_cpu_float = tensor_b.float().cpu()
            
            hist_a_raw, _ = torch.histogram(tensor_a_cpu_float, bins=bins_raw_cpu)
            hist_b_raw, _ = torch.histogram(tensor_b_cpu_float, bins=bins_raw_cpu)
            hist_a_raw_counts_cpu += hist_a_raw
            hist_b_raw_counts_cpu += hist_b_raw

            hist_a_abs, _ = torch.histogram(tensor_a_cpu_float.abs(), bins=bins_abs_cpu)
            hist_b_abs, _ = torch.histogram(tensor_b_cpu_float.abs(), bins=bins_abs_cpu)
            hist_a_abs_counts_cpu += hist_a_abs
            hist_b_abs_counts_cpu += hist_b_abs

            hist_a_sq, _ = torch.histogram(tensor_a_cpu_float.pow(2), bins=bins_sq_cpu)
            hist_b_sq, _ = torch.histogram(tensor_b_cpu_float.pow(2), bins=bins_sq_cpu)
            hist_a_sq_counts_cpu += hist_a_sq
            hist_b_sq_counts_cpu += hist_b_sq

            # --- Granular Per-Layer/Module Analysis ---
            match = layer_re.search(param_name)
            if match:
                layer_idx = int(match.group(1))

                if layer_idx not in sub_module_l1: sub_module_l1[layer_idx] = {}
                if layer_idx not in sub_module_l2: sub_module_l2[layer_idx] = {}
                if layer_idx not in sub_module_counts: sub_module_counts[layer_idx] = {}
                if layer_idx not in attn_head_l1: attn_head_l1[layer_idx] = np.zeros(num_q_heads)
                if layer_idx not in attn_head_l2: attn_head_l2[layer_idx] = np.zeros(num_q_heads)
                if layer_idx not in attn_head_counts: attn_head_counts[layer_idx] = np.zeros(num_q_heads)

                l2_energy_val = energies.sum().item()
                l1_norm_val = magnitudes.sum().item()

                if "self_attn" in param_name and 'weight' in param_name:
                    if "q_proj" in param_name:
                        head_param_count = param_count / num_q_heads
                        delta_reshaped = delta.view(num_q_heads, -1, delta.shape[-1])
                        attn_head_l2[layer_idx] += delta_reshaped.pow(2).sum(dim=(1,2)).cpu().numpy()
                        attn_head_l1[layer_idx] += delta_reshaped.abs().sum(dim=(1,2)).cpu().numpy()
                        attn_head_counts[layer_idx] += head_param_count
                    
                    elif any(p in param_name for p in ["k_proj", "v_proj"]):
                        head_param_count = param_count / num_q_heads
                        delta_reshaped = delta.view(num_kv_heads, -1, delta.shape[-1])
                        per_kv_head_l2 = delta_reshaped.pow(2).sum(dim=(1,2)).cpu().numpy()
                        per_kv_head_l1 = delta_reshaped.abs().sum(dim=(1,2)).cpu().numpy()
                        if num_q_heads != num_kv_heads:
                            repetition_factor = num_q_heads // num_kv_heads
                            per_kv_head_l2_repeated = np.repeat(per_kv_head_l2, repetition_factor)
                            per_kv_head_l1_repeated = np.repeat(per_kv_head_l1, repetition_factor)
                            attn_head_l2[layer_idx] += per_kv_head_l2_repeated
                            attn_head_l1[layer_idx] += per_kv_head_l1_repeated
                        else:
                            attn_head_l2[layer_idx] += per_kv_head_l2
                            attn_head_l1[layer_idx] += per_kv_head_l1
                        attn_head_counts[layer_idx] += head_param_count

                    elif "o_proj" in param_name:
                        head_param_count = param_count / num_q_heads
                        delta_reshaped = delta.view(delta.shape[0], num_q_heads, -1)
                        attn_head_l2[layer_idx] += delta_reshaped.pow(2).sum(dim=(0,2)).cpu().numpy()
                        attn_head_l1[layer_idx] += delta_reshaped.abs().sum(dim=(0,2)).cpu().numpy()
                        attn_head_counts[layer_idx] += head_param_count

                if "self_attn.q_proj" in param_name: sub_module = "Attn Q Proj"
                elif "self_attn.k_proj" in param_name: sub_module = "Attn K Proj"
                elif "self_attn.v_proj" in param_name: sub_module = "Attn V Proj"
                elif "self_attn.o_proj" in param_name: sub_module = "Attn O Proj"
                elif "mlp.gate_proj" in param_name: sub_module = "MLP Gate Proj"
                elif "mlp.up_proj" in param_name: sub_module = "MLP Up Proj"
                elif "mlp.down_proj" in param_name: sub_module = "MLP Down Proj"
                elif "input_layernorm" in param_name: sub_module = "Input LN"
                elif "post_attention_layernorm" in param_name: sub_module = "Post-Attn LN"
                else: continue

                sub_module_l2[layer_idx][sub_module] = sub_module_l2[layer_idx].get(sub_module, 0) + l2_energy_val
                sub_module_l1[layer_idx][sub_module] = sub_module_l1[layer_idx].get(sub_module, 0) + l1_norm_val
                sub_module_counts[layer_idx][sub_module] = sub_module_counts[layer_idx].get(sub_module, 0) + param_count

    del model_a, model_b, params_a, params_b
    torch.cuda.empty_cache()

    # 4. Save Global Stats and Generate Plots
    save_global_stats(global_stats, model_pair_name, output_dir)
    print(f"\nGenerating detailed plots (v5.10 Delta Dist)...")
    
    l2_dir = os.path.join(output_dir, "l2_norm_avg_per_param")
    os.makedirs(l2_dir, exist_ok=True)
    plot_attention_head_spotlight(attn_head_l2, attn_head_counts, model_pair_name, l2_dir, config, "Avg L2 Norm (Energy)")
    generate_combined_trend_plot(sub_module_l2, sub_module_counts, model_pair_name, l2_dir, config, "Avg L2 Norm (Energy)")
    plot_per_module_trends(sub_module_l2, sub_module_counts, model_pair_name, l2_dir, config, "Avg L2 Norm (Energy)")

    l1_dir = os.path.join(output_dir, "l1_norm_avg_per_param")
    os.makedirs(l1_dir, exist_ok=True)
    plot_attention_head_spotlight(attn_head_l1, attn_head_counts, model_pair_name, l1_dir, config, "Avg L1 Norm (Abs Change)")
    generate_combined_trend_plot(sub_module_l1, sub_module_counts, model_pair_name, l1_dir, config, "Avg L1 Norm (Abs Change)")
    plot_per_module_trends(sub_module_l1, sub_module_counts, model_pair_name, l1_dir, config, "Avg L1 Norm (Abs Change)")

    plot_norm_distributions(attn_head_l1, attn_head_l2, attn_head_counts, model_pair_name, output_dir, config)
    plot_cumulative_l2_energy_final(l2_hist_counts_cpu, l2_bins_cpu, total_params, model_pair_name, output_dir)
    plot_cumulative_l1_norm_final(l1_hist_counts_cpu, l1_bins_cpu, total_params, model_pair_name, output_dir)
    
    plot_parameter_distributions(
        hist_a_raw_counts_cpu, hist_b_raw_counts_cpu, bins_raw_cpu,
        hist_a_abs_counts_cpu, hist_b_abs_counts_cpu, bins_abs_cpu,
        hist_a_sq_counts_cpu, hist_b_sq_counts_cpu, bins_sq_cpu,
        model_name_a, model_name_b, model_pair_name, output_dir
    )

    # <<< 新增开始 >>>
    # Call the new plotting function for the delta distribution
    plot_delta_distribution(
        hist_delta_counts_cpu, bins_delta_cpu,
        model_pair_name, output_dir
    )
    # <<< 新增结束 >>>

    print("\n--- Advanced Pillar 2 Analysis (v5.10 Delta Dist) Complete! ---")

# --- Plotting and Helper Functions ---

def save_global_stats(stats, model_pair_name, output_dir):
    summary = {
        "model_A": {"L1_Norm": stats["model_a_l1"], "L2_Norm_Energy_Sqrt": np.sqrt(stats["model_a_l2_sq"])},
        "model_B": {"L1_Norm": stats["model_b_l1"], "L2_Norm_Energy_Sqrt": np.sqrt(stats["model_b_l2_sq"])},
        "delta_diff": {"L1_Norm": stats["delta_l1"], "L2_Norm_Energy_Sqrt": np.sqrt(stats["delta_l2_sq"])},
        "relative_change_vs_model_A": {
            "relative_L1_change": f"{stats['delta_l1'] / (stats['model_a_l1'] + 1e-9):.4%}",
            "relative_L2_change": f"{np.sqrt(stats['delta_l2_sq']) / (np.sqrt(stats['model_a_l2_sq']) + 1e-9):.4%}",
        }
    }
    filepath = os.path.join(output_dir, "global_norm_summary.json")
    with open(filepath, 'w') as f: json.dump(summary, f, indent=4)
    print(f"\nSaved global norm statistics to: {filepath}")
    print(json.dumps(summary["relative_change_vs_model_A"], indent=4))

def plot_attention_head_spotlight(head_norms, head_counts, model_pair_name, output_dir, config, norm_type):
    print(f"  - Generating {norm_type} Head Spotlight Heatmap...")
    num_layers, num_heads = config.num_hidden_layers, config.num_attention_heads
    is_l2 = "L2" in norm_type
    avg_norms = np.zeros((num_layers, num_heads))
    for i in range(num_layers):
        counts = head_counts.get(i, np.ones(num_heads))
        norms = head_norms.get(i, np.zeros(num_heads))
        avg_norms[i, :] = norms / np.maximum(counts, 1)
    transform = np.sqrt if is_l2 else lambda x: x
    heatmap_data = transform(avg_norms)
    fig, ax = plt.subplots(figsize=(16, 12))
    sns.heatmap(heatmap_data, ax=ax, cmap="inferno", linewidths=0)
    ax.set_title(f"Attention Head Spotlight: {norm_type} per Parameter\n({model_pair_name})", fontsize=16)
    ax.set_xlabel("Attention Head Index", fontsize=12)
    ax.set_ylabel("Transformer Layer Index", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"spotlight_attention_heads_{norm_type.split()[1].lower()}.png"), dpi=300)
    plt.close(fig)

def generate_combined_trend_plot(norms, counts, model_pair_name, output_dir, config, norm_type):
    print(f"  - Generating Combined {norm_type} Trend Overview...")
    num_layers = config.num_hidden_layers
    trends_data = {}
    all_sub_modules = sorted(list(set(sm for layer_data in norms.values() for sm in layer_data.keys())))
    is_l2 = "L2" in norm_type
    transform = np.sqrt if is_l2 else lambda x: x
    for sm_name in all_sub_modules:
        trends_data[sm_name] = [transform(norms.get(i, {}).get(sm_name, 0) / max(counts.get(i, {}).get(sm_name, 1), 1)) for i in range(num_layers)]
    fig, ax = plt.subplots(figsize=(16, 9))
    colors = plt.colormaps.get_cmap('tab10')
    for i, (module_name, layer_norms_data) in enumerate(trends_data.items()):
        ax.plot(range(num_layers), layer_norms_data, marker='.', linestyle='-', label=module_name, color=colors(i))
    ax.set_yscale('log')
    ax.set_title(f"Combined Sub-Module {norm_type} per Parameter Across Layers (Log Scale)\n({model_pair_name})", fontsize=16)
    ax.set_xlabel("Transformer Layer Index", fontsize=12)
    ax.set_ylabel(f"{norm_type} per Parameter (Log Scale)", fontsize=12)
    ax.grid(True, which="both", linestyle='--', alpha=0.5)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"combined_trends_{norm_type.split()[1].lower()}.png"), dpi=300)
    plt.close(fig)

def plot_per_module_trends(norms, counts, model_pair_name, base_output_dir, config, norm_type):
    print(f"  - Generating Per-Module {norm_type} Trend Plots...")
    num_layers = config.num_hidden_layers
    all_sub_modules = sorted(list(set(sm for layer_data in norms.values() for sm in layer_data.keys())))
    is_l2 = "L2" in norm_type
    transform = np.sqrt if is_l2 else lambda x: x
    output_dir = os.path.join(base_output_dir, "per_module_trends")
    os.makedirs(output_dir, exist_ok=True)
    for sm_name in all_sub_modules:
        layer_norms_data = [transform(norms.get(i, {}).get(sm_name, 0) / max(counts.get(i, {}).get(sm_name, 1), 1)) for i in range(num_layers)]
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(range(num_layers), layer_norms_data, marker='o', linestyle='-')
        ax.set_title(f"Trend for {sm_name}: {norm_type} per Parameter\n({model_pair_name})", fontsize=16)
        ax.set_xlabel("Transformer Layer Index", fontsize=12)
        ax.set_ylabel(f"{norm_type} per Parameter", fontsize=12)
        ax.grid(True, which="both", linestyle='--', alpha=0.5)
        plt.tight_layout()
        safe_sm_name = sm_name.replace(" ", "_").replace("/", "_")
        filename = f"trend_{norm_type.split()[1].lower()}_{safe_sm_name}.png"
        fig.savefig(os.path.join(output_dir, filename), dpi=150)
        plt.close(fig)

def plot_norm_distributions(attn_head_l1, attn_head_l2, attn_head_counts, model_pair_name, output_dir, config):
    print("  - Generating L1/L2 Norm Distribution Histograms...")
    num_layers, num_heads = config.num_hidden_layers, config.num_attention_heads
    l1_norms_flat, l2_norms_flat = [], []
    for i in range(num_layers):
        counts = attn_head_counts.get(i, np.ones(num_heads))
        l1_norms = attn_head_l1.get(i, np.zeros(num_heads))
        l2_energies = attn_head_l2.get(i, np.zeros(num_heads))
        l1_norms_flat.extend(l1_norms / np.maximum(counts, 1))
        l2_norms_flat.extend(np.sqrt(l2_energies / np.maximum(counts, 1)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    ax1.hist(l1_norms_flat, bins=50, color='skyblue', edgecolor='black')
    ax1.set_title(f"Distribution of Avg L1 Norm per Parameter\nAcross All {num_heads*num_layers} Attention Heads", fontsize=14)
    ax1.set_xlabel("Avg L1 Norm per Parameter", fontsize=12)
    ax1.set_ylabel("Frequency (Count of Heads)", fontsize=12)
    ax1.grid(alpha=0.6)
    ax2.hist(l2_norms_flat, bins=50, color='salmon', edgecolor='black')
    ax2.set_title(f"Distribution of Avg L2 Norm per Parameter\nAcross All {num_heads*num_layers} Attention Heads", fontsize=14)
    ax2.set_xlabel("Avg L2 Norm per Parameter", fontsize=12)
    ax2.grid(alpha=0.6)
    fig.suptitle(f"Attention Head Change Distribution (Normalized by Parameter Count)\n({model_pair_name})", fontsize=18, weight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join(output_dir, "distribution_of_head_norms_avg.png"), dpi=300)
    plt.close(fig)

def plot_cumulative_l1_norm_final(hist_counts_cpu, hist_bins_cpu, total_params, model_pair_name, output_dir):
    print("  - Generating Cumulative L1 Norm Curve from histogram...")
    counts = hist_counts_cpu
    bin_centers = (hist_bins_cpu[:-1] + hist_bins_cpu[1:]) / 2
    counts_desc, bin_centers_desc = torch.flip(counts, dims=[0]), torch.flip(bin_centers, dims=[0])
    cumulative_counts = torch.cumsum(counts_desc, dim=0)
    cumulative_l1 = torch.cumsum(counts_desc * bin_centers_desc, dim=0)
    approximated_total_l1 = cumulative_l1[-1]
    y_axis_percent = (cumulative_l1 / (approximated_total_l1 + 1e-12)).numpy()
    x_axis_percent = (cumulative_counts / total_params * 100).numpy()
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(x_axis_percent, y_axis_percent, color='dodgerblue', lw=2.5)
    ax.set_title(f"Cumulative L1 Norm of Sorted Parameter Differences\n({model_pair_name})", fontsize=16)
    ax.set_xlabel("Percentage of Parameters (Sorted by |$\Delta$| magnitude)", fontsize=12)
    ax.set_ylabel("Cumulative Percentage of Total L1 Norm ($\sum |\Delta|$)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    for percentile in [1, 5, 10, 20]:
        idx = np.searchsorted(x_axis_percent, percentile)
        if idx < len(y_axis_percent):
            y_val = y_axis_percent[idx]
            ax.axhline(y=y_val, color='grey', linestyle=':', xmax=x_axis_percent[idx]/100.0)
            ax.axvline(x=x_axis_percent[idx], color='grey', linestyle=':', ymax=y_val)
            ax.text(x_axis_percent[idx] + 1, y_val - 0.05, f"Top {percentile:.0f}% params\n~{y_val*100:.1f}% of L1 norm", fontsize=10)
    ax.set_xlim(0, 100); ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter()); ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    plt.tight_layout(); fig.savefig(os.path.join(output_dir, "cumulative_l1_norm_curve.png"), dpi=300)
    plt.close(fig); print("  - Saved cumulative L1 norm curve plot.")

def plot_cumulative_l2_energy_final(hist_counts_cpu, hist_bins_cpu, total_params, model_pair_name, output_dir):
    print("  - Generating Cumulative L2 Energy Curve from histogram...")
    counts = hist_counts_cpu
    bin_centers = (hist_bins_cpu[:-1] + hist_bins_cpu[1:]) / 2
    counts_desc, bin_centers_desc = torch.flip(counts, dims=[0]), torch.flip(bin_centers, dims=[0])
    cumulative_counts = torch.cumsum(counts_desc, dim=0)
    cumulative_energy = torch.cumsum(counts_desc * bin_centers_desc, dim=0)
    approximated_total_energy = cumulative_energy[-1]
    y_axis_percent = (cumulative_energy / (approximated_total_energy + 1e-18)).numpy()
    x_axis_percent = (cumulative_counts / total_params * 100).numpy()
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(x_axis_percent, y_axis_percent, color='darkviolet', lw=2.5)
    ax.set_title(f"Cumulative L2 Energy of Sorted Parameter Differences\n({model_pair_name})", fontsize=16)
    ax.set_xlabel("Percentage of Parameters (Sorted by |$\Delta$| magnitude)", fontsize=12)
    ax.set_ylabel("Cumulative Percentage of Total L2 Energy ($\sum \Delta^2$)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    for percentile in [1, 5, 10, 20]:
        idx = np.searchsorted(x_axis_percent, percentile)
        if idx < len(y_axis_percent):
            y_val = y_axis_percent[idx]
            ax.axhline(y=y_val, color='grey', linestyle=':', xmax=x_axis_percent[idx]/100.0)
            ax.axvline(x=x_axis_percent[idx], color='grey', linestyle=':', ymax=y_val)
            ax.text(x_axis_percent[idx] + 1, y_val - 0.05, f"Top {percentile:.0f}% params\n~{y_val*100:.1f}% energy", fontsize=10)
    ax.set_xlim(0, 100); ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter()); ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    plt.tight_layout(); fig.savefig(os.path.join(output_dir, "cumulative_l2_energy_curve.png"), dpi=300)
    plt.close(fig); print("  - Saved cumulative L2 energy curve plot.")

def plot_parameter_distributions(
    hist_a_raw, hist_b_raw, bins_raw, hist_a_abs, hist_b_abs, bins_abs,
    hist_a_sq, hist_b_sq, bins_sq, model_name_a, model_name_b, model_pair_name, output_dir):
    print("  - Generating original parameter distribution histograms...")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))
    model_a_label, model_b_label = model_name_a.split('/')[-1], model_name_b.split('/')[-1]
    bin_centers_raw = (bins_raw[:-1] + bins_raw[1:]) / 2
    ax1.plot(bin_centers_raw, hist_a_raw, lw=2, alpha=0.8, label=model_a_label)
    ax1.plot(bin_centers_raw, hist_b_raw, lw=2, alpha=0.8, label=model_b_label)
    ax1.set_title("Raw Parameter Value Distribution", fontsize=14); ax1.set_xlabel("Parameter Value", fontsize=12)
    ax1.set_ylabel("Frequency (Count)", fontsize=12); ax1.set_yscale('log'); ax1.grid(True, linestyle='--', alpha=0.6); ax1.legend()
    bin_centers_abs = (bins_abs[:-1] + bins_abs[1:]) / 2
    ax2.plot(bin_centers_abs, hist_a_abs, lw=2, alpha=0.8, label=model_a_label)
    ax2.plot(bin_centers_abs, hist_b_abs, lw=2, alpha=0.8, label=model_b_label)
    ax2.set_title("Absolute Parameter Value Distribution", fontsize=14); ax2.set_xlabel("Parameter Absolute Value |w|", fontsize=12)
    ax2.set_ylabel("Frequency (Count)", fontsize=12); ax2.set_xscale('log'); ax2.set_yscale('log'); ax2.grid(True, which="both", linestyle='--', alpha=0.6); ax2.legend()
    bin_centers_sq = (bins_sq[:-1] + bins_sq[1:]) / 2
    ax3.plot(bin_centers_sq, hist_a_sq, lw=2, alpha=0.8, label=model_a_label)
    ax3.plot(bin_centers_sq, hist_b_sq, lw=2, alpha=0.8, label=model_b_label)
    ax3.set_title("Squared Parameter Value Distribution", fontsize=14); ax3.set_xlabel("Parameter Squared Value w^2", fontsize=12)
    ax3.set_ylabel("Frequency (Count)", fontsize=12); ax3.set_xscale('log'); ax3.set_yscale('log'); ax3.grid(True, which="both", linestyle='--', alpha=0.6); ax3.legend()
    fig.suptitle(f"Original Parameter Distributions\n({model_pair_name})", fontsize=18, weight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filename = os.path.join(output_dir, "original_parameter_distributions.png")
    fig.savefig(filename, dpi=300); plt.close(fig)
    print(f"  - Saved original parameter distribution plot to: {filename}")

# <<< 新增函数开始 >>>
def plot_delta_distribution(hist_delta, bins_delta, model_pair_name, output_dir):
    """
    Plots a histogram of the parameter differences (deltas).
    """
    print("  - Generating parameter difference (delta) distribution histogram...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bin_centers_delta = (bins_delta[:-1] + bins_delta[1:]).numpy() / 2
    hist_delta_np = hist_delta.numpy()
    
    # Use a bar plot for a classic histogram look
    ax.bar(bin_centers_delta, hist_delta_np, width=(bins_delta[1] - bins_delta[0]).item(),
           color='mediumseagreen', edgecolor=None, alpha=0.9)
    
    ax.set_title(f"Parameter Difference ($\Delta$) Distribution\n({model_pair_name})", fontsize=16)
    ax.set_xlabel("Parameter Difference Value ($\Delta$ = Model B - Model A)", fontsize=12)
    ax.set_ylabel("Frequency (Count)", fontsize=12)
    ax.set_yscale('log') # Log scale is essential due to the high peak at zero
    ax.grid(True, which="both", linestyle='--', alpha=0.5)
    
    # Add text about the log scale for clarity
    ax.text(0.98, 0.98, 'Y-axis is log-scaled',
            verticalalignment='top', horizontalalignment='right',
            transform=ax.transAxes, color='red', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.5))

    plt.tight_layout()
    
    filename = os.path.join(output_dir, "parameter_delta_distribution.png")
    fig.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"  - Saved delta distribution plot to: {filename}")
# <<< 新增函数结束 >>>


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg') # Use non-interactive backend

    try:
        analyze_model_difference(MODEL_A_NAME, MODEL_B_NAME, device_id=TARGET_GPU_ID)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()