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
TARGET_GPU_ID = 3

# 2. DEFINE MODELS TO ANALYZE
MODEL_A_NAME = "Qwen/Qwen3-4B-Instruct-2507" 
MODEL_B_NAME = "Qwen/Qwen3-4B-Thinking-2507"

# MODEL_A_NAME = "Qwen/Qwen3-8B"
# MODEL_B_NAME = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"

# --- Main Analysis Function ---
def analyze_model_difference(model_name_a, model_name_b, device_id):
    """
    Performs a deeply detailed and highly efficient Pillar 2 analysis.
    v5.2: Fixes logical inconsistency. Both cumulative plots now use the
          same fast and correct histogram-based method.
    """
    print("--- Starting Advanced Pillar 2 Analysis (v5.2) ---")
    
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
        model_name_a, dtype=torch.bfloat16, device_map=target_device, trust_remote_code=True
    )
    model_a.eval()

    print(f"Loading model B ({model_name_b}) onto {target_device}")
    model_b = AutoModelForCausalLM.from_pretrained(
        model_name_b, dtype=torch.bfloat16, device_map=target_device, trust_remote_code=True
    )
    model_b.eval()

    config = model_a.config
    num_heads = config.num_attention_heads
    params_a = model_a.state_dict()
    params_b = model_b.state_dict()
    
    # 3. Calculate Differences and Collect Granular & Global Data
    print("\nCalculating parameter differences and collecting data...")
    sub_module_l1, sub_module_l2, sub_module_counts = {}, {}, {}
    attn_head_l1, attn_head_l2, attn_head_counts = {}, {}, {}
    delta_magnitudes_gpu = [] 
    
    global_stats = {
        "model_a_l1": 0.0, "model_a_l2_sq": 0.0,
        "model_b_l1": 0.0, "model_b_l2_sq": 0.0,
        "delta_l1": 0.0, "delta_l2_sq": 0.0
    }
    
    layer_re = re.compile(r"model\.layers\.(\d+)\.")
    param_iterator = tqdm(params_a.keys(), desc="Processing parameters")

    for param_name in param_iterator:
        if param_name not in params_b: continue
        
        tensor_a, tensor_b = params_a[param_name], params_b[param_name]
        delta = (tensor_b - tensor_a).float()
        
        delta_magnitudes_gpu.append(delta.abs())

        global_stats["model_a_l1"] += tensor_a.float().abs().sum().item()
        global_stats["model_a_l2_sq"] += tensor_a.float().pow(2).sum().item()
        global_stats["model_b_l1"] += tensor_b.float().abs().sum().item()
        global_stats["model_b_l2_sq"] += tensor_b.float().pow(2).sum().item()
        global_stats["delta_l1"] += delta.abs().sum().item()
        global_stats["delta_l2_sq"] += delta.pow(2).sum().item()

        match = layer_re.search(param_name)
        if match:
            layer_idx = int(match.group(1))
            
            if layer_idx not in sub_module_l1: sub_module_l1[layer_idx] = {}
            if layer_idx not in sub_module_l2: sub_module_l2[layer_idx] = {}
            if layer_idx not in sub_module_counts: sub_module_counts[layer_idx] = {}
            if layer_idx not in attn_head_l1: attn_head_l1[layer_idx] = np.zeros(num_heads)
            if layer_idx not in attn_head_l2: attn_head_l2[layer_idx] = np.zeros(num_heads)
            if layer_idx not in attn_head_counts: attn_head_counts[layer_idx] = np.zeros(num_heads)

            l2_energy_tensor = delta.pow(2)
            l1_norm_val = delta.abs().sum().item()
            param_count = delta.numel()

            if "self_attn" in param_name:
                head_param_count = param_count / num_heads
                if any(p in param_name for p in ["q_proj", "k_proj", "v_proj"]):
                    delta_reshaped = delta.view(num_heads, -1, delta.shape[-1])
                    attn_head_l2[layer_idx] += delta_reshaped.pow(2).sum(dim=(1,2)).cpu().numpy()
                    attn_head_l1[layer_idx] += delta_reshaped.abs().sum(dim=(1,2)).cpu().numpy()
                    attn_head_counts[layer_idx] += head_param_count
                elif "o_proj" in param_name:
                    delta_reshaped = delta.view(delta.shape[0], num_heads, -1)
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
            
            sub_module_l2[layer_idx][sub_module] = sub_module_l2[layer_idx].get(sub_module, 0) + l2_energy_tensor.sum().item()
            sub_module_l1[layer_idx][sub_module] = sub_module_l1[layer_idx].get(sub_module, 0) + l1_norm_val
            sub_module_counts[layer_idx][sub_module] = sub_module_counts[layer_idx].get(sub_module, 0) + param_count
    
    del model_a, model_b, params_a, params_b
    torch.cuda.empty_cache()

    # 4. Save Global Stats and Generate Plots
    save_global_stats(global_stats, model_pair_name, output_dir)
    print("\nGenerating detailed plots (v5.2)...")
    
    l2_dir = os.path.join(output_dir, "l2_norm_avg_per_param")
    os.makedirs(l2_dir, exist_ok=True)
    plot_attention_head_spotlight(attn_head_l2, attn_head_counts, model_pair_name, l2_dir, config, "Avg L2 Norm (Energy)")
    generate_combined_trend_plot(sub_module_l2, sub_module_counts, model_pair_name, l2_dir, config, "Avg L2 Norm (Energy)")
    
    l1_dir = os.path.join(output_dir, "l1_norm_avg_per_param")
    os.makedirs(l1_dir, exist_ok=True)
    plot_attention_head_spotlight(attn_head_l1, attn_head_counts, model_pair_name, l1_dir, config, "Avg L1 Norm (Abs Change)")
    generate_combined_trend_plot(sub_module_l1, sub_module_counts, model_pair_name, l1_dir, config, "Avg L1 Norm (Abs Change)")
    
    plot_norm_distributions(attn_head_l1, attn_head_l2, attn_head_counts, model_pair_name, output_dir, config)
    plot_cumulative_l2_energy_final(delta_magnitudes_gpu, model_pair_name, output_dir)
    plot_cumulative_l1_norm_final(delta_magnitudes_gpu, model_pair_name, output_dir) # This will now be fast
    
    del delta_magnitudes_gpu
    torch.cuda.empty_cache()

    print("\n--- Advanced Pillar 2 Analysis (v5.2) Complete! ---")

# --- Plotting and Helper Functions ---

def save_global_stats(stats, model_pair_name, output_dir):
    # ... (This function is correct, no changes needed)
    summary = {
        "model_A_instruct": {"L1_Norm": stats["model_a_l1"], "L2_Norm_Energy": np.sqrt(stats["model_a_l2_sq"])},
        "model_B_thinking": {"L1_Norm": stats["model_b_l1"], "L2_Norm_Energy": np.sqrt(stats["model_b_l2_sq"])},
        "delta_diff": {"L1_Norm": stats["delta_l1"], "L2_Norm_Energy": np.sqrt(stats["delta_l2_sq"])},
        "relative_change_vs_instruct_model": {
            "relative_L1_change": f"{stats['delta_l1'] / stats['model_a_l1']:.4%}",
            "relative_L2_change": f"{np.sqrt(stats['delta_l2_sq']) / np.sqrt(stats['model_a_l2_sq']):.4%}",
        }
    }
    filepath = os.path.join(output_dir, "global_norm_summary.json")
    with open(filepath, 'w') as f: json.dump(summary, f, indent=4)
    print(f"\nSaved global norm statistics to: {filepath}")
    print(json.dumps(summary["relative_change_vs_instruct_model"], indent=4))

def plot_attention_head_spotlight(head_norms, head_counts, model_pair_name, output_dir, config, norm_type):
    # ... (This function is correct, no changes needed)
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
    # ... (This function is correct, no changes needed)
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

def plot_norm_distributions(attn_head_l1, attn_head_l2, attn_head_counts, model_pair_name, output_dir, config):
    # ... (This function is correct, no changes needed)
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
    ax1.set_title(f"Distribution of Avg L1 Norm per Parameter\nAcross All {num_layers*num_heads} Attention Heads", fontsize=14)
    ax1.set_xlabel("Avg L1 Norm per Parameter", fontsize=12)
    ax1.set_ylabel("Frequency (Count of Heads)", fontsize=12)
    ax1.grid(alpha=0.6)
    ax2.hist(l2_norms_flat, bins=50, color='salmon', edgecolor='black')
    ax2.set_title(f"Distribution of Avg L2 Norm per Parameter\nAcross All {num_layers*num_heads} Attention Heads", fontsize=14)
    ax2.set_xlabel("Avg L2 Norm per Parameter", fontsize=12)
    ax2.grid(alpha=0.6)
    fig.suptitle(f"Attention Head Change Distribution (Normalized by Parameter Count)\n({model_pair_name})", fontsize=18, weight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join(output_dir, "distribution_of_head_norms_avg.png"), dpi=300)
    plt.close(fig)

def plot_cumulative_l1_norm_final(magnitudes_gpu, model_pair_name, output_dir):
    """
    MODIFIED in v5.2: Now uses the same fast and correct histogram method as the L2 plot.
    """
    print("  - Generating Cumulative L1 Norm Curve...")
    with torch.no_grad():
        print("    - Step 1: Concatenating magnitudes on GPU...")
        all_magnitudes_flat_gpu = torch.cat([t.flatten() for t in magnitudes_gpu])
        
        print("    - Step 2: Moving magnitude tensor to CPU RAM...")
        all_magnitudes_flat_cpu = all_magnitudes_flat_gpu.cpu()
        del all_magnitudes_flat_gpu
        torch.cuda.empty_cache()

        print("    - Step 3: Building correct log-spaced histogram on CPU...")
        max_val = all_magnitudes_flat_cpu.max().item()
        bins = torch.logspace(start=np.log10(1e-9), end=np.log10(max_val + 1e-9), steps=5000)
        
        counts, bin_edges = torch.histogram(all_magnitudes_flat_cpu, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        print("    - Step 4: Calculating cumulative L1 norm from histogram...")
        total_params = all_magnitudes_flat_cpu.numel()
        counts_desc = torch.flip(counts, dims=[0])
        bin_centers_desc = torch.flip(bin_centers, dims=[0])

        cumulative_counts = torch.cumsum(counts_desc, dim=0)
        cumulative_l1 = torch.cumsum(counts_desc * bin_centers_desc, dim=0)

        approximated_total_l1 = cumulative_l1[-1]
        y_axis_percent = (cumulative_l1 / (approximated_total_l1 + 1e-9)).numpy()
        x_axis_percent = (cumulative_counts / total_params * 100).numpy()
        
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(x_axis_percent, y_axis_percent, color='dodgerblue', lw=2.5)
    ax.set_title(f"Cumulative L1 Norm of Sorted Parameter Differences\n({model_pair_name})", fontsize=16)
    ax.set_xlabel("Percentage of Parameters (Sorted by |$\Delta_{{diff}}$| magnitude)", fontsize=12)
    ax.set_ylabel("Cumulative Percentage of Total L1 Norm ($\sum |\Delta_{{diff}}|$)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    for percentile in [1, 5, 10, 20]:
        idx = np.searchsorted(x_axis_percent, percentile)
        if idx < len(y_axis_percent):
            y_val = y_axis_percent[idx]
            ax.axhline(y=y_val, color='grey', linestyle=':', xmax=x_axis_percent[idx]/100.0)
            ax.axvline(x=x_axis_percent[idx], color='grey', linestyle=':', ymax=y_val)
            ax.text(x_axis_percent[idx] + 1, y_val - 0.05, f"Top {percentile:.0f}% params\n~{y_val*100:.1f}% of L1 norm", fontsize=10)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "cumulative_l1_norm_curve.png"), dpi=300)
    plt.close(fig)
    print("  - Saved cumulative L1 norm curve plot.")

def plot_cumulative_l2_energy_final(magnitudes_gpu, model_pair_name, output_dir):
    # ... (This function is correct, no changes needed)
    print("  - Generating Cumulative L2 Energy Curve...")
    with torch.no_grad():
        print("    - Step 1: Concatenating energies on GPU...")
        all_energies_flat_gpu = torch.cat([t.pow(2).flatten() for t in magnitudes_gpu])
        print("    - Step 2: Moving energy tensor to CPU RAM...")
        all_energies_flat_cpu = all_energies_flat_gpu.cpu()
        del all_energies_flat_gpu
        torch.cuda.empty_cache()
        print("    - Step 3: Building correct log-spaced histogram on CPU...")
        max_energy = all_energies_flat_cpu.max().item()
        bins = torch.logspace(start=np.log10(1e-18), end=np.log10(max_energy + 1e-18), steps=5000)
        counts, bin_edges = torch.histogram(all_energies_flat_cpu, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        print("    - Step 4: Calculating cumulative energy from histogram...")
        total_params = all_energies_flat_cpu.numel()
        counts_desc = torch.flip(counts, dims=[0])
        bin_centers_desc = torch.flip(bin_centers, dims=[0])
        cumulative_counts = torch.cumsum(counts_desc, dim=0)
        cumulative_energy = torch.cumsum(counts_desc * bin_centers_desc, dim=0)
        approximated_total_energy = cumulative_energy[-1]
        y_axis_percent = (cumulative_energy / (approximated_total_energy + 1e-9)).numpy()
        x_axis_percent = (cumulative_counts / total_params * 100).numpy()
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(x_axis_percent, y_axis_percent, color='darkviolet', lw=2.5)
    ax.set_title(f"Cumulative L2 Energy of Sorted Parameter Differences\n({model_pair_name})", fontsize=16)
    ax.set_xlabel("Percentage of Parameters (Sorted by |$\Delta_{{diff}}$| magnitude)", fontsize=12)
    ax.set_ylabel("Cumulative Percentage of Total L2 Energy ($\sum \Delta_{{diff}}^2$)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    for percentile in [1, 5, 10, 20]:
        idx = np.searchsorted(x_axis_percent, percentile)
        if idx < len(y_axis_percent):
            y_val = y_axis_percent[idx]
            ax.axhline(y=y_val, color='grey', linestyle=':', xmax=x_axis_percent[idx]/100.0)
            ax.axvline(x=x_axis_percent[idx], color='grey', linestyle=':', ymax=y_val)
            ax.text(x_axis_percent[idx] + 1, y_val - 0.05, f"Top {percentile:.0f}% params\n~{y_val*100:.1f}% energy", fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "cumulative_l2_energy_curve.png"), dpi=300)
    plt.close(fig)
    print("  - Saved cumulative L2 energy curve plot.")

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    
    try:
        analyze_model_difference(MODEL_A_NAME, MODEL_B_NAME, device_id=TARGET_GPU_ID)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()