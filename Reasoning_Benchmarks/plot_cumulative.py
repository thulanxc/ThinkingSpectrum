import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import FuncFormatter # <<< 1. IMPORT FuncFormatter
import os

# --- Configuration ---
ANALYSIS_RESULTS_DIR = "diff_analysis_results/Qwen3-4B-Instruct_vs_Qwen3-4B-Thinking"
GAUSSIAN_SAMPLES = 20_000_000

# --- Main Plotting Function ---

def plot_comparison_l2_energy_curve(analysis_dir, num_gaussian_samples):
    """
    Loads L2 energy data from a model analysis cache and compares it to the
    L2 energy curve of the difference between two standard normal distributions.
    Generates a high-quality PDF plot with custom styling.
    """
    print("--- Starting L2 Energy Curve Comparison Plot (PDF Version) ---")

    # === Part 1: Process Model Data from Cache ===
    # (This part is unchanged)
    print(f"Loading model data from: {analysis_dir}")
    cache_path_light = os.path.join(analysis_dir, "analysis_cache_lightweight.pt")
    cache_path_full = os.path.join(analysis_dir, "analysis_cache_full.pt")
    if os.path.exists(cache_path_light):
        cache_path = cache_path_light
    elif os.path.exists(cache_path_full):
        cache_path = cache_path_full
    else:
        raise FileNotFoundError(
            f"Error: No cache file found in '{analysis_dir}'. "
            "Please run the main analysis script first."
        )
    print(f"Reading cache file: {cache_path}")
    cache_data = torch.load(cache_path, map_location=torch.device('cpu'))
    model_hist_counts = cache_data['l2_hist_counts_cpu']
    model_hist_bins = cache_data['l2_bins_cpu']
    model_total_params = cache_data['total_params']
    model_pair_name = os.path.basename(analysis_dir)
    model_x, model_y = calculate_cumulative_curve_from_hist(
        model_hist_counts, model_hist_bins, model_total_params
    )
    print("Successfully processed model data.")

    # === Part 2: Generate Gaussian Distribution Data ===
    # (This part is unchanged)
    print(f"\nGenerating {num_gaussian_samples:,} samples for N(0, 1) distributions...")
    samples_a = np.random.normal(loc=0.0, scale=1.0, size=num_gaussian_samples)
    samples_b = np.random.normal(loc=0.0, scale=1.0, size=num_gaussian_samples)
    delta = samples_b - samples_a
    energies = np.square(delta)
    num_hist_bins = 10000
    hist_bins = torch.logspace(start=-26, end=4, steps=num_hist_bins)
    hist_counts, _ = np.histogram(energies, bins=hist_bins.numpy())
    gaussian_x, gaussian_y = calculate_cumulative_curve_from_hist(
        torch.from_numpy(hist_counts), hist_bins, num_gaussian_samples
    )
    print("Successfully processed Gaussian data.")

    # === Part 3: Create the Comparison Plot ===
    print("\nGenerating the comparison plot...")
    
    # <<< 2. DEFINE THE CUSTOM FORMATTER FUNCTION >>>
    # This function formats the label. If the value is 0, it returns an empty string.
    def custom_y_formatter(y, pos):
        if y == 0:
            return ''
        else:
            return f'{int(y * 100)}%'

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(13, 9))

    model_legend_label = model_pair_name.replace("_vs_", " vs ")
    gaussian_legend_label = "Difference of two independent N(0,1) Distributions"
    
    ax.plot(model_x, model_y, color='darkviolet', lw=3.5, label=model_legend_label)
    ax.plot(gaussian_x, gaussian_y, color='deepskyblue', lw=3.5, linestyle='--', label=gaussian_legend_label)

    ax.set_xlabel("Top % Parameters (by $|\Delta|^2)$", fontsize=48,labelpad=15)
    ax.set_ylabel("$\sum \Delta^2$", fontsize=48, labelpad=15)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.05)
    
    # Keep the original formatter for the x-axis (shows "0%")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    
    # <<< 3. APPLY THE CUSTOM FORMATTER TO THE Y-AXIS >>>
    # This will hide the "0%" label on the y-axis only.
    ax.yaxis.set_major_formatter(FuncFormatter(custom_y_formatter))
    
    ax.tick_params(axis='both', which='major', labelsize=35)

    legend = ax.legend(fontsize=22, frameon=True, facecolor='white', framealpha=0.8, loc='best')
    legend.get_frame().set_edgecolor('white')

    plt.tight_layout()
    
    # --- Save the figure ---
    # (This part is unchanged)
    parent_dir = os.path.dirname(analysis_dir)
    if os.path.basename(parent_dir) != "diff_analysis_results":
        parent_dir = "diff_analysis_results"
        os.makedirs(parent_dir, exist_ok=True)
    
    output_filename = os.path.join(parent_dir, f"comparison_l2_energy_curve_{model_pair_name}.pdf")
    plt.savefig(output_filename, format='pdf', bbox_inches='tight')
    
    print(f"\nPlot saved successfully as PDF to: {output_filename}")


# --- Helper Function (Unchanged) ---
def calculate_cumulative_curve_from_hist(hist_counts, hist_bins, total_items):
    bin_centers = (hist_bins[:-1] + hist_bins[1:]) / 2.0
    counts_desc = torch.flip(hist_counts, dims=[0])
    bin_centers_desc = torch.flip(bin_centers, dims=[0])
    cumulative_counts = torch.cumsum(counts_desc, dim=0)
    cumulative_energy = torch.cumsum(counts_desc * bin_centers_desc, dim=0)
    total_approximated_energy = cumulative_energy[-1]
    if total_items == 0 or total_approximated_energy <= 1e-18:
        return np.array([0]), np.array([0])
    x_axis_percent = (cumulative_counts / total_items) * 100
    y_axis_percent = cumulative_energy / total_approximated_energy
    return x_axis_percent.numpy(), y_axis_percent.numpy()


if __name__ == '__main__':
    try:
        plot_comparison_l2_energy_curve(ANALYSIS_RESULTS_DIR, GAUSSIAN_SAMPLES)
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()