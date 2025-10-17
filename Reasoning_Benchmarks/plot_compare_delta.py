import torch
import os
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---

# 定义要比较的第一组模型
MODEL_PAIR_1 = (
    "mlabonne/NeuralBeagle14-7B", 
    "udkai/Turdus"
)

# 定义要比较的第二组模型
MODEL_PAIR_2 = (
    "Qwen/Qwen3-4B-Instruct",
    "Qwen/Qwen3-4B-Thinking"
)
# MODEL_PAIR_2 = (
#     "Qwen/Qwen3-4B-Instruct",
#     "Qwen/Qwen3-4B-Thinking"
# )



# 定义结果图的输出目录
OUTPUT_DIR = "diff_analysis_results"


def get_pair_name(model_pair: tuple) -> str:
    """根据模型名称元组生成用于文件夹的名称"""
    name_a = model_pair[0].split('/')[-1]
    name_b = model_pair[1].split('/')[-1]
    return f"{name_a}_vs_{name_b}"
    
def get_pair_label(model_pair: tuple) -> str:
    """生成更美观的、用于图例的标签"""
    name_a = model_pair[0].split('/')[-1]
    name_b = model_pair[1].split('/')[-1]
    # 使用箭头表示从基础模型到微调模型的转变
    return f"{name_a} → {name_b}"


def load_delta_distribution_from_cache(model_pair: tuple) -> tuple:
    """从缓存文件中加载指定的模型对的差值分布数据"""
    pair_name = get_pair_name(model_pair)
    cache_dir = os.path.join(OUTPUT_DIR, pair_name)
    
    cache_path = os.path.join(cache_dir, "analysis_cache_full.pt")
    
    print(f"Attempting to load data for '{pair_name}' from: {cache_path}")

    if not os.path.exists(cache_path):
        lightweight_cache_path = os.path.join(cache_dir, "analysis_cache_lightweight.pt")
        if not os.path.exists(lightweight_cache_path):
            raise FileNotFoundError(
                f"Cache file not found for model pair '{pair_name}'.\n"
                f"Checked paths:\n- {cache_path}\n- {lightweight_cache_path}\n"
                "Please run the main analysis script for this model pair first."
            )
        cache_path = lightweight_cache_path

    cache_data = torch.load(cache_path, map_location='cpu', weights_only=False)

    required_keys = ['hist_delta_counts_cpu', 'bins_delta_cpu']
    if not all(key in cache_data for key in required_keys):
        raise KeyError(
            f"Cache file '{cache_path}' is missing required data. "
            "It might be from an older version of the analysis script."
        )
    
    print(f"Successfully loaded data for '{pair_name}'.")
    return cache_data['hist_delta_counts_cpu'], cache_data['bins_delta_cpu']


def plot_comparison(pair1_tuple: tuple, pair2_tuple: tuple):
    """
    绘制两个模型对的参数差值分布对比图 (Final Style)
    """
    try:
        hist1, bins1 = load_delta_distribution_from_cache(pair1_tuple)
        hist2, bins2 = load_delta_distribution_from_cache(pair2_tuple)
    except (FileNotFoundError, KeyError) as e:
        print(f"\nError: {e}")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(13, 9))

    # --- 关键修改：让背景网格线变淡 ---
    # alpha=0.5 使虚线透明度为50%，看起来更柔和
    ax.grid(True, linestyle='--', alpha=0.3)

    # --- 绘制数据 ---
    ax.bar(
        (bins1[:-1] + bins1[1:]) / 2, hist1, 
        width=(bins1[1] - bins1[0]).item(), 
        color='dodgerblue', 
        alpha=0.6, 
        label=get_pair_label(pair1_tuple)
    )

    ax.bar(
        (bins2[:-1] + bins2[1:]) / 2, hist2, 
        width=(bins2[1] - bins2[0]).item(), 
        color='orangered', 
        alpha=0.6, 
        label=get_pair_label(pair2_tuple)
    )

    # --- 美化和标注 (Final Style) ---
    ax.set_yscale('log')
    
    ax.axvline(x=0.002, color='gray', linestyle='--', linewidth=2, alpha=0.9)
    ax.axvline(x=-0.002, color='gray', linestyle='--', linewidth=2, alpha=0.9)

    ax.set_xlabel("Parameter Difference", fontsize=48)
    ax.set_ylabel("Frequency", fontsize=48)

    # 1. 移除强制的X轴刻度标注
    # ax.set_xticks(...) # This line is removed

    # 2. 设置Y轴范围，抬高顶部，为图例腾出空间
    ax.set_ylim(bottom=1, top=1e12) # Y轴底部设为1，顶部设为10^11
    
    ax.set_xlim(-0.04, 0.04)
    
    ax.tick_params(axis='both', which='major', labelsize=35)

    # 3. 设置图例为不透明白底，确保能盖住虚线
    legend = ax.legend(fontsize=22, frameon=True, facecolor='white',edgecolor='white',framealpha=0.8, loc='best')
    plt.tight_layout()

    # --- 保存图像 ---
    pair1_name = get_pair_name(pair1_tuple)
    pair2_name = get_pair_name(pair2_tuple)
    output_filename = f"comparison_{pair1_name}_vs_{pair2_name}.pdf"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFinal comparison plot saved successfully to: {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    plot_comparison(MODEL_PAIR_1, MODEL_PAIR_2)