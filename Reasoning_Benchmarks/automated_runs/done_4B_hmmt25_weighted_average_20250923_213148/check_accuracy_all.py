import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
import scipy.stats as st
# Auto-generated on 2025-09-23 09:34:25 to check multiple models.

MODELS_TO_EVALUATE = [
    {"model_name": "Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_0I-100T", "results_path": "outputs/Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_0I-100T", "tokenizer_path": "../../../../../models/Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_0I-100T"},
    {"model_name": "Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_10I-90T", "results_path": "outputs/Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_10I-90T", "tokenizer_path": "../../../../../models/Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_10I-90T"},
    {"model_name": "Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_20I-80T", "results_path": "outputs/Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_20I-80T", "tokenizer_path": "../../../../../models/Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_20I-80T"},
    {"model_name": "Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_30I-70T", "results_path": "outputs/Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_30I-70T", "tokenizer_path": "../../../../../models/Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_30I-70T"},
    {"model_name": "Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_40I-60T", "results_path": "outputs/Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_40I-60T", "tokenizer_path": "../../../../../models/Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_40I-60T"},
    {"model_name": "Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_50I-50T", "results_path": "outputs/Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_50I-50T", "tokenizer_path": "../../../../../models/Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_50I-50T"},
    {"model_name": "Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_60I-40T", "results_path": "outputs/Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_60I-40T", "tokenizer_path": "../../../../../models/Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_60I-40T"},
    {"model_name": "Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_70I-30T", "results_path": "outputs/Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_70I-30T", "tokenizer_path": "../../../../../models/Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_70I-30T"},
    {"model_name": "Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_80I-20T", "results_path": "outputs/Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_80I-20T", "tokenizer_path": "../../../../../models/Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_80I-20T"},
    {"model_name": "Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_90I-10T", "results_path": "outputs/Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_90I-10T", "tokenizer_path": "../../../../../models/Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_90I-10T"},
    {"model_name": "Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_100I-0T", "results_path": "outputs/Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_100I-0T", "tokenizer_path": "../../../../../models/Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_100I-0T"}
]
DEFAULT_BENCHMARKS = [
    "hmmt25"
]

def process_file(file_path, tokenizer):
    correct_counts = [0] * 10
    total_counts = [0] * 10
    token_sums = [0] * 10
    response_counts = [0] * 10

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                scores = data.get('score', [])
                codes = data.get('code', [])

                if not isinstance(scores, list) or len(scores) != 10: continue
                if not isinstance(codes, list) or len(codes) != 10: continue

                for i in range(10):
                    s = scores[i]
                    if isinstance(s, bool) or isinstance(s, int):
                        correct_counts[i] += bool(s)
                        total_counts[i] += 1

                    code = codes[i]
                    if isinstance(code, str):
                        token_sums[i] += len(tokenizer.encode(code))
                        response_counts[i] += 1

        sample_accuracies = [(c / t) * 100 if t > 0 else 0 for c, t in zip(correct_counts, total_counts)]
        total_tokens_list = [ts / rc if rc > 0 else 0 for ts, rc in zip(token_sums, response_counts)]

        return sample_accuracies, total_tokens_list

    except FileNotFoundError:
        return None, None

# --- Plotting Function (No Change) ---
def plot_results(results):
    sns.set_theme(style="whitegrid", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(14, 9))

    # --- 从所有结果中提取数据 ---
    mean_tokens = [r['mean_tokens'] for r in results]
    mean_acc = [r['mean_acc'] for r in results]
    
    # 计算误差棒的长度
    y_err = [[r['mean_acc'] - r['ci_90_acc'][0] for r in results], 
             [r['ci_90_acc'][1] - r['mean_acc'] for r in results]]
    
    x_err = [[r['mean_tokens'] - r['ci_90_tokens'][0] for r in results],
             [r['ci_90_tokens'][1] - r['mean_tokens'] for r in results]]

    # --- 绘制所有数据点(样式统一) ---
    ax.errorbar(
        x=mean_tokens, 
        y=mean_acc, 
        xerr=x_err, 
        yerr=y_err,
        fmt='o',             # 'o' 代表圆形标记
        color='royalblue',   # 统一颜色
        capsize=5,
        markersize=8,
        alpha=0.8,
        linestyle='none'
    )

    # --- 为每个数据点添加文本标签 ---
    for r in results:
        try:
            label = "slerp=" + r['model_name'].split('slerp')[-1]
        except:
            label = r['model_name']
        
        ax.annotate(
            label,
            (r['mean_tokens'], r['mean_acc']),
            textcoords="offset points",
            xytext=(10, -10), # 文本偏移量，防止遮挡数据点
            ha='left',        # 水平对齐方式
            fontsize=12
        )

    # --- 美化图表 ---
    ax.set_title('Model Performance: Accuracy vs. Token Consumption', fontsize=20, pad=20)
    ax.set_xlabel('Average Token Consumption per Response', fontsize=14, labelpad=15)
    ax.set_ylabel('Accuracy (%)', fontsize=14, labelpad=15)
    
    # 调整Y轴，显示百分号
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}%'))
    
    plt.tight_layout() # 自动调整布局
    
    # 保存图表
    save_path = 'model_performance_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅Plot successfully saved to: {save_path}")
    
    # 显示图表
    plt.show()


# --- MODIFIED: Main Execution Logic ---
def main():
    all_results = []

    for config in MODELS_TO_EVALUATE:
        # --- ADDED: Print the name of the results directory ---
        directory_name = os.path.basename(config['results_path'])
        print(f"\n--- Processing results from: {directory_name} ---")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_path'], trust_remote_code=True)
        except Exception as e:
            print(f"      ❌ ERROR: Failed to load tokenizer. {e}. Skipping.")
            continue

        for benchmark in DEFAULT_BENCHMARKS:
            bench_path = os.path.join(config['results_path'], benchmark)
            print(f"      ▶️ Benchmark: {benchmark}")
            if not os.path.isdir(bench_path):
                print(f"              ❗️ SKIPPED: Directory not found: {bench_path}")
                continue
            
            jsonl_files = [f for f in os.listdir(bench_path) if f.endswith('.jsonl')]
            if not jsonl_files:
                print("              ❗️ SKIPPED: No .jsonl result files found.")
                continue

            for fname in sorted(jsonl_files):
                fpath = os.path.join(bench_path, fname)
                sample_accuracies, sample_tokens = process_file(fpath, tokenizer)
                
                if sample_accuracies is None:
                    print(f"              ❗️ SKIPPED: File not found or unreadable: {fname}")
                    continue
                
                n = len(sample_accuracies)
                print(f"              📊 Summary for {n} samples from '{fname}':")
                
                if n > 1:
                    mean_acc = np.mean(sample_accuracies)
                    std_acc = np.std(sample_accuracies, ddof=1)
                    sem_acc = st.sem(sample_accuracies)
                    ci_95_acc = st.t.interval(0.95, n - 1, loc=mean_acc, scale=sem_acc)
                    ci_90_acc = st.t.interval(0.90, n - 1, loc=mean_acc, scale=sem_acc)
                    
                    print("                   --- Accuracy ---")
                    print(f"                   Mean:               {mean_acc:>6.2f}%")
                    print(f"                   Std Deviation:      {std_acc:>6.2f}")
                    print(f"                   95% Confidence Int.: ({ci_95_acc[0]:>5.2f}%, {ci_95_acc[1]:>5.2f}%)")
                    print(f"                   90% Confidence Int.: ({ci_90_acc[0]:>5.2f}%, {ci_90_acc[1]:>5.2f}%)")

                    mean_tokens, std_tokens, ci_95_tokens, ci_90_tokens = (0, 0, (0,0), (0,0))
                    if sample_tokens and len(sample_tokens) == n:
                        mean_tokens = np.mean(sample_tokens)
                        std_tokens = np.std(sample_tokens, ddof=1)
                        sem_tokens = st.sem(sample_tokens)
                        ci_95_tokens = st.t.interval(0.95, n - 1, loc=mean_tokens, scale=sem_tokens)
                        ci_90_tokens = st.t.interval(0.90, n - 1, loc=mean_tokens, scale=sem_tokens)
                        
                        print("                   --- Token Length ---")
                        print(f"                   Mean:               {mean_tokens:>6.0f}")
                        print(f"                   Std Deviation:      {std_tokens:>6.0f}")
                        print(f"                   95% Confidence Int.: ({ci_95_tokens[0]:>5.0f}, {ci_95_tokens[1]:>5.0f})")
                        print(f"                   90% Confidence Int.: ({ci_90_tokens[0]:>5.0f}, {ci_90_tokens[1]:>5.0f})")

                        all_results.append({
                            "model_name": config['model_name'],
                            "benchmark": benchmark,
                            "file": fname,
                            "mean_acc": mean_acc,
                            "mean_tokens": mean_tokens,
                            "ci_90_acc": ci_90_acc,
                            "ci_90_tokens": ci_90_tokens
                        })

    if all_results:
        plot_results(all_results)
    else:
        print("No data was collected for plotting.")

if __name__ == "__main__":
    main()