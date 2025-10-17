# 文件名: run_experiment.py
import os
import stat
from datetime import datetime
import model_merger  # 确保 model_merger.py 在同一个文件夹下

# 设置使用的GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# ======================================================================================
# --- ⚙️ 实验控制面板 (在这里修改) ---
# ======================================================================================
EXPERIMENT_CONTROL_PANEL = {
    # "ties_merge_ta": [(80,[0.2,0.8]),(80,[0.4,0.6]),(80,[0.6,0.4]),(80,[0.8,0.2])],#ok
    "lore_merge": [(2,[0.2,0.8]),(2,[0.4,0.6]),(2,[0.6,0.4]),(2,[0.8,0.2])],
    # "emr": [[0.2,0.8],[0.4,0.6],[0.6,0.4],[0.8,0.2]],
    # "emr": [[0.2,0.8]], 

    # "dare_merge_TA": [(0.2,0.2),(0.2, 0.4),(0.2,0.6),(0.2,0.8)],# ok

    # 'surgical_merge': [1,2,5,10,20,50],
    # "avg_override_top_k_thinking": [1,2,5,10,20,50],
    # --- DARE 合并方法的控制开关 ---
    # 参数格式: [(drop_rate, scaling_lambda), ...]
    # drop_rate: 随机丢弃率 p (论文推荐 0.9 或 0.99)
    # scaling_lambda: 任务算术的合并系数 λ (通常设为 0.5 到 1.0 之间)
    # "dare_merge": [(0.1,0.5),(0.2, 0.5),(0.3,0.5),(0.4,0.5),(0.8,0.5)],
    # --- TIES-Merging 合并方法的控制开关 ---
    # 参数格式: [(top_k_percentage, scaling_lambda), ...]
    # top_k_percentage: Trim 步骤保留的 top k% 参数 (论文中 k=20 效果不错)
    # scaling_lambda: 任务向量的缩放系数 λ (论文中 λ=1)
    # "ties_merge": [(95,1.0),(90,1.0),(80,1.0),(70,1.0),(50,1.0)], # 示例: 运行一次 TIES 合并
    # "weighted_average":[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
    # "slerp":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
    # "top_k_avg_keep_thinking": [],
    # "top_k_avg_keep_instruct": [1,2,5,10,20,50],
    # "avg_override_top_k_thinking": []
    # "weighted_average":[0,1],
}

# ======================================================================================
# --- 📂 全局路径与任务配置 (在这里修改) ---
# ======================================================================================

# --- 定义所有需要进行合并实验的模型对 ---
# !!! 重要: DARE 和 TIES 合并需要一个共同的 "base" 基础模型 !!!
MODEL_PAIRS_TO_MERGE = [
    {
    "base": "../../../models/Qwen3-30B-A3B",
    "instruct": "../../../models/Qwen3-30B-A3B-Instruct-2507",
    "thinking": "../../../models/Qwen3-30B-A3B-Thinking-2507"
    },
    # {
    # "base": "Qwen/Qwen3-4B",
    # "instruct": "../../../models/Qwen3-4B-Instruct-2507",
    # "thinking": "../../../models/Qwen3-4B-Thinking-2507"
    # },   
    # {
    # "base": "Qwen/Qwen2-Math-1.5B",
    # "instruct": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    # "thinking": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    # },
    # {
    # "base": "Qwen/Qwen2-Math-7B",
    # "instruct": "Qwen/Qwen2.5-Math-7B-Instruct",
    # "thinking": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    # }
]

# --- 全局路径配置 ---
# 合并后模型的顶级输出目录
BASE_MODEL_OUTPUT_DIR = "../../../models" 
# 存放每次运行生成的脚本的目录
AUTOMATED_RUNS_DIR = "automated_runs"
# 【新增】指定你的数据集文件夹相对于本脚本的位置
# 假设你的 data 文件夹和 run_experiment.py 在同一级目录
DATA_DIR_PATH_IN_PROJECT = "../../data" 

# --- 评估任务配置 ---
EVAL_TASKS = [
    "aime24 10"
    # "gpqa_diamond 1"
]

# ======================================================================================
# --- 📜 脚本模板 (路径已校准，无需修改) ---
# ======================================================================================
RUN_EVAL_SH_TEMPLATE = """#!/bin/bash
set -ex
export CUDA_VISIBLE_DEVICES="0,1,2,3"
PROMPT_TYPE="qwen25-no-cot"

MODELS=(
{model_paths_string}
)

TASKS=(
{tasks_string}
)

run_evaluation() {{
    local MODEL_NAME_OR_PATH=$1
    local DATA_NAME=$2
    local N_SAMPLING=$3
    local OUTPUT_DIR

    echo "================================================================"
    echo "STARTING EVALUATION FOR: ${{MODEL_NAME_OR_PATH}}"
    echo "ON DATA: ${{DATA_NAME}} with N_SAMPLING: ${{N_SAMPLING}}"
    echo "================================================================"
    
    CLEAN_MODEL_NAME=$(basename "${{MODEL_NAME_OR_PATH}}")
    OUTPUT_DIR="${{CLEAN_MODEL_NAME}}"

    # 【重要修改】显式传入数据集的路径，确保脚本能找到数据
    TOKENIZERS_PARALLELISM=false \\
    python3 -u ../../math_eval.py \\
        --model_name_or_path "${{MODEL_NAME_OR_PATH}}" \\
        --data_name "${{DATA_NAME}}" \\
        --data_dir "{data_dir_path}" \\
        --output_dir "${{OUTPUT_DIR}}" \\
        --split "test" \\
        --prompt_type "${{PROMPT_TYPE}}" \\
        --num_test_sample -1 \\
        --seed 0 \\
        --temperature 0.6 \\
        --n_sampling "${{N_SAMPLING}}" \\
        --top_p 0.95 \\
        --top_k 20 \\
        --start 0 \\
        --end -1 \\
        --use_vllm \\
        --save_outputs \\
        --overwrite \\
        --max_tokens_per_call 39512 \\
        --presence_penalty 0 \\
        --frequency_penalty 0
}}

for model in "${{MODELS[@]}}"; do
    for task in "${{TASKS[@]}}"; do
        read -r data_names n_sampling <<< "$task"
        run_evaluation "$model" "$data_names" "$n_sampling"
    done
done

echo "All evaluation tasks are complete."
"""

CHECK_ACCURACY_PY_TEMPLATE = """import os, json
from transformers import AutoTokenizer
# Auto-generated on {generation_time} to check multiple models.

MODELS_TO_EVALUATE = [
{model_configs_string}
]
DEFAULT_BENCHMARKS = [
{benchmarks_string}
]

def process_file(file_path, tokenizer):
    correct, total_preds, total_tokens, total_responses = 0, 0, 0, 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                if 'score' in data and isinstance(data['score'], list):
                    correct += sum(bool(s) for s in data['score'])
                    total_preds += len(data['score'])
                if 'code' in data and isinstance(data['code'], list):
                    for resp in data['code']:
                        if isinstance(resp, str): total_tokens += len(tokenizer.encode(resp)); total_responses += 1
    except FileNotFoundError: return None
    acc = (correct / total_preds) * 100 if total_preds > 0 else 0
    tokens = total_tokens / total_responses if total_responses > 0 else 0
    return acc, correct, total_preds, tokens

def main():
    print("="*60 + "\\nStarting Accuracy Evaluation\\n" + "="*60)
    for config in MODELS_TO_EVALUATE:
        print("\\n--- Model: '{{}}' ---".format(config['model_name']))
        try:
            tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_path'], trust_remote_code=True)
        except Exception as e:
            print("     ❌ ERROR: Failed to load tokenizer from {{}}. Details: {{}}. Skipping.".format(config['tokenizer_path'], e))
            continue
        for benchmark in DEFAULT_BENCHMARKS:
            bench_path = os.path.join(config['results_path'], benchmark)
            print("     ▶️ Benchmark: {{}}".format(benchmark))
            if not os.path.isdir(bench_path):
                print("         ❗️ SKIPPED: Directory not found: {{}}".format(bench_path))
                continue
            
            jsonl_files = [f for f in os.listdir(bench_path) if f.endswith('.jsonl')]
            if not jsonl_files:
                print("         ❗️ SKIPPED: No .jsonl result files found in directory.")
                continue

            for fname in sorted(jsonl_files):
                fpath = os.path.join(bench_path, fname)
                res = process_file(fpath, tokenizer)
                if res:
                    accuracy, correct, total, avg_tokens = res
                    print("         ✔️ {{:<40}} | Accuracy: {{:>6.2f}}% ({{:>3}}/{{:<3}}) | Avg Length: {{:.0f}} tokens".format(
                        fname, accuracy, correct, total, avg_tokens
                    ))
if __name__ == "__main__": main()
"""

# ======================================================================================
# --- 🤖 主执行逻辑 (路径已校准，无需修改) ---
# ======================================================================================
def main():
    active_method, params_to_run = None, None
    for method, params in EXPERIMENT_CONTROL_PANEL.items():
        if params:
            if active_method is not None:
                print("❌ Error: Multiple experiments are activated in the control panel. Please ensure that parameters are provided for only one method at a time.")
                return
            active_method = method
            params_to_run = params

    if not active_method:
        print("🤷 Info: No experiment to run was found in the control panel. Exiting script.")
        return

    print(f"🚀 Preparing to execute experiment: Method='{active_method}', Parameters={params_to_run}")

    # 【重要修改】文件夹命名规则: 方法名_日期_时间
    run_folder_name = f"{active_method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    current_run_dir = os.path.join(AUTOMATED_RUNS_DIR, run_folder_name)
    os.makedirs(current_run_dir, exist_ok=True)
    print(f"Created main directory for this run: {current_run_dir}")

    generated_models_info = []

    for model_pair in MODEL_PAIRS_TO_MERGE:
        instruct_model_path = model_pair["instruct"]
        thinking_model_path = model_pair["thinking"]
        base_model_path = model_pair.get("base")

        if (active_method == 'dare_merge' or active_method == 'ties_merge') and not base_model_path:
            print(f"❌ Error: {active_method.upper()} merge requires a 'base' model. Skipping model pair: {os.path.basename(instruct_model_path)} and {os.path.basename(thinking_model_path)}")
            continue

        print("\n" + "#"*80)
        print(f"## Processing model pair: Instruct='{os.path.basename(instruct_model_path)}', Thinking='{os.path.basename(thinking_model_path)}'")
        if base_model_path:
            print(f"## Base model: Base='{os.path.basename(base_model_path)}'")
        print("#"*80)

        instruct_name = os.path.basename(instruct_model_path)
        thinking_name = os.path.basename(thinking_model_path)

    for param_value in params_to_run:
        print("\n" + "="*80 + f"\n▶️ Processing parameters: {param_value}\n" + "="*80)
        model_output_path, merge_func = "", None
        
        if active_method == 'dare_merge':
            drop_rate, scaling_lambda = param_value
            output_name = f"{instruct_name}_{thinking_name}_DARE_p{int(drop_rate*100)}_lambda{str(scaling_lambda).replace('.', '')}"
            model_output_path = os.path.join(BASE_MODEL_OUTPUT_DIR, output_name)
            merge_func = lambda: model_merger.dare_merge_models(
                base_model_path, instruct_model_path, thinking_model_path, 
                drop_rate, scaling_lambda, model_output_path
            )
        
        elif active_method == 'ties_merge':
            top_k, scaling_lambda = param_value
            output_name = f"{instruct_name}_{thinking_name}_TIES_k{top_k}_lambda{str(scaling_lambda).replace('.', '')}"
            model_output_path = os.path.join(BASE_MODEL_OUTPUT_DIR, output_name)
            merge_func = lambda: model_merger.ties_merge_models(
                base_model_path,
                [instruct_model_path, thinking_model_path], # TIES 可以合并多个模型
                top_k,
                scaling_lambda,
                model_output_path
            )
        elif active_method == 'sce_merge':
            sparsity_threshold, scaling_lambda ,fusion_weights= param_value
            output_name = f"{instruct_name}_{thinking_name}_SCE_sparsity{sparsity_threshold}_lambda{str(scaling_lambda).replace('.', '')}"
            model_output_path = os.path.join(BASE_MODEL_OUTPUT_DIR, output_name)
            merge_func = lambda: model_merger.sce_merge_models(
                base_model_path, [instruct_model_path, thinking_model_path], sparsity_threshold, scaling_lambda, model_output_path, fusion_weights
            )
        # ... 其他合并方法的逻辑可以按需添加 ...
        elif active_method == 'weighted_average':
            output_name =f"{instruct_name}_{thinking_name}_weighted_avg_{int(param_value*100)}I-{100-int(param_value*100)}T"
            model_output_path = os.path.join(BASE_MODEL_OUTPUT_DIR, output_name)
            merge_func = lambda: model_merger.weighted_average_models(instruct_model_path, thinking_model_path, param_value, model_output_path)
        elif active_method == 'surgical_merge':
            output_name = f"{instruct_name}_{thinking_name}_surgical_merge_top_{param_value}pct".replace('.0pct', 'pct')
            model_output_path = os.path.join(BASE_MODEL_OUTPUT_DIR, output_name)
            merge_func = lambda: model_merger.merge_top_k_diff_models(instruct_model_path, thinking_model_path, param_value, model_output_path)
        elif active_method == 'dare_merge_TA':
            drop_rate, scaling_lambda = param_value
            output_name = f"{instruct_name}_{thinking_name}_DARE_TA_p{int(drop_rate*100)}_lambda_1{str(scaling_lambda).replace('.', '')}"
            model_output_path = os.path.join(BASE_MODEL_OUTPUT_DIR, output_name)
            merge_func = lambda: model_merger.dare_merge_models_TA(
                base_model_path, instruct_model_path, thinking_model_path, 
                drop_rate, scaling_lambda, model_output_path
            )
        elif active_method == 'lore_merge':
            max_iter,  task_weight = param_value
            output_name = f"{instruct_name}_{thinking_name}_LORE_max_iter{max_iter}_lambda{str( task_weight[0]).replace('.', '')}"
            model_output_path = os.path.join(BASE_MODEL_OUTPUT_DIR, output_name)
            merge_func = lambda: model_merger.lore_merge_models(
                base_model_path, [instruct_model_path, thinking_model_path], task_weight,max_iter,  model_output_path,0.1
            )
        elif active_method == 'ties_merge_ta':
            top_k, weights = param_value
            output_name = f"{instruct_name}_{thinking_name}_TIES_TA_k{top_k}_weight{str(weights[0]).replace('.', '')}"
            model_output_path = os.path.join(BASE_MODEL_OUTPUT_DIR, output_name)
            merge_func = lambda: model_merger.ties_merge_models_TA(
                base_model_path, [instruct_model_path, thinking_model_path], top_k, weights, model_output_path
            )
        elif active_method == 'twin_merge':
            mask_rate, scaling_lambda = param_value
            output_name = f"{instruct_name}_{thinking_name}_TWIN_mask{int(mask_rate*100)}_lambda{str(scaling_lambda).replace('.', '')}"
            model_output_path = os.path.join(BASE_MODEL_OUTPUT_DIR, output_name)
            merge_func = lambda: model_merger.twin_merge_models(
                base_model_path, [instruct_model_path, thinking_model_path], mask_rate, scaling_lambda, model_output_path
            )
        elif active_method == 'emr':
            cnt= param_value
            output_name = f"{instruct_name}_{thinking_name}_EMR_cnt{cnt[0]}"
            model_output_path = os.path.join(BASE_MODEL_OUTPUT_DIR, output_name)
            merge_func = lambda: model_merger.emr_merge_models(
                base_model_path, [instruct_model_path, thinking_model_path], cnt, model_output_path
            )
        elif active_method == 'slerp':
            cnt= param_value
            output_name = f"{instruct_name}_{thinking_name}_slerp{cnt}"
            model_output_path = os.path.join(BASE_MODEL_OUTPUT_DIR, output_name)
            merge_func = lambda: model_merger.slerp_merge_models(
                instruct_model_path, thinking_model_path, cnt, model_output_path
            )
        elif active_method == 'top_k_avg_keep_thinking':
            # 逻辑: k%差异最大的参数取平均，其余保留 thinking model 的参数.
            # 实现: 调用 merge_top_k_avg_keep_instruct, 它保留第二个(instruct/donor)模型的参数, 我们将 thinking model 作为第二个参数传入.
            output_name = f"{instruct_name}_{thinking_name}_top_k_avg_{param_value}pct_keep_T".replace('.0pct', 'pct')
            merge_func = lambda: model_merger.merge_top_k_avg_keep_instruct(instruct_model_path, thinking_model_path, param_value, model_output_path)


        elif active_method == 'bottom_k_avg_keep_thinking':
            # 逻辑: k%差异最min的参数取平均，其余保留 thinking model 的参数.
            output_name = f"{instruct_name}_{thinking_name}_bottom_k_avg_{param_value}pct_keep_T"
            model_output_path = os.path.join(BASE_MODEL_OUTPUT_DIR, output_name)
            merge_func = lambda: model_merger.merge_bottom_k_avg_keep_instruct(instruct_model_path, thinking_model_path, param_value, model_output_path)
        elif active_method == 'bottom_k_avg_keep_instruct':
            # 逻辑: k%差异最min的参数取平均，其余保留 an instruct model 的参数.
            output_name = f"{instruct_name}_{thinking_name}_bottom_k_avg_{param_value}pct_keep_I"
            model_output_path = os.path.join(BASE_MODEL_OUTPUT_DIR, output_name)
            merge_func = lambda: model_merger.merge_bottom_k_avg_keep_instruct(thinking_model_path,instruct_model_path, param_value, model_output_path)
            #换一下传参的顺序应该就可以。
        elif active_method == 'top_k_avg_keep_instruct':
            # 逻辑: k%差异最大的参数取平均，其余保留 an instruct model 的参数.
            # 实现: 调用 merge_top_k_avg_keep_base, 它保留第一个(base)模型的参数, 我们将 instruct model 作为第一个参数传入.
            output_name = f"{instruct_name}_{thinking_name}_top_k_avg_{param_value}pct_keep_I".replace('.0pct', 'pct')
            model_output_path = os.path.join(BASE_MODEL_OUTPUT_DIR, output_name)
            merge_func = lambda: model_merger.merge_top_k_avg_keep_base(instruct_model_path, thinking_model_path, param_value, model_output_path)

        elif active_method == 'avg_override_top_k_thinking':
            # 逻辑: 所有参数平均，但差异最大的k%参数换回 thinking model 的参数.
            # 实现: 调用 merge_avg_override_top_k_base, 它会用第一个(base)模型的参数覆盖.
            # 因此, 我们必须将 thinking_model_path 作为第一个参数传入才能实现所需逻辑.
            # 注意: 这也会导致保存的 tokenizer 来自 thinking model.
            output_name = f"{instruct_name}_{thinking_name}_avg_override_top_{param_value}pct_with_T".replace('.0pct', 'pct')
            model_output_path = os.path.join(BASE_MODEL_OUTPUT_DIR, output_name)
            merge_func = lambda: model_merger.merge_avg_override_top_k_base(thinking_model_path, instruct_model_path, param_value, model_output_path)
        if not merge_func:
            print(f"🤷 Warning: Implementation logic for method '{active_method}' not found. Skipping.")
            continue
        print(f"模型将保存到: {model_output_path}")
        if not os.path.exists(model_output_path):
            try: 
                merge_func()
                print(f"✅ 模型 '{output_name}' 生成成功!")
            except Exception as e: 
                print(f"❌ 模型生成时出错: {e}. 跳过此参数。")
                continue
        else: 
            print("⚠️  警告: 模型目录已存在, 跳过生成步骤。")
            generated_models_info.append({"model_name": output_name, "output_path": model_output_path})
    print("\n" + "="*80 + "\n✅ All models processed, now generating unified evaluation scripts...\n" + "="*80)

    tasks_string_for_sh = "\n".join([f'    "{task}"' for task in EVAL_TASKS])
    
    all_benchmarks = {data_name.strip() for task in EVAL_TASKS for data_name in task.split(' ')[0].split(',')}
    benchmarks_string_for_py = ",\n".join([f'    "{b}"' for b in sorted(list(all_benchmarks))])

    model_paths_str_list = [f'    "{os.path.relpath(info["output_path"], start=current_run_dir)}"' for info in generated_models_info]
    model_paths_str = "\n".join(model_paths_str_list)
    
    # 【重要修改】计算评估脚本到数据集文件夹的相对路径
    data_dir_relative_path = os.path.relpath(DATA_DIR_PATH_IN_PROJECT, start=current_run_dir).replace("\\", "/")
    
    sh_content = RUN_EVAL_SH_TEMPLATE.format(
        model_paths_string=model_paths_str, 
        tasks_string=tasks_string_for_sh,
        data_dir_path=data_dir_relative_path
    )
    sh_filepath = os.path.join(current_run_dir, "run_eval_all.sh")
    with open(sh_filepath, 'w', encoding='utf-8') as f: f.write(sh_content)
    os.chmod(sh_filepath, os.stat(sh_filepath).st_mode | stat.S_IEXEC)
    print(f"   -> Generated multi-task evaluation script: {sh_filepath}")

    model_configs = []
    for info in generated_models_info:
        model_name = info["model_name"]
        results_path = os.path.join("outputs", model_name).replace("\\", "/")
        tokenizer_path = os.path.relpath(info["output_path"], start=current_run_dir).replace("\\", "/")
        model_configs.append(f'    {{"model_name": "{model_name}", "results_path": "{results_path}", "tokenizer_path": "{tokenizer_path}"}}')
    model_configs_str = ",\n".join(model_configs)
    
    py_content = CHECK_ACCURACY_PY_TEMPLATE.format(
        generation_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        model_configs_string=model_configs_str,
        benchmarks_string=benchmarks_string_for_py
    )
    py_filepath = os.path.join(current_run_dir, "check_accuracy_all.py")
    with open(py_filepath, 'w', encoding='utf-8') as f: f.write(py_content)
    print(f"   -> Generated multi-task accuracy check script: {py_filepath}")

    print("\n\n🎉 All experiment scripts have been generated successfully!")
    print("Then execute the evaluation: ./run_eval_all.sh")
    print("After evaluation is complete, check the accuracy: python check_accuracy_all.py")


if __name__ == '__main__':
    main()