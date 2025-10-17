# run_benchmark_final.py (Final Version - Corrected with Dual Response Saving and GPU Selection)
import os
import re
import uuid
import time
import json
import logging
import sys
import gc
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List

# --- 关键依赖 ---
try:
    import torch
    from transformers import AutoTokenizer
    import openai
    from openai import OpenAI
except ImportError:
    print("错误：PyTorch, Transformers 或 OpenAI 未安装。请运行 'pip install torch transformers openai' 来安装。")
    sys.exit(1)

import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams

# =====================================================================================
# ⚙️ 1. 配置区域
# =====================================================================================

# <--- 新增功能: 指定要使用的GPU ---
# 设置 CUDA_VISIBLE_DEVICES 环境变量。
# 例如: "0,1" 表示仅使用 GPU 0 和 GPU 1。
# 设置为 None 或 "" 表示使用所有可见的GPU (默认行为)。
VISIBLE_GPUS = "6,7" # <--- 在这里修改为你想要的GPU ID，例如 "0,1" 或 "2"

# MODELS_TO_TEST = ["../../models/Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_slerp0.8"]
MODELS_TO_TEST = ["../../models/Qwen3-30B-A3B-Instruct-2507_Qwen3-30B-A3B-Thinking-2507_weighted_avg_20I-80T",
    "../../models/Qwen3-30B-A3B-Instruct-2507_Qwen3-30B-A3B-Thinking-2507_weighted_avg_30I-70T",
    "../../models/Qwen3-30B-A3B-Instruct-2507_Qwen3-30B-A3B-Thinking-2507_weighted_avg_40I-60T",
    "../../models/Qwen3-30B-A3B-Instruct-2507_Qwen3-30B-A3B-Thinking-2507_weighted_avg_50I-50T",
    "../../models/Qwen3-30B-A3B-Instruct-2507_Qwen3-30B-A3B-Thinking-2507_weighted_avg_60I-40T",
    "../../models/Qwen3-30B-A3B-Instruct-2507_Qwen3-30B-A3B-Thinking-2507_weighted_avg_70I-30T",
    "../../models/Qwen3-30B-A3B-Instruct-2507_Qwen3-30B-A3B-Thinking-2507_weighted_avg_80I-20T",
    "../../models/Qwen3-30B-A3B-Instruct-2507_Qwen3-30B-A3B-Thinking-2507_weighted_avg_90I-10T",
    "../../models/Qwen3-30B-A3B-Instruct-2507_Qwen3-30B-A3B-Thinking-2507_weighted_avg_100I-0T",
    "../../models/Qwen3-30B-A3B-Instruct-2507_Qwen3-30B-A3B-Thinking-2507_top_k_avg_1pct_keep_I",
    "../../models/Qwen3-30B-A3B-Instruct-2507_Qwen3-30B-A3B-Thinking-2507_top_k_avg_2pct_keep_I",
    "../../models/Qwen3-30B-A3B-Instruct-2507_Qwen3-30B-A3B-Thinking-2507_top_k_avg_5pct_keep_I",
    "../../models/Qwen3-30B-A3B-Instruct-2507_Qwen3-30B-A3B-Thinking-2507_top_k_avg_10pct_keep_I",
    "../../models/Qwen3-30B-A3B-Instruct-2507_Qwen3-30B-A3B-Thinking-2507_top_k_avg_20pct_keep_I",
    "../../models/Qwen3-30B-A3B-Instruct-2507_Qwen3-30B-A3B-Thinking-2507_top_k_avg_50pct_keep_I",
    "../../models/Qwen3-30B-A3B-Instruct-2507_Qwen3-30B-A3B-Thinking-2507_surgical_merge_top_1pct",
    "../../models/Qwen3-30B-A3B-Instruct-2507_Qwen3-30B-A3B-Thinking-2507_surgical_merge_top_2pct",
    "../../models/Qwen3-30B-A3B-Instruct-2507_Qwen3-30B-A3B-Thinking-2507_surgical_merge_top_5pct",
    "../../models/Qwen3-30B-A3B-Instruct-2507_Qwen3-30B-A3B-Thinking-2507_surgical_merge_top_10pct",
    "../../models/Qwen3-30B-A3B-Instruct-2507_Qwen3-30B-A3B-Thinking-2507_surgical_merge_top_20pct",
    "../../models/Qwen3-30B-A3B-Instruct-2507_Qwen3-30B-A3B-Thinking-2507_surgical_merge_top_50pct",
    "../../models/Qwen3-30B-A3B-Instruct-2507_Qwen3-30B-A3B-Thinking-2507_slerp0.1",
    "../../models/Qwen3-30B-A3B-Instruct-2507_Qwen3-30B-A3B-Thinking-2507_slerp0.2",
    "../../models/Qwen3-30B-A3B-Instruct-2507_Qwen3-30B-A3B-Thinking-2507_slerp0.3",
    "../../models/Qwen3-30B-A3B-Instruct-2507_Qwen3-30B-A3B-Thinking-2507_slerp0.4",
    "../../models/Qwen3-30B-A3B-Instruct-2507_Qwen3-30B-A3B-Thinking-2507_slerp0.5",
    "../../models/Qwen3-30B-A3B-Instruct-2507_Qwen3-30B-A3B-Thinking-2507_slerp0.6",
    "../../models/Qwen3-30B-A3B-Instruct-2507_Qwen3-30B-A3B-Thinking-2507_slerp0.7",
    "../../models/Qwen3-30B-A3B-Instruct-2507_Qwen3-30B-A3B-Thinking-2507_slerp0.8",
    "../../models/Qwen3-30B-A3B-Instruct-2507_Qwen3-30B-A3B-Thinking-2507_slerp0.9",
    "../../models/Qwen3-30B-A3B-Instruct-2507_Qwen3-30B-A3B-Thinking-2507_avg_override_top_1pct_with_T",
    "../../models/Qwen3-30B-A3B-Instruct-2507_Qwen3-30B-A3B-Thinking-2507_avg_override_top_2pct_with_T",
    "../../models/Qwen3-30B-A3B-Instruct-2507_Qwen3-30B-A3B-Thinking-2507_avg_override_top_5pct_with_T",
    "../../models/Qwen3-30B-A3B-Instruct-2507_Qwen3-30B-A3B-Thinking-2507_avg_override_top_10pct_with_T",
    "../../models/Qwen3-30B-A3B-Instruct-2507_Qwen3-30B-A3B-Thinking-2507_avg_override_top_20pct_with_T",
    "../../models/Qwen3-30B-A3B-Instruct-2507_Qwen3-30B-A3B-Thinking-2507_avg_override_top_50pct_with_T"]

JUDGE_CONFIG = {
    "model_name": "kimi-k2-turbo-preview",
    "base_url": "https://api.moonshot.cn/v1",
    "api_key": "sk-RXvmowS7kvaEcPpH6E2u4soQE5PZ1N02ZnUALsEscG60uygF", # 请替换为您的有效 API Key
}

VLLM_PARAMS = {
    "tensor_parallel_size": 2,
    "gpu_memory_utilization": 0.90,
    "max_model_len": 4096,
    "trust_remote_code": True,
}

SAMPLING_PARAMS = {
    "temperature": 0.7,
    "min_p": 0.1,
    "max_tokens": 4096
}

BENCHMARK_PARAMS = {
    "output_dir": "cw_output",
    "num_threads_judging": 64,
    "iterations": 3,
    "creative_prompts_file": "data/creative_writing_prompts_v3.json",
    "criteria_file": "data/creative_writing_criteria.txt",
    "negative_criteria_file": "data/negative_criteria.txt",
    "judge_prompt_file": "data/creative_writing_judging_prompt.txt",
}

# =====================================================================================
# 📚 2. 辅助函数 (Scoring部分已替换为原版逻辑)
# =====================================================================================

logging.basicConfig(level="INFO", format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# --- 初始化 Judge API 客户端 ---
try:
    judge_client = OpenAI(
        api_key=JUDGE_CONFIG["api_key"],
        base_url=JUDGE_CONFIG["base_url"],
    )
except Exception as e:
    logging.error(f"初始化 Judge API 客户端失败: {e}")
    judge_client = None

def load_data_file(file_path):
    """直接加载文件，如果失败则程序崩溃并报错。"""
    with open(file_path, 'r', encoding='utf-8') as f:
        if file_path.endswith('.json'):
            return json.load(f)
        else:
            return [line.strip() for line in f if line.strip()]

# ========== 核心修改：从这里开始，使用您提供的原版 scoring 函数 ==========

SCORE_RANGE_MAX = 20 # 原版代码中的全局变量

def parse_judge_scores_creative(judge_model_response: str) -> Dict[str, float]:
    """
    原版解析函数: 使用正则表达式从自然语言文本中提取分数。
    """
    scores = {}
    score_pattern1 = r'(.*?):\s*(?:Score\s+)?(-?\d+(?:\.\d+)?)'
    score_pattern2 = r'(.*?):\s*\[(-?\d+(?:\.\d+)?)\]'
    
    matches1 = re.findall(score_pattern1, judge_model_response)
    matches2 = re.findall(score_pattern2, judge_model_response)
    
    for matches in [matches1, matches2]:
        for match in matches:
            metric_name = match[0].strip()
            # 过滤掉一些非预期的匹配
            if len(metric_name) > 50 or len(metric_name) < 2:
                continue
            try:
                score = float(match[1])
                if score <= SCORE_RANGE_MAX:
                    scores[metric_name] = score
            except ValueError:
                continue
    return scores

def call_judge_api(prompt: str) -> dict:
    """使用 openai 库调用打分API，并使用原版解析函数。"""
    if not judge_client:
        return {"raw_judge_text": "[ERROR: Judge client not initialized]", "judge_scores": {}}
        
    for attempt in range(10):
        try:
            response = judge_client.chat.completions.create(
                model=JUDGE_CONFIG["model_name"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1000,
                timeout=240,
            )
            raw_text = response.choices[0].message.content
            # 调用原版的解析函数
            scores = parse_judge_scores_creative(raw_text)
            return {"raw_judge_text": raw_text, "judge_scores": scores}
        except Exception as e:
            logging.warning(f"打分API调用失败 (尝试 {attempt+1}/10): {e.__class__.__name__} - {e}")
            time.sleep(2)
            
    return {"raw_judge_text": "[ERROR: API call failed after multiple retries]", "judge_scores": {}}

# --- 原版计分逻辑函数 ---

def invert_if_negative_original(metric: str, score: float, negative_criteria: List[str]) -> float:
    """原版负面指标分数翻转逻辑"""
    # 注意：原版代码检查 metric 是否在 negative_criteria 列表中
    # 而不是像您脚本中那样检查子字符串
    if metric in negative_criteria:
        return 20.0 - score
    return score

def compute_creative_scores_original(tasks: List[Dict[str, Any]], negative_criteria: List[str]) -> float:
    """原版计算所有作品平均分的逻辑 (0-20分制)"""
    piece_scores = []
    for task_data in tasks:
        # 在我们的简化脚本中，每个task就是一个作品，没有results_by_modifier这层
        j_scores = task_data.get("judge_scores", {})
        if not j_scores:
            continue
        
        local_vals = []
        for metric, val in j_scores.items():
            if isinstance(val, (int, float)):
                # 检查 metric 是否与加载的负面标准完全匹配
                # 我们需要找到原始的 criterion name 来做这个判断
                # 这是一个挑战，因为解析出的 metric name 可能不完全一致
                # 我们采用一种近似方法：检查是否有任何负面标准是解析出的metric的子串
                is_negative = any(neg_crit in metric for neg_crit in negative_criteria)
                
                # 简化的翻转逻辑
                if is_negative:
                    new_val = 20.0 - val
                else:
                    new_val = val

                if new_val <= SCORE_RANGE_MAX:
                    local_vals.append(new_val)

        if local_vals:
            # 原版逻辑是先对每个作品的所有评分项取平均
            piece_score = sum(local_vals) / len(local_vals)
            piece_scores.append(piece_score)

    if not piece_scores:
        return 0.0
    # 最后对所有作品的平均分再取平均
    return sum(piece_scores) / len(piece_scores)

def compute_final_scores_original(tasks: List[Dict[str, Any]], negative_criteria: List[str]) -> dict:
    """原版最终计分函数，整合所有逻辑"""
    avg_0_20 = compute_creative_scores_original(tasks, negative_criteria)
    eqbench_score = avg_0_20 * 5.0
    eqbench_score = round(eqbench_score, 2)
    
    # 收集所有详细分数用于报告
    scores_by_metric = {}
    for task in tasks:
        if task.get("judge_scores"):
            for metric, val in task["judge_scores"].items():
                if isinstance(val, (int, float)):
                    scores_by_metric.setdefault(metric, []).append(val)
    
    detailed_scores_avg = {m: np.mean(v) for m, v in scores_by_metric.items() if v}

    return {
        "overall_score_0_20": round(avg_0_20, 2),
        "eqbench_creative_score": eqbench_score,
        "detailed_scores_raw": detailed_scores_avg
    }

# ========== 核心修改结束 ==========

# =====================================================================================
# 🚀 3. 主执行流程 (阶段8被替换)
# =====================================================================================

if __name__ == "__main__":
    # <--- 新增: 应用GPU配置 ---
    # 这个操作必须在任何CUDA库（如torch或vLLM）初始化之前完成。
    if VISIBLE_GPUS:
        os.environ["CUDA_VISIBLE_DEVICES"] = VISIBLE_GPUS
    
    # --- 阶段 1: 加载所有必需文件 ---
    logging.info(f"阶段 1: 加载 prompts 和 criteria 文件... (可见GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'All')})")
    try:
        prompts_data = load_data_file(BENCHMARK_PARAMS['creative_prompts_file'])
        creative_criteria = load_data_file(BENCHMARK_PARAMS['criteria_file'])
        negative_criteria = load_data_file(BENCHMARK_PARAMS['negative_criteria_file'])
        judge_prompt_template = "".join(load_data_file(BENCHMARK_PARAMS['judge_prompt_file']))
    except FileNotFoundError as e:
        logging.error(f"加载数据文件失败: {e}. 请确保 'data' 文件夹及其中所有必需文件都存在。")
        sys.exit(1)
    logging.info("文件加载完毕。")

    # --- 对每个模型进行完整的评测循环 ---
    for model_identifier in MODELS_TO_TEST:
        logging.info(f"===== 开始评测模型: {model_identifier} =====")
        
        logging.info(f"正在为模型 '{model_identifier}' 加载 Tokenizer...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_identifier, trust_remote_code=True)
        except Exception as e:
            logging.error(f"加载 Tokenizer 失败: {e}. 跳过此模型。")
            continue
        logging.info("Tokenizer 加载成功。")

        logging.info("阶段 2: 准备任务列表和应用聊天模板...")
        tasks = []
        prompts_to_generate = []
        # 注意：这里的循环为每个prompt创建了iterations次任务
        for prompt_id, prompt_obj in prompts_data.items():
            for i in range(1, BENCHMARK_PARAMS['iterations'] + 1):
                seed = prompt_obj.get("seed_modifiers", ["default"])[(i - 1) % len(prompt_obj.get("seed_modifiers", ["default"]))]
                raw_prompt = prompt_obj["writing_prompt"].replace("<SEED>", seed)
                messages = [{"role": "user", "content": raw_prompt}]
                final_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                task = {
                    "prompt_id": prompt_id, "iteration": i, "base_prompt": prompt_obj["writing_prompt"],
                    "model_response": None,
                    "cleaned_model_response": None, # <--- 修改点 1: 为清理后的响应添加占位符
                    "judge_scores": None, "raw_judge_text": None,
                    "response_token_length": None,
                }
                tasks.append(task)
                prompts_to_generate.append(final_prompt)
        logging.info(f"已准备 {len(tasks)} 个格式化后的生成任务。")

        logging.info("阶段 3: 初始化 vLLM 引擎...")
        llm = LLM(model=model_identifier, **VLLM_PARAMS)
        logging.info("vLLM 引擎初始化成功。")

        logging.info("阶段 4: 开始批量生成文本...")
        generated_responses = llm.generate(prompts_to_generate, SamplingParams(**SAMPLING_PARAMS), use_tqdm=True)
        
        # --- 计算Token长度的逻辑 ---
        all_token_lengths = []
        for i, response in enumerate(generated_responses):
            raw_response_text = response.outputs[0].text
            tasks[i]["model_response"] = raw_response_text
            # 使用加载的tokenizer计算token数量并存储
            token_count = len(tokenizer.encode(raw_response_text))
            tasks[i]["response_token_length"] = token_count
            all_token_lengths.append(token_count)
            
        # --- 计算平均Token长度 ---
        avg_token_length = sum(all_token_lengths) / len(all_token_lengths) if all_token_lengths else 0.0
        logging.info(f"成功生成了 {len(generated_responses)} 条响应。平均Token长度: {avg_token_length:.2f}")

        logging.info("阶段 5: 清理GPU资源...")
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        logging.info("GPU资源已释放。")
        
        logging.info("阶段 6: 准备打分任务...")
        tasks_to_judge = []
        judge_prompts = []
        for task in tasks:
            if task["model_response"] and len(task["model_response"].strip()) >= 50:
                response_for_judge = re.sub(r'<think>.*?</think>', '', task["model_response"], flags=re.DOTALL).strip()
                task["cleaned_model_response"] = response_for_judge # <--- 修改点 2: 将清理后的响应保存回task字典
                
                if len(response_for_judge) >= 50:
                    judge_prompts.append(judge_prompt_template.format(
                        writing_prompt=task["base_prompt"], 
                        test_model_response=response_for_judge,
                        creative_writing_criteria="\n".join([f"- {c}" for c in creative_criteria]),
                        lower_is_better_criteria=", ".join(negative_criteria)))
                    tasks_to_judge.append(task)
                else:
                    task["judge_scores"], task["raw_judge_text"] = {}, "[SKIPPED - Response too short after cleaning <think> tags]"
            else:
                short_response_info = f"len={len(task['model_response'].strip())}" if task["model_response"] else "failed"
                task["judge_scores"], task["raw_judge_text"] = {}, f"[SKIPPED - Generation too short or failed ({short_response_info})]"
        
        if tasks_to_judge:
            logging.info(f"阶段 7: 开始对 {len(tasks_to_judge)} 个有效文本进行并发打分...")
            with ThreadPoolExecutor(max_workers=BENCHMARK_PARAMS['num_threads_judging']) as executor:
                judging_results = list(tqdm(executor.map(call_judge_api, judge_prompts), total=len(judge_prompts), desc="Judging"))

            for i, result in enumerate(judging_results):
                tasks_to_judge[i].update(result)
            logging.info("打分任务完成。")
        else:
            logging.info("阶段 7: 没有有效的生成文本需要打分。")

        # --- 阶段 8: 使用原版逻辑计算最终分数 ---
        logging.info("阶段 8: 计算最终分数...")
        final_scores_dict = compute_final_scores_original(tasks_to_judge, negative_criteria)
        logging.info("分数计算完成。")


        # --- 阶段 9: 保存结果并打印报告 ---
        logging.info("阶段 9: 保存详细结果并打印总结报告...")
        sanitized_name = re.sub(r'[^a-zA-Z0-9_-]+', '_', model_identifier.split('/')[-1])
        output_dir = BENCHMARK_PARAMS['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        results_file = os.path.join(output_dir, f"final_results_{sanitized_name}_{uuid.uuid4().hex[:6]}.json")
        
        final_output = {
            "model_tested": model_identifier, "judge_model": JUDGE_CONFIG["model_name"],
            "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "final_scores": {
                "rubric_score_0_20": final_scores_dict["overall_score_0_20"], 
                "eqbench_score_0_100": final_scores_dict["eqbench_creative_score"],
                "avg_response_token_length": round(avg_token_length, 2),
                "detailed_scores_raw": final_scores_dict["detailed_scores_raw"],
            },
            "individual_results": tasks # <--- 现在这里保存的tasks包含了两个版本的response
        }
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)

        logging.info(f"完整结果已保存至: {results_file}")
        print("\n" + "="*60)
        print(f"  模型评测完成: {model_identifier}")
        print("-"*60)
        print(f"  Rubric Score (0-20):    {final_scores_dict['overall_score_0_20']:.2f}")
        print(f"  EQ-Bench Score (0-100): {final_scores_dict['eqbench_creative_score']:.2f}")
        print(f"  Avg. Token Length:      {avg_token_length:.2f}")
        print("="*60 + "\n")

    logging.info("所有模型均已评测完毕。")