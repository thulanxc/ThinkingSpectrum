import os
import json
import argparse

# 导入 transformers 库
try:
    from transformers import AutoTokenizer
except ImportError:
    print("错误: 'transformers' 库未找到。")
    print("请运行 'pip install transformers torch sentencepiece' 进行安装。")
    exit()

# --- 配置 ---
# 指定要使用的 Hugging Face Tokenizer 模型
TOKENIZER_MODEL = "Qwen/Qwen2.5-Math-7B-Instruct"

def process_jsonl_file(file_path, tokenizer):
    """
    计算单个 .jsonl 文件中所有 score 的平均正确率，
    并计算 'code' 字段中所有回答的平均 Token 数量。
    """
    # 用于计算准确率的变量
    total_predictions = 0
    correct_predictions = 0
    
    # 用于计算Token数的变量
    total_tokens = 0
    total_responses = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                    
                    # --- 1. 计算准确率 (原有逻辑) ---
                    if 'score' in data and isinstance(data['score'], list):
                        scores = data['score']
                        total_predictions += len(scores)
                        # bool(True) 是 1, bool(False) 是 0，可以直接求和
                        correct_predictions += sum(bool(s) for s in scores)
                    else:
                        print(f"   - 警告: 文件 '{os.path.basename(file_path)}' 的第 {i+1} 行缺少 'score' 列表。")

                    # --- 2. 新增: 计算Token数 ---
                    if 'code' in data and isinstance(data['code'], list):
                        for response_text in data['code']:
                            # 确保元素是字符串类型再进行tokenize
                            if isinstance(response_text, str):
                                token_ids = tokenizer.encode(response_text)
                                total_tokens += len(token_ids)
                                total_responses += 1
                            else:
                                print(f"   - 警告: 文件 '{os.path.basename(file_path)}' 的第 {i+1} 行 'code' 列表中包含非字符串元素。")
                    else:
                        print(f"   - 警告: 文件 '{os.path.basename(file_path)}' 的第 {i+1} 行缺少 'code' 列表。")

                except json.JSONDecodeError:
                    print(f"   - 警告: 文件 '{os.path.basename(file_path)}' 的第 {i+1} 行 JSON 格式错误。")
                    continue
    
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到。")
        return None, None, None, None

    # --- 3. 计算最终结果 ---
    # 计算准确率，避免除以零
    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0.0
    # 计算平均Token数，避免除以零
    avg_tokens = total_tokens / total_responses if total_responses > 0 else 0.0

    return accuracy, correct_predictions, total_predictions, avg_tokens

def main():
    """
    主函数，用于解析参数并遍历文件夹。
    """
    parser = argparse.ArgumentParser(
        description="计算文件夹中所有 .jsonl 文件的‘拍平后’的平均准确率和平均Token数。"
    )
    parser.add_argument(
        "folder_path",
        type=str,
        nargs='?', # 使参数变为可选
        default='.', # 默认值为当前文件夹
        help="包含 .jsonl 文件的文件夹路径 (默认为当前文件夹)"
    )
    args = parser.parse_args()
    
    target_folder = args.folder_path

    if not os.path.isdir(target_folder):
        print(f"❌ 错误: '{target_folder}' 不是一个有效的文件夹路径。")
        return

    # --- 新增: 加载 Tokenizer ---
    print(f"🔄 正在加载 Tokenizer: '{TOKENIZER_MODEL}'...")
    try:
        # trust_remote_code=True 是某些模型（如Qwen）所必需的
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL, trust_remote_code=True)
        print("Tokenizer 加载成功！\n")
    except Exception as e:
        print(f"❌ 错误: 加载 Tokenizer 失败。")
        print(f"   请确保网络连接正常，并已正确安装所需库 (pip install transformers torch sentencepiece)。")
        print(f"   详细错误: {e}")
        return

    print(f"🔍 正在扫描文件夹: '{os.path.abspath(target_folder)}'...\n")

    jsonl_files = [f for f in os.listdir(target_folder) if f.endswith('.jsonl')]

    if not jsonl_files:
        print("在该文件夹中未找到 .jsonl 文件。")
        return
        
    for filename in sorted(jsonl_files):
        file_path = os.path.join(target_folder, filename)
        accuracy, correct, total, avg_tokens = process_jsonl_file(file_path, tokenizer)
        
        if accuracy is not None:
            print(f"📄 文件: {filename}")
            print(f"   => 平均准确率: {accuracy:.2f}% ({correct}/{total} 个预测正确)")
            print(f"   => 平均Token数: {avg_tokens:.0f}\n") # 新增输出

if __name__ == "__main__":
    main()