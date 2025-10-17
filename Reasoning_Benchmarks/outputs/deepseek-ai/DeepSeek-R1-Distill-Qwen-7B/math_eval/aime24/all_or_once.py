import os
import json
import argparse

def calculate_flattened_accuracy(file_path):
    """
    计算单个 .jsonl 文件中所有 score 的平均正确率。
    这相当于把所有问题的 n_sampling 次尝试“拍平”后计算总的 pass@1。
    """
    total_predictions = 0
    correct_predictions = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                    
                    # 检查 'score' 字段是否存在且为列表
                    if 'score' in data and isinstance(data['score'], list):
                        scores = data['score']
                        # 累加总预测次数
                        total_predictions += len(scores)
                        # 累加正确次数 (True计为1, False计为0)
                        correct_predictions += sum(scores)
                    else:
                        print(f"  - 警告: 文件 '{os.path.basename(file_path)}' 的第 {i+1} 行缺少 'score' 列表。")

                except json.JSONDecodeError:
                    print(f"  - 警告: 文件 '{os.path.basename(file_path)}' 的第 {i+1} 行 JSON 格式错误。")
                    continue
    
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到。")
        return None, None, None

    # 避免除以零
    if total_predictions == 0:
        return 0.0, 0, 0

    accuracy = (correct_predictions / total_predictions) * 100
    return accuracy, correct_predictions, total_predictions

def main():
    """
    主函数，用于解析参数并遍历文件夹。
    """
    parser = argparse.ArgumentParser(
        description="计算文件夹中所有 .jsonl 文件的‘拍平后’的平均准确率。"
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
        print(f"错误: '{target_folder}' 不是一个有效的文件夹路径。")
        return

    print(f"🔍 正在扫描文件夹: '{os.path.abspath(target_folder)}'...\n")

    # 获取所有 .jsonl 文件
    jsonl_files = [f for f in os.listdir(target_folder) if f.endswith('.jsonl')]

    if not jsonl_files:
        print("在该文件夹中未找到 .jsonl 文件。")
        return
        
    # 对文件进行排序以保证输出顺序一致
    for filename in sorted(jsonl_files):
        file_path = os.path.join(target_folder, filename)
        accuracy, correct, total = calculate_flattened_accuracy(file_path)
        
        if accuracy is not None:
            print(f"📄 文件: {filename}")
            print(f"   => 平均准确率: {accuracy:.2f}% ({correct}/{total} 个预测正确)\n")

if __name__ == "__main__":
    main()