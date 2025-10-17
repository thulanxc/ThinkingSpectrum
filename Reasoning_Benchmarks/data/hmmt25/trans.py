import pandas as pd
import json

# --- 前提 ---
# 在运行此代码之前，请确保您已经安装了必要的库。
# 您可以使用 pip 来安装：
# pip install pandas pyarrow

# 定义输入和输出文件名
input_file = 'train-00000-of-00001.parquet'
output_file = 'test.jsonl'

try:
    # 1. 使用 pandas 读取 Parquet 文件
    print(f"正在读取文件: {input_file}...")
    df = pd.read_parquet(input_file)
    print("文件读取成功！")

    # 2. 打开输出文件准备写入
    with open(output_file, 'w', encoding='utf-8') as f:
        # 3. 遍历 DataFrame 的每一行
        for index, row in df.iterrows():
            # 4. 按照您指定的格式创建一个字典
            #    - "problem" 来自 'problem' 列
            #    - "answer" 来自 'answer' 列
            #    - "id" 来自 'problem_idx' 列，并转换为字符串
            record = {
                "problem": row['problem'],
                "answer": row['answer'],
                "id": str(row['problem_idx'])
            }
            
            # 5. 将字典转换为 JSON 字符串并写入文件，
            #    然后在末尾添加换行符，以符合 jsonl 格式
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"文件 '{output_file}' 已成功创建！")

except FileNotFoundError:
    print(f"错误：输入文件 '{input_file}' 未找到。请确保文件与脚本在同一目录下。")
except Exception as e:
    print(f"处理过程中发生错误: {e}")