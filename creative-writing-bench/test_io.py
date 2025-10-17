# test_io.py
import os
import json
import time

output_dir = "cw_output"
test_file = os.path.join(output_dir, "io_test.json")
data_to_write = {"status": "ok", "timestamp": time.time()}

print(f"--- 1. 准备在 '{output_dir}' 文件夹下写入文件...")
os.makedirs(output_dir, exist_ok=True)
print(f"--- 1. 目录 '{output_dir}' 已确认存在。")

print(f"\n--- 2. 正在尝试写入文件: {test_file} ---")
print("--- 如果程序卡在这里，说明您的文件系统存在写入问题。 ---")

try:
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(data_to_write, f, indent=4)
    
    print(f"\n--- 3. 写入成功！ ---")
    print("--- 这意味着您的文件系统没有问题。问题可能更复杂。 ---")

except Exception as e:
    print(f"\n--- 3. 写入失败！---")
    print(f"--- 错误信息: {e} ---")