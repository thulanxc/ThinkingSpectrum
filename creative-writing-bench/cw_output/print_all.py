import os
import json

def extract_scores_from_json():
    """
    Finds all JSON files in the current directory, extracts specific scores,
    and saves them to a summary text file.
    """
    output_filename = "scores_summary.txt"
    files_in_directory = os.listdir('.')
    json_files = [f for f in files_in_directory if f.endswith('.json')]

    if not json_files:
        print("在当前文件夹中没有找到 JSON 文件。")
        return

    print(f"找到了 {len(json_files)} 个 JSON 文件。正在处理...")

    with open(output_filename, 'w', encoding='utf-8') as f_out:
        for filename in json_files:
            filepath = os.path.join('.', filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f_in:
                    data = json.load(f_in)
                    
                    # Safely access nested keys
                    final_scores = data.get('final_scores', {})
                    eqbench_score = final_scores.get('eqbench_score_0_100')
                    avg_response_length = final_scores.get('avg_response_token_length')

                    # Check if both values were found before writing
                    if eqbench_score is not None and avg_response_length is not None:
                        f_out.write(f"文件名: {filename}\n")
                        f_out.write(f"  eqbench_score_0_100: {eqbench_score}\n")
                        f_out.write(f"  avg_response_token_length: {avg_response_length}\n")
                        f_out.write("-" * 30 + "\n")
                    else:
                        print(f"警告: 在文件 '{filename}' 中未能找到所需的键，已跳过。")

            except json.JSONDecodeError:
                print(f"警告: 文件 '{filename}' 不是一个有效的JSON文件，已跳过。")
            except Exception as e:
                print(f"处理文件 '{filename}' 时发生未知错误: {e}")

    print(f"\n处理完成！结果已保存至 '{output_filename}' 文件中。")

if __name__ == "__main__":
    extract_scores_from_json()