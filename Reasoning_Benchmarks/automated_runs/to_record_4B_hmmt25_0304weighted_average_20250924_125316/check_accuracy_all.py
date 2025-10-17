import os, json
from transformers import AutoTokenizer
# Auto-generated on 2025-09-24 12:53:15 to check multiple models.

MODELS_TO_EVALUATE = [
    {"model_name": "Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_31I-69T", "results_path": "outputs/Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_31I-69T", "tokenizer_path": "../../../../../models/Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_31I-69T"},
    {"model_name": "Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_32I-68T", "results_path": "outputs/Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_32I-68T", "tokenizer_path": "../../../../../models/Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_32I-68T"},
    {"model_name": "Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_33I-67T", "results_path": "outputs/Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_33I-67T", "tokenizer_path": "../../../../../models/Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_33I-67T"},
    {"model_name": "Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_34I-66T", "results_path": "outputs/Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_34I-66T", "tokenizer_path": "../../../../../models/Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_34I-66T"},
    {"model_name": "Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_35I-65T", "results_path": "outputs/Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_35I-65T", "tokenizer_path": "../../../../../models/Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_35I-65T"},
    {"model_name": "Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_36I-64T", "results_path": "outputs/Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_36I-64T", "tokenizer_path": "../../../../../models/Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_36I-64T"},
    {"model_name": "Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_37I-63T", "results_path": "outputs/Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_37I-63T", "tokenizer_path": "../../../../../models/Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_37I-63T"},
    {"model_name": "Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_38I-62T", "results_path": "outputs/Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_38I-62T", "tokenizer_path": "../../../../../models/Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_38I-62T"},
    {"model_name": "Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_39I-61T", "results_path": "outputs/Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_39I-61T", "tokenizer_path": "../../../../../models/Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_39I-61T"}
]
DEFAULT_BENCHMARKS = [
    "hmmt25"
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
    print("="*60 + "\nStarting Accuracy Evaluation\n" + "="*60)
    for config in MODELS_TO_EVALUATE:
        print("\n--- Model: '{}' ---".format(config['model_name']))
        try:
            tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_path'], trust_remote_code=True)
        except Exception as e:
            print("     ❌ ERROR: Failed to load tokenizer from {}. Details: {}. Skipping.".format(config['tokenizer_path'], e))
            continue
        for benchmark in DEFAULT_BENCHMARKS:
            bench_path = os.path.join(config['results_path'], benchmark)
            print("     ▶️ Benchmark: {}".format(benchmark))
            if not os.path.isdir(bench_path):
                print("         ❗️ SKIPPED: Directory not found: {}".format(bench_path))
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
                    print("         ✔️ {:<40} | Accuracy: {:>6.2f}% ({:>3}/{:<3}) | Avg Length: {:.0f} tokens".format(
                        fname, accuracy, correct, total, avg_tokens
                    ))
if __name__ == "__main__": main()
