import os
import json
import argparse
import sys
from datetime import datetime

# Attempt to import the transformers library
try:
    from transformers import AutoTokenizer
except ImportError:
    print("FATAL ERROR: The 'transformers' library was not found.")
    print("Please install the required libraries by running: pip install transformers torch sentencepiece")
    sys.exit(1)

# --- SCRIPT CONFIGURATION ---
# SCRIPT LOCATION: /datadisk/MergingL2S/Qwen2.5-Math/evaluation/
#
# IMPORTANT: Configure your models to evaluate here.
# Paths are now relative to the new script location.

MODELS_TO_EVALUATE = [
    # {
    #     "results_path": "outputs/Qwen/Qwen3-4B-Thinking-2507",
    #     "tokenizer_path": "Qwen/Qwen3-4B-Instruct-2507"  # Using a known valid HF ID for tokenizer
    # },
    #     {
    #     "results_path": "outputs/Qwen/Qwen3-4B-Instruct-2507",
    #     "tokenizer_path": "Qwen/Qwen3-4B-Instruct-2507"  # Using a known valid HF ID for tokenizer
    # },
    # {
    #     # Path for results is now 'outputs/models/...'
    #     "results_path": "outputs/models/Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_80I-20T",
    #     # Path to tokenizer is adjusted for the script's new parent directory location
    #     "tokenizer_path": "../../../models/Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_80I-20T"
    # },
    # {
    #     # Path for results is now 'outputs/models/...'
    #     "results_path": "outputs/models/Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_20I-80T",
    #     # Path to tokenizer is adjusted for the script's new parent directory location
    #     "tokenizer_path": "../../../models/Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_20I-80T"
    # }
    {
        "results_path": "outputs/Qwen/Qwen3-4B-Instruct-2507",
        "tokenizer_path": "Qwen/Qwen3-4B-Instruct-2507"  # Using a known valid HF ID for tokenizer
    },
    {
        "results_path": "outputs/models/simple_averaged_Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507",
        "tokenizer_path": "../../../models/simple_averaged_Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507"  # Using a known valid HF ID for tokenizer
    },
]

# MODELS_TO_EVALUATE = [
#     {
#         "results_path": "outputs/Qwen/Qwen3-4B-Instruct-2507",
#         "tokenizer_path": "Qwen/Qwen3-4B-Instruct-2507"  # Using a known valid HF ID for tokenizer
#     },
# ]

# NOTE: Benchmark paths should be relative to the model's result directory.
DEFAULT_BENCHMARKS = [
    "math_eval/gpqa_diamond",
    # "math_eval/math-500",
    "math_eval/aime24",
    "math_eval/aime25",
    # "math_eval/mmlu_redux"
]

# --- CORE PROCESSING FUNCTION ---

def process_jsonl_file(file_path, tokenizer):
    """
    Calculates the average accuracy from the 'score' field and the average
    token count for all responses in the 'code' field of a .jsonl file.
    """
    total_predictions = 0
    correct_predictions = 0
    total_tokens = 0
    total_responses = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())

                    if 'score' in data and isinstance(data['score'], list):
                        scores = data['score']
                        total_predictions += len(scores)
                        correct_predictions += sum(bool(s) for s in scores)
                    
                    if 'code' in data and isinstance(data['code'], list):
                        for response_text in data['code']:
                            if isinstance(response_text, str):
                                token_ids = tokenizer.encode(response_text)
                                total_tokens += len(token_ids)
                                total_responses += 1

                except json.JSONDecodeError:
                    print(f"    - WARNING: JSON format error on line {i} of '{os.path.basename(file_path)}'.")
                    continue

    except FileNotFoundError:
        print(f"ERROR: The file '{file_path}' was not found during processing.")
        return None, None, None, None, None

    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0.0
    avg_tokens = total_tokens / total_responses if total_responses > 0 else 0.0

    return accuracy, correct_predictions, total_predictions, avg_tokens

# --- MAIN EXECUTION (Revised for per-file output) ---

def main():
    parser = argparse.ArgumentParser(
        description="A script to automatically evaluate multiple models across multiple benchmarks.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--benchmarks",
        nargs='+',
        default=DEFAULT_BENCHMARKS,
        help="A list of benchmark subdirectories to evaluate for each model.\n(default: %(default)s)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="outputs/evaluation_summary.txt", # Default output location is now inside 'outputs'
        help="The name of the file to save the summary results to.\n(default: %(default)s)"
    )
    args = parser.parse_args()

    models_to_run = MODELS_TO_EVALUATE
    benchmarks_to_run = args.benchmarks
    output_filename = args.output_file

    print("==================================================")
    print("Starting Automated Model Evaluation (Per-File Mode)")
    print(f"Script location: {os.getcwd()}")
    print("==================================================")
    print(f"Models to evaluate (see config): {len(models_to_run)} total")
    print(f"Benchmarks to run: {benchmarks_to_run}")
    print(f"Results will be saved to: {output_filename}\n")

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    with open(output_filename, 'w', encoding='utf-8') as summary_file:
        summary_file.write(f"Evaluation Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        summary_file.write("="*80 + "\n")

        for model_config in models_to_run:
            results_path = model_config["results_path"]
            tokenizer_path = model_config["tokenizer_path"]
            
            print(f"--- Processing Model Results From: '{results_path}' ---")
            summary_file.write(f"Model Results: {results_path}\n")

            tokenizer = None
            try:
                print(f"  🔄 Loading tokenizer from '{tokenizer_path}'...")
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
                print("  ✅ Tokenizer loaded successfully.")
            except Exception as e:
                error_msg = f"ERROR: Could not load tokenizer. Skipping all benchmarks for this model."
                print(f"  ❌ {error_msg}")
                summary_file.write(f"  {error_msg}\n\n")
                continue

            for benchmark_dir_rel_path in benchmarks_to_run:
                full_benchmark_path = os.path.join(results_path, benchmark_dir_rel_path)
                
                print(f"\n  ▶️  Evaluating Benchmark Directory: {benchmark_dir_rel_path}")
                print(f"      Scanning for .jsonl files in: {full_benchmark_path}")

                if not os.path.isdir(full_benchmark_path):
                    message = "SKIPPED: Directory not found."
                    print(f"      ❗️ {message}")
                    summary_file.write(f"  - Directory: {benchmark_dir_rel_path:<40} | {message}\n")
                    continue

                jsonl_files = sorted([f for f in os.listdir(full_benchmark_path) if f.endswith('.jsonl')])

                if not jsonl_files:
                    message = "SKIPPED: No .jsonl files found in directory."
                    print(f"      ❗️ {message}")
                    summary_file.write(f"  - Directory: {benchmark_dir_rel_path:<40} | {message}\n")
                    continue
                
                # *** MODIFIED LOGIC: Process and print each file individually ***
                for filename in jsonl_files:
                    full_file_path = os.path.join(full_benchmark_path, filename)
                    
                    results = process_jsonl_file(full_file_path, tokenizer)
                    
                    if results:
                        accuracy, correct, total, avg_tokens = results

                        # Create a unique identifier for the output line, e.g., "math_eval/gpqa_diamond/file.jsonl"
                        output_label = os.path.join(benchmark_dir_rel_path, filename)

                        console_output = f"      ✔️  File: {filename} | Accuracy: {accuracy:.2f}% ({correct}/{total}), Avg Tokens: {avg_tokens:.0f}"
                        file_output = (f"  - File: {output_label:<50} | "
                                       f"Accuracy: {accuracy:>6.2f}% ({correct:>4}/{total:<4}), "
                                       f"Avg Tokens: {avg_tokens:.0f}")
                        
                        print(console_output)
                        summary_file.write(file_output + "\n")

            summary_file.write("\n")
            print("\n" + "-"*50 + "\n")

    print(f"🎉 Evaluation complete. Full results saved to '{output_filename}'.")

if __name__ == "__main__":
    main()