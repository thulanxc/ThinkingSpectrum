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
# IMPORTANT: Configure your models to evaluate here.
# Each model is a dictionary with two keys:
# 1. 'results_path': The folder path where the result .jsonl files are located,
#                    relative to this script's location (the 'outputs' directory).
# 2. 'tokenizer_path': The identifier for the tokenizer. This can be:
#                      a) A Hugging Face Hub model ID (e.g., "Qwen/Qwen2-7B-Instruct").
#                      b) A relative path to a local directory containing model files
#                         (e.g., "../../../../models/my_local_model").

MODELS_TO_EVALUATE = [
    {
        "results_path": "Qwen/Qwen3-4B-Thinking-2507",
        "tokenizer_path": "Qwen/Qwen3-4B-Thinking-2507"  # Use a valid HF ID for the tokenizer
    },
    {
        "results_path": "Qwen/Qwen3-4B-Instruct-2507",
        "tokenizer_path": "Qwen/Qwen3-4B-Instruct-2507"  # Use a valid HF ID for the tokenizer
    },
    {
        # Example for a locally saved/merged model
        "results_path": "models/simple_averaged_Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507",
        "tokenizer_path": "../../../../models/simple_averaged_Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507"
    }
    # Add more model configurations here
]

# NOTE: Benchmark paths should be relative to the model's result directory.
DEFAULT_BENCHMARKS = [
    "math_eval/gpqa_diamond",
    # e.g., "gsm8k_eval/gsm8k",
]

# --- CORE PROCESSING FUNCTION (Unchanged) ---

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
                    else:
                        print(f"    - WARNING: Line {i} in '{os.path.basename(file_path)}' is missing a 'score' list.")

                    if 'code' in data and isinstance(data['code'], list):
                        for response_text in data['code']:
                            if isinstance(response_text, str):
                                token_ids = tokenizer.encode(response_text)
                                total_tokens += len(token_ids)
                                total_responses += 1
                            else:
                                print(f"    - WARNING: Non-string element found in 'code' list on line {i} of '{os.path.basename(file_path)}'.")
                    else:
                        print(f"    - WARNING: Line {i} in '{os.path.basename(file_path)}' is missing a 'code' list.")

                except json.JSONDecodeError:
                    print(f"    - WARNING: JSON format error on line {i} of '{os.path.basename(file_path)}'.")
                    continue

    except FileNotFoundError:
        print(f"ERROR: The file '{file_path}' was not found.")
        return None, None, None, None

    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0.0
    avg_tokens = total_tokens / total_responses if total_responses > 0 else 0.0
    return accuracy, correct_predictions, total_predictions, avg_tokens

# --- MAIN EXECUTION (Revised) ---

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
        default="evaluation_summary.txt",
        help="The name of the file to save the summary results to.\n(default: %(default)s)"
    )
    args = parser.parse_args()

    # Models are now configured in the list at the top of the script
    models_to_run = MODELS_TO_EVALUATE
    benchmarks_to_run = args.benchmarks
    output_filename = args.output_file

    print("==================================================")
    print("Starting Automated Model Evaluation (Revised)")
    print("==================================================")
    print(f"Models to evaluate (see config): {len(models_to_run)} total")
    print(f"Benchmarks to run: {benchmarks_to_run}")
    print(f"Results will be saved to: {output_filename}\n")

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
                print(f"     Details: {e}")
                summary_file.write(f"  {error_msg}\n\n")
                continue

            for benchmark_path in benchmarks_to_run:
                benchmark_name = os.path.basename(benchmark_path)
                jsonl_filename = f"{benchmark_name}.jsonl"
                
                # Construct path to the RESULTS file
                full_results_path = os.path.join(results_path, benchmark_path, jsonl_filename)

                print(f"\n  ▶️  Evaluating Benchmark: {benchmark_path}")
                print(f"      File path: {full_results_path}")

                if not os.path.exists(full_results_path):
                    message = "SKIPPED: File not found."
                    print(f"      ❗️ {message}")
                    summary_file.write(f"  - Benchmark: {benchmark_path:<40} | {message}\n")
                    continue

                accuracy, correct, total, avg_tokens = process_jsonl_file(full_results_path, tokenizer)

                if accuracy is not None:
                    console_output = f"      ✔️  Result: Accuracy: {accuracy:.2f}% ({correct}/{total}), Avg Tokens: {avg_tokens:.0f}"
                    file_output = (f"  - Benchmark: {benchmark_path:<40} | "
                                   f"Accuracy: {accuracy:>6.2f}% ({correct:>4}/{total:<4}), "
                                   f"Avg Tokens: {avg_tokens:.0f}")
                    
                    print(console_output)
                    summary_file.write(file_output + "\n")

            summary_file.write("\n")
            print("\n" + "-"*50 + "\n")

    print(f"🎉 Evaluation complete. Full results saved to '{output_filename}'.")

if __name__ == "__main__":
    main()