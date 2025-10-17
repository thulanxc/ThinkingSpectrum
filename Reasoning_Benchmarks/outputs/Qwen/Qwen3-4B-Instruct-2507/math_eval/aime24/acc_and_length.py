import os
import json
import argparse

# Import the transformers library
try:
    from transformers import AutoTokenizer
except ImportError:
    print("Error: 'transformers' library not found.")
    print("Please run 'pip install transformers torch sentencepiece' to install.")
    exit()

# --- Configuration ---
# Specify the Hugging Face Tokenizer model to use
# TOKENIZER_MODEL = "Qwen/Qwen2.5-Math-7B-Instruct"
TOKENIZER_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

def process_jsonl_file(file_path, tokenizer):
    """
    Calculates the average accuracy of all 'score' entries in a single .jsonl file,
    and also calculates the average token count for all responses in the 'code' field.
    """
    # Variables for accuracy calculation
    total_predictions = 0
    correct_predictions = 0
    
    # Variables for token count calculation
    total_tokens = 0
    total_responses = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                    
                    # --- 1. Calculate accuracy (original logic) ---
                    if 'score' in data and isinstance(data['score'], list):
                        scores = data['score']
                        total_predictions += len(scores)
                        # bool(True) is 1, bool(False) is 0, so they can be summed directly
                        correct_predictions += sum(bool(s) for s in scores)
                    else:
                        print(f"   - Warning: Missing 'score' list in line {i+1} of file '{os.path.basename(file_path)}'.")

                    # --- 2. New: Calculate token count ---
                    if 'code' in data and isinstance(data['code'], list):
                        for response_text in data['code']:
                            # Ensure the element is a string before tokenizing
                            if isinstance(response_text, str):
                                token_ids = tokenizer.encode(response_text)
                                total_tokens += len(token_ids)
                                total_responses += 1
                            else:
                                print(f"   - Warning: Non-string element found in 'code' list in line {i+1} of file '{os.path.basename(file_path)}'.")
                    else:
                        print(f"   - Warning: Missing 'code' list in line {i+1} of file '{os.path.basename(file_path)}'.")

                except json.JSONDecodeError:
                    print(f"   - Warning: JSON format error in line {i+1} of file '{os.path.basename(file_path)}'.")
                    continue
    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None, None, None, None

    # --- 3. Calculate final results ---
    # Calculate accuracy, avoiding division by zero
    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0.0
    # Calculate average tokens, avoiding division by zero
    avg_tokens = total_tokens / total_responses if total_responses > 0 else 0.0

    return accuracy, correct_predictions, total_predictions, avg_tokens

def main():
    """
    Main function to parse arguments and iterate through the folder.
    """
    parser = argparse.ArgumentParser(
        description="Calculate the 'flattened' average accuracy and average token count for all .jsonl files in a folder."
    )
    parser.add_argument(
        "folder_path",
        type=str,
        nargs='?', # Make the argument optional
        default='.', # Default value is the current directory
        help="Path to the folder containing .jsonl files (defaults to the current directory)"
    )
    args = parser.parse_args()
    
    target_folder = args.folder_path

    if not os.path.isdir(target_folder):
        print(f"❌ Error: '{target_folder}' is not a valid directory path.")
        return

    # --- New: Load Tokenizer ---
    print(f"🔄 Loading Tokenizer: '{TOKENIZER_MODEL}'...")
    try:
        # trust_remote_code=True is required for some models (like Qwen)
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL, trust_remote_code=True)
        print("Tokenizer loaded successfully!\n")
    except Exception as e:
        print(f"❌ Error: Failed to load Tokenizer.")
        print(f"   Please ensure you have a working internet connection and the required libraries are installed (pip install transformers torch sentencepiece).")
        print(f"   Detailed error: {e}")
        return

    print(f"🔍 Scanning folder: '{os.path.abspath(target_folder)}'...\n")

    jsonl_files = [f for f in os.listdir(target_folder) if f.endswith('.jsonl')]

    if not jsonl_files:
        print("No .jsonl files found in this directory.")
        return
        
    for filename in sorted(jsonl_files):
        file_path = os.path.join(target_folder, filename)
        accuracy, correct, total, avg_tokens = process_jsonl_file(file_path, tokenizer)
        
        if accuracy is not None:
            print(f"📄 File: {filename}")
            print(f"   => Average Accuracy: {accuracy:.2f}% ({correct}/{total} predictions correct)")
            print(f"   => Average Token Count: {avg_tokens:.0f}\n") # New output

if __name__ == "__main__":
    main()