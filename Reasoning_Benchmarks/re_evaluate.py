import json
from tqdm import tqdm
import os

# --- Imports from your custom files ---
# Make sure parser.py and grader.py are in the same directory as this script.
try:
    from parser import extract_answer
    from grader import math_equal
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure 'parser.py' and 'grader.py' are in the same directory as this script.")
    exit(1)

# --- Configuration: Specify your filenames here ---
# 1. Input file: The JSONL file with the data to be corrected.
INPUT_FILE = "outputs/Qwen/Qwen3-4B-Thinking-2507/math_eval/mmlu_redux/test_qwen25-no-cot_-1_seed0_t0.6_s0_e-1.jsonl" 

# 2. Output file: The new JSONL file where the corrected data will be saved.
#    It's now safe to use the same name as the input file, but a different name is recommended.
OUTPUT_FILE = "outputs/Qwen/Qwen3-4B-Thinking-2507/math_eval/mmlu_redux/test_qwen25-no-cot_-1_seed0_t0.6_s0_e-1_corrected.jsonl"
# ----------------------------------------------------

def re_evaluate_jsonl_and_calculate_acc(input_file_path: str, output_file_path: str):
    """
    Reads a JSONL file, re-parses the 'pred' and 'score' fields, calculates accuracy,
    and writes the results to a new JSONL file.

    Args:
        input_file_path (str): The path to the input JSONL file.
        output_file_path (str): The path to the output JSONL file.
    """
    # Initialize statistics
    lines_processed = 0
    lines_failed = 0
    total_correct_predictions = 0
    total_predictions = 0
    
    # --- Step 1: Read all lines from the input file into memory first. ---
    # This is crucial to prevent data loss if input and output files are the same.
    try:
        print(f"Reading data from '{input_file_path}'...")
        with open(input_file_path, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()
        if not lines:
            print("Warning: Input file is empty. No data to process.")
            # Create an empty output file
            open(output_file_path, 'w').close()
            return
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_file_path}'")
        return
        
    # --- Step 2: Process the lines from memory and write to the output file. ---
    try:
        # Create parent directory for output file if it doesn't exist
        output_dir = os.path.dirname(output_file_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            for line in tqdm(lines, desc="Processing file"):
                try:
                    data = json.loads(line.strip())
                    
                    model_outputs = data.get("code", [])
                    ground_truth = data.get("gt", "")
                    
                    if not model_outputs or ground_truth is None:
                        outfile.write(json.dumps(data) + '\n')
                        continue

                    new_preds = []
                    new_scores = []

                    for output in model_outputs:
                        # Assuming the dataset is 'aime25' based on the file path
                        predicted_answer = extract_answer(output, data_name="aime25")
                        new_preds.append(predicted_answer)

                        is_correct = math_equal(prediction=predicted_answer, reference=str(ground_truth))
                        new_scores.append(is_correct)
                    
                    # Update statistics for accuracy calculation
                    total_correct_predictions += sum(new_scores) # sum() works on booleans (True=1, False=0)
                    total_predictions += len(new_scores)
                    
                    # Update the JSON object with new predictions and scores
                    data['pred'] = new_preds
                    data['score'] = new_scores
                    
                    outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                    lines_processed += 1

                except json.JSONDecodeError:
                    print(f"\nWarning: Skipping invalid JSON line: {line.strip()}")
                    lines_failed += 1
                    continue
                except Exception as e:
                    print(f"\nAn unknown error occurred while processing a line: {e}")
                    print(f"Line content: {line.strip()}")
                    lines_failed += 1
                    outfile.write(line) # Write the original line on error

    except IOError as e:
        print(f"Error writing to file '{output_file_path}': {e}")
        return
        
    # --- Step 3: Print the final summary report after processing all lines. ---
    print("\nProcessing complete!")
    print(f"Successfully processed and wrote {lines_processed} records.")
    if lines_failed > 0:
        print(f"Failed to process {lines_failed} records.")
    print(f"Corrected file saved to: {output_file_path}")

    # --- Step 4: Calculate and print the overall accuracy. ---
    if total_predictions > 0:
        accuracy = (total_correct_predictions / total_predictions) * 100
        print("\n" + "="*30)
        print("  Overall Accuracy")
        print("="*30)
        print(f"  Total Correct Predictions: {total_correct_predictions}")
        print(f"  Total Predictions:         {total_predictions}")
        print(f"  Accuracy (ACC):            {accuracy:.2f}%")
        print("="*30)
    else:
        print("\nNo predictions were found, cannot calculate accuracy.")


def main():
    """
    Main function to run the script.
    """
    print(f"Input file:  {INPUT_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    if INPUT_FILE == OUTPUT_FILE:
        print("\nWarning: Input and output filenames are the same. The original file will be overwritten.")
    
    re_evaluate_jsonl_and_calculate_acc(INPUT_FILE, OUTPUT_FILE)


if __name__ == "__main__":
    main()