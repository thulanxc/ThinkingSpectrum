#!/bin/bash

# This script runs the math evaluation for a specified model.
# Exit immediately if a command exits with a non-zero status.
set -ex

# =========================================================================
#                           CONFIGURATION
#       Please edit the variables in this section to match your setup.
# =========================================================================

# --- Model & Hardware Configuration ---
# Specify the GPU device to use.
# export CUDA_VISIBLE_DEVICES="0,1"
export CUDA_VISIBLE_DEVICES="0"

MODEL_NAME_OR_PATH="Qwen/Qwen3-4B-Thinking-2507"
# MODEL_NAME_OR_PATH="Qwen/Qwen2.5-Math-7B-Instruct"
# MODEL_NAME_OR_PATH="Qwen/Qwen3-4B-Thinking-2507"
# MODEL_NAME_OR_PATH="../../../models/simple_averaged_Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507"
# --- Evaluation Dataset & Prompt Configuration ---
# Set the prompt type.
PROMPT_TYPE="qwen25-math-cot"
# Set the dataset name for evaluation.
# DATA_NAME="aime24" # Other common options include "gsm8k"
# DATA_NAME="math-500"
# DATA_NAME="aime25"
# DATA_NAME="mmlu_stem"
DATA_NAME="mmlu_redux"

OUTPUT_DIR="${MODEL_NAME_OR_PATH}/math_eval"

# If the model path is local, clean up the output directory path.
if [[ "$OUTPUT_DIR" == *models* ]]; then
  # If it does, remove the "../../../" prefix.
  echo "Found 'models' in OUTPUT_DIR. Removing prefix '../../../'."
  OUTPUT_DIR=${OUTPUT_DIR#"../../../"}
  echo "New OUTPUT_DIR is: ${OUTPUT_DIR}"
fi

SPLIT="test"
NUM_TEST_SAMPLE=-1 # Use -1 to evaluate all samples in the test split.

# --- Run Python Evaluation Script ---
# Disable tokenizer parallelism to avoid potential issues.
TOKENIZERS_PARALLELISM=false \
python3 -u math_eval.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --data_name "${DATA_NAME}" \
    --output_dir "${OUTPUT_DIR}" \
    --split "${SPLIT}" \
    --prompt_type "${PROMPT_TYPE}" \
    --num_test_sample "${NUM_TEST_SAMPLE}" \
    --seed 0 \
    --temperature 0.6 \
    --n_sampling 1 \
    --top_p 0.95 \
    --top_k 20 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --overwrite \
    --max_tokens_per_call 39512

MODEL_NAME_OR_PATH="Qwen/Qwen3-4B-Thinking-2507"
# MODEL_NAME_OR_PATH="Qwen/Qwen2.5-Math-7B-Instruct"
# MODEL_NAME_OR_PATH="Qwen/Qwen3-4B-Thinking-2507"
# MODEL_NAME_OR_PATH="../../../models/simple_averaged_Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507"
# --- Evaluation Dataset & Prompt Configuration ---
# Set the prompt type.
PROMPT_TYPE="qwen25-no-cot"
# Set the dataset name for evaluation.
# DATA_NAME="aime24" # Other common options include "gsm8k"
# DATA_NAME="math-500"
# DATA_NAME="aime25"
# DATA_NAME="mmlu_stem"
DATA_NAME="mmlu_redux"

OUTPUT_DIR="${MODEL_NAME_OR_PATH}/math_eval"

# If the model path is local, clean up the output directory path.
if [[ "$OUTPUT_DIR" == *models* ]]; then
  # If it does, remove the "../../../" prefix.
  echo "Found 'models' in OUTPUT_DIR. Removing prefix '../../../'."
  OUTPUT_DIR=${OUTPUT_DIR#"../../../"}
  echo "New OUTPUT_DIR is: ${OUTPUT_DIR}"
fi

SPLIT="test"
NUM_TEST_SAMPLE=-1 # Use -1 to evaluate all samples in the test split.

# --- Run Python Evaluation Script ---
# Disable tokenizer parallelism to avoid potential issues.
TOKENIZERS_PARALLELISM=false \
python3 -u math_eval.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --data_name "${DATA_NAME}" \
    --output_dir "${OUTPUT_DIR}" \
    --split "${SPLIT}" \
    --prompt_type "${PROMPT_TYPE}" \
    --num_test_sample "${NUM_TEST_SAMPLE}" \
    --seed 0 \
    --temperature 0.6 \
    --n_sampling 1 \
    --top_p 0.95 \
    --top_k 20 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --overwrite \
    --max_tokens_per_call 39512

MODEL_NAME_OR_PATH="Qwen/Qwen3-4B-Instruct-2507"
# MODEL_NAME_OR_PATH="Qwen/Qwen2.5-Math-7B-Instruct"
# MODEL_NAME_OR_PATH="Qwen/Qwen3-4B-Thinking-2507"
# MODEL_NAME_OR_PATH="../../../models/simple_averaged_Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507"
# --- Evaluation Dataset & Prompt Configuration ---
# Set the prompt type.
PROMPT_TYPE="qwen25-math-cot"
# Set the dataset name for evaluation.
# DATA_NAME="aime24" # Other common options include "gsm8k"
# DATA_NAME="math-500"
# DATA_NAME="aime25"
# DATA_NAME="mmlu_stem"
DATA_NAME="mmlu_redux"

OUTPUT_DIR="${MODEL_NAME_OR_PATH}/math_eval"

# If the model path is local, clean up the output directory path.
if [[ "$OUTPUT_DIR" == *models* ]]; then
  # If it does, remove the "../../../" prefix.
  echo "Found 'models' in OUTPUT_DIR. Removing prefix '../../../'."
  OUTPUT_DIR=${OUTPUT_DIR#"../../../"}
  echo "New OUTPUT_DIR is: ${OUTPUT_DIR}"
fi

SPLIT="test"
NUM_TEST_SAMPLE=-1 # Use -1 to evaluate all samples in the test split.

# --- Run Python Evaluation Script ---
# Disable tokenizer parallelism to avoid potential issues.
TOKENIZERS_PARALLELISM=false \
python3 -u math_eval.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --data_name "${DATA_NAME}" \
    --output_dir "${OUTPUT_DIR}" \
    --split "${SPLIT}" \
    --prompt_type "${PROMPT_TYPE}" \
    --num_test_sample "${NUM_TEST_SAMPLE}" \
    --seed 0 \
    --temperature 0.7 \
    --n_sampling 1 \
    --top_p 0.8 \
    --top_k 20 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --overwrite \
    --max_tokens_per_call 39512

MODEL_NAME_OR_PATH="Qwen/Qwen3-4B-Instruct-2507"
# MODEL_NAME_OR_PATH="Qwen/Qwen2.5-Math-7B-Instruct"
# MODEL_NAME_OR_PATH="Qwen/Qwen3-4B-Thinking-2507"
# MODEL_NAME_OR_PATH="../../../models/simple_averaged_Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507"
# --- Evaluation Dataset & Prompt Configuration ---
# Set the prompt type.
PROMPT_TYPE="qwen25-no-cot"
# Set the dataset name for evaluation.
# DATA_NAME="aime24" # Other common options include "gsm8k"
# DATA_NAME="math-500"
# DATA_NAME="aime25"
# DATA_NAME="mmlu_stem"
DATA_NAME="mmlu_redux"

OUTPUT_DIR="${MODEL_NAME_OR_PATH}/math_eval"

# If the model path is local, clean up the output directory path.
if [[ "$OUTPUT_DIR" == *models* ]]; then
  # If it does, remove the "../../../" prefix.
  echo "Found 'models' in OUTPUT_DIR. Removing prefix '../../../'."
  OUTPUT_DIR=${OUTPUT_DIR#"../../../"}
  echo "New OUTPUT_DIR is: ${OUTPUT_DIR}"
fi

SPLIT="test"
NUM_TEST_SAMPLE=-1 # Use -1 to evaluate all samples in the test split.

# --- Run Python Evaluation Script ---
# Disable tokenizer parallelism to avoid potential issues.
TOKENIZERS_PARALLELISM=false \
python3 -u math_eval.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --data_name "${DATA_NAME}" \
    --output_dir "${OUTPUT_DIR}" \
    --split "${SPLIT}" \
    --prompt_type "${PROMPT_TYPE}" \
    --num_test_sample "${NUM_TEST_SAMPLE}" \
    --seed 0 \
    --temperature 0.7 \
    --n_sampling 1 \
    --top_p 0.8 \
    --top_k 20 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --overwrite \
    --max_tokens_per_call 39512



# MODEL_NAME_OR_PATH="Qwen/Qwen2.5-Math-7B-Instruct"
# MODEL_NAME_OR_PATH="Qwen/Qwen3-4B-Thinking-2507"
MODEL_NAME_OR_PATH="../../../models/simple_averaged_Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507"
# --- Evaluation Dataset & Prompt Configuration ---
# Set the prompt type.
PROMPT_TYPE="qwen25-no-cot"
# Set the dataset name for evaluation.
# DATA_NAME="aime24" # Other common options include "gsm8k"
# DATA_NAME="math-500"
# DATA_NAME="aime25"
# DATA_NAME="mmlu_stem"
DATA_NAME="mmlu_redux"

OUTPUT_DIR="${MODEL_NAME_OR_PATH}/math_eval"

# If the model path is local, clean up the output directory path.
if [[ "$OUTPUT_DIR" == *models* ]]; then
  # If it does, remove the "../../../" prefix.
  echo "Found 'models' in OUTPUT_DIR. Removing prefix '../../../'."
  OUTPUT_DIR=${OUTPUT_DIR#"../../../"}
  echo "New OUTPUT_DIR is: ${OUTPUT_DIR}"
fi

SPLIT="test"
NUM_TEST_SAMPLE=-1 # Use -1 to evaluate all samples in the test split.

# --- Run Python Evaluation Script ---
# Disable tokenizer parallelism to avoid potential issues.
TOKENIZERS_PARALLELISM=false \
python3 -u math_eval.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --data_name "${DATA_NAME}" \
    --output_dir "${OUTPUT_DIR}" \
    --split "${SPLIT}" \
    --prompt_type "${PROMPT_TYPE}" \
    --num_test_sample "${NUM_TEST_SAMPLE}" \
    --seed 0 \
    --temperature 0.7 \
    --n_sampling 1 \
    --top_p 0.8 \
    --top_k 20 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --overwrite \
    --max_tokens_per_call 39512


# MODEL_NAME_OR_PATH="Qwen/Qwen2.5-Math-7B-Instruct"
# MODEL_NAME_OR_PATH="Qwen/Qwen3-4B-Thinking-2507"
MODEL_NAME_OR_PATH="../../../models/simple_averaged_Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507"
# --- Evaluation Dataset & Prompt Configuration ---
# Set the prompt type.
PROMPT_TYPE="qwen25-math-cot"
# Set the dataset name for evaluation.
# DATA_NAME="aime24" # Other common options include "gsm8k"
# DATA_NAME="math-500"
# DATA_NAME="aime25"
# DATA_NAME="mmlu_stem"
DATA_NAME="mmlu_redux"

OUTPUT_DIR="${MODEL_NAME_OR_PATH}/math_eval"

# If the model path is local, clean up the output directory path.
if [[ "$OUTPUT_DIR" == *models* ]]; then
  # If it does, remove the "../../../" prefix.
  echo "Found 'models' in OUTPUT_DIR. Removing prefix '../../../'."
  OUTPUT_DIR=${OUTPUT_DIR#"../../../"}
  echo "New OUTPUT_DIR is: ${OUTPUT_DIR}"
fi

SPLIT="test"
NUM_TEST_SAMPLE=-1 # Use -1 to evaluate all samples in the test split.

# --- Run Python Evaluation Script ---
# Disable tokenizer parallelism to avoid potential issues.
TOKENIZERS_PARALLELISM=false \
python3 -u math_eval.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --data_name "${DATA_NAME}" \
    --output_dir "${OUTPUT_DIR}" \
    --split "${SPLIT}" \
    --prompt_type "${PROMPT_TYPE}" \
    --num_test_sample "${NUM_TEST_SAMPLE}" \
    --seed 0 \
    --temperature 0.6 \
    --n_sampling 1 \
    --top_p 0.95 \
    --top_k 20 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --overwrite \
    --max_tokens_per_call 39512