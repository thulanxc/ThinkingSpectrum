#!/bin/bash
# This script runs the math evaluation for a specified model.
# Exit immediately if a command exits with a non-zero status.
set -ex

# =========================================================================
#                           CONFIGURATION
#       Please edit the variables in this section to match your setup.
# =========================================================================

export CUDA_VISIBLE_DEVICES="3"
MODEL_NAME_OR_PATH="../../../models/Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_20I-80T"
PROMPT_TYPE="qwen25-no-cot"
DATA_NAME="aime24,aime25"

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
    --n_sampling 10 \
    --top_p 0.8 \
    --top_k 20 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --overwrite \
    --max_tokens_per_call 39512

MODEL_NAME_OR_PATH="../../../models/Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_20I-80T"
PROMPT_TYPE="qwen25-no-cot"
DATA_NAME="gpqa_diamond"

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
    --n_sampling 2 \
    --top_p 0.8 \
    --top_k 20 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --overwrite \
    --max_tokens_per_call 39512

MODEL_NAME_OR_PATH="../../../models/Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_80I-20T"
PROMPT_TYPE="qwen25-no-cot"
DATA_NAME="aime24,aime25"

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
    --n_sampling 10 \
    --top_p 0.8 \
    --top_k 20 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --overwrite \
    --max_tokens_per_call 39512

MODEL_NAME_OR_PATH="../../../models/Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_80I-20T"
PROMPT_TYPE="qwen25-no-cot"
DATA_NAME="gpqa_diamond"

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
    --n_sampling 2 \
    --top_p 0.8 \
    --top_k 20 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --overwrite \
    --max_tokens_per_call 39512
