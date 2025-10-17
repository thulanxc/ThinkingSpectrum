#!/bin/bash
# This script runs the math evaluation for a list of specified models and tasks.
# Exit immediately if a command exits with a non-zero status.
set -ex

# =========================================================================
#                             CONFIGURATION
#           Please edit the variables in this section to match your setup.
# =========================================================================

export CUDA_VISIBLE_DEVICES="0"
PROMPT_TYPE="qwen25-no-cot"

# --- Add all the model paths you want to evaluate into this list ---
MODELS=(
    # --- Weighted Average Models ---
    "Qwen/Qwen3-4B-Thinking-2507"
    # "../../../models/simple_averaged_Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507"
    # "../../../models/Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_30I-70T"
    # "../../../models/Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_70I-30T"
    # "../../../models/Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_weighted_avg_90I-10T"

    # # --- Surgical Merge Models ---
    # "../../../models/Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_surgical_merge_top_1pct"
    # "../../../models/Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_surgical_merge_top_5pct"
    # "../../../models/Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_surgical_merge_top_10pct"
    # "../../../models/Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507_surgical_merge_top_20pct"
)

# --- Define the evaluation tasks. Each task is a string: "DATA_NAME N_SAMPLING" ---
# TASKS=(
#     "aime24,aime25 64"
#     "gpqa_diamond 2"
# )
TASKS=(
    "gpqa_diamond 2"
)

# =========================================================================
#                       EVALUATION EXECUTION
#        You shouldn't need to edit anything below this line.
# =========================================================================

# --- Function to run a single evaluation ---
run_evaluation() {
    local MODEL_NAME_OR_PATH=$1
    local DATA_NAME=$2
    local N_SAMPLING=$3

    echo "========================================================================="
    echo "Starting evaluation for MODEL: ${MODEL_NAME_OR_PATH}"
    echo "on DATA: ${DATA_NAME} with N_SAMPLING: ${N_SAMPLING}"
    echo "========================================================================="

    local OUTPUT_DIR="${MODEL_NAME_OR_PATH}/math_eval"

    # If the model path is local, clean up the output directory path.
    if [[ "$OUTPUT_DIR" == *models* ]]; then
        # If it does, remove the "../../../" prefix.
        echo "Found 'models' in OUTPUT_DIR. Removing prefix '../../../'."
        OUTPUT_DIR=${OUTPUT_DIR#"../../../"}
        echo "New OUTPUT_DIR is: ${OUTPUT_DIR}"
    fi

    local SPLIT="test"
    local NUM_TEST_SAMPLE=-1 # Use -1 to evaluate all samples in the test split.

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
        --n_sampling "${N_SAMPLING}" \
        --top_p 0.8 \
        --top_k 20 \
        --start 0 \
        --end -1 \
        --use_vllm \
        --save_outputs \
        --overwrite \
        --max_tokens_per_call 39512

    echo "-------------------------------------------------------------------------"
    echo "Finished evaluation for MODEL: ${MODEL_NAME_OR_PATH} on DATA: ${DATA_NAME}"
    echo "-------------------------------------------------------------------------"
    echo ""
}

# --- Loop through all configured models and tasks ---
for model in "${MODELS[@]}"; do
    for task in "${TASKS[@]}"; do
        # Split the task string into DATA_NAME and N_SAMPLING
        read -r data_name n_sampling <<< "$task"
        run_evaluation "$model" "$data_name" "$n_sampling"
    done
done

echo "========================================================================="
echo "All evaluations completed."
echo "========================================================================="

