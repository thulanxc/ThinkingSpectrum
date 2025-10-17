#!/bin/bash
set -ex
export CUDA_VISIBLE_DEVICES="6,7"
PROMPT_TYPE="qwen25-no-cot"

MODELS=(
    "../../../../../models/Qwen3-30B-A3B-Instruct-2507_Qwen3-30B-A3B-Thinking-2507_weighted_avg_0I-100T"
    "../../../../../models/Qwen3-30B-A3B-Instruct-2507_Qwen3-30B-A3B-Thinking-2507_weighted_avg_10I-90T"
    "../../../../../models/Qwen3-30B-A3B-Instruct-2507_Qwen3-30B-A3B-Thinking-2507_weighted_avg_20I-80T"
    "../../../../../models/Qwen3-30B-A3B-Instruct-2507_Qwen3-30B-A3B-Thinking-2507_weighted_avg_30I-70T"
    "../../../../../models/Qwen3-30B-A3B-Instruct-2507_Qwen3-30B-A3B-Thinking-2507_weighted_avg_40I-60T"
    "../../../../../models/Qwen3-30B-A3B-Instruct-2507_Qwen3-30B-A3B-Thinking-2507_weighted_avg_50I-50T"
    "../../../../../models/Qwen3-30B-A3B-Instruct-2507_Qwen3-30B-A3B-Thinking-2507_weighted_avg_60I-40T"
    "../../../../../models/Qwen3-30B-A3B-Instruct-2507_Qwen3-30B-A3B-Thinking-2507_weighted_avg_70I-30T"
    "../../../../../models/Qwen3-30B-A3B-Instruct-2507_Qwen3-30B-A3B-Thinking-2507_weighted_avg_80I-20T"
    "../../../../../models/Qwen3-30B-A3B-Instruct-2507_Qwen3-30B-A3B-Thinking-2507_weighted_avg_90I-10T"
    "../../../../../models/Qwen3-30B-A3B-Instruct-2507_Qwen3-30B-A3B-Thinking-2507_weighted_avg_100I-0T"
)

TASKS=(
    "aime25 10"
)

run_evaluation() {
    local MODEL_NAME_OR_PATH=$1
    local DATA_NAME=$2
    local N_SAMPLING=$3
    local OUTPUT_DIR

    echo "================================================================"
    echo "STARTING EVALUATION FOR: ${MODEL_NAME_OR_PATH}"
    echo "ON DATA: ${DATA_NAME} with N_SAMPLING: ${N_SAMPLING}"
    echo "================================================================"
    
    CLEAN_MODEL_NAME=$(basename "${MODEL_NAME_OR_PATH}")
    OUTPUT_DIR="${CLEAN_MODEL_NAME}"

    # 【重要修改】显式传入数据集的路径，确保脚本能找到数据
    TOKENIZERS_PARALLELISM=false \
    python3 -u ../../math_eval.py \
        --model_name_or_path "${MODEL_NAME_OR_PATH}" \
        --data_name "${DATA_NAME}" \
        --data_dir "../../../../data" \
        --output_dir "${OUTPUT_DIR}" \
        --split "test" \
        --prompt_type "${PROMPT_TYPE}" \
        --num_test_sample -1 \
        --seed 0 \
        --temperature 0.6 \
        --n_sampling "${N_SAMPLING}" \
        --top_p 0.95 \
        --top_k 20 \
        --start 0 \
        --end -1 \
        --use_vllm \
        --save_outputs \
        --overwrite \
        --max_tokens_per_call 39512 \
        --presence_penalty 0 \
        --frequency_penalty 0
}

for model in "${MODELS[@]}"; do
    for task in "${TASKS[@]}"; do
        read -r data_names n_sampling <<< "$task"
        run_evaluation "$model" "$data_names" "$n_sampling"
    done
done

echo "All evaluation tasks are complete."
