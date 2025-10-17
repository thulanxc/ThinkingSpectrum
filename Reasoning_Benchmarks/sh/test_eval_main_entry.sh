# Evaluate Qwen2.5-Math-Instruct
PROMPT_TYPE="qwen25-math-cot"

# Qwen2.5-Math-1.5B-Instruct
export CUDA_VISIBLE_DEVICES="0"
MODEL_NAME_OR_PATH="Qwen/Qwen2.5-Math-1.5B-Instruct"
bash sh/simple_eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH
