import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def average_models(model_names, output_path):
    """
    Averages the parameters of two models with the same architecture and saves the result.

    Args:
        model_names (list): A list containing the names of the two models,
                            in the format "organization/model_name".
        output_path (str): The local path to save the merged model.
    """
    if len(model_names) != 2:
        raise ValueError("This function is designed to average two models. Please provide a list with two model names.")

    model_name_a, model_name_b = model_names
    print(f"Loading model 1: {model_name_a}")
    # device_map="auto" automatically distributes model layers across available hardware (GPU/CPU) to avoid OOM errors.
    model_a = AutoModelForCausalLM.from_pretrained(
        model_name_a,
        torch_dtype=torch.bfloat16,  # Use bfloat16 to save memory
        device_map="auto",
        trust_remote_code=True
    )
    model_a.eval()  # Set to evaluation mode

    print(f"Loading model 2: {model_name_b}")
    model_b = AutoModelForCausalLM.from_pretrained(
        model_name_b,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model_b.eval()

    # Get the state dictionaries of the two models
    params_a = model_a.state_dict()
    params_b = model_b.state_dict()

    # Create a new state dictionary to store the averaged parameters
    averaged_params = {}

    print("\nStarting to average model parameters...")
    # Use tqdm to create a progress bar
    for param_name in tqdm(params_a.keys(), desc="Averaging weights"):
        # Check if the parameter exists in both models
        if param_name in params_b:
            # Get the corresponding parameter tensors from both models
            tensor_a = params_a[param_name]
            tensor_b = params_b[param_name].to(tensor_a.device)  # Ensure the tensors are on the same device

            # Calculate the average
            averaged_params[param_name] = (tensor_a + tensor_b) / 2.0
        else:
            print(f"Warning: Parameter '{param_name}' found only in model {model_name_a}. Using it directly.")
            averaged_params[param_name] = params_a[param_name]

    # Load the averaged parameters back into the first model's architecture
    # We reuse model_a's structure to hold the new weights
    print("\nLoading averaged parameters into the model structure...")
    model_a.load_state_dict(averaged_params)

    # Ensure the output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Created output directory: {output_path}")

    # Save the merged model
    print(f"Saving the merged model to: {output_path}")
    model_a.save_pretrained(output_path)

    # Load and save the tokenizer
    # Tokenizers from source-compatible models are usually compatible. We'll save one of them.
    print(f"Saving the tokenizer to: {output_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_b, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    print("\nModel averaging and saving process completed successfully!")

if __name__ == '__main__':
    # --- Configuration ---
    # Names of the two models to be averaged
    # MODELS_TO_AVERAGE = [
    #     "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    #     "Qwen/Qwen2.5-Math-7B-Instruct"
    # ]
    MODELS_TO_AVERAGE = [
        "Qwen/Qwen3-4B-Instruct-2507",
        "Qwen/Qwen3-4B-Thinking-2507"
    ]
    # Path to save the averaged model
    # OUTPUT_MODEL_PATH = "../../../models/simple_averaged_R1_7B_Qwen_Math_7B"
    OUTPUT_MODEL_PATH = "../../../models/simple_averaged_Qwen3-4B-Instruct-2507_Qwen3-4B-Thinking-2507"
    # --- Execute Script ---
    try:
        average_models(MODELS_TO_AVERAGE, OUTPUT_MODEL_PATH)
    except Exception as e:
        print(f"\nAn error occurred during the process: {e}")