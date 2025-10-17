import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from tqdm import tqdm

# --- Configuration ---
# Model A: The reference model whose statistics we want to match.
MODEL_A_NAME = "Qwen/Qwen2.5-1.5B"
# Model B: The model we want to modify.
MODEL_B_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# Output Directory: Where to save the new, normalized model.
OUTPUT_DIR = "../../../models/DeepSeek-R1-Qwen-1.5B-normalized"

def normalize_model_parameters(model_a_name, model_b_name, output_dir):
    """
    Normalizes the parameters of model B to match the per-tensor mean and variance
    of model A and saves the new model.
    """
    print("--- Starting Model Parameter Normalization ---")

    # 1. Setup device and data type
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    print(f"Using device: {device} with dtype: {dtype}")

    # 2. Load models and the tokenizer for the model to be saved
    print(f"Loading reference model (A): {model_a_name}")
    model_a = AutoModelForCausalLM.from_pretrained(
        model_a_name,
        dtype=dtype,
        device_map=device,
        trust_remote_code=True
    )
    model_a.eval()

    print(f"Loading target model (B): {model_b_name}")
    model_b = AutoModelForCausalLM.from_pretrained(
        model_b_name,
        dtype=dtype,
        device_map=device,
        trust_remote_code=True
    )
    tokenizer_b = AutoTokenizer.from_pretrained(model_b_name, trust_remote_code=True)
    model_b.eval()

    # 3. Get the state dictionaries (parameter collections)
    params_a = model_a.state_dict()
    params_b = model_b.state_dict()

    print("\nStarting parameter normalization process...")
    # A small value to add to the denominator for numerical stability
    epsilon = 1e-8

    # Perform operations without tracking gradients to save memory and compute
    with torch.no_grad():
        # Iterate over each parameter in the target model
        for param_name in tqdm(params_b.keys(), desc="Normalizing Tensors"):
            # Ensure the parameter also exists in the reference model
            if param_name in params_a:
                tensor_a = params_a[param_name]
                tensor_b = params_b[param_name]

                # Sanity check: ensure tensors have matching shapes
                if tensor_a.shape != tensor_b.shape:
                    print(f"Warning: Skipping {param_name} due to shape mismatch. "
                          f"A: {tensor_a.shape}, B: {tensor_b.shape}")
                    continue
                
                # Skip scalar tensors which have no variance
                if tensor_b.dim() == 0:
                    continue

                # Use float32 for higher precision during statistical calculations
                tensor_a_f32 = tensor_a.float()
                tensor_b_f32 = tensor_b.float()

                # Calculate statistics for the reference tensor (A)
                mean_a = torch.mean(tensor_a_f32)
                var_a = torch.var(tensor_a_f32, unbiased=False)
                std_a = torch.sqrt(var_a + epsilon)

                # Calculate statistics for the target tensor (B)
                mean_b = torch.mean(tensor_b_f32)
                var_b = torch.var(tensor_b_f32, unbiased=False)
                std_b = torch.sqrt(var_b + epsilon)
                
                # --- The Core Transformation ---
                # Step 1: Standardize tensor_b (z-score normalization)
                # This gives it a mean of 0 and a standard deviation of 1.
                normalized_b = (tensor_b_f32 - mean_b) / std_b
                
                # Step 2: Rescale the standardized tensor to match tensor_a's statistics
                new_tensor_b = normalized_b * std_a + mean_a

                # Step 3: Update the parameter in model B's state dictionary
                # Use .copy_() for an efficient in-place update of the tensor's data.
                # Cast back to the original data type.
                params_b[param_name].copy_(new_tensor_b.to(tensor_b.dtype))

    # Clean up GPU memory
    del model_a, params_a
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 4. Load the modified state dictionary back into model B's architecture
    print("\nLoading normalized state dictionary into the model...")
    model_b.load_state_dict(params_b)

    # 5. Save the newly created model and its tokenizer
    print(f"Saving normalized model to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    model_b.save_pretrained(output_dir)
    tokenizer_b.save_pretrained(output_dir)

    print("\n✅ Normalization complete. New model saved successfully!")


if __name__ == '__main__':
    try:
        normalize_model_parameters(MODEL_A_NAME, MODEL_B_NAME, OUTPUT_DIR)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()