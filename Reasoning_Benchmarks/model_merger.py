# 文件名: model_merger.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from tqdm import tqdm
import gc
from collections import defaultdict
from typing import List,Optional

# Set the visible CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def save_model_and_tokenizer(model, tokenizer_name, output_path):
    """Saves the model and its corresponding tokenizer to the specified path."""
    print(f"\nEnsuring output directory exists: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    print(f"Saving the merged model to: {output_path}")
    model.save_pretrained(output_path)
    print(f"Loading and saving the tokenizer from {tokenizer_name} to: {output_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        tokenizer.save_pretrained(output_path)
    except Exception as e:
        print(f"Could not save tokenizer. Error: {e}")
    print(f"Process for '{output_path}' completed successfully!")
def extract_magnitude_components(tensor, sparsity_threshold, component_ratio):
    """
    基于幅度提取稀疏组件
    """
    # 计算幅度
    magnitudes = torch.abs(tensor)
    
    # 计算阈值
    num_params = tensor.numel()
    num_to_keep = int(num_params * component_ratio)
    
    if num_to_keep > 0:
        # 找到阈值
        threshold = torch.kthvalue(magnitudes.flatten(), num_params - num_to_keep + 1).values
        
        # 创建掩码
        mask = magnitudes >= threshold
        
        # 应用稀疏性阈值
        mask = mask & (magnitudes >= sparsity_threshold)
        
        return tensor * mask
    else:
        return torch.zeros_like(tensor)

def extract_gradient_components(tensor, sparsity_threshold, component_ratio):
    """
    基于梯度信息提取稀疏组件
    """
    # 计算梯度的近似（使用数值差分）
    epsilon = 1e-6
    gradient_approx = torch.abs(tensor)
    
    # 应用幅度提取逻辑
    return extract_magnitude_components(gradient_approx, sparsity_threshold, component_ratio)

def extract_activation_components(tensor, sparsity_threshold, component_ratio):
    """
    基于激活模式提取稀疏组件
    """
    # 计算激活强度
    activation_strength = torch.abs(tensor)
    
    # 应用非线性变换模拟激活
    activation_strength = torch.tanh(activation_strength)
    
    # 应用幅度提取逻辑
    return extract_magnitude_components(activation_strength, sparsity_threshold, component_ratio)

def sce_merge_models(base_model_name, model_names_list, sparsity_threshold, scaling_lambda,  output_path, fusion_weights=None,
                     extraction_method="magnitude", component_ratio=0.1):
    """
    使用 SCE (Sparse Component Extraction) 方法合并多个模型。
    SCE 通过识别和提取模型中的稀疏组件来进行合并。
    
    Args:
        base_model_name (str): 基础预训练模型的路径。
        model_names_list (list[str]): 需要合并的多个微调模型的路径列表。
        sparsity_threshold (float): 稀疏性阈值，用于识别稀疏组件。
        scaling_lambda (float): 任务算术的缩放系数 λ。
        output_path (str): 合并后模型的保存路径。
        extraction_method (str): 组件提取方法，支持 "magnitude", "gradient", "activation"。
        component_ratio (float): 保留的组件比例 (0.0 到 1.0)。
        fusion_weights (list[float], optional): 各模型的融合权重，如果为None则使用等权重。
    """
    print("-" * 80)
    print(f"Starting SCE-Merge with sparsity_threshold={sparsity_threshold} and scaling_lambda={scaling_lambda}")
    print(f"Extraction method: {extraction_method}, Component ratio: {component_ratio}")
    
    # --- 处理融合权重 ---
    if fusion_weights is None:
        # 使用等权重
        fusion_weights = [1.0 / len(model_names_list)] * len(model_names_list)
        print(f"Using equal weights: {fusion_weights}")
    else:
        # 验证权重
        if len(fusion_weights) != len(model_names_list):
            raise ValueError(f"fusion_weights length ({len(fusion_weights)}) must match model_names_list length ({len(model_names_list)})")
        
        # 归一化权重
        total_weight = sum(fusion_weights)
        fusion_weights = [w / total_weight for w in fusion_weights]
        print(f"Using custom weights: {fusion_weights}")
    
    # --- 1. 加载所有模型 ---
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    )
    base_model.eval()
    params_base = base_model.state_dict()
    
    fine_tuned_params_list = []
    for model_name in model_names_list:
        print(f"Loading fine-tuned model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
        )
        model.eval()
        fine_tuned_params_list.append(model.state_dict())
        del model
    gc.collect()
    
    # --- 2. 计算任务向量 ---
    print("Computing task vectors...")
    task_vectors = []
    param_names = list(params_base.keys())
    
    for i, ft_params in enumerate(fine_tuned_params_list):
        print(f"Computing task vector {i+1}/{len(fine_tuned_params_list)}")
        task_vector = {}
        
        for param_name in tqdm(param_names, desc=f"Computing TV {i+1}"):
            if param_name in ft_params:
                base_tensor = params_base[param_name]
                ft_tensor = ft_params[param_name]
                task_vector[param_name] = ft_tensor - base_tensor
        
        task_vectors.append(task_vector)
    
    # --- 3. 提取稀疏组件 ---
    print(f"Extracting sparse components using {extraction_method} method...")
    sparse_components = []
    
    for i, task_vector in enumerate(task_vectors):
        print(f"Extracting sparse components for model {i+1}/{len(task_vectors)}")
        sparse_component = {}
        
        for param_name in tqdm(param_names, desc=f"Extracting SC {i+1}"):
            if param_name in task_vector:
                raw_vector = task_vector[param_name]
                
                if raw_vector.ndim > 0:  # 跳过标量张量
                    # 根据提取方法处理
                    if extraction_method == "magnitude":
                        sparse_component[param_name] = extract_magnitude_components(
                            raw_vector, sparsity_threshold, component_ratio
                        )
                    elif extraction_method == "gradient":
                        sparse_component[param_name] = extract_gradient_components(
                            raw_vector, sparsity_threshold, component_ratio
                        )
                    elif extraction_method == "activation":
                        sparse_component[param_name] = extract_activation_components(
                            raw_vector, sparsity_threshold, component_ratio
                        )
                    else:
                        raise ValueError(f"Unsupported extraction method: {extraction_method}")
                else:
                    sparse_component[param_name] = raw_vector
        
        sparse_components.append(sparse_component)
    
    # --- 4. 合并稀疏组件 ---
    print("Merging sparse components with fusion weights...")
    merged_components = {}
    
    for param_name in tqdm(param_names, desc="Merging components"):
        if param_name in params_base:
            # 收集所有稀疏组件中该参数的值
            component_values = []
            for sc in sparse_components:
                if param_name in sc:
                    component_values.append(sc[param_name])
            
            if component_values:
                # 使用融合权重进行加权平均合并
                weighted_sum = torch.zeros_like(component_values[0])
                for i, component in enumerate(component_values):
                    weighted_sum += fusion_weights[i] * component
                merged_component = weighted_sum
                merged_components[param_name] = merged_component
    
    # --- 5. 构建最终模型 ---
    print("Building final merged model...")
    final_params = params_base.copy()
    
    for param_name in tqdm(param_names, desc="Building final model"):
        if param_name in merged_components:
            final_params[param_name] = params_base[param_name] + scaling_lambda * merged_components[param_name]
    
    # --- 6. 加载合并后的参数并保存模型 ---
    print("Loading merged parameters...")
    base_model.load_state_dict(final_params)
    
    # --- 7. 保存模型 ---
    save_model_and_tokenizer(base_model, base_model_name, output_path)
    
    # --- 8. 清理内存 ---
    del base_model, fine_tuned_params_list, task_vectors, sparse_components, merged_components, final_params
    gc.collect()
    torch.cuda.empty_cache()
    
def weighted_average_models(model_name_a, model_name_b, lambda_val, output_path):
    """Performs a weighted average: lambda * model_a + (1 - lambda) * model_b."""
    print("-" * 80)
    print(f"Starting weighted average with lambda = {lambda_val}")
    model_a = AutoModelForCausalLM.from_pretrained(model_name_a, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)
    model_a.eval()
    model_b = AutoModelForCausalLM.from_pretrained(model_name_b, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)
    model_b.eval()
    params_a = model_a.state_dict()
    params_b = model_b.state_dict()
    merged_params = {}
    for param_name in tqdm(params_a.keys(), desc=f"Averaging (lambda={lambda_val})"):
        if param_name in params_b:
            tensor_a = params_a[param_name]
            tensor_b = params_b[param_name].to(tensor_a.device, dtype=tensor_a.dtype)
            merged_params[param_name] = (lambda_val * tensor_a) + ((1 - lambda_val) * tensor_b)
        else:
            merged_params[param_name] = params_a[param_name]
    model_a.load_state_dict(merged_params)
    save_model_and_tokenizer(model_a, model_name_a, output_path)
    del model_a, model_b, params_a, params_b, merged_params
    gc.collect()
    torch.cuda.empty_cache()


def merge_and_amplify_top_k_diff_models(base_model_name, donor_model_name, percentage, output_path):
    """Merges and amplifies the top k% of differing parameters"""
    print("-" * 80)
    print(f"Starting SURGICAL MERGE WITH AMPLIFICATION for top {percentage}% parameters")
    model_a = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model_a.eval()
    model_b = AutoModelForCausalLM.from_pretrained(donor_model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model_b.eval()
    params_a = model_a.state_dict()
    params_b = model_b.state_dict()
    all_diffs_cpu = []
    total_params = 0
    for param_name in tqdm(params_a.keys(), desc="Calculating Diffs"):
        if param_name in params_b:
            tensor_a = params_a[param_name]
            total_params += tensor_a.numel()
            tensor_b = params_b[param_name].to(tensor_a.device, dtype=tensor_a.dtype)
            diff = torch.abs(tensor_b - tensor_a)
            all_diffs_cpu.append(diff.flatten().cpu())
    flat_all_diffs_cpu = torch.cat(all_diffs_cpu)
    num_to_merge = int(total_params * (percentage / 100.0))
    k_val_for_threshold = max(1, total_params - num_to_merge)
    threshold = torch.kthvalue(flat_all_diffs_cpu.to(torch.float32), k_val_for_threshold).values
    del flat_all_diffs_cpu, all_diffs_cpu
    gc.collect()
    alpha = 1.0 / (percentage / 100.0) if percentage > 0 else 1.0
    merged_params = model_a.state_dict()
    for param_name in tqdm(params_a.keys(), desc=f"Amplifying Top {percentage}%"):
        if param_name in params_b:
            tensor_a = params_a[param_name]
            tensor_b = params_b[param_name].to(tensor_a.device, dtype=tensor_a.dtype)
            diff_mask = torch.abs(tensor_b - tensor_a) >= threshold
            amplified_update = tensor_a + alpha * (tensor_b - tensor_a)
            merged_params[param_name] = torch.where(diff_mask, amplified_update, tensor_a)
    model_a.load_state_dict(merged_params)
    save_model_and_tokenizer(model_a, base_model_name, output_path)
    del model_a, model_b, params_a, params_b, merged_params
    gc.collect()
    torch.cuda.empty_cache()


def merge_top_k_diff_models(base_model_name, donor_model_name, percentage, output_path):
    """Merges the top k% of differing parameters from a donor model into a base model. 这个是surgical merge"""
    print("-" * 80)
    print(f"Starting surgical merge of top {percentage}% differing parameters")
    model_a = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)
    model_a.eval()
    model_b = AutoModelForCausalLM.from_pretrained(donor_model_name, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)
    model_b.eval()
    params_a = model_a.state_dict()
    params_b = model_b.state_dict()
    all_diffs_cpu = []
    total_params = 0
    for param_name in tqdm(params_a.keys(), desc="Calculating Diffs"):
        if param_name in params_b:
            tensor_a = params_a[param_name]
            total_params += tensor_a.numel()
            tensor_b = params_b[param_name].to(tensor_a.device, dtype=tensor_a.dtype)
            diff = torch.abs(tensor_b - tensor_a)
            all_diffs_cpu.append(diff.flatten().cpu())
    flat_all_diffs_cpu = torch.cat(all_diffs_cpu)
    num_to_merge = int(total_params * (percentage / 100.0))
    k_val_for_threshold = max(1, total_params - num_to_merge)
    threshold = torch.kthvalue(flat_all_diffs_cpu.to(torch.float32), k_val_for_threshold).values
    del flat_all_diffs_cpu, all_diffs_cpu
    gc.collect()
    merged_params = model_a.state_dict()
    for param_name in tqdm(params_a.keys(), desc=f"Merging Top {percentage}%"):
        if param_name in params_b:
            tensor_a = params_a[param_name]
            tensor_b = params_b[param_name].to(tensor_a.device, dtype=tensor_a.dtype)
            diff_mask = torch.abs(tensor_b - tensor_a) >= threshold
            merged_params[param_name] = torch.where(diff_mask, tensor_b, tensor_a)
    model_a.load_state_dict(merged_params)
    save_model_and_tokenizer(model_a, base_model_name, output_path)
    del model_a, model_b, params_a, params_b, merged_params
    gc.collect()
    torch.cuda.empty_cache()
    
def merge_top_k_avg_keep_base(base_model_name, instruct_model_name, percentage, output_path):
    """
    1. k%差异最大的参数取平均，其他的参数保留推理模型的参数。
    Averages the top k% most differing parameters, keeps the rest from the base (inference) model.
    """
    print("-" * 80)
    print(f"Starting merge: Top {percentage}% Avg, Keep Base")
    # Load models
    model_base = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)
    model_base.eval()
    model_instruct = AutoModelForCausalLM.from_pretrained(instruct_model_name, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)
    model_instruct.eval()
    
    params_base = model_base.state_dict()
    params_instruct = model_instruct.state_dict()
    
    # Calculate differences and find threshold for top k%
    all_diffs_cpu = []
    total_params = 0
    for param_name in tqdm(params_base.keys(), desc="Calculating Diffs"):
        if param_name in params_instruct:
            tensor_base = params_base[param_name]
            total_params += tensor_base.numel()
            tensor_instruct = params_instruct[param_name].to(tensor_base.device, dtype=tensor_base.dtype)
            diff = torch.abs(tensor_instruct - tensor_base)
            all_diffs_cpu.append(diff.flatten().cpu())
            
    flat_all_diffs_cpu = torch.cat(all_diffs_cpu)
    num_to_merge = int(total_params * (percentage / 100.0))
    k_val_for_threshold = min(num_to_merge, total_params)  # 第 k 小的值
    threshold = torch.kthvalue(flat_all_diffs_cpu.to(torch.float32), k_val_for_threshold).values
    del flat_all_diffs_cpu, all_diffs_cpu
    gc.collect()
    
    # Apply merging logic
    merged_params = model_base.state_dict()
    for param_name in tqdm(params_base.keys(), desc=f"Merging Top {percentage}% Avg"):
        if param_name in params_instruct:
            tensor_base = params_base[param_name]
            tensor_instruct = params_instruct[param_name].to(tensor_base.device, dtype=tensor_base.dtype)
            
            diff_mask = torch.abs(tensor_instruct - tensor_base) <= threshold
            avg_tensor = (tensor_base + tensor_instruct) / 2.0
            
            # Where mask is True (top k%), use the average; otherwise, keep the base model's parameter.
            merged_params[param_name] = torch.where(diff_mask, avg_tensor, tensor_base)
            
    # Load merged state and save
    model_base.load_state_dict(merged_params)
    save_model_and_tokenizer(model_base, base_model_name, output_path)
    
    # Cleanup
    del model_base, model_instruct, params_base, params_instruct, merged_params
    gc.collect()
    torch.cuda.empty_cache()

def merge_top_k_avg_keep_base_min_diff(base_model_name, instruct_model_name, percentage, output_path):
    """
    对差异最小的 k% 参数取平均，其余参数保留 base 模型。
    Averages the top k% LEAST differing parameters, keeps the rest from the base model.
    """
    print("-" * 80)
    print(f"Starting merge: Top {percentage}% Min-Diff Avg, Keep Base")
    model_base = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model_base.eval()
    model_instruct = AutoModelForCausalLM.from_pretrained(instruct_model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model_instruct.eval()
    
    params_base = model_base.state_dict()
    params_instruct = model_instruct.state_dict()
    
    # Calculate absolute differences
    all_diffs_cpu = []
    total_params = 0
    for param_name in tqdm(params_base.keys(), desc="Calculating Diffs"):
        if param_name in params_instruct:
            tensor_base = params_base[param_name]
            total_params += tensor_base.numel()
            tensor_instruct = params_instruct[param_name].to(tensor_base.device, dtype=tensor_base.dtype)
            diff = torch.abs(tensor_instruct - tensor_base)
            all_diffs_cpu.append(diff.flatten().cpu())
            
    flat_all_diffs_cpu = torch.cat(all_diffs_cpu)
    num_to_merge = int(total_params * (percentage / 100.0))
    k_val_for_threshold = min(num_to_merge, total_params)  # 第 k 小的值
    threshold = torch.kthvalue(flat_all_diffs_cpu.to(torch.float32), k_val_for_threshold).values
    del flat_all_diffs_cpu, all_diffs_cpu
    gc.collect()
    
    # Apply merging: avg for min-diff params, keep base for others
    merged_params = model_base.state_dict()
    for param_name in tqdm(params_base.keys(), desc=f"Merging Top {percentage}% Min-Diff Avg"):
        if param_name in params_instruct:
            tensor_base = params_base[param_name]
            tensor_instruct = params_instruct[param_name].to(tensor_base.device, dtype=tensor_base.dtype)
            diff_mask = torch.abs(tensor_instruct - tensor_base) <= threshold  # 最小差异区域
            avg_tensor = (tensor_base + tensor_instruct) / 2.0
            merged_params[param_name] = torch.where(diff_mask, avg_tensor, tensor_base)
            
    model_base.load_state_dict(merged_params)
    save_model_and_tokenizer(model_base, base_model_name, output_path)
    
    del model_base, model_instruct, params_base, params_instruct, merged_params
    gc.collect()
    torch.cuda.empty_cache()

def merge_avg_override_top_k_base(base_model_name, instruct_model_name, percentage, output_path):
    """
    3. 所有参数都取平均，但差异最大的k%参数换成推理模型的参数。
    Averages all parameters, but then replaces the top k% most differing parameters with the base (inference) model's original parameters.
    """
    print("-" * 80)
    print(f"Starting merge: Average All, Override Top {percentage}% with Base")
    # Load models
    model_base = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)
    model_base.eval()
    model_instruct = AutoModelForCausalLM.from_pretrained(instruct_model_name, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)
    model_instruct.eval()

    params_base = model_base.state_dict()
    params_instruct = model_instruct.state_dict()

    # Calculate differences and find threshold for top k%
    all_diffs_cpu = []
    total_params = 0
    for param_name in tqdm(params_base.keys(), desc="Calculating Diffs"):
        if param_name in params_instruct:
            tensor_base = params_base[param_name]
            total_params += tensor_base.numel()
            tensor_instruct = params_instruct[param_name].to(tensor_base.device, dtype=tensor_base.dtype)
            diff = torch.abs(tensor_instruct - tensor_base)
            all_diffs_cpu.append(diff.flatten().cpu())

    flat_all_diffs_cpu = torch.cat(all_diffs_cpu)
    num_to_merge = int(total_params * (percentage / 100.0))
    k_val_for_threshold = max(1, total_params - num_to_merge)
    threshold = torch.kthvalue(flat_all_diffs_cpu.to(torch.float32), k_val_for_threshold).values
    del flat_all_diffs_cpu, all_diffs_cpu
    gc.collect()

    # Apply merging logic
    merged_params = model_base.state_dict()
    for param_name in tqdm(params_base.keys(), desc=f"Averaging and Overriding Top {percentage}%"):
        if param_name in params_instruct:
            tensor_base = params_base[param_name]
            tensor_instruct = params_instruct[param_name].to(tensor_base.device, dtype=tensor_base.dtype)
            
            diff_mask = torch.abs(tensor_instruct - tensor_base) >= threshold
            avg_tensor = (tensor_base + tensor_instruct) / 2.0
            
            # Where mask is True (top k%), override the average with the base model's parameter; otherwise, keep the average.
            merged_params[param_name] = torch.where(diff_mask, tensor_base, avg_tensor)

    # Load merged state and save
    model_base.load_state_dict(merged_params)
    save_model_and_tokenizer(model_base, base_model_name, output_path)

    # Cleanup
    del model_base, model_instruct, params_base, params_instruct, merged_params
    gc.collect()
    torch.cuda.empty_cache()

def merge_bottom_k_avg_keep_instruct(base_model_name, instruct_model_name, percentage, output_path):
    """
    k%差异最min的参数取平均，其他的参数保留Instruct模型的参数。
    Averages the bottom k% most differing parameters, keeps the rest from the Instruct model.
    """
    print("-" * 80)
    print(f"Starting merge: Top {percentage}% Avg, Keep Instruct")
    # Load models
    model_base = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model_base.eval()
    model_instruct = AutoModelForCausalLM.from_pretrained(instruct_model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model_instruct.eval()

    params_base = model_base.state_dict()
    params_instruct = model_instruct.state_dict()

    # Calculate differences and find threshold for top k%
    all_diffs_cpu = []
    total_params = 0
    for param_name in tqdm(params_base.keys(), desc="Calculating Diffs"):
        if param_name in params_instruct:
            tensor_base = params_base[param_name]
            total_params += tensor_base.numel()
            tensor_instruct = params_instruct[param_name].to(tensor_base.device, dtype=tensor_base.dtype)
            diff = torch.abs(tensor_instruct - tensor_base)
            all_diffs_cpu.append(diff.flatten().cpu())

    flat_all_diffs_cpu = torch.cat(all_diffs_cpu)
    num_to_merge = int(total_params * (percentage / 100.0))
    k_val_for_threshold = min(num_to_merge, total_params)  # 第 k 小的值
    threshold = torch.kthvalue(flat_all_diffs_cpu.to(torch.float32), k_val_for_threshold).values
    del flat_all_diffs_cpu, all_diffs_cpu
    gc.collect()
    
    # Apply merging logic
    merged_params = model_base.state_dict()
    for param_name in tqdm(params_base.keys(), desc=f"Merging Top {percentage}% Avg"):
        if param_name in params_instruct:
            tensor_base = params_base[param_name]
            tensor_instruct = params_instruct[param_name].to(tensor_base.device, dtype=tensor_base.dtype)
            
            diff_mask = torch.abs(tensor_instruct - tensor_base) <= threshold
            avg_tensor = (tensor_base + tensor_instruct) / 2.0
            
            # Where mask is True (top k%), use the average; otherwise, keep the instruct model's parameter.
            merged_params[param_name] = torch.where(diff_mask, avg_tensor, tensor_instruct)

    # Load merged state and save
    model_base.load_state_dict(merged_params)
    # We use the base model's tokenizer as is standard practice
    save_model_and_tokenizer(model_base, base_model_name, output_path)

    # Cleanup
    del model_base, model_instruct, params_base, params_instruct, merged_params
    gc.collect()
    torch.cuda.empty_cache()

def dare_merge_models_TA(base_model_name, model_a_name, model_b_name, drop_rate, scaling_lambda, output_path):
    """
    使用 DARE (Drop And REscale) 和任务算术 (Task Arithmetic) 方法合并两个模型。
    公式: Merged = Base + λ * (DARE(Model_A - Base) + (1 - λ) * (DARE(Model_B - Base)

    Args:
        base_model_name (str): 基础预训练模型的路径。
        model_a_name (str): 第一个微调模型的路径。
        model_b_name (str): 第二个微调模型的路径。
        drop_rate (float): DARE 的丢弃率 p (0.0 到 1.0)。例如，0.9 表示随机丢弃90%的增量参数。
        scaling_lambda (float): 任务算术的缩放系数 λ。
        output_path (str): 合并后模型的保存路径。
    """
    print("-" * 80)
    print(f"Starting DARE merge with drop_rate={drop_rate} and scaling_lambda={scaling_lambda}")

    # --- 1. 加载所有模型 ---
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    base_model.eval()
    print("Loading model A...")
    model_a = AutoModelForCausalLM.from_pretrained(model_a_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model_a.eval()
    print("Loading model B...")
    model_b = AutoModelForCausalLM.from_pretrained(model_b_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model_b.eval()

    params_base = base_model.state_dict()
    params_a = model_a.state_dict()
    params_b = model_b.state_dict()
    
    merged_params = {}

    # --- 2. 遍历参数，应用 DARE 和任务算术 ---
    for param_name in tqdm(params_base.keys(), desc="Applying DARE and Merging"):
        if param_name in params_a and param_name in params_b:
            base_tensor = params_base[param_name].to(torch.device("cuda"))
            tensor_a = params_a[param_name].to(base_tensor.device, dtype=base_tensor.dtype)
            tensor_b = params_b[param_name].to(base_tensor.device, dtype=base_tensor.dtype)

            # --- 计算增量参数 (Deltas) ---
            delta_a = tensor_a - base_tensor
            delta_b = tensor_b - base_tensor

            # --- 对每个 Delta 应用 DARE ---
            # DARE for Delta A
            mask_a = torch.rand_like(delta_a) > drop_rate
            dare_delta_a = (delta_a * mask_a) / (1 - drop_rate)
            
            # DARE for Delta B
            mask_b = torch.rand_like(delta_b) > drop_rate
            dare_delta_b = (delta_b * mask_b) / (1 - drop_rate)

            # --- 使用任务算术合并处理后的 Deltas ---
            merged_delta = scaling_lambda * dare_delta_a + (1 - scaling_lambda) * dare_delta_b
            
            # --- 将合并后的 Delta 应用于基础模型 ---
            merged_params[param_name] = base_tensor + merged_delta
    #   else:
    #       # 如果某个参数不存在于某个模型中，则直接使用基础模型的参数
    #       merged_params[param_name] = params_base[param_name]

    # --- 3. 加载合并后的参数并保存模型 ---
    base_model.load_state_dict(merged_params)
    # Tokenizer 通常使用基础模型或其中一个微调模型的即可
    save_model_and_tokenizer(base_model, base_model_name, output_path)

    # --- 4. 清理内存 ---
    del base_model, model_a, model_b, params_base, params_a, params_b, merged_params
    gc.collect()
    torch.cuda.empty_cache()


def dare_merge_models(base_model_name, model_a_name, model_b_name, drop_rate, scaling_lambda, output_path):
    """
    使用 DARE (Drop And REscale) 和任务算术 (Task Arithmetic) 方法合并两个模型。
    公式: Merged = Base + λ * (DARE(Model_A - Base) + λ * (DARE(Model_B - Base)

    Args:
        base_model_name (str): 基础预训练模型的路径。
        model_a_name (str): 第一个微调模型的路径。
        model_b_name (str): 第二个微调模型的路径。
        drop_rate (float): DARE 的丢弃率 p (0.0 到 1.0)。例如，0.9 表示随机丢弃90%的增量参数。
        scaling_lambda (float): 任务算术的缩放系数 λ。
        output_path (str): 合并后模型的保存路径。
    """
    print("-" * 80)
    print(f"Starting DARE merge with drop_rate={drop_rate} and scaling_lambda={scaling_lambda}")

    # --- 1. 加载所有模型 ---
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    base_model.eval()
    print("Loading model A...")
    model_a = AutoModelForCausalLM.from_pretrained(model_a_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model_a.eval()
    print("Loading model B...")
    model_b = AutoModelForCausalLM.from_pretrained(model_b_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model_b.eval()

    params_base = base_model.state_dict()
    params_a = model_a.state_dict()
    params_b = model_b.state_dict()
    
    merged_params = {}

    # --- 2. 遍历参数，应用 DARE 和任务算术 ---
    for param_name in tqdm(params_base.keys(), desc="Applying DARE and Merging"):
        if param_name in params_a and param_name in params_b:
            base_tensor = params_base[param_name].to(torch.device("cuda"))
            tensor_a = params_a[param_name].to(base_tensor.device, dtype=base_tensor.dtype)
            tensor_b = params_b[param_name].to(base_tensor.device, dtype=base_tensor.dtype)

            # --- 计算增量参数 (Deltas) ---
            delta_a = tensor_a - base_tensor
            delta_b = tensor_b - base_tensor

            # --- 对每个 Delta 应用 DARE ---
            # DARE for Delta A
            mask_a = torch.rand_like(delta_a) > drop_rate
            dare_delta_a = (delta_a * mask_a) / (1 - drop_rate)
            
            # DARE for Delta B
            mask_b = torch.rand_like(delta_b) > drop_rate
            dare_delta_b = (delta_b * mask_b) / (1 - drop_rate)

            # --- 使用任务算术合并处理后的 Deltas ---
            merged_delta = scaling_lambda * dare_delta_a + scaling_lambda * dare_delta_b
            
            # --- 将合并后的 Delta 应用于基础模型 ---
            merged_params[param_name] = base_tensor + merged_delta
    #   else:
    #       # 如果某个参数不存在于某个模型中，则直接使用基础模型的参数
    #       merged_params[param_name] = params_base[param_name]

    # --- 3. 加载合并后的参数并保存模型 ---
    base_model.load_state_dict(merged_params)
    # Tokenizer 通常使用基础模型或其中一个微调模型的即可
    save_model_and_tokenizer(base_model, base_model_name, output_path)

    # --- 4. 清理内存 ---
    del base_model, model_a, model_b, params_base, params_a, params_b, merged_params
    gc.collect()
    torch.cuda.empty_cache()

def ties_merge_models(base_model_name, model_names_list, top_k_percentage, scaling_lambda, output_path):
    """
    使用内存优化的 TIES-Merging (Trim, Elect Sign & Merge) 方法合并多个模型。
    此版本通过逐个参数处理来避免创建巨大的中间张量，从而解决内存溢出（OOM）问题。
    注意：此实现是 TIES 论文算法的一种近似，它在每个参数张量上进行局部修剪（local trimming），
    而非全局修剪（global trimming），但这对于解决内存瓶颈至关重要。

    Args:
        base_model_name (str): 基础预训练模型的路径。
        model_names_list (list[str]): 需要合并的多个微调模型的路径列表。
        top_k_percentage (float): Trim 步骤中保留的参数百分比 (0-100)。它将应用于每个参数张量。
        scaling_lambda (float): 最终合并任务向量的缩放系数 λ。
        output_path (str): 合并后模型的保存路径。
    """
    print("-" * 80)
    print(f"Starting Memory-Optimized TIES-Merging with top_k={top_k_percentage}% and scaling_lambda={scaling_lambda}")

    # --- 1. 加载所有模型 ---
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    )
    base_model.eval()
    params_base = base_model.state_dict()

    fine_tuned_params_list = []
    for model_name in model_names_list:
        print(f"Loading fine-tuned model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
        )
        model.eval()
        fine_tuned_params_list.append(model.state_dict())
        del model
    gc.collect()

    final_merged_task_vector = {}
    param_names = list(params_base.keys())
    
    # --- 逐个参数进行 TIES 合并 ---
    for param_name in tqdm(param_names, desc="TIES-Merging (per-parameter)"):
        base_tensor = params_base[param_name]
        
        # --- 计算当前参数的任务向量 ---
        tvs_for_param = [ft_params[param_name] - base_tensor for ft_params in fine_tuned_params_list]

        # --- Step 1: Trim (Local Approximation) ---
        trimmed_tvs_for_param = []
        for tensor in tvs_for_param:
            if tensor.ndim == 0:  # Skip scalar tensors
                trimmed_tvs_for_param.append(tensor)
                continue

            num_params_to_trim = int(tensor.numel() * (1 - top_k_percentage / 100.0))
            if num_params_to_trim > 0:
                # 找到局部阈值
                threshold = torch.kthvalue(torch.abs(tensor).flatten(), num_params_to_trim).values
                mask = torch.abs(tensor) >= threshold
                trimmed_tvs_for_param.append(tensor * mask)
            else:
                trimmed_tvs_for_param.append(tensor)
        
        # --- Step 2: Elect Sign ---
        sum_of_trimmed_tvs = torch.sum(torch.stack(trimmed_tvs_for_param), dim=0)
        elected_sign_tensor = torch.sign(sum_of_trimmed_tvs)

        # --- Step 3: Disjoint Merge ---
        final_sum = torch.zeros_like(base_tensor)
        final_counts = torch.zeros_like(base_tensor)

        for trimmed_tensor in trimmed_tvs_for_param:
            agreement_mask = (torch.sign(trimmed_tensor) == elected_sign_tensor).float()
            final_sum += trimmed_tensor * agreement_mask
            final_counts += agreement_mask

        # 避免除以零
        merged_tensor = torch.where(
            final_counts > 0,
            final_sum / final_counts,
            0.0
        )
        final_merged_task_vector[param_name] = merged_tensor

    # 清理内存
    del fine_tuned_params_list, tvs_for_param, trimmed_tvs_for_param
    gc.collect()

    # --- 将合并后的任务向量应用到基础模型 ---
    print("Applying merged task vector to the base model...")
    merged_params = base_model.state_dict()
    for param_name, tensor in final_merged_task_vector.items():
        merged_params[param_name] += (scaling_lambda * tensor).to(merged_params[param_name].device)

    base_model.load_state_dict(merged_params)
    
    # --- 保存模型 ---
    save_model_and_tokenizer(base_model, base_model_name, output_path)

    del base_model
    gc.collect()
    torch.cuda.empty_cache()


# def ties_merge_models_TA(base_model_name, model_names_list, top_k_percentage, scaling_lambda, output_path):
#     """
#     【混合版本】
#     - 当 model_names_list 长度为 2 时：执行凸组合（Convex Combination）
#         Final = λ * Model1 + (1-λ) * Model2
#     - 当长度 > 2 时：执行非标准 TIES（先乘 λ，再投票）

#     Args:
#         base_model_name (str): 基础预训练模型的路径。
#         model_names_list (list[str]): 需要合并的多个微调模型的路径列表。
#         top_k_percentage (float): Trim 步骤中保留的参数百分比 (0-100)。全局应用。
#         scaling_lambda (float): 缩放系数 λ（在凸组合中表示 Model1 的权重）。
#         output_path (str): 合并后模型的保存路径。
#     """
#     print("-" * 80)
#     n_models = len(model_names_list)
#     if n_models == 2:
#         print(f"Starting CONVEX COMBINATION for 2 models with weight_λ={scaling_lambda}")
#     else:
#         print(f"Starting TIES-Merging (NON-STANDARD: λ FIRST) with GLOBAL top_k={top_k_percentage}% and scaling_lambda={scaling_lambda}")

#     # --- 1. 加载所有模型 ---
#     print("Loading base model...")
#     base_model = AutoModelForCausalLM.from_pretrained(
#         base_model_name, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
#     )
#     base_model.eval()
#     params_base = base_model.state_dict()

#     fine_tuned_params_list = []
#     for model_name in model_names_list:
#         print(f"Loading fine-tuned model: {model_name}")
#         model = AutoModelForCausalLM.from_pretrained(
#             model_name, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
#         )
#         model.eval()
#         fine_tuned_params_list.append(model.state_dict())
#         del model
#     gc.collect()

#     # --- 2. 如果是 2 个模型，直接凸组合 ---
# # --- 2. 如果是 2 个模型，执行带 Top-K 修剪的凸组合 ---
#     if n_models == 2:
#         print(f"Computing convex combination with Top-{top_k_percentage}% pruning...")
        
#         # --- 2.1 计算全局阈值 ---
#         print("Computing global threshold for Top-K pruning...")
#         all_tvs_abs_cpu = []
#         total_params = 0
#         param_names = list(params_base.keys())
        
#         for param_name in tqdm(param_names, desc="Collecting magnitudes for threshold", leave=False):
#             base_tensor = params_base[param_name]
#             tv1 = fine_tuned_params_list[0][param_name] - base_tensor
#             total_params += tv1.numel()
#             all_tvs_abs_cpu.append(torch.abs(tv1).flatten().cpu())
        
#         flat_all_tvs_abs_cpu = torch.cat(all_tvs_abs_cpu)
#         num_to_keep = int(total_params * (top_k_percentage / 100.0))
#         k_val_for_threshold = max(1, total_params - num_to_keep)
#         global_threshold = torch.kthvalue(flat_all_tvs_abs_cpu.to(torch.float32), k_val_for_threshold).values
        
#         del flat_all_tvs_abs_cpu, all_tvs_abs_cpu
#         gc.collect()
        
#         # --- 2.2 执行凸组合 ---
#         print("Applying convex combination with pruning...")
#         final_merged_task_vector = {}
        
#         for param_name in tqdm(param_names, desc="Convex Combination with Pruning"):
#             base_tensor = params_base[param_name]
#             tv1 = fine_tuned_params_list[0][param_name] - base_tensor
#             tv2 = fine_tuned_params_list[1][param_name] - base_tensor
            
#             # 对两个任务向量分别应用 Top-K 修剪
#             mask1 = torch.abs(tv1) >= global_threshold
#             mask2 = torch.abs(tv2) >= global_threshold
#             tv1_pruned = tv1 * mask1
#             tv2_pruned = tv2 * mask2
            
#             # 凸组合：λ * TV1 + (1-λ) * TV2
#             merged_tv = scaling_lambda * tv1_pruned + (1.0 - scaling_lambda) * tv2_pruned
#             final_merged_task_vector[param_name] = merged_tv
    
#         del fine_tuned_params_list
#         gc.collect()

#     # --- 3. 如果 >2 个模型，执行原“先乘 λ 再投票”逻辑 ---
#     else:
#         # --- 3.1 全局计算任务向量绝对值，用于 Global Trim 阈值计算 ---
#         print("Computing global task vector magnitudes for global top-k threshold...")
#         all_tvs_abs_cpu = []
#         total_params = 0

#         param_names = list(params_base.keys())
#         for param_name in tqdm(param_names, desc="Collecting TV magnitudes"):
#             base_tensor = params_base[param_name]
#             if len(fine_tuned_params_list) == 0:
#                 continue
#             tv_tensor = fine_tuned_params_list[0][param_name] - base_tensor
#             total_params += tv_tensor.numel()
#             all_tvs_abs_cpu.append(torch.abs(tv_tensor).flatten().cpu())

#         flat_all_tvs_abs_cpu = torch.cat(all_tvs_abs_cpu)
#         num_to_keep = int(total_params * (top_k_percentage / 100.0))
#         k_val_for_threshold = max(1, total_params - num_to_keep)
#         global_threshold = torch.kthvalue(flat_all_tvs_abs_cpu.to(torch.float32), k_val_for_threshold).values

#         del flat_all_tvs_abs_cpu, all_tvs_abs_cpu
#         gc.collect()

#         # --- 3.2 逐个参数进行 TIES 合并 ---
#         final_merged_task_vector = {}
#         for param_name in tqdm(param_names, desc="TIES-Merging with Global Trim (λ FIRST)"):
#             base_tensor = params_base[param_name]
#             tvs_for_param = [ft_params[param_name] - base_tensor for ft_params in fine_tuned_params_list]

#             # Trim
#             trimmed_tvs_for_param = []
#             for tensor in tvs_for_param:
#                 if tensor.ndim == 0:
#                     trimmed_tvs_for_param.append(tensor)
#                     continue
#                 mask = torch.abs(tensor) >= global_threshold
#                 trimmed_tvs_for_param.append(tensor * mask)
            
#             # 先乘 lambda
#             scaled_tvs_for_param = [tv * scaling_lambda for tv in trimmed_tvs_for_param]

#             # Elect Sign (on SCALED vectors)
#             sum_of_scaled_tvs = torch.sum(torch.stack(scaled_tvs_for_param), dim=0)
#             elected_sign_tensor = torch.sign(sum_of_scaled_tvs)

#             # Disjoint Merge
#             final_sum = torch.zeros_like(base_tensor)
#             final_counts = torch.zeros_like(base_tensor)
#             for scaled_tensor in scaled_tvs_for_param:
#                 agreement_mask = (torch.sign(scaled_tensor) == elected_sign_tensor).float()
#                 final_sum += scaled_tensor * agreement_mask
#                 final_counts += agreement_mask

#             merged_tensor = torch.where(final_counts > 0, final_sum / final_counts, 0.0)
#             final_merged_task_vector[param_name] = merged_tensor

#         del fine_tuned_params_list, tvs_for_param, trimmed_tvs_for_param, scaled_tvs_for_param
#         gc.collect()

#     # --- 4. 将合并后的任务向量应用到基础模型 ---
#     print("Applying merged task vector to the base model...")
#     merged_params = base_model.state_dict()
#     for param_name, tensor in final_merged_task_vector.items():
#         merged_params[param_name] += tensor.to(merged_params[param_name].device)

#     base_model.load_state_dict(merged_params)
#     save_model_and_tokenizer(base_model, base_model_name, output_path)

#     # --- 5. 清理内存 ---
#     del base_model, final_merged_task_vector, merged_params
#     gc.collect()
#     torch.cuda.empty_cache()

#     if n_models == 2:
#         print("✅ Convex Combination completed successfully!")
#     else:
#         print("⚠️  WARNING: This is a NON-STANDARD version (λ applied BEFORE sign voting). Use for experimental purposes only.")

def twin_merge_models(base_model_name, model_names_list, mask_rate, scaling_lambda, output_path, router_model_path=None):
    """
    使用 Twin-Merging 方法合并多个模型。
    Twin-Merging 包含两个阶段：
    1. 模块化知识分解：将专家模型知识分解为共享和专属组件
    2. 动态知识融合：根据输入动态融合共享和任务特定知识
    
    Args:
        base_model_name (str): 基础预训练模型的路径。
        model_names_list (list[str]): 需要合并的多个微调模型的路径列表。
        mask_rate (float): Twin Vector 的稀疏化率 (0.0 到 1.0)。例如，0.8 表示保留20%的参数。
        scaling_lambda (float): 任务算术的缩放系数 λ。
        output_path (str): 合并后模型的保存路径。
        router_model_path (str, optional): 路由器模型的路径，如果为None则使用简单平均。
    """
    print("-" * 80)
    print(f"Starting Twin-Merging with mask_rate={mask_rate} and scaling_lambda={scaling_lambda}")
    
    # --- 1. 加载所有模型 ---
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    )
    base_model.eval()
    params_base = base_model.state_dict()
    
    fine_tuned_params_list = []
    for model_name in model_names_list:
        print(f"Loading fine-tuned model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
        )
        model.eval()
        fine_tuned_params_list.append(model.state_dict())
        del model
    gc.collect()
    
    # --- 2. 第一阶段：计算共享模型（使用简单平均，不乘 scaling_lambda） ---
    print("Computing shared model using Simple Average...")
    shared_params = {}
    param_names = list(params_base.keys())
    
    for param_name in tqdm(param_names, desc="Computing Shared Model"):
        base_tensor = params_base[param_name]
        
        # 计算任务向量
        task_vectors = []
        for ft_params in fine_tuned_params_list:
            if param_name in ft_params:
                task_vector = ft_params[param_name] - base_tensor
                task_vectors.append(task_vector)
        
        if task_vectors:
            # 简单平均（不乘 scaling_lambda，留给第三阶段控制）
            merged_task_vector = sum(task_vectors) / len(task_vectors)
            shared_params[param_name] = base_tensor + merged_task_vector  # 👈 不乘 lambda
        else:
            shared_params[param_name] = base_tensor
    
    # --- 3. 第二阶段：提取Twin Vectors ---
    print("Extracting Twin Vectors...")
    twin_vectors = []
    
    for i, ft_params in enumerate(fine_tuned_params_list):
        print(f"Processing Twin Vector {i+1}/{len(fine_tuned_params_list)}")
        twin_vector = {}
        
        for param_name in tqdm(param_names, desc=f"Extracting TV {i+1}"):
            if param_name in ft_params and param_name in shared_params:
                base_tensor = params_base[param_name]
                ft_tensor = ft_params[param_name]
                shared_tensor = shared_params[param_name]
                
                # 计算Twin Vector: θ^t - θ*
                raw_twin_vector = ft_tensor - shared_tensor
                
                # 应用稀疏化（magnitude-based）
                if raw_twin_vector.ndim > 0:  # 跳过标量张量
                    # 计算要保留的参数数量
                    num_params = raw_twin_vector.numel()
                    num_to_keep = int(num_params * (1 - mask_rate))
                    
                    if num_to_keep > 0:
                        # 找到阈值
                        abs_values = torch.abs(raw_twin_vector).flatten()
                        threshold = torch.kthvalue(abs_values, num_params - num_to_keep + 1).values
                        
                        # 创建掩码
                        mask = torch.abs(raw_twin_vector) >= threshold
                        twin_vector[param_name] = raw_twin_vector * mask
                    else:
                        twin_vector[param_name] = torch.zeros_like(raw_twin_vector)
                else:
                    twin_vector[param_name] = raw_twin_vector
        
        twin_vectors.append(twin_vector)
    # --- 4. 第三阶段：动态融合（支持动态比例） ---
    print("Computing final merged model...")
    final_params = shared_params.copy()
    
    n_models = len(model_names_list)
    
    if router_model_path is None:
        print("Using dynamic weighting for Twin Vector combination...")
        for param_name in tqdm(param_names, desc="Combining Twin Vectors"):
            if param_name not in shared_params:
                continue

            tv_values = []
            for tv in twin_vectors:
                if param_name in tv:
                    tv_values.append(tv[param_name])
            
            if not tv_values:
                continue

            # 核心修改：根据模型数量选择融合策略
            if n_models == 2 and len(tv_values) == 2:
                # 动态比例：λ * TV1 + (1-λ) * TV2
                combined_tv = scaling_lambda * tv_values[0] + (1.0 - scaling_lambda) * tv_values[1]
                print(f"  → Using convex combination for {param_name} with λ={scaling_lambda}")
            else:
                # 简单平均（或未来可扩展为自定义权重）
                combined_tv = sum(tv_values) / len(tv_values)

            final_params[param_name] = shared_params[param_name] + combined_tv
    else:
        print(f"Router-based combination not implemented yet. Using dynamic weighting...")
        # 同上逻辑
        for param_name in tqdm(param_names, desc="Combining Twin Vectors"):
            if param_name not in shared_params:
                continue

            tv_values = []
            for tv in twin_vectors:
                if param_name in tv:
                    tv_values.append(tv[param_name])
            
            if not tv_values:
                continue

            if n_models == 2 and len(tv_values) == 2:
                combined_tv = scaling_lambda * tv_values[0] + (1.0 - scaling_lambda) * tv_values[1]
            else:
                combined_tv = sum(tv_values) / len(tv_values)

            final_params[param_name] = shared_params[param_name] + combined_tv
    # --- 5. 加载合并后的参数并保存模型 ---
    print("Loading merged parameters...")
    base_model.load_state_dict(final_params)
    
    # --- 6. 保存模型 ---
    save_model_and_tokenizer(base_model, base_model_name, output_path)
    
    # --- 7. 清理内存 ---
    del base_model, fine_tuned_params_list, shared_params, twin_vectors, final_params
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"Twin-Merging completed successfully! Model saved to: {output_path}")


def singular_value_thresholding(matrix: torch.Tensor, mu: float, device: str) -> torch.Tensor:
    """
    实现论文中的 SVT(δ; µ) 算子：
    1. 对矩阵做 SVD: δ = U Σ V^T
    2. 对奇异值做软阈值: Σ⁺ = diag(max(σᵢ - µ, 0))
    3. 重构矩阵: U Σ⁺ V^T
    """
    if matrix.numel() == 0:
        return matrix

    original_shape = matrix.shape
    if matrix.ndim != 2:
        m = matrix.size(0)
        matrix_2d = matrix.view(m, -1)
    else:
        matrix_2d = matrix

    matrix_2d = matrix_2d.to(device, dtype=torch.float32)

    try:
        U, S, Vt = torch.svd(matrix_2d)
        S_thresh = torch.clamp(S - mu, min=0.0)
        matrix_2d_lore = torch.mm(torch.mm(U, torch.diag(S_thresh)), Vt.T)
        result = matrix_2d_lore.view(original_shape).cpu()
    except Exception as e:
        print(f"[WARNING] SVD failed for shape {matrix.shape}, using zero matrix. Error: {e}")
        result = torch.zeros_like(matrix)

    return result

def lore_merge_models(
    base_model_name: str,
    model_names_list: List[str],
    task_weights: Optional[List[float]] = None,  # 👈 新增：动态融合权重
    max_iter: int = 20,
    output_path: str = "./lore_merged_model",
    mu: float = 0.01,
    scaling_lambda: float = 1.0,

):
    """
    【支持动态权重的 LoRE-Merging】
    在最终合并阶段，使用 task_weights 加权求和任务向量 δᵢ。

    Args:
        base_model_name (str): 基础预训练模型路径。
        model_names_list (list[str]): 微调模型路径列表。
        task_weights (list[float], optional): 每个模型的融合权重。若为 None，则平均。
        mu (float): 核范数系数 µ（默认 0.01）。
        max_iter (int): 坐标下降迭代次数（默认 20）。
        scaling_lambda (float): 最终缩放系数 λ（默认 1.0）。
        output_path (str): 合并后模型保存路径。
    """
    print("-" * 80)
    n_models = len(model_names_list)
    print(f"Starting LORE-MERGING (Dynamic Weighted) | µ={mu}, max_iter={max_iter}, λ={scaling_lambda}")

    # 👇 新增：处理动态权重
    if task_weights is None:
        task_weights = [1.0 / n_models] * n_models
        print("→ Using equal weights for all models")
    else:
        assert len(task_weights) == n_models, "task_weights 长度必须等于模型数量"
        total_weight = sum(task_weights)
        task_weights = [w / total_weight for w in task_weights]  # 归一化
        print(f"→ Using custom weights: {task_weights}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- 1. 加载所有模型 ---
    print("Loading models...")
    model_params_list = []
    for model_name in model_names_list:
        print(f"Loading fine-tuned model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True
        )
        model.eval()
        model_params_list.append(model.state_dict())
        del model
    gc.collect()

    # --- 2. 初始化：δᵢ = 0, θ₀ = mean(θᵢ) ---
    print("Initializing approximate base model θ₀ and task vectors δᵢ...")
    param_names = list(model_params_list[0].keys())

    deltas = [{} for _ in range(n_models)]
    for i in range(n_models):
        for param_name in param_names:
            deltas[i][param_name] = torch.zeros_like(model_params_list[i][param_name], dtype=torch.float32)

    theta_0 = {}
    for param_name in param_names:
        stacked = torch.stack([model_params_list[i][param_name] for i in range(n_models)])
        theta_0[param_name] = torch.mean(stacked, dim=0)
        del stacked
    gc.collect()

    # --- 3. 坐标下降优化：θ₀ 和 δᵢ 迭代更新 ---
    print(f"Running coordinate descent for {max_iter} iterations...")
    for iteration in range(max_iter):
        # Step 1: 更新 θ₀ = 1/n Σ (θᵢ - δᵢ)
        print(f"  → Iteration {iteration + 1}/{max_iter}: Updating θ₀...")
        for param_name in tqdm(param_names, desc="Updating θ₀", leave=False):
            stacked_diff = torch.stack([
                model_params_list[i][param_name] - deltas[i][param_name]
                for i in range(n_models)
            ])
            theta_0[param_name] = torch.mean(stacked_diff, dim=0)
            del stacked_diff

        # Step 2: 更新 δᵢ = SVT(θᵢ - θ₀; µ)
        print(f"  → Iteration {iteration + 1}/{max_iter}: Updating δᵢ with SVT...")
        for i in range(n_models):
            for param_name in tqdm(param_names, desc=f"Updating δ_{i}", leave=False):
                residual = model_params_list[i][param_name] - theta_0[param_name]
                deltas[i][param_name] = singular_value_thresholding(residual, mu, device)

        gc.collect()

    # --- 4. 计算最终合并模型（支持动态权重！）---
    print("Computing final merged model with dynamic weights...")

    # 👇 核心改动：加权求和 τ = Σ (wᵢ * δᵢ)
    tau = {}
    for param_name in tqdm(param_names, desc="Weighted sum of task vectors"):
        weighted_sum = None
        for i in range(n_models):
            delta_weighted = deltas[i][param_name] * task_weights[i]  # 👈 动态权重生效！
            if weighted_sum is None:
                weighted_sum = delta_weighted
            else:
                weighted_sum += delta_weighted
        tau[param_name] = weighted_sum

    # Step 3: θ* = θ₀ + λ * τ
    final_params = {}
    for param_name in tqdm(param_names, desc="Applying scaling and merging"):
        final_params[param_name] = theta_0[param_name] + scaling_lambda * tau[param_name]

    # --- 5. 加载合并后的参数并保存模型 ---
    print("Loading merged parameters...")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="cpu", trust_remote_code=True)
    final_params_casted = {
        k: v.to(base_model.state_dict()[k].dtype)
        for k, v in final_params.items()
    }
    base_model.load_state_dict(final_params_casted)

    # --- 6. 保存模型 ---
    save_model_and_tokenizer(base_model, base_model_name, output_path)

    # --- 7. 清理内存 ---
    del base_model, model_params_list, deltas, theta_0, tau, final_params, final_params_casted
    gc.collect()
    torch.cuda.empty_cache()

    print(f"✅ LORE-MERGING with dynamic weights completed! Model saved to: {output_path}")

def emr_merge_models(
    base_model_name: str,
    model_names_list: List[str],
    task_weights: Optional[List[float]] = None,  # 👈 新增参数：任务权重
    output_path: str = "./emr_merged_model"
):
    """
    【支持动态比例的 EMR-Merging】
    通过 task_weights 动态调节每个任务的贡献比例。

    Args:
        base_model_name (str): 基础预训练模型的路径。
        model_names_list (list[str]): 需要合并的多个微调模型的路径列表。
        task_weights (list[float], optional): 每个任务的权重。若为 None，则使用平均权重。
        output_path (str): 合并后模型的保存路径。
    """
    print("-" * 80)
    n_models = len(model_names_list)
    print(f"Starting EMR-Merging for {n_models} models")

    if task_weights is None:
        task_weights = [1.0 / n_models] * n_models  # 默认平均
        print("→ Using equal weights for all tasks")
    else:
        assert len(task_weights) == n_models, "task_weights 长度必须等于模型数量"
        # 可选：归一化权重（非必须）
        total_weight = sum(task_weights)
        task_weights = [w / total_weight for w in task_weights]
        print(f"→ Using custom weights: {task_weights}")

    # --- 1. 加载所有模型 ---
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    )
    base_model.eval()
    params_base = base_model.state_dict()

    fine_tuned_params_list = []
    for model_name in model_names_list:
        print(f"Loading fine-tuned model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
        )
        model.eval()
        fine_tuned_params_list.append(model.state_dict())
        del model
    gc.collect()

    # --- 2. 计算任务向量 ---
    print("Computing task vectors...")
    task_vectors = []
    param_names = list(params_base.keys())

    for i, ft_params in enumerate(fine_tuned_params_list):
        print(f"Computing task vector {i+1}/{n_models}")
        task_vector = {}
        for param_name in tqdm(param_names, desc=f"TV {i+1}", leave=False):
            if param_name in ft_params:
                base_tensor = params_base[param_name]
                ft_tensor = ft_params[param_name]
                task_vector[param_name] = ft_tensor - base_tensor
        task_vectors.append(task_vector)

    # --- 3. EMR合并算法 ---
    print("Applying EMR merging algorithm...")

    # 步骤1: 计算平均参数（用于方向标志）
    sum_param = {}
    for param_name in tqdm(param_names, desc="Computing average parameters", leave=False):
        if param_name in task_vectors[0]:
            param_values = []
            for tv in task_vectors:
                if param_name in tv:
                    param_values.append(tv[param_name])
            if param_values:
                sum_param[param_name] = torch.stack(param_values, 0).mean(0)

    # 步骤2-6: EMR核心算法
    vector_unified = {}
    scales = torch.zeros(n_models)
    masks = {}

    for param_name in tqdm(param_names, desc="EMR core algorithm", leave=False):
        if param_name not in sum_param:
            continue

        masks[param_name] = []
        flag = (sum_param[param_name] > 0) * 2 - 1  # 方向标志
        param_max = torch.zeros_like(task_vectors[0][param_name])

        for m in range(n_models):
            if param_name in task_vectors[m]:
                param = task_vectors[m][param_name]
                mask = (param * flag) > 0
                masks[param_name].append(mask)
                param_abs = torch.abs(mask * param)
                param_max = torch.where(param_abs > param_max, param_abs, param_max)
                scales[m] += torch.mean(torch.abs(param))

        vector_unified[param_name] = param_max * flag

    # 步骤6: 计算重缩放因子
    print("Computing rescaling factors...")
    new_scales = torch.zeros(n_models)

    for m in range(n_models):
        for param_name in vector_unified:
            if param_name in masks and len(masks[param_name]) > m:
                p = vector_unified[param_name] * masks[param_name][m]
                new_scales[m] += torch.mean(torch.abs(p))

    rescalers = torch.where(new_scales > 0, scales / new_scales, torch.ones_like(scales))

    # --- 4. 应用EMR合并结果（使用动态权重）---
    print("Applying EMR merged parameters with dynamic weights...")
    merged_params = base_model.state_dict()

    for param_name in tqdm(param_names, desc="Applying EMR results", leave=False):
        if param_name in vector_unified:
            base_tensor = params_base[param_name]
            weighted_task_vector = torch.zeros_like(base_tensor)

            for m in range(n_models):
                if param_name in masks and len(masks[param_name]) > m:
                    task_vector_recon = vector_unified[param_name] * masks[param_name][m] * rescalers[m]
                    weighted_task_vector += task_weights[m] * task_vector_recon  # 👈 动态权重！

            merged_params[param_name] = base_tensor + weighted_task_vector

    # --- 5. 保存模型 ---
    print("Loading merged parameters...")
    base_model.load_state_dict(merged_params)
    save_model_and_tokenizer(base_model, base_model_name, output_path)

    # --- 6. 清理内存 ---
    del base_model, fine_tuned_params_list, task_vectors, vector_unified, masks, merged_params
    gc.collect()
    torch.cuda.empty_cache()

    print(f"✅ EMR-Merging with dynamic weights completed! Model saved to: {output_path}")


import numpy as np

def normalize(v: np.ndarray, eps: float = 1e-8):
    norm_v = np.linalg.norm(v)
    if norm_v > eps:
        v = v / norm_v
    return v

def lerp(t: float, v0: np.ndarray, v1: np.ndarray) -> np.ndarray:
    return (1 - t) * v0 + t * v1

def slerp(
    t: float,
    v0: torch.Tensor,
    v1: torch.Tensor,
    DOT_THRESHOLD: float = 0.9995,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Spherical Linear Interpolation for PyTorch tensors.
    """
    # 转为 numpy 处理
    v0_np = v0.detach().cpu().float().numpy()
    v1_np = v1.detach().cpu().float().numpy()
    v0_copy, v1_copy = np.copy(v0_np), np.copy(v1_np)

    # 归一化方向
    v0_norm = normalize(v0_np, eps)
    v1_norm = normalize(v1_np, eps)

    dot = np.sum(v0_norm * v1_norm)

    # 若接近共线，退化为线性插值
    if np.abs(dot) > DOT_THRESHOLD:
        res = lerp(t, v0_copy, v1_copy)
    else:
        theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / (sin_theta_0 + eps)
        s1 = sin_theta_t / (sin_theta_0 + eps)
        res = s0 * v0_copy + s1 * v1_copy

    # 转回 torch tensor，保留原设备和 dtype
    res_tensor = torch.from_numpy(res).to(v0.dtype).to(v0.device)
    return res_tensor

# --- 主函数：仿照 weighted_average_models 接口 ---

def slerp_merge_models(
    model_name_a: str,
    model_name_b: str,
    t: float,
    output_path: str = "./slerp_merged_model"
):
    """
    使用球面线性插值（Slerp）合并两个模型：
        result = slerp(t, model_a, model_b)
    当 t=0 时完全等于 model_a，t=1 时完全等于 model_b。

    Args:
        model_name_a (str): 起始模型路径
        model_name_b (str): 目标模型路径
        t (float): 插值系数，范围 [0, 1]
        output_path (str): 合并后模型保存路径
    """
    print("-" * 80)
    print(f"Starting SLERP merge with t = {t}")

    # 加载模型
    print("Loading model A...")
    model_a = AutoModelForCausalLM.from_pretrained(
        model_name_a, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    )
    model_a.eval()
    params_a = model_a.state_dict()

    print("Loading model B...")
    model_b = AutoModelForCausalLM.from_pretrained(
        model_name_b, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    )
    model_b.eval()
    params_b = model_b.state_dict()

    # 执行 SLERP 合并
    merged_params = {}
    param_names = list(params_a.keys())

    for param_name in tqdm(param_names, desc=f"SLERP (t={t})"):
        if param_name in params_b:
            tensor_a = params_a[param_name]
            tensor_b = params_b[param_name].to(tensor_a.device, dtype=tensor_a.dtype)
            # 👇 核心：使用 SLERP 替代线性插值
            merged_params[param_name] = slerp(t, tensor_a, tensor_b)
        else:
            # 若 model_b 没有该参数，直接保留 model_a 的
            merged_params[param_name] = params_a[param_name]

    # 加载合并后参数
    print("Loading merged parameters...")
    model_a.load_state_dict(merged_params)

    # 保存模型
    save_model_and_tokenizer(model_a, model_name_a, output_path)

    # 清理内存
    del model_a, model_b, params_a, params_b, merged_params
    gc.collect()
    torch.cuda.empty_cache()

    print(f"✅ SLERP merging completed! Model saved to: {output_path}")

def ties_merge_models_TA(base_model_name, model_names_list, density, weights=None,  output_path=None,scaling_lambda=1.0):
    """
    使用标准 TIES-Merging (Trim, Elect Sign & Merge) 方法合并多个模型。
    此实现遵循 mergekit 库的标准算法，使用局部阈值和加权符号一致性。

    Args:
        base_model_name (str): 基础预训练模型的路径。
        model_names_list (list[str]): 需要合并的多个微调模型的路径列表。
        density (float): Trim 步骤中保留的参数密度。每个tensor独自计算阈值。
        weights(list[float],optinal):每个模型权重。
        scaling_lambda (float): 最终合并任务向量的缩放系数 λ。
        output_path (str, optional): 合并后模型的保存路径。如果为 None，则返回合并后的模型。
    
    Returns:
        AutoModelForCausalLM: 合并后的模型
    """
    print("-" * 80)
    print(f"Starting Standard TIES-Merging with density={density} and scaling_lambda={scaling_lambda}")
    
    if weights is None:
        weights = [1.0] * len(model_names_list)
    elif len(weights) != len(model_names_list):
        raise ValueError("Weights list length must match model_names_list length")

    # --- 1. 加载所有模型 ---
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    )
    base_model.eval()
    params_base = base_model.state_dict()

    fine_tuned_params_list = []
    for model_name in model_names_list:
        print(f"Loading fine-tuned model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
        )
        model.eval()
        fine_tuned_params_list.append(model.state_dict())
        del model
    gc.collect()

    # --- 2. 逐个参数进行标准 TIES 合并 ---
    final_merged_task_vector = {}
    param_names = list(params_base.keys())

    for param_name in tqdm(param_names, desc="TIES-Merging with Local Trim"):
        base_tensor = params_base[param_name]
        
        # --- 计算当前参数的任务向量 ---
        tvs_for_param = []
        for i, ft_params in enumerate(fine_tuned_params_list):
            delta = ft_params[param_name] - base_tensor
            tvs_for_param.append({
                'delta': delta,
                'weight': weights[i],
                'density': density
            })

        # --- Step 1: Trim (Local Threshold per tensor) ---
        trimmed_tvs_for_param = []
        for tv_info in tvs_for_param:
            delta = tv_info['delta']
            density = tv_info['density']
            
            if delta.ndim == 0 or density >= 1.0:  # Skip scalar tensors or no trimming
                trimmed_tvs_for_param.append(delta)
                continue

            # 使用局部阈值进行修剪（按张量独立计算）
            k = int(density * delta.numel())
            if k <= 0:
                trimmed_tvs_for_param.append(torch.zeros_like(delta))
                continue
                
            # 计算局部阈值
            flat_delta = delta.abs().view(-1)
            if flat_delta.device.type == "cpu":
                flat_delta = flat_delta.float()
            
            # 找到第 k 大的值作为阈值
            topk_values, _ = torch.topk(flat_delta, k, largest=True)
            threshold = topk_values[-1] if k > 0 else 0.0
            
            # 应用阈值掩码
            mask = delta.abs() >= threshold
            trimmed_tvs_for_param.append(delta * mask)
        
        # --- Step 2: Elect Sign (Weighted Consensus) ---
        # 按照 mergekit 库的标准实现加权符号一致性
        deltas = torch.stack(trimmed_tvs_for_param, dim=0)
        weights_tensor = torch.tensor(weights, dtype=deltas.dtype, device=deltas.device)
        
        # 计算加权和来确定多数符号
        weighted_deltas = deltas * weights_tensor.view(-1, *([1] * (deltas.ndim - 1)))
        sign_weight = weighted_deltas.sum(dim=0)
        majority_sign = (sign_weight >= 0).float() * 2 - 1
        
        # --- Step 3: Merge with Sign Consensus ---
        # 创建符号一致性掩码
        signs = deltas.sign()
        consensus_mask = (signs == majority_sign).float()
        
        # 应用权重和符号一致性
        weighted_consensus_deltas = weighted_deltas * consensus_mask
        final_sum = weighted_consensus_deltas.sum(dim=0)
        
        # 计算归一化因子
        weight_mask_sum = (weights_tensor.view(-1, *([1] * (deltas.ndim - 1))) * consensus_mask).sum(dim=0)
        weight_mask_sum[weight_mask_sum.abs() < 1e-8] = 1.0  # 避免除以零
        
        # 归一化
        merged_tensor = final_sum / weight_mask_sum
        final_merged_task_vector[param_name] = merged_tensor

    # 清理内存
    del fine_tuned_params_list
    gc.collect()

    # --- 3. 将合并后的任务向量应用到基础模型 ---
    print("Applying merged task vector to the base model...")
    merged_params = base_model.state_dict()
    for param_name, tensor in final_merged_task_vector.items():
        merged_params[param_name] += (scaling_lambda * tensor).to(merged_params[param_name].device)

    base_model.load_state_dict(merged_params)
    
    # --- 4. 保存模型（如果指定了输出路径） ---
    if output_path:
        print(f"Saving merged model to {output_path}")
        base_model.save_pretrained(output_path)
        # 如果基础模型有 tokenizer，也保存它
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            tokenizer.save_pretrained(output_path)
        except Exception as e:
            print(f"Warning: Could not save tokenizer: {e}")

    # 最终清理
    del final_merged_task_vector, merged_params
    gc.collect()
    torch.cuda.empty_cache()
    
    return base_model