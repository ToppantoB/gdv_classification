import os
import random
import numpy as np
import torch
import re
from collections import defaultdict

def enforce_reproducibility(seed=42):
    # Standard libraries
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # PyTorch and CUDA
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    # cuDNN configurations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Strict algorithmic determinism
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)


def process_and_extract_best_metrics(input_filepath, output_filepath):
    # data structure: {model_name: {metric_name: [array_from_seed_1, array_from_seed_2, ...]}}
    data = defaultdict(lambda: defaultdict(list))
    target_metrics = ["accuracy", "auc", "loss", "recall"]
    
    with open(input_filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    current_model = None
    
    # Parse and collect data across seeds
    for line in lines:
        line = line.strip()
        
        # Skip empty lines or seed headers
        if not line or line.startswith("Random seed:"):
            continue
            
        # Identify metric lines
        if ":" in line and "[" in line and "]" in line:
            metric_name, values_str = line.split(":", 1)
            metric_name = metric_name.strip()
            
            # Filter only specified metrics
            if not any(target in metric_name.lower() for target in target_metrics):
                continue
            
            # Extract float values from the np.float64(...) wrappers
            float_strings = re.findall(r"np\.float64\((.*?)\)", values_str)
            float_values = [float(x) for x in float_strings]
            
            if current_model and float_values:
                data[current_model][metric_name].append(float_values)
        else:
            # Non-metric lines represent the model name
            current_model = line

    # Compute averages, find best epochs, and write to output file
    with open(output_filepath, 'w', encoding='utf-8') as f:
        for model_name, metrics in data.items():
            f.write(f"{model_name}\n")
            
            for metric_name, seed_arrays in metrics.items():
                # Compute element-wise mean across all collected seeds
                avg_array = np.mean(seed_arrays, axis=0).tolist()
                
                # Losses should be minimized; other metrics should be maximized
                if "loss" in metric_name.lower():
                    best_val = min(avg_array)
                else:
                    best_val = max(avg_array)
                    
                best_epoch = avg_array.index(best_val) + 1
                f.write(f"  Best {metric_name}: {best_val:.6f} (Epoch {best_epoch})\n")
            
            f.write("\n")