import torch
from torch.utils.data import Dataset
import random
from collections import defaultdict
from pathlib import Path

class MultiViewDataset(Dataset):
    def __init__(self, data_path, labels_dict=None):
        self.data_dir = Path(data_path)
        self.labels_dict = labels_dict if labels_dict is not None else {}
        self.patient_to_files = defaultdict(list)
                
        for pt_file in self.data_dir.glob("*.pt"):
            parts = pt_file.stem.rsplit('_', 1)
            
            if len(parts) == 2:
                patient_id = parts[0]
                self.patient_to_files[patient_id].append(pt_file)
                
        self.patient_ids = list(self.patient_to_files.keys())
        
    def __len__(self):
        return len(self.patient_ids)
    
    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        all_files = self.patient_to_files[patient_id]
        
        label = self.labels_dict.get(patient_id, 0)
        
        num_views = random.randint(1, len(all_files))
        selected_files = all_files[:num_views]
        
        tensors = [torch.load(f, weights_only=True) for f in selected_files]
        
        image_tensor = torch.stack(tensors, dim=0)
        
        return patient_id, label