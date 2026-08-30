import torch
from torch.utils.data import Dataset
from collections import defaultdict
from pathlib import Path
import pandas as pd
import warnings

class MultiViewDataset(Dataset):
    """Dataset that groups multiple tensor views into one case-level sample."""

    def __init__(self, data_path, labels_dict=None, is_train=True, multi_modal=False, view="multi"):
        self.data_dir = Path(data_path)
        self.labels_dict = labels_dict if labels_dict is not None else {}
        self.patient_to_files = defaultdict(list)

        # Tensor filenames encode the case id before the final underscore.
        for pt_file in self.data_dir.glob("*.pt"):
            parts = pt_file.stem.rsplit('_', 1)
            
            if len(parts) == 2:
                patient_id = parts[0]
                self.patient_to_files[patient_id].append(pt_file)
                
        self.patient_ids = list(self.patient_to_files.keys())
        self.is_train = is_train
        self.multi_modal = multi_modal
        
        if view not in ['single', 'multi', 'mixed']:
            warnings.warn(
                f"view: '{view}' is not a valid option. Use one of 'single', 'multi' or 'mixed'. Defaulting to 'multi'",
                stacklevel=2,
            )
            self.view = 'multi'
        else:
            self.view = view
        
        if multi_modal:
            emd = pd.read_csv(self.data_dir / "metadata.csv", dtype={"case_id": str})
            emd = emd.fillna({'Age': -1, 'Weight': -1})
                        
            self.weights_dict = dict(zip(emd['case_id'], emd['Weight']))
            self.ages_dict = dict(zip(emd['case_id'], emd['Age']))
        
        
    def __len__(self):
        return len(self.patient_ids)
    
    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        all_files = self.patient_to_files[patient_id]
        label = self.labels_dict.get(patient_id, 0)       

        # "mixed" trains with all views but evaluates on a single deterministic view.
        if self.view == "single":
            selected_files = [all_files[0]]
        elif self.view == "multi":
            selected_files = all_files
        else:
            if self.is_train:
                selected_files = all_files
            else:
                selected_files = [all_files[0]]

        tensors = [torch.load(f, weights_only=True) for f in selected_files]
        
        image_tensor = torch.stack(tensors, dim=0)
        
        if self.multi_modal:
            weight = self.weights_dict.get(patient_id, 0)
            age = self.ages_dict.get(patient_id, 0)
            return image_tensor, (label, age, weight), ""
        
        return image_tensor, (label, 0, 0), patient_id