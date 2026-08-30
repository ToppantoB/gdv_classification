from dataset.dataset import MultiViewDataset
from torch.utils.data import DataLoader
import torch
from pathlib import Path

def get_dataloader(directory, batch_size=32, shuffle=True, multi_modal=False, view="multi"):
    labels_dict = extract_labels_from_csv(Path(directory) / "metadata.csv")  
    dataset = MultiViewDataset(directory, labels_dict=labels_dict, multi_modal=multi_modal, view=view)    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=multi_view_collate_fn)
    return dataloader

def multi_view_collate_fn(batch):
    """Keep each case's variable number of views separate inside the batch."""

    images = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    
    return images, labels
  
def extract_labels_from_csv(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path, dtype={"case_id": str})
    labels_dict = dict(zip(df['case_id'], df['label']))
    return labels_dict