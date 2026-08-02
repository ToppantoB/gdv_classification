import argparse
from pathlib import Path

import torch

from torch.utils.data import DataLoader, Subset

from dataset.dataloader import get_dataloader
from model.model import get_model, get_xrv_model
from training.training import train
from utils import enforce_reproducibility

import numpy as np
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    enforce_reproducibility(42)

    parser = argparse.ArgumentParser()
    # parser.add_argument("--mode", choices=["train", "pretrain", "process", "eval"], default="eval",
    #                     help="Mode to run: train, pretrain, or process")
    parser.add_argument("--data_type", default="256", choices=["flat", "256", "512"],
                        help="Type of input data: flat (one channel 512x512), 256 (three channel 256x256), or 512 (three channel 512x512)")
    parser.add_argument("--model_type", default="resnet-50", choices=["resnet-50", "xrv_resnet", "resnet-34", "resnet-18"],
                        help="Model architecture to use")
    parser.add_argument("--multi_modal", action=argparse.BooleanOptionalAction, default=False,
                        help="Whether to use multi-modal inputs")
    parser.add_argument("--weights_path", type=str, default=None,
                        help="Path to the model weights for evaluation or fine-tuning")

    args = parser.parse_args()
    
    print(f"Running with data type {args.data_type} and model type {args.model_type}. Multi-modal: {args.multi_modal}. Weights path: {args.weights_path}")

    dataloader = get_dataloader(f"data\\combined", batch_size=4, shuffle=True)
    # test_loader = get_dataloader(f"files\\tensors\\{args.data_type}\\labeled\\test", batch_size=32, shuffle=False)

    # if args.model_type == "xrv_resnet":
    #     model = get_xrv_model(weights_path=args.weights_path, multi_modal=args.multi_modal)
    # else:
    model = get_model(model_type=args.model_type, fine_tune=True, weights_path=args.weights_path, multi_modal=args.multi_modal)
    # model = get_xrv_model()

    indices = np.arange(len(dataloader.dataset))
    labels = [dataloader.dataset.labels_dict[dataloader.dataset.patient_ids[i]] for i in indices]


    X_train, X_test, y_train, y_test = train_test_split(indices, labels, test_size=0.2, stratify=labels, random_state=42)
    
    # test data loader:
    # batch = next(iter(dataloader))

    train_sub = Subset(dataloader.dataset, X_train)
    test_sub = Subset(dataloader.dataset, X_test)
    
    test_loader = DataLoader(test_sub, batch_size=4, shuffle=False, collate_fn=dataloader.collate_fn)
    train_loader = DataLoader(train_sub, batch_size=4, shuffle=True, collate_fn=dataloader.collate_fn)

    # train(model, train_loader, test_loader, multi_modal=args.multi_modal)
    
    
    
    