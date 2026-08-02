import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.model_selection import StratifiedKFold
import numpy as np
import torchxrayvision as xrv

# 1. Reconstruct identical backbone structure from Phase 2
def get_model(weights_path=None, freeze_backbone=True, from_xrv=False):
    is_valid_path = weights_path and os.path.exists(weights_path)
    
    if from_xrv:
        # Load the pre-trained xrv model
        model = list(xrv.models.ResNet(weights="resnet50-res512-all").children())[0]
        # Remove the classifier head to get the backbone
        # model = nn.Sequential(*list(model.children())[:-1])
    else:
        # model = resnet50()
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    
    # Apply GroupNorm conversion to match the saved state dict structure
    def convert_to_groupnorm(m, num_groups=32):
        for name, module in m.named_children():
            if isinstance(module, nn.BatchNorm2d):
                num_channels = module.num_features
                groups = num_channels if num_channels % num_groups != 0 else num_groups
                setattr(m, name, nn.GroupNorm(num_groups=groups, num_channels=num_channels))
            else:
                convert_to_groupnorm(module, num_groups)
    
    if is_valid_path:
        convert_to_groupnorm(model)
        
    model.fc = nn.Identity()
    
    # Load weights, convert tensors to double, and lock gradients
    if is_valid_path:
        print(f"Loading weights from {weights_path}")
        raw_state = torch.load(weights_path, map_location="cpu")
    else:
        raw_state = model.state_dict()  # Use the current state if no weights are provided
        
    state_dict = {}
    for k, v in raw_state.items():
        if isinstance(v, torch.Tensor):
            state_dict[k] = v.double()
        else:
            state_dict[k] = v
            
    if is_valid_path:
        model.load_state_dict(state_dict)
    # ensure backbone parameters use double precision
    model = model.double()

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    return model

# 2. Linear Probe Class
class LinearProbe(nn.Module):
    def __init__(self, backbone, num_classes=2):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(2048, num_classes)
        
    def forward(self, x):
        # ensure input matches backbone dtype (double)
        with torch.no_grad():
            features = self.backbone(x.double())
        return self.classifier(features)

# 3. 5-Fold Cross-Validation Loop
def run_5fold_evaluation(dataset, targets, weights_path, freeze_backbone=True, from_xrv=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_accuracies = []
    criterion = nn.CrossEntropyLoss()
    
    # Stratified split requires arrays
    indices = np.arange(len(dataset))
    targets_arr = np.array(targets)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(indices, targets_arr)):
        print(f"\n--- Training Fold {fold + 1}/5 ---")
        
        # Create DataLoaders for current fold
        train_sub = Subset(dataset, train_idx)
        val_sub = Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_sub, batch_size=16, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_sub, batch_size=16, shuffle=False, drop_last=False)
        
        # Initialize fresh probe for each fold (backbone returned as double)
        backbone = get_model(weights_path, freeze_backbone=freeze_backbone, from_xrv=from_xrv)
        model = LinearProbe(backbone, num_classes=2).to(device).double()
        
        params = model.classifier.parameters() if freeze_backbone else model.parameters()
        learning_rate = 1e-3 if freeze_backbone else 1e-4
        
        # Only optimize the single linear layer parameters
        optimizer = Adam(params, lr=learning_rate, weight_decay=1e-5)
        
        # Train linear classifier (Fast convergence since backbone is frozen)
        model.train()
        for epoch in range(10): # 10 epochs is standard for linear probing on small datasets
            for images, labels, _ in train_loader:
                # cast images to double and labels to long for CrossEntropyLoss
                images = images.to(device).double()
                labels = labels.to(device).long()

                outputs = model(images)
                loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # Evaluate current fold
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device).double()
                labels = labels.to(device).long()
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        fold_acc = 100 * correct / total
        fold_accuracies.append(fold_acc)
        print(f"Fold {fold + 1} Validation Accuracy: {fold_acc:.2f}%")
        
    print(f"\n======================================")
    print(f"Mean CV Accuracy: {np.mean(fold_accuracies):.2f}% (+/- {np.std(fold_accuracies):.2f}%)")
    print(f"======================================")

    return fold_accuracies