import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights, resnet34, ResNet34_Weights
# import torchxrayvision as xrv

def convert_to_groupnorm(m, num_groups=32):
    for name, module in m.named_children():
        if isinstance(module, nn.BatchNorm2d):
            num_channels = module.num_features
            groups = num_channels if num_channels % num_groups != 0 else num_groups
            setattr(m, name, nn.GroupNorm(num_groups=groups, num_channels=num_channels))
        else:
            convert_to_groupnorm(module, num_groups)

class FineTuneResNet(nn.Module):
  def __init__(self, 
               model_type, 
               weights_path=None,
               multi_modal=False, 
               multi_modal_input_size=1, 
               multi_modal_reduction_factor=32):
    super().__init__()
    # 1. Recreate the pretraining structural container to align dictionary keys
    # base_model = resnet50()
    # base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    # base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # # base_model = resnet50()
    # self.backbone = nn.Sequential(*list(base_model.children())[:-1])
    
    base_model = None
    
    if model_type == "resnet-50":
      print("Using ResNet-50 backbone")
      # base_model = resnet50()
      base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
      # base_model = nn.Sequential(*list(base_model.children())[:-1])
    elif model_type == "resnet-34":
      print("Using ResNet-34 backbone")
      base_model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
      # base_model = nn.Sequential(*list(base_model.children())[:-1])
    elif model_type == "resnet-18":
      print("Using ResNet-18 backbone")
      base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
      # base_model = nn.Sequential(*list(base_model.children())[:-1])
    # elif model_type == "xrv_resnet":
    #   print("Using xrv ResNet backbone")
    #   base_model = list(xrv.models.ResNet(weights="resnet50-res512-all").children())[0]
    
    if multi_modal:
      print("with multi-modal input")
    else:
      print("with single-modal input")
      
    if weights_path is not None:
      print(f"Loading weights from {weights_path}")
      convert_to_groupnorm(base_model)
      # base_model.fc = nn.Identity()
      
    base_model.fc = nn.Identity()
    self.backbone = base_model
    
    if weights_path is not None:
      raw_state = torch.load(weights_path, map_location="cpu")
      state_dict = {}
      for k, v in raw_state.items():
          if isinstance(v, torch.Tensor):
              state_dict[k] = v.double()
          else:
              state_dict[k] = v
              
      self.backbone.load_state_dict(state_dict)
    
    output_dim = 2048 if (model_type == "resnet-50" or model_type == "xrv_resnet") else 512
    
    multi_modal_feature_dim = output_dim // multi_modal_reduction_factor
    print(f"Multi-modal feature dimension: {multi_modal_feature_dim}")
    
    output_dim += multi_modal_feature_dim if multi_modal else 0  # If using multi-modal, increase output dimension for the additional features
    
    # 3. Add flattening and the new binary classification head
    self.flatten = nn.Flatten(start_dim=1)
    self.fc = nn.Sequential(
        nn.Dropout(p=0.1), # Regularization for the small dataset
        # nn.Linear(2048, 1)
        nn.Linear(output_dim, 1)
    )
          
    # If using multi-modal, add a parallel branch for the second input
    self.multi_modal = multi_modal
    if multi_modal:
      # small network to process dog's weight
      self.weight_nn = nn.Linear(multi_modal_input_size, multi_modal_feature_dim)  
        
  def forward(self, x, weight=None):    
    x = self.backbone(x)
    x = self.flatten(x)
    
    if self.multi_modal and weight is not None:
      weight_features = self.weight_nn(weight)
      x = torch.cat((x, weight_features), dim=1)  # Concatenate along the feature dimension
    
    x = self.fc(x)
    return x
  
  
def get_model(model_type="resnet-50", fine_tune=True, weights_path=None, multi_modal=False, multi_modal_input_size=1, multi_modal_reduction_factor=16):
  model = FineTuneResNet(model_type=model_type,
                         weights_path=weights_path,
                         multi_modal=multi_modal, 
                         multi_modal_input_size=multi_modal_input_size, 
                         multi_modal_reduction_factor=multi_modal_reduction_factor)
  
  # 4. Enable fine-tuning by unfreezing all parameters
  if fine_tune:  
    for param in model.parameters():
        param.requires_grad = True
  else:
    for param in model.parameters():
        param.requires_grad = False
      
  # 5. Move to GPU
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = model.to(device)
  
  return model

def get_xrv_model(weights_path=None, multi_modal=False, multi_modal_input_size=1, multi_modal_reduction_factor=32):
  model = FineTuneResNet(model_type="xrv_resnet",
                         weights_path=weights_path,
                         multi_modal=multi_modal, 
                         multi_modal_input_size=multi_modal_input_size, 
                         multi_modal_reduction_factor=multi_modal_reduction_factor)
  
  # 2. Disable internal operating thresholds (since you aren't using their 18 classes anymore)
  model.op_threshs = None
  
  if weights_path is not None:
    convert_to_groupnorm(model)
    raw_state = torch.load(weights_path, map_location="cpu")
    
  model.fc = nn.Identity()
  
  state_dict = {}
  for k, v in raw_state.items():
      if isinstance(v, torch.Tensor):
          state_dict[k] = v.double()
      else:
          state_dict[k] = v
          
  if weights_path is not None:
      model.load_state_dict(state_dict)
  # ensure backbone parameters use double precision
  model = model.double()
  
  # # 3. Replace the final classification head
  # # The internal torchvision resnet is stored in `model.model`
  # num_features = model.model.fc.in_features # This is 2048 for ResNet-50
  
  # # Replace it with a single output node for binary classification
  # model.model.fc = nn.Linear(num_features, 1)
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
  model = model.to(device)
  
  
  # # For linear probe:
  # for param in model.parameters():
  #   param.requires_grad = False

  # # Unfreeze ONLY your new final classification layer
  # for param in model.model.fc.parameters():
  #     param.requires_grad = True 
  model.multi_modal = multi_modal  
  return model