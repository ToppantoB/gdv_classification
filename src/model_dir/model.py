import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torch.nn import functional as F

class FineTuneResNet(nn.Module):
  """ResNet-50 classifier with optional view pooling and metadata fusion."""

  def __init__(self,
               weights_path=None,
               multi_modal=False,
               pooling="attention"):
    super().__init__()
    self.pooling = pooling
    
    base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    
    if multi_modal:
      print("With multi-modal input")
    else:
      print("With single-modal input")
      
    if weights_path is not None:
      print(f"Loading weights from {weights_path}")
      
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
    
    output_dim = 2048
      
    final_feature_dim = output_dim + (32 if multi_modal else 0)

    self.flatten = nn.Flatten(start_dim=1)    
    self.classifier = nn.Sequential(
            nn.Linear(final_feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
          nn.Linear(128, 1)
        )
    
    self.age_net = nn.Sequential(
      nn.Linear(1, 16),
      nn.ReLU(),      
      nn.Linear(16, 16),
    )

    self.weight_net = nn.Sequential(
      nn.Linear(1, 16),
      nn.ReLU(),      
      nn.Linear(16, 16),
    )
    
    self.multi_modal = multi_modal
    self.attention_pooling = AttentionPooling(feature_dim=output_dim, hidden_dim=128)
        
  def forward(self, x, metadata=None):    
    # x arrives as per-case views: (num_views, C, H, W).
    # backbone/flatten converts this to per-view embeddings: (num_views, 2048).
    
    x = self.backbone(x)
    x = self.flatten(x)       
        
    # Pool per-view embeddings into one case-level representation: (1, 2048).
    if self.pooling == "attention":
      merged_features, _ = self.attention_pooling(x)
    elif self.pooling == "max":
      merged_features = torch.max(x, dim=0).values.unsqueeze(0)
    else:
      merged_features = torch.mean(x, dim=0).unsqueeze(0)
    
    # metadata is a tuple (age, weight); each branch produces 16 features.
    # After concatenation, metadata contributes +32 dims before classification.
    if self.multi_modal:      
      age_mask = (metadata[0][0] != -1).float().view(-1, 1)
      weight_mask = (metadata[1][0] != -1).float().view(-1, 1)
      
      age_features = self.age_net(metadata[0]).unsqueeze(0)
      weight_features = self.weight_net(metadata[1]).unsqueeze(0)
      
      age_features = age_features * age_mask
      weight_features = weight_features * weight_mask
            
      metadata_features = torch.cat((age_features, weight_features), 1)
      merged_features = torch.cat((merged_features, metadata_features), 1) 
      
    x = self.classifier(merged_features)
    
    return x
  
def get_model(weights_path=None, 
              multi_modal=False,
              pooling="attention",
              device=None):
  
  model = FineTuneResNet(weights_path=weights_path,
                         multi_modal=multi_modal,
                         pooling=pooling)  

  for param in model.parameters():
      param.requires_grad = True
      
  if device is None:
    raise ValueError("device must be provided from local_config.DEVICE")

  model = model.to(device)
  
  return model

def freeze_batchnorm_layers(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            
            if module.weight is not None:
                module.weight.requires_grad = False
            if module.bias is not None:
                module.bias.requires_grad = False
                
class AttentionPooling(nn.Module):
    """Learn attention weights for combining multiple view embeddings."""

    def __init__(self, feature_dim=512, hidden_dim=128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """
        x shape: (N, feature_dim) - where N is the number of views
        """
        attn_scores = self.attention(x)
        
        attn_weights = F.softmax(attn_scores, dim=0)
        
        weighted_features = x * attn_weights
        
        pooled_features = torch.sum(weighted_features, dim=0, keepdim=True)
        
        return pooled_features, attn_weights