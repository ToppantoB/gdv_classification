import torch
import torch.nn as nn
from tqdm import tqdm
from model_dir.model import freeze_batchnorm_layers
from sklearn.metrics import recall_score, roc_auc_score
from local_config import CONFIG

def train(model, train_loader, test_loader, device, multi_modal=False, epochs=20):
  """Train one fold: freeze backbone first, then unfreeze for fine-tuning."""
  criterion = nn.BCEWithLogitsLoss()

  # Backbone parameters are added later with a lower learning rate after the frozen phase.
  optimizer = torch.optim.AdamW([
      {'params': model.weight_net.parameters(), 'lr': 1e-3},
      {'params': model.age_net.parameters(), 'lr': 1e-3},
      {'params': model.classifier.parameters(), 'lr': 1e-3},
  ], weight_decay=1e-4)
    
  epoch_accuracies=[]
  epoch_aucs=[]
  epoch_losses=[]
  epoch_recalls=[]

  for param in model.backbone.parameters():
    param.requires_grad = False
  
  for epoch in range(epochs):
      model.train()
      running_loss = 0.0
          
      if epoch == CONFIG.frozen_epoch_count:
        for param in model.backbone.parameters():
            param.requires_grad = True

        optimizer.add_param_group({
            'params': model.backbone.parameters(),
            'lr': 1e-5
        })
        
      freeze_batchnorm_layers(model)
      
      pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
      
      # Each batch element contains one case with 1..N views; forward pass runs per case.
      for images, data in pbar:
        predictions = []
                
        labels = data[:,0].to(device).float().unsqueeze(1)
        age = None
        weight = None
    
        for i in range(len(images)):
            x = images[i].to(device).float()
            
            if multi_modal:
              age = data[i][1].to(device).float().unsqueeze(0)
              weight = data[i][2].to(device).float().unsqueeze(0)

            optimizer.zero_grad()

            prediction = model(x, metadata=(age, weight) if model.multi_modal else None)
            predictions.append(prediction)
            
        batch_predictions = torch.cat(predictions, dim=0)

        loss = criterion(batch_predictions, labels)
        
        loss.backward()
        optimizer.step()          
        running_loss += loss.item()
      
      if CONFIG.with_per_epoch_output:
        val_loss, val_accuracy, recall, auc, = evaluate(model, test_loader, device, multi_modal=multi_modal)
        
        epoch_accuracies.append(val_accuracy)
        epoch_aucs.append(auc)
        epoch_losses.append(val_loss)
        epoch_recalls.append(recall)
    
  
  val_loss, val_accuracy, recall, auc = evaluate(model, test_loader, device, multi_modal=multi_modal)

  return val_accuracy, recall, auc, (epoch_accuracies, epoch_aucs, epoch_losses, epoch_recalls)


def evaluate(model, test_loader, device, multi_modal=False ):
    """Evaluate one fold and return (loss, accuracy, recall, auc)."""
    model.eval()
    # For mixed-view mode, evaluation uses single-view selection through dataset state.
    test_loader.dataset.dataset.is_train = False
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    all_probabilities = []
    total_loss = 0.0
    
    criterion = nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for images, data in test_loader:
            predictions = []
            
            labels = data[:,0].to(device).float()
            age = None
            weight = None
                        
            for i in range(len(images)):
                x = images[i].to(device).float()
                
                if multi_modal:
                  age = data[i][1].to(device).float().unsqueeze(0)
                  weight = data[i][2].to(device).float().unsqueeze(0)                
                
                prediction = model(x, metadata=(age, weight) if model.multi_modal else None)
                predictions.append(prediction)

            labels = labels.to(device).float().unsqueeze(1)
            logits = torch.cat(predictions)
                        
            probs = torch.sigmoid(logits)
            predicted = (probs > 0.5).float()
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.view(-1).cpu().numpy().tolist())
            all_predictions.extend(predicted.view(-1).cpu().numpy().tolist())
            all_probabilities.extend(probs.view(-1).cpu().numpy().tolist())
            
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)

    
    accuracy = correct / total
    
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    
    try:
        auc = roc_auc_score(all_labels, all_probabilities)
    except ValueError:
        # AUC is undefined when predictions/labels contain only one class.
        auc = float('nan')
        
    test_loader.dataset.dataset.is_train = True
        
    mean_loss = total_loss / total

    return mean_loss, accuracy, recall, auc