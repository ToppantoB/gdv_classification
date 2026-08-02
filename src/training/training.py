import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def train(model, train_loader, test_loader, multi_modal=False, device="cuda"):
  # Use BCEWithLogitsLoss for binary classification
  criterion = nn.BCEWithLogitsLoss()

  # AdamW is a robust default. Use a lower learning rate for fine-tuning.
  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
  
  num_epochs = 50

#   evaluate(model, test_loader, -1, device)  # Evaluate before training
  best_accuracy = 0.0
  best_val_loss = float('inf')
  patience = 8
  epoch_since_improvement = 0
  
  losses = []
  accuracies = []

  for epoch in range(num_epochs):
      model.train()
      running_loss = 0.0
      
      pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
      
      # Assuming 'train_loader' yields (images, labels, dog_weights)
      for images, labels in pbar:
        predictions = []
        labels = labels.to(device).float().unsqueeze(1)      
    
        for i in range(len(images)):
            x = images[i][0].to(device).float().unsqueeze(0) # [0] <=== take only the first image to test
          
            # Ensure labels are floats and shaped (batch_size, 1)
            
            # If using multi-modal, pass the dog weights to the model
            if model.multi_modal:
                weights = weights.to(device).float().unsqueeze(1)

            # 1. Zero gradients
            optimizer.zero_grad()
            
            # 2. Forward pass
            prediction = model(x, weight=weights if model.multi_modal else None)
            predictions.append(prediction)
            
        batch_predictions = torch.cat(predictions, dim=0)
        # 3. Calculate loss
        loss = criterion(batch_predictions, labels)
        
        # 4. Backward pass and optimize
        loss.backward()
        optimizer.step()          
        running_loss += loss.item()
          
      val_loss, val_accuracy = evaluate(model, test_loader, epoch, device)
      losses.append(val_loss)
      accuracies.append(val_accuracy)
      
      scheduler.step(val_loss)
      
      best_accuracy = max(best_accuracy, val_accuracy)
            
      if val_loss < best_val_loss:
          epoch_since_improvement = 0
      else:
          epoch_since_improvement += 1
      
      best_val_loss = min(best_val_loss, val_loss)
          
    #   if epoch_since_improvement >= patience:
    #       print(f"No improvement for {patience} epochs. Stopping early.")
    #       break
        
      print(f"Loss: {running_loss/len(train_loader):.4f}")
  print(f"Best Validation Accuracy: {best_accuracy:.4f}")
  
#   plt.figure(figsize=(10, 5))
#   plt.subplot(1, 2, 1)
#   plt.plot(losses)
#   plt.title('Validation Loss')
#   plt.xlabel('Epoch')
#   plt.ylabel('Loss')

#   plt.subplot(1, 2, 2)
#   plt.plot(accuracies)
#   plt.title('Validation Accuracy')
#   plt.xlabel('Epoch')
#   plt.ylabel('Accuracy')
#   plt.tight_layout()
#   plt.savefig('output/loss_accuracy.png')
#   plt.close()

def evaluate(model, test_loader, epoch, device="cuda" ):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            predictions = []
            
            for i in range(len(images)):
                x = images[i][0].to(device).float().unsqueeze(0) # [0] <=== take only the first image to test
                prediction = model(x, weight=weights.to(device).float().unsqueeze(1) if model.multi_modal else None)
                predictions.append(prediction)
            
            labels = labels.to(device).float().unsqueeze(1)
            
            predicted = (torch.sigmoid(torch.cat(predictions)) > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.view(-1).cpu().numpy().tolist())
            all_predictions.extend(predicted.view(-1).cpu().numpy().tolist())
            
            loss = nn.BCEWithLogitsLoss()(torch.cat(predictions), labels)
    
    accuracy = correct / total
    
    print(f"Test Accuracy: {accuracy:.4f}")

    # cm = confusion_matrix(all_labels, all_predictions)
    # plt.figure(figsize=(5, 5))
    # plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    # plt.title('Confusion Matrix')
    # plt.colorbar()
    # plt.xlabel('Predicted label')
    # plt.ylabel('True label')
    # plt.xticks([0, 1])
    # plt.yticks([0, 1])

    # for i in range(cm.shape[0]):
    #     for j in range(cm.shape[1]):
    #         plt.text(j, i, str(cm[i, j]), ha='center', va='center',
    #                   color='white' if cm[i, j] > cm.max() / 2 else 'black')

    # plt.tight_layout()
    # plt.savefig(f'output/confusion_matrix_epoch_{epoch}.png')
    # plt.close()
    
    return loss.item(), accuracy  # Return validation loss for scheduler (1 - accuracy) and accuracy