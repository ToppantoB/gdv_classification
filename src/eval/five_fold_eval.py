from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
import numpy as np
from training.training import train
from dataset.dataloader import multi_view_collate_fn
import copy
from local_config import CONFIG

def run_5fold_evaluation(dataset, targets, model, epochs, device, multi_modal=False, output_file=CONFIG.accuracy_result_outputs, random_state=42):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    
    fold_accuracies = []
    fold_recalls = []
    fold_aucs = []
    fold_epoch_stats = []
        
    # Stratified split requires arrays
    indices = np.arange(len(dataset))
    targets_arr = np.array(targets)
    
    for _, (train_idx, val_idx) in enumerate(skf.split(indices, targets_arr)):
        train_sub = Subset(dataset, train_idx)
        val_sub = Subset(dataset, val_idx)

        # Each fold starts from the same initial weights before fold-specific training.
        model_copy = copy.deepcopy(model)
        
        train_loader = DataLoader(train_sub, batch_size=4, shuffle=True, drop_last=False, collate_fn=multi_view_collate_fn)
        val_loader = DataLoader(val_sub, batch_size=4, shuffle=False, drop_last=False, collate_fn=multi_view_collate_fn)
        
        accuracy, recall, auc, epoch_stats = train(
            model_copy,
            train_loader,
            val_loader,
            device=device,
            epochs=epochs,
            multi_modal=multi_modal,
        )
        fold_accuracies.append(accuracy)
        fold_recalls.append(recall)
        fold_aucs.append(auc)
        fold_epoch_stats.append(epoch_stats)

    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)

    mean_recall = np.mean(fold_recalls)
    std_recall = np.std(fold_recalls)
    
    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)    

    print(f"\n======================================")
    print(f"Mean CV Accuracy: {mean_acc:.4f}% (+/- {std_acc:.4f}%)")
    print(f"======================================")
    
    if CONFIG.with_accuracy_outputs:
        with open(output_file, "a") as f:
            f.write(f"Mean Accuracy: {mean_acc:.4f}% (+/- {std_acc:.4f}%)\n")
            f.write(f"Mean Recall: {mean_recall:.4f}% (+/- {std_recall:.4f}%)\n")
            f.write(f"Mean AUC: {mean_auc:.4f}% (+/- {std_auc:.4f}%)\n")
            f.write("\n" + "-" * 40 + "\n\n")

    if CONFIG.with_per_epoch_output:
        averaged_array = np.mean(fold_epoch_stats, axis=0)
        avg_acc, avg_auc, avg_loss, avg_recall = averaged_array

        with open(CONFIG.per_epoch_report_output, 'a') as f:
            f.write(f"Accuracy: {str(list(avg_acc))}\n")
            f.write(f"AUC: {str(list(avg_auc))}\n")
            f.write(f"Loss: {str(list(avg_loss))}\n")
            f.write(f"Recall: {str(list(avg_recall))}\n")

    return fold_accuracies