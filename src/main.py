from dataset.dataloader import get_dataloader
from model_dir.model import get_model

import numpy as np
from pathlib import Path
from eval.five_fold_eval import run_5fold_evaluation
from local_config import CONFIG, DEVICE
from utils import enforce_reproducibility, process_and_extract_best_metrics
   
    
def run_training(number_of_runs, 
                 data_path):
  
    epochs = CONFIG.training_epochs
    output_paths = [CONFIG.accuracy_result_outputs]
    if CONFIG.with_per_epoch_output:
      output_paths.extend([CONFIG.per_epoch_report_output, CONFIG.summary_report_output])

    for output_path in output_paths:
      Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    randoms = np.arange(number_of_runs)
    
    for random_seed in randoms:
      if CONFIG.with_accuracy_outputs:
        with open(CONFIG.accuracy_result_outputs, "a") as f:
          f.write(f"\nRandom seed: {random_seed}\n")
      
      if CONFIG.with_per_epoch_output:
        with open(CONFIG.per_epoch_report_output, "a") as f:
          f.write(f"\nRandom seed: {random_seed}\n")

      for (multi_modal, weights_path, pooling, view) in CONFIG.training_config:
        
        # logging
        modality = f"{'Standard' if not multi_modal else 'Multi-Modal'}"
        backbone_type = f"{'(MoCo pre-trained), ' if weights_path is not None else ''}"
        pooling_type = f"{'Attention pooling' if pooling == 'attention' else ('Max pooling' if pooling == 'max' else 'Mean pooling with')}"
        view_type = f"{view} view."
          
        if CONFIG.with_accuracy_outputs:
          with open(CONFIG.accuracy_result_outputs, "a") as f:
            f.write(f"\n{modality}, {backbone_type}{pooling_type}, {view_type}\n\n")
        
        if CONFIG.with_per_epoch_output:
          with open(CONFIG.per_epoch_report_output, "a") as f:
            f.write(f"\n{modality}, {backbone_type}{pooling_type}, {view_type}\n\n")
        # logging end
          
        enforce_reproducibility(random_seed)
        
        dataloader = get_dataloader(data_path, batch_size=4, shuffle=True, multi_modal=multi_modal, view=view)
        
        indices = np.arange(len(dataloader.dataset))
        labels = [dataloader.dataset.labels_dict[dataloader.dataset.patient_ids[i]] for i in indices]
        
        model = get_model(weights_path=weights_path, 
                    multi_modal=multi_modal, 
                    pooling=pooling,
                    device=DEVICE)
        
        run_5fold_evaluation(dataloader.dataset, 
                            labels, 
                            model, 
                            epochs=epochs, 
                            multi_modal=multi_modal,
                            device=DEVICE,
                            random_state=random_seed)

if __name__ == "__main__":
    print(f"Configured global multi-modal flag: {CONFIG.multi_modal}.")
    print(f"Using device: {DEVICE}.")

    data_path = CONFIG.data_path

    run_training(number_of_runs=CONFIG.random_seeds_to_run_for,
                 data_path=data_path)

    if CONFIG.with_per_epoch_output:
      process_and_extract_best_metrics(CONFIG.per_epoch_report_output, CONFIG.summary_report_output)