# GDV classification with ResNet-50

Master's thesis codebase for binary GDV classification from tensorized image views, with optional metadata fusion (age/weight), multiple pooling strategies, and 5-fold cross-validation evaluation.

## Setup

Install base dependencies (everything except PyTorch):

```bash
pip install -r requirements.txt
```

Install PyTorch separately so you can choose the CUDA/CPU build matching your machine:

1. Go to https://pytorch.org/get-started/locally/
2. Select your OS/package manager/CUDA version
3. Run the generated install command (typically includes `torch` and `torchvision`)

## Project structure

- [src/main.py](src/main.py): experiment entry point
- [src/local_config.py](src/local_config.py): all configurable experiment settings
- [src/dataset/](src/dataset/): dataset and dataloader logic
- [src/model_dir/](src/model_dir/): model definition
- [src/training/](src/training/): train/evaluate loop
- [src/eval/](src/eval/): 5-fold CV orchestration
- [src/utils.py](src/utils.py): reproducibility and report post-processing utilities

## Data protection

Due to data protection/privacy requirements, I was not allowed to upload or publish the original dataset to any public domain (public repositories, public file shares, or other openly accessible platforms).

## Data layout

By default, the entry script uses `data\tensors` as data root.

Expected files:
- `*.pt` tensor files (case/image-view tensors)
- `metadata.csv` in the same folder, containing at least:
  - `case_id`
  - `label`
  - `Age` (used when multi-modal input is enabled)
  - `Weight` (used when multi-modal input is enabled)

Filename/case grouping convention (matching the current loader behavior):
- Files for the same case must share the same prefix before the final underscore.
- The part after the final underscore is interpreted as the view index/suffix.
- Example of one case with two views:
  - `CASE_AAA111_0000_0.pt`
  - `CASE_AAA111_0000_1.pt`
- In this example, `CASE_AAA111_0000` is the case identifier prefix and both files are grouped together as one patient/case.

## Running experiments

From repository root:

```bash
python src/main.py
```

The script loops through all entries in `CONFIG.training_config`, runs multiple random seeds, and writes reports to the configured output files.

## Configuration reference

All variables below are defined in **src/local_config.py**.

### Output flags and paths

- `with_accuracy_outputs`  
  Flag intended to control writing accuracy results.  

- `with_per_epoch_output`  
  If `True`, per-epoch validation metrics are collected and written.

- `log_epoch_stats`  
  Toggle for epoch-level metric logging behavior.

- `accuracy_result_outputs`  
  File path where aggregate run/fold metrics are appended.

- `per_epoch_report_output`  
  File path where per-epoch metric arrays are appended.

- `summary_report_output`  
  File path for post-processed best-epoch summary generated from per-epoch logs.

### Training controls

- `training_epochs`  
  Number of epochs used in each fold training run.

- `frozen_epoch_count`  
  Epoch index threshold for unfreezing the backbone during fine-tuning.

- `random_seeds_to_run_for`  
  Number of random seeds (0 to N-1) to execute end-to-end.

### Mode toggle

- `multi_modal`  
  Global mode indicator printed at startup (actual per-run behavior is controlled by each `training_config` entry).

### Experiment matrix

- `training_config`  
  List of run tuples with shape:
  `(multi_modal, weights_path, pooling, view)`

  Where:
  - `multi_modal` (`bool`): enables/disables age+weight metadata branch
  - `weights_path` (`str | None`): optional pretrained checkpoint path for the backbone
  - `pooling` (`str`): feature aggregation strategy (e.g., `"attention"`, `"max"`, `"average"`)
  - `view` (`str`): view selection strategy (configured values include `"single"`, `"multi"`, `"mixed"`)

## Outputs

The pipeline writes results under the configured output folder (default `results\`):
- accuracy/cross-validation report
- per-epoch metric arrays
- summary of best metric epochs (when enabled)