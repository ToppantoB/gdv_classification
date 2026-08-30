from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import torch

TrainingConfigEntry = Tuple[bool, Optional[str], str, str]

@dataclass(frozen=True)
class Config:
  # Output toggles and file paths.
  with_accuracy_outputs: bool = True
  with_per_epoch_output: bool = True
  log_epoch_stats: bool = True
  accuracy_result_outputs: str = "results/accuracy_results.txt"
  per_epoch_report_output: str = "results/epoch_stats.txt"
  summary_report_output: str = "results/stats_summary.txt"

  # Dataset location relative to the repository root.
  data_path: str = "data/tensors"

  # Training loop controls.
  training_epochs: int = 40
  frozen_epoch_count: int = 5
  random_seeds_to_run_for: int = 5

  # Requested execution device: "auto", "cpu", or "cuda".
  device: str = "auto"

  # Global mode toggle, printed at startup.
  multi_modal: bool = False

  # Runs to execute: (multi_modal, weights_path, pooling, view).
  training_config: List[TrainingConfigEntry] = field(default_factory=lambda: [
      (False, None, "attention", "multi"),
      # (False, None, "attention", "single"),
      # (False, None, "attention", "mixed"),
      # (False, None, "max", "multi"),
      # (False, None, "average", "multi"),
      # (True, None, "attention", "multi"),
      # (True, None, "max", "multi"),
      # (True, None, "average", "multi"),
      # (False, "weights/moco_pretrained_backbone.pth", "attention", "multi"),
      # (True, "weights/moco_pretrained_backbone.pth", "attention", "multi"),
  ])

  def resolve_device(self) -> torch.device:
    if self.device not in {"auto", "cpu", "cuda"}:
      raise ValueError("device must be one of: 'auto', 'cpu', 'cuda'")

    if self.device == "cpu":
      return torch.device("cpu")

    if self.device == "cuda" and not torch.cuda.is_available():
      print("CUDA was requested but is unavailable; falling back to CPU.")
      return torch.device("cpu")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
CONFIG = Config()
DEVICE = CONFIG.resolve_device()