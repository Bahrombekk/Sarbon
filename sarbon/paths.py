"""Where data, runs and released weights live. Override with environment variables:

  SARBON_DATA    datasets and recordings      (default C:/data)
  SARBON_RUNS    training and benchmark runs  (default <repo>/runs)
  SARBON_WEIGHTS released model weights       (default <repo>/weights)
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("SARBON_DATA", "C:/data"))
RUNS = Path(os.environ.get("SARBON_RUNS", REPO / "runs"))
WEIGHTS = Path(os.environ.get("SARBON_WEIGHTS", REPO / "weights"))

# the product detector used by the camera tools unless --weights says otherwise
DEFAULT_DETECTOR = WEIGHTS / "sarbon-s-railsafe.pt"
