"""
Legacy entry point for training the House Price Prediction model.
Preserved for backwards compatibility, delegating to the modernized
multi-model training pipeline in scripts/train.py.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train import run_training

if __name__ == "__main__":
    run_training()
