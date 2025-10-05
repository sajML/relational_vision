"""
Dataset utilities for handling Visual Genome dataset paths.
"""
import os
from pathlib import Path

def get_dataset_path(config_path=None):
    """
    Get the Visual Genome dataset path with priority:
    1. Environment variable VG_DATASET_PATH (set by CLI for Kaggle)
    2. Config path if provided
    3. Default download location
    
    Returns:
        Path: The dataset path to use
    """
    # Check for Kaggle environment variable first
    kaggle_path = os.environ.get("VG_DATASET_PATH")
    if kaggle_path and Path(kaggle_path).exists():
        return Path(kaggle_path)
    
    # Use config path if provided
    if config_path:
        config_path = Path(config_path)
        if config_path.exists():
            return config_path
    
    # Fall back to default
    return None  # Let existing download logic handle it
