#!/usr/bin/env python3
"""Simple debug script to check logits distribution."""

import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

print("Starting debug script...")

# Add the source directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from scene_graph_vit.utils.config import load_cfg, resolve_cfg, validate_config, adapt_yaml_to_model_cfg
    from scene_graph_vit.models.detectorT import ViTDetector
    from PIL import Image
    print("Imports successful")
except Exception as e:
    print(f"Import error: {e}")
    sys.exit(1)

def simple_analysis():
    print("Loading config...")
    raw_cfg = load_cfg("configs/base.yaml")
    model_cfg = adapt_yaml_to_model_cfg(raw_cfg)
    validate_config(model_cfg)
    
    print("Loading model...")
    model = ViTDetector(model_cfg).eval()
    
    print("Loading checkpoint...")
    state = torch.load("experiments/runs/OWV_expT1/final.pt", map_location="cpu")
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    elif "model" in state: 
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    
    print("Loading image...")
    img = Image.open("dataset/raw_DATA/VG_100K/10.jpg").convert("RGB")
    arr = np.asarray(img, np.float32) / 255.0
    img_tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    
    print("Running inference...")
    with torch.no_grad():
        out = model(img_tensor, train=False)
    
    logits = out["logits"][0]  # [N, K]
    print(f"Logits shape: {logits.shape}")
    print(f"Logits range: {logits.min():.4f} to {logits.max():.4f}")
    
    # Check probabilities
    probs = F.softmax(logits, dim=-1)
    class_probs = probs[:, :-1]  # Exclude EOS
    max_probs, _ = class_probs.max(dim=-1)
    
    print(f"Max class probabilities range: {max_probs.min():.6f} to {max_probs.max():.6f}")
    
    # Count detections at different thresholds
    for thresh in [0.01, 0.1, 0.3]:
        count = (max_probs > thresh).sum().item()
        total = max_probs.shape[0]
        print(f"Detections with confidence > {thresh}: {count}/{total}")
    
    return max_probs.max().item()

if __name__ == "__main__":
    try:
        max_confidence = simple_analysis()
        print(f"\nResult: Maximum confidence = {max_confidence:.6f}")
        if max_confidence < 0.1:
            print("ISSUE: Very low confidence scores detected!")
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
