#!/usr/bin/env python3
"""
Debug script to analyze why object detection is failing completely.
This script loads a checkpoint and analyzes the logits, probabilities, and confidence scores.
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# Add the source directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from scene_graph_vit.utils.config import load_cfg, resolve_cfg, validate_config, adapt_yaml_to_model_cfg
from scene_graph_vit.models.detectorT import ViTDetector
from PIL import Image


def analyze_detection_failure(checkpoint_path, config_path, image_path):
    """Detailed analysis of detection failure."""
    
    print("=== DETECTION FAILURE ANALYSIS ===")
    
    # Load configuration
    print(f"Loading config from: {config_path}")
    raw_cfg = load_cfg(config_path)
    model_cfg = adapt_yaml_to_model_cfg(raw_cfg)
    validate_config(model_cfg)
    
    # Load model
    print(f"Loading checkpoint from: {checkpoint_path}")
    model = ViTDetector(model_cfg).eval()
    state = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    elif "model" in state: 
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    
    # Load and preprocess image
    print(f"Loading image from: {image_path}")
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    print(f"Original image size: {W}x{H}")
    
    # Convert to tensor
    arr = np.asarray(img, np.float32) / 255.0  # HxWx3 in [0,1]
    img_tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    print(f"Input tensor shape: {img_tensor.shape}")
    
    # Run inference
    print("\n=== RUNNING INFERENCE ===")
    with torch.no_grad():
        out = model(img_tensor, train=False)
    
    # Analyze outputs
    logits = out["logits"]  # [B, N, K]
    boxes = out["boxes"]    # [B, N, 4]
    
    print(f"Logits shape: {logits.shape}")
    print(f"Boxes shape: {boxes.shape}")
    
    # Flatten for analysis
    B, N, K = logits.shape
    logits_flat = logits[0]  # [N, K]
    boxes_flat = boxes[0]    # [N, 4]
    
    print(f"\n=== LOGITS ANALYSIS ===")
    print(f"Logits min/max: {logits_flat.min():.4f}/{logits_flat.max():.4f}")
    print(f"Logits mean: {logits_flat.mean():.4f}")
    print(f"Logits std: {logits_flat.std():.4f}")
    
    # Analyze class-wise logits (excluding EOS)
    eos_channel = K - 1
    class_logits = logits_flat[:, :eos_channel]  # [N, K-1]
    eos_logits = logits_flat[:, eos_channel]     # [N]
    
    print(f"\nClass logits (excluding EOS):")
    print(f"  Shape: {class_logits.shape}")
    print(f"  Min/Max: {class_logits.min():.4f}/{class_logits.max():.4f}")
    print(f"  Mean: {class_logits.mean():.4f}")
    
    print(f"\nEOS logits:")
    print(f"  Shape: {eos_logits.shape}")
    print(f"  Min/Max: {eos_logits.min():.4f}/{eos_logits.max():.4f}")
    print(f"  Mean: {eos_logits.mean():.4f}")
    
    # Convert to probabilities
    print(f"\n=== PROBABILITY ANALYSIS ===")
    probs = F.softmax(logits_flat, dim=-1)  # [N, K]
    class_probs = probs[:, :eos_channel]    # [N, K-1]
    eos_probs = probs[:, eos_channel]       # [N]
    
    print(f"Class probabilities:")
    print(f"  Min/Max: {class_probs.min():.6f}/{class_probs.max():.6f}")
    print(f"  Mean: {class_probs.mean():.6f}")
    
    print(f"EOS probabilities:")
    print(f"  Min/Max: {eos_probs.min():.6f}/{eos_probs.max():.6f}")
    print(f"  Mean: {eos_probs.mean():.6f}")
    
    # Get max class probabilities and IDs
    max_class_probs, max_class_ids = class_probs.max(dim=-1)
    
    print(f"\n=== CONFIDENCE ANALYSIS ===")
    print(f"Max class confidence scores:")
    print(f"  Min/Max: {max_class_probs.min():.6f}/{max_class_probs.max():.6f}")
    print(f"  Mean: {max_class_probs.mean():.6f}")
    print(f"  Std: {max_class_probs.std():.6f}")
    
    # Count how many tokens pass different thresholds
    thresholds = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    for thresh in thresholds:
        count = (max_class_probs > thresh).sum().item()
        print(f"  Tokens with confidence > {thresh}: {count}/{N} ({100*count/N:.2f}%)")
    
    # Analyze top confident detections
    print(f"\n=== TOP DETECTIONS ANALYSIS ===")
    sorted_indices = torch.argsort(max_class_probs, descending=True)
    top_k = min(20, N)
    
    print(f"Top {top_k} most confident detections:")
    for i in range(top_k):
        idx = sorted_indices[i]
        conf = max_class_probs[idx].item()
        class_id = max_class_ids[idx].item()
        eos_prob = eos_probs[idx].item()
        
        print(f"  Token {idx:4d}: conf={conf:.6f}, class={class_id:2d}, eos_prob={eos_prob:.6f}")
    
    # Analyze class distribution
    print(f"\n=== CLASS DISTRIBUTION ANALYSIS ===")
    class_counts = torch.bincount(max_class_ids, minlength=eos_channel)
    top_classes = torch.argsort(class_counts, descending=True)[:10]
    
    print("Most predicted classes:")
    for i, class_id in enumerate(top_classes):
        count = class_counts[class_id].item()
        print(f"  Class {class_id:2d}: {count:5d} tokens ({100*count/N:.2f}%)")
    
    # Load class names if available
    try:
        import json
        with open("dataset/raw_DATA/object_synsets.json", "r") as f:
            class_dict = json.load(f)
        id_to_name = {v-1: k for k, v in class_dict.items()}  # Convert to 0-based indexing
        
        print("\nTop classes with names:")
        for i, class_id in enumerate(top_classes[:5]):
            count = class_counts[class_id].item()
            name = id_to_name.get(class_id.item(), f"unknown_{class_id}")
            print(f"  Class {class_id:2d} ({name}): {count:5d} tokens")
            
    except Exception as e:
        print(f"Could not load class names: {e}")
    
    # Analyze boxes
    print(f"\n=== BOXES ANALYSIS ===")
    print(f"Box coordinates (cx, cy, w, h) in [0,1]:")
    print(f"  cx - min/max: {boxes_flat[:, 0].min():.4f}/{boxes_flat[:, 0].max():.4f}")
    print(f"  cy - min/max: {boxes_flat[:, 1].min():.4f}/{boxes_flat[:, 1].max():.4f}")
    print(f"  w  - min/max: {boxes_flat[:, 2].min():.4f}/{boxes_flat[:, 2].max():.4f}")
    print(f"  h  - min/max: {boxes_flat[:, 3].min():.4f}/{boxes_flat[:, 3].max():.4f}")
    
    # Check for degenerate boxes
    box_areas = boxes_flat[:, 2] * boxes_flat[:, 3]  # w * h
    valid_boxes = (box_areas > 1e-6).sum().item()
    print(f"  Valid boxes (area > 1e-6): {valid_boxes}/{N}")
    print(f"  Box areas - min/max: {box_areas.min():.8f}/{box_areas.max():.8f}")
    
    return {
        'max_class_probs': max_class_probs,
        'max_class_ids': max_class_ids,
        'eos_probs': eos_probs,
        'boxes': boxes_flat,
        'logits': logits_flat
    }


if __name__ == "__main__":
    checkpoint = "experiments/runs/OWV_expT1/final.pt"
    config = "configs/base.yaml"
    image = "dataset/raw_DATA/VG_100K/10.jpg"
    
    results = analyze_detection_failure(checkpoint, config, image)
    
    print("\n=== SUMMARY ===")
    max_conf = results['max_class_probs'].max().item()
    print(f"Maximum confidence achieved: {max_conf:.6f}")
    
    if max_conf < 0.1:
        print("🚨 CRITICAL ISSUE: Maximum confidence is extremely low!")
        print("   This suggests the model was not properly trained or there's a fundamental issue.")
        print("   Possible causes:")
        print("   1. Model weights not properly loaded")
        print("   2. Training did not converge")
        print("   3. Wrong loss function or optimization")
        print("   4. Open vocabulary text bank mismatch")
        print("   5. Backbone frozen during training")
    elif max_conf < 0.3:
        print("⚠️  WARNING: Confidence scores are very low.")
        print("   Consider lowering the detection threshold or investigating training.")
    else:
        print("✅ Confidence scores look reasonable.")
