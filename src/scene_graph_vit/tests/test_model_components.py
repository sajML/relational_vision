import torch
import pytest
from src.scene_graph_vit.models.detectorT import ViTDetector
from src.scene_graph_vit.utils.config import resolve_cfg

def test_model_forward():
    """Test end-to-end model forward pass with dummy data."""
    config = {
        "backbone": {
            "type": "owlvit",
            "hf_id": "google/owlvit-base-patch32", 
            "force_cpu": True,
            "return_grid": True,
            "interpolate_pos_encoding": True
        },
        "head": {
            "type": "owlvit_heads_only",
            "num_classes": 150,
            "hf_id": "google/owlvit-base-patch32",
            "device": "cpu"
        },
        "rel_attn": {
            "top_instances": 16,
            "top_pairs": 64,
            "proj_depth": 2
        }
    }
    
    model = ViTDetector(config)
    
    # Test forward pass
    batch_size, channels, height, width = 2, 3, 896, 896
    images = torch.rand(batch_size, channels, height, width)
    
    outputs = model(images)
    
    # Check output structure - verify all expected keys are present
    expected_keys = ["logits", "boxes", "sel_indices", "rel_embs", "rel_pairs", 
                     "rel_scores", "diag_rel_embs", "diag_indices", "diag_scores"]
    for key in expected_keys:
        assert key in outputs, f"Missing expected key: {key}"
    
    # Check shapes
    num_instances = config["rel_attn"]["top_instances"]
    num_pairs = config["rel_attn"]["top_pairs"]
    
    # Basic batch dimension checks
    assert outputs["logits"].shape[0] == batch_size
    assert outputs["boxes"].shape[0] == batch_size
    assert outputs["sel_indices"].shape[0] == batch_size
    
    # The sel_indices contains diag_indices + flattened rel_pairs
    # So it should be [B, top_instances + 2*top_pairs] (subject + object for each pair)
    expected_sel_length = num_instances + 2 * num_pairs
    assert outputs["sel_indices"].shape[1] == expected_sel_length, \
        f"Expected sel_indices shape[1] to be {expected_sel_length}, got {outputs['sel_indices'].shape[1]}"
    
    # Logits and boxes should match the number of selected tokens
    assert outputs["logits"].shape[1] == expected_sel_length
    assert outputs["boxes"].shape[1] == expected_sel_length
    
    # Check relationship attention outputs
    assert outputs["rel_pairs"].shape == (batch_size, num_pairs, 2)
    assert outputs["diag_indices"].shape == (batch_size, num_instances)
    
    # Check that logits have correct class dimension
    assert outputs["logits"].shape[2] == config["head"]["num_classes"]
    
    # Check box format (should be 4D - cxcywh)
    assert outputs["boxes"].shape[2] == 4
    
    print("✓ Model forward pass test passed")

if __name__ == "__main__":
    test_model_forward()



