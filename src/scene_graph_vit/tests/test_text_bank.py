import torch
from scene_graph_vit.losses.text_bank import CLIPTextBank

def test_text_bank_encoding():
    """Test CLIP text bank encodes object and predicate vocabularies."""
    text_bank = CLIPTextBank(force_cpu=True)
    
    # Test object vocabulary
    obj_prompts = ["person", "car", "dog", "chair"]
    obj_features = text_bank.encode_prompts(obj_prompts)
    
    assert obj_features.shape[0] == len(obj_prompts)
    assert obj_features.shape[1] > 0  # Feature dimension
    assert torch.allclose(obj_features.norm(dim=-1), torch.ones(len(obj_prompts)), atol=1e-6)
    
    # Test predicate vocabulary  
    pred_prompts = ["on", "holding", "wearing", "riding"]
    pred_features = text_bank.encode_prompts(pred_prompts)
    
    assert pred_features.shape[0] == len(pred_prompts)
    assert pred_features.shape[1] == obj_features.shape[1]  # Same feature dim
    
    # Test similarity computation
    similarity = torch.mm(obj_features, pred_features.t())
    assert similarity.shape == (len(obj_prompts), len(pred_prompts))
    
    print("✓ Text bank encoding test passed")

if __name__ == "__main__":
    test_text_bank_encoding()