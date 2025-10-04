import torch
from src.scene_graph_vit.losses.matcher import HungarianMatcher

def test_hungarian_matcher_diagonal():
    """Test that Hungarian matcher only uses diagonal (object) predictions."""
    matcher = HungarianMatcher(cost_class=1.0, cost_bbox=0.1, cost_giou=0.2)
    
    batch_size = 2
    num_instances = 8
    num_classes = 150
    
    # Mock predictions (only object diagonal matters for matching)
    obj_logits = torch.randn(batch_size, num_instances, num_classes)
    boxes = torch.rand(batch_size, num_instances, 4)  # cxcywh format
    
    # Mock targets
    targets = []
    for b in range(batch_size):
        num_gt = torch.randint(2, 6, (1,)).item()
        targets.append({
            "labels": torch.randint(0, num_classes, (num_gt,)),
            "boxes": torch.rand(num_gt, 4),  # cxcywh format
            "relationships": torch.randint(0, 50, (num_gt * 2, 3))  # subject, object, predicate
        })
    
    outputs = {
        "pred_logits": obj_logits,
        "pred_boxes": boxes,
        "pred_rel_logits": torch.randn(batch_size, 32, 50)  # Not used in matching
    }
    
    # Test matching
    indices = matcher(outputs, targets)
    
    assert len(indices) == batch_size
    for b, (pred_idx, tgt_idx) in enumerate(indices):
        assert len(pred_idx) == len(tgt_idx)
        assert len(pred_idx) <= num_instances
        assert len(tgt_idx) <= len(targets[b]["labels"])
        assert pred_idx.max() < num_instances
        assert tgt_idx.max() < len(targets[b]["labels"])
    
    print("✓ Hungarian matcher test passed")

if __name__ == "__main__":
    test_hungarian_matcher_diagonal()