"""Model Generation module for Scene Graph ViT..

This package provides components for Vision Transformer-based scene graph generation:
- ViTDetector: Main detector class that integrates all components
- Backbone modules: Vision encoders (OWL-ViT, DINOv2)
- Relationship Attention: Top-K relation detection module
- Detection Heads: Classification and box regression modules
"""


from .detectorT import ViTDetector

from .relationship_attentionT import RelationshipAttention

from .backboneT import (
    VisionBackbone,
    _OWLViTAdapter,
    _DINOv2Adapter,
)

from .headT import (
    create_head,
    DetHead,
    OwlViTHeadsOnly,
)

__all__ = [
    # Main detector
    "ViTDetector",
    
    # Relationship attention
    "RelationshipAttention",
    
    # Backbones
    "VisionBackbone",
    "_OWLViTAdapter",
    "_DINOv2Adapter",
    
    # Heads
    "create_head",
    "DetHead",
    "OwlViTHeadsOnly",
]