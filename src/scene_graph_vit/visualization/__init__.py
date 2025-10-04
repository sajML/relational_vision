"""
Scene Graph Visualization Module

This module provides utilities for visualizing scene graphs including:
- Decoding model outputs into detections and relations
- Drawing scene graphs on images with bounding boxes and relation arrows
- Creating side-by-side visualizations with networkx graphs
- Plotting training metrics
"""

from .predict import (
    decode_detections,
    decode_relations,
    cxcywh_to_xyxy,
    iou_xyxy,
    nms,
)

from .plot import (
    draw_scene,
    show_side_by_side,
    print_scene_graph,
    interactive_scene_graph,
    plot_metrics,
    get_color_palette,
    Det,
    Rel,
)

__all__ = [
    # Prediction/decoding functions
    "decode_detections",
    "decode_relations",
    "cxcywh_to_xyxy", 
    "iou_xyxy",
    "nms",
    
    # Plotting functions
    "draw_scene",
    "show_side_by_side",
    "print_scene_graph", 
    "interactive_scene_graph",
    "plot_metrics",
    "get_color_palette",
    
    # Data classes
    "Det",
    "Rel",
]