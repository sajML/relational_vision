from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math


import numpy as np
from PIL import Image, ImageDraw, ImageFont
import networkx as nx
import matplotlib.pyplot as plt

# ========================Color + text helpers========================

def get_color_palette(n_colors: int) -> List[Tuple[int, int, int]]:
    """Return n visually distinct RGB colors (0-255)."""
    cmap = plt.cm.get_cmap("tab20", max(1, n_colors))
    cols = cmap(np.arange(max(1, n_colors)))[:, :3]
    return [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in cols]


def _text_with_bg(
    draw: ImageDraw.Draw,
    xy: Tuple[int, int],
    text: str,
    *,
    font: Optional[ImageFont.ImageFont] = None,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    pad: int = 2,
) -> None:
    """Draw text with a solid background rectangle for readability."""
    bbox = draw.textbbox(xy, text, font=font)
    x1, y1, x2, y2 = bbox
    draw.rectangle([x1 - pad, y1 - pad, x2 + pad, y2 + pad], fill=bg_color)
    draw.text(xy, text, fill=text_color, font=font)


# ========================Core scene drawing========================

_DEF_FONT: Optional[ImageFont.ImageFont] = None # Pillow default font

def _center(box_xyxy: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box_xyxy
    return (float(x1 + x2) / 2.0, float(y1 + y2) / 2.0)

def _draw_arrow(
    draw: ImageDraw.Draw,
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    *,
    color: Tuple[int, int, int] = (200, 200, 200),
    width: int = 2,
    head_len: float = 10.0,
    head_width: float = 6.0,
    ) -> None:
    """Draw a line with a triangular arrowhead pointing from p1 -> p2."""
    (x1, y1), (x2, y2) = p1, p2
    draw.line([x1, y1, x2, y2], fill=color, width=width)
    ang = math.atan2(y2 - y1, x2 - x1)
    hx = x2 - head_len * math.cos(ang)
    hy = y2 - head_len * math.sin(ang)
    left_ang = ang + math.pi / 2
    right_ang = ang - math.pi / 2
    lx = hx + (head_width / 2.0) * math.cos(left_ang)
    ly = hy + (head_width / 2.0) * math.sin(left_ang)
    rx = hx + (head_width / 2.0) * math.cos(right_ang)
    ry = hy + (head_width / 2.0) * math.sin(right_ang)
    draw.polygon([(x2, y2), (lx, ly), (rx, ry)], fill=color)


@dataclass
class Det:
    token: int
    label_id: int
    label_name: str
    score: float
    box_xyxy: Tuple[float, float, float, float]


@dataclass
class Rel:
    sub: int                # index into detections list
    obj: int                # index into detections list
    score: float
    pred_id:    Optional[int] = None
    pred_name:  Optional[str] = None
    pred_score: Optional[float] = None


def draw_scene(
    image: Image.Image,
    detections: List[Dict],
    relations: List[Dict],
    *,
    draw_scores: bool = True,
    edge_limit: int = 50,
    palette: Optional[List[Tuple[int, int, int]]] = None,
) -> Image.Image:
    """Overlay detections + relations directly on the image and return a new PIL image."""
    img = image.convert("RGB").copy()
    drw = ImageDraw.Draw(img)

    n_det = max(1, len(detections))
    colors = palette or get_color_palette(n_det)

    # Draw detections
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = map(float, det["box_xyxy"])
        color = colors[i % len(colors)]
        drw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = det.get("label_name", f"obj_{i}")
        if draw_scores and ("score" in det):
            label = f"{label} {det['score']:.2f}"
        _text_with_bg(drw, (int(x1) + 3, int(y1) + 3), label, font=_DEF_FONT, 
                      text_color=(255, 255, 255), bg_color=(0, 0, 0))

    # Draw relations (subset) - fix variable scoping
    for r in relations[:edge_limit]:
        si, oi = int(r.get("sub", -1)), int(r.get("obj", -1))
        if not (0 <= si < len(detections) and 0 <= oi < len(detections)):
            continue
        p1 = _center(tuple(detections[si]["box_xyxy"]))
        p2 = _center(tuple(detections[oi]["box_xyxy"]))
        _draw_arrow(drw, p1, p2, color=(220, 220, 220), width=2)

        # predicate label near midpoint - moved inside loop
        mid = ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)
        if "pred_name" in r and r["pred_name"] is not None:
            txt = r["pred_name"]
            if draw_scores and ("pred_score" in r) and r["pred_score"] is not None:
                txt += f" ({r['pred_score']:.2f})"
            _text_with_bg(drw, (int(mid[0]) + 4, int(mid[1]) + 4), txt, 
                          font=_DEF_FONT, text_color=(255, 255, 255), bg_color=(0, 0, 0))
        elif draw_scores and ("score" in r):
            _text_with_bg(drw, (int(mid[0]) + 4, int(mid[1]) + 4), f"rel {r['score']:.2f}", 
                          font=_DEF_FONT, text_color=(255, 255, 255), bg_color=(0, 0, 0))

    return img


# ========================combined figure========================

def print_scene_graph(
    detections: List[Dict],
    relations: List[Dict],
    *,
    layout: str = "spring",
    seed: int = 42,
):
    """Return a NetworkX DiGraph with labels on nodes and edges."""

    G = nx.DiGraph()
    for i, det in enumerate(detections):
        G.add_node(i, label=det.get("label_name", f"obj_{i}"))
    for rel in relations:
        si, oi = int(rel.get("sub", -1)), int(rel.get("obj", -1))
        if 0 <= si < len(detections) and 0 <= oi < len(detections):
            lab = rel.get("pred_name", "related-to")
            G.add_edge(si, oi, label=lab)


    # positions
    if layout == "spring":
        pos = nx.spring_layout(G, seed=seed)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, seed=seed)


    return G, pos

def show_side_by_side(
    image: Image.Image,
    detections: List[Dict],
    relations: List[Dict],
    *,
    figsize: Tuple[int, int] = (16, 7),
    layout: str = "spring",
    seed: int = 42,
) -> plt.Figure:
    """Display (image overlay) | (graph) side by side using matplotlib."""
    
    vis = draw_scene(image, detections, relations)
    G, pos = print_scene_graph(detections, relations, layout=layout, seed=seed)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.imshow(vis)
    ax1.axis("off")

    node_labels = {n: d["label"] for n, d in G.nodes(data=True)}
    edge_labels = nx.get_edge_attributes(G, "label")

    nx.draw(G, pos, with_labels=False, node_color="#9ecae1", node_size=1200, ax=ax2, arrows=True)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, ax=ax2)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="#333333", font_size=9, ax=ax2)
    ax2.axis("off")
    fig.tight_layout()
    return fig


# ==============================================================================

def interactive_scene_graph(
    image: Image.Image,
    detections: List[Dict],
    relations: List[Dict],
    figsize: Tuple[int, int] = (16, 7)
) -> None:
    """Notebook helper that shows the overlay + simple directed graph."""
    show_side_by_side(image, detections, relations, figsize=figsize)


def plot_metrics(metrics: Dict[str, List[float]], figsize: Tuple[int, int] = (10, 6)) -> None:
    """Plot training curves given a dict of name -> list of scalars. Requires matplotlib."""

    plt.figure(figsize=figsize)
    for name, values in metrics.items():
        plt.plot(values, label=name)
    plt.legend(); plt.grid(True)
    plt.xlabel("Steps"); plt.ylabel("Value")
    plt.tight_layout()
    plt.show()
