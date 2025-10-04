# src/scene_graph_vit/viz.py
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List

import numpy as np
from PIL import Image
import torch

from .utils.config import load_cfg, resolve_cfg, validate_config, adapt_yaml_to_model_cfg
from .utils.checkpointing import _load_latest_step
from .models.detectorT import ViTDetector
from .visualization.predict import decode_detections, decode_relations, _to_numpy, cxcywh_to_xyxy, iou_xyxy
from .visualization.plot import draw_scene, show_side_by_side


# Image helper
def _prep_image(img_path: Path):
    img = Image.open(img_path).convert("RGB")
    arr = np.asarray(img, np.float32) / 255.0  # HxWx3 in [0,1]
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    return t, img


# ----------------------------- Auto-threshold decoding (optional) -----------------------------
def _decode_auto(
    out: Dict[str, torch.Tensor],
    image_size: Tuple[int, int],
    class_names: Optional[List[str]],
    start_conf: float,
    topk: int,
    nms_iou: Optional[float],
):
    ladder = [start_conf, 0.20, 0.10, 0.05, 0.02, 0.01, 0.005]
    for c in ladder:
        dets = decode_detections(
            out,
            image_size=image_size,
            class_names=class_names,
            conf_thresh=c,
            topk=topk,
            nms_iou=nms_iou,
        )
        if len(dets) > 0:
            if c != start_conf:
                print(f"[auto] lowered conf_thresh {start_conf} → {c} to keep {len(dets)} dets")
            return dets, c
    dets = decode_detections(
        out,
        image_size=image_size,
        class_names=class_names,
        conf_thresh=0.0,
        topk=max(20, topk),
        nms_iou=nms_iou,
    )
    print(f"[auto] no dets at min ladder; fallback conf=0.0 (kept {len(dets)})")
    return dets, 0.0


def decode_relations_robust(
    out: Dict[str, torch.Tensor],
    detections: List[Dict],
    *,
    top_rel: int = 50,
    pred_text_bank: Optional[np.ndarray] = None,
    temp_pred: float = 50.0,
    predicate_names: Optional[List[str]] = None,
) -> List[Dict]:
    """Robust relation decoding that handles token index mismatches."""
    
    # Support either key name
    pairs_key = 'rel_pairs' if 'rel_pairs' in out else ('pair_idx' if 'pair_idx' in out else None)
    if pairs_key is None or ("rel_scores" not in out):
        print("[WARNING] Missing relation outputs in model!")
        return []

    pairs = _to_numpy(out[pairs_key][0]).astype(np.int64)     # [Krel,2] absolute token indices
    scores = _to_numpy(out["rel_scores"][0])                  # [Krel]
    
    if len(pairs) == 0 or len(scores) == 0:
        print("[WARNING] Empty relation pairs or scores!")
        return []
    
    order = np.argsort(-scores)
    pairs = pairs[order]
    scores = scores[order]

    print(f"[DEBUG] Raw pairs shape: {pairs.shape}, scores shape: {scores.shape}")
    print(f"[DEBUG] Pair token range: {pairs.min()} to {pairs.max()}")
    print(f"[DEBUG] Score range: {scores.min():.4f} to {scores.max():.4f}")

    # Build a more flexible token mapping
    # Instead of exact token matching, use spatial overlap
    if 'logits' in out and 'boxes' in out:
        # Get all model predictions (before filtering)
        all_logits = out['logits'][0]  # [N, K]
        all_boxes = out['boxes'][0]   # [N, 4]
        
        # Create a mapping from all tokens to detection indices based on IoU
        W, H = 896, 896  # Default size, should match your preprocessing
        
        # Convert all model boxes to absolute coordinates
        all_boxes_abs = []
        for i in range(len(all_boxes)):
            box_abs = cxcywh_to_xyxy(all_boxes[i].cpu().numpy(), W, H)
            all_boxes_abs.append(box_abs)
        
        # For each detection, find the best matching model token by IoU
        token_to_det = {}
        det_to_token = {}
        
        for det_idx, det in enumerate(detections):
            det_box = det['box_xyxy']
            best_iou = -1
            best_token = -1
            
            for token_idx in range(len(all_boxes_abs)):
                if token_idx in token_to_det:  # Already assigned
                    continue
                    
                token_box = all_boxes_abs[token_idx]
                iou = iou_xyxy(det_box, token_box)
                
                if iou > best_iou and iou > 0.5:  # Minimum IoU threshold
                    best_iou = iou
                    best_token = token_idx
            
            if best_token >= 0:
                token_to_det[best_token] = det_idx
                det_to_token[det_idx] = best_token
                print(f"[DEBUG] Mapped token {best_token} -> detection {det_idx} (IoU: {best_iou:.3f})")
    else:
        # Fallback to original token-based mapping
        token_to_det = {}
        for i, det in enumerate(detections):
            t = int(det["token"])
            if t not in token_to_det:
                token_to_det[t] = i

    print(f"[DEBUG] Token-to-detection mapping: {len(token_to_det)} mappings")

    # Optional OV predicate processing
    have_pred = (pred_text_bank is not None) and ("rel_embs" in out)
    if have_pred:
        emb = _to_numpy(out["rel_embs"][0])
        n = np.linalg.norm(emb, axis=-1, keepdims=True) + 1e-7
        emb = emb / n
        bank = pred_text_bank.astype(np.float32)

    rels = []
    max_k = min(top_rel, len(pairs))
    valid_pairs = 0
    
    for k in range(max_k):
        ti, tj = int(pairs[k, 0]), int(pairs[k, 1])
        
        if ti not in token_to_det or tj not in token_to_det:
            continue
            
        si, oi = token_to_det[ti], token_to_det[tj]
        if si == oi:  # Skip self-relations
            continue
            
        valid_pairs += 1
        rel = {"sub": si, "obj": oi, "score": float(scores[k])}
        
        if have_pred and k < len(emb):
            v = emb[k:k+1]
            logits = temp_pred * (v @ bank.T)
            p = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()[0]
            pid = int(np.argmax(p))
            rel["pred_id"] = pid
            rel["pred_score"] = float(p[pid])
            if predicate_names is not None and 0 <= pid < len(predicate_names):
                rel["pred_name"] = predicate_names[pid]
        
        rels.append(rel)

    print(f"[DEBUG] Valid pairs found: {valid_pairs}/{max_k}")
    return rels


# -----------------------------
# Main
# -----------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser("Scene-Graph ViT — single-image visualization")
    ap.add_argument("--base", type=str, required=True, help="Path to base config (YAML)")
    ap.add_argument("--ckpt", type=str, required=True, help="Path or dir to checkpoint (final.pt / step_*.pt)")
    ap.add_argument("--image", type=str, required=True, help="Path to an RGB image")
    ap.add_argument("--outdir", type=str, default="/home/romina/SG_Generation/experiments/output", help="Output dir (default: ./outputs)")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--auto-conf", action="store_true", help="Auto-lower conf if no detections")
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--nms", type=float, default=0.50, help=">=0 for NMS IOU, negative to disable")
    ap.add_argument("--top_rel", type=int, default=50)
    ap.add_argument("--show-graph", action="store_true", help="Also save side-by-side graph view")
    args = ap.parse_args(argv)

    base_path = Path(args.base).resolve()
    cfg = load_cfg(str(base_path))
    cfg = resolve_cfg(cfg)
    try:
        validate_config(cfg)
    except Exception:
        pass
    model_cfg = adapt_yaml_to_model_cfg(cfg.get("model_resolved", cfg.get("model", cfg)))

    # Device hints
    device = torch.device("cuda" if torch.cuda.is_available() and not bool(model_cfg.get("backbone", {}).get("force_cpu", False)) else "cpu")
    model_cfg.setdefault("backbone", {}).update({"force_cpu": device.type != "cuda"})
    model_cfg.setdefault("head", {}).update({"force_cpu": device.type != "cuda", "device": device.type})

    # Build model
    model = ViTDetector(model_cfg).to(device)
    model.eval()

    # Load checkpoint
    ckpt_path = Path(args.ckpt)
    if ckpt_path.is_dir():
        ckpt_path = _load_latest_step(ckpt_path)
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    print(f"[ckpt] Loaded: {ckpt_path.as_posix()}")

    # Image
    x, pil_img = _prep_image(Path(args.image))
    x = x.to(device)
    W, H = pil_img.size

    # Forward
    with torch.no_grad():
        out = model(x, train=False)

    # Debug: Print model outputs
    print(f"[DEBUG] Model output keys: {list(out.keys())}")
    for key, value in out.items():
        if isinstance(value, torch.Tensor):
            print(f"[DEBUG] {key}: shape={value.shape}, dtype={value.dtype}")
            if key == "logits":
                print(f"[DEBUG] logits min/max: {value.min().item():.4f}/{value.max().item():.4f}")
            elif key == "rel_scores" and value.numel() > 0:
                print(f"[DEBUG] rel_scores min/max: {value.min().item():.4f}/{value.max().item():.4f}")
                print(f"[DEBUG] rel_scores top 10: {torch.topk(value.flatten(), min(10, value.numel())).values}")
        else:
            print(f"[DEBUG] {key}: {type(value)}")

    # Class names (fallback)
    n_cls = int(model_cfg.get("head", {}).get("num_classes", 0))
    class_names = [f"class_{i}" for i in range(max(1, n_cls))] if n_cls > 0 else None

    # Decode
    nms_iou = float(args.nms) if args.nms >= 0 else None
    if args.auto_conf:
        dets, used_conf = _decode_auto(out, image_size=(W, H), class_names=class_names,
                                       start_conf=float(args.conf), topk=int(args.topk), nms_iou=nms_iou)
    else:
        dets = decode_detections(out, image_size=(W, H), class_names=class_names,
                                 conf_thresh=float(args.conf), topk=int(args.topk), nms_iou=nms_iou)
        used_conf = float(args.conf)

    rels = decode_relations_robust(out, dets, top_rel=int(args.top_rel), pred_text_bank=None, predicate_names=None)

    # Enhanced debugging for relation decoding issues
    print(f"\n[ENHANCED DEBUG] Relationship Detection Analysis:")
    print(f"[DEBUG] Number of detections: {len(dets)}")
    
    # Check if RelationshipAttention is being used at all
    if 'rel_pairs' in out or 'pair_idx' in out:
        pairs_key = 'rel_pairs' if 'rel_pairs' in out else 'pair_idx'
        pairs = out[pairs_key]
        rel_scores = out.get('rel_scores', None)
        
        print(f"[DEBUG] Raw relation outputs:")
        print(f"  - {pairs_key} shape: {pairs.shape if pairs is not None else 'None'}")
        print(f"  - rel_scores shape: {rel_scores.shape if rel_scores is not None else 'None'}")
        
        if pairs is not None and pairs.numel() > 0:
            print(f"  - Pairs tensor min/max: {pairs.min().item()}/{pairs.max().item()}")
            print(f"  - First 5 pairs: {pairs[0][:5] if pairs.shape[1] >= 5 else pairs[0]}")
        
        if rel_scores is not None and rel_scores.numel() > 0:
            print(f"  - Scores min/max/mean: {rel_scores.min().item():.4f}/{rel_scores.max().item():.4f}/{rel_scores.mean().item():.4f}")
            print(f"  - Top 5 scores: {torch.topk(rel_scores.flatten(), min(5, rel_scores.numel())).values}")
            
        # Check token mapping
        if len(dets) > 0:
            print(f"[DEBUG] Detection tokens: {[det['token'] for det in dets[:10]]}")
            max_token = max(det['token'] for det in dets)
            min_token = min(det['token'] for det in dets)
            print(f"[DEBUG] Token range in detections: {min_token} to {max_token}")
            
            if pairs is not None and pairs.numel() > 0:
                pair_tokens = pairs[0].cpu().numpy()  # [K, 2]
                unique_pair_tokens = np.unique(pair_tokens.flatten())
                print(f"[DEBUG] Unique tokens in pairs: {len(unique_pair_tokens)} tokens")
                print(f"[DEBUG] Pair token range: {unique_pair_tokens.min()} to {unique_pair_tokens.max()}")
                
                # Check overlap
                det_tokens = set(det['token'] for det in dets)
                pair_token_set = set(unique_pair_tokens)
                overlap = det_tokens.intersection(pair_token_set)
                print(f"[DEBUG] Token overlap between detections and pairs: {len(overlap)}/{len(det_tokens)} detections have matching pairs")
                
                if len(overlap) == 0:
                    print(f"[WARNING] NO TOKEN OVERLAP! This is why no relations are decoded.")
                    print(f"[WARNING] Detection tokens: {sorted(list(det_tokens))[:10]}...")
                    print(f"[WARNING] Pair tokens: {sorted(list(pair_token_set))[:10]}...")
        
    else:
        print(f"[WARNING] No relation pair outputs found in model!")
        print(f"[WARNING] Available keys: {list(out.keys())}")
        
        # Check if RelationshipAttention module exists and is configured
        print(f"[DEBUG] Model has rel_attn module: {hasattr(model, 'rel_attn') and model.rel_attn is not None}")
        if hasattr(model, 'rel_attn') and model.rel_attn is not None:
            print(f"[DEBUG] rel_attn config - top_instances: {model.rel_attn.top_instances}, top_pairs: {model.rel_attn.top_pairs}")
        
    # Check if model is in correct mode
    print(f"[DEBUG] Model training mode: {model.training}")

    # Additional debugging for relations
    print(f"[DEBUG] Decoded detections: {len(dets)}")
    for i, det in enumerate(dets[:5]):  # Show first 5 detections
        print(f"[DEBUG]   Det {i}: token={det['token']}, label={det['label_name']}, score={det['score']:.3f}")
    
    print(f"[DEBUG] Decoded relations: {len(rels)}")
    for i, rel in enumerate(rels[:5]):  # Show first 5 relations
        print(f"[DEBUG]   Rel {i}: sub={rel['sub']}, obj={rel['obj']}, score={rel['score']:.3f}")
        if 'pred_name' in rel:
            print(f"[DEBUG]     pred_name={rel['pred_name']}, pred_score={rel.get('pred_score', 'N/A')}")

    # Check if model has relation outputs
    has_rel_pairs = 'rel_pairs' in out or 'pair_idx' in out
    has_rel_scores = 'rel_scores' in out
    has_rel_embs = 'rel_embs' in out
    print(f"[DEBUG] Model relation outputs - pairs: {has_rel_pairs}, scores: {has_rel_scores}, embeddings: {has_rel_embs}")
    
    if not has_rel_pairs or not has_rel_scores:
        print(f"[WARNING] Model missing relation outputs! Expected 'rel_pairs'/'pair_idx' and 'rel_scores'")

    # Visualize
    out_dir = Path(args.outdir) if args.outdir else Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = Path(args.image).stem
    overlay_path = out_dir / f"scene_{tag}.png"
    draw_scene(pil_img, dets, rels).save(overlay_path)
    print(f"[saved] {overlay_path.as_posix()}")

    if args.show_graph:
        try:
            fig = show_side_by_side(pil_img, dets, rels)
            graph_path = out_dir / f"graph_{tag}.png"
            fig.savefig(graph_path, bbox_inches="tight")
            print(f"[saved] {graph_path.as_posix()}")
        except Exception as e:
            print(f"[warn] Could not render graph view: {e}")

    print(f"[viz] detections={len(dets)}, relations={len(rels)}, conf_used={used_conf}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
