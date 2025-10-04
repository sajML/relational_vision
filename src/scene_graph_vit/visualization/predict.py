from __future__ import annotations
from typing import Dict, List, Tuple, Optional


import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.ops import nms as tv_nms, box_iou as tv_box_iou
from ..utils.box_ops import box_cxcywh_to_xyxy


# ----------------------- Torch-based utilities -----------------------
def cxcywh_to_xyxy(box: np.ndarray, W: int, H: int) -> np.ndarray:
    """Use the project’s Torch op, return NumPy for plotting."""
    t = torch.as_tensor(box, dtype=torch.float32).view(1, 4)     # [1,4] cx,cy,w,h in [0..1]
    xyxy = box_cxcywh_to_xyxy(t)[0]                            # [4] in [0..1] coords
    scale = torch.tensor([W, H, W, H], dtype=torch.float32)
    xyxy = (xyxy * scale).cpu().numpy().astype(np.float32)
    return xyxy

def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """Delegate to torchvision’s box_iou for a single pair."""
    A = torch.as_tensor(a, dtype=torch.float32).view(1, 4)
    B = torch.as_tensor(b, dtype=torch.float32).view(1, 4)
    return float(tv_box_iou(A, B)[0, 0].item())

def nms(boxes: List[np.ndarray], scores: List[float], iou_thr: float = 0.5) -> List[int]:
    """Use torchvision.ops.nms and return Python indices."""
    B = torch.as_tensor(np.asarray(boxes, dtype=np.float32))  # [N,4] absolute xyxy
    S = torch.as_tensor(np.asarray(scores, dtype=np.float32)) # [N]
    keep = tv_nms(B, S, float(iou_thr))  # Use the aliased import
    return keep.cpu().numpy().tolist()

def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


# ----------------------- Decoding: detections -----------------------

def _flatten_logits_boxes(logits: torch.Tensor, boxes: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Return (pl, pb) as NumPy with shapes [Q, K] and [Q, 4]."""
    if logits.ndim == 4:  # [B,H,W,K]
        _, H, W, K = logits.shape
        Q = H * W
        pl = _to_numpy(logits[0].reshape(Q, K))
        pb = _to_numpy(boxes[0].reshape(Q, 4))
    else:  # [B,Q,K]
        _, Q, K = logits.shape
        pl = _to_numpy(logits[0])
        pb = _to_numpy(boxes[0])
    return pl, pb


def decode_detections(
    out: Dict[str, torch.Tensor],
    *,
    image_size: Tuple[int, int],               # (W,H)
    class_names: Optional[List[str]] = None,
    conf_thresh: float = 0.3,
    topk: int = 100,
    nms_iou: Optional[float] = 0.5,
) -> List[Dict]:
    """Turn model outputs into detection dicts with token indices preserved.

    Returns a list of dicts: {token, label_id, label_name, score, box_xyxy}
    """
    logits = out["logits"]
    boxes  = out["boxes"] if "boxes" in out else out.get("pred_boxes")
    if boxes is None:
        raise KeyError("Expected 'boxes' or 'pred_boxes' in model output")

    pl, pb = _flatten_logits_boxes(logits, boxes)  # [Q,K], [Q,4]
    W, H = image_size
    K = pl.shape[1]
    eos = K - 1

    probs = F.softmax(torch.from_numpy(pl), dim=-1).numpy()  # [Q,K]
    cls_scores = probs[:, :eos].max(axis=-1)
    cls_ids = probs[:, :eos].argmax(axis=-1)

    if class_names is None:
        class_names = [f"class_{i}" for i in range(eos)]

    # build candidates
    cand = []
    for q in range(pl.shape[0]):
        s = float(cls_scores[q])
        if s < conf_thresh:
            continue
        box_xyxy = cxcywh_to_xyxy(pb[q], W, H)
        lab = int(cls_ids[q])
        name = class_names[lab] if 0 <= lab < len(class_names) else f"class_{lab}"
        cand.append({
            "token": q,
            "label_id": lab,
            "label_name": name,
            "score": s,
            "box_xyxy": box_xyxy,
        })

    # sort & topk
    cand.sort(key=lambda d: d["score"], reverse=True)
    cand = cand[:topk]

    # optional NMS
    if nms_iou is not None and len(cand) > 0:
        boxes_np = [c["box_xyxy"] for c in cand]
        scores = [c["score"] for c in cand]
        keep = nms(boxes_np, scores, iou_thr=nms_iou)
        cand = [cand[i] for i in keep]

    return cand


# ----------------------- Decoding: relations -----------------------

def decode_relations(
    out: Dict[str, torch.Tensor],
    detections: List[Dict],
    *,
    top_rel: int = 50,
    pred_text_bank: Optional[np.ndarray] = None,   # [Cpred, D], ideally L2-normalized
    temp_pred: float = 50.0,
    predicate_names: Optional[List[str]] = None,
) -> List[Dict]:
    """Use RA outputs to form relations between kept detections.
    If `pred_text_bank` is provided and `rel_embs` present, computes OV predicate labels.
    """
    # Support either key name
    pairs_key = 'rel_pairs' if 'rel_pairs' in out else ('pair_idx' if 'pair_idx' in out else None)
    if pairs_key is None or ("rel_scores" not in out):
        return []

    pairs = _to_numpy(out[pairs_key][0]).astype(np.int64)     # [Krel,2] absolute token indices
    scores = _to_numpy(out["rel_scores"][0])                  # [Krel]
    order = np.argsort(-scores)
    pairs = pairs[order]
    scores = scores[order]

    # map from token index -> detection index (choose best score per token)
    tok2det: Dict[int, int] = {}
    for i, det in enumerate(detections):
        t = int(det["token"])  # token id from detection decoding
        if t not in tok2det:
            tok2det[t] = i

    # optional OV predicate scores
    have_pred = (pred_text_bank is not None) and ("rel_embs" in out)
    if have_pred:
        emb = _to_numpy(out["rel_embs"][0])              # [Krel, D]
        # l2 normalize
        n = np.linalg.norm(emb, axis=-1, keepdims=True) + 1e-7
        emb = emb / n
        bank = pred_text_bank.astype(np.float32)
        # (optionally) normalize bank if needed
        # bank = bank / (np.linalg.norm(bank, axis=-1, keepdims=True) + 1e-7)

    rels: List[Dict] = []
    max_k = int(min(top_rel, pairs.shape[0]))
    for k in range(max_k):
        ti, tj = int(pairs[k, 0]), int(pairs[k, 1])
        if (ti not in tok2det) or (tj not in tok2det):
            continue
        si, oi = tok2det[ti], tok2det[tj]
        rel = {"sub": si, "obj": oi, "score": float(scores[k])}
        if have_pred:
            v = emb[k:k+1]                             # [1,D]
            logits = temp_pred * (v @ bank.T)          # [1,C]
            p = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()[0]
            pid = int(np.argmax(p))
            rel["pred_id"] = pid
            rel["pred_score"] = float(p[pid])
            if predicate_names is not None and 0 <= pid < len(predicate_names):
                rel["pred_name"] = predicate_names[pid]
        rels.append(rel)
    return rels

