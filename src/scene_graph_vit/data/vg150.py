from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF

def _xyxy_to_cxcywh_norm(b: List[float], W: int, H: int) -> List[float]:
    x0,y0,x1,y1 = b
    cx = ((x0 + x1) / 2.0) / W
    cy = ((y0 + y1) / 2.0) / H
    w  = (x1 - x0) / W
    h  = (y1 - y0) / H
    return [cx, cy, w, h]

def _load_names(path: str) -> List[str]:
    # keep SAME order
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    keys = sorted(d.keys(), key=str.lower)
    return keys  # id 1..N


def letterbox(img: Image.Image, size: int) -> Tuple[Image.Image, float, Tuple[int,int]]:
    W, H = img.size
    scale = min(size / W, size / H)
    newW, newH = int(W * scale), int(H * scale)
    img_resized = img.resize((newW, newH), Image.BICUBIC)
    padW, padH = size - newW, size - newH
    pad_left = padW // 2; pad_top = padH // 2
    out = Image.new("RGB", (size, size), (114,114,114))
    out.paste(img_resized, (pad_left, pad_top))
    return out, scale, (pad_left, pad_top)

class VG150(Dataset):
    """
    Returns:
      img: FloatTensor [3,H,W], in [0,1]
      target: Dict with:
        - boxes:  FloatTensor [N,4] cx,cy,w,h normalized to [0,1]
        - labels: LongTensor [N] (0..C-1)
        - relations: LongTensor [R,3] (sub_idx, obj_idx, pred_id 0..P-1)
    """
    def __init__(self, img_root, ann_path, split: str = "train", transform = None):
        self.img_root = Path(img_root) 
        self.size = 896 # Default image size for ViT 
        self.t = transform
        with open(ann_path, "r") as f:
            data = json.load(f)
        # Load object and predicate names from meta section
        meta = data.get("meta", {})
        self._obj_names = meta.get("object_names", [])
        self._pred_names = meta.get("predicate_names", [])
        self.recs = data["data"][split]

    def __len__(self): return len(self.recs)
    def object_names(self): return self._obj_names
    def predicate_names(self): return self._pred_names

    def __getitem__(self, i: int):
        R = self.recs[i]
        img = Image.open(self.img_root / R["image"]).convert("RGB")
        W, H = img.size

        # image -> fixed square while preserving aspect (letterbox)
        # img = TF.resize(img, [self.size, self.size], antialias=True)
        img, scale, (pad_left, pad_top) = letterbox(img, self.size)

        # boxes -> cxcywh normalized IN ORIGINAL W,H then no need to rescale (model is scale-invariant via backbone)
        boxes_cxcywh = torch.tensor(
            [_xyxy_to_cxcywh_norm(b, W, H) for b in R["boxes"]],
            dtype=torch.float32
        )
        labels_1based = torch.tensor(R["labels"], dtype=torch.int64)  #    1..C
        labels = labels_1based - 1                                    # -> 0..C-1

        # relations: [sub,obj,pred] with 1-based pred ids -> 0-based
        # relations: [sub, obj, pred] (JSON may be 1-based for sub/obj and pred)
        if "rels" in R and len(R["rels"]) > 0:
            rels = torch.tensor(R["rels"], dtype=torch.int64)  # [R,3]
            # Detect if sub/obj look 1-based (max equals N or any zero absent)
            if rels[:, :2].min().item() >= 1:
                rels[:, :2] -= 1                     # -> sub,obj in 0..N-1
            rels[:, 2] -= 1                          # -> pred in 0..P-1
            # Safety: all indices are valid
            N = len(R["boxes"])
            assert (rels[:,0] >= 0).all() and (rels[:,0] < N).all()
            assert (rels[:,1] >= 0).all() and (rels[:,1] < N).all()
        else:
            rels = torch.zeros((0,3), dtype=torch.int64)

        img = TF.to_tensor(img)  # PIL -> Tensor [3,H,W] in [0,1]
        
        # # DEBUG: Print relation info
        # if len(rels) > 0:
        #     print(f"VG150 item {i}: relations shape {rels.shape}, sample: {rels[:3].tolist()}")
        #     print(f"VG150 item {i}: max subject_idx={rels[:,0].max()}, max object_idx={rels[:,1].max()}, num_boxes={len(labels)}")
        
        target = {
            "boxes": boxes_cxcywh, 
            "labels": labels, 
            "relations": rels,  # Should be [R, 3] with [subj_idx, obj_idx, pred_class]
        }
        return img, target

def collate_vg(batch):
    imgs, t = zip(*batch)
    return torch.stack(imgs, 0), list(t)