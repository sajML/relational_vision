"""
Probe the backbone + head.

Usage:
  python debug_dataset_probe.py \
    --img-root /path/to/VG150/images \
    --ann /path/to/annotations.json \
    --index 0 \
    --use-detector   # optional: run through ViTDetector instead of raw head
"""

import argparse
from pathlib import Path
import torch
from PIL import Image
import torchvision.transforms.functional as TF


from ..models.backboneT import VisionBackbone
from ..models.headT import OwlViTReuseHeads
from ..data.vg150 import VG150

def cxcywh_to_xyxy_norm(box):
    cx, cy, w, h = box
    x0 = cx - w / 2
    y0 = cy - h / 2
    x1 = cx + w / 2
    y1 = cy + h / 2
    return (x0, y0, x1, y1)

def overlay_boxes(img_tensor, boxes_cxcywh, out_path):
    """Save a quick overlay to visually spot-check boxes."""
    import torchvision
    import torch

    H, W = img_tensor.shape[-2:]
    # convert to xyxy in pixels
    xyxy_pix = []
    for b in boxes_cxcywh:
        x0,y0,x1,y1 = cxcywh_to_xyxy_norm(b.tolist())
        xyxy_pix.append([
            max(0, min(W-1, x0 * W)),
            max(0, min(H-1, y0 * H)),
            max(0, min(W-1, x1 * W)),
            max(0, min(H-1, y1 * H)),
        ])
    if len(xyxy_pix) == 0:
        print("[overlay] no boxes to draw")
        return
    boxes_tensor = torch.tensor(xyxy_pix, dtype=torch.float32)

    # torchvision wants [C,H,W] uint8 0..255
    img_uint8 = (img_tensor.clamp(0,1) * 255).to(torch.uint8)
    drawn = torchvision.utils.draw_bounding_boxes(
        img_uint8.cpu(), boxes_tensor.cpu(), width=2
    )
    out = TF.to_pil_image(drawn)
    out.save(out_path)
    print(f"[overlay] saved {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-root", required=True, type=str)
    ap.add_argument("--ann", required=True, type=str)
    ap.add_argument("--index", type=int, default=0, help="dataset index to probe")
    ap.add_argument("--hf-id", type=str, default="google/owlvit-base-patch32")
    ap.add_argument("--save-overlay", type=str, default="debug_overlay.jpg")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # --- Load one REAL sample from your VG150 dataset ---
    ds = VG150(img_root=args.img_root, ann_path=args.ann, split="train")
    img, target = ds[args.index]   # img: [3,H,W] in [0,1]; boxes normalized cx,cy,w,h

    print(f"[sample] idx={args.index} image={img.shape} "
          f"boxes={tuple(target['boxes'].shape)} labels={tuple(target['labels'].shape)}")

    # --- Build either (backbone+head) or full detector ---
    model_cfg = {
        "backbone": {"type": "owlvit", "hf_id": args.hf_id, "return_grid": True},
        "head": {"type": "owlvit_reuse", "hf_id": args.hf_id, "freeze": True,
                 "open_vocab_file": "/home/romina/SG_Generation/dataset/raw_DATA/object_synsets_150.json"},
    }

    img_b = img.unsqueeze(0).to(device)  # [1,3,H,W]

    # Raw backbone + head (no RA)
    backbone = VisionBackbone(model_cfg).to(device).eval()
    head = OwlViTReuseHeads(model_cfg).to(device).eval()

    with torch.no_grad():
        tokens, cls = backbone(img_b)  # tokens: [1,N,C] or [1,H,W,C]
        print("[backbone] tokens:", tuple(tokens.shape), "cls:", tuple(cls.shape))
        B, H, W, C = tokens.shape
        grid_hw = (H, W)
        tokens = tokens.view(B, H * W, C)
        head_out = head(tokens=tokens, cls=cls, grid_hw=grid_hw)
        boxes = head_out["boxes"]  # [1,N,4] normalized cx,cy,w,h in [0,1]
        logits = head_out.get("logits", None)

    
    print("[head] boxes:", tuple(boxes.shape))
    if logits is not None:
        print("[head] logits:", tuple(logits.shape))

    print("[head] boxes stats:",
          f"min={float(boxes.min()):.4f}, max={float(boxes.max()):.4f}, std={float(boxes.std()):.4f}")

    # Quick visual: draw a handful of boxes (top 100 tokens to keep it readable)
    overlay_boxes(img, boxes[0][:100].cpu(), args.save_overlay)

if __name__ == "__main__":
    main()
