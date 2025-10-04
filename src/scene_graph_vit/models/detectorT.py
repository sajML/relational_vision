from typing import Dict, Any, Tuple, Optional, List, Iterable
import math
import json
import torch
import torch.nn as nn

from .backboneT import VisionBackbone
from .relationship_attentionT import RelationshipAttention
from .headT import create_head

def _infer_grid_hw_from_image(N: int, images_01: torch.Tensor, patch: Optional[int]) -> Tuple[int,int]:
    """Prefer image/patch; fallback to sqrt(N)."""
    H_in, W_in = images_01.shape[-2:]
    if patch and (H_in % patch == 0) and (W_in % patch == 0):
        return (H_in // patch, W_in // patch)
    g = int(round(math.sqrt(N)))
    if g * g != N:
        # Last resort: use divisors closest to square
        for h in range(int(math.sqrt(N)), 0, -1):
            if N % h == 0:
                return (h, N // h)
    return (g, g)

def _clean_label(s: str) -> str:
    return (
        s.replace("_"," ").replace("-"," ").replace("("," ").replace(")"," ")
         .replace("/", " ").strip()
    )

def _labels_from_any_json(obj: Any) -> List[str]:
    """
    Accepts: 
      - ["tree","traffic light",...]
      - [{"name":"tree"}, {"name":"traffic light"}, ...]
      - [["n123","tree"], ["n456","car"], ...]
    Returns: ["tree", "traffic light", ...]
    """
    if isinstance(obj, dict) and "objects" in obj:
        obj = obj["objects"]
    out = []
    for it in obj:
        if isinstance(it, str):
            out.append(_clean_label(it))
        elif isinstance(it, dict):
            # try common keys
            for k in ("name","label","object","text"):
                if k in it:
                    out.append(_clean_label(str(it[k])))
                    break
        elif isinstance(it, (list, tuple)) and len(it) >= 1:
            out.append(_clean_label(str(it[-1])))
    # dedupe preserving order
    seen = set(); uniq=[]
    for s in out:
        if s and s not in seen:
            seen.add(s); uniq.append(s)
    return uniq

class ViTDetector(nn.Module):
    """
    Encoder-only detector with RelationshipAttention gating.

    Outputs:
      - logits:      [B, S, K]
      - boxes:       [B, S, 4]  (cx,cy,w,h)
      - sel_indices: [B, S]     (token ids sent to the head)
      - RA extras:   rel_embs, rel_pairs, rel_scores, diag_*  (if RA enabled)
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.backbone = VisionBackbone(cfg)
        self.rel_attn: Optional[RelationshipAttention] = None
        self.head = create_head(cfg)

        # Relationship Attention
        C = int(self.backbone.get_feature_dim())
        self.rel_attn = RelationshipAttention(cfg, dim=C)

        # Load open-vocab once (if provided via file)
        head_cfg = cfg.get("head", {})
        open_vocab_file = head_cfg.get("open_vocab_file")
        if open_vocab_file:
            try:
                # Try to resolve path if it's relative
                from pathlib import Path
                vocab_path = Path(open_vocab_file)
                if not vocab_path.is_absolute():
                    # Try relative to project root
                    try:
                        from ..utils.dataset_setup import get_project_root, resolve_path
                        vocab_path = resolve_path(open_vocab_file)
                    except:
                        # Fallback to relative path from current working directory
                        vocab_path = Path.cwd() / open_vocab_file
                
                with open(vocab_path, "r") as f:
                    raw = json.load(f)
                self.open_vocab = _labels_from_any_json(raw)
                print(f"[ViTDetector] Loaded open_vocab from: {vocab_path}")
            except Exception as e:
                print(f"[ViTDetector] WARNING: failed to load open_vocab_file={open_vocab_file}: {e}")
        else:
            self.open_vocab = None

    @staticmethod
    def _extract_tokens(bb_out) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[int,int]]]:
        """
        Normalize backbone outputs to:
          - tokens [B,N,C]
          - cls    [B,C] or None
          - grid_hw (H,W) if known when input was [B,H,W,C]
        """
        cls = None
        grid_hw = None
        x = bb_out
        if isinstance(x, tuple) and len(x) == 2:
            x, cls = x
        if x.ndim == 4:                     # [B,H,W,C] -> [B,N,C]
            B, H, W, C = x.shape
            grid_hw = (H, W)
            x = x.view(B, H * W, C)
        assert x.ndim == 3, f"expected tokens [B,N,C], got {tuple(x.shape)}"
        return x, cls, grid_hw

    def forward(self, images_01: torch.Tensor, train: bool = True):
        """
        images_01: [B,3,H,W] float32 in [0,1]
        """
        bb_out = self.backbone(images_01)
        tokens, cls, grid_hw = self._extract_tokens(bb_out)  # tokens: [B,N,C]
        B, N, C = tokens.shape

        # Move to device
        dev = tokens.device
        self.rel_attn.to(dev)
        self.head.to(dev)

        # ---- Relationship Attention (selection) ----
        ra_out = {}
        sel_indices = None
        if self.rel_attn is not None:
            ra_out = self.rel_attn(tokens)                # e.g., rel_pairs [B,Kp,2], diag_indices [B,I]
            rel_pairs = ra_out["rel_pairs"]               # [B,Kp,2]
            sel_indices = rel_pairs.view(B, -1)           # [B, 2*Kp]
            if "diag_indices" in ra_out:
                sel_indices = torch.cat([ra_out["diag_indices"], sel_indices], dim=1)

        # ---- Head (pass cls/grid/open_vocab so OWL heads can work correctly) ----
        head_out = self.head(
            tokens,
            token_indices=sel_indices,
            cls=cls,
            grid_hw=grid_hw,
            open_vocab=self.open_vocab,
        )

        # Validate and return
        if not isinstance(head_out, dict) or "boxes" not in head_out or "logits" not in head_out:
            raise RuntimeError("Head must return a dict with 'logits' and 'boxes'.")

        assert head_out["logits"].dim() == 3, f"Expected logits [B,S,K], got {head_out['logits'].shape}"
        assert head_out["boxes"].dim() == 3 and head_out["boxes"].shape[-1] == 4

        out = {
            "logits": head_out["logits"],
            "boxes": head_out["boxes"],
            "sel_indices": sel_indices,
            **ra_out,
        }
        return out
