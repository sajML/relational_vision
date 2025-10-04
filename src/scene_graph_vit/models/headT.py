from __future__ import annotations
import logging
from typing import Dict, Any, Optional, List, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from transformers import AutoTokenizer, OwlViTForObjectDetection

def _select_by_indices(x: torch.Tensor, idx: Optional[torch.Tensor]) -> torch.Tensor:
    if idx is None: return x
    B, _, C = x.shape
    idx_exp = idx.unsqueeze(-1).expand(B, idx.shape[1], C)
    return torch.gather(x, 1, idx_exp)

def _square_hw_from_N(N: int) -> Tuple[int, int]:
    g = int(round(math.sqrt(N)))
    if g * g != N:
        raise ValueError(f"N={N} is not a square; pass grid_hw=(H,W).")
    return g, g

class DetHead(nn.Module):
    """
    Token-based detection head for [B, N, C] features.

    Produces:
      logits: [B, N or M, K] where K = num_classes + 1 (EOS)
      boxes : [B, N or M, 4] in cxcywh, optionally passed through sigmoid
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        head_cfg = cfg.get("head", {})
        self.num_classes = int(head_cfg.get("num_classes", 150))
        self.sigmoid_boxes = bool(head_cfg.get("sigmoid_boxes", True))
        self.dropout = float(head_cfg.get("dropout", 0.1))

        self._shared: Optional[nn.Sequential] = None
        self._cls: Optional[nn.Linear] = None
        self._box: Optional[nn.Linear] = None
        self._in_dim: Optional[int] = None

    def _lazy_build(self, C: int, device: torch.device):
        if self._shared is not None:
            return
        self._shared = nn.Sequential(
            nn.Linear(C, C),
            nn.GELU(),
            nn.Dropout(self.dropout),
        ).to(device)
        self._cls = nn.Linear(C, self.num_classes + 1).to(device)  # +EOS
        self._box = nn.Linear(C, 4).to(device)
        self._in_dim = C

    def forward(self, features: torch.Tensor, *, token_indices: Optional[torch.Tensor] = None, train: bool = False):
        assert features.ndim == 3, f"expected [B,N,C], got {features.shape}"
        B, N, C = features.shape
        self._lazy_build(C, features.device) 

        x = features
        if token_indices is not None:
            x = _select_by_indices(x, token_indices)
        x = self._shared(x)  # [B,M,H]
        logits = self._cls(x)
        boxes  = self._box(x)
        if self.sigmoid_boxes:
            boxes = torch.sigmoid(boxes)
        return {"logits": logits, "boxes": boxes}


class OwlViTHeadsOnly(nn.Module):
    """
    Reuses OWL-ViT's detection heads (box + optional cls) on arbitrary tokens [B,N,C].
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        head_cfg = cfg.get("head", {})
        self.sigmoid_boxes = bool(head_cfg.get("sigmoid_boxes", True))
        self.num_classes = head_cfg.get("num_classes", None)  # optional
        self.hf_id = head_cfg.get("hf_id", "google/owlvit-base-patch32")
        force_cpu = bool(head_cfg.get("force_cpu", True))
        self.device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")
        self.freeze = True
        self._box: Optional[nn.Linear] = None
        self._cls: Optional[nn.Linear] = None
        self._built = False
        self.to(self.device)

    def _best_effort_hf_init(self):
        """Try to steal OWL-ViT's final box predictor weights if dims match."""
        try:
            hf = OwlViTForObjectDetection.from_pretrained(self.hf_id).to(self.device).eval()
            # Try known attribute paths
            candidate_paths = [
                "box_predictor",
                "detector.box_predictor",
                "object_detection_head.box_predictor",
            ]
            box_lin = None
            for dotted in candidate_paths:
                cur = hf
                ok = True
                for part in dotted.split("."):
                    if not hasattr(cur, part):
                        ok = False
                        break
                    cur = getattr(cur, part)
                if ok and hasattr(cur, "out_proj"):
                    maybe = getattr(cur, "out_proj")
                    if isinstance(maybe, nn.Linear) and maybe.out_features == 4:
                        box_lin = maybe
                        break
            if box_lin is not None and self._box is not None and self._box.weight.shape == box_lin.weight.shape:
                with torch.no_grad():
                    self._box.weight.copy_(box_lin.weight)
                    if self._box.bias is not None and box_lin.bias is not None:
                        self._box.bias.copy_(box_lin.bias)
            del hf
        except Exception as e:
            logging.warning(f"[OwlViTHeadsOnly] HF init failed (non-fatal): {e}")

    def _build(self, C: int, device: torch.device):
        if self._built:
            return
        self._box = nn.Linear(C, 4).to(device)
        if isinstance(self.num_classes, int) and self.num_classes > 0:
            self._cls = nn.Linear(C, self.num_classes).to(device) 
        self._best_effort_hf_init()
        if self.freeze:
            for m in [self._box, self._cls]:
                if m is None:
                    continue
                for p in m.parameters():
                    p.requires_grad_(False)
        self._built = True

    def forward(self, tokens: torch.Tensor, *, token_indices: Optional[torch.Tensor] = None, train: bool = False):
        assert tokens.ndim == 3, f"expected [B,N,C], got {tokens.shape}"
        B, N, C = tokens.shape
        self._build(C, tokens.device)

        x = tokens
        if token_indices is not None:
            x = _select_by_indices(x, token_indices)  # [B,M,C]
        boxes = self._box(x)
        if self.sigmoid_boxes:
            boxes = torch.sigmoid(boxes)
        out = {"boxes": boxes}
        if self._cls is not None:
            out["logits"] = self._cls(x)
        else:
            # dummy logits for API parity
            out["logits"] = torch.zeros(x.shape[0], x.shape[1], 1, device=x.device, dtype=x.dtype)
        return out


class OwlViTReuseHeads(nn.Module):
    """
    Reuses the OWL-ViT heads (class_head + box_head) with the
    feature path, given:
      - tokens: [B, N, C]   (patch tokens from your backbone)
      - cls:    [B, C]      (CLS from your backbone)
    Returns:
      boxes : [B, M, 4]  (cx,cy,w,h in [0,1])
      logits: [B, M, Q+1]  Q=len(queries) or 0 (only EOS)
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        h = cfg.get("head", {})
        self.hf_id = h.get("hf_id", "google/owlvit-base-patch32")
        self.sigmoid_boxes = True  # OWL does sigmoid after bias
        self.freeze = True

        # Load HF model once; reuse its components
        self.hf = OwlViTForObjectDetection.from_pretrained(self.hf_id)
        self.class_head = self.hf.class_head
        self.box_head   = self.hf.box_head
        self.ln_img     = nn.LayerNorm(
            self.hf.config.vision_config.hidden_size,
            eps=self.hf.config.vision_config.layer_norm_eps
        )

        # tokenizer + text tower to build query embeddings when a string list is provided
        self.tok = AutoTokenizer.from_pretrained(self.hf_id)
        self.text_model = self.hf.owlvit.text_model

        if self.freeze:
            for m in [self.class_head, self.box_head, self.ln_img, self.text_model]:
                for p in m.parameters(): p.requires_grad_(False)
            for p in self.hf.parameters(): p.requires_grad_(False)
        self.hf.eval()

        self._bias_cache: Dict[Tuple[int,int], torch.Tensor] = {}

    @staticmethod
    def _compute_box_bias(H: int, W: int, device: torch.device) -> torch.Tensor:
        # identical to HF implementation
        x = torch.arange(1, W + 1, dtype=torch.float32, device=device) / W
        y = torch.arange(1, H + 1, dtype=torch.float32, device=device) / H
        xx, yy = torch.meshgrid(x, y, indexing="xy")
        coords = torch.stack((xx, yy), dim=-1).view(-1, 2).clamp_(0, 1)
        coord_bias = torch.log(coords + 1e-4) - torch.log1p(-coords + 1e-4)

        size = torch.ones_like(coord_bias)
        size[..., 0] /= W
        size[..., 1] /= H
        size_bias = torch.log(size + 1e-4) - torch.log1p(-size + 1e-4)

        return torch.cat([coord_bias, size_bias], dim=-1)  # [H*W, 4]

    def _get_bias(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        key = (H, W)
        if key not in self._bias_cache:
            self._bias_cache[key] = self._compute_box_bias(H, W, device)
        return self._bias_cache[key].to(device)

    @torch.no_grad()
    def _build_query_embeds(self, queries: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          query_embeds: [1, Q, D]  (batch dimension kept = 1)
          query_mask  : [1, Q]     (True where valid)
        """
        if not queries:
            # Q=0: class_head will output only EOS logit
            return torch.empty(1, 0, self.hf.config.projection_dim), torch.zeros(1, 0, dtype=torch.bool)
        tk = self.tok(queries, return_tensors="pt", padding=True)
        out = self.text_model(**tk, output_hidden_states=True, return_dict=True)
        pooled = out.pooler_output  # [Q, D_text]
        # add batch dim and mask
        q = pooled.unsqueeze(0)               # [1, Q, D_text]
        mask = (tk["input_ids"][..., 0] > -999).unsqueeze(0)
        return q, mask

    def forward(
        self,
        tokens: torch.Tensor,                 # [B, N, C]
        *,
        cls: Optional[torch.Tensor] = None,   # [B, C] (required to mimic OWL feature build)
        grid_hw: Optional[Tuple[int,int]] = None,
        open_vocab: Optional[List[str]] = None,
        token_indices: Optional[torch.Tensor] = None,
        train: bool = False,  # Add this parameter
    ):
        assert tokens.ndim == 3, f"expected [B,N,C], got {tokens.shape}"
        B, N, C = tokens.shape
        H, W = grid_hw if grid_hw is not None else _square_hw_from_N(N)

        if cls is None:
            raise ValueError("OwlViTReuseHeads needs the CLS vector from the backbone (cls=[B,C]).")

        # 1) broadcast class token and multiply
        cls_b = cls.unsqueeze(1).expand(B, N, C)      # [B,N,C]
        x = tokens * cls_b
        # 2) layernorm 
        x = self.ln_img(x)                            # [B,N,C]
        # 3) Use directly for heads
        image_feats = x  # [B,N,C]

        # Optional selection (RA selected tokens)
        if token_indices is not None:
            image_feats = _select_by_indices(image_feats, token_indices)

        # ---- box ----
        pred_boxes = self.box_head(image_feats)
        bias = self._get_bias(H, W, image_feats.device)
        if token_indices is not None:
            # gather the corresponding bias rows for selected tokens
            bias = bias.index_select(0, token_indices[0]).unsqueeze(0).expand_as(pred_boxes)
        else:
            bias = bias.unsqueeze(0).expand_as(pred_boxes)
        pred_boxes = torch.sigmoid(pred_boxes + bias)  # [B,M,4] cxcywh in [0,1]

        # ---- Class logits: need text queries (open-vocab) ----
        if open_vocab is None:
            # produce EOS-only channel (API parity)
            logits = image_feats.new_zeros(B, image_feats.size(1), 1)
        else:
            q_embeds, q_mask = self._build_query_embeds(open_vocab)  # [1,Q,D] 
            # class_head returns (pred_logits, image_class_embeds)
            logits, _ = self.class_head(image_embeds=image_feats, query_embeds=q_embeds , query_mask = q_mask)

        return {"boxes": pred_boxes, "logits": logits}




def create_head(cfg: Dict[str, Any]) -> nn.Module:
    t = str(cfg.get("head", {}).get("type", "mlp_token")).lower()
    if t in ("mlp_token", "det_token"):
        return DetHead(cfg)
    if t in ("owlvit_heads_only", "owl_head_only", "owl_head"):
        return OwlViTHeadsOnly(cfg)
    if t in ("owlvit_reuse", "owlvit_true_heads"):
        return OwlViTReuseHeads(cfg)
    raise ValueError(f"Unknown head type: {t}")
