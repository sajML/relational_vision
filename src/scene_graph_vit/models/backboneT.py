from typing import Dict, Any, Tuple
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, OwlViTModel, AutoModel

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

def _to_bhwc(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if x.dim() != 4:
        raise ValueError(f"expected 4D, got {x.dim()}D")
    # accept [B,3,H,W] or [B,H,W,3]
    if x.shape[1] == 3:  # CHW -> HWC
        x = x.permute(0, 2, 3, 1).contiguous()
    assert x.shape[-1] == 3, "expected channel-last RGB"
    return x

class _OWLViTAdapter(nn.Module):
    def __init__(self, bcfg: Dict[str, Any], device: torch.device):
        super().__init__()
        self.hf_id = bcfg.get("hf_id", "google/owlvit-base-patch32")
        self.return_grid = bool(bcfg.get("return_grid", False))
        self.interpolate_pos = bool(bcfg.get("interpolate_pos_encoding", True))
        self.freeze = True
        self.device = device

        self.processor = AutoImageProcessor.from_pretrained(self.hf_id)
        self.vision = OwlViTModel.from_pretrained(self.hf_id).vision_model.eval().to(self.device)
        if self.freeze:
            for p in self.vision.parameters(): p.requires_grad_(False)

    def feature_dim(self) -> int:
        return int(self.vision.config.hidden_size)

    def patch_size(self) -> int:
        return int(self.vision.config.patch_size)

    @torch.inference_mode()
    def _forward_frozen(self, images_bhwc_01: torch.Tensor) -> torch.Tensor:
        imgs = images_bhwc_01.detach().cpu().numpy()  # processor expects PIL/np
        inputs = self.processor(images=[img for img in imgs], return_tensors="pt", do_rescale=False)
        pv = inputs["pixel_values"].to(self.device)
        out = self.vision(pixel_values=pv, interpolate_pos_encoding=self.interpolate_pos)
        return (
            out.last_hidden_state[:, 1:, :],  # patch tokens
            out.last_hidden_state[:, 0, :],   # CLS token
        )

    def forward(self, images_f32_01: torch.Tensor) -> torch.Tensor:
        x = _to_bhwc(images_f32_01)
        if self.freeze:
            tokens, CLS = self._forward_frozen(x)
        else:
            imgs = x.detach().cpu().numpy()
            inputs = self.processor(images=[img for img in imgs], return_tensors="pt", do_rescale=False)
            pv = inputs["pixel_values"].to(self.device)
            out = self.vision(pixel_values=pv, interpolate_pos_encoding=self.interpolate_pos)
            tokens = out.last_hidden_state[:, 1:, :]
            CLS = out.last_hidden_state[:, 0, :]

        if self.return_grid:
            B, N, C = tokens.shape
            g = int(N ** 0.5)
            if g * g == N:
                return tokens.view(B, g, g, C), CLS
            logging.warning(f"[OWLViT] {N} tokens not square; returning flat tokens.")
        return tokens, CLS

class _DINOv2Adapter(nn.Module):
    def __init__(self, bcfg: Dict[str, Any], device: torch.device):
        super().__init__()
        self.hf_id = bcfg.get("hf_id", "facebook/dinov2-large")
        self.return_grid = bool(bcfg.get("return_grid", False))
        self.pad_to_patch = bool(bcfg.get("pad_to_patch", True))
        self.freeze = True
        self.device = device

        self.model = AutoModel.from_pretrained(self.hf_id).eval().to(self.device)
        if self.freeze:
            for p in self.model.parameters(): p.requires_grad_(False)

        self.ps = int(getattr(self.model.config, "patch_size", 14))
        mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std  = torch.tensor(_IMAGENET_STD,  dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("_mean", mean, persistent=False)
        self.register_buffer("_std",  std,  persistent=False)

    def feature_dim(self) -> int:
        return int(self.model.config.hidden_size)

    def patch_size(self) -> int:
        return int(self.ps)

    def _prep(self, images_bhwc_01: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = images_bhwc_01.permute(0, 3, 1, 2).contiguous()  # -> BCHW
        if self.pad_to_patch:
            B, C, H, W = x.shape
            ph = (self.ps - (H % self.ps)) % self.ps
            pw = (self.ps - (W % self.ps)) % self.ps
            if ph or pw:
                x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        B, C, H, W = x.shape
        gh, gw = H // self.ps, W // self.ps
        x = (x - self._mean) / self._std
        return x, gh, gw

    def forward(self, images_f32_01: torch.Tensor) -> torch.Tensor:
        x = _to_bhwc(images_f32_01).to(self.device)
        x, gh, gw = self._prep(x)
        out = self.model(pixel_values=x).last_hidden_state  # [B,1+N,C]
        tokens = out[:, 1:, :]  # drop CLS
        if self.return_grid:
            return tokens.view(tokens.size(0), gh, gw, tokens.size(-1))
        return tokens

class VisionBackbone(nn.Module):
    """
    Unified backbone:
      cfg['backbone']['type'] in {'owlvit','dinov2'}
      shared flags: freeze, return_grid
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        bcfg = cfg.get("backbone", {})
        force_cpu = bool(bcfg.get("force_cpu", False))
        self.device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")
        btype = str(bcfg.get("type", "owlvit")).lower()
        if btype == "owlvit":
            self.impl = _OWLViTAdapter(bcfg, self.device)
        elif btype == "dinov2":
            self.impl = _DINOv2Adapter(bcfg, self.device)
        else:
            raise ValueError(f"Unknown backbone type: {btype}")

    def forward(self, images_f32_01: torch.Tensor) -> torch.Tensor:
        return self.impl(images_f32_01)
    def get_feature_dim(self) -> int: return self.impl.feature_dim()
    def get_patch_size(self) -> int:  return self.impl.patch_size()
    def set_return_grid(self, flag: bool): setattr(self.impl, "return_grid", bool(flag))
    def unfreeze(self):
        for p in self.impl.parameters(): p.requires_grad_(True)
        setattr(self.impl, "freeze", False)