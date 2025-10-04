from typing import Sequence, List
import torch, open_clip

class CLIPTextBank:
    def __init__(self, model_name: str = "ViT-L-14", pretrained: str = "openai", force_cpu: bool = False):
        use_cuda = torch.cuda.is_available() and not force_cpu
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.tokenizer = open_clip.get_tokenizer(model_name)

    @classmethod
    def from_owlvit_config(cls, owlvit_hf_id: str, force_cpu: bool = False):
        """
        Create CLIPTextBank with matching dimensions to OwlViT model.
        """
        clip_model_map = {
            "google/owlvit-base-patch32": ("ViT-L-14", "openai"),              # 768-D
            "google/owlvit-large-patch14": ("ViT-H-14", "laion2b_s32b_b79k"),  # 1024-D
        }
        model_name, pretrained = clip_model_map.get(
            owlvit_hf_id, ("ViT-H-14", "laion2b_s32b_b79k")
        )
        return cls(model_name=model_name, pretrained=pretrained, force_cpu=force_cpu)

    @torch.inference_mode()
    def encode_prompts(self, prompts: Sequence[str]) -> torch.Tensor:
        toks = self.tokenizer(list(prompts)).to(self.device)
        feats = self.model.encode_text(toks).float()
        feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-7)
        return feats

    @torch.inference_mode()
    def build_bank(self, labels: Sequence[str], templates: Sequence[str]) -> torch.Tensor:
        """
        Build a normalized text embedding bank [C, D] by averaging template prompts per class.
        """
        assert len(labels) > 0 and len(templates) > 0, "labels and templates must be non-empty"
        all_embeds: List[torch.Tensor] = []
        for name in labels:
            prompts = [
                (t.format(label=name) if "{label}" in t
                 else (t.format(name) if "{}" in t
                 else f"{t} {name}"))
                for t in templates
            ]
            emb = self.encode_prompts(prompts).mean(dim=0, keepdim=True)
            emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-7)
            all_embeds.append(emb)
        bank = torch.cat(all_embeds, dim=0)
        bank = bank / (bank.norm(dim=-1, keepdim=True) + 1e-7)
        assert bank.dim() == 2, f"Expected [C,D], got {bank.shape}"
        return bank


# Keep templates here (used by callers)
OBJ_TEMPLATES  = [
    "a photo of a {label}.",
    "a cropped photo of a {label}.",
    "an image of a {label}.",
]
PRED_TEMPLATES = [
    "is {label}",
    "is being {label}",
    "the relation is {label}",
]