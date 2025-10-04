import os, math, argparse, json
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
from torch.utils.data import DataLoader

# --- project imports ---
from .models.detectorT import ViTDetector
from .losses.ov_loss import OpenVocabSetCriterion, SetCriterion
from .losses.matcher import HungarianMatcher
from .data.vg150 import VG150, collate_vg
from .losses.text_bank import CLIPTextBank, OBJ_TEMPLATES, PRED_TEMPLATES
from .utils.checkpointing import save_ckpt, load_ckpt
from .utils.config import load_cfg, resolve_cfg, adapt_yaml_to_model_cfg, validate_config
from .utils.dataset_setup import setup_vg150_dataset, get_project_root, resolve_path
# --------------------------------------------------------------

ROOT = Path(__file__).parent.resolve()
REPO = ROOT

# ----------- Helpers ----------------
def count_trainable(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def linear_warm(step, warm=200):  # match your LR warmup
    return min(1.0, (step+1)/warm)

class CosineWithWarmup:
    def __init__(self, optimizer, base_lr, final_lr, warmup_steps, total_steps):
        self.opt = optimizer
        self.base_lr = float(base_lr)
        self.final_lr = float(final_lr)
        self.warmup = int(warmup_steps)
        self.total  = int(total_steps)
        self.step_n = 0
        self._set_lr(0.0)

    def _set_lr(self, lr: float):
        for g in self.opt.param_groups:
            g["lr"] = lr

    def step(self):
        self.step_n += 1
        if self.step_n <= self.warmup:
            lr = self.base_lr * self.step_n / max(1, self.warmup)
        else:
            t = (self.step_n - self.warmup) / max(1, self.total - self.warmup)
            lr = self.final_lr + 0.5*(self.base_lr - self.final_lr)*(1 + math.cos(math.pi*t))
        self._set_lr(lr)

def build_optimizer(params, opt_cfg: Dict[str, Any]):
    name = str(opt_cfg.get("name", "adamw")).lower()
    lr = float(opt_cfg.get("base_lr", 1e-3))
    wd = float(opt_cfg.get("weight_decay", 0.0))
    betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
    eps = float(opt_cfg.get("eps", 1e-8))
    if name in ("adamw","adam"):
        return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=wd)
    raise ValueError(f"Unknown optimizer {name}")

def build_scheduler(optimizer, sch_cfg: Dict[str, Any], total_steps: int):
    name = str(sch_cfg.get("name", "cosine")).lower()
    if name == "cosine":
        return CosineWithWarmup(
            optimizer,
            base_lr=sch_cfg.get("base_lr", 1e-3),
            final_lr=sch_cfg.get("final_lr", 1e-5),
            warmup_steps=int(sch_cfg.get("warmup_steps", 0)),
            total_steps=int(total_steps),
        )
    raise ValueError(f"Unknown scheduler {name}")

# ------------- Train / Eval steps --------------
def forward_loss(model, criterion, batch, device):
    images, targets = batch
    images = images.to(device, non_blocking=True)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    out = model(images, train=True)                 # ViTDetector forward
    out_for_loss = dict(out)
    # DETR-style SetCriterion expects 'pred_boxes', OpenVocab expects RA fields too
    if "boxes" in out_for_loss and "pred_boxes" not in out_for_loss:
        out_for_loss["pred_boxes"] = out_for_loss.pop("boxes")

    losses = criterion(out_for_loss, targets)
    return losses, out

@torch.no_grad()
def run_eval(model, criterion, loader, device) -> Dict[str, float]:
    model.eval()
    agg: Dict[str, float] = {}
    n = 0
    for batch in loader:
        losses, _ = forward_loss(model, criterion, batch, device)
        for k, v in losses.items():
            agg[k] = agg.get(k, 0.0) + float(v.item())
        n += 1
    if n > 0:
        for k in list(agg.keys()):
            agg[k] /= n
    model.train()
    return agg


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, default=str(REPO / "configs/base.yaml"),
                    help="Path to base.yaml")
    ap.add_argument("--override_out", type=str, default="", help="Override output dir")
    ap.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume")
    args = ap.parse_args()

    # Load base bundle and resolve nested model/algo yamls
    base_cfg = load_cfg(args.base)
    base_cfg = resolve_cfg(base_cfg)
    validate_config(base_cfg)

    data_cfg = base_cfg["data"]
    train_cfg = base_cfg["train"]
    algo_cfg = base_cfg["algo_resolved"]
    model_cfg = adapt_yaml_to_model_cfg(base_cfg["model_resolved"])

    # Get project root once for all path resolutions
    project_root = get_project_root()
    
    # Output dir - resolve dynamically
    work_dir = resolve_path(base_cfg.get("work_dir", "./runs"), project_root)
    exp_name = base_cfg.get("exp_name", "sgdet_exp")
    out_dir = Path(args.override_out) if args.override_out else (work_dir / exp_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # Device
    device_cfg = base_cfg.get("device", {})
    device_type = device_cfg.get("type", "auto")
    force_cpu_config = device_cfg.get("force_cpu", False)

    # Determine actual device
    if device_type == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu_config else "cpu")
    elif device_type == "cuda":
        if not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
            force_cpu_config = True
        else:
            device = torch.device("cuda")
            force_cpu_config = False if not force_cpu_config else True
    elif device_type == "cpu":
        device = torch.device("cpu")
        force_cpu_config = True
    else:
        raise ValueError(f"Invalid device type '{device_type}'. Use 'auto', 'cuda', or 'cpu'.")


    model_cfg.setdefault("backbone", {}).update({
        "force_cpu": force_cpu_config
    })
    model_cfg.setdefault("head", {}).update({
        "force_cpu": force_cpu_config,
        "device": device.type
    })

    # Dataset - resolve paths dynamically
    root = resolve_path(data_cfg["root"], project_root)
    images_dir = data_cfg.get("images_dir", "images")
    annotation_file = data_cfg.get("ann_dir", "annotations")
    
    # Auto-download dataset if needed
    print(f"Setting up dataset at: {root}")
    setup_vg150_dataset(root)
    
    train_set = VG150(img_root=root / images_dir, ann_path=root / annotation_file, split="train")
    val_set   = VG150(img_root=root / images_dir, ann_path=root / annotation_file, split="val")
    
    batch_size = int(train_cfg.get("batch_size", 8))
    n_workers  = int(train_cfg.get("num_workers", 4))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=n_workers, pin_memory=True, collate_fn=collate_vg)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              num_workers=n_workers, pin_memory=True, collate_fn=collate_vg)

    # Text banks (OpenCLIP)
    owlvit_model_id = model_cfg.get("backbone", {}).get("hf_id", "google/owlvit-base-patch32")
    tb = CLIPTextBank.from_owlvit_config(owlvit_model_id, force_cpu=force_cpu_config)
    # tb = CLIPTextBank(force_cpu=force_cpu_config)
    obj_names = train_set.object_names() if hasattr(train_set, "object_names") else [f"class_{i}" for i in range(model_cfg.get("head",{}).get("num_classes",150))]
    pred_names= train_set.predicate_names() if hasattr(train_set, "predicate_names") else [f"pred_{i}" for i in range(50)]
    obj_bank = tb.build_bank(obj_names, OBJ_TEMPLATES)  # [C_obj, D]
    pred_bank= tb.build_bank(pred_names, PRED_TEMPLATES) # [C_pred, D]

    # Model
    model = ViTDetector(model_cfg).to(device)
    model.train()
    print(f"Model params (trainable): {count_trainable(model)/1e6:.2f}M")

    # Matcher
    mcfg = algo_cfg.get("matcher", {})
    m = HungarianMatcher(
        cost_class=float(mcfg.get("alpha", 1.0)),
        cost_bbox=float(mcfg.get("beta",  0.2)),
        cost_giou=float(mcfg.get("gamma", 0.2)),
    )

    # Loss / Criterion
    lcfg = algo_cfg.get("loss", {})
    ovcfg = algo_cfg.get("open_vocab", {"enabled": True})
    if bool(ovcfg.get("enabled", True)):
        criterion = OpenVocabSetCriterion(
            obj_text_bank=obj_bank, pred_text_bank=pred_bank,
            matcher=m,
            temp_obj=float(ovcfg.get("temp_obj", 10.0)),
            temp_pred=float(ovcfg.get("temp_pred", 10.0)),
            weight_dict={
                "loss_obj_bce": float(lcfg.get("class_weight", 2.0)),
                "loss_pred_bce": float(lcfg.get("pred_weight", 2.0)),
                "loss_bbox": float(lcfg.get("box_weight", 0.2)),
                "loss_giou": float(lcfg.get("giou_weight", 0.2)),
                "loss_ra_bce": float(lcfg.get("ra_selfsup_weight", 1.0)),
            },
            losses=["ov_labels","boxes","predicates","ra"]
        ).to(device)
    else:
        # DETR-style fallback (not open-vocab)
        criterion = SetCriterion(
            num_classes=model_cfg.get("head",{}).get("num_classes", 150),
            matcher=m,
            weight_dict={
                "loss_ce": float(lcfg.get("class_weight", 2.0)),
                "loss_bbox": float(lcfg.get("box_weight", 5.0)),
                "loss_giou": float(lcfg.get("giou_weight", 1.0)),
            },
            eos_coef=float(lcfg.get("eos_coef", 0.1)),
            losses=["labels","boxes"]
        ).to(device)

    # Optimizer / Scheduler
    optimizer = build_optimizer(model.parameters(), algo_cfg.get("optimizer", {}))
    scheduler = build_scheduler(optimizer, algo_cfg.get("scheduler", {}), total_steps=int(train_cfg.get("total_steps",1000)))

    # Optional resume
    start_step = 0
    if args.resume:
        ckpt = load_ckpt(Path(args.resume))
        if ckpt:
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optim"])
            start_step = int(ckpt.get("step", 0))
            print(f"Resumed from {args.resume} @ step {start_step}")

    # Train loop
    total_steps = int(train_cfg.get("total_steps", 1000))
    log_every   = int(train_cfg.get("log_every", 10))
    eval_every  = int(train_cfg.get("eval_every", 500))
    ckpt_every  = int(train_cfg.get("ckpt_every", 100))
    grad_clip   = float(train_cfg.get("grad_clip_norm", 1.0))

    step = start_step
    data_iter = iter(train_loader)
    while step < total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        losses, _ = forward_loss(model, criterion, batch, device)
        loss_total = sum(v for k,v in losses.items() if k.startswith("loss_"))

        optimizer.zero_grad(set_to_none=True)
        loss_total.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        if (step % log_every) == 0:
            msg = f"[{step:6d}/{total_steps}] "
            msg += " | ".join(f"{k}:{float(v.item()):.4f}" for k,v in losses.items() if k.startswith("loss_"))
            print(msg, flush=True)

        if step>0 and (step % eval_every) == 0:
            val = run_eval(model, criterion, val_loader, device)
            print("VAL:", " | ".join(f"{k}:{v:.4f}" for k,v in val.items()))

        if step>0 and (step % ckpt_every) == 0:
            payload = {
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
            }
            save_ckpt(out_dir / f"step_{step:07d}.pt", step, payload)

        step += 1

    # Final eval + ckpt
    val = run_eval(model, criterion, val_loader, device)
    print("FINAL VAL:", " | ".join(f"{k}:{v:.4f}" for k,v in val.items()))
    payload = {"model": model.state_dict(), "optim": optimizer.state_dict()}
    save_ckpt(out_dir / f"final.pt", step, payload)


if __name__ == "__main__":
    main()
