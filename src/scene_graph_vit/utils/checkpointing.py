from __future__ import annotations
from pathlib import Path
import re
from typing import Optional, Tuple
import torch



def save_ckpt(path: Path, step: int, payload: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**payload, "step": int(step)}, path)

def load_ckpt(path: Path) -> dict:
    if path.exists():
        return torch.load(Path(path), map_location="cpu")


def _load_latest_step(path: Path) -> Path:
    if path.is_file():
        return path
    finals = list(path.glob("final.pt"))
    steps = sorted(path.glob("step_*.pt"))
    if finals:
        return finals[0]
    if steps:
        return steps[-1]
    raise FileNotFoundError(f"No checkpoint found under {path}")