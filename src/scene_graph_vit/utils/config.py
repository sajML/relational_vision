from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Union
import yaml

YamlLike = Union[str, Path, Dict[str, Any]]

def _load_yaml(path_or_dict: YamlLike) -> Dict[str, Any]:
    """
    Load a YAML file or a dictionary containing a 'cfg' key.
    Accepts:
      • str path
      • pathlib.Path
      • {"cfg": <str|Path>}
    """
    if isinstance(path_or_dict, (str, Path)):
        p = Path(path_or_dict)
        if not p.exists():
            raise FileNotFoundError(f"YAML not found: {p}")
        with open(p, "r") as f:
            return yaml.safe_load(f)
    if isinstance(path_or_dict, dict) and "cfg" in path_or_dict:
        p = Path(path_or_dict["cfg"])
        if not p.exists():
            raise FileNotFoundError(f"YAML not found: {p}")
        with open(p, "r") as f:
            return yaml.safe_load(f)
    raise ValueError("Expected YAML path (str/Path) or a dict with key 'cfg' pointing to a YAML path.")

def load_cfg(path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with open(p, "r") as f:
        return yaml.safe_load(f)

def resolve_cfg(base: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve the configuration by loading YAML files
    """
    
    if "model" not in base or "algo" not in base:
        missing = [k for k in ("model","algo") if k not in base]
        raise KeyError(f"Base config must contain keys: {missing}")
    base["model_resolved"] = _load_yaml(base["model"])
    base["algo_resolved"]  = _load_yaml(base["algo"])
    return base

def validate_config(cfg: Dict[str, Any], check_fs: bool = False) -> None:
    """
    Quick validation for common config errors.
    """
    errors = []

    # Required top-level keys
    for key in ["model_resolved", "algo_resolved", "data", "train"]:
        if key not in cfg:
            errors.append(f"Missing required key: {key}")

    # Train schedule sanity
    if "train" in cfg:
        total_steps = int(cfg["train"].get("total_steps", 0))
        if total_steps < 0:
            errors.append(f"train.total_steps must be >= 0, got {total_steps}")

        if "algo_resolved" in cfg:
            sched_cfg = cfg["algo_resolved"].get("scheduler", {})
            warmup = int(sched_cfg.get("warmup_steps", 0))
            # Only check when we actually use step-based training
            if total_steps > 0 and warmup >= total_steps:
                errors.append(f"scheduler.warmup_steps ({warmup}) must be < train.total_steps ({total_steps}).")

    # Model config
    if "model_resolved" in cfg:
        m = cfg["model_resolved"]

        # Backbone validation
        b = m.get("backbone", {})
        btype = b.get("type")
        if btype not in ["owlvit", "dino_v2"]:
            errors.append(f"Invalid backbone type: {btype} (expected 'owlvit' or 'dino_v2').")

        # Head validation
        h = m.get("head", {})
        htype = h.get("type", "")
        num_classes = h.get("num_classes", 0)

        # In open-vocab heads (e.g., 'owlvit_heads_only'), num_classes can be 0 by design.
        ov_head_types = {"owlvit_heads_only"}
        if htype not in ov_head_types:
            if not isinstance(num_classes, int) or num_classes <= 0:
                errors.append(f"head.num_classes must be a positive integer for head.type='{htype}', got {num_classes}.")

    # Optional filesystem checks for data
    if check_fs and "data" in cfg:
        d = cfg["data"]
        root = Path(d.get("root", ""))
        if not root.exists():
            errors.append(f"data.root does not exist: {root}")
        img_dir = root / d.get("images_dir", "images")
        ann_dir = root / d.get("ann_dir", "annotations")
        if not img_dir.exists():
            errors.append(f"Images directory not found: {img_dir}")
        if not ann_dir.exists():
            errors.append(f"Annotations directory not found: {ann_dir}")
        splits = (d.get("splits") or {})
        for split_name, relpath in splits.items():
            p = ann_dir / relpath
            if not p.exists():
                errors.append(f"Split file for '{split_name}' not found: {p}")

    if errors:
        raise ValueError("Configuration errors:\n  " + "\n  ".join(errors))

def adapt_yaml_to_model_cfg(model_cfg: dict) -> dict:
    """
    Map YAML field names/values to what the modules expect.
      backbone.type: 'owlvit'|'dino_v2' -> 'owlvit'|'dinov2'
    """
    cfg = dict(model_cfg)
    b = dict(cfg.get("backbone", {}))
    t = str(b.get("type", "owlvit")).lower()
    b["type"] = {"owlvit": "owlvit", "dino_v2": "dinov2"}.get(t, t)
    cfg["backbone"] = b
    return cfg
