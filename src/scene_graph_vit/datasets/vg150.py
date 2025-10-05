"""
Unified command-line interface.

Commands:
  - train           Run the training loop (wraps train.main)
  - config          Resolve & validate YAML, pretty-print (optionally save)
  - model-summary   Build the model from YAML and print parameter counts
  - infer           Single-image inference + overlay/graph export

Examples:
  python -m src.scene_graph_vit.cli train -- --base configs/base.yaml --num_workers 8
  python -m src.scene_graph_vit.cli config --base configs/base.yaml --save resolved.json
  python -m src.scene_graph_vit.cli model-summary --base configs/base.yaml
  python -m src.scene_graph_vit.cli infer --base configs/base.yaml --ckpt runs/exp/final.pt --image demo.jpg --show-graph
"""

import sys
import json
import argparse
import subprocess
import os
from pathlib import Path

import torch

# --- project imports (relative) ---
from .utils.config import load_cfg, resolve_cfg, validate_config, adapt_yaml_to_model_cfg
from .models.detectorT import ViTDetector
from .torch_viz import main as viz_main

def count_trainable(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def check_kaggle_dataset():
    """
    Check if running on Kaggle with the Visual Genome dataset available.
    Returns the dataset path if found, otherwise None.
    """
    kaggle_path = Path("/kaggle/input/the-visual-genome-dataset/images2")
    if kaggle_path.exists() and kaggle_path.is_dir():
        print(f"✓ Found Kaggle dataset at: {kaggle_path}")
        return str(kaggle_path)
    print("✗ Kaggle dataset not found")
    return None

def setup_dataset_path(cfg):
    """
    Check for Kaggle dataset and update config accordingly.
    If Kaggle path exists, override the dataset path in config.
    """
    kaggle_path = check_kaggle_dataset()
    if kaggle_path:
        # Set environment variable that can be used by other modules
        os.environ["VG_DATASET_PATH"] = kaggle_path
        
        # Update config if it has dataset paths
        if "data_resolved" in cfg:
            if "root" in cfg["data_resolved"]:
                cfg["data_resolved"]["root"] = kaggle_path
            if "image_dir" in cfg["data_resolved"]:
                cfg["data_resolved"]["image_dir"] = kaggle_path
        
        print(f"→ Using Kaggle dataset path: {kaggle_path}")
    else:
        print("→ Kaggle dataset not found, using configured paths")
    
    return cfg

def cmd_train(argv):
    """
    Pass-through to train.main().
    """
    filtered_argv = [arg for arg in argv if arg and arg != '--']
    
    # Check for Kaggle dataset before training
    kaggle_path = check_kaggle_dataset()
    if kaggle_path:
        os.environ["VG_DATASET_PATH"] = kaggle_path
    
    # Build command to run torch_train.py as a module
    cmd = [sys.executable, "-m", "src.scene_graph_vit.torch_train"] + filtered_argv
    result = subprocess.run(cmd, cwd=Path.cwd())
    return result.returncode

def cmd_config(args):
    """
    Resolve and validate config, then pretty-print (and optionally save).
    """
    cfg = load_cfg(args.base)
    cfg = resolve_cfg(cfg)
    cfg = setup_dataset_path(cfg)
    validate_config(cfg)
    if args.save:
        out = Path(args.save)
        out.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
        print(f"Saved resolved config to: {out}")
    else:
        print(json.dumps(cfg, indent=2, ensure_ascii=False))

def cmd_model_summary(args):
    """
    Build model from resolved YAML and print a quick summary.
    """
    cfg = load_cfg(args.base)
    cfg = resolve_cfg(cfg)
    validate_config(cfg)

    model_yaml = adapt_yaml_to_model_cfg(cfg["model_resolved"])

    # If open-vocab is enabled, closed-vocab head logits are not required
    ov = (cfg.get("algo_resolved", {}).get("open_vocab") or {})
    if ov.get("enabled", False):
        model_yaml.setdefault("head", {})
        model_yaml["head"]["num_classes"] = 0
        
    # Choose device based on availability unless overridden
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure YAML respects device (same nudge as train.py)
    model_yaml.setdefault("backbone", {}).update({
        "force_cpu": False if device.type == "cuda" else True
    })
    model_yaml.setdefault("head", {}).update({
        "force_cpu": False if device.type == "cuda" else True,
        "device": "cuda" if device.type == "cuda" else "cpu"
    })

    model = ViTDetector(model_yaml).to(device)
    trainable = count_trainable(model)
    total = sum(p.numel() for p in model.parameters())

    print("Model summary")
    print("-------------")
    print(f"Device:            {device.type}")
    print(f"Trainable params:  {trainable:,} ({trainable/1e6:.2f} M)")
    print(f"Total params:      {total:,} ({total/1e6:.2f} M)")
    # Optional: print important head/backbone config bits
    bb = model_yaml.get("backbone", {})
    hd = model_yaml.get("head", {})
    print("Backbone config (subset):")
    print(json.dumps({k: bb[k] for k in bb if k in ("type","hf_id","interpolate_pos_encoding","return_grid","force_cpu")}, indent=2))
    print("Head config (subset):")
    print(json.dumps({k: hd[k] for k in hd if k in ("type","num_classes","num_queries","device","force_cpu")}, indent=2))


def cmd_infer(argv):
    """
    Pass-through to viz.main().
    """
    filtered_argv = [arg for arg in argv if arg and arg != '--']
    
    # Check for Kaggle dataset before inference
    kaggle_path = check_kaggle_dataset()
    if kaggle_path:
        os.environ["VG_DATASET_PATH"] = kaggle_path
    
    # Call viz.main with our argv list
    return viz_main(filtered_argv)


def build_parser():
    p = argparse.ArgumentParser(prog="src.scene_graph_vit.cli", description="Scene Graph Generation CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    # train (delegates to train.py)
    p_train = sub.add_parser("train", help="Run training (delegates to torch_train.py)")
    p_train.add_argument("rest", nargs=argparse.REMAINDER, help="Arguments to pass to torch_train.py")
    p_train.set_defaults(func=lambda args: cmd_train(args.rest))
    
    # config
    p_cfg = sub.add_parser("config", help="Resolve, validate, and print config")
    p_cfg.add_argument("--base", type=str, default="configs/base.yaml", help="Path to base.yaml")
    p_cfg.add_argument("--save", type=str, default="", help="Optional path to save resolved config as JSON")
    p_cfg.set_defaults(func=cmd_config)

    # model-summary
    p_sum = sub.add_parser("model-summary", help="Build model from YAML and print parameter counts")
    p_sum.add_argument("--base", type=str, default="configs/base.yaml", help="Path to base.yaml")
    p_sum.add_argument("--device", type=str, default="", choices=["cpu","cuda","mps",""], help="Override device")
    p_sum.set_defaults(func=cmd_model_summary)

    # infer (delegates to viz.py)
    p_inf = sub.add_parser("infer", help="Single-image inference + overlay/graph export (delegates to viz.py)")
    p_inf.add_argument("rest", nargs=argparse.REMAINDER, help="Arguments to pass to torch_viz.py")
    p_inf.set_defaults(func=lambda args: cmd_infer(args.rest))

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)

if __name__ == "__main__":
    raise SystemExit(main())