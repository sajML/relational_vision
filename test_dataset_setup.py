#!/usr/bin/env python3
"""
Test script to verify dataset download and setup.
Run this before training to ensure everything is configured correctly.
"""
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from scene_graph_vit.utils.dataset_setup import setup_vg150_dataset, get_project_root, resolve_path

def main():
    print("=" * 60)
    print("VG150 Dataset Setup Test")
    print("=" * 60)
    
    # Get project root
    project_root = get_project_root()
    print(f"\n✓ Project root: {project_root}")
    
    # Test path resolution
    dataset_path = resolve_path("./dataset/vg150", project_root)
    print(f"✓ Dataset path (resolved): {dataset_path}")
    
    work_dir = resolve_path("./experiments/runs/", project_root)
    print(f"✓ Work directory (resolved): {work_dir}")
    
    # Setup dataset
    print("\n" + "=" * 60)
    print("Setting up VG150 dataset...")
    print("=" * 60)
    
    images_dir = setup_vg150_dataset(dataset_path)
    
    # Verify annotations exist
    annotations_file = dataset_path / "annotations" / "vg150_clean.json"
    if annotations_file.exists():
        print(f"\n✓ Annotations file found: {annotations_file}")
    else:
        print(f"\n✗ WARNING: Annotations file not found: {annotations_file}")
        print("  You may need to add the annotations manually.")
    
    # Verify object synsets file
    synsets_file = dataset_path / "annotations" / "object_synsets_150.json"
    if synsets_file.exists():
        print(f"✓ Object synsets file found: {synsets_file}")
    else:
        print(f"✗ WARNING: Object synsets file not found: {synsets_file}")
        print("  You may need to add this file manually.")
    
    print("\n" + "=" * 60)
    print("Setup complete! You can now run training.")
    print("=" * 60)
    print("\nTo train, run:")
    print("  python -m src.scene_graph_vit.cli train -- --base configs/base.yaml")

if __name__ == "__main__":
    main()
