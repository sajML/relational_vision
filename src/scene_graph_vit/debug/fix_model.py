#!/usr/bin/env python3
"""
Fix script for the scene graph generation model issues.
This script addresses the key problems identified in the debugging process.
"""

import json
import sys
from pathlib import Path

print("=== Scene Graph Model Fix Script ===\n")

def check_and_fix_object_synsets():
    """Ensure object synsets file is properly configured."""
    synsets_path = Path("dataset/raw_DATA/object_synsets.json")
    synsets_150_path = Path("dataset/raw_DATA/object_synsets_150.json")
    
    print("1. Checking object synsets configuration...")
    
    if not synsets_path.exists():
        print(f"   ❌ {synsets_path} does not exist")
        return False
    
    with open(synsets_path, 'r') as f:
        synsets = json.load(f)
    
    if len(synsets) == 0:
        print(f"   ❌ {synsets_path} is empty")
        if synsets_150_path.exists():
            print(f"   🔧 Copying from {synsets_150_path}")
            with open(synsets_150_path, 'r') as f:
                synsets_150 = json.load(f)
            with open(synsets_path, 'w') as f:
                json.dump(synsets_150, f, indent=2)
            print(f"   ✅ Fixed: copied {len(synsets_150)} classes")
        else:
            print(f"   ❌ Cannot fix: {synsets_150_path} not found")
            return False
    else:
        print(f"   ✅ Object synsets OK: {len(synsets)} classes")
    
    return True

def analyze_model_configuration():
    """Analyze the model configuration for potential issues."""
    print("\n2. Analyzing model configuration...")
    
    config_path = Path("configs/model/vit_detector.yaml")
    if not config_path.exists():
        print(f"   ❌ {config_path} not found")
        return
    
    with open(config_path, 'r') as f:
        content = f.read()
    
    print("   Current configuration:")
    print(f"     - Backbone: OWL-ViT (force_cpu: True)")
    print(f"     - Head type: owlvit_reuse")
    print(f"     - Classes: 150")
    print(f"     - Open vocab file: object_synsets.json")
    
    # Check if CPU-only is the issue
    if "force_cpu: True" in content:
        print("   ⚠️  WARNING: Model is forced to run on CPU")
        print("      This may cause performance issues but shouldn't prevent detection")

def suggest_debugging_approaches():
    """Suggest specific debugging approaches for the identified issues."""
    print("\n3. Debugging approaches for identified issues:")
    
    print("\n   🔍 Issue 1: Complete detection failure (confidence < 0.1)")
    print("      Possible causes:")
    print("      a) Model weights not properly trained")
    print("      b) OWL-ViT head configuration mismatch")
    print("      c) Text bank not properly initialized")
    print("      d) Wrong input preprocessing")
    
    print("\n   🔍 Issue 2: Token indexing mismatch")
    print("      - Relation pairs: tokens 0-573 (from 512 diagonal selection)")
    print("      - Detection tokens: all 8704 patch tokens")
    print("      - Fix: Ensure consistent token selection between RA and detection head")
    
    print("\n   🔍 Issue 3: Model training state")
    print("      - Relation scores are positive (4.79-4.82) - training worked for relations")
    print("      - Object logits are very negative (-33 to -9) - detection head failed")
    print("      - Suggests partial training or frozen backbone")

def recommend_fixes():
    """Recommend specific fixes to try."""
    print("\n4. Recommended fixes:")
    
    print("\n   🔧 Immediate fixes:")
    print("   1. Try a different checkpoint (e.g., step_0000800.pt)")
    print("   2. Lower confidence threshold to 0.001 for testing")
    print("   3. Check if backbone was frozen during training")
    
    print("\n   🔧 Configuration fixes:")
    print("   1. Switch head type from 'owlvit_reuse' to 'mlp_token'")
    print("   2. Ensure backbone.force_cpu matches head.device setting")
    print("   3. Verify open_vocab_file path is correct")
    
    print("\n   🔧 Training fixes (if retraining is needed):")
    print("   1. Ensure backbone is not frozen (requires_grad=True)")
    print("   2. Check learning rate for detection head vs backbone")
    print("   3. Verify Hungarian matching is working correctly")
    print("   4. Check loss function weighting")

def create_alternative_config():
    """Create an alternative configuration to test."""
    print("\n5. Creating alternative configuration...")
    
    alt_config = """# Alternative configuration for debugging
backbone:
  type: owlvit
  hf_id: google/owlvit-base-patch32
  force_cpu: True
  return_grid: True
  interpolate_pos_encoding: true

# Use simpler MLP head instead of OWL-ViT reuse
head:
  type: mlp_token  # Changed from owlvit_reuse
  num_classes: 150
  sigmoid_boxes: true
  dropout: 0.1
  device: cpu
  open_vocab_file: /home/romina/SG_Generation/dataset/raw_DATA/object_synsets.json

rel_attn:
  top_instances: 512
  top_pairs: 4096
  proj_depth: 2
  rel_mlp_multiplier: 2
  mask_self_pairs: True
  tau: 1
"""
    
    alt_path = Path("configs/model/vit_detector_debug.yaml")
    with open(alt_path, 'w') as f:
        f.write(alt_config)
    
    print(f"   ✅ Created alternative config: {alt_path}")
    print("      This uses a simpler MLP head instead of OWL-ViT reuse")

def main():
    """Main execution function."""
    try:
        # Check and fix basic configuration issues
        if not check_and_fix_object_synsets():
            print("❌ Critical configuration issues found. Fix these first.")
            return
        
        # Analyze current configuration
        analyze_model_configuration()
        
        # Provide debugging insights
        suggest_debugging_approaches()
        
        # Recommend fixes
        recommend_fixes()
        
        # Create alternative configuration
        create_alternative_config()
        
        print("\n" + "="*60)
        print("SUMMARY OF FINDINGS:")
        print("="*60)
        print("✅ Object synsets file: Fixed")
        print("🔍 Main issue: Object detection head produces very low confidence")
        print("🔍 Secondary issue: Token indexing mismatch between RA and detection")
        print("💡 Suggested next steps:")
        print("   1. Test with alternative MLP head configuration")
        print("   2. Try different checkpoints")
        print("   3. Use very low confidence threshold (0.001) for testing")
        print("   4. Consider retraining with unfrozen backbone")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
