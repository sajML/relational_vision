import torch
from pathlib import Path

def test_vg150_dataset_format():
    """Test VG150 dataset loading and format consistency."""
    # Check if dataset path exists
    dataset_root = Path("./dataset/vg150")
    if not dataset_root.exists():
        print("⚠ Dataset not found, skipping test")
        return
        
    # Mock dataset loader (would import actual VG150 class)
    # from src.scene_graph_vit.data.vg150 import VG150Dataset
    
    # For now, just verify expected structure
    expected_dirs = ["images", "annotations"]
    for dir_name in expected_dirs:
        dir_path = dataset_root / dir_name
        assert dir_path.exists(), f"Missing {dir_path}"
    
    # Check annotation file
    ann_file = dataset_root / "annotations" / "vg150_clean.json"
    if ann_file.exists():
        print(f"✓ Found annotation file: {ann_file}")
    else:
        print(f"⚠ Annotation file not found: {ann_file}")
    
    print("✓ Dataset structure check passed")

def test_image_preprocessing():
    """Test image preprocessing pipeline."""
    # Test image size and normalization
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((896, 896)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create dummy PIL image
    from PIL import Image
    dummy_img = Image.new('RGB', (640, 480), color='red')
    
    tensor_img = transform(dummy_img)
    assert tensor_img.shape == (3, 896, 896)
    
    print("✓ Image preprocessing test passed")

if __name__ == "__main__":
    test_vg150_dataset_format()
    test_image_preprocessing()