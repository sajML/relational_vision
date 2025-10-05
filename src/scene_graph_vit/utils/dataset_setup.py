"""
Utility to automatically download and setup VG150 dataset.
This ensures the dataset is available before training, useful for Kaggle/Colab environments.
"""
from __future__ import annotations
import os
import zipfile
from pathlib import Path
from typing import Optional
import urllib.request
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url: str, output_path: Path) -> None:
    """Download a file from URL with progress bar."""
    print(f"Downloading {url} to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
    
    print(f"Download complete: {output_path}")


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract a zip file to the specified directory."""
    print(f"Extracting {zip_path} to {extract_to}...")
    extract_to.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get total size for progress bar
        total_size = sum(info.file_size for info in zip_ref.infolist())
        
        with tqdm(total=total_size, unit='B', unit_scale=True, desc='Extracting') as pbar:
            for member in zip_ref.infolist():
                zip_ref.extract(member, extract_to)
                pbar.update(member.file_size)
    
    print(f"Extraction complete: {extract_to}")


def setup_vg150_dataset(
    dataset_root: str | Path,
    images_url: str = "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip",
    force_download: bool = False
) -> Path:
    """
    Setup VG150 dataset by downloading and extracting images if needed.
    
    Args:
        dataset_root: Root directory for the dataset (e.g., ./dataset/vg150)
        images_url: URL to download the images zip file
        force_download: If True, re-download even if files exist
    
    Returns:
        Path to the images directory
    """
    dataset_root = Path(dataset_root).resolve()
    images_dir = dataset_root / "images"
    
    # Check for Kaggle dataset first
    kaggle_path = os.environ.get("VG_DATASET_PATH")
    if kaggle_path:
        kaggle_images = Path(kaggle_path)
        if kaggle_images.exists() and kaggle_images.is_dir():
            print(f"✓ Using Kaggle dataset at: {kaggle_images}")
            
            # Create symlink or copy reference to Kaggle path
            if not images_dir.exists():
                print(f"→ Creating symlink: {images_dir} -> {kaggle_images}")
                try:
                    images_dir.symlink_to(kaggle_images)
                except OSError:
                    # If symlink fails (permissions), update the path expectation
                    print(f"→ Symlink failed, dataset will use: {kaggle_images}")
                    # Store the actual path for later use
                    (dataset_root / "kaggle_images_path.txt").write_text(str(kaggle_images))
            
            print("✓ Kaggle dataset setup complete")
            return
    
    # Check if images directory exists and has files
    if images_dir.exists() and not force_download:
        image_files = list(images_dir.glob("*.jpg"))
        if len(image_files) > 0:
            print(f"✓ Dataset images found: {len(image_files)} images in {images_dir}")
            return images_dir
    
    print(f"Dataset images not found in {images_dir}")
    print("Starting automatic download and setup...")
    
    # Download zip file
    zip_filename = images_url.split('/')[-1]
    zip_path = dataset_root / zip_filename
    
    if not zip_path.exists() or force_download:
        download_url(images_url, zip_path)
    else:
        print(f"Zip file already exists: {zip_path}")
    
    # Extract zip file
    # The zip file contains VG_100K_2/ folder, we need to handle this
    temp_extract = dataset_root / "temp_extract"
    extract_zip(zip_path, temp_extract)
    
    # Move extracted images to the correct location
    extracted_folders = list(temp_extract.glob("VG_100K*"))
    if extracted_folders:
        source_images = extracted_folders[0]
        print(f"Moving images from {source_images} to {images_dir}...")
        
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Move all images
        image_files = list(source_images.glob("*.jpg"))
        for img_file in tqdm(image_files, desc="Moving images"):
            dest = images_dir / img_file.name
            if not dest.exists():
                img_file.rename(dest)
        
        # Clean up
        import shutil
        shutil.rmtree(temp_extract, ignore_errors=True)
        print(f"Cleaned up temporary files")
    
    # Optionally remove the zip file to save space
    if zip_path.exists():
        zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
        print(f"Zip file ({zip_size_mb:.1f} MB) kept at: {zip_path}")
        print(f"You can delete it manually to save space.")
    
    # Verify setup
    final_image_count = len(list(images_dir.glob("*.jpg")))
    print(f"✓ Dataset setup complete: {final_image_count} images in {images_dir}")
    
    return images_dir


def get_project_root() -> Path:
    """
    Get the project root directory dynamically.
    Looks for the directory containing 'src' folder.
    """
    current = Path(__file__).resolve()
    
    # Go up from utils/dataset_setup.py to find project root
    # Path: src/scene_graph_vit/utils/dataset_setup.py -> go up 3 levels
    for parent in current.parents:
        if (parent / "src").exists() and (parent / "configs").exists():
            return parent
    
    # Fallback: go up 3 levels from this file
    return current.parents[3]


def resolve_path(path: str | Path, base_dir: Optional[Path] = None) -> Path:
    """
    Resolve a path that may be relative or absolute.
    If relative, resolves relative to base_dir (or project root if None).
    
    Args:
        path: Path to resolve
        base_dir: Base directory for relative paths (default: project root)
    
    Returns:
        Absolute resolved Path
    """
    path = Path(path)
    
    # If already absolute, return as is
    if path.is_absolute():
        return path
    
    # Resolve relative to base_dir or project root
    if base_dir is None:
        base_dir = get_project_root()
    
    return (base_dir / path).resolve()


if __name__ == "__main__":
    # Test the setup
    import argparse
    parser = argparse.ArgumentParser(description="Setup VG150 dataset")
    parser.add_argument("--dataset-root", default="./dataset/vg150", help="Dataset root directory")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    args = parser.parse_args()
    
    setup_vg150_dataset(args.dataset_root, force_download=args.force)
