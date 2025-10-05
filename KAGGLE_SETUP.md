# Running on Kaggle

This guide helps you run the Scene-Graph ViT training on Kaggle.

## Quick Start

### 1. Upload Your Code to Kaggle

Create a new Kaggle notebook and upload your code or clone from GitHub:

```python
# Option 1: Clone from GitHub (if you have a repo)
!git clone https://github.com/yourusername/relational_vision.git
%cd relational_vision

# Option 2: Upload files directly to Kaggle
# Use Kaggle's "Add Data" button to upload your code as a dataset
```

### 2. Install Dependencies

```python
# Install required packages
!pip install -q torch torchvision torchaudio transformers open-clip-torch Pillow numpy scipy matplotlib networkx PyYAML tqdm pytest
```

### 3. Dataset Auto-Download

The dataset will be automatically downloaded when you run training! The code now includes automatic dataset setup.

You can test the dataset setup first:

```python
!python test_dataset_setup.py
```

### 4. Run Training

```python
# The dataset will be downloaded automatically if not present
!python -m src.scene_graph_vit.cli train -- --base configs/base.yaml
```

## Configuration for Kaggle

The configuration files now use **relative paths** that work automatically in any environment (local, Kaggle, Colab, etc.).

### Default Paths (automatically resolved):
- **Dataset**: `./dataset/vg150/` (auto-downloaded from Stanford)
- **Output**: `./experiments/runs/`
- **Configs**: `./configs/`

### Kaggle-Specific Settings

Update `configs/base.yaml` for Kaggle GPU:

```yaml
device:
  type: "auto"          # Will use GPU if available
  force_cpu: False      # Allow GPU usage

train:
  batch_size: 32        # Increase for GPU
  num_workers: 4        # More workers for faster data loading
```

## Features for Cloud Environments

✅ **Automatic Dataset Download**: Images are downloaded from Stanford's VG dataset  
✅ **Dynamic Path Resolution**: Paths work in any environment  
✅ **Progress Bars**: See download and extraction progress  
✅ **Resumable**: If download fails, it can resume  
✅ **Space Efficient**: Option to keep or delete zip files after extraction

## Complete Kaggle Notebook Example

```python
# === KAGGLE SETUP ===

# 1. Install dependencies
!pip install -q torch torchvision transformers open-clip-torch \
              Pillow numpy scipy matplotlib networkx PyYAML tqdm

# 2. Clone or upload your code
!git clone https://github.com/yourusername/relational_vision.git
%cd relational_vision

!git clone https://github.com/sajML/relational_vision
%cd relational_vision

# 3. Verify setup (optional but recommended)
!python test_dataset_setup.py

# 4. Update config for GPU (optional)
import yaml
with open('configs/base.yaml', 'r') as f:
    config = yaml.safe_load(f)

config['device'] = {'type': 'auto', 'force_cpu': False}
config['train']['batch_size'] = 32
config['train']['num_workers'] = 4

with open('configs/base.yaml', 'w') as f:
    yaml.dump(config, f)

# 5. Start training (dataset downloads automatically)
!python -m src.scene_graph_vit.cli train -- --base configs/base.yaml

# 6. Monitor training
# The output will be in ./experiments/runs/OWV_test3/
```

## Manual Dataset Setup (Alternative)

If you prefer to upload the dataset manually to Kaggle:

1. Download images from: https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
2. Upload as a Kaggle dataset
3. Update `configs/base.yaml`:

```yaml
data:
  root: /kaggle/input/your-dataset-name/vg150
```

## Tips for Kaggle

- **Enable GPU**: Go to Settings → Accelerator → GPU T4 x2
- **Internet Access**: Make sure "Internet" is enabled for downloads
- **Save Outputs**: Copy checkpoints to `/kaggle/working/` to keep them after session
- **Monitor Progress**: Use `watch -n 10 tail -n 20 experiments/runs/OWV_test3/train.log`

## Troubleshooting

### "Download Failed"
- Check internet is enabled in Kaggle notebook settings
- Try running setup script separately: `python test_dataset_setup.py`

### "CUDA Out of Memory"
- Reduce batch size in `configs/base.yaml`
- Set `num_workers: 0` to reduce memory usage

### "Path Not Found"
- Make sure you're running from the project root directory
- Paths are now automatically resolved, but ensure folder structure is intact

## Storage Considerations

The VG dataset images are approximately:
- **Compressed**: ~5GB (zip file)
- **Extracted**: ~12GB (images)

Kaggle provides 20GB of disk space per notebook, which is sufficient.
