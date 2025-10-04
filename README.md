# Scene-Graph ViT

An implementation of **Scene-Graph ViT (SG-ViT)** — a simple, decoder‑free architecture for **open‑vocabulary visual relationship detection (VRD)**. SG‑ViT models objects **and** their pairwise relations directly in the image encoder and uses a lightweight **Relationship Attention** layer to select high‑confidence subject–object pairs for classification.

> Paper: *Scene‑Graph ViT: End‑to‑End Open‑Vocabulary Visual Relationship Detection* (PDF: `paper/2403.14270v2.pdf`).

---

## Highlights

* **Encoder‑only design** built on a ViT image backbone (e.g., OWL‑ViT or DINOv2). No transformer decoder required.
* **Relationship Attention (RA)**:

  1. selects top object instances from token diagonals;
  2. selects top subject–object pairs; and
  3. forms relation embeddings by combining subject/object projections.
* **Open‑vocabulary recognition** for objects and predicates via CLIP/OpenCLIP text banks with prompt templates.
* **Single‑stage training** with DETR‑style matching and losses extended to VRD (objects, predicates, boxes, and RA self‑supervision).
* **Reference VG150 dataset** loader and collate utilities.
* **Batteries‑included training loop**, cosine‑warmup scheduler, checkpointing, and a small CLI.

---

## Repository Structure

```
<repo-root>/
  paper/
    2403.14270v2.pdf              # Paper (for reference)

  models/
    backboneT.py                   # OWL‑ViT & DINOv2 adapters (optionally frozen)
    detectorT.py                   # ViTDetector wrapper + RA + head wiring
    headT.py                       # Token heads (MLP token head, etc.)
    relationship_attentionT.py     # Relationship Attention (instances → pairs)

  losses/
    matcher.py                     # Hungarian matcher + cosine logits helpers
    ov_loss.py                     # Open‑vocab criterion (objects, predicates, RA, box)

  data/
    vg150.py                       # VG150 dataset class + transforms + collate

  utils/
    text_bank.py                   # CLIP/OpenCLIP text bank + OBJ/PRED templates
    box_ops.py                     # IoU / GIoU / box conversions
    checkpointing.py               # save_ckpt / load_ckpt helpers
    misc.py                        # logging, seeding, distributed helpers
    config.py                      # YAML resolve/validate/adapt → runtime cfg

  torch_train.py                   # Main training loop (PyTorch)
  cli.py                           # Tiny CLI: train/config/model summary
```

> Note: If your working tree is flat, keep the above relative paths in mind; the code assumes package folders as shown.

---

## Installation

Tested with Python ≥3.10 and PyTorch ≥2.1.

```bash
# 1) Create environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install PyTorch (choose CUDA/CPU build you need)
# Example (CUDA 12.x):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3) Install project requirements (if you have a requirements.txt)
# pip install -r requirements.txt

# 4) Install extras
pip install numpy tqdm pyyaml einops opencv-python pillow
pip install transformers open_clip_torch  # for OWL‑ViT / CLIP text encoders
```

If you plan to run OWL‑ViT from HuggingFace, ensure `transformers` is installed and that you have network access to download model weights on first run.

---

## Dataset: Visual Genome VG150

1. **Download** VG images and the **VG150** splits/annotations.
2. **Organize** your folder (example):

```
DATA/
  VG/
    images/                # *.jpg
    annotations/           # VG150 annotations (objects, attributes, relations)
```

3. **Point** the loader to your data root via config or environment variable, e.g. `DATA_ROOT=DATA/VG`.

The dataset class and collate fn live in `data/vg150.py`. They return images, boxes, object labels, and relation triplets (subject, predicate, object).

---

## Configuration

All configuration is driven by a Python dict resolved from YAML (see `utils/config.py`). You can supply either a YAML file or CLI flags.

Key config sections:

* `backbone`: `type` (`owlvit`, `dinov2`), `hf_id` (for OWL‑ViT), `return_grid`, `interpolate_pos_encoding`, `force_cpu`.
* `rel_attn`: `top_instances`, `top_pairs`, `mask_self_pairs`, `proj_depth`, `rel_mlp_multiplier`.
* `head`: choose a head (e.g., `mlp_token`) and its dims/classes.
* `optim/schedule`: base LR, weight decay, warmup steps, cosine decay.

You can also adapt YAML into the runtime cfg with `adapt_yaml_to_model_cfg` and sanity‑check via `validate_config`.

---

## Quickstart (Training)

### 1) Minimal CLI

```bash
# Example: train with OWL‑ViT backbone on VG150
python -m cli \
  train \
  --data_root /path/to/DATA/VG \
  --hf_id google/owlvit-base-patch32 \
  --epochs 30 \
  --batch_size 8 \
  --top_instances 100 \
  --top_pairs 300
```

### 2) Direct script

```bash
python -m torch_train \
  --cfg configs/sgvit_vg150.yaml \
  --data_root /path/to/DATA/VG
```

Typical flags (depending on your CLI wiring):

* `--hf_id` — HuggingFace model id for OWL‑ViT backbone.
* `--top_instances`, `--top_pairs` — RA selection budgets.
* `--epochs`, `--batch_size`, `--lr`, `--wd` — training schedule.
* `--cpu` — force CPU for text encoders/backbone where supported.

> Tip: Use mixed precision (`torch.cuda.amp`) if available for speed.

---

## Model Components

* **Backbone (`models/backboneT.py`)** — Wraps OWL‑ViT/DINOv2. Returns token embeddings for detection and relations.
* **Relationship Attention (`models/relationship_attentionT.py`)** —

  * scores diagonal tokens to pick **instances**;
  * builds candidate **pairs**; and
  * projects & combines subject/object tokens to relation embeddings.
* **Head (`models/headT.py`)** — Token classification/regression heads (e.g., MLP over tokens) for objects & predicates.
* **Detector (`models/detectorT.py`)** — Wires backbone + RA + heads; exposes forward for loss/inference.

---

## Losses & Matching

* **Hungarian matching** (`losses/matcher.py`) aligns predictions↔targets with a cost over boxes, object logits, and predicate logits.
* **Open‑vocab losses** (`losses/ov_loss.py`) extend DETR‑style criteria with:

  * object classification (text‑to‑vision cosine logits),
  * predicate classification (open‑vocab),
  * box regression (ℓ1 + GIoU),
  * RA supervision (optional), and
  * no‑object / background handling.

---

## Text Bank (Open‑Vocab)

`utils/text_bank.py` builds CLIP/OpenCLIP text embeddings for **objects** and **predicates** using prompt templates like:

```python
OBJ_TEMPLATES = [
  "a photo of a {}.",
  "a blurry photo of a {}.",
  ...
]
PRED_TEMPLATES = [
  "{} {} {}",
  "a photo of {} {} {}",
]
```

You can swap the label set by editing the bank or passing your own label file.

---

## Checkpointing

Use `utils/checkpointing.py`:

```python
from utils.checkpointing import save_ckpt, load_ckpt
save_ckpt(model, optimizer, step, path)
model, optimizer, step = load_ckpt(model, optimizer, path)
```

Trained weights (and EMA if any) are stored with optimizer/state for resume.

---

## Evaluation

Add an evaluation pass to compute:

* **Object detection**: mAP/mAP50, recall.
* **Predicate classification**: top‑k accuracy.
* **VRD triplets**: Recall@k under standard SGDet/PredCls settings.

(If metrics aren’t yet implemented, export predictions and evaluate with a standard VG150 VRD script.)

---

## Inference Example

```python
import torch
from models.detectorT import ViTDetector
from utils.text_bank import CLIPTextBank

model = ViTDetector(cfg).eval().cuda()
text_bank = CLIPTextBank(obj_labels, pred_labels)

with torch.no_grad():
    out = model(images, text_bank)  # dict with boxes, obj_scores, pred_scores, pairs
```

Expect outputs including:

* `boxes`: [N, 4] in xyxy or cxcywh (per cfg)
* `obj_scores` / `obj_labels`
* `pairs`: indices (subject_idx, object_idx)
* `pred_scores` / `pred_labels` for each pair

---

## CLI Cheatsheet

```bash
# Show resolved config
python -m cli config --cfg configs/sgvit_vg150.yaml

# Train
python -m cli train --cfg configs/sgvit_vg150.yaml

# Summarize model
python -m cli summary --cfg configs/sgvit_vg150.yaml
```

---

## Reproducibility

* Set seeds via `utils/misc.py`.
* Enable deterministic ops if needed (slower on GPU).
* Log config snapshots with each run.

---

## Troubleshooting

* **Out of memory**: lower `top_instances`/`top_pairs`, use smaller backbone, reduce batch size, enable gradient checkpointing.
* **Slow data loading**: increase workers, use JPEG decoding libs, pre‑resize images.
* **No CLIP weights**: confirm `transformers`/`open_clip_torch` installed and `hf_id` reachable.
* **NaN losses**: reduce LR, disable mixed precision for a sanity check.

---

## Acknowledgements

* OWL‑ViT and CLIP/OpenCLIP authors and open‑source communities.
* DETR for Hungarian matching and set‑based training inspiration.

---

## Citation

If you use this codebase or ideas from the paper, please cite the original work. (Add BibTeX here when ready.)
