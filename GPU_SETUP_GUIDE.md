# MBSSNet — Complete GPU Training Guide (From Scratch)

## Step 1: Copy Project to GPU Machine

Copy the entire `btp` folder to the GPU machine via USB/Google Drive/SCP.

```bash
# If using SCP (Linux GPU server):
scp -r btp/ username@gpu-server:/home/username/

# Then SSH into the server:
ssh username@gpu-server
cd /home/username/btp
```

---

## Step 2: Create Python Environment

```bash
# Option A: Using conda (recommended)
conda create -n mbssnet python=3.10 -y
conda activate mbssnet

# Option B: Using venv
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows
```

---

## Step 3: Install PyTorch with CUDA

```bash
# Check your CUDA version first:
nvidia-smi

# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CUDA 12.4+:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

---

## Step 4: Install Other Dependencies

```bash
pip install opencv-python numpy einops
```

---

## Step 5: Install mamba-ssm (CUDA-optimized SS2D)

```bash
# This is the key package for fast SS2D scanning
pip install mamba-ssm

# If mamba-ssm fails, try:
pip install mamba-ssm --no-build-isolation

# If still fails, the code will auto-fallback to pure PyTorch (slower but works)
```

---

## Step 6: Verify GPU is Detected

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

Expected output:
```
CUDA available: True
GPU: NVIDIA RTX 4090    (or whatever GPU you have)
```

---

## Step 7: Run Preprocessing (if not already done)

```bash
# Skip this if you already have the 'processed/' folder with patches
python preprocess_data.py
```

Expected output:
```
Train: 23520 patches | Val: 2940 patches | Test: 2940 patches
```

---

## Step 8: Quick Sanity Check (Optional, ~1 min on GPU)

```bash
python test_demo.py
```

---

## Step 9: Train the Full Model

```bash
# Standard training (paper settings):
python train.py --batch_size 4 --num_workers 4 --epochs 30

# If you have a powerful GPU (RTX 3090/4090 with 24GB):
python train.py --batch_size 8 --num_workers 4 --epochs 30

# If you get OUT OF MEMORY error, reduce batch size:
python train.py --batch_size 2 --num_workers 4 --epochs 30

# If you get killed/timeout, resume from checkpoint:
python train.py --batch_size 4 --num_workers 4 --epochs 30 --resume checkpoints/latest_checkpoint.pth
```

### Expected Training Time:
| GPU | Batch Size | Time per Epoch | Total (30 epochs) |
|-----|-----------|----------------|-------------------|
| RTX 4090 | 8 | ~8 min | ~4 hours |
| RTX 3090 | 4 | ~15 min | ~7.5 hours |
| RTX 2080 Ti | 4 | ~25 min | ~12 hours |
| V100 | 4 | ~20 min | ~10 hours |
| T4 (Colab) | 2 | ~40 min | ~20 hours |

---

## Step 10: Generate Visualizations

```bash
# After training completes, generate Fig.3-style results:
python visualize.py --num_samples 10 --split test

# Results saved in visualizations/ folder
```

---

## All Commands in One Block (Copy-Paste Ready)

```bash
# === SETUP (run once) ===
conda create -n mbssnet python=3.10 -y
conda activate mbssnet
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python numpy einops mamba-ssm
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# === PREPROCESS (run once, skip if processed/ exists) ===
python preprocess_data.py

# === TRAIN ===
python train.py --batch_size 4 --num_workers 4 --epochs 30

# === VISUALIZE ===
python visualize.py --num_samples 10 --split test
```

---

## Google Colab Alternative

If your college doesn't have a GPU, use Google Colab:

```python
# Cell 1: Upload btp folder to Google Drive, then mount it
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/btp

# Cell 2: Install dependencies
!pip install torch torchvision mamba-ssm opencv-python einops

# Cell 3: Check GPU
!nvidia-smi
import torch
print(torch.cuda.get_device_name(0))

# Cell 4: Preprocess (if needed)
!python preprocess_data.py

# Cell 5: Train
!python train.py --batch_size 4 --num_workers 2 --epochs 30

# Cell 6: Visualize
!python visualize.py --num_samples 10 --split test
```

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `CUDA out of memory` | Reduce `--batch_size` to 2 or 1 |
| `mamba-ssm install fails` | Skip it — code falls back to PyTorch (slower) |
| `No module named 'model'` | Make sure you're in the `btp/` directory |
| `FileNotFoundError: processed/` | Run `python preprocess_data.py` first |
| `Training killed/timeout` | Use `--resume checkpoints/latest_checkpoint.pth` |
| `num_workers error on Windows` | Use `--num_workers 0` |
