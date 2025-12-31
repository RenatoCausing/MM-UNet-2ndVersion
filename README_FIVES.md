# MM-UNet FIVES - Retinal Vessel Segmentation

This repository contains the MM-UNet implementation optimized for the **FIVES dataset** (800 images, 1024x1024) for retinal vessel segmentation.

## 📋 Overview

- **Dataset**: FIVES (Fundus Image Vessel Segmentation)
- **Image Size**: 1024 × 1024
- **Split**: 95% Training / 5% Testing
- **Augmentation**: Flipping, Rotation, Scaling, Brightness/Contrast adjustments
- **Metrics**: Accuracy, Precision, Recall, F1-Score, AUC-ROC, Dice, IoU

## 🚀 Quick Start

### Step 1: Local Preprocessing (on your PC)

```bash
# Install local requirements
pip install -r requirements_local.txt

# Run preprocessing
python preprocess_fives.py \
    --original_dir "D:\DRIVE\FIVES\Original\PNG" \
    --segmented_dir "D:\DRIVE\FIVES\Segmented\PNG" \
    --output_dir ./fives_preprocessed \
    --augmentation_factor 3

# Or use the batch script (Windows)
preprocess_windows.bat
```

### Step 2: Upload to Google Drive

1. Zip the `fives_preprocessed` folder
2. Upload to Google Drive
3. Get the file ID from the shareable link:
   - Link format: `https://drive.google.com/file/d/FILE_ID/view`
   - Copy the `FILE_ID` part

### Step 3: Cloud Training (vast.ai)

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/MM-UNet-2ndVersion.git
cd MM-UNet-2ndVersion

# Install dependencies
chmod +x install_cloud.sh && bash install_cloud.sh

# Download dataset
gdown --id YOUR_FILE_ID
unzip fives_preprocessed.zip

# Set environment
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Train
python3.10 train_fives.py --batch_size 4 --lr 0.0005 --epochs 40

# Test
python3.10 test_fives.py --checkpoint ./model_store/MM_Net_FIVES/best --save-vis
```

## 📁 Project Structure

```
MM-UNet-2ndVersion/
├── preprocess_fives.py      # Dataset preprocessing & augmentation
├── train_fives.py           # Training script with CLI args
├── test_fives.py            # Testing with all metrics
├── config.yml               # Configuration (updated for FIVES)
├── install_cloud.sh         # Cloud installation script
├── run_training.sh          # One-click training script
├── run_testing.sh           # One-click testing script
├── requirements_local.txt   # Requirements for local preprocessing
├── requirements_cloud.txt   # Requirements for cloud training
├── preprocess_windows.bat   # Windows batch script for preprocessing
├── CLOUD_COMMANDS.md        # Quick reference for cloud commands
└── src/                     # Model and utility code
```

## 🔧 Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--batch_size` | 4 | Batch size (reduce if OOM) |
| `--lr` | 0.0005 | Learning rate (auto-scales with batch size) |
| `--epochs` | 40 | Number of training epochs |
| `--image_size` | 1024 | Input image size |
| `--warmup` | 3 | Warmup epochs |
| `--save_every` | 1 | Save checkpoint every N epochs |
| `--resume` | False | Resume from checkpoint |

### Learning Rate Scaling

The learning rate automatically scales with batch size using the linear scaling rule:
- Base: `batch_size=4`, `lr=0.0005`
- Formula: `lr = 0.0005 × (batch_size / 4)`

| Batch Size | Learning Rate |
|------------|---------------|
| 1 | 0.000125 |
| 2 | 0.00025 |
| 4 | 0.0005 |
| 8 | 0.001 |

## 📊 Metrics

The testing script computes:

- **Classification**: Accuracy, Precision, Recall (Sensitivity), Specificity, F1-Score
- **Segmentation**: Dice Score, IoU (Jaccard Index)
- **AUC**: ROC-AUC, PR-AUC
- **Visualizations**: ROC/PR curves, prediction overlays

## 📦 Dataset Structure

After preprocessing, the dataset has this structure:

```
fives_preprocessed/
├── train/
│   ├── Original/       # Training images (original + augmented)
│   └── Segmented/      # Training masks (original + augmented)
├── test/
│   ├── Original/       # Test images (no augmentation)
│   └── Segmented/      # Test masks
└── dataset_metadata.json  # Normalization stats & split info
```

### Important Notes

1. **Normalization**: Training set normalization is computed INDEPENDENTLY from test set (no data leakage)
2. **Augmentation**: Only applied to training set
3. **File naming**: Original images 0001-0800, augmented start at 0801

## 🎯 Preprocessing Options

```bash
python preprocess_fives.py \
    --original_dir PATH           # Path to original images
    --segmented_dir PATH          # Path to segmented images
    --output_dir ./output         # Output directory
    --augmentation_factor 3       # Augmentations per image
    --train_ratio 0.95            # Train/test split ratio
    --seed 42                     # Random seed
    --num_workers 4               # Parallel workers
```

### Augmentations Applied

- Horizontal/Vertical Flip
- 90°/180°/270° Rotation
- Random Rotation (±30°)
- Random Scaling (0.85-1.15x)
- Brightness/Contrast Adjustment
- Gaussian Blur
- Elastic Deformation
- Combined augmentations

## 🔥 GPU Memory

For Tesla V100 (16GB/32GB):

| Batch Size | Image Size | Memory Usage |
|------------|------------|--------------|
| 4 | 1024 | ~12GB |
| 2 | 1024 | ~8GB |
| 1 | 1024 | ~5GB |
| 4 | 512 | ~6GB |

If you encounter OOM errors:
1. Reduce `--batch_size` (also reduce `--lr` proportionally)
2. Reduce `--image_size`
3. Use `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

## 📝 Checkpoints

Checkpoints are saved to `./model_store/MM_Net_FIVES/`:

- `best/` - Best model (highest F1 score)
- `checkpoint/` - Latest checkpoint
- `epoch_N/` - Checkpoint at epoch N

## 🐛 Troubleshooting

### gdown fails
```bash
pip install --upgrade gdown
# Or use wget with direct link
```

### CUDA out of memory
```bash
# Reduce batch size
python3.10 train_fives.py --batch_size 2 --lr 0.00025 --epochs 40
```

### Package conflicts
```bash
python3.10 -m pip install --force-reinstall PACKAGE_NAME
```

### Missing images during preprocessing
- Ensure images are named `0001.png` to `0800.png`
- Ensure masks are named `0001_segment.png` to `0800_segment.png`

## 📄 License

See [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

- FIVES Dataset
- MM-UNet original implementation
