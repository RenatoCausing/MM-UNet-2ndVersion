#!/bin/bash
# =============================================================================
# MM-UNet FIVES - Complete Cloud Training Script
# =============================================================================
# This is a single script that handles the entire cloud training workflow
# Run: chmod +x run_training.sh && bash run_training.sh
# =============================================================================

set -e

echo "=============================================="
echo "MM-UNet FIVES - Cloud Training"
echo "=============================================="
echo ""

# =============================================
# CONFIGURATION - MODIFY THESE VALUES
# =============================================
GDRIVE_FILE_ID="YOUR_GDRIVE_FILE_ID"  # Replace with your Google Drive file ID
BATCH_SIZE=4
LEARNING_RATE=0.0005
EPOCHS=40
IMAGE_SIZE=1024
# =============================================

# Determine Python command
if command -v python3.10 &> /dev/null; then
    PY="python3.10"
elif command -v python3 &> /dev/null; then
    PY="python3"
else
    PY="python"
fi

echo "Using Python: $PY"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "Epochs: $EPOCHS"
echo "Image Size: $IMAGE_SIZE"
echo ""

# Check if dataset exists, if not download it
if [ ! -d "fives_preprocessed" ]; then
    echo "[1/4] Downloading dataset from Google Drive..."
    if [ "$GDRIVE_FILE_ID" = "YOUR_GDRIVE_FILE_ID" ]; then
        echo "ERROR: Please set your Google Drive file ID in this script!"
        echo "Edit the GDRIVE_FILE_ID variable at the top of this file."
        exit 1
    fi
    gdown --id $GDRIVE_FILE_ID
    echo "[1/4] Extracting dataset..."
    unzip -q fives_preprocessed.zip
    rm fives_preprocessed.zip
else
    echo "[1/4] Dataset already exists, skipping download..."
fi

# Set CUDA memory config
echo "[2/4] Setting environment variables..."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Verify GPU
echo "[3/4] Checking GPU..."
$PY -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}, Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')" || echo "GPU check skipped"

# Start training
echo "[4/4] Starting training..."
echo ""
echo "=============================================="

$PY train_fives.py \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --epochs $EPOCHS \
    --image_size $IMAGE_SIZE \
    --num_workers 4 \
    --save_every 1 \
    --data_root ./fives_preprocessed

echo ""
echo "=============================================="
echo "Training Complete!"
echo "=============================================="
echo ""
echo "Model saved to: ./model_store/MM_Net_FIVES/"
echo ""
echo "To run testing:"
echo "$PY test_fives.py --checkpoint ./model_store/MM_Net_FIVES/best --save-vis"
echo ""
