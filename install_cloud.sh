#!/bin/bash
# =============================================================================
# MM-UNet FIVES Cloud Installation Script
# =============================================================================
# For use with vast.ai PyTorch template (Python 3.10.x, Tesla V100)
# Run: chmod +x install_cloud.sh && bash install_cloud.sh
# =============================================================================

set -e  # Exit on error

echo "=============================================="
echo "MM-UNet FIVES - Cloud Environment Setup"
echo "=============================================="
echo ""

# Check Python version
PYTHON_VERSION=$(python3.10 --version 2>&1 || python3 --version 2>&1 || python --version 2>&1)
echo "Python version: $PYTHON_VERSION"

# Determine Python command
if command -v python3.10 &> /dev/null; then
    PY="python3.10"
    PIP="python3.10 -m pip"
elif command -v python3 &> /dev/null; then
    PY="python3"
    PIP="python3 -m pip"
else
    PY="python"
    PIP="python -m pip"
fi

echo "Using Python: $PY"
echo "Using pip: $PIP"
echo ""

# Upgrade pip
echo "[1/12] Upgrading pip..."
$PIP install --upgrade pip

# Install PyTorch with CUDA 11.8 (compatible with V100)
echo "[2/12] Installing PyTorch with CUDA support..."
$PIP install torch==2.0.0+cu118 torchvision==0.15.0+cu118 torchaudio==2.0.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install core dependencies
echo "[3/12] Installing core dependencies..."
$PIP install \
    numpy>=1.21.0 \
    opencv-python>=4.5.0 \
    Pillow>=9.0.0 \
    scipy>=1.7.0 \
    scikit-learn>=1.0.0 \
    scikit-image>=0.19.0

# Install deep learning libraries
echo "[4/12] Installing deep learning libraries..."
$PIP install \
    timm==0.4.12 \
    monai>=1.0.0 \
    accelerate>=0.18.0 \
    einops>=0.6.0

# Install utilities
echo "[5/12] Installing utility packages..."
$PIP install \
    easydict \
    objprint>=0.2.3 \
    pyyaml \
    tqdm \
    matplotlib \
    pandas \
    tensorboard \
    wandb

# Install medical imaging packages
echo "[6/12] Installing medical imaging packages..."
$PIP install \
    SimpleITK \
    nibabel \
    albumentations

# Install additional requirements
echo "[7/12] Installing additional packages..."
$PIP install \
    mmengine \
    yacs \
    pathlib \
    openpyxl

# Install gdown for Google Drive downloads
echo "[8/12] Installing gdown for Google Drive..."
$PIP install gdown

# Install packaging (needed for mamba)
echo "[9/12] Installing packaging..."
$PIP install packaging

# Install causal-conv1d (required by mamba)
echo "[10/12] Installing causal-conv1d..."
$PIP install causal-conv1d>=1.1.0

# Install mamba-ssm
echo "[11/12] Installing mamba-ssm..."
$PIP install mamba-ssm

# Verify installations
echo "[12/12] Verifying installations..."
$PY -c "import torch; print(f'PyTorch: {torch.__version__}')"
$PY -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
$PY -c "import torch; print(f'CUDA version: {torch.version.cuda}')" || echo "CUDA version check skipped"
$PY -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')" || echo "GPU count check skipped"
$PY -c "import monai; print(f'MONAI: {monai.__version__}')"
$PY -c "import timm; print(f'timm: {timm.__version__}')"
$PY -c "import accelerate; print(f'accelerate: {accelerate.__version__}')"
$PY -c "import mamba_ssm; print(f'mamba_ssm: installed')" || echo "mamba_ssm check skipped"

# Set CUDA memory allocation config for V100
echo ""
echo "Setting environment variables..."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo 'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True' >> ~/.bashrc

echo ""
echo "=============================================="
echo "Installation Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. Download your preprocessed dataset from Google Drive:"
echo "   gdown --id YOUR_GDRIVE_FILE_ID"
echo "   unzip fives_preprocessed.zip"
echo ""
echo "2. Start training:"
echo "   $PY train_fives.py --batch_size 4 --lr 0.0005 --epochs 40"
echo ""
echo "3. Run testing:"
echo "   $PY test_fives.py --checkpoint ./model_store/MM_Net_FIVES/best"
echo ""
echo "=============================================="
