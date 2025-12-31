# =============================================================================
# MM-UNet FIVES - Quick Start Commands Reference
# =============================================================================
# Copy and paste these commands in your cloud terminal (vast.ai)
# =============================================================================

# =============================================================================
# STEP 1: Clone Repository
# =============================================================================
git clone https://github.com/YOUR_USERNAME/MM-UNet-2ndVersion.git
cd MM-UNet-2ndVersion

# =============================================================================
# STEP 2: Install Dependencies
# =============================================================================
chmod +x install_cloud.sh && bash install_cloud.sh

# OR Manual installation:
# python3.10 -m pip install torch==2.0.0+cu118 torchvision==0.15.0+cu118 --index-url https://download.pytorch.org/whl/cu118
# python3.10 -m pip install -r requirements_cloud.txt
# python3.10 -m pip install mmengine yacs

# =============================================================================
# STEP 3: Download Dataset from Google Drive
# =============================================================================
# Replace YOUR_FILE_ID with your actual Google Drive file ID
gdown --id YOUR_FILE_ID
unzip fives_preprocessed.zip

# =============================================================================
# STEP 4: Set Environment Variables
# =============================================================================
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# =============================================================================
# STEP 5: Training
# =============================================================================

# Default training (batch_size=4, lr=0.0005, epochs=40)
python3.10 train_fives.py

# Custom training parameters
python3.10 train_fives.py --batch_size 4 --lr 0.0005 --epochs 40

# If you have less GPU memory, reduce batch size:
python3.10 train_fives.py --batch_size 2 --lr 0.00025 --epochs 40

# Resume from checkpoint
python3.10 train_fives.py --batch_size 4 --lr 0.0005 --epochs 40 --resume

# =============================================================================
# STEP 6: Testing
# =============================================================================

# Test with best checkpoint
python3.10 test_fives.py --checkpoint ./model_store/MM_Net_FIVES/best

# Test with visualizations
python3.10 test_fives.py --checkpoint ./model_store/MM_Net_FIVES/best --save-vis

# Test specific epoch
python3.10 test_fives.py --checkpoint ./model_store/MM_Net_FIVES/epoch_40

# =============================================================================
# USEFUL COMMANDS
# =============================================================================

# Check GPU status
nvidia-smi

# Monitor GPU during training
watch -n 1 nvidia-smi

# Check disk usage
df -h

# Check Python version
python3.10 --version

# Check PyTorch CUDA
python3.10 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"

# =============================================================================
# TROUBLESHOOTING
# =============================================================================

# If CUDA out of memory:
# 1. Reduce batch size: --batch_size 2 or --batch_size 1
# 2. Reduce image size: --image_size 512
# 3. Enable gradient checkpointing (if available)

# If gdown fails:
# pip install --upgrade gdown
# Or use wget with direct link

# If package conflicts:
# python3.10 -m pip install --force-reinstall PACKAGE_NAME
