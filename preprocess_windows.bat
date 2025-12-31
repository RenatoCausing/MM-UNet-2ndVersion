@echo off
REM =============================================================================
REM MM-UNet FIVES - Windows Preprocessing Script
REM =============================================================================
REM Run this on your local PC to preprocess the FIVES dataset
REM Before running, make sure you have Python 3.10+ installed with:
REM   pip install numpy opencv-python Pillow tqdm
REM =============================================================================

echo ============================================
echo MM-UNet FIVES Dataset Preprocessor
echo ============================================
echo.

REM Set paths - UPDATE THESE TO YOUR ACTUAL PATHS
set ORIGINAL_DIR=D:\DRIVE\FIVES\Original\PNG
set SEGMENTED_DIR=D:\DRIVE\FIVES\Segmented\PNG
set OUTPUT_DIR=.\fives_preprocessed
set AUG_FACTOR=3

echo Original images: %ORIGINAL_DIR%
echo Segmented images: %SEGMENTED_DIR%
echo Output directory: %OUTPUT_DIR%
echo Augmentation factor: %AUG_FACTOR%
echo.

REM Check if directories exist
if not exist "%ORIGINAL_DIR%" (
    echo ERROR: Original directory not found: %ORIGINAL_DIR%
    echo Please update the ORIGINAL_DIR path in this script.
    pause
    exit /b 1
)

if not exist "%SEGMENTED_DIR%" (
    echo ERROR: Segmented directory not found: %SEGMENTED_DIR%
    echo Please update the SEGMENTED_DIR path in this script.
    pause
    exit /b 1
)

echo Starting preprocessing...
echo.

python preprocess_fives.py ^
    --original_dir "%ORIGINAL_DIR%" ^
    --segmented_dir "%SEGMENTED_DIR%" ^
    --output_dir "%OUTPUT_DIR%" ^
    --augmentation_factor %AUG_FACTOR% ^
    --train_ratio 0.95 ^
    --num_workers 4

echo.
echo ============================================
echo Preprocessing Complete!
echo ============================================
echo.
echo Output saved to: %OUTPUT_DIR%
echo.
echo Next steps:
echo 1. Zip the fives_preprocessed folder
echo 2. Upload to Google Drive
echo 3. Get the shareable link and extract the file ID
echo 4. Use gdown in cloud to download
echo.
pause
