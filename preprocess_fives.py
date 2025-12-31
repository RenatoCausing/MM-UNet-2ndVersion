"""
FIVES Dataset Preprocessor
==========================
Preprocesses the FIVES retinal vessel segmentation dataset (800 images, 1024x1024).
- Performs data augmentation (flipping, rotation, scaling, brightness/contrast)
- Splits data into train/test (95:5 ratio)
- Normalizes training set INDEPENDENTLY from test set (no data leakage)
- Outputs preprocessed folders ready for cloud training

Usage:
    python preprocess_fives.py --original_dir D:\DRIVE\FIVES\Original\PNG \
                               --segmented_dir D:\DRIVE\FIVES\Segmented\PNG \
                               --output_dir ./fives_preprocessed \
                               --augmentation_factor 3

Author: For MM-UNet FIVES Training
"""

import os
import argparse
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
from tqdm import tqdm
import json
from typing import Tuple, List, Dict
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def load_image(path: str) -> Image.Image:
    """Load image as PIL Image."""
    return Image.open(path).convert('RGB')


def load_mask(path: str) -> Image.Image:
    """Load segmentation mask as grayscale PIL Image."""
    return Image.open(path).convert('L')


def horizontal_flip(image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
    """Apply horizontal flip."""
    return image.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)


def vertical_flip(image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
    """Apply vertical flip."""
    return image.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(Image.FLIP_TOP_BOTTOM)


def rotate_image(image: Image.Image, mask: Image.Image, angle: float) -> Tuple[Image.Image, Image.Image]:
    """Rotate image and mask by given angle (preserves dimensions)."""
    rotated_img = image.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=(0, 0, 0))
    rotated_mask = mask.rotate(angle, resample=Image.NEAREST, expand=False, fillcolor=0)
    return rotated_img, rotated_mask


def random_rotation(image: Image.Image, mask: Image.Image, 
                    angle_range: Tuple[float, float] = (-30, 30)) -> Tuple[Image.Image, Image.Image]:
    """Apply random rotation within specified range."""
    angle = random.uniform(angle_range[0], angle_range[1])
    return rotate_image(image, mask, angle)


def adjust_brightness(image: Image.Image, factor: float) -> Image.Image:
    """Adjust brightness of image."""
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def adjust_contrast(image: Image.Image, factor: float) -> Image.Image:
    """Adjust contrast of image."""
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


def adjust_saturation(image: Image.Image, factor: float) -> Image.Image:
    """Adjust saturation of image."""
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(factor)


def random_brightness_contrast(image: Image.Image, 
                                brightness_range: Tuple[float, float] = (0.8, 1.2),
                                contrast_range: Tuple[float, float] = (0.8, 1.2)) -> Image.Image:
    """Apply random brightness and contrast adjustments."""
    brightness_factor = random.uniform(brightness_range[0], brightness_range[1])
    contrast_factor = random.uniform(contrast_range[0], contrast_range[1])
    
    image = adjust_brightness(image, brightness_factor)
    image = adjust_contrast(image, contrast_factor)
    return image


def random_scale(image: Image.Image, mask: Image.Image, 
                 scale_range: Tuple[float, float] = (0.8, 1.2)) -> Tuple[Image.Image, Image.Image]:
    """Apply random scaling (resize then crop/pad to original size)."""
    original_size = image.size  # (width, height)
    scale_factor = random.uniform(scale_range[0], scale_range[1])
    
    new_size = (int(original_size[0] * scale_factor), int(original_size[1] * scale_factor))
    
    # Resize
    scaled_img = image.resize(new_size, Image.BILINEAR)
    scaled_mask = mask.resize(new_size, Image.NEAREST)
    
    # Crop or pad to original size
    if scale_factor > 1.0:
        # Crop center
        left = (new_size[0] - original_size[0]) // 2
        top = (new_size[1] - original_size[1]) // 2
        scaled_img = scaled_img.crop((left, top, left + original_size[0], top + original_size[1]))
        scaled_mask = scaled_mask.crop((left, top, left + original_size[0], top + original_size[1]))
    else:
        # Pad with black
        padded_img = Image.new('RGB', original_size, (0, 0, 0))
        padded_mask = Image.new('L', original_size, 0)
        
        paste_x = (original_size[0] - new_size[0]) // 2
        paste_y = (original_size[1] - new_size[1]) // 2
        
        padded_img.paste(scaled_img, (paste_x, paste_y))
        padded_mask.paste(scaled_mask, (paste_x, paste_y))
        
        scaled_img = padded_img
        scaled_mask = padded_mask
    
    return scaled_img, scaled_mask


def gaussian_blur(image: Image.Image, radius: float = 1.0) -> Image.Image:
    """Apply Gaussian blur to image."""
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def add_gaussian_noise(image: Image.Image, mean: float = 0, std: float = 10) -> Image.Image:
    """Add Gaussian noise to image."""
    img_array = np.array(image, dtype=np.float32)
    noise = np.random.normal(mean, std, img_array.shape)
    noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)


def elastic_transform(image: Image.Image, mask: Image.Image, 
                      alpha: float = 50, sigma: float = 5) -> Tuple[Image.Image, Image.Image]:
    """Apply elastic deformation to image and mask."""
    img_array = np.array(image)
    mask_array = np.array(mask)
    
    shape = img_array.shape[:2]
    
    # Generate random displacement fields
    dx = cv2.GaussianBlur((np.random.rand(*shape) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((np.random.rand(*shape) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
    
    # Create meshgrid
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    
    # Apply displacement
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    
    # Remap images
    transformed_img = cv2.remap(img_array, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    transformed_mask = cv2.remap(mask_array, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    
    return Image.fromarray(transformed_img), Image.fromarray(transformed_mask)


class AugmentationPipeline:
    """Augmentation pipeline with various transformations."""
    
    def __init__(self, 
                 enable_flip: bool = True,
                 enable_rotation: bool = True,
                 enable_scale: bool = True,
                 enable_brightness_contrast: bool = True,
                 enable_blur: bool = True,
                 enable_noise: bool = True,
                 enable_elastic: bool = True):
        
        self.enable_flip = enable_flip
        self.enable_rotation = enable_rotation
        self.enable_scale = enable_scale
        self.enable_brightness_contrast = enable_brightness_contrast
        self.enable_blur = enable_blur
        self.enable_noise = enable_noise
        self.enable_elastic = enable_elastic
    
    def augment(self, image: Image.Image, mask: Image.Image, 
                aug_type: str = 'random') -> Tuple[Image.Image, Image.Image, str]:
        """
        Apply augmentation based on type.
        Returns: (augmented_image, augmented_mask, augmentation_description)
        """
        aug_img = image.copy()
        aug_mask = mask.copy()
        aug_desc = []
        
        if aug_type == 'hflip' and self.enable_flip:
            aug_img, aug_mask = horizontal_flip(aug_img, aug_mask)
            aug_desc.append('hflip')
            
        elif aug_type == 'vflip' and self.enable_flip:
            aug_img, aug_mask = vertical_flip(aug_img, aug_mask)
            aug_desc.append('vflip')
            
        elif aug_type == 'hvflip' and self.enable_flip:
            aug_img, aug_mask = horizontal_flip(aug_img, aug_mask)
            aug_img, aug_mask = vertical_flip(aug_img, aug_mask)
            aug_desc.append('hvflip')
            
        elif aug_type == 'rotate' and self.enable_rotation:
            aug_img, aug_mask = random_rotation(aug_img, aug_mask, (-30, 30))
            aug_desc.append('rotate')
            
        elif aug_type == 'rotate90':
            aug_img, aug_mask = rotate_image(aug_img, aug_mask, 90)
            aug_desc.append('rotate90')
            
        elif aug_type == 'rotate180':
            aug_img, aug_mask = rotate_image(aug_img, aug_mask, 180)
            aug_desc.append('rotate180')
            
        elif aug_type == 'rotate270':
            aug_img, aug_mask = rotate_image(aug_img, aug_mask, 270)
            aug_desc.append('rotate270')
            
        elif aug_type == 'scale' and self.enable_scale:
            aug_img, aug_mask = random_scale(aug_img, aug_mask, (0.85, 1.15))
            aug_desc.append('scale')
            
        elif aug_type == 'brightness_contrast' and self.enable_brightness_contrast:
            aug_img = random_brightness_contrast(aug_img, (0.7, 1.3), (0.7, 1.3))
            aug_desc.append('bc')
            
        elif aug_type == 'combo1':
            # Flip + rotation + brightness/contrast
            if random.random() > 0.5:
                aug_img, aug_mask = horizontal_flip(aug_img, aug_mask)
                aug_desc.append('hflip')
            aug_img, aug_mask = random_rotation(aug_img, aug_mask, (-20, 20))
            aug_desc.append('rotate')
            aug_img = random_brightness_contrast(aug_img, (0.8, 1.2), (0.8, 1.2))
            aug_desc.append('bc')
            
        elif aug_type == 'combo2':
            # Scale + flip + contrast
            aug_img, aug_mask = random_scale(aug_img, aug_mask, (0.9, 1.1))
            aug_desc.append('scale')
            if random.random() > 0.5:
                aug_img, aug_mask = vertical_flip(aug_img, aug_mask)
                aug_desc.append('vflip')
            aug_img = adjust_contrast(aug_img, random.uniform(0.8, 1.2))
            aug_desc.append('contrast')
            
        elif aug_type == 'combo3':
            # Elastic + rotation + brightness
            aug_img, aug_mask = elastic_transform(aug_img, aug_mask, alpha=30, sigma=4)
            aug_desc.append('elastic')
            aug_img, aug_mask = random_rotation(aug_img, aug_mask, (-15, 15))
            aug_desc.append('rotate')
            aug_img = adjust_brightness(aug_img, random.uniform(0.85, 1.15))
            aug_desc.append('brightness')
            
        elif aug_type == 'random':
            # Apply random combination of augmentations
            if random.random() > 0.5 and self.enable_flip:
                aug_img, aug_mask = horizontal_flip(aug_img, aug_mask)
                aug_desc.append('hflip')
            if random.random() > 0.5 and self.enable_flip:
                aug_img, aug_mask = vertical_flip(aug_img, aug_mask)
                aug_desc.append('vflip')
            if random.random() > 0.5 and self.enable_rotation:
                aug_img, aug_mask = random_rotation(aug_img, aug_mask, (-25, 25))
                aug_desc.append('rotate')
            if random.random() > 0.5 and self.enable_scale:
                aug_img, aug_mask = random_scale(aug_img, aug_mask, (0.85, 1.15))
                aug_desc.append('scale')
            if random.random() > 0.5 and self.enable_brightness_contrast:
                aug_img = random_brightness_contrast(aug_img, (0.8, 1.2), (0.8, 1.2))
                aug_desc.append('bc')
            if random.random() > 0.3 and self.enable_blur:
                aug_img = gaussian_blur(aug_img, random.uniform(0.5, 1.5))
                aug_desc.append('blur')
                
        return aug_img, aug_mask, '_'.join(aug_desc) if aug_desc else 'none'


def get_image_pairs(original_dir: str, segmented_dir: str) -> List[Dict[str, str]]:
    """Get list of image pairs (original, segmented)."""
    pairs = []
    
    # FIVES naming: 0001.png -> 0001_segment.png
    for i in range(1, 801):
        img_name = f"{i:04d}.png"
        seg_name = f"{i:04d}_segment.png"
        
        img_path = os.path.join(original_dir, img_name)
        seg_path = os.path.join(segmented_dir, seg_name)
        
        if os.path.exists(img_path) and os.path.exists(seg_path):
            pairs.append({
                'id': i,
                'original': img_path,
                'segmented': seg_path,
                'name': img_name
            })
        else:
            if not os.path.exists(img_path):
                print(f"Warning: Original image not found: {img_path}")
            if not os.path.exists(seg_path):
                print(f"Warning: Segmented image not found: {seg_path}")
    
    return pairs


def split_dataset(pairs: List[Dict], train_ratio: float = 0.95, seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """Split dataset into train and test sets."""
    random.seed(seed)
    shuffled = pairs.copy()
    random.shuffle(shuffled)
    
    split_idx = int(len(shuffled) * train_ratio)
    train_pairs = shuffled[:split_idx]
    test_pairs = shuffled[split_idx:]
    
    return train_pairs, test_pairs


def compute_normalization_stats(image_paths: List[str]) -> Dict[str, List[float]]:
    """
    Compute mean and std for normalization from training images only.
    This ensures no data leakage from test set.
    """
    print("Computing normalization statistics from training set...")
    
    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    num_pixels = 0
    
    for img_path in tqdm(image_paths, desc="Computing stats"):
        img = np.array(Image.open(img_path).convert('RGB'), dtype=np.float64) / 255.0
        pixel_sum += img.sum(axis=(0, 1))
        pixel_sq_sum += (img ** 2).sum(axis=(0, 1))
        num_pixels += img.shape[0] * img.shape[1]
    
    mean = pixel_sum / num_pixels
    std = np.sqrt(pixel_sq_sum / num_pixels - mean ** 2)
    
    return {
        'mean': mean.tolist(),
        'std': std.tolist()
    }


def save_image(image: Image.Image, path: str) -> None:
    """Save image to path."""
    image.save(path, 'PNG')


def process_and_save_pair(args_tuple) -> bool:
    """Process and save a single image pair (for parallel processing)."""
    try:
        img_path, mask_path, out_img_path, out_mask_path = args_tuple
        
        img = load_image(img_path)
        mask = load_mask(mask_path)
        
        save_image(img, out_img_path)
        save_image(mask, out_mask_path)
        
        return True
    except Exception as e:
        print(f"Error processing {args_tuple[0]}: {e}")
        return False


def process_and_save_augmented(args_tuple) -> bool:
    """Process, augment, and save a single image pair."""
    try:
        img_path, mask_path, out_img_path, out_mask_path, aug_pipeline, aug_type = args_tuple
        
        img = load_image(img_path)
        mask = load_mask(mask_path)
        
        aug_img, aug_mask, _ = aug_pipeline.augment(img, mask, aug_type)
        
        save_image(aug_img, out_img_path)
        save_image(aug_mask, out_mask_path)
        
        return True
    except Exception as e:
        print(f"Error processing augmentation {args_tuple[0]}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='FIVES Dataset Preprocessor')
    parser.add_argument('--original_dir', type=str, required=True,
                        help='Path to original images directory')
    parser.add_argument('--segmented_dir', type=str, required=True,
                        help='Path to segmented images directory')
    parser.add_argument('--output_dir', type=str, default='./fives_preprocessed',
                        help='Output directory for preprocessed dataset')
    parser.add_argument('--augmentation_factor', type=int, default=3,
                        help='Number of augmented versions per original image (for training only)')
    parser.add_argument('--train_ratio', type=float, default=0.95,
                        help='Ratio of training data (default: 0.95 for 95:5 split)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of parallel workers')
    
    args = parser.parse_args()
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print("=" * 60)
    print("FIVES Dataset Preprocessor")
    print("=" * 60)
    print(f"Original directory: {args.original_dir}")
    print(f"Segmented directory: {args.segmented_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Train ratio: {args.train_ratio} ({int(args.train_ratio * 100)}:{int((1-args.train_ratio) * 100)})")
    print(f"Augmentation factor: {args.augmentation_factor}")
    print("=" * 60)
    
    # Create output directories
    train_original_dir = os.path.join(args.output_dir, 'train', 'Original')
    train_segmented_dir = os.path.join(args.output_dir, 'train', 'Segmented')
    test_original_dir = os.path.join(args.output_dir, 'test', 'Original')
    test_segmented_dir = os.path.join(args.output_dir, 'test', 'Segmented')
    
    for d in [train_original_dir, train_segmented_dir, test_original_dir, test_segmented_dir]:
        ensure_dir(d)
    
    # Get all image pairs
    print("\nFinding image pairs...")
    pairs = get_image_pairs(args.original_dir, args.segmented_dir)
    print(f"Found {len(pairs)} image pairs")
    
    if len(pairs) == 0:
        print("ERROR: No image pairs found! Check your input directories.")
        return
    
    # Split into train/test
    train_pairs, test_pairs = split_dataset(pairs, args.train_ratio, args.seed)
    print(f"\nDataset split:")
    print(f"  Training: {len(train_pairs)} images")
    print(f"  Testing: {len(test_pairs)} images")
    
    # Initialize augmentation pipeline
    aug_pipeline = AugmentationPipeline(
        enable_flip=True,
        enable_rotation=True,
        enable_scale=True,
        enable_brightness_contrast=True,
        enable_blur=True,
        enable_noise=True,
        enable_elastic=True
    )
    
    # Define augmentation types for variety
    augmentation_types = [
        'hflip', 'vflip', 'hvflip',
        'rotate', 'rotate90', 'rotate180', 'rotate270',
        'scale', 'brightness_contrast',
        'combo1', 'combo2', 'combo3', 'random'
    ]
    
    # Process TEST set first (no augmentation, just copy)
    print("\n" + "=" * 60)
    print("Processing TEST set (no augmentation)...")
    print("=" * 60)
    
    test_tasks = []
    for pair in test_pairs:
        img_name = f"{pair['id']:04d}.png"
        seg_name = f"{pair['id']:04d}.png"  # Output naming convention
        
        out_img_path = os.path.join(test_original_dir, img_name)
        out_mask_path = os.path.join(test_segmented_dir, seg_name)
        
        test_tasks.append((pair['original'], pair['segmented'], out_img_path, out_mask_path))
    
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        list(tqdm(executor.map(process_and_save_pair, test_tasks), 
                  total=len(test_tasks), desc="Saving test images"))
    
    # Process TRAINING set with augmentation
    print("\n" + "=" * 60)
    print("Processing TRAINING set (with augmentation)...")
    print("=" * 60)
    
    # First, save original training images
    print("\nSaving original training images...")
    train_original_tasks = []
    for pair in train_pairs:
        img_name = f"{pair['id']:04d}.png"
        seg_name = f"{pair['id']:04d}.png"
        
        out_img_path = os.path.join(train_original_dir, img_name)
        out_mask_path = os.path.join(train_segmented_dir, seg_name)
        
        train_original_tasks.append((pair['original'], pair['segmented'], out_img_path, out_mask_path))
    
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        list(tqdm(executor.map(process_and_save_pair, train_original_tasks),
                  total=len(train_original_tasks), desc="Saving original training images"))
    
    # Now create augmented versions (starting from ID 0801)
    print(f"\nGenerating {args.augmentation_factor} augmented versions per image...")
    
    aug_tasks = []
    aug_id = 801  # Start augmented images at 0801
    
    for pair in train_pairs:
        for aug_idx in range(args.augmentation_factor):
            # Select augmentation type (cycle through types for variety)
            aug_type = augmentation_types[(pair['id'] + aug_idx) % len(augmentation_types)]
            
            img_name = f"{aug_id:04d}.png"
            seg_name = f"{aug_id:04d}.png"
            
            out_img_path = os.path.join(train_original_dir, img_name)
            out_mask_path = os.path.join(train_segmented_dir, seg_name)
            
            aug_tasks.append((pair['original'], pair['segmented'], 
                            out_img_path, out_mask_path, aug_pipeline, aug_type))
            aug_id += 1
    
    # Process augmentations in parallel
    successful = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_and_save_augmented, task) for task in aug_tasks]
        for future in tqdm(as_completed(futures), total=len(aug_tasks), desc="Generating augmentations"):
            if future.result():
                successful += 1
            else:
                failed += 1
    
    print(f"Augmentation complete: {successful} successful, {failed} failed")
    
    # Compute normalization statistics from TRAINING SET ONLY
    print("\n" + "=" * 60)
    print("Computing normalization statistics (from training set only)...")
    print("=" * 60)
    
    train_image_paths = [os.path.join(train_original_dir, f) 
                         for f in os.listdir(train_original_dir) if f.endswith('.png')]
    norm_stats = compute_normalization_stats(train_image_paths)
    
    print(f"Training set normalization:")
    print(f"  Mean: {norm_stats['mean']}")
    print(f"  Std:  {norm_stats['std']}")
    
    # Save metadata
    metadata = {
        'dataset': 'FIVES',
        'original_images': 800,
        'image_size': [1024, 1024],
        'train_test_split': f"{int(args.train_ratio * 100)}:{int((1-args.train_ratio) * 100)}",
        'train_original_count': len(train_pairs),
        'train_augmented_count': len(train_pairs) * args.augmentation_factor,
        'train_total_count': len(train_pairs) + len(train_pairs) * args.augmentation_factor,
        'test_count': len(test_pairs),
        'augmentation_factor': args.augmentation_factor,
        'normalization': {
            'mean': norm_stats['mean'],
            'std': norm_stats['std'],
            'computed_from': 'training_set_only'
        },
        'train_ids': [p['id'] for p in train_pairs],
        'test_ids': [p['id'] for p in test_pairs],
        'seed': args.seed
    }
    
    metadata_path = os.path.join(args.output_dir, 'dataset_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nMetadata saved to: {metadata_path}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE!")
    print("=" * 60)
    print(f"\nOutput directory structure:")
    print(f"  {args.output_dir}/")
    print(f"    ├── train/")
    print(f"    │   ├── Original/    ({metadata['train_total_count']} images)")
    print(f"    │   └── Segmented/   ({metadata['train_total_count']} masks)")
    print(f"    ├── test/")
    print(f"    │   ├── Original/    ({metadata['test_count']} images)")
    print(f"    │   └── Segmented/   ({metadata['test_count']} masks)")
    print(f"    └── dataset_metadata.json")
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  • Original images: 800")
    print(f"  • Training samples (with augmentation): {metadata['train_total_count']}")
    print(f"  • Testing samples: {metadata['test_count']}")
    print(f"  • Total samples: {metadata['train_total_count'] + metadata['test_count']}")
    
    print(f"\n🔧 Normalization (use these values in config.yml):")
    print(f"  image_mean: {norm_stats['mean']}")
    print(f"  image_std: {norm_stats['std']}")
    
    print(f"\n✅ Ready for cloud training!")
    print(f"   Zip this folder and upload to Google Drive.")


if __name__ == '__main__':
    main()
