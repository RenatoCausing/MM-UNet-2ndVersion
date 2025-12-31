"""
FIVES Dataset Testing Script for MM-UNet
=========================================
Evaluates trained model on FIVES test set with comprehensive metrics:
- Accuracy, Precision, Recall, F1 Score, AUC-ROC
- Dice Score, Mean IoU
- Optionally saves visualization of predictions

Usage:
    python test_fives.py --checkpoint ./model_store/MM_Net_FIVES/best
    python test_fives.py --checkpoint ./model_store/MM_Net_FIVES/best --save-vis

Author: MM-UNet FIVES Testing
"""

import os
import sys
import argparse
import json
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
import monai
from PIL import Image
from tqdm import tqdm
from easydict import EasyDict
from monai.utils import ensure_tuple_rep
from sklearn.metrics import (
    roc_auc_score, 
    precision_recall_fscore_support, 
    accuracy_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
import matplotlib.pyplot as plt

from src.models import give_model
from src import utils
import warnings

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description='MM-UNet FIVES Testing')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint directory')
    parser.add_argument('--data_root', type=str, default='./fives_preprocessed',
                        help='Path to preprocessed FIVES dataset')
    parser.add_argument('--image_size', type=int, default=1024,
                        help='Image size (default: 1024)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--save-vis', action='store_true',
                        help='Save visualization of predictions')
    parser.add_argument('--vis_dir', type=str, default='./visualization/FIVES',
                        help='Directory to save visualizations')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU device ID')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary prediction')
    return parser.parse_args()


def get_test_dataloader(args):
    """Create test data loader for FIVES dataset."""
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
    
    # Load metadata for normalization
    metadata_path = os.path.join(args.data_root, 'dataset_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        image_mean = metadata['normalization']['mean']
        image_std = metadata['normalization']['std']
        print(f"Loaded normalization from metadata: mean={image_mean}, std={image_std}")
    else:
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        print(f"Using default normalization: mean={image_mean}, std={image_std}")
    
    class FIVESTestDataset(Dataset):
        def __init__(self, data_root, image_size=1024, 
                     image_mean=None, image_std=None):
            self.data_root = data_root
            self.image_size = image_size
            self.image_mean = image_mean or [0.485, 0.456, 0.406]
            self.image_std = image_std or [0.229, 0.224, 0.225]
            
            # Get test image paths
            original_dir = os.path.join(data_root, 'test', 'Original')
            segmented_dir = os.path.join(data_root, 'test', 'Segmented')
            
            self.image_paths = []
            self.mask_paths = []
            self.image_names = []
            
            for img_name in sorted(os.listdir(original_dir)):
                if img_name.endswith('.png'):
                    img_path = os.path.join(original_dir, img_name)
                    mask_path = os.path.join(segmented_dir, img_name)
                    
                    if os.path.exists(mask_path):
                        self.image_paths.append(img_path)
                        self.mask_paths.append(mask_path)
                        self.image_names.append(img_name)
            
            print(f"Loaded {len(self.image_paths)} test samples")
            
            self.img_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.image_mean, std=self.image_std)
            ])
            
            self.mask_transform = transforms.Compose([
                transforms.Resize((image_size, image_size), interpolation=Image.NEAREST),
                transforms.ToTensor()
            ])
        
        def __len__(self):
            return len(self.image_paths)
        
        def __getitem__(self, idx):
            image = Image.open(self.image_paths[idx]).convert('RGB')
            mask = Image.open(self.mask_paths[idx]).convert('L')
            
            # Store original for visualization
            original_image = image.copy()
            
            image = self.img_transform(image)
            mask = self.mask_transform(mask)
            mask = (mask > 0.5).float()
            
            return image, mask, self.image_names[idx], self.image_paths[idx]
    
    test_dataset = FIVESTestDataset(
        args.data_root, 
        image_size=args.image_size,
        image_mean=image_mean, 
        image_std=image_std
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    return test_loader


def compute_metrics(all_preds: np.ndarray, all_targets: np.ndarray, 
                    threshold: float = 0.5) -> Dict[str, float]:
    """Compute comprehensive metrics."""
    # Flatten arrays
    preds_flat = all_preds.flatten()
    targets_flat = all_targets.flatten()
    
    # Binary predictions
    preds_binary = (preds_flat > threshold).astype(np.int32)
    targets_binary = (targets_flat > 0.5).astype(np.int32)
    
    # Basic metrics
    accuracy = accuracy_score(targets_binary, preds_binary)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets_binary, preds_binary, average='binary', zero_division=0
    )
    
    # AUC-ROC
    try:
        auc_roc = roc_auc_score(targets_binary, preds_flat)
    except ValueError:
        auc_roc = 0.0
    
    # Average Precision (AUC-PR)
    try:
        auc_pr = average_precision_score(targets_binary, preds_flat)
    except ValueError:
        auc_pr = 0.0
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(targets_binary, preds_binary).ravel()
    
    # Specificity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Dice Score (same as F1 for binary)
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    
    # IoU (Jaccard)
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,  # Same as sensitivity
        'specificity': specificity,
        'f1_score': f1,
        'dice': dice,
        'iou': iou,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    }


def save_visualization(image: np.ndarray, mask: np.ndarray, pred: np.ndarray,
                       save_path: str, image_name: str) -> None:
    """Save visualization of image, ground truth, and prediction."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Ground truth mask
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Prediction
    axes[2].imshow(pred, cmap='gray')
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    # Overlay
    overlay = image.copy()
    # Make prediction red channel
    pred_rgb = np.zeros_like(image)
    pred_rgb[:, :, 0] = (pred * 255).astype(np.uint8)  # Red for prediction
    # Make ground truth green channel
    mask_rgb = np.zeros_like(image)
    mask_rgb[:, :, 1] = (mask * 255).astype(np.uint8)  # Green for ground truth
    
    # Blend
    overlay = (0.5 * overlay + 0.25 * pred_rgb + 0.25 * mask_rgb).astype(np.uint8)
    axes[3].imshow(overlay)
    axes[3].set_title('Overlay (Red=Pred, Green=GT)')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{image_name}_comparison.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_curves(all_preds: np.ndarray, all_targets: np.ndarray, save_dir: str) -> None:
    """Plot ROC and PR curves."""
    preds_flat = all_preds.flatten()
    targets_flat = all_targets.flatten()
    targets_binary = (targets_flat > 0.5).astype(np.int32)
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(targets_binary, preds_flat)
    roc_auc = roc_auc_score(targets_binary, preds_flat)
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    
    # PR Curve
    precision, recall, _ = precision_recall_curve(targets_binary, preds_flat)
    avg_precision = average_precision_score(targets_binary, preds_flat)
    
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AP = {avg_precision:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'curves.png'), dpi=150, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load config
    config = EasyDict(yaml.load(open('config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))
    
    # Load model
    print("Loading model...")
    model = give_model(config)
    
    # Load checkpoint
    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint_path = os.path.join(args.checkpoint, 'pytorch_model.bin')
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        # Try accelerate format
        from accelerate import Accelerator
        accelerator = Accelerator()
        accelerator.load_state(args.checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # Load data
    print("Loading test data...")
    test_loader = get_test_dataloader(args)
    
    # Setup visualization directory
    if args.save_vis:
        vis_dir = os.path.join(args.vis_dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(vis_dir, exist_ok=True)
        os.makedirs(os.path.join(vis_dir, 'predictions'), exist_ok=True)
        print(f"Saving visualizations to: {vis_dir}")
    
    # Setup inference
    inference = monai.inferers.SlidingWindowInferer(
        roi_size=ensure_tuple_rep(args.image_size, 2),
        overlap=0.5,
        sw_device=device,
        device=device
    )
    
    # Post-processing
    post_trans = monai.transforms.Compose([
        monai.transforms.Activations(sigmoid=True),
        monai.transforms.AsDiscrete(threshold=args.threshold)
    ])
    
    # Run inference
    print("\nRunning inference...")
    all_preds = []
    all_targets = []
    all_preds_prob = []  # For AUC calculation
    
    with torch.no_grad():
        for images, masks, names, paths in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            logits = inference(images, model)
            probs = torch.sigmoid(logits)
            preds = post_trans(logits)
            
            # Store for metrics
            all_preds_prob.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_targets.append(masks.cpu().numpy())
            
            # Save visualizations
            if args.save_vis:
                for i in range(len(names)):
                    # Denormalize image for visualization
                    img = images[i].cpu().numpy().transpose(1, 2, 0)
                    # Approximate denormalization
                    img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
                    img = np.clip(img * 255, 0, 255).astype(np.uint8)
                    
                    mask = masks[i].cpu().numpy().squeeze()
                    pred = preds[i].cpu().numpy().squeeze()
                    
                    save_visualization(
                        img, mask, pred,
                        os.path.join(vis_dir, 'predictions'),
                        names[i].replace('.png', '')
                    )
    
    # Concatenate all predictions and targets
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_preds_prob = np.concatenate(all_preds_prob, axis=0)
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(all_preds_prob, all_targets, args.threshold)
    
    # Print results
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"\n📊 Classification Metrics:")
    print(f"   Accuracy:    {metrics['accuracy']:.4f}")
    print(f"   Precision:   {metrics['precision']:.4f}")
    print(f"   Recall:      {metrics['recall']:.4f}")
    print(f"   Specificity: {metrics['specificity']:.4f}")
    print(f"   F1 Score:    {metrics['f1_score']:.4f}")
    
    print(f"\n📊 Segmentation Metrics:")
    print(f"   Dice Score:  {metrics['dice']:.4f}")
    print(f"   IoU (Jaccard): {metrics['iou']:.4f}")
    
    print(f"\n📊 AUC Metrics:")
    print(f"   AUC-ROC:     {metrics['auc_roc']:.4f}")
    print(f"   AUC-PR:      {metrics['auc_pr']:.4f}")
    
    print(f"\n📊 Confusion Matrix:")
    print(f"   True Positives:  {metrics['true_positives']}")
    print(f"   True Negatives:  {metrics['true_negatives']}")
    print(f"   False Positives: {metrics['false_positives']}")
    print(f"   False Negatives: {metrics['false_negatives']}")
    print("=" * 60)
    
    # Save metrics to file
    if args.save_vis:
        metrics_path = os.path.join(vis_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to: {metrics_path}")
        
        # Plot curves
        plot_curves(all_preds_prob, all_targets, vis_dir)
        print(f"Curves saved to: {os.path.join(vis_dir, 'curves.png')}")
    
    # Also save metrics to checkpoint directory
    checkpoint_metrics_path = os.path.join(args.checkpoint, 'test_metrics.json')
    with open(checkpoint_metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {checkpoint_metrics_path}")
    
    return metrics


if __name__ == '__main__':
    main()
