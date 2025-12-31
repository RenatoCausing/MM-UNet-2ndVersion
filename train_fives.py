"""
FIVES Dataset Training Script for MM-UNet
==========================================
Optimized for FIVES dataset (1024x1024 retinal vessel segmentation)
Supports adjustable batch size, learning rate, and epochs from command line.

Usage:
    python train_fives.py --batch_size 4 --lr 0.0005 --epochs 40
    python train_fives.py --batch_size 2 --lr 0.00025 --epochs 40  # For smaller GPU memory

Author: MM-UNet FIVES Training
"""

import os
import sys
import argparse
from datetime import datetime
from typing import Dict

import monai
import torch
import yaml
import numpy as np
from accelerate import Accelerator
from easydict import EasyDict
from monai.utils import ensure_tuple_rep
from objprint import objstr
from timm.optim import optim_factory
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score

from src import utils
from src.models import give_model
from src.optimizer import LinearWarmupCosineAnnealingLR
from src.utils import Logger, load_pretrain_model
import warnings
import torch.nn.functional as F
import torch.nn as nn

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description='MM-UNet FIVES Training')
    parser.add_argument('--batch_size', type=int, default=4, 
                        help='Batch size for training (default: 4)')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Learning rate (default: 0.0005, scaled for batch_size=4)')
    parser.add_argument('--epochs', type=int, default=40,
                        help='Number of training epochs (default: 40)')
    parser.add_argument('--warmup', type=int, default=3,
                        help='Warmup epochs (default: 3)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint')
    parser.add_argument('--data_root', type=str, default='./fives_preprocessed',
                        help='Path to preprocessed FIVES dataset')
    parser.add_argument('--image_size', type=int, default=1024,
                        help='Image size (default: 1024)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--save_every', type=int, default=1,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU device ID')
    return parser.parse_args()


def compute_metrics_sklearn(y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    """Compute additional metrics using sklearn."""
    # Flatten arrays
    y_pred_flat = y_pred.flatten()
    y_true_flat = y_true.flatten()
    
    # Binary predictions (threshold 0.5)
    y_pred_binary = (y_pred_flat > 0.5).astype(np.int32)
    y_true_binary = (y_true_flat > 0.5).astype(np.int32)
    
    # Accuracy
    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    
    # Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_binary, y_pred_binary, average='binary', zero_division=0
    )
    
    # AUC-ROC (using probabilities)
    try:
        auc = roc_auc_score(y_true_binary, y_pred_flat)
    except ValueError:
        auc = 0.0  # If only one class present
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }


def train_one_epoch(model: torch.nn.Module, loss_functions: Dict[str, torch.nn.modules.loss._Loss],
                    train_loader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler,
                    config, metrics: Dict[str, monai.metrics.CumulativeIterationMetric],
                    post_trans: monai.transforms.Compose, accelerator: Accelerator, epoch: int, 
                    step: int, loss_weights, args):
    """Train for one epoch."""
    model.train()
    
    all_preds = []
    all_targets = []
    
    for i, image_batch in enumerate(train_loader):
        logits = model(image_batch[0])
        total_loss = 0
        log = ''
        
        for name, loss_fn in loss_functions.items():
            current_loss = loss_fn(logits, image_batch[1])
            weighted_loss = loss_weights[name] * current_loss
            accelerator.log({'Train/' + name: float(current_loss)}, step=step)
            accelerator.log({'Train/Weighted_' + name: float(weighted_loss)}, step=step)
            total_loss += weighted_loss
            log += f'{name}: {current_loss:.4f} '
        
        val_outputs = post_trans(logits)
        
        for metric_name in metrics:
            metrics[metric_name](y_pred=val_outputs, y=image_batch[1])
        
        # Collect predictions for sklearn metrics
        all_preds.append(torch.sigmoid(logits).cpu().detach().numpy())
        all_targets.append(image_batch[1].cpu().detach().numpy())
        
        accelerator.backward(total_loss)
        optimizer.step()
        optimizer.zero_grad()
        
        accelerator.log({'Train/Total Loss': float(total_loss)}, step=step)
        accelerator.print(
            f'Epoch [{epoch + 1}/{args.epochs}] Training [{i + 1}/{len(train_loader)}] '
            f'Loss: {total_loss:1.5f} {log}', flush=True)
        step += 1
    
    scheduler.step(epoch)
    
    # Compute MONAI metrics
    metric = {}
    for metric_name in metrics:
        batch_acc = metrics[metric_name].aggregate()[0].to(accelerator.device)
        if accelerator.num_processes > 1:
            batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes
        metric.update({f'Train/mean {metric_name}': float(batch_acc.mean())})
        metrics[metric_name].reset()
    
    # Compute sklearn metrics
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    sklearn_metrics = compute_metrics_sklearn(all_preds, all_targets)
    
    for name, value in sklearn_metrics.items():
        metric[f'Train/{name}'] = value
        accelerator.log({f'Train/{name}': value}, step=epoch)
    
    accelerator.print(f'Epoch [{epoch + 1}/{args.epochs}] Training metric {metric}')
    accelerator.log(metric, step=epoch)
    
    return step


@torch.no_grad()
def val_one_epoch(model: torch.nn.Module, loss_functions: Dict[str, torch.nn.modules.loss._Loss],
                  inference: monai.inferers.Inferer, val_loader: torch.utils.data.DataLoader,
                  config: EasyDict, metrics: Dict[str, monai.metrics.CumulativeIterationMetric], 
                  step: int, post_trans: monai.transforms.Compose, accelerator: Accelerator, 
                  epoch: int, args):
    """Validate for one epoch."""
    model.eval()
    
    all_preds = []
    all_targets = []
    
    for i, image_batch in enumerate(val_loader):
        logits = inference(image_batch[0], model)
        total_loss = 0
        log = ''
        
        for name in loss_functions:
            loss = loss_functions[name](logits, image_batch[1])
            accelerator.log({'Val/' + name: float(loss)}, step=step)
            log += f' {name} {float(loss):1.5f} '
            total_loss += loss
        
        val_outputs = post_trans(logits)
        
        for metric_name in metrics:
            metrics[metric_name](y_pred=val_outputs, y=image_batch[1])
        
        # Collect predictions for sklearn metrics
        all_preds.append(torch.sigmoid(logits).cpu().detach().numpy())
        all_targets.append(image_batch[1].cpu().detach().numpy())
        
        accelerator.log({'Val/Total Loss': float(total_loss)}, step=step)
        accelerator.print(
            f'Epoch [{epoch + 1}/{args.epochs}] Validation [{i + 1}/{len(val_loader)}] '
            f'Loss: {total_loss:1.5f} {log}', flush=True)
        step += 1
    
    # Compute MONAI metrics
    metric = {}
    for metric_name in metrics:
        batch_acc = metrics[metric_name].aggregate()[0].to(accelerator.device)
        if accelerator.num_processes > 1:
            batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes
        metrics[metric_name].reset()
        metric.update({f'Val/mean {metric_name}': float(batch_acc.mean())})
    
    # Compute sklearn metrics
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    sklearn_metrics = compute_metrics_sklearn(all_preds, all_targets)
    
    for name, value in sklearn_metrics.items():
        metric[f'Val/{name}'] = value
        accelerator.log({f'Val/{name}': value}, step=epoch)
    
    accelerator.print(f'Epoch [{epoch + 1}/{args.epochs}] Validation metric {metric}')
    accelerator.log(metric, step=epoch)
    
    # Return F1 score as primary metric for model selection
    return torch.Tensor([metric['Val/f1']]).to(accelerator.device), metric, step


def get_fives_dataloader(args, config):
    """Create data loaders for FIVES dataset."""
    import os
    import json
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image
    import torchvision.transforms as transforms
    import torchvision.transforms.functional as TF
    
    # Load metadata
    metadata_path = os.path.join(args.data_root, 'dataset_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        image_mean = metadata['normalization']['mean']
        image_std = metadata['normalization']['std']
        print(f"Loaded normalization from metadata: mean={image_mean}, std={image_std}")
    else:
        # Default ImageNet normalization
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        print(f"Using default normalization: mean={image_mean}, std={image_std}")
    
    class FIVESDataset(Dataset):
        def __init__(self, data_root, split='train', image_size=1024, 
                     image_mean=None, image_std=None):
            self.data_root = data_root
            self.split = split
            self.image_size = image_size
            self.image_mean = image_mean or [0.485, 0.456, 0.406]
            self.image_std = image_std or [0.229, 0.224, 0.225]
            
            # Get image paths
            original_dir = os.path.join(data_root, split, 'Original')
            segmented_dir = os.path.join(data_root, split, 'Segmented')
            
            self.image_paths = []
            self.mask_paths = []
            
            for img_name in sorted(os.listdir(original_dir)):
                if img_name.endswith('.png'):
                    img_path = os.path.join(original_dir, img_name)
                    mask_path = os.path.join(segmented_dir, img_name)
                    
                    if os.path.exists(mask_path):
                        self.image_paths.append(img_path)
                        self.mask_paths.append(mask_path)
            
            print(f"Loaded {len(self.image_paths)} {split} samples")
            
            # Transforms
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
            # Load image and mask
            image = Image.open(self.image_paths[idx]).convert('RGB')
            mask = Image.open(self.mask_paths[idx]).convert('L')
            
            # Apply transforms
            image = self.img_transform(image)
            mask = self.mask_transform(mask)
            
            # Binarize mask
            mask = (mask > 0.5).float()
            
            return image, mask
    
    # Create datasets
    train_dataset = FIVESDataset(
        args.data_root, split='train', image_size=args.image_size,
        image_mean=image_mean, image_std=image_std
    )
    
    test_dataset = FIVESDataset(
        args.data_root, split='test', image_size=args.image_size,
        image_mean=image_mean, image_std=image_std
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader


if __name__ == '__main__':
    args = parse_args()
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # Load config
    config = EasyDict(yaml.load(open('config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))
    
    # Override config with command line args
    config.trainer.num_epochs = args.epochs
    config.trainer.warmup = args.warmup
    config.trainer.lr = args.lr
    config.trainer.resume = args.resume
    
    # Scale learning rate based on batch size (linear scaling rule)
    # Base: batch_size=4, lr=0.0005
    # lr = base_lr * (batch_size / 4)
    if args.lr == 0.0005:  # If using default, apply scaling
        scaled_lr = 0.0005 * (args.batch_size / 4)
        config.trainer.lr = scaled_lr
        print(f"Learning rate scaled to {scaled_lr} for batch size {args.batch_size}")
    
    utils.same_seeds(50)
    
    # Setup logging
    logging_dir = os.path.join(
        os.getcwd(), 'logs',
        f"MM_Net_FIVES_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    accelerator = Accelerator(cpu=False)
    Logger(logging_dir if accelerator.is_local_main_process else None)
    
    accelerator.init_trackers(os.path.split(__file__)[-1].split(".")[0])
    
    print("=" * 60)
    print("MM-UNet FIVES Training Configuration")
    print("=" * 60)
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {config.trainer.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"Image Size: {args.image_size}")
    print(f"Data Root: {args.data_root}")
    print(f"Warmup Epochs: {args.warmup}")
    print(f"Save Every: {args.save_every} epochs")
    print("=" * 60)
    
    accelerator.print(objstr(config))
    
    # Load model
    accelerator.print('Loading Model...')
    model = give_model(config)
    
    # Load data
    accelerator.print('Loading Dataloader...')
    train_loader, val_loader = get_fives_dataloader(args, config)
    
    # Setup inference
    inference = monai.inferers.SlidingWindowInferer(
        roi_size=ensure_tuple_rep(args.image_size, 2), 
        overlap=0.5,
        sw_device=accelerator.device, 
        device=accelerator.device
    )
    
    # Setup metrics (including all requested metrics)
    include_background = True
    metrics = {
        'dice_metric': monai.metrics.DiceMetric(
            include_background=include_background,
            reduction=monai.utils.MetricReduction.MEAN_BATCH, 
            get_not_nans=True
        ),
        'miou_metric': monai.metrics.MeanIoU(
            include_background=include_background, 
            reduction="mean_channel"
        ),
        'f1': monai.metrics.ConfusionMatrixMetric(
            include_background=include_background, 
            metric_name='f1 score'
        ),
        'precision': monai.metrics.ConfusionMatrixMetric(
            include_background=include_background,
            metric_name="precision"
        ),
        'recall': monai.metrics.ConfusionMatrixMetric(
            include_background=include_background, 
            metric_name="recall"
        ),
        'ACC': monai.metrics.ConfusionMatrixMetric(
            include_background=include_background, 
            metric_name="accuracy"
        ),
    }
    
    post_trans = monai.transforms.Compose([
        monai.transforms.Activations(sigmoid=True), 
        monai.transforms.AsDiscrete(threshold=0.5)
    ])
    
    # Setup optimizer - using torch.optim directly for compatibility
    if config.trainer.optimizer.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.trainer.lr,
            weight_decay=config.trainer.weight_decay,
            betas=(0.9, 0.95)
        )
    elif config.trainer.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.trainer.lr,
            weight_decay=config.trainer.weight_decay,
            betas=(0.9, 0.95)
        )
    elif config.trainer.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.trainer.lr,
            weight_decay=config.trainer.weight_decay,
            momentum=0.9
        )
    else:
        # Fallback to timm's create_optimizer_v2
        optimizer = optim_factory.create_optimizer_v2(
            model,
            opt=config.trainer.optimizer,
            weight_decay=config.trainer.weight_decay,
            lr=config.trainer.lr,
            betas=(0.9, 0.95)
        )
    
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer, 
        warmup_epochs=config.trainer.warmup,
        max_epochs=args.epochs
    )
    
    # Loss function
    loss_functions = {
        'dice_focal_loss': monai.losses.DiceFocalLoss(
            smooth_nr=0, smooth_dr=1e-5, 
            to_onehot_y=False, sigmoid=True
        ),
    }
    
    loss_weights = {
        'dice_focal_loss': 1.0
    }
    
    # Initialize training state
    step = 0
    best_epoch = -1
    val_step = 0
    starting_epoch = 0
    best_acc = torch.tensor(0)
    best_class = []
    
    # Prepare with accelerator
    model, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(
        model, optimizer, scheduler, train_loader, val_loader
    )
    
    # Resume if requested
    if config.trainer.resume:
        model_store_path = f'{os.getcwd()}/model_store/MM_Net_FIVES/checkpoint'
        if os.path.exists(model_store_path):
            model, optimizer, scheduler, starting_epoch, train_step, best_acc, best_class = utils.resume_train_state(
                model, 'MM_Net_FIVES', optimizer, scheduler, train_loader, accelerator
            )
            val_step = train_step
            print(f"Resumed from epoch {starting_epoch}")
        else:
            print("No checkpoint found, starting fresh")
    
    best_acc = best_acc.to(accelerator.device)
    
    # Create model store directory
    model_store_dir = f"{os.getcwd()}/model_store/MM_Net_FIVES"
    os.makedirs(model_store_dir, exist_ok=True)
    os.makedirs(f"{model_store_dir}/best", exist_ok=True)
    os.makedirs(f"{model_store_dir}/checkpoint", exist_ok=True)
    
    # Start training
    accelerator.print("=" * 60)
    accelerator.print("Starting Training!")
    accelerator.print("=" * 60)
    
    for epoch in range(starting_epoch, args.epochs):
        # Train
        step = train_one_epoch(
            model, loss_functions, train_loader,
            optimizer, scheduler, config, metrics,
            post_trans, accelerator, epoch, step, loss_weights, args
        )
        
        # Validate
        mean_acc, batch_acc, val_step = val_one_epoch(
            model, loss_functions, inference, val_loader,
            config, metrics, val_step,
            post_trans, accelerator, epoch, args
        )
        
        # Save best model
        if mean_acc > best_acc:
            accelerator.wait_for_everyone()
            accelerator.save_state(output_dir=f"{model_store_dir}/best")
            best_acc = mean_acc
            best_class = batch_acc
            best_epoch = epoch
            accelerator.print(f"New best model saved! F1: {mean_acc:.4f}")
        
        # Save checkpoint every epoch (or every N epochs)
        if (epoch + 1) % args.save_every == 0:
            accelerator.print('Saving checkpoint...')
            accelerator.wait_for_everyone()
            accelerator.save_state(output_dir=f"{model_store_dir}/checkpoint")
            torch.save(
                {'epoch': epoch, 'best_acc': best_acc, 'best_class': batch_acc},
                f'{model_store_dir}/checkpoint/epoch.pth.tar'
            )
            
            # Also save epoch-specific checkpoint
            epoch_dir = f"{model_store_dir}/epoch_{epoch+1}"
            os.makedirs(epoch_dir, exist_ok=True)
            accelerator.save_state(output_dir=epoch_dir)
            torch.save(
                {'epoch': epoch, 'best_acc': mean_acc, 'metrics': batch_acc},
                f'{epoch_dir}/epoch.pth.tar'
            )
        
        accelerator.print(
            f'Epoch [{epoch + 1}/{args.epochs}] Best F1: {best_acc:.4f}, '
            f'Current F1: {mean_acc:.4f}'
        )
    
    # Final summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best F1 Score: {best_acc:.4f}")
    print(f"Best Epoch: {best_epoch + 1}")
    print(f"Best Metrics: {best_class}")
    print(f"Model saved to: {model_store_dir}/best")
    print("=" * 60)
    
    sys.exit(0)
