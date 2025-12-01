"""
Unified training script for video summarization models
Supports: CSTA, VideoSAGE, EDSNet, LoGoNet
"""

import argparse
import os
import sys
import json
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import numpy as np
from tqdm import tqdm
import gc

# Check required packages
try:
    import h5py
except ImportError:
    print("ERROR: h5py is not installed.")
    print("Please activate the conda environment: conda activate mrhisum")
    print("Or install h5py: pip install h5py")
    sys.exit(1)

try:
    from scipy.stats import kendalltau, spearmanr
except ImportError:
    print("ERROR: scipy is not installed.")
    print("Please install scipy: pip install scipy")
    sys.exit(1)

from dataset import VideoSummarizationDataset, BatchCollator
from utils import compute_metrics, EarlyStopping
from models import CSTA, VideoSAGE, EDSNet, LoGoNet


def get_model(model_name, **kwargs):
    """Get model instance"""
    if model_name == 'csta':
        return CSTA(**kwargs)
    elif model_name == 'videosage':
        return VideoSAGE(**kwargs)
    elif model_name == 'edsnet':
        return EDSNet(**kwargs)
    elif model_name == 'logonet':
        return LoGoNet(**kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_optimizer(model_name, model, lr, **kwargs):
    """Get optimizer based on model"""
    if model_name == 'csta':
        # CSTA: Adam with weight decay 1e-7
        return optim.Adam(model.parameters(), lr=lr, weight_decay=1e-7)
    elif model_name == 'logonet':
        # LoGoNet: AdamW with weight decay 1e-4
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    elif model_name in ['videosage', 'edsnet']:
        # VideoSAGE and EDSNet: Adam (no weight decay by default)
        return optim.Adam(model.parameters(), lr=lr, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_scheduler(model_name, optimizer, num_epochs, use_scheduler=False, **kwargs):
    """
    Get learning rate scheduler based on model
    
    If use_scheduler is False, returns None (fixed LR).
    Otherwise, returns a scheduler based on model-specific settings.
    """
    if not use_scheduler:
        # Fixed LR (no scheduler)
        return None
    
    base_lr = optimizer.param_groups[0]['lr']
    
    if model_name == 'csta':
        # CSTA: Warmup + Cosine Annealing
        warmup_epochs = kwargs.get('warmup_epochs', 3)
        max_lr = kwargs.get('max_lr', 3e-4)
        min_lr = kwargs.get('min_lr', 1e-7)
        
        if base_lr > max_lr or base_lr <= 0:
            base_lr = max_lr
            optimizer.param_groups[0]['lr'] = max_lr
        
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch / warmup_epochs) if warmup_epochs > 0 and base_lr > 0 else 1.0
            else:
                progress = (epoch - warmup_epochs) / max(1, (num_epochs - warmup_epochs))
                cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
                lr_range = max_lr - min_lr
                current_lr = min_lr + lr_range * cosine_factor
                return current_lr / base_lr if base_lr > 0 else 1.0
        
        return LambdaLR(optimizer, lr_lambda)
    
    elif model_name == 'edsnet':
        # EDSNet: Fixed LR (no scheduler)
        return None
    
    elif model_name == 'videosage':
        # VideoSAGE: Warmup + Cosine Annealing
        warmup_epochs = kwargs.get('warmup_epochs', 5)
        max_lr = kwargs.get('max_lr', 5e-4)
        min_lr = kwargs.get('min_lr', 1e-7)
        
        if base_lr > max_lr or base_lr <= 0:
            base_lr = max_lr
            optimizer.param_groups[0]['lr'] = max_lr
        
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch / warmup_epochs) if warmup_epochs > 0 and base_lr > 0 else 1.0
            else:
                progress = (epoch - warmup_epochs) / max(1, (num_epochs - warmup_epochs))
                cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
                lr_range = max_lr - min_lr
                current_lr = min_lr + lr_range * cosine_factor
                return current_lr / base_lr if base_lr > 0 else 1.0
        
        return LambdaLR(optimizer, lr_lambda)
    
    elif model_name == 'logonet':
        # LoGoNet: Linear Warmup (10% of epochs) + Cosine Annealing
        warmup_epochs = kwargs.get('warmup_epochs', max(1, int(num_epochs * 0.1)))  # 10% of total epochs
        max_lr = kwargs.get('max_lr', 5e-4)
        min_lr = kwargs.get('min_lr', 1e-7)
        
        if base_lr > max_lr or base_lr <= 0:
            base_lr = max_lr
            optimizer.param_groups[0]['lr'] = max_lr
        
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch / warmup_epochs) if warmup_epochs > 0 and base_lr > 0 else 1.0
            else:
                progress = (epoch - warmup_epochs) / max(1, (num_epochs - warmup_epochs))
                cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
                lr_range = max_lr - min_lr
                current_lr = min_lr + lr_range * cosine_factor
                return current_lr / base_lr if base_lr > 0 else 1.0
        
        return LambdaLR(optimizer, lr_lambda)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_epoch(model, dataloader, optimizer, criterion, device, scaler, 
                gradient_accumulation_steps=1, max_grad_norm=1.0, 
                use_wandb=False, epoch=0, log_interval=100):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Train")):
        features = batch['features'].to(device)
        gtscores = batch['gtscores'].to(device)
        mask = batch['mask'].to(device)
        
        # Forward pass with mixed precision
        with autocast():
            scores = model(features, mask)
            # Apply mask to loss (only compute loss on valid positions)
            loss_per_element = criterion(scores, gtscores)  # [B, T]
            masked_loss = (loss_per_element * mask.float()).sum() / mask.float().sum()
            loss = masked_loss / gradient_accumulation_steps  # Scale loss for gradient accumulation
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Log to wandb periodically during training
            if use_wandb and (batch_idx + 1) % (gradient_accumulation_steps * log_interval) == 0:
                try:
                    import wandb
                    global_step = epoch * len(dataloader) + batch_idx
                    wandb.log({
                        'train/batch_loss': loss.item() * gradient_accumulation_steps,
                        'train/grad_norm': grad_norm.item(),
                        'global_step': global_step
                    })
                except:
                    pass  # wandb not available, skip
        
        total_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    all_masks = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Val"):
            features = batch['features'].to(device)
            gtscores = batch['gtscores'].to(device)
            mask = batch['mask'].to(device)
            
            # Forward pass with mixed precision
            with autocast():
                scores = model(features, mask)
                # Apply mask to loss
                loss_per_element = criterion(scores, gtscores)  # [B, T]
                masked_loss = (loss_per_element * mask.float()).sum() / mask.float().sum()
            
            total_loss += masked_loss.item()
            
            # Store predictions per video (handle variable length sequences)
            batch_size = scores.shape[0]
            for i in range(batch_size):
                valid_len = mask[i].sum().item()
                all_predictions.append(scores[i, :valid_len].cpu())
                all_targets.append(gtscores[i, :valid_len].cpu())
                all_masks.append(mask[i, :valid_len].cpu())
    
    # Stack predictions (each video is separate, so we can stack them)
    # But we need to handle variable lengths, so we'll pad them to max length
    if len(all_predictions) > 0:
        max_len = max(p.shape[0] for p in all_predictions)
        predictions_padded = []
        targets_padded = []
        masks_padded = []
        
        for pred, target, msk in zip(all_predictions, all_targets, all_masks):
            pad_len = max_len - pred.shape[0]
            if pad_len > 0:
                pred = torch.cat([pred, torch.zeros(pad_len)], dim=0)
                target = torch.cat([target, torch.zeros(pad_len)], dim=0)
                msk = torch.cat([msk, torch.zeros(pad_len, dtype=torch.bool)], dim=0)
            predictions_padded.append(pred)
            targets_padded.append(target)
            masks_padded.append(msk)
        
        predictions = torch.stack(predictions_padded, dim=0)
        targets = torch.stack(targets_padded, dim=0)
        masks = torch.stack(masks_padded, dim=0)
    else:
        # Fallback: create empty tensors
        predictions = torch.empty(0, 0)
        targets = torch.empty(0, 0)
        masks = torch.empty(0, 0, dtype=torch.bool)
    
    # Compute metrics
    metrics = compute_metrics(predictions, targets, masks)
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, metrics


def find_optimal_batch_size(model, dataset, collator, device, start_batch_size=8,
                           max_batch_size=256, num_workers=8, use_amp=True, num_test_batches=3, 
                           log_to_wandb=False, batch_size_step=4):
    """
    Find the optimal batch size that doesn't cause OOM
    Note: Model weights are preserved (saved before and restored after)
    
    Args:
        model: PyTorch model
        dataset: Dataset to test with
        collator: Batch collator
        device: Device to run on
        start_batch_size: Starting batch size to test
        max_batch_size: Maximum batch size to test
        num_workers: Number of DataLoader workers
        use_amp: Whether to use mixed precision
        num_test_batches: Number of batches to test
        batch_size_step: Step size for increasing batch size (default: 4)
    
    Returns:
        int: Optimal batch size
    """
    print(f"\n{'='*60}")
    print(f"Finding optimal batch size")
    print(f"{'='*60}")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"Start batch size: {start_batch_size}")
    print(f"Max batch size: {max_batch_size}")
    print(f"Batch size step: {batch_size_step}")
    print(f"{'='*60}\n")
    
    # Save model state before batch size finding (to restore later)
    model_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = GradScaler() if use_amp else None
    
    current_batch_size = start_batch_size
    last_successful_batch_size = None
    # batch_size_step is now a parameter (default: 4, can be customized per model)
    
    results = []
    
    # If start_batch_size fails, try smaller values
    if start_batch_size > 1:
        # Try smaller batch sizes first if start fails
        min_batch_size = 1
    else:
        min_batch_size = start_batch_size
    
    while current_batch_size <= max_batch_size:
        # Clear cache aggressively
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        
        print(f"Testing batch size: {current_batch_size}...", end=" ", flush=True)
        
        try:
            # Create test dataloader (use 0 workers to reduce memory during batch size finding)
            test_loader = DataLoader(
                dataset,
                batch_size=current_batch_size,
                shuffle=False,
                num_workers=0,  # Use 0 workers during batch size finding to save memory
                collate_fn=collator,
                pin_memory=False,
                persistent_workers=False
            )
            
            # Get initial memory
            if device.type == 'cuda':
                initial_memory = torch.cuda.memory_allocated(device) / 1024**2  # MB
            
            # Test a few batches
            for batch_idx, batch in enumerate(test_loader):
                if batch_idx >= num_test_batches:
                    break
                
                features = batch['features'].to(device)
                gtscores = batch['gtscores'].to(device)
                mask = batch['mask'].to(device)
                
                optimizer.zero_grad()
                
                if use_amp:
                    with autocast():
                        scores = model(features, mask)
                        loss_per_element = criterion(scores, gtscores)
                        loss = (loss_per_element * mask.float()).sum() / mask.float().sum()
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    scores = model(features, mask)
                    loss_per_element = criterion(scores, gtscores)
                    loss = (loss_per_element * mask.float()).sum() / mask.float().sum()
                    loss.backward()
                    optimizer.step()
                
                # Check for OOM
                if device.type == 'cuda':
                    torch.cuda.synchronize()
            
            # Get peak memory
            if device.type == 'cuda':
                peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2  # MB
                memory_used = peak_memory - initial_memory
            else:
                memory_used = 0.0
            
            print(f"✓ Success (Memory: {memory_used:.1f} MB)")
            last_successful_batch_size = current_batch_size
            results.append((current_batch_size, memory_used, True))
            
            # Clean up ALL variables before next batch size test
            del test_loader
            if 'features' in locals():
                del features
            if 'gtscores' in locals():
                del gtscores
            if 'mask' in locals():
                del mask
            if 'scores' in locals():
                del scores
            if 'loss' in locals():
                del loss
            if 'loss_per_element' in locals():
                del loss_per_element
            
            # Clear GPU cache and synchronize
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()  # Reset peak memory stats
            
            # Force garbage collection
            gc.collect()
            
            # Wait a bit to ensure memory is freed
            import time
            time.sleep(0.1)
            
            # Move to next batch size AFTER cleanup
            current_batch_size += batch_size_step
            
        except RuntimeError as e:
            error_msg = str(e)
            if "out of memory" in error_msg.lower() or "cuda" in error_msg.lower():
                print(f"✗ OOM Error")
                results.append((current_batch_size, 0.0, False))
                
                # If this was the first attempt and it failed, try smaller batch sizes
                if last_successful_batch_size is None and current_batch_size > min_batch_size:
                    print(f"Trying smaller batch sizes...")
                    # Try smaller batch sizes
                    for smaller_bs in [current_batch_size // 2, current_batch_size // 4, 1]:
                        if smaller_bs < min_batch_size:
                            continue
                        if smaller_bs >= current_batch_size:
                            continue
                        
                        # Clear cache before testing smaller batch size
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                        gc.collect()
                        
                        print(f"Testing batch size: {smaller_bs}...", end=" ", flush=True)
                        try:
                            test_loader = DataLoader(
                                dataset,
                                batch_size=smaller_bs,
                                shuffle=False,
                                num_workers=0,  # Use 0 workers to reduce memory
                                collate_fn=collator,
                                pin_memory=False,
                                persistent_workers=False
                            )
                            
                            if device.type == 'cuda':
                                initial_memory = torch.cuda.memory_allocated(device) / 1024**2
                            
                            for batch_idx, batch in enumerate(test_loader):
                                if batch_idx >= num_test_batches:
                                    break
                                
                                features = batch['features'].to(device)
                                gtscores = batch['gtscores'].to(device)
                                mask = batch['mask'].to(device)
                                
                                optimizer.zero_grad()
                                
                                if use_amp:
                                    with autocast():
                                        scores = model(features, mask)
                                        loss_per_element = criterion(scores, gtscores)
                                        loss = (loss_per_element * mask.float()).sum() / mask.float().sum()
                                    
                                    scaler.scale(loss).backward()
                                    scaler.step(optimizer)
                                    scaler.update()
                                else:
                                    scores = model(features, mask)
                                    loss_per_element = criterion(scores, gtscores)
                                    loss = (loss_per_element * mask.float()).sum() / mask.float().sum()
                                    loss.backward()
                                    optimizer.step()
                                
                                if device.type == 'cuda':
                                    torch.cuda.synchronize()
                            
                            if device.type == 'cuda':
                                peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2
                                memory_used = peak_memory - initial_memory
                            else:
                                memory_used = 0.0
                            
                            print(f"✓ Success (Memory: {memory_used:.1f} MB)")
                            last_successful_batch_size = smaller_bs
                            results.append((smaller_bs, memory_used, True))
                            
                            # Clean up ALL variables before next batch size test
                            del test_loader
                            if 'features' in locals():
                                del features
                            if 'gtscores' in locals():
                                del gtscores
                            if 'mask' in locals():
                                del mask
                            if 'scores' in locals():
                                del scores
                            if 'loss' in locals():
                                del loss
                            if 'loss_per_element' in locals():
                                del loss_per_element
                            
                            # Clear GPU cache and synchronize
                            if device.type == 'cuda':
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()
                                torch.cuda.reset_peak_memory_stats()
                            
                            # Force garbage collection
                            gc.collect()
                            
                            # Wait a bit to ensure memory is freed
                            import time
                            time.sleep(0.1)
                            
                            # Found a working batch size, break
                            break
                            
                        except RuntimeError as e2:
                            error_msg2 = str(e2)
                            if "out of memory" in error_msg2.lower():
                                print(f"✗ OOM")
                                results.append((smaller_bs, 0.0, False))
                                # Clean up after OOM
                                if 'test_loader' in locals():
                                    del test_loader
                                if device.type == 'cuda':
                                    torch.cuda.empty_cache()
                                    torch.cuda.synchronize()
                                gc.collect()
                                continue
                            else:
                                print(f"✗ Error: {error_msg2}")
                                results.append((smaller_bs, 0.0, False))
                                break
                        except Exception as e2:
                            print(f"✗ Error: {str(e2)}")
                            results.append((smaller_bs, 0.0, False))
                            break
                
                break
            else:
                print(f"✗ Error: {error_msg}")
                results.append((current_batch_size, 0.0, False))
                break
        except Exception as e:
            print(f"✗ Unexpected error: {str(e)}")
            results.append((current_batch_size, 0.0, False))
            break
    
    if last_successful_batch_size is None:
        print(f"\n❌ No successful batch size found. Trying batch size 1...")
        # Last resort: try batch size 1
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        
        try:
            test_loader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                collate_fn=collator,
                pin_memory=False
            )
            
            for batch_idx, batch in enumerate(test_loader):
                if batch_idx >= 1:
                    break
                features = batch['features'].to(device)
                gtscores = batch['gtscores'].to(device)
                mask = batch['mask'].to(device)
                with autocast():
                    _ = model(features, mask)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
            
            optimal_batch_size = 1
            print(f"✓ Batch size 1 works")
            del test_loader
        except:
            print(f"✗ Even batch size 1 failed. Please check your GPU memory or model size.")
            optimal_batch_size = 1  # Use 1 anyway, might work with gradient accumulation
    else:
        optimal_batch_size = last_successful_batch_size
    
    print(f"\n{'='*60}")
    print(f"Optimal batch size: {optimal_batch_size}")
    print(f"{'='*60}\n")
    
    # Print summary
    print("Test Results Summary:")
    print(f"{'Batch Size':<12} {'Memory (MB)':<15} {'Status':<10}")
    print("-" * 40)
    for bs, mem, success in results:
        status = "✓ Success" if success else "✗ Failed"
        print(f"{bs:<12} {mem:<15.1f} {status:<10}")
    print()
    
    # Log to wandb if requested
    if log_to_wandb:
        try:
            import wandb
            wandb.log({
                'batch_size_finder/optimal_batch_size': optimal_batch_size,
                'batch_size_finder/start_batch_size': start_batch_size,
                'batch_size_finder/max_batch_size': max_batch_size,
            })
            # Log each test result
            for bs, mem, success in results:
                wandb.log({
                    f'batch_size_finder/test_batch_size_{bs}': 1 if success else 0,
                    f'batch_size_finder/memory_mb_{bs}': mem if success else 0,
                })
        except:
            pass  # wandb not available, skip
    
    # Restore model state (undo any weight changes during batch size finding)
    model.load_state_dict(model_state_dict)
    model = model.to(device)
    
    # Clean up
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    
    print(f"Model weights restored to initial state.\n")
    
    return optimal_batch_size


def main():
    parser = argparse.ArgumentParser(description='Train video summarization model')
    
    # Model selection
    parser.add_argument('--model', type=str, required=True, choices=['csta', 'videosage', 'edsnet', 'logonet'],
                        help='Model to train')
    
    # Data arguments
    parser.add_argument('--dataset_path', type=str, default='dataset/mr_hisum.h5',
                        help='Path to dataset H5 file')
    parser.add_argument('--split_file', type=str, default='dataset/mr_hisum_split.json',
                        help='Path to split JSON file')
    parser.add_argument('--split_id', type=int, default=0,
                        help='Canonical split ID (default: 0)')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--find_batch_size', action='store_true',
                        help='Automatically find optimal batch size before training (skipped if not specified)')
    parser.add_argument('--start_batch_size', type=int, default=8,
                        help='Starting batch size for batch size finder')
    parser.add_argument('--max_batch_size', type=int, default=256,
                        help='Maximum batch size for batch size finder')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (model-specific defaults if not specified)')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loading workers')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum gradient norm for clipping')
    
    # Model-specific arguments
    parser.add_argument('--feature_dim', type=int, default=1024,
                        help='Input feature dimension')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension')
    parser.add_argument('--score_hidden', type=int, default=256,
                        help='Score regression hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    
    # CSTA-specific
    parser.add_argument('--spatial_hidden', type=int, default=512,
                        help='CSTA spatial hidden dimension')
    parser.add_argument('--temporal_hidden', type=int, default=512,
                        help='CSTA temporal hidden dimension')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='Warmup epochs (default: 10, 5% of 200 epochs)')
    parser.add_argument('--max_lr', type=float, default=3e-4,
                        help='Peak learning rate (default: 3e-4 for all models)')
    parser.add_argument('--min_lr', type=float, default=1e-7,
                        help='Minimum learning rate for cosine annealing (default: 1e-7)')
    
    # VideoSAGE-specific
    parser.add_argument('--num_layers', type=int, default=3,
                        help='VideoSAGE number of GCN layers')
    
    # EDSNet-specific
    parser.add_argument('--num_mixer_blocks', type=int, default=4,
                        help='EDSNet number of mixer blocks')
    parser.add_argument('--token_hidden', type=int, default=256,
                        help='EDSNet token mixing hidden dimension')
    parser.add_argument('--channel_hidden', type=int, default=2048,
                        help='EDSNet channel mixing hidden dimension')
    
    # LoGoNet-specific
    parser.add_argument('--local_kernel_size', type=int, default=3,
                        help='LoGoNet local path CNN kernel size (3 or 5)')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory to save logs')
    parser.add_argument('--early_stop_patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='video_summarization',
                        help='W&B project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='W&B run name')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set default epochs and learning rates (model-specific)
    if args.epochs == 200:  # Only override if using default
        if args.model == 'csta':
            args.epochs = 50
        elif args.model == 'edsnet':
            args.epochs = 300
        elif args.model == 'videosage':
            args.epochs = 50
        elif args.model == 'logonet':
            args.epochs = 200  # LoGoNet: 200 epochs
    
    if args.lr is None:
        if args.model == 'csta':
            args.lr = 1e-3  # CSTA: Fixed LR 1e-3
        elif args.model == 'edsnet':
            args.lr = 5e-5  # EDSNet: Fixed LR 5e-5
        elif args.model == 'videosage':
            args.lr = 1e-3  # VideoSAGE: Fixed LR 1e-3
        elif args.model == 'logonet':
            args.lr = 5e-4  # LoGoNet: Base LR 5e-4 (with warmup + cosine)
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Initialize wandb if requested (before batch size finding so we can log it)
    if args.use_wandb:
        import wandb
        run_name = args.wandb_run_name or f"{args.model}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = VideoSummarizationDataset(
        mode='train', 
        dataset_path=args.dataset_path,
        split_file=args.split_file,
        split_id=args.split_id
    )
    val_dataset = VideoSummarizationDataset(
        mode='val',
        dataset_path=args.dataset_path,
        split_file=args.split_file,
        split_id=args.split_id
    )
    test_dataset = VideoSummarizationDataset(
        mode='test',
        dataset_path=args.dataset_path,
        split_file=args.split_file,
        split_id=args.split_id
    )
    
    # Disable pin_memory for LoGoNet to avoid pin memory thread issues
    use_pin_memory = args.model != 'logonet'
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=BatchCollator(include_metadata=False),
        pin_memory=use_pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=BatchCollator(include_metadata=True),
        pin_memory=use_pin_memory
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create model
    print(f"Creating {args.model} model...")
    model_kwargs = {
        'feature_dim': args.feature_dim,
        'score_hidden': args.score_hidden,
        'dropout': args.dropout
    }
    
    if args.model == 'csta':
        model_kwargs.update({
            'spatial_hidden': args.spatial_hidden,
            'temporal_hidden': args.temporal_hidden
        })
    elif args.model == 'videosage':
        model_kwargs.update({
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers
        })
    elif args.model == 'edsnet':
        model_kwargs.update({
            'hidden_dim': args.hidden_dim,
            'num_mixer_blocks': args.num_mixer_blocks,
            'token_hidden': args.token_hidden,
            'channel_hidden': args.channel_hidden
        })
    elif args.model == 'logonet':
        model_kwargs.update({
            'hidden_dim': args.hidden_dim,
            'local_kernel_size': args.local_kernel_size
        })
    
    model = get_model(args.model, **model_kwargs)
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Create initial dataloaders (will be recreated if batch size is found)
    # Disable pin_memory for LoGoNet to avoid pin memory thread issues
    use_pin_memory = args.model != 'logonet'
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=BatchCollator(include_metadata=False),
        pin_memory=use_pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=BatchCollator(include_metadata=True),
        pin_memory=use_pin_memory
    )
    
    # Find optimal batch size if requested (BEFORE training starts)
    if args.find_batch_size:
        print("\n" + "="*60)
        print("BATCH SIZE FINDING MODE")
        print("="*60 + "\n")
        print("Finding optimal batch size before training starts...")
        
        # Set batch size step based on model (VideoSAGE uses larger steps)
        if args.model == 'videosage':
            batch_size_step = 50
        elif args.model == 'edsnet':
            batch_size_step = 8
        elif args.model == 'logonet':
            batch_size_step = 4  # LoGoNet: similar to CSTA
        else:  # CSTA
            batch_size_step = 4
        
        print(f"Batch size step: {batch_size_step}")
        
        optimal_batch_size = find_optimal_batch_size(
            model=model,
            dataset=train_dataset,
            collator=BatchCollator(include_metadata=False),
            device=device,
            start_batch_size=args.start_batch_size,
            max_batch_size=args.max_batch_size,
            num_workers=args.num_workers,
            use_amp=True,
            num_test_batches=3,
            log_to_wandb=args.use_wandb,
            batch_size_step=batch_size_step
        )
        args.batch_size = optimal_batch_size
        
        # Apply safety margin: reduce batch size by 10% for actual training
        # (batch size finding uses num_workers=0, but training uses num_workers > 0)
        if optimal_batch_size > 100:
            safety_margin = 0.9  # 10% reduction for large batch sizes
            args.batch_size = int(optimal_batch_size * safety_margin)
            # Round down to nearest multiple of 8 for efficiency
            args.batch_size = (args.batch_size // 8) * 8
            print(f"Optimal batch size found: {optimal_batch_size}")
            print(f"Applying safety margin: using batch size {args.batch_size} for training")
        else:
            print(f"Optimal batch size found: {args.batch_size}")
        
        # Auto-adjust gradient accumulation if batch size is small
        # Target effective batch size of 16
        target_effective_batch = 16
        if args.batch_size < target_effective_batch:
            suggested_grad_accum = (target_effective_batch + args.batch_size - 1) // args.batch_size
            if args.gradient_accumulation_steps == 1:
                args.gradient_accumulation_steps = suggested_grad_accum
                print(f"Auto-adjusting gradient accumulation to {args.gradient_accumulation_steps} "
                      f"(effective batch size: {args.batch_size * args.gradient_accumulation_steps})")
        
        print("Recreating dataloaders with optimal batch size...\n")
        
        # For very large batch sizes, reduce num_workers to save memory
        actual_num_workers = args.num_workers
        if args.batch_size > 500:
            actual_num_workers = min(2, args.num_workers)  # Reduce workers for large batches
            print(f"Reducing num_workers to {actual_num_workers} for large batch size")
        elif args.batch_size > 200:
            actual_num_workers = min(4, args.num_workers)
        
        # Recreate dataloaders with optimal batch size
        # Disable pin_memory for LoGoNet to avoid pin memory thread issues
        use_pin_memory = args.batch_size < 1000 and args.model != 'logonet'
        # Also disable persistent_workers for LoGoNet to avoid thread issues
        use_persistent_workers = actual_num_workers > 0 and args.model != 'logonet'
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=actual_num_workers,
            collate_fn=BatchCollator(include_metadata=False),
            pin_memory=use_pin_memory,
            persistent_workers=use_persistent_workers
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=actual_num_workers,
            collate_fn=BatchCollator(include_metadata=True),
            pin_memory=use_pin_memory,
            persistent_workers=use_persistent_workers
        )
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Optimizer
    optimizer = get_optimizer(args.model, model, args.lr)
    
    # Scheduler: None (fixed LR for all models)
    # Use scheduler for models that need it (CSTA, VideoSAGE, LoGoNet)
    use_scheduler = args.model in ['csta', 'videosage', 'logonet']
    scheduler = get_scheduler(args.model, optimizer, args.epochs, use_scheduler=use_scheduler,
                             warmup_epochs=args.warmup_epochs, max_lr=args.max_lr, min_lr=args.min_lr)
    warmup_epochs = args.warmup_epochs if use_scheduler else 0
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.early_stop_patience, mode='max')
    
    # Training loop
    best_spearman = -1.0
    best_epoch = 0
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Model: {args.model}")
    if scheduler is not None:
        print(f"Learning rate: {args.lr} (with scheduler: warmup={warmup_epochs} epochs, then cosine annealing)")
    else:
        print(f"Learning rate: {args.lr} (fixed, no scheduler)")
    optimizer_name = "AdamW" if args.model == 'logonet' else "Adam"
    print(f"Optimizer: {optimizer_name}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Train batches per epoch: {len(train_loader)}")
    print(f"Early stopping patience: {args.early_stop_patience}")
    print("-" * 80)
    
    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
            use_wandb=args.use_wandb,
            epoch=epoch
        )
        
        # Memory cleanup after training epoch
        torch.cuda.empty_cache()
        gc.collect()
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        
        # Memory cleanup after validation
        torch.cuda.empty_cache()
        gc.collect()
        
        # Update learning rate (if scheduler exists)
        if scheduler is not None:
            scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print metrics
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss: {val_loss:.6f}")
        print(f"Val Kendall's Tau: {val_metrics['kendall_tau']:.4f}")
        print(f"Val Spearman's Rho: {val_metrics['spearman_rho']:.4f}")
        print(f"LR: {current_lr:.6e}")
        
        # Log to wandb
        if args.use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': train_loss,
                'val/loss': val_loss,
                'val/kendall_tau': val_metrics['kendall_tau'],
                'val/spearman_rho': val_metrics['spearman_rho'],
                'train/lr': current_lr,
                'best_spearman_rho': best_spearman
            })
        
        # Save best model
        if val_metrics['spearman_rho'] > best_spearman:
            best_spearman = val_metrics['spearman_rho']
            best_epoch = epoch + 1
            best_model_path = os.path.join(args.save_dir, f'{args.model}_best.pth')
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_spearman': best_spearman,
                'val_metrics': val_metrics
            }
            if scheduler is not None:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            torch.save(checkpoint, best_model_path)
            print(f"Saved best model (Spearman's Rho: {best_spearman:.4f})")
        
        # Save checkpoint periodically
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.save_dir, f'{args.model}_epoch_{epoch+1}.pth')
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics
            }
            if scheduler is not None:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            torch.save(checkpoint, checkpoint_path)
        
        # Early stopping
        if early_stopping(val_metrics['spearman_rho']):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            print(f"Best Spearman's Rho: {best_spearman:.4f} at epoch {best_epoch}")
            break
    
    print(f"\nTraining completed!")
    print(f"Best Spearman's Rho: {best_spearman:.4f} at epoch {best_epoch}")
    
    # Load best model and test on test set
    print(f"\n{'='*60}")
    print("Testing on test set with best model...")
    print(f"{'='*60}")
    
    # Load best checkpoint
    best_model_path = os.path.join(args.save_dir, f'{args.model}_best.pth')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            model.load_state_dict(checkpoint)
            print(f"Loaded best checkpoint")
    else:
        print(f"Warning: Best checkpoint not found at {best_model_path}, using current model state")
    
    # Create test dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=BatchCollator(include_metadata=True),
        pin_memory=use_pin_memory
    )
    
    # Test the model
    test_loss, test_metrics = validate(model, test_loader, criterion, device)
    
    print(f"\n{'='*60}")
    print("Test Results:")
    print(f"{'='*60}")
    print(f"Test Loss: {test_loss:.6f}")
    print(f"Test Kendall's Tau: {test_metrics['kendall_tau']:.4f}")
    print(f"Test Spearman's Rho: {test_metrics['spearman_rho']:.4f}")
    print(f"Test Pearson R: {test_metrics['pearson_r']:.4f}")
    print(f"{'='*60}\n")
    
    # Log test results to wandb
    if args.use_wandb:
        wandb.log({
            'test/loss': test_loss,
            'test/kendall_tau': test_metrics['kendall_tau'],
            'test/spearman_rho': test_metrics['spearman_rho'],
            'test/pearson_r': test_metrics['pearson_r']
        })
        wandb.finish()


if __name__ == '__main__':
    main()

