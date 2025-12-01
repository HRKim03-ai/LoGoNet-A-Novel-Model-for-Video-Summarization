"""
Test script for video summarization models
Evaluates trained models on test set
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import numpy as np
from tqdm import tqdm

# Check required packages
try:
    import h5py
except ImportError:
    print("ERROR: h5py is not installed.")
    print("Please activate the conda environment: conda activate mrhisum")
    sys.exit(1)

from dataset import VideoSummarizationDataset, BatchCollator
from utils import compute_metrics
from models import CSTA, VideoSAGE, EDSNet, LoGoNet


def get_model(model_name, checkpoint_path, **kwargs):
    """Load model from checkpoint"""
    if model_name == 'csta':
        model = CSTA(**kwargs)
    elif model_name == 'videosage':
        model = VideoSAGE(**kwargs)
    elif model_name == 'edsnet':
        model = EDSNet(**kwargs)
    elif model_name == 'logonet':
        model = LoGoNet(**kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Load checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from {checkpoint_path}")
            if 'epoch' in checkpoint:
                print(f"Checkpoint epoch: {checkpoint['epoch']}")
            if 'best_spearman' in checkpoint:
                print(f"Checkpoint best Spearman's Rho: {checkpoint['best_spearman']:.4f}")
        else:
            # Try loading directly (if checkpoint is just state_dict)
            model.load_state_dict(checkpoint)
            print(f"Loaded state dict from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}, using untrained model")
    
    return model


def test(model, dataloader, criterion, device):
    """Test model on test set"""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    all_masks = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Test"):
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
    
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    
    return avg_loss, metrics


def main():
    parser = argparse.ArgumentParser(description='Test video summarization model on test set')
    
    # Model selection
    parser.add_argument('--model', type=str, required=True, choices=['csta', 'videosage', 'edsnet', 'logonet'],
                        help='Model to test')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth file)')
    
    # Data arguments
    parser.add_argument('--dataset_path', type=str, default='dataset/mr_hisum.h5',
                        help='Path to dataset H5 file')
    parser.add_argument('--split_file', type=str, default='dataset/mr_hisum_split.json',
                        help='Path to split JSON file')
    parser.add_argument('--split_id', type=int, default=0,
                        help='Canonical split ID (default: 0)')
    
    # Test arguments
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
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
    parser.add_argument('--output_file', type=str, default=None,
                        help='Optional: Save results to file')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load test dataset
    print(f"\nLoading test dataset...")
    test_dataset = VideoSummarizationDataset(
        mode='test',
        dataset_path=args.dataset_path,
        split_file=args.split_file,
        split_id=args.split_id
    )
    print(f"Test dataset size: {len(test_dataset)} videos")
    
    # Create test dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=BatchCollator(include_metadata=True),
        pin_memory=True
    )
    
    # Model arguments
    model_kwargs = {
        'feature_dim': args.feature_dim,
        'score_hidden': args.score_hidden,
        'dropout': args.dropout
    }
    
    if args.model == 'csta':
        # CSTA doesn't use hidden_dim, uses spatial_hidden and temporal_hidden instead
        model_kwargs.update({
            'spatial_hidden': args.spatial_hidden,
            'temporal_hidden': args.temporal_hidden
        })
    elif args.model == 'videosage':
        # Other models use hidden_dim
        model_kwargs['hidden_dim'] = args.hidden_dim
        model_kwargs.update({
            'num_layers': args.num_layers
        })
    elif args.model == 'edsnet':
        # Other models use hidden_dim
        model_kwargs['hidden_dim'] = args.hidden_dim
        model_kwargs.update({
            'num_mixer_blocks': args.num_mixer_blocks,
            'token_hidden': args.token_hidden,
            'channel_hidden': args.channel_hidden
        })
    elif args.model == 'logonet':
        # Other models use hidden_dim
        model_kwargs['hidden_dim'] = args.hidden_dim
        model_kwargs.update({
            'local_kernel_size': args.local_kernel_size
        })
    
    # Load model
    print(f"\nLoading {args.model} model...")
    model = get_model(args.model, args.checkpoint, **model_kwargs)
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Test
    print(f"\n{'='*60}")
    print(f"Testing {args.model} on test set")
    print(f"{'='*60}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Batch size: {args.batch_size}")
    print(f"{'='*60}\n")
    
    test_loss, test_metrics = test(model, test_loader, criterion, device)
    
    # Print results
    print(f"\n{'='*60}")
    print("TEST RESULTS")
    print(f"{'='*60}")
    print(f"Test Loss: {test_loss:.6f}")
    print(f"Test Kendall's Tau: {test_metrics['kendall_tau']:.4f}")
    print(f"Test Spearman's Rho: {test_metrics['spearman_rho']:.4f}")
    print(f"{'='*60}\n")
    
    # Save to file if requested
    if args.output_file:
        with open(args.output_file, 'w') as f:
            f.write(f"Model: {args.model}\n")
            f.write(f"Checkpoint: {args.checkpoint}\n")
            f.write(f"Test Loss: {test_loss:.6f}\n")
            f.write(f"Test Kendall's Tau: {test_metrics['kendall_tau']:.4f}\n")
            f.write(f"Test Spearman's Rho: {test_metrics['spearman_rho']:.4f}\n")
        print(f"Results saved to {args.output_file}")


if __name__ == '__main__':
    main()

