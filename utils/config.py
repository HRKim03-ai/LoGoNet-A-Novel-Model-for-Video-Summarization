"""
Configuration management for training
"""

import argparse
import json
import os
from pathlib import Path


def get_base_parser():
    """Base argument parser with common arguments"""
    parser = argparse.ArgumentParser(description='Video Summarization Training')
    
    # Data arguments
    parser.add_argument('--dataset_path', type=str, default='dataset/mr_hisum.h5',
                        help='Path to mr_hisum.h5 file')
    parser.add_argument('--split_file', type=str, default='dataset/mr_hisum_split.json',
                        help='Path to split JSON file')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (default: 32 for RTX 3080 10GB, reduce if OOM)')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loading workers (default: 8 for GPU, 4 for CPU)')
    parser.add_argument('--torch_threads', type=int, default=None,
                        help='Number of threads for PyTorch operations (None = auto, CPU only)')
    parser.add_argument('--prefetch_factor', type=int, default=4,
                        help='Number of batches prefetched by each worker (default: 4 for GPU, 2 for CPU)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs (default: 200)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=5.0,
                        help='Gradient clipping value')
    
    # LR scheduling
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Number of warmup epochs for learning rate (default: 5)')
    parser.add_argument('--lr_patience', type=int, default=10,
                        help='Patience for ReduceLROnPlateau scheduler (default: 10)')
    parser.add_argument('--lr_factor', type=float, default=0.1,
                        help='Factor for reducing learning rate (default: 0.1)')
    
    # Early stopping
    parser.add_argument('--early_stop_patience', type=int, default=15,
                        help='Patience for early stopping (default: 15, 0 to disable)')
    
    # Model arguments
    parser.add_argument('--feature_dim', type=int, default=1024,
                        help='Input feature dimension')
    
    # Logging and saving
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory to save logs')
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--eval_every', type=int, default=1,
                        help='Evaluate on validation set every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Logging / monitoring
    parser.add_argument('--use_wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='mrhisum',
                        help='Weights & Biases project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Weights & Biases run name (optional)')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='Weights & Biases entity/team (optional)')
    
    return parser


def save_config(args, save_path):
    """Save configuration to JSON file"""
    config_dict = vars(args)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=2)


def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    return argparse.Namespace(**config_dict)

