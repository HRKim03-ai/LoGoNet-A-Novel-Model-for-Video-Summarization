"""
Utility functions for training and evaluation
"""

import torch
import numpy as np
from scipy.stats import kendalltau, spearmanr
import warnings


def compute_metrics(predictions, targets, masks):
    """
    Compute Kendall's Tau and Spearman's Rho for each video in the batch
    
    Args:
        predictions: torch.Tensor [batch_size, seq_len] - predicted scores
        targets: torch.Tensor [batch_size, seq_len] - ground truth scores
        masks: torch.Tensor [batch_size, seq_len] - padding mask (True for valid positions)
    
    Returns:
        dict with:
            - kendall_tau: float - average Kendall's Tau
            - spearman_rho: float - average Spearman's Rho
    """
    batch_size = predictions.shape[0]
    kendall_taus = []
    spearman_rhos = []
    
    # Convert to numpy for scipy
    pred_np = predictions.detach().cpu().numpy()
    target_np = targets.detach().cpu().numpy()
    mask_np = masks.detach().cpu().numpy()
    
    for i in range(batch_size):
        # Get valid (non-padded) positions
        valid_mask = mask_np[i]
        if valid_mask.sum() < 2:  # Need at least 2 points for correlation
            continue
        
        pred_seq = pred_np[i][valid_mask]
        target_seq = target_np[i][valid_mask]
        
        # Check for constant sequences (correlation undefined)
        if np.std(pred_seq) < 1e-8 or np.std(target_seq) < 1e-8:
            continue
        
        # Compute Kendall's Tau
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                kendall_result = kendalltau(pred_seq, target_seq)
                if not np.isnan(kendall_result.correlation):
                    kendall_taus.append(kendall_result.correlation)
        except:
            pass
        
        # Compute Spearman's Rho
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                spearman_result = spearmanr(pred_seq, target_seq)
                if not np.isnan(spearman_result.correlation):
                    spearman_rhos.append(spearman_result.correlation)
        except:
            pass
    
    # Average over batch
    avg_kendall = np.mean(kendall_taus) if kendall_taus else 0.0
    avg_spearman = np.mean(spearman_rhos) if spearman_rhos else 0.0
    
    return {
        'kendall_tau': avg_kendall,
        'spearman_rho': avg_spearman
    }


class EarlyStopping:
    """
    Early stopping based on validation metric
    """
    
    def __init__(self, patience=10, mode='max', min_delta=0.0):
        """
        Args:
            patience: number of epochs to wait before stopping
            mode: 'max' (for metrics to maximize) or 'min' (for metrics to minimize)
            min_delta: minimum change to qualify as an improvement
        """
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == 'max':
            self.is_better = lambda current, best: current > best + min_delta
            self.best_score = float('-inf')
        else:
            self.is_better = lambda current, best: current < best - min_delta
            self.best_score = float('inf')
    
    def __call__(self, score):
        """
        Check if should stop early
        
        Args:
            score: current validation metric score
        
        Returns:
            bool: True if should stop early
        """
        if self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def reset(self):
        """Reset early stopping state"""
        self.counter = 0
        self.early_stop = False
        if self.mode == 'max':
            self.best_score = float('-inf')
        else:
            self.best_score = float('inf')

__all__ = ['compute_metrics', 'EarlyStopping']

