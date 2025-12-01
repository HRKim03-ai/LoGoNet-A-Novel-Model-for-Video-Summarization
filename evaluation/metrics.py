"""
Evaluation metrics for video summarization
"""

import torch
import numpy as np
from scipy.stats import kendalltau, spearmanr


def evaluate_batch(predicted_scores, ground_truth_scores, masks):
    """
    Evaluate batch of predictions using Kendall's tau and Spearman's rho
    
    Args:
        predicted_scores: torch.Tensor [batch_size, max_seq_len] - predicted importance scores
        ground_truth_scores: torch.Tensor [batch_size, max_seq_len] - ground truth scores
        masks: torch.Tensor [batch_size, max_seq_len] - padding masks (True for valid positions)
    
    Returns:
        dict with:
            - kendall_tau: average Kendall's tau across batch
            - spearman_rho: average Spearman's rho across batch
    """
    # Convert to numpy if needed
    if isinstance(predicted_scores, torch.Tensor):
        predicted_scores = predicted_scores.cpu().numpy()
    if isinstance(ground_truth_scores, torch.Tensor):
        ground_truth_scores = ground_truth_scores.cpu().numpy()
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()
    
    batch_size = predicted_scores.shape[0]
    kendall_taus = []
    spearman_rhos = []
    
    for i in range(batch_size):
        # Get valid positions (where mask is True)
        valid_mask = masks[i].astype(bool)
        
        if valid_mask.sum() < 2:
            # Need at least 2 elements for correlation
            continue
        
        pred = predicted_scores[i][valid_mask]
        gt = ground_truth_scores[i][valid_mask]
        
        # Compute Kendall's tau
        try:
            tau, _ = kendalltau(pred, gt)
            if not np.isnan(tau):
                kendall_taus.append(tau)
        except:
            pass
        
        # Compute Spearman's rho
        try:
            rho, _ = spearmanr(pred, gt)
            if not np.isnan(rho):
                spearman_rhos.append(rho)
        except:
            pass
    
    # Average across batch
    avg_kendall_tau = np.mean(kendall_taus) if kendall_taus else 0.0
    avg_spearman_rho = np.mean(spearman_rhos) if spearman_rhos else 0.0
    
    return {
        'kendall_tau': float(avg_kendall_tau),
        'spearman_rho': float(avg_spearman_rho)
    }


if __name__ == "__main__":
    # Test metrics
    batch_size = 4
    seq_len = 100
    
    # Create dummy data
    predicted = torch.randn(batch_size, seq_len)
    ground_truth = predicted + 0.1 * torch.randn(batch_size, seq_len)
    masks = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    # Add some padding
    masks[0, 80:] = False
    masks[1, 90:] = False
    
    results = evaluate_batch(predicted, ground_truth, masks)
    print(f"Kendall's tau: {results['kendall_tau']:.4f}")
    print(f"Spearman's rho: {results['spearman_rho']:.4f}")

