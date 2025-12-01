"""
Batch size finder: Test different batch sizes to find the maximum safe batch size
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import gc
import traceback

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.dataset import VideoSummarizationDataset, BatchCollator


def test_batch_size(model, dataset, collator, device, batch_size, num_workers=8, 
                    prefetch_factor=4, use_amp=False, num_test_batches=3):
    """
    Test if a batch size works without OOM or shared memory errors
    
    Args:
        model: PyTorch model
        dataset: Dataset to test with
        collator: Batch collator
        device: Device to run on
        batch_size: Batch size to test
        num_workers: Number of DataLoader workers
        prefetch_factor: Prefetch factor for DataLoader
        use_amp: Whether to use mixed precision
        num_test_batches: Number of batches to test (to ensure stability)
    
    Returns:
        tuple: (success: bool, memory_used_mb: float, error_message: str)
    """
    # Clear cache before test
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()
    
    try:
        # Create dataloader
        test_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=False,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=False
        )
        
        model.train()
        criterion = nn.MSELoss(reduction='none')
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        if use_amp:
            scaler = torch.cuda.amp.GradScaler()
        
        # Get initial memory
        if device.type == 'cuda':
            initial_memory = torch.cuda.memory_allocated(device) / 1024**2  # MB
        
        # Test a few batches
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= num_test_batches:
                break
            
            features = batch['features'].to(device)
            scores = batch['scores'].to(device)
            masks = batch['masks'].to(device)
            
            optimizer.zero_grad()
            
            if use_amp:
                with torch.cuda.amp.autocast():
                    pred_scores = model(features, masks)
                    # Apply mask
                    loss = criterion(pred_scores, scores) * masks
                    loss = loss.sum() / masks.sum()
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred_scores = model(features, masks)
                # Apply mask
                loss = criterion(pred_scores, scores) * masks
                loss = loss.sum() / masks.sum()
                
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
        
        # Clean up
        del test_loader, features, scores, masks, pred_scores, loss
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        
        return True, memory_used, ""
        
    except RuntimeError as e:
        error_msg = str(e)
        if "out of memory" in error_msg.lower() or "oom" in error_msg.lower():
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            return False, 0.0, f"OOM: {error_msg}"
        elif "bus error" in error_msg.lower() or "shared memory" in error_msg.lower():
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            return False, 0.0, f"Shared Memory Error: {error_msg}"
        else:
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            return False, 0.0, f"RuntimeError: {error_msg}"
    except Exception as e:
        error_msg = str(e)
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        return False, 0.0, f"Unexpected error: {error_msg}"


def find_optimal_batch_size(model, dataset, collator, device, start_batch_size=16,
                           max_batch_size=512, num_workers=8, prefetch_factor=4,
                           use_amp=False, strategy='binary_search'):
    """
    Find the optimal batch size that doesn't cause OOM or shared memory errors
    
    Args:
        model: PyTorch model
        dataset: Dataset to test with
        collator: Batch collator
        device: Device to run on
        start_batch_size: Starting batch size to test
        max_batch_size: Maximum batch size to test
        num_workers: Number of DataLoader workers
        prefetch_factor: Prefetch factor for DataLoader
        use_amp: Whether to use mixed precision
        strategy: 'linear' or 'binary_search'
    
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
    print(f"Strategy: {strategy}")
    print(f"{'='*60}\n")
    
    if strategy == 'linear':
        # Linear search: test each batch size incrementally
        current_batch_size = start_batch_size
        last_successful_batch_size = None
        batch_size_step = 8  # Increase by 8 each time
        
        results = []
        
        while current_batch_size <= max_batch_size:
            print(f"Testing batch size: {current_batch_size}...", end=" ", flush=True)
            success, memory_used, error_msg = test_batch_size(
                model, dataset, collator, device, current_batch_size,
                num_workers, prefetch_factor, use_amp
            )
            
            if success:
                print(f"✓ Success (Memory: {memory_used:.1f} MB)")
                last_successful_batch_size = current_batch_size
                results.append((current_batch_size, memory_used, True))
                current_batch_size += batch_size_step
            else:
                print(f"✗ Failed: {error_msg}")
                results.append((current_batch_size, 0.0, False))
                break
        
        if last_successful_batch_size is None:
            print(f"\n❌ No successful batch size found. Try reducing start_batch_size.")
            return start_batch_size // 2
        
        print(f"\n{'='*60}")
        print(f"Optimal batch size: {last_successful_batch_size}")
        print(f"{'='*60}\n")
        
        # Print summary
        print("Test Results Summary:")
        print(f"{'Batch Size':<12} {'Memory (MB)':<15} {'Status':<10}")
        print("-" * 40)
        for bs, mem, success in results:
            status = "✓ Success" if success else "✗ Failed"
            print(f"{bs:<12} {mem:<15.1f} {status:<10}")
        
        return last_successful_batch_size
    
    elif strategy == 'binary_search':
        # Binary search: faster but might miss some edge cases
        low = start_batch_size
        high = max_batch_size
        best_batch_size = start_batch_size
        results = []
        
        while low <= high:
            mid = (low + high) // 2
            # Round to nearest multiple of 8 for efficiency
            mid = (mid // 8) * 8
            if mid < start_batch_size:
                mid = start_batch_size
            
            print(f"Testing batch size: {mid}...", end=" ", flush=True)
            success, memory_used, error_msg = test_batch_size(
                model, dataset, collator, device, mid,
                num_workers, prefetch_factor, use_amp
            )
            
            if success:
                print(f"✓ Success (Memory: {memory_used:.1f} MB)")
                best_batch_size = mid
                results.append((mid, memory_used, True))
                low = mid + 8  # Try larger batch size
            else:
                print(f"✗ Failed: {error_msg}")
                results.append((mid, 0.0, False))
                high = mid - 8  # Try smaller batch size
        
        print(f"\n{'='*60}")
        print(f"Optimal batch size: {best_batch_size}")
        print(f"{'='*60}\n")
        
        # Print summary
        print("Test Results Summary:")
        print(f"{'Batch Size':<12} {'Memory (MB)':<15} {'Status':<10}")
        print("-" * 40)
        for bs, mem, success in results:
            status = "✓ Success" if success else "✗ Failed"
            print(f"{bs:<12} {mem:<15.1f} {status:<10}")
        
        return best_batch_size
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


if __name__ == '__main__':
    # This can be imported and used by training scripts
    pass

