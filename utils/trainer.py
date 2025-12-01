"""
Base trainer class with common training utilities
"""

import os
import time
import datetime
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import gc

from evaluation.metrics import evaluate_batch


class BaseTrainer:
    """Base trainer class for video summarization models"""
    
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, 
                 device, save_dir, log_dir, grad_clip=5.0, use_amp=False, wandb_run=None,
                 train_dataset=None, val_dataset=None, collator=None, num_workers=8, prefetch_factor=4,
                 lr_scheduler=None, warmup_epochs=5, early_stop_patience=15):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.grad_clip = grad_clip
        self.use_amp = use_amp
        self.wandb_run = wandb_run
        
        # For dynamic batch size adjustment
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.collator = collator
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.current_batch_size = train_loader.batch_size if train_loader else 32
        
        # LR scheduling
        self.lr_scheduler = lr_scheduler
        self.warmup_epochs = warmup_epochs
        self.base_lr = optimizer.param_groups[0]['lr'] if optimizer else 1e-4
        
        # Early stopping
        self.early_stop_patience = early_stop_patience
        self.early_stop_counter = 0
        self.best_metric_for_early_stop = -1.0
        
        # Initialize GradScaler for mixed precision training
        if use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # Create directories
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_kendall_tau': [],
            'val_spearman_rho': [],
            'learning_rate': []
        }
        
        # Best model tracking
        self.best_kendall_tau = -1.0
        self.best_spearman_rho = -1.0
        
    def _recreate_dataloaders(self, new_batch_size):
        """Recreate dataloaders with new batch size"""
        if self.train_dataset is None or self.collator is None:
            return False
        
        print(f"Recreating dataloaders with batch size: {new_batch_size}")
        self.current_batch_size = new_batch_size
        
        # Close old dataloaders
        if hasattr(self.train_loader, 'dataset'):
            del self.train_loader
        if hasattr(self.val_loader, 'dataset'):
            del self.val_loader
        
        # Clear CUDA cache
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
        
        # Create new dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=new_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collator,
            pin_memory=False,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=False
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=new_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collator,
            pin_memory=False,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=False
        )
        return True
    
    def _try_training_step(self, batch):
        """Try a training step, return (success, loss)"""
        try:
            features = batch['features'].to(self.device)
            gtscores = batch['gtscores'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    model_output = self.model(features, mask)
                    if isinstance(model_output, tuple):
                        scores = model_output[0]
                    else:
                        scores = model_output
                    loss = self.criterion(scores, gtscores)
                    if mask is not None:
                        loss = (loss * mask.float()).sum() / mask.float().sum()
                
                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                model_output = self.model(features, mask)
                if isinstance(model_output, tuple):
                    scores = model_output[0]
                else:
                    scores = model_output
                loss = self.criterion(scores, gtscores)
                if mask is not None:
                    loss = (loss * mask.float()).sum() / mask.float().sum()
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
            
            return True, loss.item()
        except RuntimeError as e:
            error_msg = str(e).lower()
            if 'out of memory' in error_msg or 'cuda out of memory' in error_msg:
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    gc.collect()
                return False, None
            elif 'shared memory' in error_msg or 'bus error' in error_msg:
                # Shared memory error - reduce batch size
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    gc.collect()
                return False, None
            else:
                raise
    
    def train_epoch(self, epoch):
        """Train for one epoch with automatic batch size adjustment"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        epoch_start_time = time.time()
        
        # Apply warmup LR
        if epoch < self.warmup_epochs:
            warmup_lr = self.base_lr * (epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = warmup_lr
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        
        for batch in pbar:
            success, loss_value = self._try_training_step(batch)
            
            if not success:
                # OOM occurred, reduce batch size
                new_batch_size = max(1, self.current_batch_size // 2)
                if new_batch_size < self.current_batch_size:
                    print(f"OOM detected! Reducing batch size from {self.current_batch_size} to {new_batch_size}")
                    if self._recreate_dataloaders(new_batch_size):
                        # Restart the epoch with new batch size
                        pbar.close()
                        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
                        if self.wandb_run is not None:
                            self.wandb_run.log({'training/batch_size': new_batch_size, 'epoch': epoch})
                        # Retry the same batch with new batch size
                        continue
                    else:
                        print("Failed to recreate dataloaders, skipping batch")
                        break
                else:
                    print("Batch size already at minimum, skipping batch")
                    break
            
            # Successfully processed batch
            total_loss += loss_value
            num_batches += 1
            pbar.set_postfix({'loss': loss_value, 'bs': self.current_batch_size})
        
        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.history['train_loss'].append(avg_loss)
        current_lr = self.optimizer.param_groups[0]['lr']
        self.history['learning_rate'].append(current_lr)
        
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
              f"Epoch {epoch} Train - Loss: {avg_loss:.4f}, LR: {current_lr:.2e}, "
              f"Batch Size: {self.current_batch_size}, Time: {epoch_time:.2f}s")
        return avg_loss
    
    def validate(self, epoch):
        """Validate on validation set"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        val_start_time = time.time()
        
        all_predicted = []
        all_ground_truth = []
        all_masks = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
            for batch in pbar:
                # Move to device
                features = batch['features'].to(self.device)
                gtscores = batch['gtscores'].to(self.device)
                mask = batch['mask'].to(self.device)
                
                # Forward pass (use autocast for validation too if AMP is enabled)
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        model_output = self.model(features, mask)
                        # Handle different model return formats
                        if isinstance(model_output, tuple):
                            scores = model_output[0]  # First element is always scores
                        else:
                            scores = model_output
                else:
                    model_output = self.model(features, mask)
                    # Handle different model return formats
                    if isinstance(model_output, tuple):
                        scores = model_output[0]  # First element is always scores
                    else:
                        scores = model_output
                
                # Compute loss
                loss = self.criterion(scores, gtscores)
                if mask is not None:
                    loss = (loss * mask.float()).sum() / mask.float().sum()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Collect predictions for evaluation - extract individual samples
                # Each batch may have different sequence lengths, so we need to extract
                # valid parts using the mask and store them separately
                batch_size = scores.shape[0]
                for i in range(batch_size):
                    # Get valid length for this sample
                    valid_length = mask[i].sum().item()
                    if valid_length > 0:
                        all_predicted.append(scores[i, :valid_length].cpu())
                        all_ground_truth.append(gtscores[i, :valid_length].cpu())
                        all_masks.append(mask[i, :valid_length].cpu())
        
        avg_loss = total_loss / num_batches
        self.history['val_loss'].append(avg_loss)
        
        # Evaluate metrics - now each item is a 1D tensor of variable length
        # We need to pad them to the same length for concatenation, or evaluate individually
        if len(all_predicted) > 0:
            # Get max length
            max_len = max(p.shape[0] for p in all_predicted)
            
            # Pad all tensors to max_len
            padded_predicted = []
            padded_ground_truth = []
            padded_masks = []
            
            for pred, gt, msk in zip(all_predicted, all_ground_truth, all_masks):
                pad_len = max_len - pred.shape[0]
                if pad_len > 0:
                    pred = torch.cat([pred, torch.zeros(pad_len)])
                    gt = torch.cat([gt, torch.zeros(pad_len)])
                    msk = torch.cat([msk, torch.zeros(pad_len, dtype=torch.bool)])
                padded_predicted.append(pred)
                padded_ground_truth.append(gt)
                padded_masks.append(msk)
            
            all_predicted = torch.stack(padded_predicted, dim=0)
            all_ground_truth = torch.stack(padded_ground_truth, dim=0)
            all_masks = torch.stack(padded_masks, dim=0)
        else:
            # Fallback if no predictions
            all_predicted = torch.empty(0)
            all_ground_truth = torch.empty(0)
            all_masks = torch.empty(0, dtype=torch.bool)
        
        eval_results = evaluate_batch(all_predicted, all_ground_truth, all_masks)
        
        kendall_tau = eval_results['kendall_tau']
        spearman_rho = eval_results['spearman_rho']
        
        self.history['val_kendall_tau'].append(kendall_tau)
        self.history['val_spearman_rho'].append(spearman_rho)
        
        # Use spearman_rho as the metric for LR scheduling and early stopping
        current_metric = spearman_rho
        
        # Update LR scheduler (only after warmup)
        if self.lr_scheduler is not None and epoch >= self.warmup_epochs:
            self.lr_scheduler.step(current_metric)
        
        # Update best models
        improved = False
        if kendall_tau > self.best_kendall_tau:
            self.best_kendall_tau = kendall_tau
            self.save_checkpoint(epoch, 'best_kendall_tau.pth')
            improved = True
        
        if spearman_rho > self.best_spearman_rho:
            self.best_spearman_rho = spearman_rho
            self.save_checkpoint(epoch, 'best_spearman_rho.pth')
            improved = True
        
        # Early stopping check
        if current_metric > self.best_metric_for_early_stop:
            self.best_metric_for_early_stop = current_metric
            self.early_stop_counter = 0
        else:
            self.early_stop_counter += 1
        
        val_time = time.time() - val_start_time
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
              f"Epoch {epoch} Val - Loss: {avg_loss:.4f}, "
              f"Kendall's τ: {kendall_tau:.4f}, Spearman's ρ: {spearman_rho:.4f}, "
              f"LR: {current_lr:.2e}, Time: {val_time:.2f}s")
        if self.early_stop_patience > 0:
            print(f"  Early stop counter: {self.early_stop_counter}/{self.early_stop_patience}")
        
        return avg_loss, kendall_tau, spearman_rho, improved
    
    def save_checkpoint(self, epoch, filename):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_kendall_tau': self.best_kendall_tau,
            'best_spearman_rho': self.best_spearman_rho,
            'current_batch_size': self.current_batch_size
        }
        if self.lr_scheduler is not None:
            checkpoint['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()
        save_path = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, save_path)
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        self.best_kendall_tau = checkpoint.get('best_kendall_tau', -1.0)
        self.best_spearman_rho = checkpoint.get('best_spearman_rho', -1.0)
        return checkpoint['epoch']
    
    def train(self, num_epochs, eval_every=1, save_every=5):
        """Main training loop with early stopping"""
        start_epoch = 0
        training_start_time = time.time()
        
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
              f"Starting training for {num_epochs} epochs")
        print(f"Best metrics will be saved to {self.save_dir}")
        print(f"Device: {self.device}")
        if self.warmup_epochs > 0:
            print(f"Warmup epochs: {self.warmup_epochs}")
        if self.lr_scheduler is not None:
            print(f"LR Scheduler: {type(self.lr_scheduler).__name__}")
        if self.early_stop_patience > 0:
            print(f"Early stopping patience: {self.early_stop_patience}")
        print("-" * 80)
        
        for epoch in range(start_epoch, num_epochs):
            epoch_start = time.time()
            # Train
            train_loss = self.train_epoch(epoch)
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.wandb_run is not None:
                self.wandb_run.log({
                    'train/loss': train_loss,
                    'learning_rate': current_lr,
                    'training/batch_size': self.current_batch_size,
                    'epoch': epoch + 1
                })
            
            # Validate
            if (epoch + 1) % eval_every == 0:
                val_loss, kendall_tau, spearman_rho, improved = self.validate(epoch)
                print(f"  Best Kendall's τ: {self.best_kendall_tau:.4f}")
                print(f"  Best Spearman's ρ: {self.best_spearman_rho:.4f}")
                if self.wandb_run is not None:
                    self.wandb_run.log({
                        'val/loss': val_loss,
                        'val/kendall_tau': kendall_tau,
                        'val/spearman_rho': spearman_rho,
                        'best/kendall_tau': self.best_kendall_tau,
                        'best/spearman_rho': self.best_spearman_rho,
                        'epoch': epoch + 1
                    })
                
                # Early stopping check
                if self.early_stop_patience > 0 and self.early_stop_counter >= self.early_stop_patience:
                    print(f"\nEarly stopping triggered! No improvement for {self.early_stop_patience} epochs.")
                    print(f"Best Spearman's ρ: {self.best_metric_for_early_stop:.4f}")
                    break
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(epoch, f'checkpoint_epoch_{epoch+1}.pth')
            
            epoch_time = time.time() - epoch_start
            elapsed_time = time.time() - training_start_time
            avg_time_per_epoch = elapsed_time / (epoch + 1)
            remaining_epochs = num_epochs - (epoch + 1)
            estimated_remaining = avg_time_per_epoch * remaining_epochs
            
            print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                  f"Epoch {epoch+1}/{num_epochs} completed - "
                  f"Epoch time: {epoch_time:.2f}s, "
                  f"Elapsed: {elapsed_time/3600:.2f}h, "
                  f"Est. remaining: {estimated_remaining/3600:.2f}h")
            print("-" * 80)
        
        # Save final checkpoint
        total_time = time.time() - training_start_time
        final_epoch = epoch
        self.save_checkpoint(final_epoch, 'final_checkpoint.pth')
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
              f"Training completed! Total time: {total_time/3600:.2f} hours")
        print(f"Final epoch: {final_epoch + 1}")
        print(f"Best Kendall's τ: {self.best_kendall_tau:.4f}")
        print(f"Best Spearman's ρ: {self.best_spearman_rho:.4f}")
        if self.wandb_run is not None:
            self.wandb_run.log({
                'training/total_time_hours': total_time / 3600.0,
                'final/best_kendall_tau': self.best_kendall_tau,
                'final/best_spearman_rho': self.best_spearman_rho,
                'final/epoch': final_epoch + 1,
                'epoch': final_epoch + 1
            })
            self.wandb_run.summary['best_kendall_tau'] = self.best_kendall_tau
            self.wandb_run.summary['best_spearman_rho'] = self.best_spearman_rho
            self.wandb_run.summary['final_epoch'] = final_epoch + 1

