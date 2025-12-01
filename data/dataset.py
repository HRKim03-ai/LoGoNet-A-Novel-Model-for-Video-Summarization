"""
Dataset loader for MR.HiSum with visual-only features
"""

import h5py
import numpy as np
import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class VideoSummarizationDataset(Dataset):
    """
    Dataset for video summarization with visual-only features
    
    Returns:
        - features: [seq_len, feature_dim] - frame features (visual-only)
        - gtscore: [seq_len] - ground truth importance scores
        - video_id: str - video identifier
    """
    
    def __init__(self, mode='train', dataset_path='dataset/mr_hisum.h5', 
                 split_file='dataset/mr_hisum_split.json'):
        """
        Args:
            mode: 'train', 'val', or 'test'
            dataset_path: path to mr_hisum.h5 file
            split_file: path to split JSON file
        """
        self.mode = mode
        self.dataset_path = dataset_path
        self.split_file = split_file
        
        # Don't open HDF5 file here - open it in __getitem__ for multiprocessing compatibility
        # This allows each worker process to open its own file handle
        
        # Load split information
        with open(self.split_file, 'r') as f:
            self.split_data = json.load(f)
        
        # Get video keys for this mode
        self.video_keys = self.split_data.get(f'{mode}_keys', [])
        
        # Cache for file handle (per-process)
        self._file_handle = None
    
    def _get_file_handle(self):
        """Get or create file handle (thread-safe for multiprocessing)"""
        # Use threading lock for thread safety
        import threading
        if not hasattr(self, '_lock'):
            self._lock = threading.Lock()
        
        with self._lock:
            if self._file_handle is None:
                # Open HDF5 file with libver='latest' for better multiprocessing support
                self._file_handle = h5py.File(self.dataset_path, 'r', libver='latest')
        return self._file_handle
    
    def __len__(self):
        return len(self.video_keys)
    
    def __getitem__(self, index):
        """
        Returns:
            dict with:
                - video_id: str
                - features: torch.Tensor [seq_len, feature_dim]
                - gtscore: torch.Tensor [seq_len]
                - change_points: np.ndarray (only for val/test)
                - gt_summary: np.ndarray (only for val/test)
        """
        video_id = self.video_keys[index]
        
        # Get file handle (opens file if needed)
        video_data = self._get_file_handle()
        
        # Load features (visual-only, no text)
        features = torch.Tensor(np.array(video_data[f'{video_id}/features']))
        
        # Load ground truth scores
        gtscore = torch.Tensor(np.array(video_data[f'{video_id}/gtscore']))
        
        result = {
            'video_id': video_id,
            'features': features,
            'gtscore': gtscore
        }
        
        # Add additional info for validation/test
        if self.mode in ['val', 'test']:
            result['change_points'] = np.array(video_data[f'{video_id}/change_points'])
            result['gt_summary'] = np.array(video_data[f'{video_id}/gt_summary'])
            result['n_frames'] = features.shape[0]
        
        return result
    
    def __del__(self):
        """Close HDF5 file when dataset is deleted"""
        if hasattr(self, '_file_handle') and self._file_handle is not None:
            try:
                self._file_handle.close()
            except:
                pass


class BatchCollator:
    """
    Collate function for batching variable-length sequences
    """
    
    def __init__(self, include_metadata=False):
        """
        Args:
            include_metadata: whether to include change_points, gt_summary, etc.
        """
        self.include_metadata = include_metadata
    
    def __call__(self, batch):
        """
        Args:
            batch: list of samples from dataset
        Returns:
            dict with batched tensors
        """
        video_ids = []
        features = []
        gtscores = []
        
        if self.include_metadata:
            change_points = []
            gt_summaries = []
            n_frames = []
        
        for item in batch:
            video_ids.append(item['video_id'])
            features.append(item['features'])
            gtscores.append(item['gtscore'])
            
            if self.include_metadata:
                if 'change_points' in item:
                    change_points.append(item['change_points'])
                if 'gt_summary' in item:
                    gt_summaries.append(item['gt_summary'])
                if 'n_frames' in item:
                    n_frames.append(item['n_frames'])
        
        # Pad sequences
        lengths = torch.LongTensor([f.shape[0] for f in features])
        max_len = lengths.max().item()
        
        # Create mask (True for valid positions, False for padding)
        mask = torch.arange(max_len)[None, :] < lengths[:, None]
        
        # Pad features and gtscores
        features_padded = pad_sequence(features, batch_first=True)
        gtscores_padded = pad_sequence(gtscores, batch_first=True)
        
        result = {
            'video_ids': video_ids,
            'features': features_padded,
            'gtscores': gtscores_padded,
            'mask': mask,
            'lengths': lengths
        }
        
        if self.include_metadata:
            result['change_points'] = change_points if change_points else None
            result['gt_summaries'] = gt_summaries if gt_summaries else None
            result['n_frames'] = n_frames if n_frames else None
        
        return result


if __name__ == "__main__":
    # Test dataset
    dataset = VideoSummarizationDataset(mode='train')
    print(f"Dataset size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"Video ID: {sample['video_id']}")
    print(f"Features shape: {sample['features'].shape}")
    print(f"GTScore shape: {sample['gtscore'].shape}")
    
    # Test collator
    collator = BatchCollator()
    batch = [dataset[i] for i in range(4)]
    batched = collator(batch)
    
    print(f"\nBatched features shape: {batched['features'].shape}")
    print(f"Batched mask shape: {batched['mask'].shape}")
    print(f"Batched lengths: {batched['lengths']}")

