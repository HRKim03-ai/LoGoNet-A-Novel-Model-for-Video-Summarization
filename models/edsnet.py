"""
EDSNet: Efficient-DSNet for Video Summarization
arXiv 2024

Architecture:
- Token Mixer using MLP-Mixer or Fourier Transform
- Lightweight alternative to heavy attention mechanisms
- Score regression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPMixerBlock(nn.Module):
    """
    MLP-Mixer Block: Token mixing and channel mixing
    Uses 1D convolution for token mixing to handle variable length sequences
    """
    
    def __init__(self, feature_dim, token_hidden, channel_hidden, dropout=0.1):
        """
        Args:
            feature_dim: feature dimension (for channel mixing)
            token_hidden: hidden dimension for token mixing MLP
            channel_hidden: hidden dimension for channel mixing MLP
            dropout: dropout rate
        """
        super(MLPMixerBlock, self).__init__()
        
        # Token mixing using 1D convolution (handles variable length)
        # Note: LayerNorm is applied before transpose, Conv1d operates on [B, C, L]
        self.token_mix_conv = nn.Sequential(
            nn.Conv1d(feature_dim, token_hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(token_hidden, feature_dim, kernel_size=3, padding=1),
            nn.Dropout(dropout)
        )
        
        # Channel mixing (mixing across feature dimension)
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, channel_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channel_hidden, feature_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, feature_dim]
        Returns:
            x: [batch_size, seq_len, feature_dim]
        """
        # Token mixing: apply LayerNorm first, then transpose for Conv1d
        x_norm = F.layer_norm(x, x.shape[-1:])  # [B, seq_len, feature_dim]
        x_transposed = x_norm.transpose(1, 2)  # [B, feature_dim, seq_len]
        x_token = self.token_mix_conv(x_transposed)  # [B, feature_dim, seq_len]
        x_token = x_token.transpose(1, 2)  # [B, seq_len, feature_dim]
        x = x + x_token  # Residual connection
        
        # Channel mixing
        x_channel = self.channel_mix(x)  # [B, seq_len, feature_dim]
        x = x + x_channel  # Residual connection
        
        return x


class EDSNet(nn.Module):
    """
    EDSNet Model: Efficient video summarization using Token Mixer
    """
    
    def __init__(self, feature_dim=1024, hidden_dim=512, num_mixer_blocks=4,
                 token_hidden=256, channel_hidden=2048, score_hidden=256, dropout=0.1):
        """
        Args:
            feature_dim: input feature dimension (default: 1024)
            hidden_dim: hidden dimension after projection
            num_mixer_blocks: number of MLP-Mixer blocks
            token_hidden: hidden dimension for token mixing
            channel_hidden: hidden dimension for channel mixing
            score_hidden: hidden dimension for score regression
            dropout: dropout rate
        """
        super(EDSNet, self).__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_mixer_blocks = num_mixer_blocks
        
        # Feature projection
        self.feature_proj = nn.Linear(feature_dim, hidden_dim)
        
        # MLP-Mixer blocks (using 1D conv for variable length support)
        self.mixer_blocks = nn.ModuleList()
        for _ in range(num_mixer_blocks):
            self.mixer_blocks.append(
                MLPMixerBlock(
                    hidden_dim,
                    token_hidden,
                    channel_hidden,
                    dropout
                )
            )
        
        # For fixed sequence length, we can pre-create blocks
        # For variable length, we'll use a workaround: use max_seq_len
        # In practice, we'll create blocks dynamically or use a large enough max_seq_len
        
        # Score regression network
        self.score_net = nn.Sequential(
            nn.Linear(hidden_dim, score_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(score_hidden, score_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(score_hidden // 2, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len, feature_dim] - frame features
            mask: [batch_size, seq_len] - padding mask (optional)
        
        Returns:
            scores: [batch_size, seq_len] - importance scores
        """
        batch_size, seq_len, feature_dim = x.shape
        
        # Project features
        h = self.feature_proj(x)  # [B, T, hidden_dim]
        
        # Apply mixer blocks (now supports variable length)
        for mixer_block in self.mixer_blocks:
            h = mixer_block(h)
        
        # Apply mask if provided
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            h = h * mask_expanded
        
        # Score regression
        scores = self.score_net(h).squeeze(-1)  # [B, T]
        
        # Apply mask to scores
        if mask is not None:
            scores = scores * mask.float()
        
        return scores


if __name__ == "__main__":
    # Test model
    model = EDSNet(feature_dim=1024, hidden_dim=512, num_mixer_blocks=4, 
                   token_hidden=256, channel_hidden=2048, score_hidden=256)
    batch_size = 4
    seq_len = 200
    features = torch.randn(batch_size, seq_len, 1024)
    mask = torch.ones(batch_size, seq_len).bool()
    
    scores = model(features, mask)
    print(f"Input shape: {features.shape}")
    print(f"Output scores shape: {scores.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

