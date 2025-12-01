"""
CSTA: CNN-based Spatiotemporal Attention for Video Summarization
CVPR 2024

Architecture:
- 2D CNN on reshaped video features (B, T, D) -> (B, 1, T, D)
- Attention map generation
- Input and attention mixing
- Sigmoid output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CSTA(nn.Module):
    """
    CSTA Model: CNN-based Spatiotemporal Attention
    """
    
    def __init__(self, feature_dim=1024, spatial_hidden=512, temporal_hidden=512, 
                 score_hidden=256, dropout=0.5):
        """
        Args:
            feature_dim: input feature dimension (default: 1024)
            spatial_hidden: hidden dimension for spatial processing
            temporal_hidden: hidden dimension for temporal processing
            score_hidden: hidden dimension for score regression
            dropout: dropout rate
        """
        super(CSTA, self).__init__()
        
        self.feature_dim = feature_dim
        self.spatial_hidden = spatial_hidden
        self.temporal_hidden = temporal_hidden
        
        # Feature projection
        self.feature_proj = nn.Linear(feature_dim, spatial_hidden)
        
        # 2D CNN layers for spatiotemporal modeling
        # Input: (B, 1, T, D) where T is temporal, D is feature dimension
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, temporal_hidden, kernel_size=(3, 3), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(temporal_hidden)
        
        # Attention mechanism
        self.attention_conv = nn.Conv2d(temporal_hidden, 1, kernel_size=(1, 1))
        
        # Score regression network
        self.score_net = nn.Sequential(
            nn.Linear(temporal_hidden, score_hidden),
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
        x_proj = self.feature_proj(x)  # [B, T, spatial_hidden]
        
        # Reshape for 2D CNN: (B, T, D) -> (B, 1, T, D)
        x_2d = x_proj.unsqueeze(1)  # [B, 1, T, spatial_hidden]
        
        # Apply 2D CNN
        x_conv = F.relu(self.bn1(self.conv1(x_2d)))
        x_conv = self.dropout(x_conv)
        x_conv = F.relu(self.bn2(self.conv2(x_conv)))
        x_conv = self.dropout(x_conv)
        x_conv = F.relu(self.bn3(self.conv3(x_conv)))
        x_conv = self.dropout(x_conv)  # [B, temporal_hidden, T, spatial_hidden]
        
        # Generate attention map
        attention_map = self.attention_conv(x_conv)  # [B, 1, T, spatial_hidden]
        attention_map = attention_map.squeeze(1)  # [B, T, spatial_hidden]
        # Average over spatial dimension to get temporal attention
        attention_temporal = attention_map.mean(dim=-1)  # [B, T]
        attention_weights = F.softmax(attention_temporal, dim=1).unsqueeze(-1)  # [B, T, 1]
        
        # Extract temporal features from CNN output
        # Take mean over spatial dimension from x_conv
        x_conv_temporal = x_conv.mean(dim=-1)  # [B, temporal_hidden, T]
        x_conv_temporal = x_conv_temporal.permute(0, 2, 1)  # [B, T, temporal_hidden]
        
        # Apply attention to temporal features
        attended_temporal = x_conv_temporal * attention_weights
        
        # Score regression
        scores = self.score_net(attended_temporal).squeeze(-1)  # [B, T]
        
        # Apply mask if provided
        if mask is not None:
            scores = scores * mask.float()
        
        return scores


if __name__ == "__main__":
    # Test model
    model = CSTA(feature_dim=1024, spatial_hidden=512, temporal_hidden=512, score_hidden=256)
    batch_size = 4
    seq_len = 200
    features = torch.randn(batch_size, seq_len, 1024)
    mask = torch.ones(batch_size, seq_len).bool()
    
    scores = model(features, mask)
    print(f"Input shape: {features.shape}")
    print(f"Output scores shape: {scores.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

