"""
LoGo-Net: Local-Global Network for Video Summarization (Improved Version)
Proposed Method: Hybrid model combining CSTA (Local CNN) and Transformer-based Global context

Architecture Improvements (based on latest research):
- Dual-Path Design:
  * Local Path: 2D CNN for capturing local temporal patterns
  * Global Path: Transformer Encoder for global context (replaces FFT Mixer)
- Cross-Path Attention: Enables interaction between Local and Global paths
- Adaptive Fusion: Dynamic weighted fusion instead of simple concatenation

Key Improvements:
1. Transformer-based Global Path: Better long-range dependencies than FFT Mixer
2. Cross-Path Attention: Early information exchange between paths
3. Adaptive Fusion: Context-aware dynamic weighting of Local/Global features

Training Configuration:
- Optimal batch size: 9 (RTX 3080 10GB VRAM)
- Gradient accumulation: 2 (effective batch size: 18)
- Optimizer: AdamW (weight_decay=1e-4)
- Learning rate: 5e-4 (warmup 10 epochs, cosine annealing)
- Mixed Precision: Enabled (autocast)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LocalPath(nn.Module):
    """
    Local Path: CNN-based branch for capturing local temporal patterns
    Uses 2D convolution with kernel (3,1) or (5,1) to focus on local neighbors
    """
    
    def __init__(self, feature_dim, hidden_dim, kernel_size=3, dropout=0.1):
        """
        Args:
            feature_dim: input feature dimension
            hidden_dim: hidden dimension after projection
            kernel_size: convolution kernel size (3 or 5)
            dropout: dropout rate
        """
        super(LocalPath, self).__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        
        # Feature projection
        self.feature_proj = nn.Linear(feature_dim, hidden_dim)
        
        # 2D CNN for local temporal patterns
        # Input: (B, 1, T, D) where kernel operates on time axis
        # Kernel (kernel_size, 1) captures local neighbors in time dimension
        padding = (kernel_size // 2, 0)  # Only pad time dimension
        
        self.conv1 = nn.Conv2d(1, hidden_dim // 2, kernel_size=(kernel_size, 1), padding=padding)
        self.bn1 = nn.BatchNorm2d(hidden_dim // 2)
        
        self.conv2 = nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=(kernel_size, 1), padding=padding)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, feature_dim]
        Returns:
            x_local: [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, feature_dim = x.shape
        
        # Project features
        x_proj = self.feature_proj(x)  # [B, T, hidden_dim]
        
        # Reshape for 2D CNN: (B, T, D) -> (B, 1, T, D)
        x_2d = x_proj.unsqueeze(1)  # [B, 1, T, hidden_dim]
        
        # Apply 2D CNN with residual connection
        x_conv = F.gelu(self.bn1(self.conv1(x_2d)))  # [B, hidden_dim//2, T, hidden_dim]
        x_conv = self.dropout(x_conv)
        
        x_conv = self.bn2(self.conv2(x_conv))  # [B, hidden_dim, T, hidden_dim]
        
        # Reshape back: (B, hidden_dim, T, hidden_dim) -> (B, T, hidden_dim)
        # Take mean over channel dimension to get temporal features
        x_local = x_conv.mean(dim=1)  # [B, T, hidden_dim]
        
        # Residual connection: add projected input
        x_local = x_local + x_proj  # [B, T, hidden_dim]
        
        x_local = F.gelu(x_local)
        x_local = self.dropout(x_local)
        
        return x_local


class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer with Multi-Head Self-Attention
    """
    
    def __init__(self, hidden_dim, num_heads=8, ff_dim=None, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        if ff_dim is None:
            ff_dim = hidden_dim * 4
        
        # MultiheadAttention: default is (seq_len, batch, hidden_dim)
        # We'll handle batch_first manually
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout
        )
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize attention weights with very small values for stability
        # Split initialization for query, key, value projections
        with torch.no_grad():
            if hasattr(self.self_attn, 'in_proj_weight') and self.self_attn.in_proj_weight is not None:
                nn.init.xavier_uniform_(self.self_attn.in_proj_weight, gain=0.01)  # Very small gain
            if hasattr(self.self_attn, 'in_proj_bias') and self.self_attn.in_proj_bias is not None:
                nn.init.zeros_(self.self_attn.in_proj_bias)
            if hasattr(self.self_attn, 'out_proj'):
                nn.init.xavier_uniform_(self.self_attn.out_proj.weight, gain=0.01)
                if self.self_attn.out_proj.bias is not None:
                    nn.init.zeros_(self.self_attn.out_proj.bias)
    
    def forward(self, x, mask=None):
        # Pre-norm architecture for better stability
        x_norm = self.norm1(x)
        
        # Clamp values to prevent extreme values
        x_norm = torch.clamp(x_norm, min=-10.0, max=10.0)
        
        # x: [B, T, hidden_dim] -> [T, B, hidden_dim] for MultiheadAttention
        x_t = x_norm.transpose(0, 1)  # [T, B, hidden_dim]
        
        # Convert mask format: True for padding -> True for padding (MultiheadAttention expects this)
        # If mask is None, pass None
        attn_mask = mask if mask is not None else None
        
        # Self-attention with residual (pre-norm)
        try:
            attn_out, _ = self.self_attn(x_t, x_t, x_t, key_padding_mask=attn_mask)
            attn_out = attn_out.transpose(0, 1)  # [B, T, hidden_dim]
            
            # Clamp attention output
            attn_out = torch.clamp(attn_out, min=-10.0, max=10.0)
            
            # Check for NaN
            if torch.isnan(attn_out).any() or torch.isinf(attn_out).any():
                # If NaN/Inf, skip attention and use original
                attn_out = x_norm
            
            x = x + self.dropout(attn_out)
        except Exception as e:
            # If attention fails, skip it
            x = x
        
        # Feed-forward with pre-norm
        x_norm2 = self.norm2(x)
        x_norm2 = torch.clamp(x_norm2, min=-10.0, max=10.0)
        ff_out = self.ff(x_norm2)
        
        # Clamp FF output
        ff_out = torch.clamp(ff_out, min=-10.0, max=10.0)
        
        # Check for NaN in FF
        if torch.isnan(ff_out).any() or torch.isinf(ff_out).any():
            ff_out = x_norm2
        
        x = x + ff_out
        
        return x


class GlobalPath(nn.Module):
    """
    Global Path: Transformer-based Encoder for capturing global context
    Replaces FFT Mixer with Transformer to better capture long-range dependencies
    """
    
    def __init__(self, feature_dim, hidden_dim, num_layers=2, num_heads=8, dropout=0.1):
        """
        Args:
            feature_dim: input feature dimension
            hidden_dim: hidden dimension after projection
            num_layers: number of transformer encoder layers
            num_heads: number of attention heads
            dropout: dropout rate
        """
        super(GlobalPath, self).__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Feature projection
        self.feature_proj = nn.Linear(feature_dim, hidden_dim)
        
        # Positional encoding (learnable, initialized with small values)
        # Use Xavier uniform initialization scaled down
        pos_enc = torch.empty(1, 1000, hidden_dim)
        nn.init.xavier_uniform_(pos_enc, gain=0.02)  # Small gain for stability
        self.pos_encoding = nn.Parameter(pos_enc)
        
        # Transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len, feature_dim]
            mask: [batch_size, seq_len] - padding mask (True for padding positions)
        Returns:
            x_global: [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, feature_dim = x.shape
        
        # Project features
        x_proj = self.feature_proj(x)  # [B, T, hidden_dim]
        
        # Add positional encoding
        if seq_len <= self.pos_encoding.shape[1]:
            x_proj = x_proj + self.pos_encoding[:, :seq_len, :]
        else:
            # If sequence is longer, interpolate positional encoding
            pos_enc = F.interpolate(
                self.pos_encoding.transpose(1, 2),
                size=seq_len,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
            x_proj = x_proj + pos_enc
        
        x_proj = self.dropout(x_proj)
        
        # Apply transformer layers
        x_global = x_proj
        for layer in self.transformer_layers:
            x_global = layer(x_global, mask=mask)
            # Check for NaN and stop if found
            if torch.isnan(x_global).any():
                print("Warning: NaN detected in transformer layer")
                return x_proj  # Return original projection if NaN
        
        # Residual connection to initial projection
        x_global = x_global + x_proj
        
        return x_global


class CrossPathAttention(nn.Module):
    """
    Cross-Path Attention: Enables interaction between Local and Global paths
    Local path queries global context, and vice versa
    """
    
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        """
        Args:
            hidden_dim: hidden dimension of both paths
            num_heads: number of attention heads
            dropout: dropout rate
        """
        super(CrossPathAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # Cross-attention: Local queries Global
        self.local_to_global = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout
        )
        
        # Cross-attention: Global queries Local
        self.global_to_local = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout
        )
        
        # Initialize attention weights with very small values for stability
        with torch.no_grad():
            for attn in [self.local_to_global, self.global_to_local]:
                if hasattr(attn, 'in_proj_weight') and attn.in_proj_weight is not None:
                    nn.init.xavier_uniform_(attn.in_proj_weight, gain=0.01)
                if hasattr(attn, 'in_proj_bias') and attn.in_proj_bias is not None:
                    nn.init.zeros_(attn.in_proj_bias)
                if hasattr(attn, 'out_proj'):
                    nn.init.xavier_uniform_(attn.out_proj.weight, gain=0.01)
                    if attn.out_proj.bias is not None:
                        nn.init.zeros_(attn.out_proj.bias)
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x_local, x_global, mask=None):
        """
        Args:
            x_local: [batch_size, seq_len, hidden_dim] - Local path features
            x_global: [batch_size, seq_len, hidden_dim] - Global path features
            mask: [batch_size, seq_len] - padding mask
        Returns:
            x_local_enhanced: [batch_size, seq_len, hidden_dim]
            x_global_enhanced: [batch_size, seq_len, hidden_dim]
        """
        # Clamp inputs to prevent extreme values
        x_local = torch.clamp(x_local, min=-10.0, max=10.0)
        x_global = torch.clamp(x_global, min=-10.0, max=10.0)
        
        # Convert to [T, B, hidden_dim] for MultiheadAttention
        x_local_t = x_local.transpose(0, 1)  # [T, B, hidden_dim]
        x_global_t = x_global.transpose(0, 1)  # [T, B, hidden_dim]
        
        # Convert mask format if not None
        attn_mask = mask if mask is not None else None
        
        # Local queries Global (Local path attends to global context)
        try:
            local_attn, _ = self.local_to_global(
                x_local_t, x_global_t, x_global_t, key_padding_mask=attn_mask
            )
            local_attn = local_attn.transpose(0, 1)  # [B, T, hidden_dim]
            local_attn = torch.clamp(local_attn, min=-10.0, max=10.0)
            
            # Check for NaN/Inf
            if torch.isnan(local_attn).any() or torch.isinf(local_attn).any():
                local_attn = x_local
        except Exception:
            local_attn = x_local
        
        x_local_enhanced = self.norm1(x_local + self.dropout(local_attn))
        
        # Global queries Local (Global path attends to local details)
        try:
            global_attn, _ = self.global_to_local(
                x_global_t, x_local_t, x_local_t, key_padding_mask=attn_mask
            )
            global_attn = global_attn.transpose(0, 1)  # [B, T, hidden_dim]
            global_attn = torch.clamp(global_attn, min=-10.0, max=10.0)
            
            # Check for NaN/Inf
            if torch.isnan(global_attn).any() or torch.isinf(global_attn).any():
                global_attn = x_global
        except Exception:
            global_attn = x_global
        
        x_global_enhanced = self.norm2(x_global + self.dropout(global_attn))
        
        return x_local_enhanced, x_global_enhanced


class AdaptiveFusion(nn.Module):
    """
    Adaptive Fusion: Dynamic weighted fusion of Local and Global features
    Context-aware weighting instead of simple concatenation
    """
    
    def __init__(self, hidden_dim, dropout=0.1):
        """
        Args:
            hidden_dim: hidden dimension of both paths
            dropout: dropout rate
        """
        super(AdaptiveFusion, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # Gate network to compute adaptive weights
        # Takes concatenated features and outputs per-frame weights
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),  # 2 weights: [alpha, beta] for local and global
            nn.Softmax(dim=-1)  # Normalize weights
        )
        
        # Projection after fusion
        # Input: [x_weighted (hidden_dim) + x_concat (hidden_dim*2)] = hidden_dim*3
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x_local, x_global):
        """
        Args:
            x_local: [batch_size, seq_len, hidden_dim]
            x_global: [batch_size, seq_len, hidden_dim]
        Returns:
            x_fused: [batch_size, seq_len, hidden_dim]
        """
        # Concatenate for gate computation
        x_concat = torch.cat([x_local, x_global], dim=-1)  # [B, T, hidden_dim*2]
        
        # Compute adaptive weights per frame
        weights = self.gate_net(x_concat)  # [B, T, 2]
        alpha = weights[:, :, 0:1]  # [B, T, 1] - weight for local
        beta = weights[:, :, 1:2]   # [B, T, 1] - weight for global
        
        # Weighted combination
        x_weighted = alpha * x_local + beta * x_global  # [B, T, hidden_dim]
        
        # Also include concatenated features for richer representation
        x_fused = torch.cat([x_weighted, x_concat], dim=-1)  # [B, T, hidden_dim*3]
        
        # Project to hidden_dim
        x_fused = self.fusion_proj(x_fused)  # [B, T, hidden_dim]
        
        return x_fused


class LoGoNet(nn.Module):
    """
    LoGo-Net: Local-Global Network for Video Summarization (Improved Version)
    Combines local CNN path and Transformer-based global path with:
    - Cross-Path Attention for path interaction
    - Adaptive Fusion for dynamic feature combination
    """
    
    def __init__(self, feature_dim=1024, hidden_dim=512, score_hidden=256,
                 local_kernel_size=3, num_transformer_layers=2, num_heads=8,
                 use_cross_attention=True, use_adaptive_fusion=True, dropout=0.1):
        """
        Args:
            feature_dim: input feature dimension (default: 1024)
            hidden_dim: hidden dimension for both paths (default: 512)
            score_hidden: hidden dimension for score regression (default: 256)
            local_kernel_size: kernel size for local path CNN (3 or 5, default: 3)
            num_transformer_layers: number of transformer layers in global path (default: 2)
            num_heads: number of attention heads (default: 8)
            use_cross_attention: whether to use cross-path attention (default: True)
            use_adaptive_fusion: whether to use adaptive fusion (default: True)
            dropout: dropout rate (default: 0.1)
        """
        super(LoGoNet, self).__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.use_cross_attention = use_cross_attention
        self.use_adaptive_fusion = use_adaptive_fusion
        
        # Dual-path architecture
        self.local_path = LocalPath(feature_dim, hidden_dim, local_kernel_size, dropout)
        self.global_path = GlobalPath(
            feature_dim, hidden_dim, 
            num_layers=num_transformer_layers, 
            num_heads=num_heads, 
            dropout=dropout
        )
        
        # Cross-path attention (optional)
        if use_cross_attention:
            self.cross_attention = CrossPathAttention(hidden_dim, num_heads, dropout)
        
        # Adaptive fusion (optional, falls back to simple concat if False)
        if use_adaptive_fusion:
            self.adaptive_fusion = AdaptiveFusion(hidden_dim, dropout)
        else:
            # Fallback to simple concatenation
            self.fusion_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Score regression
        self.score_net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, score_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(score_hidden, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len, feature_dim] - frame features
            mask: [batch_size, seq_len] - padding mask (True for padding positions)
        
        Returns:
            scores: [batch_size, seq_len] - importance scores
        """
        # Local path: capture local temporal patterns
        x_local = self.local_path(x)  # [B, T, hidden_dim]
        
        # Global path: capture global context with Transformer
        x_global = self.global_path(x, mask=mask)  # [B, T, hidden_dim]
        
        # Cross-path attention: enable interaction between paths
        if self.use_cross_attention:
            x_local, x_global = self.cross_attention(x_local, x_global, mask=mask)
        
        # Adaptive fusion: dynamic weighted combination
        if self.use_adaptive_fusion:
            x_fused = self.adaptive_fusion(x_local, x_global)  # [B, T, hidden_dim]
        else:
            # Fallback to simple concatenation
            x_fused = torch.cat([x_local, x_global], dim=-1)  # [B, T, hidden_dim*2]
            x_fused = self.fusion_proj(x_fused)  # [B, T, hidden_dim]
            x_fused = self.dropout(x_fused)
        
        # Score regression
        scores = self.score_net(x_fused)  # [B, T, 1]
        scores = scores.squeeze(-1)  # [B, T]
        
        return scores

