"""
VideoSAGE: Video Summarization with Graph Representation Learning
CVPRW 2024

Architecture:
- Video frames as nodes in a graph
- Temporal adjacency for sparse graph construction
- Graph Convolutional Network (GCN) for feature learning
- Score regression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GraphConvolution(nn.Module):
    """
    Graph Convolution Layer
    """
    
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, input, adj):
        """
        Args:
            input: [batch_size, num_nodes, in_features]
            adj: [batch_size, num_nodes, num_nodes] - adjacency matrix
        """
        support = torch.matmul(input, self.weight)  # [B, N, out_features]
        output = torch.matmul(adj, support)  # [B, N, out_features]
        if self.bias is not None:
            output = output + self.bias
        return output


class VideoSAGE(nn.Module):
    """
    VideoSAGE Model: Graph-based video summarization
    """
    
    def __init__(self, feature_dim=1024, hidden_dim=512, num_layers=3, 
                 score_hidden=256, dropout=0.5):
        """
        Args:
            feature_dim: input feature dimension (default: 1024)
            hidden_dim: hidden dimension for GCN layers
            num_layers: number of GCN layers
            score_hidden: hidden dimension for score regression
            dropout: dropout rate
        """
        super(VideoSAGE, self).__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Feature projection
        self.feature_proj = nn.Linear(feature_dim, hidden_dim)
        
        # GCN layers
        self.gcn_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.gcn_layers.append(GraphConvolution(hidden_dim, hidden_dim))
            else:
                self.gcn_layers.append(GraphConvolution(hidden_dim, hidden_dim))
        
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
    
    def build_adjacency_matrix(self, seq_len, device):
        """
        Build temporal adjacency matrix for sparse graph
        Each node (frame) is connected to its temporal neighbors
        
        Args:
            seq_len: sequence length
            device: device to create tensor on
        
        Returns:
            adj: [seq_len, seq_len] - adjacency matrix
        """
        # Create temporal adjacency (connect adjacent frames)
        adj = torch.zeros(seq_len, seq_len, device=device)
        
        # Self-connections
        adj.fill_diagonal_(1.0)
        
        # Temporal neighbors (previous and next frame)
        for i in range(seq_len):
            if i > 0:
                adj[i, i-1] = 1.0
            if i < seq_len - 1:
                adj[i, i+1] = 1.0
        
        # Normalize adjacency matrix (symmetric normalization)
        # D^(-1/2) * A * D^(-1/2)
        degree = adj.sum(dim=1)
        degree_sqrt_inv = torch.pow(degree + 1e-8, -0.5)
        degree_sqrt_inv = torch.diag(degree_sqrt_inv)
        adj_normalized = torch.matmul(torch.matmul(degree_sqrt_inv, adj), degree_sqrt_inv)
        
        return adj_normalized
    
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
        x_proj = self.feature_proj(x)  # [B, T, hidden_dim]
        
        # Build adjacency matrix for each sequence
        # For efficiency, we can reuse the same adjacency matrix if all sequences have same length
        # Otherwise, build per sequence
        if mask is not None:
            # Handle variable length sequences
            adj_list = []
            for i in range(batch_size):
                valid_len = mask[i].sum().item()
                adj = self.build_adjacency_matrix(valid_len, x.device)
                # Pad to max length
                if valid_len < seq_len:
                    adj_padded = torch.eye(seq_len, device=x.device)
                    adj_padded[:valid_len, :valid_len] = adj
                    adj = adj_padded
                adj_list.append(adj)
            adj_batch = torch.stack(adj_list, dim=0)  # [B, T, T]
        else:
            # All sequences have same length
            adj = self.build_adjacency_matrix(seq_len, x.device)
            adj_batch = adj.unsqueeze(0).expand(batch_size, -1, -1)  # [B, T, T]
        
        # Apply GCN layers
        h = x_proj
        for i, gcn_layer in enumerate(self.gcn_layers):
            h = gcn_layer(h, adj_batch)
            if i < len(self.gcn_layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        
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
    model = VideoSAGE(feature_dim=1024, hidden_dim=512, num_layers=3, score_hidden=256)
    batch_size = 4
    seq_len = 200
    features = torch.randn(batch_size, seq_len, 1024)
    mask = torch.ones(batch_size, seq_len).bool()
    
    scores = model(features, mask)
    print(f"Input shape: {features.shape}")
    print(f"Output scores shape: {scores.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")


