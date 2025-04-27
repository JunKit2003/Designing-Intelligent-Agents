# model.py

import math
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import HeteroData, Batch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import networkx as nx

class TemporalGATTransformer(nn.Module):
    """
    Hierarchical GAT-Transformer model for spatio-temporal traffic signal control
    Combines Graph Attention Networks with Transformer temporal encoding
    """
    
    def __init__(self, 
                 in_features: int,
                 hidden_dim: int = 128,
                 n_heads: int = 8,
                 transformer_layers: int = 2,
                 history_length: int = 5,
                 gat_layers: int = 2):
        super().__init__()
        
        self.history_length = history_length
        self.hidden_dim = hidden_dim
        
        # Spatial Encoder (GAT)
        self.gat_convs = nn.ModuleList()
        for i in range(gat_layers):
            in_dim = in_features if i == 0 else hidden_dim
            self.gat_convs.append(GATConv(in_dim, hidden_dim, heads=n_heads, concat=False))
            
        # Temporal Encoder (Transformer)
        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim*4,
                batch_first=True
            ),
            num_layers=transformer_layers
        )
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # Policy and Value Heads
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output logits for each phase
        )
        
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output value estimate
        )
        
        # History buffer
        self.history_buffer = None
        
    def forward(self, data: HeteroData) -> tuple:
        """
        Process hierarchical graph with temporal information
        Returns:
            logits: Action logits for each phase
            value: State value estimate
        """
        # Encode spatial features
        spatial_feats = self.encode_spatial(data)
        
        # Update history buffer
        self.update_history(spatial_feats)
        
        # Encode temporal features
        temporal_feats = self.encode_temporal()
        
        # Combine spatial and temporal features
        combined_feats = torch.cat([spatial_feats, temporal_feats], dim=-1)
        
        # Get policy logits and value estimate
        logits = self.actor_head(combined_feats)
        value = self.critic_head(combined_feats)
        
        return logits.squeeze(), value.squeeze()
    
    def encode_spatial(self, data) -> torch.Tensor:
        """Process hierarchical graph structure using GAT"""
        # Handle both NetworkX graph and HeteroData
        if isinstance(data, nx.Graph):
            # Extract segment nodes and features
            segment_nodes = [n for n, attrs in data.nodes(data=True) 
                            if attrs.get('type') == 'segment']
            intersection_nodes = [n for n, attrs in data.nodes(data=True) 
                                if attrs.get('type') == 'intersection']
            
            # Get segment features
            segment_features = torch.stack([
                torch.tensor(data.nodes[node]['features'], dtype=torch.float32)
                for node in segment_nodes
            ]) if segment_nodes else torch.zeros((0, self.hidden_dim))
            
            # Create edge indices for segments
            edges = []
            for u, v, attrs in data.edges(data=True):
                if (data.nodes[u].get('type') == 'segment' and 
                    data.nodes[v].get('type') == 'segment'):
                    u_idx = segment_nodes.index(u)
                    v_idx = segment_nodes.index(v)
                    edges.append([u_idx, v_idx])
            
            edge_index = torch.tensor(edges, dtype=torch.long).t() if edges else torch.zeros((2, 0), dtype=torch.long)
            
            # Process through GAT layers
            x = segment_features
            for conv in self.gat_convs:
                if x.shape[0] > 0 and edge_index.shape[1] > 0:
                    x = F.relu(conv(x, edge_index))
                    x = F.dropout(x, p=0.1, training=self.training)
            
            # Aggregate to intersection level
            intersection_feats = []
            for tl_id in intersection_nodes:
                # Find segments connected to this intersection
                connected_segments = []
                for i, seg_node in enumerate(segment_nodes):
                    if data.has_edge(seg_node, tl_id) or data.has_edge(tl_id, seg_node):
                        connected_segments.append(i)
                
                if connected_segments and x.shape[0] > 0:
                    intersection_feats.append(x[connected_segments].mean(dim=0))
                else:
                    intersection_feats.append(torch.zeros(self.hidden_dim, dtype=torch.float32))
            
            return torch.stack(intersection_feats) if intersection_feats else torch.zeros((len(intersection_nodes), self.hidden_dim))
        else:
            # Extract segment features
            x = data['segment'].features
            edge_index = data['segment', 'to_down', 'movement'].edge_index
            
            # Process through GAT layers
            for conv in self.gat_convs:
                x = F.relu(conv(x, edge_index))
                x = F.dropout(x, p=0.1, training=self.training)
                
            # Aggregate to intersection level
            intersection_feats = []
            for tl_id in data['intersection'].node_indices:
                segments = data.segment_nodes_for_intersection(tl_id)  # Implement based on your graph structure
                intersection_feats.append(x[segments].mean(dim=0))
                
            return torch.stack(intersection_feats)
    
    def encode_temporal(self) -> torch.Tensor:
        """Process temporal sequence using Transformer"""
        if self.history_buffer is None or len(self.history_buffer) < self.history_length:
            return torch.zeros_like(self.history_buffer[0])
            
        # Add positional encoding
        encoded = self.pos_encoder(self.history_buffer)
        
        # Transformer expects [batch, seq, features]
        temporal_out = self.transformer_encoder(encoded.unsqueeze(0))[0]
        
        # Use last timestep output
        return temporal_out[-1]
    
    def update_history(self, current_feats: torch.Tensor):
        """Maintain rolling buffer of historical features"""
        if self.history_buffer is None:
            self.history_buffer = current_feats.unsqueeze(0)
        else:
            self.history_buffer = torch.cat([
                self.history_buffer[1:],  # Remove oldest entry
                current_feats.unsqueeze(0)
            ])
            
    def reset_history(self):
        """Reset temporal memory between episodes"""
        self.history_buffer = None

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences"""
    
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return x

class PPOWrapper(nn.Module):
    """
    PPO-specific wrapper that handles action distribution and value estimation
    """
    
    def __init__(self, model: TemporalGATTransformer):
        super().__init__()
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def forward(self, 
                states: List[HeteroData],
                actions: torch.Tensor = None) -> tuple:
        # Process batch of states
        logits = []
        values = []
        
        for state in states:
            l, v = self.model(state)
            logits.append(l)
            values.append(v)
            
        logits = torch.stack(logits)
        values = torch.stack(values)
        
        # Create action distribution
        dist = torch.distributions.Categorical(logits=logits)
        
        if actions is None:
            actions = dist.sample()
            
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return actions, log_probs, entropy, values
    
    def get_value(self, states: List[HeteroData]) -> torch.Tensor:
        with torch.no_grad():
            _, values = self.model(Batch.from_data_list(states))
        return values
    
    def reset_history(self):
        self.model.reset_history()

