"""
Phase 4: ST-GNN Model Architecture

This module implements the HydroGraph Spatiotemporal Graph Neural Network,
combining GraphSAGE for spatial message passing and GRU for temporal modeling.
"""

from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from loguru import logger


class SpatialModule(nn.Module):
    """
    Spatial module using GraphSAGE for message passing.
    
    Implements multi-layer GraphSAGE convolutions with mean aggregation
    to propagate flood risk spatially through the urban topology.
    
    Parameters
    ----------
    in_channels : int
        Number of input features
    hidden_channels : int
        Number of hidden units per layer
    num_layers : int
        Number of GraphSAGE layers
    dropout : float
        Dropout rate for regularization
    aggregator : str
        Aggregation function ('mean', 'max', 'sum')
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 3,
        dropout: float = 0.3,
        aggregator: str = 'mean',
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Build GraphSAGE layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(
            SAGEConv(in_channels, hidden_channels, aggr=aggregator)
        )
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(
                SAGEConv(hidden_channels, hidden_channels, aggr=aggregator)
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        logger.info(
            f"SpatialModule: {num_layers} layers, "
            f"{hidden_channels} hidden channels, "
            f"aggregator={aggregator}"
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through spatial layers.
        
        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix (num_nodes, in_channels)
        edge_index : torch.Tensor
            Edge indices (2, num_edges)
            
        Returns
        -------
        torch.Tensor
            Spatially aggregated node embeddings (num_nodes, hidden_channels)
        """
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


class TemporalModule(nn.Module):
    """
    Temporal module using GRU for sequential modeling.
    
    Processes the lag window of rainfall data to capture temporal dependencies
    and generate a temporal hidden state.
    
    Parameters
    ----------
    input_size : int
        Size of temporal input features (lag_window)
    hidden_size : int
        Size of GRU hidden state
    num_layers : int
        Number of GRU layers
    dropout : float
        Dropout rate between GRU layers
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU for temporal modeling
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        
        logger.info(
            f"TemporalModule: GRU with {num_layers} layers, "
            f"hidden_size={hidden_size}"
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through temporal module.
        
        Parameters
        ----------
        x : torch.Tensor
            Temporal feature sequence (batch_size, sequence_length, input_size)
            
        Returns
        -------
        torch.Tensor
            Final hidden state (batch_size, hidden_size)
        """
        # GRU forward pass
        # output: (batch_size, sequence_length, hidden_size)
        # h_n: (num_layers, batch_size, hidden_size)
        output, h_n = self.gru(x)
        
        # Return final hidden state from last layer
        # h_n[-1]: (batch_size, hidden_size)
        return h_n[-1]


class FusionModule(nn.Module):
    """
    Fusion module to combine static, temporal, and spatial embeddings.
    
    Implements a multi-layer perceptron to fuse different feature modalities.
    
    Parameters
    ----------
    static_dim : int
        Dimension of static features
    temporal_dim : int
        Dimension of temporal embeddings
    spatial_dim : int
        Dimension of spatial embeddings
    hidden_dims : List[int]
        Hidden layer dimensions
    dropout : float
        Dropout rate
    """
    
    def __init__(
        self,
        static_dim: int,
        temporal_dim: int,
        spatial_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.3,
    ):
        super().__init__()
        
        # Input dimension is concatenation of all modalities
        input_dim = static_dim + temporal_dim + spatial_dim
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]
        
        logger.info(
            f"FusionModule: input_dim={input_dim}, "
            f"hidden_dims={hidden_dims}"
        )
    
    def forward(
        self,
        static_features: torch.Tensor,
        temporal_embedding: torch.Tensor,
        spatial_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through fusion module.
        
        Parameters
        ----------
        static_features : torch.Tensor
            Static node features (num_nodes, static_dim)
        temporal_embedding : torch.Tensor
            Temporal embeddings (num_nodes, temporal_dim)
        spatial_embedding : torch.Tensor
            Spatial embeddings (num_nodes, spatial_dim)
            
        Returns
        -------
        torch.Tensor
            Fused embeddings (num_nodes, output_dim)
        """
        # Concatenate all modalities
        x = torch.cat([static_features, temporal_embedding, spatial_embedding], dim=1)
        
        # Pass through MLP
        x = self.mlp(x)
        
        return x


class HydroGraphSTGNN(nn.Module):
    """
    HydroGraph Spatiotemporal Graph Neural Network.
    
    Main model architecture combining spatial (GraphSAGE), temporal (GRU),
    and fusion modules for urban flood forecasting.
    
    Architecture:
    1. Temporal module processes lag window through GRU
    2. Static features are passed directly
    3. Concatenated features go through spatial GraphSAGE layers
    4. Fusion module combines all modalities
    5. Output head predicts flood probability
    
    Parameters
    ----------
    num_static_features : int
        Number of static node features
    lag_window : int
        Temporal lag window size
    spatial_config : dict
        Configuration for spatial module
    temporal_config : dict
        Configuration for temporal module
    fusion_config : dict
        Configuration for fusion module
    """
    
    def __init__(
        self,
        num_static_features: int,
        lag_window: int,
        spatial_config: dict,
        temporal_config: dict,
        fusion_config: dict,
    ):
        super().__init__()
        
        self.num_static_features = num_static_features
        self.lag_window = lag_window
        
        logger.info("=" * 80)
        logger.info("PHASE 4: ST-GNN MODEL ARCHITECTURE")
        logger.info("=" * 80)
        
        # Temporal module (processes lag window)
        self.temporal_module = TemporalModule(
            input_size=1,  # Single rainfall value per timestep
            hidden_size=temporal_config['hidden_size'],
            num_layers=temporal_config['num_layers'],
            dropout=temporal_config['dropout'],
        )
        
        # Spatial module (GraphSAGE)
        # Input: static features + temporal embedding
        spatial_input_dim = num_static_features + temporal_config['hidden_size']
        
        self.spatial_module = SpatialModule(
            in_channels=spatial_input_dim,
            hidden_channels=spatial_config['hidden_channels'],
            num_layers=spatial_config['num_layers'],
            dropout=spatial_config['dropout'],
            aggregator=spatial_config['aggregator'],
        )
        
        # Fusion module
        self.fusion_module = FusionModule(
            static_dim=num_static_features,
            temporal_dim=temporal_config['hidden_size'],
            spatial_dim=spatial_config['hidden_channels'],
            hidden_dims=fusion_config['hidden_dims'],
            dropout=fusion_config.get('dropout', 0.3),
        )
        
        # Output head (flood probability)
        self.output_head = nn.Sequential(
            nn.Linear(self.fusion_module.output_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid(),  # Output probability [0, 1]
        )
        
        logger.success("=" * 80)
        logger.success("PHASE 4 COMPLETE: Model architecture created")
        logger.success(f"Total parameters: {self.count_parameters():,}")
        logger.success("=" * 80)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the ST-GNN.
        
        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix (num_nodes, num_static_features + lag_window)
        edge_index : torch.Tensor
            Edge indices (2, num_edges)
            
        Returns
        -------
        torch.Tensor
            Flood probability predictions (num_nodes, 1)
        """
        num_nodes = x.size(0)
        
        # Split features into static and temporal
        static_features = x[:, :self.num_static_features]  # (num_nodes, num_static_features)
        temporal_features = x[:, self.num_static_features:]  # (num_nodes, lag_window)
        
        # Process temporal features through GRU
        # Reshape for GRU: (num_nodes, lag_window, 1)
        temporal_features = temporal_features.unsqueeze(-1)
        
        # Get temporal embedding
        temporal_embedding = self.temporal_module(temporal_features)  # (num_nodes, hidden_size)
        
        # Concatenate static and temporal for spatial processing
        spatial_input = torch.cat([static_features, temporal_embedding], dim=1)
        
        # Process through spatial module
        spatial_embedding = self.spatial_module(spatial_input, edge_index)
        
        # Fuse all modalities
        fused_embedding = self.fusion_module(
            static_features,
            temporal_embedding,
            spatial_embedding,
        )
        
        # Output head
        output = self.output_head(fused_embedding)
        
        return output
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract intermediate embeddings for visualization/analysis.
        
        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix
        edge_index : torch.Tensor
            Edge indices
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            (temporal_embedding, spatial_embedding, fused_embedding)
        """
        with torch.no_grad():
            # Split features
            static_features = x[:, :self.num_static_features]
            temporal_features = x[:, self.num_static_features:].unsqueeze(-1)
            
            # Get embeddings
            temporal_embedding = self.temporal_module(temporal_features)
            spatial_input = torch.cat([static_features, temporal_embedding], dim=1)
            spatial_embedding = self.spatial_module(spatial_input, edge_index)
            fused_embedding = self.fusion_module(
                static_features,
                temporal_embedding,
                spatial_embedding,
            )
            
            return temporal_embedding, spatial_embedding, fused_embedding


def create_model_from_config(config: dict, num_static_features: int) -> HydroGraphSTGNN:
    """
    Create model from configuration dictionary.
    
    Parameters
    ----------
    config : dict
        Model configuration
    num_static_features : int
        Number of static features
        
    Returns
    -------
    HydroGraphSTGNN
        Initialized model
    """
    model = HydroGraphSTGNN(
        num_static_features=num_static_features,
        lag_window=config['features']['temporal']['lag_window'],
        spatial_config=config['model']['architecture']['spatial'],
        temporal_config=config['model']['architecture']['temporal'],
        fusion_config=config['model']['architecture']['fusion'],
    )
    
    return model
