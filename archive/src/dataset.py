"""
Phase 3: Temporal Data Encoding

This module creates the sliding window time-series dataset,
processing sequential rainfall data and compiling PyTorch Geometric datasets.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data, Dataset
from loguru import logger


class RainfallDataLoader:
    """
    Load and preprocess rainfall data from GPM IMERG and IMD Gauge.
    
    Parameters
    ----------
    gpm_dir : Path
        Directory containing GPM IMERG hourly data
    imd_file : Optional[Path]
        Path to IMD gauge data CSV file
    bbox : Tuple[float, float, float, float]
        Bounding box (South, West, North, East) for spatial filtering
    """
    
    def __init__(
        self,
        gpm_dir: Optional[Path] = None,
        imd_file: Optional[Path] = None,
        bbox: Tuple[float, float, float, float] = (12.8, 80.1, 13.2, 80.3),
    ):
        self.gpm_dir = Path(gpm_dir) if gpm_dir else None
        self.imd_file = Path(imd_file) if imd_file else None
        self.bbox = bbox
        
        logger.info("RainfallDataLoader initialized")
    
    def load_gpm_imerg(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Load GPM IMERG precipitation data for date range.
        
        Parameters
        ----------
        start_date : datetime
            Start date for data extraction
        end_date : datetime
            End date for data extraction
            
        Returns
        -------
        pd.DataFrame
            Hourly precipitation data with columns [timestamp, lon, lat, precip_mm]
        """
        logger.info(f"Loading GPM IMERG data: {start_date} to {end_date}")
        
        if self.gpm_dir is None or not self.gpm_dir.exists():
            logger.warning("GPM IMERG directory not found. Generating synthetic data...")
            return self._generate_synthetic_rainfall(start_date, end_date)
        
        # TODO: Implement actual GPM IMERG data loading
        # This would typically involve reading HDF5/NetCDF files
        logger.warning("GPM IMERG data loading not implemented. Using synthetic data.")
        return self._generate_synthetic_rainfall(start_date, end_date)
    
    def load_imd_gauge(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Load IMD gauge precipitation data.
        
        Parameters
        ----------
        start_date : datetime
            Start date for data extraction
        end_date : datetime
            End date for data extraction
            
        Returns
        -------
        pd.DataFrame
            Hourly gauge data with columns [timestamp, station_id, precip_mm]
        """
        logger.info(f"Loading IMD gauge data: {start_date} to {end_date}")
        
        if self.imd_file is None or not self.imd_file.exists():
            logger.warning("IMD gauge file not found. Skipping gauge data.")
            return pd.DataFrame()
        
        # Load IMD data
        df = pd.read_csv(self.imd_file, parse_dates=['timestamp'])
        df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
        
        logger.success(f"Loaded {len(df)} IMD gauge records")
        return df
    
    def _generate_synthetic_rainfall(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Generate synthetic rainfall data for testing.
        
        Parameters
        ----------
        start_date : datetime
            Start date
        end_date : datetime
            End date
            
        Returns
        -------
        pd.DataFrame
            Synthetic hourly precipitation data
        """
        logger.info("Generating synthetic rainfall data...")
        
        # Create hourly timestamps
        timestamps = pd.date_range(start_date, end_date, freq='H')
        
        # Create grid of locations within bbox
        south, west, north, east = self.bbox
        lats = np.linspace(south, north, 10)
        lons = np.linspace(west, east, 10)
        
        # Generate synthetic data
        data = []
        for ts in timestamps:
            for lat in lats:
                for lon in lons:
                    # Simulate rainfall with some temporal and spatial variation
                    base_precip = np.random.exponential(2.0)  # Mean 2mm/hr
                    # Add temporal pattern (higher at certain hours)
                    hour_factor = 1.5 if ts.hour in [14, 15, 16, 17, 18] else 0.5
                    precip = base_precip * hour_factor
                    
                    data.append({
                        'timestamp': ts,
                        'lon': lon,
                        'lat': lat,
                        'precip_mm': precip,
                    })
        
        df = pd.DataFrame(data)
        logger.success(f"Generated {len(df)} synthetic rainfall records")
        return df
    
    def spatial_interpolation(
        self,
        rainfall_df: pd.DataFrame,
        target_coords: np.ndarray,
    ) -> np.ndarray:
        """
        Interpolate rainfall to target coordinates using inverse distance weighting.
        
        Parameters
        ----------
        rainfall_df : pd.DataFrame
            Rainfall data with columns [lon, lat, precip_mm]
        target_coords : np.ndarray
            Target coordinates (N, 2) array of [lon, lat]
            
        Returns
        -------
        np.ndarray
            Interpolated rainfall values at target locations
        """
        if len(rainfall_df) == 0:
            return np.zeros(len(target_coords))
        
        # Extract source coordinates and values
        source_coords = rainfall_df[['lon', 'lat']].values
        source_values = rainfall_df['precip_mm'].values
        
        # Compute distances (simple Euclidean for small areas)
        interpolated = np.zeros(len(target_coords))
        
        for i, target_coord in enumerate(target_coords):
            # Calculate distances
            distances = np.sqrt(
                ((source_coords[:, 0] - target_coord[0]) ** 2) +
                ((source_coords[:, 1] - target_coord[1]) ** 2)
            )
            
            # Avoid division by zero
            distances = np.maximum(distances, 1e-10)
            
            # Inverse distance weighting
            weights = 1 / (distances ** 2)
            weights /= weights.sum()
            
            # Weighted average
            interpolated[i] = (source_values * weights).sum()
        
        return interpolated


class TemporalDatasetCreator:
    """
    Create temporal sliding window dataset for ST-GNN training.
    
    Parameters
    ----------
    graph : nx.DiGraph
        Urban topology graph with node features
    rainfall_loader : RainfallDataLoader
        Rainfall data loader
    lag_window : int
        Number of time steps to look back (default: 6 hours)
    forecast_horizon : int
        Number of time steps to forecast (default: 1 hour)
    """
    
    def __init__(
        self,
        graph: nx.DiGraph,
        rainfall_loader: RainfallDataLoader,
        lag_window: int = 6,
        forecast_horizon: int = 1,
    ):
        self.graph = graph
        self.rainfall_loader = rainfall_loader
        self.lag_window = lag_window
        self.forecast_horizon = forecast_horizon
        
        # Extract node coordinates
        self.node_coords = np.array([
            [data['lon'], data['lat']] 
            for _, data in graph.nodes(data=True)
        ])
        
        # Extract edge indices
        edge_list = list(graph.edges())
        node_to_idx = {node: idx for idx, node in enumerate(graph.nodes())}
        
        self.edge_index = torch.tensor([
            [node_to_idx[u] for u, v in edge_list],
            [node_to_idx[v] for u, v in edge_list],
        ], dtype=torch.long)
        
        logger.info("TemporalDatasetCreator initialized")
        logger.info(f"Lag window: {lag_window} hours")
        logger.info(f"Forecast horizon: {forecast_horizon} hour(s)")
    
    def extract_static_features(self) -> torch.Tensor:
        """
        Extract static node features from graph.
        
        Returns
        -------
        torch.Tensor
            Static feature matrix (num_nodes, num_static_features)
        """
        # Define feature names (these should match Phase 2 output)
        static_feature_names = [
            'elevation', 'slope', 'twi',
            'ndvi', 'ndwi', 'ndbi', 'imperviousness',
            'sar_vv',
        ]
        
        static_features = []
        for node, data in self.graph.nodes(data=True):
            node_features = []
            for feat_name in static_feature_names:
                # Use 0.0 as default if feature not present
                value = data.get(feat_name, 0.0)
                node_features.append(value)
            static_features.append(node_features)
        
        static_features_tensor = torch.tensor(static_features, dtype=torch.float32)
        
        logger.info(f"Static features shape: {static_features_tensor.shape}")
        return static_features_tensor
    
    def create_temporal_sequence(
        self,
        start_date: datetime,
        end_date: datetime,
        labels: Optional[np.ndarray] = None,
    ) -> List[Data]:
        """
        Create temporal sequence of graph snapshots.
        
        Parameters
        ----------
        start_date : datetime
            Start date for sequence
        end_date : datetime
            End date for sequence
        labels : Optional[np.ndarray]
            Ground truth flood labels (num_timestamps, num_nodes)
            If None, generates synthetic labels
            
        Returns
        -------
        List[Data]
            List of PyTorch Geometric Data objects
        """
        logger.info("=" * 80)
        logger.info("PHASE 3: TEMPORAL DATA ENCODING")
        logger.info("=" * 80)
        
        logger.info(f"Creating temporal sequence: {start_date} to {end_date}")
        
        # Load rainfall data
        rainfall_df = self.rainfall_loader.load_gpm_imerg(start_date, end_date)
        
        # Extract static features
        static_features = self.extract_static_features()
        num_nodes = static_features.shape[0]
        
        # Create hourly timestamps
        timestamps = pd.date_range(start_date, end_date, freq='H')
        
        # Create temporal rainfall matrix (num_timestamps, num_nodes)
        logger.info("Interpolating rainfall to graph nodes...")
        temporal_rainfall = np.zeros((len(timestamps), num_nodes))
        
        for t_idx, ts in enumerate(timestamps):
            # Filter rainfall for this timestamp
            ts_rainfall = rainfall_df[rainfall_df['timestamp'] == ts]
            
            if len(ts_rainfall) > 0:
                # Interpolate to node locations
                interpolated = self.rainfall_loader.spatial_interpolation(
                    ts_rainfall,
                    self.node_coords,
                )
                temporal_rainfall[t_idx, :] = interpolated
        
        logger.success(f"Temporal rainfall matrix shape: {temporal_rainfall.shape}")
        
        # Generate labels if not provided
        if labels is None:
            logger.warning("No ground truth labels provided. Generating synthetic labels...")
            labels = self._generate_synthetic_labels(temporal_rainfall)
        
        # Create sliding window dataset
        logger.info("Creating sliding window dataset...")
        dataset = []
        
        for t in range(self.lag_window, len(timestamps) - self.forecast_horizon):
            # Extract lag window [t-lag_window:t]
            lag_features = temporal_rainfall[t-self.lag_window:t, :]  # (lag_window, num_nodes)
            
            # Transpose to (num_nodes, lag_window)
            lag_features = lag_features.T
            
            # Concatenate static and temporal features
            # static: (num_nodes, num_static_features)
            # temporal: (num_nodes, lag_window)
            x = torch.cat([
                static_features,
                torch.tensor(lag_features, dtype=torch.float32)
            ], dim=1)
            
            # Extract label at t+forecast_horizon
            y = torch.tensor(
                labels[t + self.forecast_horizon - 1, :],
                dtype=torch.float32
            ).unsqueeze(1)  # (num_nodes, 1)
            
            # Create Data object
            data = Data(
                x=x,
                edge_index=self.edge_index,
                y=y,
                timestamp=timestamps[t],
            )
            
            dataset.append(data)
        
        logger.success("=" * 80)
        logger.success("PHASE 3 COMPLETE: Temporal dataset created")
        logger.success(f"Dataset size: {len(dataset)} snapshots")
        logger.success(f"Feature dimension: {dataset[0].x.shape[1]}")
        logger.success(f"Number of nodes: {dataset[0].x.shape[0]}")
        logger.success(f"Number of edges: {dataset[0].edge_index.shape[1]}")
        logger.success("=" * 80)
        
        return dataset
    
    def _generate_synthetic_labels(self, rainfall: np.ndarray) -> np.ndarray:
        """
        Generate synthetic flood labels based on rainfall accumulation.
        
        Parameters
        ----------
        rainfall : np.ndarray
            Rainfall matrix (num_timestamps, num_nodes)
            
        Returns
        -------
        np.ndarray
            Binary flood labels (num_timestamps, num_nodes)
        """
        logger.info("Generating synthetic flood labels...")
        
        # Calculate 6-hour rolling sum
        window_size = 6
        rainfall_cumsum = np.zeros_like(rainfall)
        
        for t in range(window_size, rainfall.shape[0]):
            rainfall_cumsum[t, :] = rainfall[t-window_size:t, :].sum(axis=0)
        
        # Define flood threshold (e.g., 50mm in 6 hours)
        flood_threshold = 50.0
        
        # Generate binary labels
        labels = (rainfall_cumsum > flood_threshold).astype(np.float32)
        
        # Add some spatial spreading (nodes near flooded nodes also flood)
        # This is a simplified model for demonstration
        
        flood_count = labels.sum()
        logger.info(f"Generated {int(flood_count)} flood instances")
        logger.info(f"Flood ratio: {flood_count / labels.size * 100:.2f}%")
        
        return labels


class HydroGraphDataset(Dataset):
    """
    PyTorch Geometric Dataset for Hydro-Graph ST-GNN.
    
    Parameters
    ----------
    data_list : List[Data]
        List of PyTorch Geometric Data objects
    """
    
    def __init__(self, data_list: List[Data]):
        super().__init__()
        self.data_list = data_list
        logger.info(f"HydroGraphDataset created with {len(data_list)} samples")
    
    def len(self) -> int:
        """Return dataset length."""
        return len(self.data_list)
    
    def get(self, idx: int) -> Data:
        """Get data object at index."""
        return self.data_list[idx]
    
    def train_val_test_split(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Tuple['HydroGraphDataset', 'HydroGraphDataset', 'HydroGraphDataset']:
        """
        Split dataset chronologically into train/val/test.
        
        Parameters
        ----------
        train_ratio : float
            Fraction for training set
        val_ratio : float
            Fraction for validation set
        test_ratio : float
            Fraction for test set
            
        Returns
        -------
        Tuple[HydroGraphDataset, HydroGraphDataset, HydroGraphDataset]
            Train, validation, and test datasets
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        
        n = len(self.data_list)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_data = self.data_list[:train_end]
        val_data = self.data_list[train_end:val_end]
        test_data = self.data_list[val_end:]
        
        logger.info(f"Dataset split: train={len(train_data)}, "
                   f"val={len(val_data)}, test={len(test_data)}")
        
        return (
            HydroGraphDataset(train_data),
            HydroGraphDataset(val_data),
            HydroGraphDataset(test_data),
        )
