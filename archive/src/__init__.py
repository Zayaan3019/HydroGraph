"""Hydro-Graph ST-GNN: Spatiotemporal Graph Neural Network for Urban Flood Forecasting."""

__version__ = "1.0.0"
__author__ = "Hydro-Graph Team"
__description__ = "Production-grade ST-GNN for hyper-local urban flood forecasting"

from .graph_construction import GraphConstructor
from .feature_engineering import FeatureEngineer
from .dataset import HydroGraphDataset
from .model import HydroGraphSTGNN
from .trainer import Trainer
from .inference import FloodPredictor

__all__ = [
    "GraphConstructor",
    "FeatureEngineer",
    "HydroGraphDataset",
    "HydroGraphSTGNN",
    "Trainer",
    "FloodPredictor",
]
