"""
Hydro-Graph DS-STGAT: Dual-Scale Spatiotemporal Graph Attention Network
for Hyper-Local Urban Flood Forecasting
=======================================================================
IIT Madras -- CV5600 Geospatial Data Science
Mohamed Zayaan S (CE23B092) * Krishna Satyam (CE23B085)

Architecture (v2)
-----------------
  DS-STGAT: GATv2Conv (spatial) + Dual-GRU with cross-temporal attention gate
  Multi-horizon: predicts lead times [1, 3, 6, 12] hr simultaneously
  Uncertainty: Monte Carlo Dropout for epistemic uncertainty quantification
  Loss: Focal Tversky Loss (alpha=0.30, beta=0.70, gamma=0.75)
"""

__version__ = "2.0.0"
__author__  = "Mohamed Zayaan S & Krishna Satyam"

from .config import HydroGraphConfig, load_config
from .phase1_graph import GraphConstructor
from .phase2_features import FeatureEngineer
from .phase3_temporal import TemporalEncoder, get_chronological_split, create_val_event_encoder
from .phase4_model import DualScaleSTGAT, build_model, FocalTverskyLoss, MultiLeadFocalTverskyLoss
from .phase5_training import HydroGraphDataset, Trainer
from .phase6_inference import InferenceEngine

# Backward compat aliases
HydroGraphSTGNN = DualScaleSTGAT
FocalLoss = FocalTverskyLoss

__all__ = [
    "HydroGraphConfig",
    "load_config",
    "GraphConstructor",
    "FeatureEngineer",
    "TemporalEncoder",
    "get_chronological_split",
    "create_val_event_encoder",
    "DualScaleSTGAT",
    "HydroGraphSTGNN",
    "build_model",
    "FocalTverskyLoss",
    "FocalLoss",
    "MultiLeadFocalTverskyLoss",
    "HydroGraphDataset",
    "Trainer",
    "InferenceEngine",
]
