"""
Quick start script for testing the Hydro-Graph ST-GNN setup.

This script performs a minimal test to verify all components are working.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
import torch
import networkx as nx
import numpy as np

# Test imports
try:
    from config import load_config
    logger.success("✓ Config module imported successfully")
except Exception as e:
    logger.error(f"✗ Config import failed: {e}")
    sys.exit(1)

try:
    from src import (
        GraphConstructor,
        FeatureEngineer,
        HydroGraphDataset,
        HydroGraphSTGNN,
        Trainer,
        FloodPredictor,
    )
    logger.success("✓ All source modules imported successfully")
except Exception as e:
    logger.error(f"✗ Source import failed: {e}")
    sys.exit(1)

# Test PyTorch Geometric
try:
    from torch_geometric.data import Data
    logger.success("✓ PyTorch Geometric imported successfully")
except Exception as e:
    logger.error(f"✗ PyTorch Geometric import failed: {e}")
    logger.warning("Install with: pip install torch-geometric")
    sys.exit(1)

# Test CUDA
if torch.cuda.is_available():
    logger.success(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
else:
    logger.warning("✗ CUDA not available. Training will use CPU (slow).")

# Create a minimal test graph
logger.info("Creating test graph...")
G = nx.DiGraph()
G.add_nodes_from([
    (0, {'x': 0.0, 'y': 0.0, 'lon': 80.2, 'lat': 13.0}),
    (1, {'x': 100.0, 'y': 0.0, 'lon': 80.21, 'lat': 13.0}),
    (2, {'x': 0.0, 'y': 100.0, 'lon': 80.2, 'lat': 13.01}),
    (3, {'x': 100.0, 'y': 100.0, 'lon': 80.21, 'lat': 13.01}),
])
G.add_edges_from([
    (0, 1, {'length': 100}),
    (1, 3, {'length': 100}),
    (0, 2, {'length': 100}),
    (2, 3, {'length': 100}),
])

# Add dummy features
for node in G.nodes():
    G.nodes[node].update({
        'elevation': 10.0 + np.random.randn(),
        'slope': 1.0,
        'twi': 5.0,
        'ndvi': 0.3,
        'ndwi': 0.1,
        'ndbi': 0.5,
        'imperviousness': 60.0,
        'sar_vv': -12.0,
    })

logger.success(f"✓ Test graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Create test data object
logger.info("Creating test PyG Data object...")
num_nodes = G.number_of_nodes()
num_static_features = 8
lag_window = 6

# Node features (static + temporal)
x = torch.randn(num_nodes, num_static_features + lag_window)

# Edge index
edge_list = list(G.edges())
edge_index = torch.tensor([[u, v] for u, v in edge_list], dtype=torch.long).t()

# Labels
y = torch.randint(0, 2, (num_nodes, 1), dtype=torch.float32)

# Create Data object
data = Data(x=x, edge_index=edge_index, y=y)
logger.success(f"✓ PyG Data object created: {data}")

# Test model creation
logger.info("Creating test model...")
try:
    model = HydroGraphSTGNN(
        num_static_features=num_static_features,
        lag_window=lag_window,
        spatial_config={
            'num_layers': 2,
            'hidden_channels': 32,
            'aggregator': 'mean',
            'dropout': 0.3,
        },
        temporal_config={
            'hidden_size': 16,
            'num_layers': 1,
            'dropout': 0.2,
        },
        fusion_config={
            'hidden_dims': [64, 32],
            'dropout': 0.3,
        },
    )
    logger.success(f"✓ Model created with {model.count_parameters():,} parameters")
except Exception as e:
    logger.error(f"✗ Model creation failed: {e}")
    sys.exit(1)

# Test forward pass
logger.info("Testing forward pass...")
try:
    model.eval()
    with torch.no_grad():
        output = model(data.x, data.edge_index)
    logger.success(f"✓ Forward pass successful: output shape = {output.shape}")
    logger.info(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")
except Exception as e:
    logger.error(f"✗ Forward pass failed: {e}")
    sys.exit(1)

# Final summary
logger.info("\n" + "=" * 80)
logger.success("ALL TESTS PASSED! ✓")
logger.info("=" * 80)
logger.info("System is ready for:")
logger.info("  1. Graph construction with OSMnx")
logger.info("  2. Feature engineering with rasterio")
logger.info("  3. Model training with PyTorch Geometric")
logger.info("  4. Flood prediction and visualization")
logger.info("\nNext steps:")
logger.info("  - Prepare your data (DEM, Sentinel-2, Sentinel-1, rainfall)")
logger.info("  - Update config/config.yaml with your paths")
logger.info("  - Run: python examples/end_to_end_pipeline.py")
logger.info("=" * 80)
