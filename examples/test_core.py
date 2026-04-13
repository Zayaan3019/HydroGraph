"""
Minimal integration test for Hydro-Graph ST-GNN core components.
Tests each module independently without heavy dependencies.
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 80)
print("HYDRO-GRAPH ST-GNN: CORE FUNCTIONALITY TEST")
print("=" * 80)

# Test 1: Configuration
print("\n[1/6] Testing Configuration...")
try:
    from config import load_config
    config = load_config()
    assert config.location.city == "Chennai"
    assert config.location.target_crs == "EPSG:32644"
    assert config.features.temporal['lag_window'] == 6
    print("✓ Configuration loading works")
except Exception as e:
    print(f"✗ Configuration failed: {e}")
    sys.exit(1)

# Test 2: Model Architecture (no data required)
print("\n[2/6] Testing Model Architecture...")
try:
    import torch
    import torch.nn as nn
    from src.model import HydroGraphSTGNN
    
    # Create minimal model
    model = HydroGraphSTGNN(
        num_static_features=8,
        lag_window=6,
        spatial_config={'num_layers': 2, 'hidden_channels': 32, 'aggregator': 'mean', 'dropout': 0.3},
        temporal_config={'hidden_size': 16, 'num_layers': 1, 'dropout': 0.2},
        fusion_config={'hidden_dims': [64, 32], 'dropout': 0.3},
    )
    
    assert model.count_parameters() > 0
    print(f"✓ Model created with {model.count_parameters():,} parameters")
except Exception as e:
    print(f"✗ Model creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Forward Pass with Dummy Data
print("\n[3/6] Testing Forward Pass...")
try:
    from torch_geometric.data import Data
    
    # Create dummy graph data
    num_nodes = 10
    x = torch.randn(num_nodes, 8 + 6)  # 8 static + 6 temporal
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long)
    y = torch.rand(num_nodes, 1)
    
    data = Data(x=x, edge_index=edge_index, y=y)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(data.x, data.edge_index)
    
    assert output.shape == (num_nodes, 1)
    assert (output >= 0).all() and (output <= 1).all()  # Sigmoid output
    print(f"✓ Forward pass successful: output shape = {output.shape}")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Loss Function
print("\n[4/6] Testing Focal Loss...")
try:
    from src.trainer import FocalLoss
    
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    pred = torch.rand(10, 1)
    target = torch.randint(0, 2, (10, 1), dtype=torch.float32)
    
    loss = criterion(pred, target)
    
    assert loss.item() >= 0
    print(f"✓ Focal Loss computed: {loss.item():.4f}")
except Exception as e:
    print(f"✗ Focal Loss failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Metrics Tracker
print("\n[5/6] Testing Metrics...")
try:
    from src.trainer import MetricsTracker
    
    metrics_tracker = MetricsTracker(device='cpu')
    
    # Update with dummy predictions
    preds = torch.rand(100, 1)
    targets = torch.randint(0, 2, (100, 1), dtype=torch.long)
    
    metrics_tracker.update(preds, targets)
    metrics = metrics_tracker.compute()
    
    assert 'f1_score' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'roc_auc' in metrics
    
    print(f"✓ Metrics computed:")
    print(f"  - F1 Score: {metrics['f1_score']:.4f}")
    print(f"  - Precision: {metrics['precision']:.4f}")
    print(f"  - Recall: {metrics['recall']:.4f}")
    print(f"  - ROC-AUC: {metrics['roc_auc']:.4f}")
except Exception as e:
    print(f"✗ Metrics failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Dataset Creation (without real data)
print("\n[6/6] Testing Dataset Structure...")
try:
    from src.dataset import HydroGraphDataset
    
    # Create dummy dataset
    data_list = []
    for i in range(10):
        x = torch.randn(5, 14)  # 5 nodes, 14 features
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        y = torch.rand(5, 1)
        data_list.append(Data(x=x, edge_index=edge_index, y=y))
    
    dataset = HydroGraphDataset(data_list)
    
    assert len(dataset) == 10
    
    # Test train/val/test split
    train_data, val_data, test_data = dataset.train_val_test_split(0.7, 0.15, 0.15)
    
    assert len(train_data) == 7
    assert len(val_data) == 1
    assert len(test_data) == 2
    
    print(f"✓ Dataset created with {len(dataset)} samples")
    print(f"  - Train: {len(train_data)} samples")
    print(f"  - Val: {len(val_data)} samples")
    print(f"  - Test: {len(test_data)} samples")
except Exception as e:
    print(f"✗ Dataset failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Final Summary
print("\n" + "=" * 80)
print("ALL CORE TESTS PASSED! ✓")
print("=" * 80)
print("\nCore components verified:")
print("  ✓ Configuration system")
print("  ✓ Model architecture (GraphSAGE + GRU)")
print("  ✓ Forward propagation")
print("  ✓ Focal Loss")
print("  ✓ Metrics tracking")
print("  ✓ Dataset handling")
print("\n" + "=" * 80)
print("SYSTEM IS FUNCTIONAL AND READY TO USE")
print("=" * 80)
print("\nNote: Graph construction and feature engineering require:")
print("  - OSMnx (for graph extraction)")
print("  - Rasterio (for raster sampling)")
print("  - Real geographic data (DEM, Sentinel-2/1, rainfall)")
print("\nTo test with real data:")
print("  1. Ensure all dependencies are installed")
print("  2. Prepare your data (see data/README.md)")
print("  3. Run: python examples/end_to_end_pipeline.py")
print("=" * 80)
