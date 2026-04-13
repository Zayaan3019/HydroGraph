# Hydro-Graph ST-GNN

**Production-grade Spatiotemporal Graph Neural Network for Urban Flood Forecasting**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.3+-3C2179.svg)](https://pytorch-geometric.readthedocs.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

**Hydro-Graph** is a state-of-the-art Spatiotemporal Graph Neural Network (ST-GNN) designed for hyper-local urban flood forecasting. Unlike traditional grid-based pixel mapping approaches, Hydro-Graph extracts the physical street and drainage topology of a city as a mathematical graph $G=(V,E)$, binds physical terrain and spectral features to nodes, processes sequential rainfall data using a Gated Recurrent Unit (GRU), and routes flood potential spatially using GraphSAGE.

### Key Features

- 🌆 **Graph-Based Urban Modeling**: Extracts street networks and drainage systems from OpenStreetMap
- 🛰️ **Multi-Modal Feature Integration**: Combines terrain (DEM), spectral (Sentinel-2), and SAR (Sentinel-1) data
- 🧠 **Hybrid Architecture**: GraphSAGE for spatial propagation + GRU for temporal modeling
- ⚖️ **Class Imbalance Handling**: Focal Loss to address rare flood events
- 📊 **Production-Ready**: Strict typing, modular design, comprehensive logging
- 🗺️ **Interactive Visualization**: Static and interactive flood risk maps

## Architecture

The model implements a sophisticated fusion of spatial and temporal processing:

```
Input Features:
├── Static Physical Features (Per Node)
│   ├── Terrain: Elevation, Slope, TWI
│   ├── Spectral: NDVI, NDWI, NDBI, Imperviousness
│   └── SAR: Sentinel-1 VV Polarization
└── Temporal Features (Time Series)
    └── Rainfall: 6-hour lag window (GPM IMERG + IMD Gauge)

Model Pipeline:
┌─────────────────────────────────────────────────────────┐
│ 1. Temporal Module (GRU)                                │
│    - Processes 6-hour rainfall lag window               │
│    - Generates temporal hidden state                    │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ 2. Spatial Module (GraphSAGE)                          │
│    - 3-layer message passing                            │
│    - Propagates features through urban topology         │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ 3. Fusion Module (MLP)                                  │
│    - Combines static, temporal, spatial embeddings      │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ 4. Output Head (Sigmoid)                                │
│    - Node-level flood probability: P(Flood) ∈ [0, 1]   │
└─────────────────────────────────────────────────────────┘
```

## Installation

### Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for training)
- 16GB+ RAM for large urban graphs

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/hydro-graph.git
cd hydro-graph

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install PyTorch Geometric (adjust for your CUDA version)
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
```

## Quick Start

### 1. Graph Construction

```python
from config import load_config
from src import GraphConstructor

# Load configuration
config = load_config()

# Initialize graph constructor
constructor = GraphConstructor(
    bbox=config.location.bbox,
    target_crs=config.location.target_crs,
    network_type=config.graph.network_type,
)

# Build urban topology graph
graph = constructor.build_graph()

# Save graph
constructor.save_graph(Path('data/graphs'))
```

### 2. Feature Engineering

```python
from src import FeatureEngineer

# Initialize feature engineer
engineer = FeatureEngineer(
    graph=graph,
    node_gdf=constructor.node_gdf,
    raster_paths=config.data.rasters,
    target_crs=config.location.target_crs,
)

# Extract and bind all features
graph_with_features = engineer.engineer_all_features()
```

### 3. Dataset Creation

```python
from datetime import datetime
from src import RainfallDataLoader, TemporalDatasetCreator, HydroGraphDataset

# Initialize rainfall loader
rainfall_loader = RainfallDataLoader(
    gpm_dir=config.data.precipitation.gpm_imerg_dir,
    imd_file=config.data.precipitation.imd_gauge_file,
    bbox=config.location.bbox,
)

# Create temporal dataset
dataset_creator = TemporalDatasetCreator(
    graph=graph_with_features,
    rainfall_loader=rainfall_loader,
    lag_window=config.features.temporal.lag_window,
)

# Generate dataset for storm event
start_date = datetime(2015, 11, 1)
end_date = datetime(2015, 12, 31)
data_list = dataset_creator.create_temporal_sequence(start_date, end_date)

# Create PyG dataset and split
dataset = HydroGraphDataset(data_list)
train_data, val_data, test_data = dataset.train_val_test_split(
    train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
)
```

### 4. Model Training

```python
from torch_geometric.loader import NeighborLoader
from src import HydroGraphSTGNN, Trainer

# Create data loaders
train_loader = NeighborLoader(
    train_data,
    num_neighbors=config.model.neighbor_sampling.num_neighbors,
    batch_size=config.model.neighbor_sampling.batch_size,
)

val_loader = NeighborLoader(val_data, num_neighbors=[15, 10, 5], batch_size=512)

# Initialize model
model = HydroGraphSTGNN(
    num_static_features=8,  # elevation, slope, twi, ndvi, ndwi, ndbi, imperviousness, sar_vv
    lag_window=6,
    spatial_config=config.model.architecture.spatial,
    temporal_config=config.model.architecture.temporal,
    fusion_config=config.model.architecture.fusion,
)

# Initialize trainer
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config,
    device='cuda',
)

# Train model
trainer.train(num_epochs=config.training.epochs)
```

### 5. Inference and Visualization

```python
from src import FloodPredictor, GeospatialVisualizer

# Load trained model
predictor = FloodPredictor(
    model=model,
    checkpoint_path=Path('checkpoints/best_model.pth'),
    device='cuda',
)

# Generate predictions
test_data_obj = test_data.get(0)  # First test snapshot
predictions = predictor.predict(test_data_obj)

# Create visualizations
visualizer = GeospatialVisualizer(
    graph=graph,
    node_gdf=constructor.node_gdf,
    edge_gdf=constructor.edge_gdf,
)

# Static map
visualizer.plot_static_map(
    predictions=predictions,
    output_path=Path('outputs/flood_risk_map.png'),
    title='Chennai Flood Risk - December 2015',
)

# Interactive map
visualizer.create_interactive_map(
    predictions=predictions,
    output_path=Path('outputs/flood_risk_map.html'),
)
```

## Project Structure

```
hydro-graph/
├── config/
│   ├── __init__.py
│   ├── config.yaml              # Main configuration file
│   └── config_loader.py         # Pydantic configuration loader
├── src/
│   ├── __init__.py
│   ├── graph_construction.py    # Phase 1: Graph extraction
│   ├── feature_engineering.py   # Phase 2: Feature extraction
│   ├── dataset.py               # Phase 3: Dataset creation
│   ├── model.py                 # Phase 4: ST-GNN architecture
│   ├── trainer.py               # Phase 5: Training pipeline
│   └── inference.py             # Phase 6: Inference & visualization
├── data/
│   ├── raw/                     # Raw data (DEM, Sentinel, GPM)
│   ├── processed/               # Processed datasets
│   ├── graphs/                  # Graph pickle files
│   └── rasters/                 # GeoTIFF files
├── outputs/
│   ├── predictions/             # Model predictions
│   └── visualizations/          # Maps and plots
├── checkpoints/                 # Model checkpoints
├── logs/                        # Training logs
├── examples/
│   ├── end_to_end_pipeline.py   # Complete workflow
│   └── custom_training.py       # Advanced training
├── requirements.txt
└── README.md
```

## Configuration

The system is fully configurable via `config/config.yaml`. Key parameters:

### Location Settings
```yaml
location:
  city: "Chennai"
  bbox: [12.8, 80.1, 13.2, 80.3]  # [South, West, North, East]
  target_crs: "EPSG:32644"         # UTM Zone 44N for Chennai
```

### Model Architecture
```yaml
model:
  architecture:
    spatial:
      num_layers: 3
      hidden_channels: 128
      aggregator: "mean"
    temporal:
      hidden_size: 64
      num_layers: 2
    fusion:
      hidden_dims: [256, 128, 64]
```

### Training Parameters
```yaml
training:
  epochs: 200
  learning_rate: 0.001
  loss:
    type: "FocalLoss"
    alpha: 0.25
    gamma: 2.0
```

## Data Preparation

### Required Data Sources

1. **DEM (Digital Elevation Model)**
   - Source: SRTM 30m or ASTER GDEM
   - Resolution: 30m
   - Format: GeoTIFF

2. **Sentinel-2 Optical Data**
   - Bands: B2 (Blue), B3 (Green), B4 (Red), B8 (NIR), B11 (SWIR)
   - Pre-computed indices: NDVI, NDWI, NDBI, Imperviousness
   - Resolution: 10m
   - Format: GeoTIFF

3. **Sentinel-1 SAR Data**
   - Polarization: VV
   - Resolution: 10m
   - Format: GeoTIFF

4. **Precipitation Data**
   - Source: GPM IMERG (0.1° hourly) or IMD Gauge
   - Format: NetCDF/CSV
   - Temporal coverage: Storm event period

### Data Organization

Place data files in the following structure:
```
data/
├── rasters/
│   ├── srtm_30m_chennai.tif
│   ├── sentinel2_ndvi_10m.tif
│   ├── sentinel2_ndwi_10m.tif
│   ├── sentinel2_ndbi_10m.tif
│   ├── imperviousness_10m.tif
│   └── sentinel1_vv.tif
└── raw/
    ├── gpm_imerg/
    │   └── *.nc4
    └── imd_chennai_2015.csv
```

## Performance

### Metrics

The model is evaluated using:
- **F1-Score**: Harmonic mean of precision and recall
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)
- **ROC-AUC**: Area under ROC curve
- **Average Precision**: Area under precision-recall curve

### Computational Requirements

| Operation | GPU Memory | Time (Chennai, ~5000 nodes) |
|-----------|------------|------------------------------|
| Graph Construction | - | ~5 minutes |
| Feature Engineering | - | ~10 minutes |
| Training (200 epochs) | 8GB | ~2 hours |
| Inference (1 snapshot) | 2GB | <1 second |

## Advanced Usage

### Custom Loss Functions

```python
from src.trainer import Trainer

class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        # Your custom loss
        return loss

trainer.criterion = CustomLoss()
```

### Mini-Batch Training for Large Graphs

For graphs with >10,000 nodes, use inductive mini-batching:

```python
from torch_geometric.loader import NeighborLoader

loader = NeighborLoader(
    dataset,
    num_neighbors=[15, 10, 5],  # Per layer
    batch_size=512,
    num_workers=4,
)
```

## Citation

If you use this code for your research, please cite:

```bibtex
@software{hydro_graph_2026,
  title={Hydro-Graph: Spatiotemporal Graph Neural Network for Urban Flood Forecasting},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/hydro-graph}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenStreetMap contributors for urban topology data
- NASA for GPM IMERG precipitation data
- ESA for Sentinel-1 and Sentinel-2 satellite data
- IMD (India Meteorological Department) for gauge data

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Contact

For questions or collaboration:
- Email: your.email@example.com
- GitHub Issues: [https://github.com/yourusername/hydro-graph/issues](https://github.com/yourusername/hydro-graph/issues)

---

**Built with ❤️ for resilient cities**
