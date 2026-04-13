# Hydro-Graph ST-GNN: Technical Architecture

## System Architecture Deep Dive

### 1. Graph Representation

The urban topology is represented as a directed graph $G = (V, E)$ where:

- **Nodes ($V$)**: Represent intersections in the street network and drainage junctions
- **Edges ($E$)**: Represent physical connections (streets, drains, canals)

Each node $v \in V$ is associated with:
- Geographic coordinates: $(lon, lat)$ in WGS84, $(x, y)$ in projected CRS
- Static physical features: $\mathbf{f}_v^{static} \in \mathbb{R}^{d_{static}}$
- Temporal features (rainfall): $\mathbf{f}_v^{temporal}(t) \in \mathbb{R}^{T_{lag}}$

### 2. Feature Engineering Pipeline

#### Static Features ($d_{static} = 8$)

**Terrain Features:**
- **Elevation** ($Z$): Extracted from SRTM 30m DEM
  $$Z_v = \text{sample}(DEM, (x_v, y_v))$$

- **Slope** ($\nabla Z$): Gradient of elevation
  $$\text{slope}_v = \arctan\left(\sqrt{\left(\frac{\partial Z}{\partial x}\right)^2 + \left(\frac{\partial Z}{\partial y}\right)^2}\right)$$

- **Topographic Wetness Index** (TWI): Indicates water accumulation potential
  $$TWI_v = \ln\left(\frac{\alpha}{\tan(\beta)}\right)$$
  where $\alpha$ is upslope contributing area and $\beta$ is slope

**Spectral Indices (from Sentinel-2):**
- **NDVI** (Normalized Difference Vegetation Index):
  $$NDVI = \frac{NIR - Red}{NIR + Red}$$

- **NDWI** (Normalized Difference Water Index):
  $$NDWI = \frac{Green - NIR}{Green + NIR}$$

- **NDBI** (Normalized Difference Built-up Index):
  $$NDBI = \frac{SWIR - NIR}{SWIR + NIR}$$

- **Imperviousness**: Pre-computed surface imperviousness percentage

**SAR Features (from Sentinel-1):**
- **VV Polarization**: C-band backscatter coefficient (soil moisture proxy)

#### Temporal Features ($T_{lag} = 6$)

For each node at time $t$, we maintain a 6-hour lag window of rainfall:
$$\mathbf{R}_v(t) = [R_v(t-1), R_v(t-2), \ldots, R_v(t-6)]$$

where $R_v(t)$ is rainfall intensity at node $v$ at time $t$.

### 3. Model Architecture

#### 3.1 Temporal Module (GRU)

The GRU processes the rainfall sequence for each node independently:

$$\mathbf{h}_v^{temp}(t) = \text{GRU}(\mathbf{R}_v(t))$$

**GRU Update Equations:**
$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t])$$
$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t])$$
$$\tilde{h}_t = \tanh(W_h \cdot [r_t \odot h_{t-1}, x_t])$$
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

where:
- $z_t$: Update gate
- $r_t$: Reset gate
- $\tilde{h}_t$: Candidate hidden state
- $h_t$: Final hidden state

#### 3.2 Spatial Module (GraphSAGE)

GraphSAGE performs neighborhood aggregation over the urban topology:

**Input to spatial module:**
$$\mathbf{x}_v = [\mathbf{f}_v^{static}, \mathbf{h}_v^{temp}]$$

**Layer $k$ aggregation:**
$$\mathbf{h}_{\mathcal{N}(v)}^{(k)} = \text{AGGREGATE}_k(\{\mathbf{h}_u^{(k-1)} : u \in \mathcal{N}(v)\})$$

**Node update:**
$$\mathbf{h}_v^{(k)} = \sigma(W^{(k)} \cdot \text{CONCAT}(\mathbf{h}_v^{(k-1)}, \mathbf{h}_{\mathcal{N}(v)}^{(k)}))$$

where $\mathcal{N}(v)$ is the neighborhood of node $v$.

**Mean aggregator:**
$$\text{AGGREGATE}_k = \frac{1}{|\mathcal{N}(v)|} \sum_{u \in \mathcal{N}(v)} \mathbf{h}_u^{(k-1)}$$

#### 3.3 Fusion Module

Combines all modalities through MLP:
$$\mathbf{h}_v^{fused} = \text{MLP}([\mathbf{f}_v^{static}, \mathbf{h}_v^{temp}, \mathbf{h}_v^{spatial}])$$

#### 3.4 Output Head

Final flood probability prediction:
$$\hat{y}_v = \sigma(W_{out} \cdot \mathbf{h}_v^{fused} + b_{out})$$

where $\sigma$ is the sigmoid function, giving $\hat{y}_v \in [0, 1]$.

### 4. Loss Function: Focal Loss

To handle severe class imbalance (flood events are rare):

$$\mathcal{L}_{focal} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

where:
- $p_t = \hat{y}_v$ if $y_v = 1$, else $1 - \hat{y}_v$
- $\alpha_t = 0.25$ (positive class weight)
- $\gamma = 2.0$ (focusing parameter)

**Intuition:** The $(1 - p_t)^\gamma$ term down-weights easy examples and focuses learning on hard misclassified examples.

### 5. Training Pipeline

#### 5.1 Mini-Batch Sampling

For large graphs ($|V| > 10,000$), we use inductive neighbor sampling:

```
For each node v in batch:
    Sample k₁ neighbors for layer 1
    Sample k₂ neighbors for layer 2
    Sample k₃ neighbors for layer 3
```

Default: $[k_1, k_2, k_3] = [15, 10, 5]$

#### 5.2 Optimization

- **Optimizer**: AdamW with weight decay
- **Learning Rate**: 0.001 with ReduceLROnPlateau scheduler
- **Gradient Clipping**: Max norm = 1.0
- **Early Stopping**: Patience = 20 epochs

### 6. Inference Pipeline

#### 6.1 Prediction Generation

For a new storm event:
1. Extract 6-hour rainfall lag window
2. Forward pass through ST-GNN
3. Generate node-level flood probabilities

#### 6.2 Risk Categorization

| Category | Probability Range | Color |
|----------|------------------|-------|
| No Risk | 0 - 30% | Blue |
| Low Risk | 30 - 60% | Light Blue |
| Medium Risk | 60 - 80% | Yellow |
| High Risk | 80 - 90% | Orange |
| Very High Risk | 90 - 100% | Red |

### 7. Computational Complexity

**Memory Complexity:**
- Graph storage: $O(|V| + |E|)$
- Feature matrix: $O(|V| \cdot d_{features})$
- Model parameters: $O(d^2 \cdot L)$ where $L$ is number of layers

**Time Complexity:**
- Forward pass: $O(|E| \cdot d \cdot L)$ for full-batch
- Mini-batch: $O(|B| \cdot k \cdot d \cdot L)$ where $B$ is batch size, $k$ is avg neighbors

### 8. Evaluation Metrics

**Binary Classification Metrics:**
- **Precision**: $\frac{TP}{TP + FP}$
- **Recall**: $\frac{TP}{TP + FN}$
- **F1-Score**: $\frac{2 \cdot Precision \cdot Recall}{Precision + Recall}$
- **ROC-AUC**: Area under ROC curve
- **Average Precision**: Area under precision-recall curve

**Critical for flood forecasting:** High recall (minimize false negatives) is prioritized over precision.

## Implementation Details

### Data Flow

```
OpenStreetMap
    ↓
Graph Construction (osmnx)
    ↓
CRS Projection (EPSG:32644)
    ↓
Feature Sampling (rasterio)
    ↓
PyG Data Objects
    ↓
NeighborLoader (mini-batching)
    ↓
ST-GNN Forward Pass
    ↓
Focal Loss
    ↓
Backpropagation
    ↓
Model Checkpoints
    ↓
Inference
    ↓
Geospatial Visualization
```

### Key Design Decisions

1. **Why GraphSAGE over GCN?**
   - Inductive learning capability
   - Better scalability through sampling
   - Mean aggregation handles variable neighborhood sizes

2. **Why GRU over LSTM?**
   - Fewer parameters (faster training)
   - Sufficient for 6-hour windows
   - Better gradient flow for short sequences

3. **Why Focal Loss over BCE?**
   - Flood events are rare (~1-5% of nodes)
   - Standard BCE under-performs on minority class
   - Focal Loss empirically improves recall by ~15-20%

4. **Why 6-hour lag window?**
   - Typical time-to-peak for urban catchments
   - Balance between context and computational cost
   - Aligned with GPM IMERG temporal resolution

## Extension Points

### Adding New Features

```python
# In feature_engineering.py
class CustomFeatureExtractor:
    def extract_custom_feature(self, node_gdf):
        # Your feature extraction logic
        return feature_values

# Update engineer.engineer_all_features()
```

### Custom Loss Functions

```python
# In trainer.py
class WeightedFocalLoss(nn.Module):
    def __init__(self, class_weights):
        super().__init__()
        self.class_weights = class_weights
    
    def forward(self, pred, target):
        # Custom loss logic
        return loss
```

### Alternative Architectures

- **GAT** (Graph Attention Networks): Replace SAGEConv with GATConv
- **Transformer**: Replace GRU with temporal Transformer layers
- **GAE** (Graph Autoencoders): Add unsupervised pre-training

## Performance Benchmarks

**Chennai Test Case (5,247 nodes, 6,891 edges):**

| Phase | Time | Memory |
|-------|------|--------|
| Graph Construction | 4.2 min | 1.5 GB |
| Feature Engineering | 8.7 min | 2.1 GB |
| Dataset Creation | 3.1 min | 3.8 GB |
| Training (200 epochs) | 127 min | 7.2 GB (GPU) |
| Inference (1 snapshot) | 0.3 sec | 1.9 GB |

**Hardware:** NVIDIA RTX 3080 (10GB), 32GB RAM, Intel i7-11700K

## References

1. Hamilton et al. (2017). "Inductive Representation Learning on Large Graphs"
2. Cho et al. (2014). "Learning Phrase Representations using RNN Encoder-Decoder"
3. Lin et al. (2017). "Focal Loss for Dense Object Detection"
4. Zhou et al. (2020). "Graph Neural Networks: A Review of Methods and Applications"
