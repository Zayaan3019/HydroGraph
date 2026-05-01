# -*- coding: utf-8 -*-
"""
Hydro-Graph ST-GNN -- CTO Demo Script
======================================
End-to-end demonstration of the complete pipeline in one script.
Designed to run in ~5-15 minutes on CPU with synthetic data.

What it demonstrates:
  1. Real OSM street/drain graph extraction for Chennai, India
  2. Physics-aware feature engineering (elevation, NDVI, TWI, imperviousness)
  3. 2015 Chennai Flood temporal dataset construction
  4. ST-GNN training (GRU + GraphSAGE) with Focal Loss
  5. Node-level flood risk inference with class-imbalance metrics
  6. Interactive Folium risk map + high-DPI static map

Run with:
    python demo.py
    python demo.py --use-real-osm     # download actual Chennai OSM (requires internet)
    python demo.py --epochs 50        # longer training for better metrics
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

# Force UTF-8 for stdout to avoid Windows cp1252 issues
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("demo")

# ── Imports ───────────────────────────────────────────────────────────────────

from hydro_graph.config import load_config
from hydro_graph.phase1_graph import GraphConstructor
from hydro_graph.phase2_features import FeatureEngineer
from hydro_graph.phase3_temporal import TemporalEncoder, get_chronological_split
from hydro_graph.phase4_model import build_model
from hydro_graph.phase5_training import HydroGraphDataset, Trainer
from hydro_graph.phase6_inference import InferenceEngine


# ── Helpers ───────────────────────────────────────────────────────────────────

BANNER = """
+======================================================================+
|        HYDRO-GRAPH  ST-GNN  --  Urban Flood Forecasting             |
|        IIT Madras   CV5600 Geospatial Data Science                  |
|        Mohamed Zayaan S (CE23B092) & Krishna Satyam (CE23B085)      |
+======================================================================+
"""


def section(n: int, title: str) -> None:
    print(f"\n{'-'*65}")
    print(f"  PHASE {n}  {title}")
    print(f"{'-'*65}")


def metrics_table(metrics: dict) -> None:
    print("\n  +------------------------------------------+")
    print("  |           TEST SET METRICS              |")
    print("  +------------------------------------------+")
    for k, v in metrics.items():
        if isinstance(v, float):
            label = k.upper().replace("_", " ")
            bar_len = int(v * 20)
            bar = "#" * bar_len + "-" * (20 - bar_len)
            print(f"  |  {label:<14} {v:.4f}  [{bar}] |")
    print("  +------------------------------------------+")


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_demo(use_real_osm: bool = False, epochs: int = 30) -> None:

    t_total = time.time()
    print(BANNER)

    cfg = load_config()
    cfg.ensure_dirs(ROOT)

    cfg.features.use_synthetic  = not use_real_osm
    cfg.training.epochs         = epochs
    cfg.training.early_stopping_patience = 10
    cfg.training.batch_size     = 512
    cfg.training.lr             = 0.001

    torch.manual_seed(42)
    np.random.seed(42)

    bbox = cfg.get_bbox("demo")
    print(f"  Study Area : {cfg.study_area.name}")
    print(f"  BBox       : {bbox}")
    print(f"  Device     : {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"  Data mode  : {'Synthetic (realistic simulation)' if cfg.features.use_synthetic else 'Real rasters'}")

    # -- Phase 1: Graph Construction -------------------------------------------
    section(1, "Spatial Topology Extraction  G = (V, E)")

    gc = GraphConstructor(
        bbox=bbox,
        crs_projected=cfg.study_area.crs_projected,
        use_synthetic_fallback=not use_real_osm,
    )

    graph_cache = ROOT / cfg.paths.graph_gpickle
    if graph_cache.exists():
        print("  Loading cached graph from disk...")
        G, gdf_nodes = GraphConstructor.load(str(cfg.paths.graph_gpickle), ROOT)
    else:
        G, gdf_nodes = gc.build()
        gc.save(str(cfg.paths.graph_gpickle), ROOT)

    N = G.number_of_nodes()
    E = G.number_of_edges()
    print(f"\n  Graph G = (V, E):")
    print(f"    Nodes  |V| = {N:,}")
    print(f"    Edges  |E| = {E:,}")
    print(f"    Mean degree = {E/N:.1f}")
    print(f"  [PHASE 1 COMPLETE]")

    # -- Phase 2: Feature Engineering ------------------------------------------
    section(2, "Physics-Aware Node Attributes (Static Features)")

    feat_cache = ROOT / cfg.paths.node_features
    if feat_cache.exists():
        print("  Loading cached features from disk...")
        import pandas as pd
        df_feat = pd.read_parquet(feat_cache)
        static_features = df_feat.values.astype(np.float32)
    else:
        fe = FeatureEngineer(
            crs_projected=cfg.study_area.crs_projected,
            use_synthetic=cfg.features.use_synthetic,
        )
        static_features, df_feat = fe.compute_features(G, gdf_nodes)
        df_feat.to_parquet(feat_cache)

    print(f"\n  Feature matrix shape : {static_features.shape}")
    print(f"  Feature names        : {', '.join(df_feat.columns.tolist())}")
    print(f"  NaN count            : {np.isnan(static_features).sum()}")
    print(f"  Elevation range      : {static_features[:,0].min():.1f} - {static_features[:,0].max():.1f} m")
    print(f"  NDVI range           : {static_features[:,4].min():.3f} - {static_features[:,4].max():.3f}")
    print(f"  Imperviousness range : {static_features[:,7].min():.3f} - {static_features[:,7].max():.3f}")
    print(f"  [PHASE 2 COMPLETE]")

    # -- Phase 3: Temporal Encoding --------------------------------------------
    section(3, "Temporal Data -- 2015 Chennai Flood Event")

    temp_cache = ROOT / cfg.paths.temporal_data
    if temp_cache.exists():
        print("  Loading cached temporal data from disk...")
        enc = TemporalEncoder.load(str(cfg.paths.temporal_data), ROOT)
    else:
        enc = TemporalEncoder(
            seq_len=cfg.features.temporal_seq_len,
            event_start=cfg.temporal.event_start,
            event_end=cfg.temporal.event_end,
            use_synthetic=cfg.features.use_synthetic,
        )
        enc.encode(
            static_features,
            node_lons=gdf_nodes["lon"].values,
            node_lats=gdf_nodes["lat"].values,
        )
        enc.save(str(cfg.paths.temporal_data), ROOT)

    train_idx, val_idx, test_idx = get_chronological_split(
        enc.T, enc.seq_len,
        cfg.training.train_ratio, cfg.training.val_ratio,
    )

    print(f"\n  Storm Event    : {cfg.temporal.storm_event}")
    print(f"  Time steps (T) : {enc.T} hours")
    print(f"  Nodes (N)      : {enc.N:,}")
    print(f"  Max rainfall   : {enc.rainfall.max():.1f} mm/hr")
    print(f"  Mean flood rate: {enc.labels.mean()*100:.1f}%  (class imbalance -> Focal Loss)")
    print(f"  Lag window     : {cfg.features.temporal_seq_len} hours (GRU input sequence)")
    print(f"  Chronological split: train={len(train_idx)} | val={len(val_idx)} | test={len(test_idx)} steps")
    print(f"  [PHASE 3 COMPLETE]")

    # Build dataset
    edges = list(G.edges())
    edge_index_np = np.array(edges, dtype=np.int64).T
    dataset = HydroGraphDataset(
        static_features=static_features,
        rainfall=enc.rainfall,
        labels=enc.labels,
        edge_index=edge_index_np,
        seq_len=cfg.features.temporal_seq_len,
    )

    # -- Phase 4: Model Architecture -------------------------------------------
    section(4, "ST-GNN Model Architecture")

    model = build_model(cfg)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    from hydro_graph.phase5_training import NEIGHBOR_LOADER_OK
    batching = "NeighborLoader (pyg-lib)" if NEIGHBOR_LOADER_OK else "k_hop_subgraph (pure-PyTorch)"

    print(f"\n  Class          : HydroGraphSTGNN(nn.Module)")
    print(f"  Static enc.    : MLP Linear({cfg.model.static_dim} -> {cfg.model.sage_hidden//2} -> {cfg.model.sage_hidden//4})")
    print(f"  Temporal enc.  : GRU(input=1, hidden={cfg.model.gru_hidden}, layers={cfg.model.gru_layers}, batch_first=True)")
    print(f"  Spatial enc.   : {cfg.model.sage_layers}x SAGEConv(aggr=mean) hidden={cfg.model.sage_hidden}")
    print(f"  Output head    : Linear({cfg.model.sage_hidden} -> 64 -> 1) + Sigmoid")
    print(f"  Input tensor   : [N, {cfg.features.input_dim}]  (static={cfg.features.static_dim} + rainfall_lags={cfg.features.temporal_seq_len})")
    print(f"  Parameters     : {params:,} trainable")
    print(f"  Mini-batching  : {batching}")
    print(f"  [PHASE 4 COMPLETE]")

    # -- Phase 5: Training -----------------------------------------------------
    section(5, "Training & Evaluation Pipeline")

    trainer = Trainer(model, cfg, base_dir=ROOT)
    best_ckpt = ROOT / cfg.paths.best_checkpoint

    if best_ckpt.exists():
        print("  Found existing checkpoint -- loading best model...")
        trainer.load_best_checkpoint()
    else:
        print(f"\n  Loss function  : Focal Loss (alpha={cfg.training.focal_alpha}, gamma={cfg.training.focal_gamma})")
        print(f"  Optimizer      : Adam (lr={cfg.training.lr}, wd={cfg.training.weight_decay})")
        print(f"  LR scheduler   : CosineAnnealingLR")
        print(f"  Training for max {epochs} epochs (early-stop patience={cfg.training.early_stopping_patience})...")
        print()
        t0 = time.time()
        history = trainer.train(dataset, train_idx, val_idx)
        elapsed = time.time() - t0

        print(f"\n  Training finished in {elapsed:.0f}s")
        print(f"  Epochs completed : {len(history['train_loss'])}")
        print(f"  Best val F1      : {max(history['val_f1']):.4f}")
        print(f"  Best val AUC-ROC : {max(history['val_auroc']):.4f}")

        trainer.load_best_checkpoint()

        # Save training curves
        engine_tmp = InferenceEngine(model, cfg, base_dir=ROOT)
        curves_path = engine_tmp.plot_training_curves(history)
        print(f"  Training curves saved: {curves_path}")

    print("\n  Evaluating on held-out test set...")
    test_metrics = trainer.evaluate(dataset, test_idx)
    trainer.save_metrics(test_metrics)
    metrics_table(test_metrics)
    print(f"  [PHASE 5 COMPLETE]")

    # -- Phase 6: Inference & Mapping ------------------------------------------
    section(6, "Geospatial Risk Mapping -- Chennai Flood Risk")

    print("\n  Running inference over test period...")
    engine = InferenceEngine(model, cfg, base_dir=ROOT)
    df_pred = engine.run(dataset, test_idx, gdf_nodes)

    low    = (df_pred["flood_prob"] < 0.30).sum()
    mod    = ((df_pred["flood_prob"] >= 0.30) & (df_pred["flood_prob"] < 0.50)).sum()
    high   = ((df_pred["flood_prob"] >= 0.50) & (df_pred["flood_prob"] < 0.70)).sum()
    vhigh  = (df_pred["flood_prob"] >= 0.70).sum()

    print(f"\n  Node-level flood risk distribution ({N:,} nodes):")
    print(f"    [GREEN]  Low risk     (0-30%)  : {low:,} nodes  ({low/N*100:.1f}%)")
    print(f"    [YELLOW] Moderate     (30-50%) : {mod:,} nodes  ({mod/N*100:.1f}%)")
    print(f"    [ORANGE] High risk    (50-70%) : {high:,} nodes  ({high/N*100:.1f}%)")
    print(f"    [RED]    Very high    (>70%)   : {vhigh:,} nodes  ({vhigh/N*100:.1f}%)")

    csv_path  = engine.save_predictions(df_pred)
    png_path  = engine.create_static_map(df_pred)
    html_path = engine.create_folium_map(df_pred)
    print(f"  [PHASE 6 COMPLETE]")

    # -- Final Summary ---------------------------------------------------------
    elapsed_total = time.time() - t_total
    print(f"\n{'='*65}")
    print("  HYDRO-GRAPH PIPELINE -- COMPLETE")
    print(f"{'='*65}")
    print(f"\n  Total runtime  : {elapsed_total:.0f}s")
    print(f"  Graph          : {N:,} nodes | {E:,} edges")
    print(f"  At-risk nodes  : {df_pred['flood_binary'].mean()*100:.1f}% (threshold={cfg.training.threshold})")
    print(f"\n  Output files saved to: data/outputs/")
    print(f"    {Path(csv_path).name:<30}  -- per-node flood probabilities")
    print(f"    {Path(png_path).name:<30}  -- static risk map (publication quality)")
    if html_path:
        print(f"    {Path(html_path).name:<30}  -- interactive Folium map")
    if html_path:
        print(f"\n  >> Open '{html_path}' in a browser to explore the interactive risk map.")
    print(f"\n  Test Metrics Summary:")
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"    {k.upper():<15}: {v:.4f}")
    print(f"\n{'='*65}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hydro-Graph ST-GNN -- Urban Flood Forecasting Demo"
    )
    p.add_argument("--use-real-osm",   action="store_true",
                   help="Download Chennai OSM (requires internet)")
    p.add_argument("--epochs",         type=int, default=30,
                   help="Max training epochs (default: 30)")
    p.add_argument("--force-retrain",  action="store_true",
                   help="Delete cached data and retrain from scratch")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.force_retrain:
        for p in [
            "data/processed/chennai_graph.gpickle",
            "data/processed/node_features.parquet",
            "data/processed/temporal_data.npz",
            "data/models/best_model.pt",
            "data/models/last_model.pt",
        ]:
            fp = ROOT / p
            if fp.exists():
                fp.unlink()
                print(f"Cleared: {fp.name}")

    run_demo(use_real_osm=args.use_real_osm, epochs=args.epochs)
