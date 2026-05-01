"""
Hydro-Graph DS-STGAT -- Main Pipeline Orchestrator (v2)
=======================================================
Runs the complete end-to-end urban flood forecasting pipeline:

  Phase 1  Graph Construction          (physics-informed topology + edge features)
  Phase 2  Feature Engineering         (16-dim static node features, SRTM DEM)
  Phase 3  Temporal Encoding           (dual-scale rainfall windows, leakage-free labels)
  Phase 4  Model Instantiation         (DS-STGAT: GATv2 + dual GRU + cross-attn gate)
  Phase 5  Training & Evaluation       (multi-lead FocalTversky, per-lead metrics)
           Baselines                   (RF, LSTM, GCN, SAGEv1 ablation)
  Phase 6  Inference & Geospatial Maps (multi-lead risk, uncertainty, calibration)

Usage
-----
    python main.py                          # synthetic data, demo bbox
    python main.py --mode full              # full Chennai metro bbox
    python main.py --skip-osm              # force synthetic graph
    python main.py --skip-train            # load checkpoint and infer only
    python main.py --force-retrain         # delete cache, re-run everything
    python main.py --epochs 100            # override training epochs
    python main.py --skip-baselines        # skip baseline ablation (faster)
    python main.py --config path/cfg.yaml  # custom config
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)-30s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("hydro_graph.main")

ROOT = Path(__file__).parent


# ─── Imports ──────────────────────────────────────────────────────────────────

from hydro_graph.config import load_config, HydroGraphConfig
from hydro_graph.phase1_graph import GraphConstructor
from hydro_graph.phase2_features import FeatureEngineer
from hydro_graph.phase3_temporal import (
    TemporalEncoder,
    get_chronological_split,
    create_val_event_encoder,
)
from hydro_graph.phase4_model import build_model
from hydro_graph.phase5_training import HydroGraphDataset, Trainer
from hydro_graph.phase6_inference import InferenceEngine


# ─── Cache Helpers ────────────────────────────────────────────────────────────

_VERSION_SUFFIX = ".cache_version"


def _read_cached_version(path: Path) -> str:
    ver_file = path.parent / (path.name + _VERSION_SUFFIX)
    if ver_file.exists():
        return ver_file.read_text().strip()
    return ""


def _write_cached_version(path: Path, version: str) -> None:
    ver_file = path.parent / (path.name + _VERSION_SUFFIX)
    ver_file.write_text(version)


def _cache_valid(path: Path, cfg: HydroGraphConfig) -> bool:
    """Return True iff cached file exists and was produced by the current cache_version."""
    return path.exists() and _read_cached_version(path) == cfg.cache_version


def _checkpoint_usable(path: Path, min_f1: float = 0.05) -> Tuple[bool, str]:
    """Return whether a checkpoint looks trained enough to reuse without retraining."""
    if not path.exists():
        return False, "missing"
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return False, f"unreadable ({exc})"

    f1 = float(ckpt.get("best_val_f1", 0.0))
    hist = ckpt.get("history", {})
    if isinstance(hist, dict):
        vals = hist.get("val_f1_lead1") or []
        if vals:
            f1 = max(f1, float(max(vals)))
    epoch = ckpt.get("epoch", "?")
    if f1 < min_f1:
        return False, f"epoch={epoch}, best_val_f1={f1:.4f} below reuse threshold {min_f1:.2f}"
    return True, f"epoch={epoch}, best_val_f1={f1:.4f}"


def _purge_cache(cfg: HydroGraphConfig) -> None:
    """Delete all derived cache files so the pipeline re-runs from scratch."""
    paths = [
        ROOT / cfg.paths.graph_gpickle,
        ROOT / cfg.paths.node_features,
        ROOT / cfg.paths.temporal_data,
        ROOT / cfg.paths.temporal_val_data,
        ROOT / cfg.paths.edge_features,
        ROOT / cfg.paths.best_checkpoint,
    ]
    for p in paths:
        if p.exists():
            p.unlink()
            logger.info("Purged cache: %s", p)
        ver = p.parent / (p.name + _VERSION_SUFFIX)
        if ver.exists():
            ver.unlink()
    logger.info("Cache purge complete.")


# ─── Pipeline ─────────────────────────────────────────────────────────────────

def run_pipeline(
    cfg: HydroGraphConfig,
    bbox_mode: str = "demo",
    skip_osm: bool = False,
    skip_train: bool = False,
    force_retrain: bool = False,
    skip_baselines: bool = False,
) -> None:
    t_start = time.time()
    cfg.ensure_dirs(ROOT)

    if force_retrain:
        _purge_cache(cfg)

    _banner("HYDRO-GRAPH DS-STGAT  --  URBAN FLOOD FORECASTING")
    logger.info("Study area    : %s", cfg.study_area.name)
    logger.info("BBox mode     : %s  %s", bbox_mode, cfg.get_bbox(bbox_mode))
    logger.info("Device        : %s", "CUDA" if torch.cuda.is_available() else "CPU")
    logger.info("Cache version : %s", cfg.cache_version)
    logger.info("Lead times    : %s hr", cfg.temporal.lead_times)

    # ── Phase 1: Graph Construction ───────────────────────────────────────────
    _phase_header(1, "Spatial Topology Extraction (Graph Construction)")

    graph_path = ROOT / cfg.paths.graph_gpickle
    gc: Optional[GraphConstructor] = None  # needed for edge refinement if fresh build

    if _cache_valid(graph_path, cfg):
        logger.info("Loading cached graph <- %s", graph_path)
        G, gdf_nodes, edge_features = GraphConstructor.load(str(graph_path), ROOT)
    else:
        gc = GraphConstructor(
            bbox=cfg.get_bbox(bbox_mode),
            crs_projected=cfg.study_area.crs_projected,
            crs_geographic=cfg.study_area.crs_geographic,
            simplify=cfg.graph.simplify,
            retain_all=cfg.graph.retain_all,
            use_synthetic_fallback=skip_osm or cfg.features.use_synthetic,
        )
        G, gdf_nodes = gc.build()
        edge_features = gc.edge_features          # [E, 4]
        gc.save(str(cfg.paths.graph_gpickle), ROOT)
        _write_cached_version(graph_path, cfg.cache_version)

    E = G.number_of_edges()
    logger.info(
        "Graph ready: %d nodes | %d edges | edge_features=%s",
        G.number_of_nodes(), E,
        edge_features.shape if edge_features is not None else "None",
    )

    # ── Phase 2: Feature Engineering ─────────────────────────────────────────
    _phase_header(2, "Physics-Aware Feature Engineering (16-dim static features)")

    feat_path = ROOT / cfg.paths.node_features
    use_cached_features = _cache_valid(feat_path, cfg)
    if use_cached_features:
        logger.info("Loading cached features <- %s", feat_path)
        import pandas as pd
        df_feat = pd.read_parquet(feat_path)
        if df_feat.shape[1] != cfg.model.static_dim:
            logger.warning(
                "Cached feature dim mismatch: got %d columns, config expects %d; rebuilding.",
                df_feat.shape[1], cfg.model.static_dim,
            )
            use_cached_features = False

    if use_cached_features:
        static_features = df_feat.values.astype(np.float32)
    else:
        fe = FeatureEngineer(
            crs_projected=cfg.study_area.crs_projected,
            use_synthetic=cfg.features.use_synthetic,
            dem_tif=str(ROOT / cfg.features.dem_tif) if cfg.features.dem_tif else None,
            sentinel2_tif=str(ROOT / cfg.features.sentinel2_tif) if cfg.features.sentinel2_tif else None,
            sentinel1_tif=str(ROOT / cfg.features.sentinel1_tif) if cfg.features.sentinel1_tif else None,
            srtm_cache_dir=str(ROOT / cfg.features.srtm_cache_dir) if cfg.features.srtm_cache_dir else None,
            coast_lat=cfg.study_area.coast_lat,
            coast_lon=cfg.study_area.coast_lon,
            rivers=[(r.lat, r.lon) for r in cfg.study_area.rivers] or None,
        )
        static_features, df_feat = fe.compute_features(G, gdf_nodes, edge_features)
        feat_path.parent.mkdir(parents=True, exist_ok=True)
        df_feat.to_parquet(feat_path)
        _write_cached_version(feat_path, cfg.cache_version)

        # Refine edge elevations with real SRTM elevations (column 0 = elevation)
        if gc is not None and edge_features is not None:
            logger.info("Refining edge elevations with Phase 2 SRTM data ...")
            node_elevations = static_features[:, 0]   # elevation is first static feature
            edge_features = gc.refine_edge_elevations(G, node_elevations)
            gc.save(str(cfg.paths.graph_gpickle), ROOT)   # re-save with refined edges
            _write_cached_version(graph_path, cfg.cache_version)

    N = static_features.shape[0]
    nan_count = int(np.isnan(static_features).sum())
    logger.info(
        "Features ready: shape=%s | NaNs=%d | feature_dim=%d",
        static_features.shape, nan_count, static_features.shape[1],
    )
    if static_features.shape[1] != cfg.model.static_dim:
        raise RuntimeError(
            "Feature dim mismatch: got %d, config expects %d. "
            "Delete stale caches or rerun with --force-retrain."
            % (
            static_features.shape[1], cfg.model.static_dim,
            )
        )

    # Convenience aliases for node coords
    node_lons = gdf_nodes["lon"].values.astype(np.float32)
    node_lats = gdf_nodes["lat"].values.astype(np.float32)

    # ── Phase 3: Temporal Encoding ────────────────────────────────────────────
    _phase_header(3, "Temporal Data Encoding (Dual-Scale Rainfall + Leakage-Free Labels)")

    temp_path = ROOT / cfg.paths.temporal_data
    short_seq = cfg.features.short_seq_len   # 6
    long_seq  = cfg.features.long_seq_len    # 12
    lead_times = cfg.temporal.lead_times     # [1, 3, 6, 12]
    max_lead   = cfg.max_lead                # 12

    if _cache_valid(temp_path, cfg):
        logger.info("Loading cached temporal data <- %s", temp_path)
        enc = TemporalEncoder.load(
            str(temp_path), ROOT,
            short_seq_len=short_seq,
            long_seq_len=long_seq,
            lead_times=lead_times,
        )
    else:
        train_evt = cfg.temporal.train_event
        enc = TemporalEncoder(
            short_seq_len=short_seq,
            long_seq_len=long_seq,
            lead_times=lead_times,
            event_start=train_evt.event_start,
            event_end=train_evt.event_end,
            flood_threshold_min_mm=cfg.temporal.flood_threshold_min_mm,
            flood_threshold_max_mm=cfg.temporal.flood_threshold_max_mm,
            flood_recovery_halflife_hr=cfg.temporal.flood_recovery_halflife_hr,
            use_synthetic=cfg.features.use_synthetic,
            rainfall_csv=str(ROOT / cfg.features.rainfall_csv) if cfg.features.rainfall_csv else None,
        )
        enc.encode(static_features, node_lons=node_lons, node_lats=node_lats)
        enc.save(str(temp_path), ROOT)
        _write_cached_version(temp_path, cfg.cache_version)

    T = enc.T
    flood_rate = float(enc.labels.mean()) * 100
    logger.info(
        "Temporal ready: %d steps x %d nodes | flood_rate=%.1f%%",
        T, enc.N, flood_rate,
    )

    # 2018 validation event (cross-event generalisation)
    val_temp_path = ROOT / cfg.paths.temporal_val_data
    if _cache_valid(val_temp_path, cfg):
        logger.info("Loading cached 2018 val event <- %s", val_temp_path)
        enc_val = TemporalEncoder.load(
            str(val_temp_path), ROOT,
            short_seq_len=short_seq,
            long_seq_len=long_seq,
            lead_times=lead_times,
        )
    else:
        logger.info("Encoding 2018 analogue validation event ...")
        enc_val = create_val_event_encoder(
            cfg, static_features,
            node_lons=node_lons,
            node_lats=node_lats,
        )
        enc_val.save(str(val_temp_path), ROOT)
        _write_cached_version(val_temp_path, cfg.cache_version)

    logger.info(
        "Val event ready: %d steps x %d nodes | flood_rate=%.1f%%",
        enc_val.T, enc_val.N, float(enc_val.labels.mean()) * 100,
    )

    # ── Build Graph Tensors ───────────────────────────────────────────────────
    edges = list(G.edges())
    if len(edges) == 0:
        raise RuntimeError("Graph has no edges — cannot train GNN.")
    edge_index = np.array(edges, dtype=np.int64).T   # [2, E]

    if edge_features is None:
        logger.warning("No edge features available; using zeros [E, 4].")
        edge_features = np.zeros((edge_index.shape[1], 4), dtype=np.float32)

    # ── Build Datasets ────────────────────────────────────────────────────────
    dataset = HydroGraphDataset(
        static_features=static_features,
        rainfall=enc.rainfall,
        labels=enc.labels,
        edge_index=edge_index,
        short_seq_len=short_seq,
        long_seq_len=long_seq,
        lead_times=lead_times,
        edge_attr=edge_features,
    )

    # Cross-event held-out validation dataset (2018)
    dataset_val_event = HydroGraphDataset(
        static_features=static_features,
        rainfall=enc_val.rainfall,
        labels=enc_val.labels,
        edge_index=edge_index,
        short_seq_len=short_seq,
        long_seq_len=long_seq,
        lead_times=lead_times,
        edge_attr=edge_features,
    )

    # Chronological splits — lookback = max(short_seq, long_seq*2) for long window
    lookback = max(short_seq, long_seq * 2)
    train_idx, val_idx, test_idx = get_chronological_split(
        T,
        lookback=lookback,
        max_lead=max_lead,
        train_ratio=cfg.training.train_ratio,
        val_ratio=cfg.training.val_ratio,
    )
    logger.info(
        "Split: train=%d | val=%d | test=%d steps  (lookback=%d, max_lead=%d)",
        len(train_idx), len(val_idx), len(test_idx), lookback, max_lead,
    )
    split_rates = {
        "train": float(enc.labels[train_idx].mean()) * 100 if len(train_idx) else 0.0,
        "val": float(enc.labels[val_idx].mean()) * 100 if len(val_idx) else 0.0,
        "test": float(enc.labels[test_idx].mean()) * 100 if len(test_idx) else 0.0,
    }
    logger.info(
        "Split flood rates: train=%.2f%% | val=%.2f%% | test=%.2f%%",
        split_rates["train"], split_rates["val"], split_rates["test"],
    )
    if split_rates["train"] < 1.0 and flood_rate >= 2.0:
        logger.warning(
            "Training split has very few flood positives; adjust train_ratio/event window "
            "or use --force-retrain after regenerating temporal data."
        )

    # 2018 val event: all valid steps as test
    val_event_steps = np.arange(lookback, enc_val.T - max_lead)
    logger.info("2018 val event test steps: %d", len(val_event_steps))

    # ── Phase 4: Model Architecture ───────────────────────────────────────────
    _phase_header(4, "DS-STGAT Model Architecture (GATv2 + Dual-GRU + Cross-Attn Gate)")

    model = build_model(cfg)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "DualScaleSTGAT | %.2fK parameters | leads=%s | static_dim=%d | input_dim=%d",
        param_count / 1000,
        cfg.temporal.lead_times,
        cfg.features.static_dim,
        cfg.features.input_dim,
    )

    # ── Phase 5: Training ─────────────────────────────────────────────────────
    _phase_header(5, "Training & Evaluation (MultiLeadFocalTverskyLoss)")

    trainer = Trainer(model, cfg, base_dir=ROOT)
    best_ckpt = ROOT / cfg.paths.best_checkpoint
    ckpt_ok, ckpt_note = _checkpoint_usable(best_ckpt)
    history = None

    if skip_train and best_ckpt.exists():
        logger.info("--skip-train: loading checkpoint %s (%s)", best_ckpt, ckpt_note)
        if not ckpt_ok:
            logger.warning("Checkpoint health warning: %s", ckpt_note)
        trainer.load_best_checkpoint()
    elif best_ckpt.exists() and not force_retrain and ckpt_ok:
        logger.info("Checkpoint found; loading (%s). Use --force-retrain to re-train.", ckpt_note)
        trainer.load_best_checkpoint()
    else:
        if best_ckpt.exists() and not force_retrain:
            logger.warning("Ignoring unusable checkpoint (%s); retraining.", ckpt_note)
        logger.info(
            "Training DS-STGAT for %d epochs | train=%d | val=%d",
            cfg.training.epochs, len(train_idx), len(val_idx),
        )
        torch.manual_seed(cfg.training.seed)
        np.random.seed(cfg.training.seed)
        history = trainer.train(dataset, train_idx, val_idx)
        trainer.load_best_checkpoint()

    # Test-set evaluation (primary 2015 event)
    logger.info("--- Test evaluation (2015 event) ---")
    test_metrics = trainer.evaluate(dataset, test_idx)
    trainer.save_metrics(test_metrics)
    _log_lead_metrics(test_metrics, cfg.temporal.lead_times, label="TEST-2015")

    # Cross-event evaluation (2018 analogue)
    if len(val_event_steps) > 0:
        logger.info("--- Cross-event evaluation (2018 analogue) ---")
        cross_metrics = trainer.evaluate(dataset_val_event, val_event_steps)
        _log_lead_metrics(cross_metrics, cfg.temporal.lead_times, label="CROSS-2018")
    else:
        cross_metrics = {}
        logger.warning("2018 val event too short for evaluation.")

    # Training curves (only if we trained this session)
    if history is not None:
        engine_tmp = InferenceEngine(model, cfg, base_dir=ROOT)
        engine_tmp.plot_training_curves(history)
        logger.info("Training curves saved.")

    # ── Baseline Ablation ─────────────────────────────────────────────────────
    baseline_metrics: dict = {}
    if not skip_baselines:
        _phase_header(5, "Ablation Study -- Baseline Comparisons")
        try:
            from hydro_graph.baselines import run_all_baselines
            logger.info("Running 4 baselines: RF, LSTM, GCN, SAGEv1 ...")
            baseline_metrics = run_all_baselines(
                dataset, train_idx, val_idx, test_idx, cfg, base_dir=ROOT
            )
            bl_path = ROOT / cfg.paths.baselines_json
            bl_path.parent.mkdir(parents=True, exist_ok=True)
            with open(bl_path, "w") as fh:
                json.dump(baseline_metrics, fh, indent=2)
            logger.info("Baseline metrics saved -> %s", bl_path)
            _log_baseline_summary(baseline_metrics)
        except Exception as exc:
            logger.warning("Baseline ablation failed: %s", exc, exc_info=True)

    # ── Phase 6: Inference & Geospatial Mapping ───────────────────────────────
    _phase_header(6, "Inference & Geospatial Risk Mapping")

    engine = InferenceEngine(model, cfg, base_dir=ROOT)

    df_pred = engine.run(dataset, test_idx, gdf_nodes, run_uncertainty=True)
    csv_path = engine.save_predictions(df_pred)
    logger.info("Predictions -> %s", csv_path)

    png_path = engine.create_static_map(df_pred)
    logger.info("Multi-lead static map -> %s", png_path)

    unc_path = engine.create_uncertainty_map(df_pred)
    logger.info("Uncertainty map -> %s", unc_path)

    html_path = engine.create_folium_map(df_pred)
    if html_path:
        logger.info("Interactive map -> %s", html_path)

    cal_path = engine.plot_calibration(dataset, test_idx)
    logger.info("Calibration curve -> %s", cal_path)

    # ── Final Summary ─────────────────────────────────────────────────────────
    total_time = time.time() - t_start
    _banner("PIPELINE COMPLETE")
    logger.info("Total runtime        : %.1f s", total_time)
    logger.info("Graph                : %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    logger.info("Training steps       : %d", len(train_idx))
    logger.info("Test steps           : %d", len(test_idx))

    # Primary metric summary per lead
    logger.info("--- DS-STGAT Performance (Test 2015) ---")
    for h_idx, h in enumerate(cfg.temporal.lead_times):
        f1  = test_metrics.get(f"f1_lead{h_idx}", 0.0)
        csi = test_metrics.get(f"csi_lead{h_idx}", 0.0)
        auc = test_metrics.get(f"auroc_lead{h_idx}", 0.0)
        logger.info("  Lead %2dhr: F1=%.4f | CSI=%.4f | AUC=%.4f", h, f1, csi, auc)

    if cross_metrics:
        logger.info("--- Cross-Event (2018 Analogue) ---")
        for h_idx, h in enumerate(cfg.temporal.lead_times):
            f1  = cross_metrics.get(f"f1_lead{h_idx}", 0.0)
            csi = cross_metrics.get(f"csi_lead{h_idx}", 0.0)
            logger.info("  Lead %2dhr: F1=%.4f | CSI=%.4f", h, f1, csi)

    if baseline_metrics:
        logger.info("--- Baseline Ablation (Lead=1hr) ---")
        for name, m in baseline_metrics.items():
            f1  = m.get("f1", m.get("f1_lead0", 0.0))
            auc = m.get("auroc", m.get("auc_roc", 0.0))
            logger.info("  %-20s: F1=%.4f | AUC=%.4f", name, f1, auc)

    logger.info("")
    logger.info("Outputs written to   : %s", ROOT / cfg.paths.outputs_dir)
    logger.info("  Predictions CSV  -> %s", csv_path)
    logger.info("  Static risk map  -> %s", png_path)
    logger.info("  Uncertainty map  -> %s", unc_path)
    if html_path:
        logger.info("  Interactive map  -> %s", html_path)
    logger.info("  Calibration      -> %s", cal_path)
    if baseline_metrics:
        logger.info("  Baselines JSON   -> %s", ROOT / cfg.paths.baselines_json)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _banner(text: str) -> None:
    bar = "=" * 70
    logger.info(bar)
    logger.info("  %s", text.center(66))
    logger.info(bar)


def _phase_header(n: int, title: str) -> None:
    logger.info("")
    logger.info("-" * 65)
    logger.info("  PHASE %d  --  %s", n, title)
    logger.info("-" * 65)


def _log_lead_metrics(metrics: dict, lead_times: list, label: str) -> None:
    logger.info("[%s] Per-lead results:", label)
    for h_idx, h in enumerate(lead_times):
        f1   = metrics.get(f"f1_lead{h_idx}", 0.0)
        prec = metrics.get(f"precision_lead{h_idx}", 0.0)
        rec  = metrics.get(f"recall_lead{h_idx}", 0.0)
        auc  = metrics.get(f"auroc_lead{h_idx}", 0.0)
        csi  = metrics.get(f"csi_lead{h_idx}", 0.0)
        br   = metrics.get(f"brier_lead{h_idx}", 0.0)
        logger.info(
            "  Lead %2dhr: F1=%.4f  Prec=%.4f  Rec=%.4f  AUC=%.4f  CSI=%.4f  Brier=%.4f",
            h, f1, prec, rec, auc, csi, br,
        )


def _log_baseline_summary(baseline_metrics: dict) -> None:
    logger.info("[BASELINES] Summary (Lead=1hr):")
    for name, m in baseline_metrics.items():
        f1  = m.get("f1", m.get("f1_lead0", 0.0))
        auc = m.get("auroc", m.get("auc_roc", 0.0))
        csi = m.get("csi", m.get("csi_lead0", 0.0))
        logger.info("  %-20s: F1=%.4f | AUC=%.4f | CSI=%.4f", name, f1, auc, csi)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hydro-Graph DS-STGAT -- Urban Flood Forecasting Pipeline"
    )
    p.add_argument(
        "--config", default=None,
        help="Path to config YAML (default: config/config.yaml)",
    )
    p.add_argument(
        "--mode", default="demo", choices=["demo", "full"],
        help="'demo' = small bbox for quick runs; 'full' = full Chennai metro",
    )
    p.add_argument(
        "--skip-osm", action="store_true",
        help="Skip OSMnx download and use synthetic fallback graph",
    )
    p.add_argument(
        "--skip-train", action="store_true",
        help="Skip training; load the best existing checkpoint and run inference",
    )
    p.add_argument(
        "--force-retrain", action="store_true",
        help="Purge all caches and re-run the full pipeline from scratch",
    )
    p.add_argument(
        "--skip-baselines", action="store_true",
        help="Skip baseline ablation (RF, LSTM, GCN, SAGEv1) to save time",
    )
    p.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of training epochs from config",
    )
    p.add_argument(
        "--synthetic", action="store_true", default=False,
        help="Force synthetic raster data (ignores real DEM/Sentinel files)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)

    # CLI overrides
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.synthetic:
        cfg.features.use_synthetic = True

    run_pipeline(
        cfg,
        bbox_mode=args.mode,
        skip_osm=args.skip_osm,
        skip_train=args.skip_train,
        force_retrain=args.force_retrain,
        skip_baselines=args.skip_baselines,
    )


if __name__ == "__main__":
    main()
