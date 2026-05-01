# -*- coding: utf-8 -*-
"""Quick validation script — tests all 6 phases end-to-end."""
import sys, warnings, time
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
import numpy as np

PASS = "[PASS]"
FAIL = "[FAIL]"

def section(n, title):
    print(f"\n{'='*60}")
    print(f"  Phase {n}: {title}")
    print('='*60)

errors = []

# ── Phase 1 ──────────────────────────────────────────────────────────────────
section(1, "Graph Construction")
try:
    from hydro_graph.phase1_graph import GraphConstructor
    gc = GraphConstructor(bbox=(80.20, 12.95, 80.25, 13.00), use_synthetic_fallback=True)
    G, gdf = gc.build()
    assert G.number_of_nodes() >= 50
    assert G.number_of_edges() > G.number_of_nodes()
    assert gdf.crs.to_epsg() == 4326
    assert 'x_proj' in gdf.columns
    print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    print(f"  GDF: {gdf.shape}, CRS: {gdf.crs}")
    print(PASS)
except Exception as e:
    print(f"{FAIL}: {e}")
    errors.append(f"Phase 1: {e}")
    import traceback; traceback.print_exc()

# ── Phase 2 ──────────────────────────────────────────────────────────────────
section(2, "Feature Engineering")
try:
    from hydro_graph.phase2_features import FeatureEngineer, STATIC_DIM, FEATURE_NAMES
    fe = FeatureEngineer(use_synthetic=True)
    feat, df_feat = fe.compute_features(G, gdf)
    assert feat.shape == (G.number_of_nodes(), STATIC_DIM), f"shape={feat.shape}"
    assert not np.isnan(feat).any(), "NaN in features"
    assert feat[:, 0].min() >= 0, "Negative elevation"
    print(f"  Shape: {feat.shape}")
    print(f"  Elevation: [{feat[:,0].min():.1f}, {feat[:,0].max():.1f}] m")
    print(f"  NDVI: [{feat[:,4].min():.3f}, {feat[:,4].max():.3f}]")
    print(PASS)
except Exception as e:
    print(f"{FAIL}: {e}")
    errors.append(f"Phase 2: {e}")
    import traceback; traceback.print_exc()

# ── Phase 3 ──────────────────────────────────────────────────────────────────
section(3, "Temporal Encoding")
try:
    from hydro_graph.phase3_temporal import TemporalEncoder, get_chronological_split
    # Use flood-peak window: Dec 1-2 2015 (absolute hrs 720-768 from Nov 1)
    enc = TemporalEncoder(
        seq_len=6,
        event_start='2015-11-30T00:00:00',
        event_end='2015-12-03T00:00:00',
        use_synthetic=True,
    )
    t0 = time.time()
    enc.encode(feat, gdf['lon'].values, gdf['lat'].values)
    elapsed = time.time() - t0
    N = G.number_of_nodes()
    assert enc.rainfall.shape[1] == N
    assert enc.rainfall.min() >= 0
    peak_rain = float(enc.rainfall.max())
    assert peak_rain > 5.0, f"No storm rainfall (max={peak_rain:.1f})"
    flood_rate = float(enc.labels.mean())
    x_t, y = enc.build_snapshot(6)
    assert x_t.shape == (N, 6, 1), f"snapshot shape={x_t.shape}"
    train_idx, val_idx, test_idx = get_chronological_split(enc.T, enc.seq_len)
    assert len(train_idx) > 0 and len(val_idx) > 0 and len(test_idx) > 0
    assert train_idx[-1] < val_idx[0]
    assert val_idx[-1]   < test_idx[0]
    print(f"  Rainfall: {enc.rainfall.shape}, max={peak_rain:.1f} mm/hr")
    print(f"  Flood rate: {flood_rate*100:.1f}%")
    print(f"  Splits: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    print(f"  Time: {elapsed:.2f}s")
    print(PASS)
except Exception as e:
    print(f"{FAIL}: {e}")
    errors.append(f"Phase 3: {e}")
    import traceback; traceback.print_exc()

# ── Phase 4 ──────────────────────────────────────────────────────────────────
section(4, "Model Architecture")
try:
    import torch
    from hydro_graph.phase4_model import HydroGraphSTGNN, FocalLoss, build_model
    from hydro_graph.config import load_config
    cfg = load_config()
    model = build_model(cfg)
    N = G.number_of_nodes()
    edges = list(G.edges())
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x = torch.rand(N, 17)

    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
    assert out.shape == (N,), f"output shape={out.shape}"
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0

    model.train()
    y = torch.randint(0, 2, (N,)).float()
    fl = FocalLoss(alpha=0.25, gamma=2.0)
    pred = model(x, edge_index)
    loss = fl(pred, y)
    assert loss.item() >= 0 and torch.isfinite(loss)
    loss.backward()
    nan_grads = sum(1 for p in model.parameters() if p.grad is not None and torch.isnan(p.grad).any())
    assert nan_grads == 0

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Output: {out.shape}, range=[{out.min():.3f}, {out.max():.3f}]")
    print(f"  Focal Loss: {loss.item():.4f}")
    print(f"  Params: {params:,}")
    print(PASS)
except Exception as e:
    print(f"{FAIL}: {e}")
    errors.append(f"Phase 4: {e}")
    import traceback; traceback.print_exc()

# ── Phase 5 ──────────────────────────────────────────────────────────────────
section(5, "Training & Evaluation")
try:
    import tempfile
    from pathlib import Path
    from hydro_graph.phase5_training import HydroGraphDataset, Trainer
    from hydro_graph.phase3_temporal import get_chronological_split

    edges = list(G.edges())
    edge_index_np = np.array(edges, dtype=np.int64).T
    ds = HydroGraphDataset(feat, enc.rainfall, enc.labels, edge_index_np, seq_len=6)

    train_idx, val_idx, test_idx = get_chronological_split(enc.T, enc.seq_len)
    train_sub = train_idx[:8]
    val_sub   = val_idx[:4]
    test_sub  = test_idx[:4]

    cfg.training.epochs = 3
    cfg.training.early_stopping_patience = 2
    cfg.training.batch_size = 256

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg.paths.best_checkpoint = str(Path(tmpdir) / 'best.pt')
        cfg.paths.last_checkpoint = str(Path(tmpdir) / 'last.pt')
        model2 = build_model(cfg)
        trainer = Trainer(model2, cfg, base_dir=Path(tmpdir))
        t0 = time.time()
        history = trainer.train(ds, train_sub, val_sub)
        elapsed = time.time() - t0
        assert 'train_loss' in history
        assert len(history['train_loss']) >= 1
        metrics = trainer.evaluate(ds, test_sub)
        for k in ['f1','precision','recall','auroc']:
            assert k in metrics, f"Missing metric: {k}"
            assert 0.0 <= metrics[k] <= 1.0, f"{k}={metrics[k]}"

    print(f"  Epochs: {len(history['train_loss'])}, time={elapsed:.1f}s")
    print(f"  Train loss: {history['train_loss'][-1]:.4f}")
    print(f"  Test F1={metrics['f1']:.4f}, AUC={metrics['auroc']:.4f}")
    print(PASS)
except Exception as e:
    print(f"{FAIL}: {e}")
    errors.append(f"Phase 5: {e}")
    import traceback; traceback.print_exc()

# ── Phase 6 ──────────────────────────────────────────────────────────────────
section(6, "Inference & Geospatial Mapping")
try:
    import tempfile
    from pathlib import Path
    from hydro_graph.phase6_inference import InferenceEngine, _risk_category
    from hydro_graph.phase3_temporal import get_chronological_split

    assert _risk_category(0.10) == ("Low Risk", "#2ECC71")
    assert _risk_category(0.85) == ("Very High Risk", "#E74C3C")

    edges = list(G.edges())
    edge_index_np = np.array(edges, dtype=np.int64).T
    ds2 = HydroGraphDataset(feat, enc.rainfall, enc.labels, edge_index_np)

    model3 = build_model(cfg)
    model3.eval()
    _, _, test_idx2 = get_chronological_split(enc.T, enc.seq_len)
    test_sub2 = test_idx2[:5]

    with tempfile.TemporaryDirectory() as tmpdir_str:
        # Resolve to full long path to avoid Windows 8.3 short-path issues
        tmpdir = Path(tmpdir_str).resolve()
        engine = InferenceEngine(model3, cfg, base_dir=tmpdir)
        df_pred = engine.run(ds2, test_sub2, gdf)

        assert len(df_pred) == G.number_of_nodes()
        for col in ['lat','lon','flood_prob','flood_binary']:
            assert col in df_pred.columns
        assert df_pred['flood_prob'].between(0,1).all()
        assert df_pred['flood_binary'].isin([0,1]).all()

        png = engine.create_static_map(df_pred, str(tmpdir/'map.png'))
        import os
        assert os.path.isfile(png), f"PNG not created: {png}"
        assert os.path.getsize(png) > 10_000, f"PNG too small: {os.path.getsize(png)}"

        csv = engine.save_predictions(df_pred, str(tmpdir/'pred.csv'))
        assert os.path.isfile(csv)

        html = engine.create_folium_map(df_pred, str(tmpdir/'map.html'))
        if html:
            assert os.path.isfile(html)

    print(f"  Nodes: {len(df_pred)}, Flood rate: {df_pred['flood_binary'].mean()*100:.1f}%")
    png_size_kb = os.path.getsize(png) // 1024 if os.path.exists(png) else 0
    print(f"  Map PNG: OK ({png_size_kb} KB)")
    print(PASS)
except Exception as e:
    print(f"{FAIL}: {e}")
    errors.append(f"Phase 6: {e}")
    import traceback; traceback.print_exc()

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  VALIDATION SUMMARY")
print('='*60)
if errors:
    print(f"  FAILURES ({len(errors)}):")
    for e in errors:
        print(f"    X {e}")
    sys.exit(1)
else:
    print("  ALL 6 PHASES PASSED")
    print('='*60)
