"""
Hydro-Graph DS-STGAT — Comprehensive Test Suite (v2)
=====================================================
Tests every phase of the v2 pipeline (dual-scale rainfall, multi-lead output,
physics-informed directed edges, leakage-free splits) on small synthetic data.
All tests are deterministic (seed=42) and should run in well under 120s on CPU.

This file replaces a v1-era test suite that called an API which no longer
exists (TemporalEncoder(seq_len=...), enc.build_snapshot(), a single-lead
17-dim model input, FocalLoss(alpha, gamma) without beta). That drift meant
the "test suite" collected but every fixture-dependent test errored before
assertions ran — a false sense of coverage. Tests below target the actual
v2 API in hydro_graph/phase1_graph.py .. phase6_inference.py, and add
regression coverage for two bugs found during the leakage/correctness audit:

  - orient_drainage_edges() must never leave a waterway edge pointing uphill
    or duplicated in both directions (Priority 3.8).
  - HydroGraphDataset's rain normaliser must be fit only on the training
    window, not on val/test-period rainfall extremes (Priority 1.2).

Run with:
    pytest tests/test_pipeline.py -v --tb=short
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from hydro_graph.config import load_config, HydroGraphConfig
from hydro_graph.phase1_graph import GraphConstructor, orient_drainage_edges, EDGE_TYPE_WATERWAY
from hydro_graph.phase2_features import FeatureEngineer, STATIC_DIM, FEATURE_NAMES
from hydro_graph.phase3_temporal import TemporalEncoder, get_chronological_split
from hydro_graph.phase4_model import (
    DualScaleSTGAT, FocalTverskyLoss, MultiLeadFocalTverskyLoss, build_model,
)
from hydro_graph.phase5_training import (
    HydroGraphDataset, Trainer, _compute_metrics, _expected_calibration_error,
)
from hydro_graph.phase6_inference import InferenceEngine, _risk_category
from hydro_graph.baselines import PersistenceBaseline

# ─── Fixtures ─────────────────────────────────────────────────────────────────

SMALL_BBOX = (80.24, 12.98, 80.26, 13.00)  # tiny ~2x2 km for tests
STATIC_INPUT_DIM = 16
SHORT_SEQ = 6
LONG_SEQ = 12
LEAD_TIMES = [1, 3, 6, 12]
MAX_LEAD = max(LEAD_TIMES)
MODEL_INPUT_DIM = STATIC_INPUT_DIM + SHORT_SEQ + LONG_SEQ  # 34
EDGE_DIM = 4

torch.manual_seed(42)
np.random.seed(42)


@pytest.fixture(scope="module")
def cfg() -> HydroGraphConfig:
    c = load_config()
    c.features.use_synthetic = True
    c.training.epochs = 3
    c.training.batch_size = 64
    c.training.early_stopping_patience = 2
    return c


@pytest.fixture(scope="module")
def small_graph():
    """Build a small synthetic graph (~50-300 nodes) with edge features."""
    gc = GraphConstructor(bbox=SMALL_BBOX, use_synthetic_fallback=True)
    G_multi = gc._build_synthetic_multigraph()
    G = gc._convert_to_digraph(G_multi)
    gdf = gc._build_node_geodataframe(G)
    edge_features = gc._compute_edge_features(G, gdf)
    gc.G, gc.gdf_nodes, gc.edge_features = G, gdf, edge_features
    return G, gdf, edge_features, gc


@pytest.fixture(scope="module")
def static_features(small_graph):
    """Compute static feature matrix for the small graph."""
    G, gdf, edge_features, _ = small_graph
    fe = FeatureEngineer(use_synthetic=True)
    feat, df = fe.compute_features(G, gdf, edge_features)
    return feat, df


@pytest.fixture(scope="module")
def directed_edges(small_graph, static_features):
    """Refine edge elevations with real (synthetic) elevation, then orient."""
    G, gdf, edge_features, gc = small_graph
    feat, _ = static_features
    refined = gc.refine_edge_elevations(G, feat[:, 0])
    edge_index, edge_attr = orient_drainage_edges(G, refined)
    return edge_index, edge_attr


@pytest.fixture(scope="module")
def temporal_data(small_graph, static_features, directed_edges):
    """Build a short temporal dataset covering the 2015 flood peak."""
    G, gdf, _, _ = small_graph
    feat, _ = static_features
    edge_index, _ = directed_edges
    enc = TemporalEncoder(
        short_seq_len=SHORT_SEQ,
        long_seq_len=LONG_SEQ,
        lead_times=LEAD_TIMES,
        # _PROFILE_2015 event hours are relative to event_start (hour 0), so
        # event_start MUST be the true event start (Nov 1) for the Dec 1-2
        # catastrophic-peak event (hours 720-768) to actually fall inside
        # this window.
        event_start="2015-11-01T00:00:00",
        event_end="2015-12-03T00:00:00",   # covers the Dec 1-2 peak (hr 720-768)
        use_synthetic=True,
    )
    result = enc.encode(
        feat,
        node_lons=gdf["lon"].values,
        node_lats=gdf["lat"].values,
        edge_index=edge_index,
    )
    return enc, result


@pytest.fixture(scope="module")
def split(temporal_data):
    enc, _ = temporal_data
    lookback = max(SHORT_SEQ, LONG_SEQ * 2)
    return get_chronological_split(enc.T, lookback=lookback, max_lead=MAX_LEAD)


@pytest.fixture(scope="module")
def dataset(static_features, temporal_data, directed_edges, split):
    feat, _ = static_features
    enc, _ = temporal_data
    edge_index, edge_attr = directed_edges
    train_idx, _, _ = split
    train_end_t = int(train_idx.max()) + 1 if len(train_idx) else enc.T

    ds = HydroGraphDataset(
        static_features=feat,
        rainfall=enc.rainfall,
        labels=enc.labels,
        edge_index=edge_index,
        short_seq_len=SHORT_SEQ,
        long_seq_len=LONG_SEQ,
        lead_times=LEAD_TIMES,
        edge_attr=edge_attr,
        rain_norm_fit_end_t=train_end_t,
    )
    return ds


# ─── Phase 1: Graph Construction ──────────────────────────────────────────────

class TestPhase1GraphConstruction:
    def test_synthetic_graph_nodes(self, small_graph):
        G, gdf, _, _ = small_graph
        assert G.number_of_nodes() >= 30, "Too few nodes in synthetic graph"

    def test_synthetic_graph_edges(self, small_graph):
        G, gdf, _, _ = small_graph
        assert G.number_of_edges() > G.number_of_nodes(), "Too few edges"

    def test_node_coordinates_present(self, small_graph):
        G, gdf, _, _ = small_graph
        for _, attrs in G.nodes(data=True):
            assert "x" in attrs, "Node missing 'x' (longitude)"
            assert "y" in attrs, "Node missing 'y' (latitude)"

    def test_edge_lengths_positive(self, small_graph):
        G, gdf, _, _ = small_graph
        for u, v, data in G.edges(data=True):
            assert data.get("length", 1.0) > 0, f"Non-positive edge length at ({u},{v})"

    def test_node_geodataframe_crs(self, small_graph):
        G, gdf, _, _ = small_graph
        assert gdf.crs is not None, "GeoDataFrame missing CRS"
        assert gdf.crs.to_epsg() == 4326, f"Expected EPSG:4326, got {gdf.crs}"

    def test_node_geodataframe_columns(self, small_graph):
        G, gdf, _, _ = small_graph
        for col in ["lat", "lon", "x_proj", "y_proj", "geometry"]:
            assert col in gdf.columns, f"Missing column: {col}"

    def test_no_selfloops(self, small_graph):
        G, gdf, _, _ = small_graph
        self_loops = [(u, v) for u, v in G.edges() if u == v]
        assert len(self_loops) == 0, f"Found {len(self_loops)} self-loops"

    def test_graph_is_weakly_connected(self, small_graph):
        G, gdf, _, _ = small_graph
        import networkx as nx
        assert nx.is_weakly_connected(G), "Graph is not weakly connected"

    def test_edge_features_shape(self, small_graph):
        G, gdf, edge_features, _ = small_graph
        assert edge_features.shape == (G.number_of_edges(), 4)

    def test_waterway_edges_not_bidirectional_pre_orientation(self, small_graph):
        """
        Regression test for the symmetric-drainage bug: the synthetic graph
        builder must add each waterway/drain edge in only ONE direction.
        (Roads are legitimately bidirectional and are excluded from this check.)
        """
        G, gdf, edge_features, _ = small_graph
        waterway_pairs = {
            (u, v) for i, (u, v) in enumerate(G.edges())
            if edge_features[i, 2] == EDGE_TYPE_WATERWAY
        }
        reciprocal = [(u, v) for (u, v) in waterway_pairs if (v, u) in waterway_pairs]
        assert not reciprocal, (
            f"{len(reciprocal)} waterway edge(s) present in both directions "
            f"before orientation — water would flow uphill: {reciprocal[:5]}"
        )


# ─── Phase 1b: Directed Drainage Orientation ──────────────────────────────────

class TestOrientDrainageEdges:
    def test_no_bidirectional_waterway_edges(self, small_graph, static_features, directed_edges):
        """
        The model's core physical claim ('water flows downhill') requires that
        no waterway edge pair (u,v)+(v,u) both survive orientation — that
        would let flood signal propagate uphill exactly as easily as downhill.
        """
        G, gdf, _, _ = small_graph
        edge_index, edge_attr = directed_edges
        node_ids = list(G.nodes())
        waterway_mask = edge_attr[:, 2] == EDGE_TYPE_WATERWAY
        pairs = {
            (int(edge_index[0, i]), int(edge_index[1, i]))
            for i in np.where(waterway_mask)[0]
        }
        reciprocal = [(u, v) for (u, v) in pairs if (v, u) in pairs]
        assert not reciprocal, f"Bidirectional waterway edges survived orientation: {reciprocal[:5]}"

    def test_waterway_edges_point_downhill(self, directed_edges):
        """Every surviving waterway edge must have elev_diff_norm >= 0 (source >= target)."""
        _, edge_attr = directed_edges
        waterway = edge_attr[edge_attr[:, 2] == EDGE_TYPE_WATERWAY]
        if len(waterway) == 0:
            pytest.skip("No waterway edges generated for this small graph/seed")
        assert (waterway[:, 0] >= -1e-6).all(), "Waterway edge(s) still point uphill after orientation"

    def test_road_edges_remain_bidirectional(self, small_graph, directed_edges):
        """Roads are a deliberate exception: both directions should survive."""
        edge_index, edge_attr = directed_edges
        road_mask = edge_attr[:, 2] != EDGE_TYPE_WATERWAY
        pairs = {
            (int(edge_index[0, i]), int(edge_index[1, i]))
            for i in np.where(road_mask)[0]
        }
        reciprocal = sum(1 for (u, v) in pairs if (v, u) in pairs)
        assert reciprocal > 0, "Expected at least some bidirectional road edges"

    def test_edge_count_mismatch_raises(self, small_graph):
        G, gdf, edge_features, _ = small_graph
        with pytest.raises(ValueError):
            orient_drainage_edges(G, edge_features[:-1])  # wrong row count


# ─── Phase 2: Feature Engineering ─────────────────────────────────────────────

class TestPhase2Features:
    def test_feature_matrix_shape(self, small_graph, static_features):
        G, gdf, _, _ = small_graph
        feat, df = static_features
        assert feat.shape == (G.number_of_nodes(), STATIC_DIM), (
            f"Expected ({G.number_of_nodes()}, {STATIC_DIM}), got {feat.shape}"
        )

    def test_feature_names_match(self, static_features):
        feat, df = static_features
        assert list(df.columns) == FEATURE_NAMES

    def test_no_nan_after_imputation(self, static_features):
        feat, df = static_features
        assert not np.isnan(feat).any(), "NaN values found after imputation"

    def test_elevation_range(self, static_features):
        feat, _ = static_features
        elev = feat[:, 0]
        assert elev.min() >= 0.0,  "Elevation < 0"
        assert elev.max() <= 100.0, "Elevation > 100m (unexpected)"

    def test_spectral_indices_bounded(self, static_features):
        feat, _ = static_features
        for idx, name in zip([5, 6, 7], ["ndvi", "ndwi", "ndbi"]):
            col = feat[:, idx]
            assert col.min() >= -1.05, f"{name} < -1"
            assert col.max() <=  1.05, f"{name} > 1"

    def test_imperviousness_bounded(self, static_features):
        feat, _ = static_features
        imp = feat[:, 8]
        assert imp.min() >= 0.0, "imperviousness < 0"
        assert imp.max() <= 1.0, "imperviousness > 1"

    def test_node_attributes_updated(self, small_graph, static_features):
        G, _, _, _ = small_graph
        for node_id in list(G.nodes())[:5]:
            attrs = G.nodes[node_id]
            assert "elevation" in attrs, f"Node {node_id} missing 'elevation'"
            assert "ndvi" in attrs, f"Node {node_id} missing 'ndvi'"


# ─── Phase 3: Temporal Encoding ────────────────────────────────────────────────

class TestPhase3Temporal:
    def test_rainfall_shape(self, small_graph, temporal_data):
        G, gdf, _, _ = small_graph
        enc, _ = temporal_data
        assert enc.rainfall is not None
        assert enc.rainfall.shape[1] == G.number_of_nodes(), "Rainfall nodes mismatch"

    def test_rainfall_non_negative(self, temporal_data):
        enc, _ = temporal_data
        assert enc.rainfall.min() >= 0.0, "Negative rainfall values"

    def test_labels_binary(self, temporal_data):
        enc, _ = temporal_data
        unique = np.unique(enc.labels)
        for v in unique:
            assert v in [0.0, 1.0], f"Non-binary label found: {v}"

    def test_build_input_shapes(self, small_graph, temporal_data):
        G, _, _, _ = small_graph
        enc, _ = temporal_data
        N = G.number_of_nodes()
        t = LONG_SEQ * 2 + 1  # a valid timestep with full lookback available
        x_short, x_long = enc.build_input(t)
        assert x_short.shape == (N, SHORT_SEQ), f"Expected ({N},{SHORT_SEQ}), got {x_short.shape}"
        assert x_long.shape == (N, LONG_SEQ), f"Expected ({N},{LONG_SEQ}), got {x_long.shape}"

    def test_build_input_does_not_peek_forward(self, small_graph, temporal_data):
        """
        Priority 1.3: verify the sign of every lag shift. build_input(t) must
        only read rainfall[< t], never rainfall[t] or later. We check this by
        mutating rainfall at and after t and confirming build_input(t) is
        unaffected — the only way that can hold is if it never reads those
        indices.
        """
        G, _, _, _ = small_graph
        enc, _ = temporal_data
        t = LONG_SEQ * 2 + 5
        x_short_before, x_long_before = enc.build_input(t)

        original = enc.rainfall.copy()
        enc.rainfall[t:, :] = 1e6   # corrupt everything from t onward
        x_short_after, x_long_after = enc.build_input(t)
        enc.rainfall = original     # restore for other tests

        np.testing.assert_allclose(x_short_before, x_short_after, err_msg=(
            "build_input(t) changed when rainfall[t:] was corrupted — "
            "short-term window is reading current/future rainfall (forward leak)."
        ))
        np.testing.assert_allclose(x_long_before, x_long_after, err_msg=(
            "build_input(t) changed when rainfall[t:] was corrupted — "
            "long-term window is reading current/future rainfall (forward leak)."
        ))

    def test_build_targets_reads_future_not_past(self, small_graph, temporal_data):
        """
        Priority 1.3 (target side): build_targets(t) must equal labels[t+h],
        not labels[t-h] or labels[t]. This is the deliberate "future state"
        the model is trained to predict, not a leak — but the sign must be
        exactly right or the model would be trained on the wrong target.
        """
        enc, _ = temporal_data
        t = LONG_SEQ * 2 + 5
        y = enc.build_targets(t)
        assert y.shape == (enc.N, len(LEAD_TIMES))
        for h_idx, h in enumerate(LEAD_TIMES):
            t_future = min(t + h, enc.T - 1)
            np.testing.assert_allclose(
                y[:, h_idx], enc.labels[t_future, :],
                err_msg=f"build_targets(t)[:, {h_idx}] does not match labels[t+{h}]",
            )

    def test_chronological_split(self, temporal_data):
        enc, _ = temporal_data
        lookback = max(SHORT_SEQ, LONG_SEQ * 2)
        train_idx, val_idx, test_idx = get_chronological_split(
            enc.T, lookback=lookback, max_lead=MAX_LEAD,
        )
        assert len(train_idx) > 0, "Empty training set"
        assert len(val_idx)   > 0, "Empty validation set"
        assert len(test_idx)  > 0, "Empty test set"
        # Verify strict chronological ordering (no leakage): every train index
        # precedes every val index, every val index precedes every test index.
        assert train_idx.max() < val_idx.min(),  "Train/val overlap"
        assert val_idx.max()   < test_idx.min(), "Val/test overlap"

    def test_chronological_split_no_duplicate_indices(self, temporal_data):
        enc, _ = temporal_data
        lookback = max(SHORT_SEQ, LONG_SEQ * 2)
        train_idx, val_idx, test_idx = get_chronological_split(
            enc.T, lookback=lookback, max_lead=MAX_LEAD,
        )
        all_idx = np.concatenate([train_idx, val_idx, test_idx])
        assert len(all_idx) == len(set(all_idx.tolist())), "Duplicate indices found in split"

    def test_flood_label_class_imbalance(self, temporal_data):
        enc, _ = temporal_data
        rate = float(enc.labels.mean())
        assert rate < 0.95, f"Flood rate unrealistically high: {rate:.2%}"
        peak_rain = float(enc.rainfall.max())
        assert peak_rain > 5.0, f"No storm rainfall detected (max={peak_rain:.1f} mm/hr)"


# ─── Phase 4: Model Architecture ──────────────────────────────────────────────

def _toy_edge_index_attr(N: int, E: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    edge_index = torch.randint(0, N, (2, E), generator=g)
    edge_attr = torch.rand(E, EDGE_DIM, generator=g)
    return edge_index, edge_attr


class TestPhase4Model:
    @pytest.fixture
    def model(self, cfg):
        return build_model(cfg)

    def test_model_instantiation(self, model):
        assert isinstance(model, DualScaleSTGAT)

    def test_model_forward_shape(self, model):
        N, E = 40, 120
        edge_index, edge_attr = _toy_edge_index_attr(N, E)
        x = torch.rand(N, MODEL_INPUT_DIM)
        out = model(x, edge_index, edge_attr)
        assert out.shape == (N, len(LEAD_TIMES)), f"Expected ({N},{len(LEAD_TIMES)}), got {out.shape}"

    def test_model_output_range(self, model):
        N, E = 40, 120
        edge_index, edge_attr = _toy_edge_index_attr(N, E)
        x = torch.rand(N, MODEL_INPUT_DIM)
        with torch.no_grad():
            out = model(x, edge_index, edge_attr)
        assert out.min().item() >= 0.0, "Output < 0"
        assert out.max().item() <= 1.0, "Output > 1"

    def test_model_gradient_flow(self, model):
        N, E = 40, 120
        edge_index, edge_attr = _toy_edge_index_attr(N, E)
        x = torch.rand(N, MODEL_INPUT_DIM)
        y = torch.randint(0, 2, (N, len(LEAD_TIMES))).float()
        criterion = MultiLeadFocalTverskyLoss(n_leads=len(LEAD_TIMES))
        out = model(x, edge_index, edge_attr)
        loss = criterion(out, y)
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

    def test_focal_tversky_loss_values(self):
        fl = FocalTverskyLoss(alpha=0.30, beta=0.70, gamma=0.75)
        preds  = torch.tensor([0.9, 0.1, 0.5])
        labels = torch.tensor([1.0, 0.0, 1.0])
        loss = fl(preds, labels)
        assert loss.item() >= 0.0, "Focal Tversky loss negative"
        assert torch.isfinite(loss), "Focal Tversky loss is NaN or inf"

    def test_focal_tversky_loss_edge_cases(self):
        fl = FocalTverskyLoss(alpha=0.30, beta=0.70, gamma=0.75)
        perfect_preds  = torch.tensor([0.99, 0.01, 0.99, 0.01])
        perfect_labels = torch.tensor([1.0,  0.0,  1.0,  0.0])
        loss_perfect = fl(perfect_preds, perfect_labels)
        random_preds = torch.full((4,), 0.5)
        loss_random = fl(random_preds, perfect_labels)
        assert loss_perfect.item() < loss_random.item(), "Loss not lower for near-perfect predictions"

    def test_focal_tversky_alpha_beta_must_sum_to_one(self):
        with pytest.raises(AssertionError):
            FocalTverskyLoss(alpha=0.25, beta=0.70)  # sums to 0.95, off by > 0.01


# ─── Phase 5: Training ────────────────────────────────────────────────────────

class TestPhase5Training:
    def test_dataset_snapshot(self, dataset, small_graph):
        G, _, _, _ = small_graph
        N = G.number_of_nodes()
        t = LONG_SEQ * 2 + 5
        snapshot = dataset.get_snapshot(t)
        assert snapshot.x.shape == (N, MODEL_INPUT_DIM)
        assert snapshot.y.shape == (N, len(LEAD_TIMES))

    def test_rain_norm_fit_only_on_training_window(self, static_features, temporal_data, directed_edges, split):
        """
        Regression test for the normalisation-leakage bug (Priority 1.2):
        corrupting rainfall strictly AFTER the training window must not
        change the fitted _rain_norm scalar.
        """
        feat, _ = static_features
        enc, _ = temporal_data
        edge_index, edge_attr = directed_edges
        train_idx, _, _ = split
        train_end_t = int(train_idx.max()) + 1

        ds_clean = HydroGraphDataset(
            static_features=feat, rainfall=enc.rainfall, labels=enc.labels,
            edge_index=edge_index, short_seq_len=SHORT_SEQ, long_seq_len=LONG_SEQ,
            lead_times=LEAD_TIMES, edge_attr=edge_attr, rain_norm_fit_end_t=train_end_t,
        )

        corrupted_rainfall = enc.rainfall.copy()
        corrupted_rainfall[train_end_t:, :] = 500.0   # implausibly large post-train rain
        ds_corrupted = HydroGraphDataset(
            static_features=feat, rainfall=corrupted_rainfall, labels=enc.labels,
            edge_index=edge_index, short_seq_len=SHORT_SEQ, long_seq_len=LONG_SEQ,
            lead_times=LEAD_TIMES, edge_attr=edge_attr, rain_norm_fit_end_t=train_end_t,
        )

        assert ds_clean._rain_norm == pytest.approx(ds_corrupted._rain_norm), (
            "Rain normaliser changed after corrupting only post-training-window "
            "rainfall — it is leaking val/test-period scale into training."
        )

    def test_rain_norm_without_split_boundary_uses_full_series(self, static_features, temporal_data, directed_edges):
        """Documents the (intentionally leaky, inference-only) fallback path."""
        feat, _ = static_features
        enc, _ = temporal_data
        edge_index, edge_attr = directed_edges
        ds_full = HydroGraphDataset(
            static_features=feat, rainfall=enc.rainfall, labels=enc.labels,
            edge_index=edge_index, short_seq_len=SHORT_SEQ, long_seq_len=LONG_SEQ,
            lead_times=LEAD_TIMES, edge_attr=edge_attr,   # no rain_norm_fit_end_t
        )
        expected = max(float(np.percentile(enc.rainfall[enc.rainfall > 0], 95)), 1.0)
        assert ds_full._rain_norm == pytest.approx(expected)

    def test_trainer_smoke(self, cfg, dataset, split, tmp_path):
        """Quick 2-epoch smoke test to verify training doesn't crash."""
        train_idx, val_idx, test_idx = split
        model = build_model(cfg)

        tmp_resolved = Path(str(tmp_path)).resolve()
        cfg.paths.best_checkpoint = str(tmp_resolved / "best.pt")
        cfg.paths.last_checkpoint = str(tmp_resolved / "last.pt")
        cfg.training.epochs = 2
        cfg.training.early_stopping_patience = 2
        cfg.training.batch_size = min(cfg.training.batch_size, dataset.N)

        trainer = Trainer(model, cfg, base_dir=tmp_resolved)
        history = trainer.train(dataset, train_idx[:5], val_idx[:3])
        assert "train_loss" in history
        assert len(history["train_loss"]) >= 1

        metrics = trainer.evaluate(dataset, test_idx[:3])
        assert "f1_lead0" in metrics
        assert "auroc_lead0" in metrics
        assert "ece_lead0" in metrics
        assert 0.0 <= metrics["auroc_lead0"] <= 1.0
        assert 0.0 <= metrics["ece_lead0"] <= 1.0

    def test_compute_metrics_keys(self):
        preds = torch.tensor([0.9, 0.1, 0.6, 0.3])
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        m = _compute_metrics(preds, labels, threshold=0.5)
        for key in ["f1", "precision", "recall", "auroc", "aucpr", "ece", "base_rate", "csi", "far", "pod", "brier"]:
            assert key in m, f"Missing metric: {key}"


# ─── ECE / Calibration ─────────────────────────────────────────────────────────

class TestExpectedCalibrationError:
    def test_ece_zero_for_perfect_calibration(self):
        # Every bin's mean prediction exactly matches its observed positive rate.
        preds = np.array([0.1] * 100 + [0.9] * 100)
        labels = np.array([0] * 90 + [1] * 10 + [1] * 90 + [0] * 10)
        ece = _expected_calibration_error(preds, labels, n_bins=10)
        assert ece == pytest.approx(0.0, abs=1e-6)

    def test_ece_large_for_confident_wrong_predictions(self):
        preds = np.array([0.95] * 100)
        labels = np.array([0] * 100)   # confidently wrong every time
        ece = _expected_calibration_error(preds, labels, n_bins=10)
        assert ece > 0.9

    def test_ece_empty_input(self):
        assert _expected_calibration_error(np.array([]), np.array([])) == 0.0


# ─── Baselines ──────────────────────────────────────────────────────────────────

class TestPersistenceBaseline:
    def test_persistence_predicts_current_state(self, dataset, split):
        train_idx, val_idx, test_idx = split
        baseline = PersistenceBaseline()
        preds, labels = baseline.predict_proba(dataset, test_idx[:5], lead_idx=0)
        h = dataset.lead_times[0]
        expected_preds = np.concatenate([dataset.labels[int(t), :] for t in test_idx[:5]])
        expected_labels = np.concatenate([
            dataset.labels[min(int(t) + h, dataset.T - 1), :] for t in test_idx[:5]
        ])
        np.testing.assert_allclose(preds, expected_preds)
        np.testing.assert_allclose(labels, expected_labels)
        assert set(np.unique(preds)).issubset({0.0, 1.0})


# ─── Phase 6: Inference & Visualization ───────────────────────────────────────

class TestPhase6Inference:
    def test_risk_category_mapping(self):
        assert _risk_category(0.10) == ("Low Risk",       "#2ECC71")
        assert _risk_category(0.40) == ("Moderate Risk",  "#F1C40F")
        assert _risk_category(0.60) == ("High Risk",      "#E67E22")
        assert _risk_category(0.85) == ("Very High Risk", "#E74C3C")

    def test_inference_output_shape(self, cfg, small_graph, dataset, split, tmp_path):
        G, gdf, _, _ = small_graph
        _, _, test_idx = split
        model = build_model(cfg)
        model.eval()

        tmp_resolved = Path(str(tmp_path)).resolve()
        engine = InferenceEngine(model, cfg, base_dir=tmp_resolved)
        df_pred = engine.run(dataset, test_idx[:5], gdf, run_uncertainty=False)

        assert len(df_pred) == G.number_of_nodes()
        for col in ["lat", "lon", "flood_prob", "flood_binary", "flood_prob_lead1h"]:
            assert col in df_pred.columns, f"Missing column: {col}"

        assert df_pred["flood_prob"].between(0, 1).all(), "Flood probs out of [0,1]"
        assert df_pred["flood_binary"].isin([0, 1]).all(), "Non-binary flood labels"

    def test_calibration_plot_returns_ece(self, cfg, dataset, split, tmp_path):
        _, _, test_idx = split
        model = build_model(cfg)
        model.eval()
        tmp_resolved = Path(str(tmp_path)).resolve()
        engine = InferenceEngine(model, cfg, base_dir=tmp_resolved)
        path, ece = engine.plot_calibration(dataset, test_idx[:5])
        assert Path(path).exists()
        assert 0.0 <= ece <= 1.0

    def test_static_map_generation(self, cfg, small_graph, dataset, split, tmp_path):
        import os
        G, gdf, _, _ = small_graph
        _, _, test_idx = split
        model = build_model(cfg)
        tmp_resolved = Path(str(tmp_path)).resolve()
        engine = InferenceEngine(model, cfg, base_dir=tmp_resolved)

        df_pred = engine.run(dataset, test_idx[:5], gdf, run_uncertainty=False)
        map_path = engine.create_static_map(df_pred, str(tmp_resolved / "test_map.png"))
        assert os.path.isfile(map_path), f"Static map PNG not created at {map_path}"
        assert os.path.getsize(map_path) > 10_000, "Map file suspiciously small"


# ─── Integration Test ─────────────────────────────────────────────────────────

class TestIntegrationPipeline:
    """End-to-end smoke test of the complete pipeline, phases 1-6 in sequence."""

    def test_full_pipeline_smoke(self, cfg, tmp_path):
        import networkx as nx
        tmp_path = Path(str(tmp_path)).resolve()

        # Phase 1
        gc = GraphConstructor(bbox=SMALL_BBOX, use_synthetic_fallback=True)
        G, gdf = gc.build()
        assert G.number_of_nodes() >= 30
        edge_features = gc.edge_features

        # Phase 2
        fe = FeatureEngineer(use_synthetic=True)
        feat, df_feat = fe.compute_features(G, gdf, edge_features)
        assert feat.shape[1] == STATIC_DIM

        edge_features = gc.refine_edge_elevations(G, feat[:, 0])
        edge_index, edge_attr = orient_drainage_edges(G, edge_features)

        # Phase 3 — use flood-peak window to ensure nonzero labels (event_start
        # must be the true Nov 1 event start; profile hours are relative to it)
        enc = TemporalEncoder(
            short_seq_len=SHORT_SEQ, long_seq_len=LONG_SEQ, lead_times=LEAD_TIMES,
            event_start="2015-11-01T00:00:00",
            event_end="2015-12-03T00:00:00",
            use_synthetic=True,
        )
        enc.encode(feat, node_lons=gdf["lon"].values, node_lats=gdf["lat"].values, edge_index=edge_index)
        assert enc.rainfall is not None

        lookback = max(SHORT_SEQ, LONG_SEQ * 2)
        train_idx, val_idx, test_idx = get_chronological_split(enc.T, lookback=lookback, max_lead=MAX_LEAD)
        train_end_t = int(train_idx.max()) + 1

        ds = HydroGraphDataset(
            feat, enc.rainfall, enc.labels, edge_index,
            short_seq_len=SHORT_SEQ, long_seq_len=LONG_SEQ, lead_times=LEAD_TIMES,
            edge_attr=edge_attr, rain_norm_fit_end_t=train_end_t,
        )

        # Phase 4
        model = build_model(cfg)

        # Phase 5 (mini-train)
        cfg.training.epochs = 2
        cfg.training.early_stopping_patience = 2
        cfg.paths.best_checkpoint = str(tmp_path / "best.pt")
        cfg.paths.last_checkpoint = str(tmp_path / "last.pt")

        trainer = Trainer(model, cfg, base_dir=tmp_path)
        history = trainer.train(ds, train_idx[:4], val_idx[:2])
        assert len(history["train_loss"]) >= 1

        # Phase 6
        engine = InferenceEngine(model, cfg, base_dir=tmp_path)
        df_pred = engine.run(ds, test_idx[:3], gdf, run_uncertainty=False)
        map_png = engine.create_static_map(df_pred, str(tmp_path / "final_map.png"))
        assert Path(map_png).exists()
        pred_csv = engine.save_predictions(df_pred, str(tmp_path / "predictions.csv"))
        assert Path(pred_csv).exists()

        print(f"\n[OK] Full pipeline smoke test passed. Nodes: {G.number_of_nodes()}, "
              f"Flood rate: {df_pred['flood_binary'].mean():.1%}")
