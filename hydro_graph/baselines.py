"""
Baselines - Comparison Models for DS-STGAT Evaluation
=======================================================
Implements 4 baseline models for rigorous ablation and comparison:

  1. RandomForestBaseline   — static features + 6hr rainfall (no graph, no temporal)
  2. LSTMOnlyBaseline       — GRU over rainfall sequence (temporal, no graph)
  3. GCNBaseline            — GCN + GRU (graph but no attention, no dual-scale)
  4. GraphSAGEv1Baseline    — GraphSAGE + GRU v1 (graph, single-scale, no edge features)

These are used to quantify the contribution of each DS-STGAT component.

Usage:
    from hydro_graph.baselines import run_all_baselines
    results = run_all_baselines(dataset, train_idx, val_idx, test_idx, cfg, base_dir)
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

try:
    from torch_geometric.nn import GCNConv, SAGEConv
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False

from .phase5_training import HydroGraphDataset, _compute_metrics, _iter_mini_batches
from .phase4_model import FocalTverskyLoss

logger = logging.getLogger(__name__)

LEAD_IDX = 0   # Evaluate baselines on 1-hr lead time for fair comparison


# ─── 1. Random Forest ────────────────────────────────────────────────────────

class RandomForestBaseline:
    """
    Random Forest classifier on concatenated static + last 6hr rainfall.
    Node-level, no spatial propagation, no temporal memory.
    """

    def __init__(self, n_estimators: int = 200, max_depth: int = 12, n_jobs: int = -1) -> None:
        from sklearn.ensemble import RandomForestClassifier
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            class_weight="balanced", n_jobs=n_jobs, random_state=42,
        )

    def fit(self, dataset: HydroGraphDataset, train_idx: np.ndarray) -> None:
        X, Y = [], []
        for t in train_idx:
            data = dataset.get_snapshot(int(t))
            X.append(data.x.numpy())
            Y.append(data.y[:, LEAD_IDX].numpy())
        X_all = np.vstack(X)
        Y_all = np.concatenate(Y).astype(int)
        logger.info("RF training: X=%s, flood_rate=%.2f%%", X_all.shape, 100 * Y_all.mean())
        self.clf.fit(X_all, Y_all)

    def predict_proba(self, dataset: HydroGraphDataset, idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        preds_all, labels_all = [], []
        for t in idx:
            data = dataset.get_snapshot(int(t))
            proba = self.clf.predict_proba(data.x.numpy())[:, 1]
            preds_all.append(proba)
            labels_all.append(data.y[:, LEAD_IDX].numpy())
        return np.concatenate(preds_all), np.concatenate(labels_all)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.clf, f)

    @classmethod
    def load(cls, path: str) -> "RandomForestBaseline":
        obj = cls.__new__(cls)
        with open(path, "rb") as f:
            obj.clf = pickle.load(f)
        return obj


# ─── 2. LSTM-Only Baseline ────────────────────────────────────────────────────

class LSTMOnlyModel(nn.Module):
    """GRU over 6hr rainfall + static features. No graph propagation."""

    def __init__(self, static_dim: int = 16, seq_len: int = 6, hidden: int = 128, dropout: float = 0.25) -> None:
        super().__init__()
        self.static_enc = nn.Sequential(
            nn.Linear(static_dim, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 32),
        )
        self.gru = nn.GRU(1, hidden, num_layers=2, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden + 32, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, static_dim + seq_len + long_seq_len]
        x_static = x[:, :16]                       # [N, 16]
        x_rain = x[:, 16:22].unsqueeze(-1)         # [N, 6, 1]
        s = self.static_enc(x_static)              # [N, 32]
        _, h_n = self.gru(x_rain)                  # [2, N, 128]
        h = h_n[-1]                                # [N, 128]
        return self.head(torch.cat([h, s], dim=-1)).squeeze(-1)   # [N]


class LSTMOnlyBaseline:
    def __init__(self, cfg, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.model = LSTMOnlyModel(
            static_dim=cfg.model.static_dim,
            seq_len=cfg.model.short_seq_len,
        ).to(device)
        self.criterion = FocalTverskyLoss(
            alpha=cfg.training.tversky_alpha,
            beta=cfg.training.tversky_beta,
            gamma=cfg.training.tversky_gamma,
        )

    def fit(self, dataset: HydroGraphDataset, train_idx: np.ndarray, epochs: int = 50) -> None:
        opt = AdamW(self.model.parameters(), lr=5e-4, weight_decay=1e-4)
        self.model.train()
        for ep in range(epochs):
            total_loss = 0.0
            perm = np.random.permutation(len(train_idx))
            for t in train_idx[perm]:
                data = dataset.get_snapshot(int(t))
                x = data.x.to(self.device)
                y = data.y[:, LEAD_IDX].to(self.device)
                opt.zero_grad()
                out = self.model(x)
                loss = self.criterion(out, y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                total_loss += loss.item()
            if (ep + 1) % 10 == 0:
                logger.info("LSTM-only epoch %d/%d  loss=%.4f", ep + 1, epochs, total_loss / len(train_idx))

    @torch.no_grad()
    def predict_proba(self, dataset: HydroGraphDataset, idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        preds_all, labels_all = [], []
        for t in idx:
            data = dataset.get_snapshot(int(t))
            x = data.x.to(self.device)
            out = self.model(x).cpu().numpy()
            preds_all.append(out)
            labels_all.append(data.y[:, LEAD_IDX].numpy())
        return np.concatenate(preds_all), np.concatenate(labels_all)


# ─── 3. GCN Baseline ─────────────────────────────────────────────────────────

class GCNBaselineModel(nn.Module):
    """2-layer GCN + single-scale GRU. No attention, no edge features."""

    def __init__(self, in_dim: int = 34, hidden: int = 128, dropout: float = 0.25) -> None:
        super().__init__()
        self.gru = nn.GRU(1, 64, num_layers=1, batch_first=True)
        self.static_enc = nn.Sequential(nn.Linear(16, 32), nn.GELU())
        fused = 64 + 32  # 96

        if PYG_AVAILABLE:
            self.gcn1 = GCNConv(fused, hidden)
            self.gcn2 = GCNConv(hidden, 64)
        else:
            self.gcn1 = nn.Linear(fused, hidden)
            self.gcn2 = nn.Linear(hidden, 64)

        self.head = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x_static = x[:, :16]
        x_rain = x[:, 16:22].unsqueeze(-1)
        s = self.static_enc(x_static)
        _, h_n = self.gru(x_rain)
        h = torch.cat([h_n[-1], s], dim=-1)

        if PYG_AVAILABLE:
            h = self.drop(torch.relu(self.gcn1(h, edge_index)))
            h = self.drop(torch.relu(self.gcn2(h, edge_index)))
        else:
            h = self.drop(torch.relu(self.gcn1(h)))
            h = self.drop(torch.relu(self.gcn2(h)))

        return self.head(h).squeeze(-1)


class GCNBaseline:
    def __init__(self, cfg, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.model = GCNBaselineModel().to(device)
        self.criterion = FocalTverskyLoss(
            alpha=cfg.training.tversky_alpha,
            beta=cfg.training.tversky_beta,
            gamma=cfg.training.tversky_gamma,
        )

    def fit(self, dataset: HydroGraphDataset, train_idx: np.ndarray, epochs: int = 50) -> None:
        opt = AdamW(self.model.parameters(), lr=5e-4, weight_decay=1e-4)
        self.model.train()
        for ep in range(epochs):
            total_loss = 0.0
            for t in np.random.permutation(train_idx):
                data = dataset.get_snapshot(int(t))
                x = data.x.to(self.device)
                ei = data.edge_index.to(self.device)
                y = data.y[:, LEAD_IDX].to(self.device)
                opt.zero_grad()
                out = self.model(x, ei)
                loss = self.criterion(out, y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                total_loss += loss.item()
            if (ep + 1) % 10 == 0:
                logger.info("GCN epoch %d/%d  loss=%.4f", ep + 1, epochs, total_loss / len(train_idx))

    @torch.no_grad()
    def predict_proba(self, dataset: HydroGraphDataset, idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        preds_all, labels_all = [], []
        for t in idx:
            data = dataset.get_snapshot(int(t))
            x = data.x.to(self.device)
            ei = data.edge_index.to(self.device)
            out = self.model(x, ei).cpu().numpy()
            preds_all.append(out)
            labels_all.append(data.y[:, LEAD_IDX].numpy())
        return np.concatenate(preds_all), np.concatenate(labels_all)


# ─── 4. GraphSAGE v1 Baseline ────────────────────────────────────────────────

class SAGEv1Model(nn.Module):
    """GraphSAGE + single-scale GRU (replicates original v1 architecture)."""

    def __init__(self, in_dim: int = 34, sage_hidden: int = 128, dropout: float = 0.25) -> None:
        super().__init__()
        self.static_enc = nn.Sequential(nn.Linear(16, 32), nn.GELU())
        self.gru = nn.GRU(1, 64, num_layers=2, batch_first=True, dropout=dropout)
        fused = 64 + 32  # 96

        if PYG_AVAILABLE:
            self.sage1 = SAGEConv(fused, sage_hidden, aggr="mean")
            self.sage2 = SAGEConv(sage_hidden, sage_hidden, aggr="mean")
            self.sage3 = SAGEConv(sage_hidden, 64, aggr="mean")
        else:
            self.sage1 = nn.Linear(fused, sage_hidden)
            self.sage2 = nn.Linear(sage_hidden, sage_hidden)
            self.sage3 = nn.Linear(sage_hidden, 64)

        self.head = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x_static = x[:, :16]
        x_rain = x[:, 16:22].unsqueeze(-1)
        s = self.static_enc(x_static)
        _, h_n = self.gru(x_rain)
        h = torch.cat([h_n[-1], s], dim=-1)

        def _apply(layer, inp, ei):
            if PYG_AVAILABLE:
                return torch.relu(self.drop(layer(inp, ei)))
            return torch.relu(self.drop(layer(inp)))

        h = _apply(self.sage1, h, edge_index)
        h = _apply(self.sage2, h, edge_index)
        h = _apply(self.sage3, h, edge_index)
        return self.head(h).squeeze(-1)


class SAGEv1Baseline:
    def __init__(self, cfg, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.model = SAGEv1Model().to(device)
        self.criterion = FocalTverskyLoss(
            alpha=cfg.training.tversky_alpha,
            beta=cfg.training.tversky_beta,
            gamma=cfg.training.tversky_gamma,
        )

    def fit(self, dataset: HydroGraphDataset, train_idx: np.ndarray, epochs: int = 50) -> None:
        opt = AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.model.train()
        for ep in range(epochs):
            total_loss = 0.0
            for t in np.random.permutation(train_idx):
                data = dataset.get_snapshot(int(t))
                x = data.x.to(self.device)
                ei = data.edge_index.to(self.device)
                y = data.y[:, LEAD_IDX].to(self.device)
                opt.zero_grad()
                out = self.model(x, ei)
                loss = self.criterion(out, y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                total_loss += loss.item()
            if (ep + 1) % 10 == 0:
                logger.info("SAGE-v1 epoch %d/%d  loss=%.4f", ep + 1, epochs, total_loss / len(train_idx))

    @torch.no_grad()
    def predict_proba(self, dataset: HydroGraphDataset, idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        preds_all, labels_all = [], []
        for t in idx:
            data = dataset.get_snapshot(int(t))
            x = data.x.to(self.device)
            ei = data.edge_index.to(self.device)
            out = self.model(x, ei).cpu().numpy()
            preds_all.append(out)
            labels_all.append(data.y[:, LEAD_IDX].numpy())
        return np.concatenate(preds_all), np.concatenate(labels_all)


# ─── Master Baseline Runner ───────────────────────────────────────────────────

def run_all_baselines(
    dataset: HydroGraphDataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    cfg,
    base_dir: Path,
) -> Dict[str, Dict[str, float]]:
    """
    Train and evaluate all 4 baseline models. Returns per-model metrics.
    Uses the 1-hr lead time for comparison (LEAD_IDX=0).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    thr = cfg.training.threshold
    epochs = cfg.training.baseline_epochs
    results: Dict[str, Dict[str, float]] = {}

    # ── 1. Random Forest ─────────────────────────────────────────────────────
    logger.info("--- Baseline 1: Random Forest ---")
    try:
        rf = RandomForestBaseline()
        rf.fit(dataset, train_idx)
        rf_path = str(base_dir / cfg.paths.baseline_rf)
        rf.save(rf_path)
        preds, labs = rf.predict_proba(dataset, test_idx)
        results["random_forest"] = _compute_metrics(
            torch.tensor(preds), torch.tensor(labs), thr
        )
        logger.info("RF  F1=%.4f  AUC=%.4f  CSI=%.4f",
                    results["random_forest"]["f1"],
                    results["random_forest"]["auroc"],
                    results["random_forest"]["csi"])
    except Exception as e:
        logger.warning("RF baseline failed: %s", e)
        results["random_forest"] = {}

    # ── 2. LSTM-Only ─────────────────────────────────────────────────────────
    logger.info("--- Baseline 2: LSTM-Only ---")
    try:
        lstm = LSTMOnlyBaseline(cfg, device)
        lstm.fit(dataset, train_idx, epochs=epochs)
        preds, labs = lstm.predict_proba(dataset, test_idx)
        results["lstm_only"] = _compute_metrics(
            torch.tensor(preds), torch.tensor(labs), thr
        )
        logger.info("LSTM F1=%.4f  AUC=%.4f  CSI=%.4f",
                    results["lstm_only"]["f1"],
                    results["lstm_only"]["auroc"],
                    results["lstm_only"]["csi"])
    except Exception as e:
        logger.warning("LSTM baseline failed: %s", e)
        results["lstm_only"] = {}

    # ── 3. GCN ───────────────────────────────────────────────────────────────
    logger.info("--- Baseline 3: GCN + GRU ---")
    try:
        gcn = GCNBaseline(cfg, device)
        gcn.fit(dataset, train_idx, epochs=epochs)
        preds, labs = gcn.predict_proba(dataset, test_idx)
        results["gcn_gru"] = _compute_metrics(
            torch.tensor(preds), torch.tensor(labs), thr
        )
        logger.info("GCN  F1=%.4f  AUC=%.4f  CSI=%.4f",
                    results["gcn_gru"]["f1"],
                    results["gcn_gru"]["auroc"],
                    results["gcn_gru"]["csi"])
    except Exception as e:
        logger.warning("GCN baseline failed: %s", e)
        results["gcn_gru"] = {}

    # ── 4. GraphSAGE v1 ──────────────────────────────────────────────────────
    logger.info("--- Baseline 4: GraphSAGE-v1 + GRU ---")
    try:
        sage = SAGEv1Baseline(cfg, device)
        sage.fit(dataset, train_idx, epochs=epochs)
        preds, labs = sage.predict_proba(dataset, test_idx)
        results["sage_v1_gru"] = _compute_metrics(
            torch.tensor(preds), torch.tensor(labs), thr
        )
        logger.info("SAGEv1 F1=%.4f  AUC=%.4f  CSI=%.4f",
                    results["sage_v1_gru"]["f1"],
                    results["sage_v1_gru"]["auroc"],
                    results["sage_v1_gru"]["csi"])
    except Exception as e:
        logger.warning("SAGEv1 baseline failed: %s", e)
        results["sage_v1_gru"] = {}

    # Save results
    out_path = base_dir / cfg.paths.baselines_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(
            {k: {mk: float(mv) if isinstance(mv, (float, np.floating)) else mv
                 for mk, mv in v.items()} for k, v in results.items()},
            fh, indent=2,
        )
    logger.info("Baseline results saved -> %s", out_path)
    return results
