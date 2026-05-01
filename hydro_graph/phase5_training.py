"""
Phase 5 - Training & Evaluation Pipeline (v2)
=============================================
Complete training loop for DS-STGAT with:

  - MultiLeadFocalTverskyLoss (alpha=0.3, beta=0.7, gamma=0.75)
  - Multi-horizon targets (1, 3, 6, 12 hr lead times)
  - k_hop_subgraph mini-batching with edge_attr support
  - Chronological train/val/test split (no temporal leakage)
  - Comprehensive metrics: F1, Precision, Recall, AUC-ROC, AUC-PR, CSI, FAR, POD, Brier
  - Early stopping on val F1 (lead=1hr)
  - Cosine annealing LR schedule
  - Gradient clipping
  - Best-checkpoint saving
  - Per-lead-time evaluation
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from torch_geometric.data import Data
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False

try:
    from torch_geometric.loader import NeighborLoader as _NL
    import torch_geometric.data as _tgd
    _probe = _tgd.Data(x=torch.zeros(4, 2), edge_index=torch.tensor([[0, 1], [1, 0]]), num_nodes=4)
    _pl = _NL(_probe, num_neighbors=[1], batch_size=2)
    next(iter(_pl))
    NEIGHBOR_LOADER_OK = True
    NeighborLoader = _NL
except Exception:
    NEIGHBOR_LOADER_OK = False
    NeighborLoader = None

try:
    from torchmetrics.classification import BinaryAUROC, BinaryF1Score, BinaryPrecision, BinaryRecall
    from torchmetrics.classification import BinaryAveragePrecision
    TORCHMETRICS_OK = True
except ImportError:
    TORCHMETRICS_OK = False

from .phase4_model import DualScaleSTGAT, MultiLeadFocalTverskyLoss

logger = logging.getLogger(__name__)


# ─── Dataset ──────────────────────────────────────────────────────────────────

class HydroGraphDataset:
    """
    Holds static features, temporal data, graph topology, and edge features.

    get_snapshot(t) returns a PyG Data object:
      x         : [N, 34]  static(16) + short_rain(6) + long_rain(12)
      edge_index : [2, E]
      edge_attr  : [E, 4]  physics-informed edge features
      y          : [N, 4]  multi-lead flood labels (no leakage: future states)
    """

    def __init__(
        self,
        static_features: np.ndarray,     # [N, 16]
        rainfall: np.ndarray,             # [T, N]
        labels: np.ndarray,               # [T, N]
        edge_index: np.ndarray,           # [2, E]
        short_seq_len: int = 6,
        long_seq_len: int = 12,
        lead_times: Optional[List[int]] = None,
        edge_attr: Optional[np.ndarray] = None,   # [E, 4]
    ) -> None:
        self.N = static_features.shape[0]
        self.T = rainfall.shape[0]
        self.short_seq_len = short_seq_len
        self.long_seq_len = long_seq_len
        self.lead_times = lead_times or [1, 3, 6, 12]
        self.max_lead = max(self.lead_times)

        self.static_t = torch.tensor(static_features, dtype=torch.float32)
        self.rainfall = rainfall.astype(np.float32)
        self.labels = labels.astype(np.float32)
        self.edge_index_t = torch.tensor(edge_index, dtype=torch.long)
        self.edge_attr_t = (
            torch.tensor(edge_attr, dtype=torch.float32)
            if edge_attr is not None
            else torch.zeros(edge_index.shape[1], 4, dtype=torch.float32)
        )

        # Global rain normaliser (95th percentile for robustness)
        self._rain_norm = max(float(np.percentile(rainfall[rainfall > 0], 95)), 1.0) \
            if (rainfall > 0).any() else 1.0

    def get_snapshot(self, t: int):
        """Build PyG Data for time step t."""
        long_start = max(0, t - self.long_seq_len * 2)
        long_indices = np.linspace(long_start, t - 1, self.long_seq_len, dtype=int)
        long_indices = np.clip(long_indices, 0, self.T - 1)

        short_start = max(0, t - self.short_seq_len)
        short_rain = self.rainfall[short_start:t, :]           # [<=6, N]
        if short_rain.shape[0] < self.short_seq_len:
            pad = np.zeros((self.short_seq_len - short_rain.shape[0], self.N), dtype=np.float32)
            short_rain = np.vstack([pad, short_rain])          # [6, N]

        long_rain = self.rainfall[long_indices, :]             # [12, N]

        short_norm = torch.tensor(short_rain.T / self._rain_norm, dtype=torch.float32)  # [N,6]
        long_norm = torch.tensor(long_rain.T / self._rain_norm, dtype=torch.float32)    # [N,12]

        x = torch.cat([self.static_t, short_norm, long_norm], dim=1)  # [N, 34]

        # Multi-lead labels (future states — no leakage)
        ys = []
        for h in self.lead_times:
            t_future = min(t + h, self.T - 1)
            ys.append(self.labels[t_future, :])
        y = torch.tensor(np.stack(ys, axis=-1), dtype=torch.float32)  # [N, 4]

        if PYG_AVAILABLE:
            from torch_geometric.data import Data as PyGData
            return PyGData(
                x=x,
                edge_index=self.edge_index_t,
                edge_attr=self.edge_attr_t,
                y=y,
                num_nodes=self.N,
            )
        else:
            class _D:
                pass
            d = _D()
            d.x, d.edge_index, d.edge_attr, d.y, d.num_nodes = x, self.edge_index_t, self.edge_attr_t, y, self.N
            return d

    def __len__(self) -> int:
        return max(0, self.T - self.short_seq_len - self.max_lead)


# ─── Trainer ──────────────────────────────────────────────────────────────────

class Trainer:
    """End-to-end training and evaluation orchestrator for DS-STGAT."""

    def __init__(
        self,
        model: DualScaleSTGAT,
        cfg,
        base_dir: Optional[Path] = None,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.base = base_dir or Path.cwd()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        tr = cfg.training
        self.criterion = MultiLeadFocalTverskyLoss(
            n_leads=cfg.model.n_lead_times,
            lead_weights=tr.lead_weights,
            alpha=tr.tversky_alpha,
            beta=tr.tversky_beta,
            gamma=tr.tversky_gamma,
        )
        self.optimizer = AdamW(
            model.parameters(), lr=tr.lr, weight_decay=tr.weight_decay
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=tr.epochs, eta_min=tr.lr_min
        )

        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
            "val_f1_lead1": [], "val_auroc_lead1": [],
            "val_f1_lead3": [], "val_f1_lead6": [], "val_f1_lead12": [],
        }
        self._best_val_f1 = -1.0
        self._patience_ctr = 0

        logger.info(
            "Trainer: device=%s | loss=MultiLeadFocalTversky | leads=%s",
            self.device, cfg.temporal.lead_times,
        )

    # ── Training Loop ─────────────────────────────────────────────────────────

    def train(
        self,
        dataset: HydroGraphDataset,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
    ) -> Dict[str, List[float]]:
        cfg = self.cfg.training
        logger.info(
            "Training: %d epochs | train_steps=%d | val_steps=%d",
            cfg.epochs, len(train_idx), len(val_idx),
        )

        for epoch in range(1, cfg.epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch(dataset, train_idx)
            val_metrics = self._eval_epoch(dataset, val_idx)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_f1_lead1"].append(val_metrics.get("f1_lead0", 0.0))
            self.history["val_auroc_lead1"].append(val_metrics.get("auroc_lead0", 0.0))
            self.history["val_f1_lead3"].append(val_metrics.get("f1_lead1", 0.0))
            self.history["val_f1_lead6"].append(val_metrics.get("f1_lead2", 0.0))
            self.history["val_f1_lead12"].append(val_metrics.get("f1_lead3", 0.0))

            self.scheduler.step()
            elapsed = time.time() - t0

            val_f1 = val_metrics.get("f1_lead0", 0.0)
            val_auc = val_metrics.get("auroc_lead0", 0.0)
            logger.info(
                "Epoch %3d/%d  loss=%.4f  val_loss=%.4f  "
                "F1@1h=%.4f  AUC@1h=%.4f  [%.1fs]",
                epoch, cfg.epochs,
                train_loss, val_metrics["loss"], val_f1, val_auc, elapsed,
            )

            self._save_checkpoint(self.base / self.cfg.paths.last_checkpoint, epoch)

            if val_f1 > self._best_val_f1:
                self._best_val_f1 = val_f1
                self._patience_ctr = 0
                self._save_checkpoint(self.base / self.cfg.paths.best_checkpoint, epoch)
                logger.info("  -> Best val F1=%.4f saved.", val_f1)
            else:
                self._patience_ctr += 1
                if self._patience_ctr >= cfg.early_stopping_patience:
                    logger.info("Early stopping at epoch %d.", epoch)
                    break

        return self.history

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, dataset: HydroGraphDataset, test_idx: np.ndarray) -> Dict[str, float]:
        """Full evaluation with per-lead-time metrics and hydrology scores."""
        metrics = self._eval_epoch(dataset, test_idx, split="test")
        logger.info("TEST results:")
        for h_idx, h in enumerate(self.cfg.temporal.lead_times):
            logger.info(
                "  Lead %2dhr: F1=%.4f  Prec=%.4f  Rec=%.4f  AUC=%.4f  CSI=%.4f  Brier=%.4f",
                h,
                metrics.get(f"f1_lead{h_idx}", 0),
                metrics.get(f"precision_lead{h_idx}", 0),
                metrics.get(f"recall_lead{h_idx}", 0),
                metrics.get(f"auroc_lead{h_idx}", 0),
                metrics.get(f"csi_lead{h_idx}", 0),
                metrics.get(f"brier_lead{h_idx}", 0),
            )
        return metrics

    # ── Private: Epoch Steps ──────────────────────────────────────────────────

    def _train_epoch(self, dataset: HydroGraphDataset, time_indices: np.ndarray) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        perm = np.random.permutation(len(time_indices))

        for t in time_indices[perm]:
            data = dataset.get_snapshot(int(t))
            loss = self._process_snapshot(data, training=True)
            total_loss += loss
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _eval_epoch(
        self,
        dataset: HydroGraphDataset,
        time_indices: np.ndarray,
        split: str = "val",
    ) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        n_leads = self.cfg.model.n_lead_times
        all_preds = [[] for _ in range(n_leads)]
        all_labels = [[] for _ in range(n_leads)]

        for t in time_indices:
            data = dataset.get_snapshot(int(t))
            for x_b, ei_b, ea_b, y_b, bs, _ in _iter_mini_batches(
                data, self.cfg.training.batch_size, self.cfg.training.num_neighbors, shuffle=False
            ):
                x_b = x_b.to(self.device)
                ei_b = ei_b.to(self.device)
                ea_b = ea_b.to(self.device) if ea_b is not None else None
                y_b = y_b.to(self.device)

                out = self.model(x_b, ei_b, ea_b)                   # [N_b, n_leads]
                preds_seed = out[:bs]                                # [bs, n_leads]
                labels_seed = y_b[:bs]                               # [bs, n_leads]

                loss = self.criterion(preds_seed, labels_seed)
                total_loss += loss.item()
                n_batches += 1

                for h in range(n_leads):
                    all_preds[h].append(preds_seed[:, h].cpu())
                    all_labels[h].append(labels_seed[:, h].cpu())

        metrics = {"loss": total_loss / max(n_batches, 1)}
        for h in range(n_leads):
            preds_h = torch.cat(all_preds[h])
            labels_h = torch.cat(all_labels[h])
            m = _compute_metrics(preds_h, labels_h, self.cfg.training.threshold)
            for k, v in m.items():
                metrics[f"{k}_lead{h}"] = v

        return metrics

    def _process_snapshot(self, data, training: bool) -> float:
        losses = []
        for x_b, ei_b, ea_b, y_b, bs, _ in _iter_mini_batches(
            data, self.cfg.training.batch_size, self.cfg.training.num_neighbors, shuffle=True
        ):
            x_b = x_b.to(self.device)
            ei_b = ei_b.to(self.device)
            ea_b = ea_b.to(self.device) if ea_b is not None else None
            y_b = y_b.to(self.device)

            if training:
                self.optimizer.zero_grad()
            out = self.model(x_b, ei_b, ea_b)
            loss = self.criterion(out[:bs], y_b[:bs])
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            losses.append(loss.item())

        return float(np.mean(losses)) if losses else 0.0

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_checkpoint(self, path: Path, epoch: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_f1": self._best_val_f1,
            "history": self.history,
        }, path)

    def load_best_checkpoint(self) -> int:
        path = self.base / self.cfg.paths.best_checkpoint
        if not path.exists():
            logger.warning("No checkpoint at %s", path)
            return 0
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Checkpoint loaded (epoch=%d, val_F1=%.4f)", ckpt["epoch"], ckpt["best_val_f1"])
        return ckpt["epoch"]

    def save_metrics(self, metrics: Dict, path: Optional[Path] = None) -> None:
        p = path or (self.base / self.cfg.paths.metrics_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as fh:
            json.dump({k: float(v) if isinstance(v, (np.floating, float)) else v
                       for k, v in metrics.items()}, fh, indent=2)
        logger.info("Metrics saved -> %s", p)


# ─── Metrics ──────────────────────────────────────────────────────────────────

def _compute_metrics(
    preds: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.50,
) -> Dict[str, float]:
    """
    Comprehensive binary classification metrics including hydrological scores.

    Standard: F1, Precision, Recall, AUC-ROC, AUC-PR, Brier Score
    Hydrology: CSI (Critical Success Index), FAR, POD (= Recall)
    """
    preds_np = preds.numpy().astype(np.float64)
    labels_np = labels.numpy().astype(int)
    pred_bin = (preds_np >= threshold).astype(int)

    TP = int(((pred_bin == 1) & (labels_np == 1)).sum())
    FP = int(((pred_bin == 1) & (labels_np == 0)).sum())
    FN = int(((pred_bin == 0) & (labels_np == 1)).sum())
    TN = int(((pred_bin == 0) & (labels_np == 0)).sum())

    eps = 1e-8
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)   # = POD
    f1 = 2 * precision * recall / (precision + recall + eps)
    far = FP / (TP + FP + eps)
    csi = TP / (TP + FP + FN + eps)

    # Brier score (MSE of probabilities vs. binary labels)
    brier = float(np.mean((preds_np - labels_np.astype(np.float64))**2))

    # AUC scores
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        if labels_np.sum() > 0 and labels_np.sum() < len(labels_np):
            auroc = float(roc_auc_score(labels_np, preds_np))
            aucpr = float(average_precision_score(labels_np, preds_np))
        else:
            auroc = 0.5
            aucpr = float(labels_np.mean())
    except Exception:
        auroc = 0.5
        aucpr = 0.0

    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "auroc": float(auroc),
        "aucpr": float(aucpr),
        "csi": float(csi),
        "far": float(far),
        "pod": float(recall),
        "brier": float(brier),
        "tp": TP, "fp": FP, "fn": FN, "tn": TN,
    }


# ─── Inductive Mini-Batch Helper (edge_attr aware) ───────────────────────────

def _iter_mini_batches(
    data,
    batch_size: int,
    num_hops_list: List[int],
    shuffle: bool,
):
    """
    Yield (x_sub, edge_index_sub, edge_attr_sub, y_sub, seed_count, seed_ids).

    Handles edge_attr throughout k_hop_subgraph batching.
    Falls back to full-graph if no k_hop available.
    """
    N = data.num_nodes
    x = data.x
    ei = data.edge_index
    ea = getattr(data, "edge_attr", None)
    y = data.y

    # ── Strategy 1: NeighborLoader ────────────────────────────────────────────
    if NEIGHBOR_LOADER_OK and NeighborLoader is not None:
        loader = NeighborLoader(data, num_neighbors=num_hops_list, batch_size=batch_size, shuffle=shuffle)
        for batch in loader:
            bs = batch.batch_size
            seed_ids = batch.n_id[:bs]
            ea_b = getattr(batch, "edge_attr", None)
            yield batch.x, batch.edge_index, ea_b, batch.y, bs, seed_ids
        return

    # ── Strategy 2: k_hop_subgraph ────────────────────────────────────────────
    try:
        from torch_geometric.utils import k_hop_subgraph
        num_hops = len(num_hops_list)
        perm = torch.randperm(N) if shuffle else torch.arange(N)

        for start in range(0, N, batch_size):
            seed_nodes = perm[start:start + batch_size]
            actual_bs = len(seed_nodes)

            subset, sub_ei, mapping, edge_mask = k_hop_subgraph(
                seed_nodes, num_hops=num_hops, edge_index=ei,
                relabel_nodes=True, num_nodes=N,
            )
            x_sub = x[subset]
            y_sub = y[subset]
            sub_ea = ea[edge_mask] if ea is not None else None

            # Reorder: put seed nodes first
            n_sub = subset.shape[0]
            mask_seed = torch.zeros(n_sub, dtype=torch.bool)
            mask_seed[mapping] = True
            seed_ord = mapping
            other_ord = torch.where(~mask_seed)[0]
            reorder = torch.cat([seed_ord, other_ord])

            inv = torch.zeros(n_sub, dtype=torch.long)
            inv[reorder] = torch.arange(n_sub)
            re_ei = inv[sub_ei]

            yield (
                x_sub[reorder],
                re_ei,
                sub_ea,   # edge_attr order unchanged by node reordering
                y_sub[reorder],
                actual_bs,
                seed_nodes,
            )
        return

    except Exception as exc:
        logger.debug("k_hop batching failed (%s); using full-graph", exc)

    # ── Strategy 3: Full-graph ────────────────────────────────────────────────
    yield x, ei, ea, y, N, torch.arange(N)
