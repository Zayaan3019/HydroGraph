"""
Phase 6 - Inference & Geospatial Mapping (v2)
=============================================
Runs DS-STGAT on test data and produces publication-quality outputs:

  1. node_predictions.csv — lat, lon, flood_prob (per lead), uncertainty, flood_binary
  2. flood_risk_map.html  — interactive Folium map with multi-lead layers
  3. flood_risk_map.png   — 2x2 panel: mean/peak probs at 1hr and 6hr leads
  4. uncertainty_map.png  — epistemic uncertainty map (MC Dropout std)
  5. training_curves.png  — loss/F1/AUC history per lead time
  6. calibration_curve.png— reliability diagram (calibration quality)

Novel outputs vs. v1:
  - Multi-lead visualisation: shows how risk evolves from 1hr -> 12hr ahead
  - Uncertainty maps: identify where the model is confident vs. uncertain
  - Calibration curves: critical for operational early warning systems
  - Watershed-level risk aggregation
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Patch

try:
    import folium
    from folium.plugins import HeatMap
    FOLIUM_OK = True
except ImportError:
    FOLIUM_OK = False

try:
    from torch_geometric.data import Data
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False

from .phase5_training import HydroGraphDataset, _iter_mini_batches
from .phase4_model import DualScaleSTGAT

logger = logging.getLogger(__name__)


# ─── Risk Categories ──────────────────────────────────────────────────────────

RISK_THRESHOLDS = [0.0, 0.30, 0.50, 0.70, 1.01]
RISK_LABELS = ["Low Risk", "Moderate Risk", "High Risk", "Very High Risk"]
RISK_COLOURS = ["#2ECC71", "#F1C40F", "#E67E22", "#E74C3C"]


def _risk_cat(prob: float) -> Tuple[str, str]:
    for i in range(len(RISK_THRESHOLDS) - 1):
        if RISK_THRESHOLDS[i] <= prob < RISK_THRESHOLDS[i + 1]:
            return RISK_LABELS[i], RISK_COLOURS[i]
    return RISK_LABELS[-1], RISK_COLOURS[-1]


def _risk_category(prob: float) -> Tuple[str, str]:
    """Backward-compatible public name used by the test suite."""
    return _risk_cat(prob)


# ─── Inference Engine ─────────────────────────────────────────────────────────

class InferenceEngine:
    """
    Orchestrates DS-STGAT inference, uncertainty estimation, and all visualisations.
    """

    def __init__(self, model: DualScaleSTGAT, cfg, base_dir: Optional[Path] = None) -> None:
        self.model = model
        self.cfg = cfg
        self.base = base_dir or Path.cwd()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    # ── Main Inference ────────────────────────────────────────────────────────

    @torch.no_grad()
    def run(
        self,
        dataset: HydroGraphDataset,
        test_idx: np.ndarray,
        gdf_nodes: gpd.GeoDataFrame,
        run_uncertainty: bool = True,
    ) -> pd.DataFrame:
        """
        Produce per-node flood predictions across all test time steps.

        Returns DataFrame with columns:
          lat, lon,
          flood_prob_lead{h}     (mean over test period)
          flood_prob_max_lead{h} (peak over test period)
          flood_uncertainty_lead{h} (MC Dropout std, if run_uncertainty=True)
          flood_binary           (threshold applied to lead_1hr mean prob)
        """
        N = dataset.N
        n_leads = self.cfg.model.n_lead_times
        lead_times = self.cfg.temporal.lead_times

        accum = np.zeros((N, n_leads), dtype=np.float64)
        max_p = np.zeros((N, n_leads), dtype=np.float64)
        accum_unc = np.zeros((N, n_leads), dtype=np.float64)
        n_steps = 0

        logger.info("Inference over %d test steps …", len(test_idx))

        for t in test_idx:
            data = dataset.get_snapshot(int(t))
            probs = self._infer_snapshot(data)            # [N, n_leads]
            accum += probs
            max_p = np.maximum(max_p, probs)
            n_steps += 1

        mean_prob = (accum / max(n_steps, 1)).astype(np.float32)  # [N, n_leads]

        # MC Dropout uncertainty on peak step
        if run_uncertainty:
            logger.info("Running MC Dropout uncertainty estimation …")
            peak_t = test_idx[int(np.argmax([
                self._infer_snapshot(dataset.get_snapshot(int(t))).mean()
                for t in test_idx[:min(20, len(test_idx))]
            ]))]
            peak_data = dataset.get_snapshot(int(peak_t))
            unc_mean, unc_std = self._infer_with_uncertainty(peak_data)
            uncertainty = unc_std.astype(np.float32)      # [N, n_leads]
        else:
            uncertainty = np.zeros_like(mean_prob)

        # Build DataFrame
        df = pd.DataFrame(index=gdf_nodes.index)
        df["lat"] = gdf_nodes["lat"].values
        df["lon"] = gdf_nodes["lon"].values

        for h_idx, h in enumerate(lead_times):
            df[f"flood_prob_lead{h}h"] = mean_prob[:, h_idx]
            df[f"flood_prob_max_lead{h}h"] = max_p[:, h_idx].astype(np.float32)
            df[f"flood_unc_lead{h}h"] = uncertainty[:, h_idx]

        thr = self.cfg.training.threshold
        df["flood_binary"] = (df["flood_prob_lead1h"] >= thr).astype(int)
        df["flood_prob"] = df["flood_prob_lead1h"]  # default alias

        flooded_pct = df["flood_binary"].mean() * 100
        mean_prob_val = df["flood_prob_lead1h"].mean()
        mean_unc = df["flood_unc_lead1h"].mean()
        logger.info(
            "Inference done: %.1f%% flooded (lead=1hr) | mean_prob=%.4f | mean_unc=%.4f",
            flooded_pct, mean_prob_val, mean_unc,
        )
        return df

    def _infer_snapshot(self, data) -> np.ndarray:
        """Deterministic inference for one snapshot. Returns [N, n_leads]."""
        N = data.num_nodes
        n_leads = self.cfg.model.n_lead_times
        probs = np.zeros((N, n_leads), dtype=np.float32)
        counts = np.zeros(N, dtype=np.int32)

        for x_b, ei_b, ea_b, y_b, bs, seed_ids in _iter_mini_batches(
            data,
            min(self.cfg.training.batch_size * 2, 2048),
            self.cfg.training.num_neighbors,
            shuffle=False,
        ):
            x_b = x_b.to(self.device)
            ei_b = ei_b.to(self.device)
            ea_b = ea_b.to(self.device) if ea_b is not None else None
            out = self.model(x_b, ei_b, ea_b)     # [N_b, n_leads]
            probs_b = out[:bs].cpu().numpy()       # [bs, n_leads]

            if bs == N:
                return probs_b

            gids = seed_ids[:bs].cpu().numpy()
            probs[gids] += probs_b
            counts[gids] += 1

        safe = np.maximum(counts, 1)
        return (probs / safe[:, np.newaxis]).astype(np.float32)

    def _infer_with_uncertainty(self, data) -> Tuple[np.ndarray, np.ndarray]:
        """MC Dropout uncertainty for a single snapshot."""
        N = data.num_nodes
        x = data.x.to(self.device)
        ei = data.edge_index.to(self.device)
        ea = data.edge_attr.to(self.device) if hasattr(data, "edge_attr") and data.edge_attr is not None else None
        mean, std = self.model.predict_with_uncertainty(
            x, ei, ea,
            n_samples=self.cfg.training.mc_dropout_samples,
            device=self.device,
        )
        return mean.cpu().numpy(), std.cpu().numpy()

    # ── Folium Interactive Map ─────────────────────────────────────────────────

    def create_folium_map(self, df_pred: pd.DataFrame, output_path: Optional[str] = None) -> Optional[str]:
        if not FOLIUM_OK:
            logger.warning("folium not installed; skipping interactive map.")
            return None

        path = output_path or str(self.base / self.cfg.paths.risk_map_html)
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        center_lat = self.cfg.study_area.center_lat
        center_lon = self.cfg.study_area.center_lon
        lead_times = self.cfg.temporal.lead_times

        m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles="CartoDB positron")

        # Add a layer per lead time
        for h in lead_times:
            col = f"flood_prob_lead{h}h"
            if col not in df_pred.columns:
                continue
            fg = folium.FeatureGroup(name=f"Lead {h}hr Forecast", show=(h == 1))
            for _, row in df_pred.iterrows():
                if not (np.isfinite(row["lat"]) and np.isfinite(row["lon"])):
                    continue
                prob = float(row[col])
                label, colour = _risk_cat(prob)
                folium.CircleMarker(
                    location=[row["lat"], row["lon"]],
                    radius=3,
                    color=colour,
                    fill=True,
                    fill_color=colour,
                    fill_opacity=0.75,
                    weight=0,
                    tooltip=folium.Tooltip(
                        f"<b>{h}hr Flood Prob:</b> {prob:.2%}<br>"
                        f"<b>Risk Level:</b> {label}"
                    ),
                ).add_to(fg)
            fg.add_to(m)

        # Heatmap overlay (1hr lead)
        col1 = "flood_prob_lead1h"
        if col1 in df_pred.columns:
            high = df_pred[df_pred[col1] >= 0.50]
            if len(high) > 0:
                HeatMap(
                    [[r["lat"], r["lon"], r[col1]] for _, r in high.iterrows()
                     if np.isfinite(r["lat"])],
                    name="Risk Heatmap (1hr)",
                    min_opacity=0.3, max_zoom=18, radius=12, blur=15,
                    gradient={0.3: "blue", 0.5: "lime", 0.75: "yellow", 1.0: "red"},
                ).add_to(m)

        # Legend
        legend_html = """
        <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                    background:white;padding:12px 16px;border-radius:8px;
                    border:2px solid #999;font-size:13px;">
            <b>Hydro-Graph v2 Flood Risk</b><br><br>
            <span style="background:#2ECC71;padding:2px 10px">&nbsp;</span> Low (0-30%)<br>
            <span style="background:#F1C40F;padding:2px 10px">&nbsp;</span> Moderate (30-50%)<br>
            <span style="background:#E67E22;padding:2px 10px">&nbsp;</span> High (50-70%)<br>
            <span style="background:#E74C3C;padding:2px 10px">&nbsp;</span> Very High (>70%)<br>
            <br><small>DS-STGAT | 2015 Chennai Flood Event</small>
        </div>"""
        m.get_root().html.add_child(folium.Element(legend_html))
        folium.LayerControl().add_to(m)
        m.save(path)
        logger.info("Interactive map saved -> %s", path)
        return path

    # ── Static Risk Map ───────────────────────────────────────────────────────

    def create_static_map(self, df_pred: pd.DataFrame, output_path: Optional[str] = None, dpi: int = 200) -> str:
        path = output_path or str(self.base / self.cfg.paths.risk_map_png)
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        lead_times = self.cfg.temporal.lead_times
        # Show 4 panels: lead 1hr, 3hr, 6hr, 12hr
        fig, axes = plt.subplots(2, 2, figsize=(20, 18))
        fig.patch.set_facecolor("#0D0D1A")
        axes_flat = axes.flatten()

        cmap = mcolors.LinearSegmentedColormap.from_list(
            "flood_risk", ["#1B4F72", "#2ECC71", "#F1C40F", "#E67E22", "#E74C3C"], N=512
        )

        for ax_idx, h in enumerate(lead_times[:4]):
            ax = axes_flat[ax_idx]
            ax.set_facecolor("#0D0D1A")
            col = f"flood_prob_lead{h}h"
            if col not in df_pred.columns:
                ax.set_visible(False)
                continue

            probs = df_pred[col].values
            sc = ax.scatter(
                df_pred["lon"].values, df_pred["lat"].values,
                c=probs, cmap=cmap, vmin=0.0, vmax=1.0,
                s=2.5, alpha=0.9, linewidths=0,
            )
            cbar = fig.colorbar(sc, ax=ax, fraction=0.036, pad=0.02)
            cbar.set_label("Flood Probability", color="white", fontsize=10)
            cbar.ax.yaxis.set_tick_params(color="white")
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

            ax.set_title(f"Lead Time: {h} Hour{'s' if h > 1 else ''}", color="white", fontsize=13, fontweight="bold")
            ax.set_xlabel("Longitude", color="white", fontsize=9)
            ax.set_ylabel("Latitude", color="white", fontsize=9)
            ax.tick_params(colors="white", labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#2C2C4A")

            flooded_pct = (probs >= self.cfg.training.threshold).mean() * 100
            ax.text(0.02, 0.98,
                    f"At-risk: {flooded_pct:.1f}%\nMax p: {probs.max():.2f}",
                    transform=ax.transAxes, fontsize=8, color="white",
                    verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#1C1C2E", alpha=0.85))

        # Legend
        patches = [Patch(facecolor=c, label=l) for l, c in zip(RISK_LABELS, RISK_COLOURS)]
        fig.legend(handles=patches, loc="lower center", ncol=4, frameon=True,
                   facecolor="#1C1C2E", edgecolor="#444466", labelcolor="white", fontsize=10)
        fig.suptitle(
            "DS-STGAT: Multi-Horizon Flood Probability — Chennai 2015 Flood Event",
            color="white", fontsize=15, fontweight="bold", y=0.99,
        )
        plt.tight_layout(rect=[0, 0.04, 1, 0.98])
        plt.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        logger.info("Static map saved -> %s", path)
        return path

    # ── Uncertainty Map ───────────────────────────────────────────────────────

    def create_uncertainty_map(self, df_pred: pd.DataFrame, output_path: Optional[str] = None, dpi: int = 180) -> str:
        path = output_path or str(self.base / self.cfg.paths.uncertainty_map_png)
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.patch.set_facecolor("#0D0D1A")

        cmap_prob = mcolors.LinearSegmentedColormap.from_list(
            "risk", ["#1B4F72", "#2ECC71", "#E74C3C"], N=256
        )
        cmap_unc = mcolors.LinearSegmentedColormap.from_list(
            "uncertainty", ["#2C3E50", "#8E44AD", "#F39C12"], N=256
        )

        for ax, (col, title, cmap) in zip(axes, [
            ("flood_prob_lead1h", "Mean Flood Probability (Lead 1hr)", cmap_prob),
            ("flood_unc_lead1h", "Epistemic Uncertainty (MC Dropout Std)", cmap_unc),
        ]):
            ax.set_facecolor("#0D0D1A")
            if col not in df_pred.columns:
                ax.set_visible(False)
                continue
            vals = df_pred[col].values
            sc = ax.scatter(df_pred["lon"], df_pred["lat"], c=vals, cmap=cmap,
                            vmin=0, vmax=vals.max() if vals.max() > 0 else 1,
                            s=2.5, alpha=0.9, linewidths=0)
            cbar = fig.colorbar(sc, ax=ax, fraction=0.036, pad=0.02)
            cbar.ax.yaxis.set_tick_params(color="white")
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
            ax.set_title(title, color="white", fontsize=12, fontweight="bold")
            ax.set_xlabel("Longitude", color="white", fontsize=9)
            ax.set_ylabel("Latitude", color="white", fontsize=9)
            ax.tick_params(colors="white", labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#2C2C4A")

        fig.suptitle("DS-STGAT Flood Forecast + Uncertainty — Chennai 2015",
                     color="white", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        logger.info("Uncertainty map saved -> %s", path)
        return path

    # ── Training Curves ───────────────────────────────────────────────────────

    def plot_training_curves(self, history: Dict, output_path: Optional[str] = None) -> str:
        path = output_path or str(self.base / self.cfg.paths.training_curves)
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        lead_times = self.cfg.temporal.lead_times[:4]
        epochs = list(range(1, len(history.get("train_loss", [])) + 1))
        if not epochs:
            return path

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.patch.set_facecolor("#0D0D1A")

        # Loss
        ax = axes[0]
        ax.set_facecolor("#0D0D1A")
        ax.plot(epochs, history.get("train_loss", []), color="#3498DB", lw=2, label="Train Loss")
        ax.plot(epochs, history.get("val_loss", []), color="#E74C3C", lw=2, ls="--", label="Val Loss")
        ax.set_title("Loss", color="white", fontsize=12)
        ax.tick_params(colors="white")
        ax.legend(frameon=False, labelcolor="white")
        ax.grid(True, alpha=0.15, color="white")

        # F1 per lead time
        ax = axes[1]
        ax.set_facecolor("#0D0D1A")
        colours = ["#2ECC71", "#F1C40F", "#E67E22", "#E74C3C"]
        keys = ["val_f1_lead1", "val_f1_lead3", "val_f1_lead6", "val_f1_lead12"]
        for i, (h, c, k) in enumerate(zip(lead_times, colours, keys)):
            if k in history:
                ax.plot(epochs, history[k], color=c, lw=2, label=f"F1 ({h}hr)")
        ax.set_title("Val F1 per Lead Time", color="white", fontsize=12)
        ax.tick_params(colors="white")
        ax.legend(frameon=False, labelcolor="white", fontsize=8)
        ax.grid(True, alpha=0.15, color="white")
        ax.set_ylim(0, 1)

        # AUC-ROC (1hr lead)
        ax = axes[2]
        ax.set_facecolor("#0D0D1A")
        if "val_auroc_lead1" in history:
            ax.plot(epochs, history["val_auroc_lead1"], color="#9B59B6", lw=2, label="AUC-ROC (1hr)")
            best_ep = int(np.argmax(history["val_auroc_lead1"])) + 1
            ax.axvline(best_ep, color="#FFD700", ls=":", alpha=0.7, label=f"Best ep {best_ep}")
        ax.set_title("Val AUC-ROC (1hr Lead)", color="white", fontsize=12)
        ax.tick_params(colors="white")
        ax.legend(frameon=False, labelcolor="white", fontsize=8)
        ax.grid(True, alpha=0.15, color="white")
        ax.set_ylim(0, 1)

        for ax in axes:
            ax.set_xlabel("Epoch", color="white", fontsize=9)
            for spine in ax.spines.values():
                spine.set_edgecolor("#444466")

        fig.suptitle("DS-STGAT Training History", color="white", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        logger.info("Training curves saved -> %s", path)
        return path

    # ── Calibration Plot ──────────────────────────────────────────────────────

    def plot_calibration(
        self,
        dataset: HydroGraphDataset,
        test_idx: np.ndarray,
        output_path: Optional[str] = None,
        n_bins: int = 10,
    ) -> str:
        """
        Reliability diagram (calibration curve) for lead=1hr predictions.
        A well-calibrated model has probabilities close to the diagonal.
        """
        path = output_path or str(self.base / self.cfg.paths.calibration_plot)
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        all_preds, all_labels = [], []
        with torch.no_grad():
            for t in test_idx:
                data = dataset.get_snapshot(int(t))
                probs = self._infer_snapshot(data)     # [N, n_leads]
                all_preds.append(probs[:, 0])          # lead=1hr
                all_labels.append(data.y[:, 0].numpy())

        preds_np = np.concatenate(all_preds)
        labels_np = np.concatenate(all_labels)

        bins = np.linspace(0, 1, n_bins + 1)
        bin_centres, mean_preds, frac_pos = [], [], []
        for i in range(n_bins):
            mask = (preds_np >= bins[i]) & (preds_np < bins[i + 1])
            if mask.sum() > 5:
                bin_centres.append((bins[i] + bins[i + 1]) / 2)
                mean_preds.append(preds_np[mask].mean())
                frac_pos.append(labels_np[mask].mean())

        fig, (ax_cal, ax_hist) = plt.subplots(1, 2, figsize=(14, 6))
        fig.patch.set_facecolor("#0D0D1A")

        for ax in [ax_cal, ax_hist]:
            ax.set_facecolor("#0D0D1A")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#444466")

        # Calibration curve
        ax_cal.plot([0, 1], [0, 1], "k--", color="#AAAAAA", lw=1, label="Perfect calibration")
        ax_cal.plot(mean_preds, frac_pos, "o-", color="#2ECC71", lw=2, ms=6, label="DS-STGAT (1hr)")
        ax_cal.fill_between(mean_preds, frac_pos, mean_preds,
                             alpha=0.15, color="#E74C3C", label="Calibration gap")
        ax_cal.set_xlabel("Mean Predicted Probability", color="white", fontsize=10)
        ax_cal.set_ylabel("Fraction of Positives (Observed)", color="white", fontsize=10)
        ax_cal.set_title("Reliability Diagram (Lead=1hr)", color="white", fontsize=12, fontweight="bold")
        ax_cal.legend(frameon=False, labelcolor="white", fontsize=8)
        ax_cal.grid(True, alpha=0.15, color="white")
        ax_cal.set_xlim(0, 1)
        ax_cal.set_ylim(0, 1)

        # Histogram of predicted probabilities
        ax_hist.hist(preds_np, bins=n_bins, color="#3498DB", alpha=0.75, label="Predicted probs")
        ax_hist.set_xlabel("Predicted Flood Probability", color="white", fontsize=10)
        ax_hist.set_ylabel("Count", color="white", fontsize=10)
        ax_hist.set_title("Probability Distribution", color="white", fontsize=12, fontweight="bold")
        ax_hist.legend(frameon=False, labelcolor="white", fontsize=8)

        fig.suptitle("DS-STGAT Calibration Analysis — Chennai 2015 Flood",
                     color="white", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        logger.info("Calibration plot saved -> %s", path)
        return path

    # ── CSV Export ────────────────────────────────────────────────────────────

    def save_predictions(self, df_pred: pd.DataFrame, output_path: Optional[str] = None) -> str:
        path = output_path or str(self.base / self.cfg.paths.predictions_csv)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df_pred.to_csv(path)
        logger.info("Predictions saved -> %s  (%d nodes)", path, len(df_pred))
        return path
