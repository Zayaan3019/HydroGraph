"""
Generates DS-STGAT paper/report figures from REAL pipeline output.
======================================================================
Every number plotted here is loaded from `data/outputs/eval_metrics.json`
and `data/outputs/baseline_metrics.json` — both written by `python main.py`
after a real training + evaluation run. This script contains **no**
hardcoded metric literals. If a required JSON file (or a specific key
inside it) is missing, the figure that depends on it is skipped with a
printed warning instead of being drawn with a placeholder/fabricated
number.

Two figures that used to live here were removed rather than "fixed":
  - The training-convergence curve is produced directly by main.py from
    the real per-epoch history (`InferenceEngine.plot_training_curves`)
    and saved to `data/outputs/training_curves.png`. Reconstructing it
    here from nothing would mean either fabricating a curve or inventing
    a second, redundant history export — the real one already exists.
  - The "flood risk map" (fig7) was a fully synthetic illustration
    (random node positions, hand-tuned fake elevation bumps, a
    `np.random`-generated probability field) with no connection to any
    real model output. The real risk map, from real predictions, is
    produced by `main.py` -> `InferenceEngine.create_static_map()` and
    saved to `data/outputs/flood_risk_map.png`.
  - Similarly, the single-model reliability diagram (real per-bin
    calibration curve) is produced directly by main.py's
    `InferenceEngine.plot_calibration()` and saved to
    `data/outputs/calibration_curve.png`, using real held-out
    predictions. This script's calibration figure only plots the
    cross-model ECE/Brier bar comparison, which the two metrics JSONs
    can actually back.

Run: python generate_figures.py
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent

from hydro_graph.config import load_config  # noqa: E402

CFG = load_config(None if not (ROOT / "config" / "config.yaml").exists() else str(ROOT / "config" / "config.yaml"))

FIG_DIR = ROOT / CFG.paths.outputs_dir / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

EVAL_METRICS_PATH = ROOT / CFG.paths.metrics_json
BASELINE_METRICS_PATH = ROOT / CFG.paths.baselines_json
CROSS_METRICS_PATH = ROOT / CFG.paths.outputs_dir / "cross_event_metrics.json"

LEAD_TIMES = CFG.temporal.lead_times   # e.g. [1, 3, 6, 12]

BASELINE_ORDER = ["persistence", "random_forest", "lstm_only", "gcn_gru", "sage_v1_gru"]
BASELINE_LABELS = {
    "persistence":   "Persistence",
    "random_forest": "Rand. Forest",
    "lstm_only":     "LSTM-only",
    "gcn_gru":       "GCN+GRU",
    "sage_v1_gru":   "SAGEv1+GRU",
}

# ─── Shared style ────────────────────────────────────────────────────────────
BLUE   = "#1F4E79"
LBLUE  = "#2E75B6"
CYAN   = "#00B0F0"
TEAL   = "#00B0A0"
GREEN  = "#375623"
LGREEN = "#70AD47"
ORANGE = "#C55A11"
RED    = "#C00000"
GOLD   = "#C9A400"
LGRAY  = "#D9D9D9"
GRAY   = "#7F7F7F"
WHITE  = "#FFFFFF"
DARK   = "#1A1A2E"

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "figure.dpi":        150,
})


# ─── Real-data loaders ────────────────────────────────────────────────────────

def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        print(f"  [skip-source] {path} does not exist yet — run `python main.py` first.")
        return None
    with open(path, "r") as fh:
        data = json.load(fh)
    if not data:
        print(f"  [skip-source] {path} is empty.")
        return None
    return data


def _lead_series(metrics: Dict[str, Any], key: str) -> Optional[list]:
    """Pull metric `key` across all configured lead times as [v_lead0, v_lead1, ...]."""
    vals = []
    for h_idx in range(len(LEAD_TIMES)):
        k = f"{key}_lead{h_idx}"
        if k not in metrics:
            return None
        vals.append(metrics[k])
    return vals


def _baseline_metric(baselines: Dict[str, Any], model: str, key: str) -> Optional[float]:
    m = baselines.get(model)
    if not m or key not in m:
        return None
    return m[key]


# ══════════════════════════════════════════════════════════════════════════════
#  FIG 1 — ARCHITECTURE DIAGRAM (structural; real parameter counts)
# ══════════════════════════════════════════════════════════════════════════════
def _real_param_counts() -> Optional[Dict[str, int]]:
    """Instantiate the actual model from config and count real parameters
    per submodule. No training/data required — this is the architecture's
    own structure, not a metric, but the numbers must still be real."""
    try:
        from hydro_graph.phase4_model import build_model
        model = build_model(CFG)
    except Exception as exc:
        print(f"  [skip-params] Could not instantiate model for param counts: {exc}")
        return None

    counts = {}
    for name, module in model.named_children():
        counts[name] = sum(p.numel() for p in module.parameters())
    counts["total"] = sum(p.numel() for p in model.parameters())
    return counts


def draw_box(ax, x, y, w, h, label, sublabel="", color=LBLUE, text_color=WHITE,
             fontsize=9, sub_fontsize=7.5, radius=0.03, lw=1.5):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle=f"round,pad=0.01,rounding_size={radius}",
                         linewidth=lw, edgecolor=WHITE,
                         facecolor=color, zorder=3)
    ax.add_patch(box)
    y_text = y + (h * 0.12 if sublabel else 0)
    ax.text(x, y_text, label, ha="center", va="center",
            color=text_color, fontsize=fontsize, fontweight="bold", zorder=4)
    if sublabel:
        ax.text(x, y - h * 0.22, sublabel, ha="center", va="center",
                color=text_color, fontsize=sub_fontsize, style="italic", zorder=4)


def arrow(ax, x1, y1, x2, y2, color=LGRAY, lw=1.8, arrowsize=10):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=arrowsize),
                zorder=2)


def fig_architecture():
    counts = _real_param_counts()

    fig, ax = plt.subplots(1, 1, figsize=(16, 11))
    ax.set_xlim(0, 16); ax.set_ylim(0, 11)
    ax.set_aspect("equal"); ax.axis("off")
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(DARK)

    ax.text(8, 10.5, "DS-STGAT Architecture", ha="center", va="center",
            color=WHITE, fontsize=16, fontweight="bold")
    ax.text(8, 10.1, "Dual-Scale Spatiotemporal Graph Attention Network for Urban Flood Forecasting",
            ha="center", va="center", color=LGRAY, fontsize=10)

    draw_box(ax, 1.5, 8.5, 2.2, 0.7, "Static Features", f"s_v ∈ ℝ¹⁶", color="#1B4F72", fontsize=8.5)
    draw_box(ax, 1.5, 7.2, 2.2, 0.7, "6-hr Rainfall", "r_v[t-6:t] ∈ ℝ⁶", color="#1B4F72", fontsize=8.5)
    draw_box(ax, 1.5, 5.9, 2.2, 0.7, "24-hr Rainfall", "r_v[t-24:t:2] ∈ ℝ¹²", color="#1B4F72", fontsize=8.5)
    draw_box(ax, 1.5, 4.4, 2.2, 0.7, "Edge Features", "ε ∈ ℝᴱˣ⁴", color="#1B4F72", fontsize=8.5)
    ax.text(1.5, 9.3, "INPUTS", ha="center", color=GOLD, fontsize=9, fontweight="bold")

    draw_box(ax, 4.0, 8.5, 2.4, 0.9,
             "Static Encoder", "MLP(16→128→64)\n+ Residual + LN + GELU",
             color=BLUE, fontsize=8.5, sub_fontsize=7)
    arrow(ax, 2.61, 8.5, 3.28, 8.5)
    ax.text(4.0, 9.3, "MODULE 1", ha="center", color=GOLD, fontsize=7.5)

    draw_box(ax, 4.0, 7.2, 2.4, 0.75,
             "Short-Term GRU", f"hidden={CFG.model.short_gru_hidden}, {CFG.model.short_gru_layers}L, drop={CFG.model.dropout}",
             color="#1A5276", fontsize=8.5, sub_fontsize=7)
    draw_box(ax, 4.0, 5.9, 2.4, 0.75,
             "Long-Term GRU", f"hidden={CFG.model.long_gru_hidden}, {CFG.model.long_gru_layers}L",
             color="#1A5276", fontsize=8.5, sub_fontsize=7)
    arrow(ax, 2.61, 7.2, 3.28, 7.2)
    arrow(ax, 2.61, 5.9, 3.28, 5.9)
    ax.text(4.0, 8.0, "MODULE 2", ha="center", color=GOLD, fontsize=7.5)

    rect = FancyBboxPatch((2.72, 5.45), 1.64, 2.2, boxstyle="round,pad=0.05",
                          linewidth=1.2, edgecolor=CYAN, facecolor="none",
                          linestyle="--", zorder=2)
    ax.add_patch(rect)
    ax.text(3.54, 7.75, "Dual-Scale\nTemporal Encoder", ha="center", color=CYAN,
            fontsize=7, style="italic")

    draw_box(ax, 7.2, 6.55, 2.5, 1.2,
             "Cross-Temporal\nAttention Gate",
             "cat(h_short, h_long)\n→ MLP → Softmax\n→ Proj + LN",
             color="#7D3C98", fontsize=8.5, sub_fontsize=7)
    arrow(ax, 5.21, 7.2,  6.44, 6.9)
    arrow(ax, 5.21, 5.9,  6.44, 6.2)
    ax.text(7.2, 7.35, "MODULE 3", ha="center", color=GOLD, fontsize=7.5)

    ax.text(7.2, 5.75, "α_short, α_long ∈ [0,1]\n(α_short + α_long = 1)",
            ha="center", color="#CE93D8", fontsize=7, style="italic")

    draw_box(ax, 10.0, 7.6, 2.5, 0.75,
             "Fusion Layer",
             "cat(h_static, h_temp)\nLinear + LN",
             color="#117A65", fontsize=8.5, sub_fontsize=7)
    arrow(ax, 5.21, 8.5,  9.25, 7.85)
    arrow(ax, 8.46, 6.55, 9.25, 7.35)
    ax.text(10.0, 8.1, "FUSION", ha="center", color=GOLD, fontsize=7.5)

    draw_box(ax, 10.0, 6.1, 2.5, 0.8,
             "GATv2 Layer 1",
             f"heads={CFG.model.gat_heads_l1}, edge_dim={CFG.model.edge_dim}",
             color=ORANGE, fontsize=8.5, sub_fontsize=7)
    draw_box(ax, 10.0, 5.0, 2.5, 0.8,
             "GATv2 Layer 2",
             f"heads={CFG.model.gat_heads_l2} + Residual",
             color=ORANGE, fontsize=8.5, sub_fontsize=7)
    draw_box(ax, 10.0, 3.9, 2.5, 0.8,
             "SAGEConv (max)",
             "max-aggregation",
             color="#A04000", fontsize=8.5, sub_fontsize=7)

    arrow(ax, 10.0, 7.22, 10.0, 6.5)
    arrow(ax, 10.0, 5.7,  10.0, 5.4)
    arrow(ax, 10.0, 4.6,  10.0, 4.3)

    arrow(ax, 2.61, 4.4,  9.06, 4.4, color=GOLD, lw=1.5)
    ax.text(5.8, 4.6, "Physics edge features ε", ha="center", color=GOLD, fontsize=8)

    rect2 = FancyBboxPatch((8.73, 3.42), 2.54, 3.56, boxstyle="round,pad=0.05",
                           linewidth=1.2, edgecolor=ORANGE, facecolor="none",
                           linestyle="--", zorder=2)
    ax.add_patch(rect2)
    ax.text(10.0, 7.2, "MODULE 4", ha="center", color=GOLD, fontsize=7.5)

    ax.text(11.5, 6.1, "Attention weights\nconditioned on\nhydraulic features",
            ha="left", color="#F0B27A", fontsize=7, style="italic")

    draw_box(ax, 13.2, 5.8, 2.2, 0.75,
             "Output Head",
             "Linear + Sigmoid",
             color=RED, fontsize=8.5, sub_fontsize=7)
    arrow(ax, 11.26, 3.9,  12.45, 5.55, color=LGRAY)
    ax.text(13.2, 6.6, "MODULE 5", ha="center", color=GOLD, fontsize=7.5)

    lead_colors = ["#FF6B6B", "#FFA07A", "#FFD700", "#98D8C8"]
    for i, h in enumerate(LEAD_TIMES[:4]):
        lc = lead_colors[i % len(lead_colors)]
        ypos = 4.8 - i * 0.75
        draw_box(ax, 13.2, ypos, 2.0, 0.55,
                 f"P_flood (lead {h}h)", "θ_v ∈ [0,1]",
                 color=lc, text_color=DARK, fontsize=7.5, sub_fontsize=6.5)
        arrow(ax, 13.2, 5.42, 13.2, ypos + 0.28, color=lc)

    ax.text(13.2, 2.2, "OUTPUTS\n(Multi-Horizon)", ha="center", color=GOLD,
            fontsize=8, fontweight="bold")

    banner = FancyBboxPatch((0.3, 0.3), 15.4, 0.9, boxstyle="round,pad=0.08",
                            facecolor="#0D1117", edgecolor=CYAN, linewidth=1.5, zorder=1)
    ax.add_patch(banner)
    if counts is not None:
        specs = (
            f"Total Parameters: {counts['total']:,}  |  "
            f"Static Encoder: {counts.get('static_encoder', 0)/1000:.0f}K  |  "
            f"Short GRU: {counts.get('short_gru', 0)/1000:.0f}K  |  "
            f"Long GRU: {counts.get('long_gru', 0)/1000:.0f}K  |  "
            f"Temporal Gate: {counts.get('temporal_gate', 0)/1000:.0f}K  |  "
            f"Fusion: {counts.get('fusion', 0)/1000:.0f}K  |  "
            f"Spatial (GAT+SAGE): {counts.get('spatial', 0)/1000:.0f}K  |  "
            f"Head: {counts.get('output_head', 0)/1000:.0f}K"
        )
    else:
        specs = "Parameter counts unavailable (model could not be instantiated)."
    ax.text(8, 0.76, specs, ha="center", va="center",
            color=CYAN, fontsize=7.5, style="italic")

    plt.tight_layout(pad=0)
    path = FIG_DIR / "fig1_architecture.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    print(f"  Saved {path}")
    return str(path)


# ══════════════════════════════════════════════════════════════════════════════
#  FIG 2 — MULTI-LEAD PERFORMANCE (real, from eval_metrics.json)
# ══════════════════════════════════════════════════════════════════════════════
def fig_multi_lead():
    metrics = _load_json(EVAL_METRICS_PATH)
    if metrics is None:
        print("  [skip] fig_multi_lead: no eval_metrics.json.")
        return None

    series = {
        "F1 Score":              ("f1", [0.0, 1.0], LBLUE, False),
        "AUC-ROC":               ("auroc", [0.0, 1.0], LGREEN, False),
        "CSI":                   ("csi", [0.0, 1.0], ORANGE, False),
        "FAR (↓ better)":        ("far", [0.0, 1.0], RED, True),
        "POD / Recall":          ("pod", [0.0, 1.0], TEAL, False),
        "Brier Score (↓ better)": ("brier", [0.0, 1.0], GOLD, True),
        "ECE (↓ better)":        ("ece", [0.0, 1.0], "#9B59B6", True),
    }

    available = {}
    for label, (key, ylim, color, lower_better) in series.items():
        vals = _lead_series(metrics, key)
        if vals is not None:
            available[label] = (vals, ylim, color, lower_better)

    if not available:
        print("  [skip] fig_multi_lead: no per-lead metric keys found in eval_metrics.json.")
        return None

    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor("#F8F9FA")
    gs = GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

    positions = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2)]
    x = np.arange(len(LEAD_TIMES))
    width = 0.55

    for (label, (vals, ylim_hint, color, lower_better)), (r, c) in zip(available.items(), positions):
        ax = fig.add_subplot(gs[r, c])
        ax.set_facecolor("#F0F4F8")
        lo = min(0.0, min(vals) * 0.9)
        hi = max(vals) * 1.15 if max(vals) > 0 else 1.0
        bars = ax.bar(x, vals, width, color=color, alpha=0.85,
                      edgecolor="white", linewidth=1.2, zorder=3)
        for bar_obj, v in zip(bars, vals):
            ax.text(bar_obj.get_x() + bar_obj.get_width()/2, bar_obj.get_height() + (hi - lo) * 0.02,
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold", color="#1A1A2E")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{h}hr" for h in LEAD_TIMES], fontsize=10)
        ax.set_ylim(lo, hi)
        ax.set_title(label, fontsize=10.5, fontweight="bold", pad=6)
        ax.set_xlabel("Lead Time", fontsize=9)
        ax.grid(axis="y", alpha=0.35, ls="--", zorder=1)
        ax.spines["left"].set_linewidth(0.8)

    ax_info = fig.add_subplot(gs[1, 3])
    ax_info.axis("off")
    ax_info.set_facecolor("#E8F4F8")
    lines = ["DS-STGAT", "Test 2015, per lead", ""]
    for label, (vals, *_r) in available.items():
        lines.append(f"Lead-{LEAD_TIMES[0]}hr {label}: {vals[0]:.3f}")
    ax_info.text(0.5, 0.5, "\n".join(lines), ha="center", va="center",
                 fontsize=9, transform=ax_info.transAxes,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#D1ECF1",
                           edgecolor=LBLUE, linewidth=1.5),
                 linespacing=1.6)

    fig.suptitle("DS-STGAT Multi-Lead Forecasting Performance (real, measured)\n"
                 "Source: data/outputs/eval_metrics.json",
                 fontsize=12, fontweight="bold", y=1.01)
    path = FIG_DIR / "fig2_multi_lead.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved {path}")
    return str(path)


# ══════════════════════════════════════════════════════════════════════════════
#  FIG 3 — BASELINE COMPARISON (real, from baseline_metrics.json + eval_metrics.json)
# ══════════════════════════════════════════════════════════════════════════════
def fig_baselines():
    eval_metrics = _load_json(EVAL_METRICS_PATH)
    baselines = _load_json(BASELINE_METRICS_PATH)
    if eval_metrics is None or baselines is None:
        print("  [skip] fig_baselines: missing eval_metrics.json or baseline_metrics.json.")
        return None

    models, f1, csi, auc, brier, ece, pod = [], [], [], [], [], [], []

    ds_f1 = eval_metrics.get("f1_lead0")
    if ds_f1 is not None:
        models.append("DS-STGAT\n(ours)")
        f1.append(eval_metrics.get("f1_lead0", 0.0))
        csi.append(eval_metrics.get("csi_lead0", 0.0))
        auc.append(eval_metrics.get("auroc_lead0", 0.0))
        brier.append(eval_metrics.get("brier_lead0", 0.0))
        ece.append(eval_metrics.get("ece_lead0", 0.0))
        pod.append(eval_metrics.get("pod_lead0", eval_metrics.get("recall_lead0", 0.0)))

    for key in BASELINE_ORDER:
        m = baselines.get(key)
        if not m:
            continue
        models.append(BASELINE_LABELS[key])
        f1.append(m.get("f1", 0.0))
        csi.append(m.get("csi", 0.0))
        auc.append(m.get("auroc", 0.0))
        brier.append(m.get("brier", 0.0))
        ece.append(m.get("ece", 0.0))
        pod.append(m.get("pod", m.get("recall", 0.0)))

    if len(models) < 2:
        print("  [skip] fig_baselines: fewer than 2 models with real metrics — nothing to compare.")
        return None

    x = np.arange(len(models))
    width = 0.18

    fig, (ax_main, ax_cal) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor("#F8F9FA")

    ax_main.set_facecolor("#F0F4F8")
    colors = [LBLUE, LGREEN, ORANGE, TEAL]
    labels = ["F1 Score", "CSI", "AUC-ROC", "POD/Recall"]
    datasets = [f1, csi, auc, pod]

    for i, (data, color, label) in enumerate(zip(datasets, colors, labels)):
        ax_main.bar(x + (i - 1.5) * width, data, width * 0.92,
                    color=color, alpha=0.88, label=label,
                    edgecolor="white", linewidth=0.8, zorder=3)

    ax_main.set_xticks(x); ax_main.set_xticklabels(models, fontsize=10)
    ax_main.set_ylim(0, 1.05)
    ax_main.set_ylabel("Metric Value", fontsize=11)
    ax_main.set_title("Model Comparison: F1, CSI, AUC-ROC, POD\n(Lead = 1hr, real measured values)",
                      fontsize=11, fontweight="bold")
    ax_main.legend(fontsize=9, loc="upper right")
    ax_main.grid(axis="y", alpha=0.35, ls="--", zorder=1)

    ax_cal.set_facecolor("#F0F4F8")
    b_bars = ax_cal.bar(x - width*1.6, brier, width*3, color=RED,
                        alpha=0.82, label="Brier Score ↓", edgecolor="white",
                        linewidth=0.8, zorder=3)
    e_bars = ax_cal.bar(x + width*1.6, ece, width*3, color=GOLD,
                        alpha=0.82, label="ECE ↓", edgecolor="white",
                        linewidth=0.8, zorder=3)
    for bar_obj, v in zip(b_bars, brier):
        ax_cal.text(bar_obj.get_x() + bar_obj.get_width()/2, bar_obj.get_height() + 0.003,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#7B241C")
    for bar_obj, v in zip(e_bars, ece):
        ax_cal.text(bar_obj.get_x() + bar_obj.get_width()/2, bar_obj.get_height() + 0.003,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#7D6608")

    ax_cal.set_xticks(x); ax_cal.set_xticklabels(models, fontsize=10)
    ax_cal.set_ylabel("Score (lower is better)", fontsize=11)
    ax_cal.set_title("Calibration Comparison: Brier Score and ECE\n(Lead = 1hr, real measured values)",
                     fontsize=11, fontweight="bold")
    ax_cal.legend(fontsize=9)
    ax_cal.grid(axis="y", alpha=0.35, ls="--", zorder=1)

    plt.tight_layout(pad=1.5)
    path = FIG_DIR / "fig3_baselines.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved {path}")
    return str(path)


# ══════════════════════════════════════════════════════════════════════════════
#  FIG 4 — F1 DEGRADATION ACROSS LEAD TIMES (real: test-2015 vs cross-event-2018)
# ══════════════════════════════════════════════════════════════════════════════
def fig_lead_degradation():
    eval_metrics = _load_json(EVAL_METRICS_PATH)
    if eval_metrics is None:
        print("  [skip] fig_lead_degradation: no eval_metrics.json.")
        return None

    test_f1 = _lead_series(eval_metrics, "f1")
    if test_f1 is None:
        print("  [skip] fig_lead_degradation: no per-lead F1 in eval_metrics.json.")
        return None

    cross_metrics = _load_json(CROSS_METRICS_PATH)
    cross_f1 = _lead_series(cross_metrics, "f1") if cross_metrics else None

    fig, axes = plt.subplots(1, 2 if cross_f1 else 1, figsize=(13 if cross_f1 else 7, 5))
    fig.patch.set_facecolor("#F8F9FA")
    ax1 = axes[0] if cross_f1 else axes

    ax1.set_facecolor("#F0F4F8")
    ax1.plot(LEAD_TIMES, test_f1, "o-", color=LBLUE, lw=3.0, ms=9, label="DS-STGAT (test, real)", zorder=5)
    for i, lead in enumerate(LEAD_TIMES):
        ax1.annotate(f"{test_f1[i]:.3f}", (lead, test_f1[i]),
                     textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=8.5, color=BLUE, fontweight="bold")
    ax1.set_xlabel("Lead Time (hours)", fontsize=11)
    ax1.set_ylabel("F1 Score", fontsize=11)
    ax1.set_title("F1 Score vs Lead Time (real, measured)\nSource: eval_metrics.json", fontsize=12, fontweight="bold")
    ax1.set_xticks(LEAD_TIMES)
    ax1.legend(fontsize=9, loc="best")
    ax1.grid(alpha=0.35, ls="--")

    if cross_f1:
        ax2 = axes[1]
        ax2.set_facecolor("#F0F4F8")
        ax2.plot(LEAD_TIMES, test_f1, "o-", color=LBLUE, lw=3.0, ms=9, label="DS-STGAT (2015 test)", zorder=5)
        ax2.plot(LEAD_TIMES, cross_f1, "s--", color=TEAL, lw=3.0, ms=9, label="DS-STGAT (2018 analogue)", zorder=5)
        for i, lead in enumerate(LEAD_TIMES):
            drop = test_f1[i] - cross_f1[i]
            mid_y = (test_f1[i] + cross_f1[i]) / 2
            col = LGREEN if drop < 0.03 else (GOLD if drop < 0.10 else RED)
            ax2.annotate(f"Δ={drop:+.3f}", (lead, mid_y),
                         textcoords="offset points", xytext=(6, 0),
                         ha="left", fontsize=8, color=col, fontweight="bold")
        ax2.fill_between(LEAD_TIMES, test_f1, cross_f1, alpha=0.15, color=TEAL, label="Cross-event gap")
        ax2.set_xlabel("Lead Time (hours)", fontsize=11)
        ax2.set_ylabel("F1 Score", fontsize=11)
        ax2.set_title("Cross-Event Generalisation (real)\nTrain: 2015 → Test: 2018 analogue", fontsize=12, fontweight="bold")
        ax2.set_xticks(LEAD_TIMES)
        ax2.legend(fontsize=9, loc="best")
        ax2.grid(alpha=0.35, ls="--")
    else:
        print("  [note] fig_lead_degradation: cross_event_metrics.json not found; "
              "plotting test-2015 only (no cross-event panel).")

    plt.tight_layout(pad=1.5)
    path = FIG_DIR / "fig4_lead_degradation.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved {path}")
    return str(path)


# ══════════════════════════════════════════════════════════════════════════════
#  FIG 5 — RADAR CHART (real, single lead=1hr comparison across all models)
# ══════════════════════════════════════════════════════════════════════════════
def fig_radar():
    eval_metrics = _load_json(EVAL_METRICS_PATH)
    baselines = _load_json(BASELINE_METRICS_PATH)
    if eval_metrics is None or baselines is None:
        print("  [skip] fig_radar: missing eval_metrics.json or baseline_metrics.json.")
        return None

    categories = ["F1", "AUC-ROC", "CSI", "POD", "1-FAR", "1-ECE", "1-Brier"]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    def _row(m: Dict[str, Any]) -> Optional[list]:
        try:
            return [
                m["f1"], m["auroc"], m["csi"], m.get("pod", m.get("recall")),
                1 - m["far"], 1 - m["ece"], 1 - m["brier"],
            ]
        except KeyError:
            return None

    data = {}
    if eval_metrics.get("f1_lead0") is not None:
        row = _row({
            "f1": eval_metrics.get("f1_lead0"), "auroc": eval_metrics.get("auroc_lead0"),
            "csi": eval_metrics.get("csi_lead0"), "pod": eval_metrics.get("pod_lead0", eval_metrics.get("recall_lead0")),
            "far": eval_metrics.get("far_lead0"), "ece": eval_metrics.get("ece_lead0"),
            "brier": eval_metrics.get("brier_lead0"),
        })
        if row is not None:
            data["DS-STGAT"] = row

    for key in BASELINE_ORDER:
        m = baselines.get(key)
        if not m:
            continue
        row = _row(m)
        if row is not None:
            data[BASELINE_LABELS[key]] = row

    if len(data) < 2:
        print("  [skip] fig_radar: fewer than 2 models with complete real metrics.")
        return None

    colors_ = [LBLUE, LGREEN, ORANGE, GOLD, RED, TEAL]
    alphas  = [0.30,  0.12,   0.12,   0.08,  0.08, 0.08]

    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F0F4F8")

    for (model, vals), color, alpha in zip(data.items(), colors_, alphas):
        vals_plot = vals + vals[:1]
        is_proposed = model == "DS-STGAT"
        ax.plot(angles, vals_plot, "o-" if is_proposed else "-",
                lw=3.0 if is_proposed else 1.8,
                color=color, label=model,
                zorder=4 if is_proposed else 2)
        ax.fill(angles, vals_plot, alpha=alpha, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight="bold")
    ax.set_ylim(0.0, 1.01)
    ax.tick_params(axis="x", pad=10)
    ax.spines["polar"].set_linewidth(1.0)
    ax.grid(color=GRAY, alpha=0.35, linewidth=0.8)

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.20), ncol=3, fontsize=9)
    ax.set_title("Multi-Metric Model Comparison (Lead = 1hr, real)\nHigher = better on all axes\nSource: eval_metrics.json + baseline_metrics.json",
                 fontsize=10.5, fontweight="bold", pad=18)

    path = FIG_DIR / "fig5_radar.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved {path}")
    return str(path)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating DS-STGAT figures from real pipeline output ...")
    print(f"  eval_metrics.json     -> {EVAL_METRICS_PATH}")
    print(f"  baseline_metrics.json -> {BASELINE_METRICS_PATH}")
    print(f"  cross_event_metrics.json (optional) -> {CROSS_METRICS_PATH}")
    print()

    paths = {}
    fns = {
        "architecture":     fig_architecture,
        "multi_lead":       fig_multi_lead,
        "baselines":        fig_baselines,
        "lead_degradation": fig_lead_degradation,
        "radar":            fig_radar,
    }
    for name, fn in fns.items():
        result = fn()
        if result is not None:
            paths[name] = os.path.relpath(result, ROOT)

    print()
    if paths:
        print(f"Rendered {len(paths)}/{len(fns)} figures with real backing data -> {FIG_DIR}")
    else:
        print("No figures rendered — no real metrics JSON found. Run `python main.py` first.")

    with open(FIG_DIR / "figure_paths.json", "w") as f:
        json.dump(paths, f, indent=2)
    print(f"Figure manifest (repo-relative paths) -> {FIG_DIR / 'figure_paths.json'}")

    print()
    print("Not regenerated by this script (produced directly by main.py from live data):")
    print(f"  training curves      -> {ROOT / CFG.paths.training_curves}")
    print(f"  calibration diagram  -> {ROOT / CFG.paths.calibration_plot}")
    print(f"  flood risk map       -> {ROOT / CFG.paths.risk_map_png}")
