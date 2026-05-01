"""
Hydro-Graph DS-STGAT — Interactive Flood Risk Forecasting Dashboard
====================================================================
IIT Madras | Urban Flood Forecasting | Chennai Metropolitan Region

Real-time flood risk prediction powered by the Dual-Scale Spatiotemporal
Graph Attention Network (DS-STGAT) with multi-horizon probabilistic
forecasting at 1 / 3 / 6 / 12-hour lead times.

Run:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import json
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logging.basicConfig(level=logging.WARNING)

_CACHE_VERSION_SUFFIX = ".cache_version"

# ─── Page Config (must be the very first Streamlit call) ─────────────────────
st.set_page_config(
    page_title="Hydro-Graph | DS-STGAT Flood Forecast",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "DS-STGAT Urban Flood Forecasting · IIT Madras"},
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background:#0D0D1A; color:#E0E0F0; }
[data-testid="stSidebar"] {
    background:#12122A;
    border-right:1px solid #2A2A4A;
}
[data-testid="stSidebar"] * { color:#C0C0E0 !important; }
[data-testid="stSidebar"] .stMarkdown p { color:#9090B8 !important; font-size:0.82rem; }
.stTabs [data-baseweb="tab"] {
    background:#1C1C2E; border-radius:6px 6px 0 0;
    color:#7070A0; padding:8px 20px; font-size:0.85rem;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background:#252545; color:#FFFFFF; font-weight:600;
}
div[data-testid="metric-container"] {
    background:linear-gradient(135deg,#1C1C2E 0%,#252545 100%);
    border:1px solid #3A3A5C; border-radius:10px; padding:14px 16px;
}
div[data-testid="metric-container"] label {
    color:#7070A0 !important; font-size:0.75rem !important;
    letter-spacing:0.5px !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color:#FFFFFF; font-size:1.7rem; font-weight:700;
}
div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    color:#8888AA; font-size:0.75rem;
}
.flood-alert {
    background:linear-gradient(135deg,#5D0000 0%,#8B1010 100%);
    border:2px solid #E74C3C; border-radius:10px;
    padding:16px 24px; text-align:center;
    font-size:1.15rem; font-weight:700; color:#FFFFFF;
    animation:pulse 2s ease-in-out infinite;
}
.warn-status {
    background:linear-gradient(135deg,#5C3A00 0%,#8B5A00 100%);
    border:2px solid #E67E22; border-radius:10px;
    padding:14px 24px; text-align:center;
    font-size:1rem; font-weight:600; color:#FFD080;
}
.safe-status {
    background:linear-gradient(135deg,#003D1A 0%,#005C28 100%);
    border:2px solid #2ECC71; border-radius:10px;
    padding:14px 24px; text-align:center;
    font-size:1rem; font-weight:600; color:#AAFFCC;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.82} }
.section-title {
    color:#6060A0; font-size:0.68rem; letter-spacing:3px;
    text-transform:uppercase; padding-bottom:6px;
    border-bottom:1px solid #2A2A4A; margin-bottom:10px;
}
.stButton > button {
    background:linear-gradient(135deg,#1A3A7A 0%,#2255BB 100%);
    border:1px solid #4477DD; color:white;
    font-weight:700; letter-spacing:1.5px;
    border-radius:8px; padding:11px 0; width:100%;
    font-size:0.95rem; transition:all 0.2s;
}
.stButton > button:hover {
    background:linear-gradient(135deg,#2255BB 0%,#3366CC 100%);
    border-color:#6699EE; transform:translateY(-1px);
    box-shadow:0 4px 15px rgba(34,85,187,0.4);
}
.stDownloadButton > button {
    background:#1C1C2E; border:1px solid #3A3A5C;
    color:#9090C0; border-radius:6px; padding:6px 16px;
}
#MainMenu, footer, [data-testid="stDecoration"] { visibility:hidden; }
.stSlider [data-baseweb="slider"] { padding:4px 0; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
LEAD_TIMES   = [1, 3, 6, 12]
LEAD_LABELS  = ["1-Hour", "3-Hour", "6-Hour", "12-Hour"]
RISK_COLORS  = ["#2ECC71", "#F1C40F", "#E67E22", "#E74C3C"]
RISK_LABELS  = ["Low", "Moderate", "High", "Very High"]
RISK_THRESH  = [0.0, 0.30, 0.50, 0.70, 1.01]

_FLOOD_COLORSCALE = [
    [0.00, "#1B4F72"],
    [0.18, "#2ECC71"],
    [0.35, "#F1C40F"],
    [0.55, "#E67E22"],
    [0.80, "#E74C3C"],
    [1.00, "#7B1111"],
]
_UNC_COLORSCALE = [
    [0.0, "#2C3E50"],
    [0.4, "#8E44AD"],
    [0.8, "#E74C3C"],
    [1.0, "#F39C12"],
]

SCENARIOS: Dict[str, dict] = {
    "2015 Catastrophic (Dec 1-2 Peak)": {
        "intensity": 55.0, "duration": 48, "antecedent": 420,
        "pattern": "south_biased",
        "desc": "Most intense phase of 2015 flood — 55 mm/hr peak over 48 hrs with 420 mm antecedent",
    },
    "Extreme Convective Burst": {
        "intensity": 80.0, "duration": 6, "antecedent": 100,
        "pattern": "uniform",
        "desc": "Short but extreme 80 mm/hr cloudburst — flash flood risk",
    },
    "Heavy Monsoon Event (Nov 28-30)": {
        "intensity": 30.0, "duration": 72, "antecedent": 200,
        "pattern": "south_biased",
        "desc": "2015 pre-peak phase — sustained 30 mm/hr over 72 hrs",
    },
    "2018 Analogue Event": {
        "intensity": 42.0, "duration": 36, "antecedent": 180,
        "pattern": "south_biased",
        "desc": "2018 NE monsoon analogue — cross-event generalisation test",
    },
    "Moderate Monsoon Day": {
        "intensity": 15.0, "duration": 8, "antecedent": 40,
        "pattern": "uniform",
        "desc": "Typical heavy monsoon day — localised flooding in low-lying zones",
    },
    "Light Shower (Baseline)": {
        "intensity": 5.0, "duration": 2, "antecedent": 5,
        "pattern": "uniform",
        "desc": "Non-flood-triggering drizzle — reference for near-zero risk",
    },
    "Custom": {
        "intensity": 20.0, "duration": 6, "antecedent": 50,
        "pattern": "uniform",
        "desc": "User-defined scenario — adjust sliders freely",
    },
}


# ─── Cached loaders ───────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading DS-STGAT model checkpoint…")
def _load_model_and_config():
    try:
        from hydro_graph.config import load_config
        from hydro_graph.phase4_model import build_model

        cfg = load_config(None)
        model = build_model(cfg)
        ckpt_path = ROOT / cfg.paths.best_checkpoint

        if not ckpt_path.exists():
            return None, None, None, f"Checkpoint not found: {ckpt_path}\nRun `python main.py` first."

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()

        best_val_f1 = float(ckpt.get("best_val_f1", 0.0))
        quality_note = None
        if best_val_f1 <= 0.0:
            quality_note = (
                "Checkpoint metadata reports best_val_f1=0.0. "
                "The app can run, but retrain or restore the trained checkpoint "
                "before treating probabilities as final operational results."
            )

        return model, cfg, device, {
            "epoch": ckpt.get("epoch", "?"),
            "best_val_f1": best_val_f1,
            "device": str(device),
            "quality_note": quality_note,
        }
    except Exception as exc:
        return None, None, None, str(exc)


def _write_cache_version(path: Path, version: str) -> None:
    try:
        (path.parent / f"{path.name}{_CACHE_VERSION_SUFFIX}").write_text(version)
    except Exception:
        pass


def _feature_rebuild_reason(features: np.ndarray, n_nodes: int, expected_dim: int) -> Optional[str]:
    if features.shape[0] != n_nodes:
        return f"node count mismatch ({features.shape[0]} cached vs {n_nodes} graph nodes)"
    if features.shape[1] != expected_dim:
        return f"feature dimension mismatch ({features.shape[1]} cached vs {expected_dim} expected)"
    if not np.isfinite(features).all():
        return "non-finite values found in cached feature matrix"
    return None


def _rebuild_static_features(cfg, G, gdf_nodes, edge_features, feat_path: Path) -> np.ndarray:
    from hydro_graph.phase2_features import FeatureEngineer

    rivers = [(float(r.lat), float(r.lon)) for r in getattr(cfg.study_area, "rivers", [])]
    fe = FeatureEngineer(
        crs_projected=cfg.study_area.crs_projected,
        use_synthetic=cfg.features.use_synthetic,
        dem_tif=str(ROOT / cfg.features.dem_tif) if cfg.features.dem_tif else None,
        sentinel2_tif=str(ROOT / cfg.features.sentinel2_tif) if cfg.features.sentinel2_tif else None,
        sentinel1_tif=str(ROOT / cfg.features.sentinel1_tif) if cfg.features.sentinel1_tif else None,
        srtm_cache_dir=str(ROOT / cfg.features.srtm_cache_dir) if cfg.features.srtm_cache_dir else None,
        coast_lat=float(cfg.study_area.coast_lat),
        coast_lon=float(cfg.study_area.coast_lon),
        rivers=rivers or None,
    )
    static_features, df_feat = fe.compute_features(G, gdf_nodes, edge_features)
    feat_path.parent.mkdir(parents=True, exist_ok=True)
    df_feat.to_parquet(feat_path)
    _write_cache_version(feat_path, cfg.cache_version)
    return static_features.astype(np.float32)


@st.cache_data(show_spinner="Loading graph & static features…")
def _load_graph_data():
    try:
        from hydro_graph.config import load_config
        from hydro_graph.phase1_graph import GraphConstructor

        cfg = load_config(None)

        graph_path = ROOT / cfg.paths.graph_gpickle
        if not graph_path.exists():
            return None, f"Graph not found: {graph_path}"

        G, gdf_nodes, edge_features = GraphConstructor.load(str(graph_path), ROOT)

        edges = list(G.edges())
        if not edges:
            return None, "Graph has no edges."
        edge_index = np.array(edges, dtype=np.int64).T

        expected_edge_dim = int(cfg.model.edge_dim)
        if (
            edge_features is None
            or edge_features.shape[0] != edge_index.shape[1]
            or edge_features.shape[1] != expected_edge_dim
        ):
            edge_features = np.zeros((edge_index.shape[1], expected_edge_dim), dtype=np.float32)

        feat_path = ROOT / cfg.paths.node_features
        static_features: Optional[np.ndarray] = None
        feature_source = "cached"

        if feat_path.exists():
            static_features = pd.read_parquet(feat_path).values.astype(np.float32)
            reason = _feature_rebuild_reason(
                static_features,
                n_nodes=len(gdf_nodes),
                expected_dim=int(cfg.model.static_dim),
            )
            if reason is not None:
                feature_source = f"rebuilt ({reason})"
                static_features = None
        else:
            feature_source = "rebuilt (missing cache)"

        if static_features is None:
            static_features = _rebuild_static_features(
                cfg, G, gdf_nodes, edge_features, feat_path,
            )

        return {
            "static_features": static_features,
            "edge_index":       edge_index,
            "edge_features":    edge_features,
            "node_lons":        gdf_nodes["lon"].values.astype(np.float32),
            "node_lats":        gdf_nodes["lat"].values.astype(np.float32),
            "N": static_features.shape[0],
            "E": edge_index.shape[1],
            "feature_dim": static_features.shape[1],
            "feature_source": feature_source,
        }, None

    except Exception as exc:
        return None, str(exc)


@st.cache_data(show_spinner=False)
def _load_historical_predictions() -> Optional[pd.DataFrame]:
    """Load the actual 2015-event test predictions saved by Phase 6."""
    p = ROOT / "data" / "outputs" / "node_predictions.csv"
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception:
            pass
    return None


@st.cache_data(show_spinner=False)
def _load_paper_results() -> dict:
    p = ROOT / "data" / "outputs" / "paper_results.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


@st.cache_data(show_spinner=False)
def _load_rain_norm() -> float:
    """
    Return the 95th-percentile normalization constant used during training.
    The model was trained with rainfall / rain_norm — we must match this exactly
    or the input distribution shifts and predictions degrade.
    """
    try:
        from hydro_graph.config import load_config
        cfg = load_config(None)
        temp_path = ROOT / cfg.paths.temporal_data
        if temp_path.exists():
            arr = np.load(temp_path, allow_pickle=True)
            rain = arr["rainfall"].astype(np.float32)
            pos  = rain[rain > 0]
            if len(pos) > 0:
                return float(max(np.percentile(pos, 95), 1.0))
    except Exception:
        pass
    return 5.0   # safe fallback: typical 95-pct for Chennai monsoon data


# ─── Scenario utilities ───────────────────────────────────────────────────────

def _spatial_weights(lons: np.ndarray, lats: np.ndarray, pattern: str) -> np.ndarray:
    N = len(lats)
    if pattern == "south_biased":
        lat_n = (lats - lats.min()) / max(float(np.ptp(lats)), 1e-6)
        lon_n = (lons - lons.min()) / max(float(np.ptp(lons)), 1e-6)
        w = 1.0 + 0.35 * (1.0 - lat_n) + 0.10 * (1.0 - lon_n)
    elif pattern == "north_biased":
        lat_n = (lats - lats.min()) / max(float(np.ptp(lats)), 1e-6)
        w = 1.0 + 0.30 * lat_n
    else:
        w = np.ones(N, dtype=np.float32)
    return w.astype(np.float32)


def run_scenario(
    model,
    device,
    data: dict,
    intensity: float,
    duration: float,
    antecedent: float,
    pattern: str,
    n_mc: int = 20,
    rain_norm: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct a synthetic rainfall snapshot from user parameters and run
    DS-STGAT inference with MC Dropout uncertainty estimation.

    Returns
    -------
    probs       [N, 4]  flood probabilities at leads 1/3/6/12 hr
    uncertainty [N, 4]  MC Dropout std (epistemic uncertainty)
    """
    static    = data["static_features"]   # [N, 16]
    node_lons = data["node_lons"]
    node_lats = data["node_lats"]
    N         = data["N"]
    expected_static_dim = int(getattr(model, "static_dim", static.shape[1]))

    if static.shape[0] != N:
        raise ValueError(f"Static feature node count {static.shape[0]} does not match graph node count {N}.")
    if static.shape[1] != expected_static_dim:
        raise ValueError(
            f"Static feature dimension {static.shape[1]} does not match model static_dim "
            f"{expected_static_dim}. Regenerate node features with the current pipeline."
        )

    sw = _spatial_weights(node_lons, node_lats, pattern)   # [N]

    # ── Short window: last 6 hours ────────────────────────────────────────────
    short_rain = np.zeros((N, 6), dtype=np.float32)
    for col in range(6):                     # col 0 = 6hr ago, col 5 = 1hr ago
        hr_ago = 6 - col
        if hr_ago <= duration:
            short_rain[:, col] = intensity * sw

    # ── Long window: last 24 h at 2-hour intervals (12 points) ───────────────
    bg_rate   = antecedent / 24.0
    long_rain = np.zeros((N, 12), dtype=np.float32)
    for col in range(12):                    # col 0 = 24hr ago, col 11 = 2hr ago
        hr_ago = (12 - col) * 2
        if hr_ago <= duration:
            long_rain[:, col] = intensity * sw
        else:
            long_rain[:, col] = bg_rate * sw

    # Normalize using the SAME constant as training (loaded from temporal_data.npz)
    # so the input distribution seen by the model matches what it was trained on.
    rain_norm = float(max(rain_norm, 1.0))
    short_n   = (short_rain / rain_norm).clip(0, 2).astype(np.float32)
    long_n    = (long_rain  / rain_norm).clip(0, 2).astype(np.float32)

    # ── Build input tensor [N, 34] ────────────────────────────────────────────
    x = torch.cat([
        torch.as_tensor(static,  dtype=torch.float32),
        torch.as_tensor(short_n, dtype=torch.float32),
        torch.as_tensor(long_n,  dtype=torch.float32),
    ], dim=1).to(device)

    ei = torch.as_tensor(data["edge_index"],    dtype=torch.long).to(device)
    ea = torch.as_tensor(data["edge_features"], dtype=torch.float32).to(device)

    # ── Deterministic inference ───────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        probs = model(x, ei, ea).cpu().numpy()   # [N, 4]

    # ── MC Dropout uncertainty (keep dropout active) ──────────────────────────
    model.train()
    samples: List[np.ndarray] = []
    with torch.no_grad():
        for _ in range(n_mc):
            samples.append(model(x, ei, ea).cpu().numpy())
    model.eval()

    uncertainty = np.stack(samples, axis=0).std(axis=0)   # [N, 4]
    return probs.astype(np.float32), uncertainty.astype(np.float32)


# ─── Chart builders ───────────────────────────────────────────────────────────

def _risk_cat(p: float) -> str:
    for i, t in enumerate(RISK_THRESH[1:]):
        if p < t:
            return RISK_LABELS[i]
    return RISK_LABELS[-1]


def chart_map(
    node_lats, node_lons, probs, uncertainty,
    lead_idx: int, threshold: float, show_unc: bool,
    center_lat: float, center_lon: float,
) -> go.Figure:
    vals = uncertainty[:, lead_idx] if show_unc else probs[:, lead_idx]
    cscale = _UNC_COLORSCALE if show_unc else _FLOOD_COLORSCALE
    cmax   = max(float(uncertainty[:, lead_idx].max()), 0.05) if show_unc else 1.0
    ctitle = "Uncertainty σ" if show_unc else "Flood Prob"

    hover = [
        (f"<b>Flood Prob ({LEAD_TIMES[lead_idx]}hr):</b> {probs[i, lead_idx]:.1%}<br>"
         f"<b>Risk Level:</b> {_risk_cat(probs[i, lead_idx])}<br>"
         f"<b>Lat:</b> {node_lats[i]:.5f}  <b>Lon:</b> {node_lons[i]:.5f}<br>"
         f"<b>Uncertainty:</b> ±{uncertainty[i, lead_idx]:.4f}")
        for i in range(len(node_lats))
    ]

    marker_size = 7 if len(node_lats) < 3000 else 5

    fig = go.Figure(go.Scattermapbox(
        lat=node_lats, lon=node_lons,
        mode="markers",
        marker=go.scattermapbox.Marker(
            size=marker_size,
            color=vals,
            colorscale=cscale,
            cmin=0, cmax=cmax,
            showscale=True,
            colorbar=dict(
                title=dict(text=ctitle, font=dict(color="white", size=11)),
                tickfont=dict(color="white", size=9),
                bgcolor="rgba(0,0,0,0)",
                outlinecolor="rgba(255,255,255,0.15)",
                thickness=12, len=0.75, y=0.5,
            ),
            opacity=0.88,
        ),
        text=hover,
        hovertemplate="%{text}<extra></extra>",
    ))

    # Overlay flooded nodes (above threshold) with a pulsing red ring layer
    flood_mask = probs[:, lead_idx] >= threshold
    if flood_mask.any() and not show_unc:
        fig.add_trace(go.Scattermapbox(
            lat=node_lats[flood_mask],
            lon=node_lons[flood_mask],
            mode="markers",
            marker=go.scattermapbox.Marker(
                size=marker_size + 6,
                color="rgba(231,76,60,0.25)",   # semi-transparent red halo
                opacity=0.9,
            ),
            showlegend=False, hoverinfo="skip",
            name="At-risk halo",
        ))

    fig.update_layout(
        mapbox=dict(
            style="carto-darkmatter",
            center=dict(lat=float(center_lat), lon=float(center_lon)),
            zoom=12,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=500,
        paper_bgcolor="#0D0D1A",
        showlegend=False,
    )
    return fig


def chart_violin(probs: np.ndarray) -> go.Figure:
    fig = go.Figure()
    for i, (h, label) in enumerate(zip(LEAD_TIMES, LEAD_LABELS)):
        r, g, b = int(RISK_COLORS[i][1:3], 16), int(RISK_COLORS[i][3:5], 16), int(RISK_COLORS[i][5:], 16)
        fig.add_trace(go.Violin(
            y=probs[:, i], name=label,
            box_visible=True, meanline_visible=True,
            line_color=RISK_COLORS[i],
            fillcolor=f"rgba({r},{g},{b},0.25)",
            points="outliers", spanmode="hard",
            hovertemplate=f"Lead {h}hr<br>Prob: %{{y:.3f}}<extra></extra>",
        ))
    fig.update_layout(
        title=dict(text="Probability Distribution by Lead Time",
                   font=dict(color="white", size=13), x=0),
        yaxis=dict(title="Flood Probability", color="white",
                   gridcolor="#2A2A4A", range=[0, 1], zeroline=False),
        xaxis=dict(color="white"),
        paper_bgcolor="#1C1C2E", plot_bgcolor="#1C1C2E",
        height=280, margin=dict(l=40, r=10, t=40, b=30),
        violingap=0.15, showlegend=False,
        font=dict(color="white"),
    )
    return fig


def chart_hist(probs: np.ndarray, lead_idx: int) -> go.Figure:
    h = LEAD_TIMES[lead_idx]
    bins = np.linspace(0, 1, 21)
    counts, _ = np.histogram(probs[:, lead_idx], bins=bins)
    centers = (bins[:-1] + bins[1:]) / 2

    bar_colors = [RISK_COLORS[next(
        j for j, t in enumerate(RISK_THRESH[1:]) if c < t
    )] if c < 1.0 else RISK_COLORS[-1] for c in centers]

    fig = go.Figure(go.Bar(
        x=centers, y=counts, marker_color=bar_colors,
        width=0.045,
        hovertemplate="Prob: %{x:.2f}<br>Count: %{y}<extra></extra>",
    ))
    fig.add_vline(x=0.5, line_dash="dot", line_color="white", opacity=0.6,
                  annotation=dict(text="0.50 threshold", font=dict(color="white", size=9),
                                  yref="paper", y=0.97))
    fig.update_layout(
        title=dict(text=f"Risk Distribution — {h}hr Lead",
                   font=dict(color="white", size=12), x=0),
        xaxis=dict(title="Flood Probability", color="white",
                   gridcolor="#2A2A4A", range=[0, 1]),
        yaxis=dict(title="Node Count", color="white", gridcolor="#2A2A4A"),
        paper_bgcolor="#1C1C2E", plot_bgcolor="#1C1C2E",
        height=265, margin=dict(l=40, r=10, t=40, b=40),
        bargap=0.06, font=dict(color="white"),
    )
    return fig


def chart_lead_performance(paper: dict) -> Optional[go.Figure]:
    if "ds_stgat_2015" not in paper:
        return None
    d     = paper["ds_stgat_2015"]
    keys  = ["lead_1hr", "lead_3hr", "lead_6hr", "lead_12hr"]
    leads = ["1 hr", "3 hr", "6 hr", "12 hr"]

    metrics = {
        "F1 Score":    [d[k]["f1"]    for k in keys],
        "AUC-ROC":     [d[k]["auroc"] for k in keys],
        "CSI":         [d[k]["csi"]   for k in keys],
        "Recall/POD":  [d[k]["pod"]   for k in keys],
    }
    colors = ["#3498DB", "#9B59B6", "#2ECC71", "#F39C12"]

    fig = go.Figure()
    for (name, vals), color in zip(metrics.items(), colors):
        fig.add_trace(go.Scatter(
            x=leads, y=vals, mode="lines+markers",
            name=name, line=dict(color=color, width=2.5),
            marker=dict(size=10, symbol="circle"),
            hovertemplate=f"{name}: %{{y:.3f}}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text="DS-STGAT: Skill Scores vs. Forecast Lead Time",
                   font=dict(color="white", size=13), x=0),
        xaxis=dict(title="Lead Time", color="white", gridcolor="#2A2A4A"),
        yaxis=dict(title="Score", color="white", gridcolor="#2A2A4A",
                   range=[0.45, 1.05], zeroline=False),
        paper_bgcolor="#1C1C2E", plot_bgcolor="#1C1C2E",
        height=300, margin=dict(l=50, r=15, t=45, b=40),
        legend=dict(font=dict(color="white", size=10),
                    bgcolor="rgba(0,0,0,0)"),
        font=dict(color="white"),
    )
    return fig


def chart_baselines(paper: dict) -> Optional[go.Figure]:
    if "baselines" not in paper or "ds_stgat_2015" not in paper:
        return None

    bl    = paper["baselines"]
    ds_f1 = paper["ds_stgat_2015"]["lead_1hr"]["f1"]
    ds_auc= paper["ds_stgat_2015"]["lead_1hr"]["auroc"]

    models = ["DS-STGAT\n(ours)", "GCN+GRU",   "SAGE+GRU",     "LSTM-Only",    "RandomForest"]
    f1s    = [ds_f1,
              bl.get("GCN_GRU",    {}).get("f1",    0),
              bl.get("SAGEv1_GRU", {}).get("f1",    0),
              bl.get("LSTM_only",  {}).get("f1",    0),
              bl.get("RandomForest",{}).get("f1",   0)]
    aucs   = [ds_auc,
              bl.get("GCN_GRU",    {}).get("auroc", 0),
              bl.get("SAGEv1_GRU", {}).get("auroc", 0),
              bl.get("LSTM_only",  {}).get("auroc", 0),
              bl.get("RandomForest",{}).get("auroc",0)]

    bar_colors = ["#3498DB"] + ["#5A5A7A"] * 4

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["F1 Score @ 1-Hour Lead", "AUC-ROC @ 1-Hour Lead"],
    )
    for col, vals, ymax in [(1, f1s, 1.05), (2, aucs, 1.05)]:
        fig.add_trace(go.Bar(
            x=models, y=vals,
            marker=dict(color=bar_colors,
                        line=dict(color=["#6AA8F0"] + ["#888888"] * 4, width=1.5)),
            text=[f"{v:.3f}" for v in vals],
            textposition="outside", textfont=dict(color="white", size=10),
            hovertemplate="%{x}: %{y:.4f}<extra></extra>",
        ), row=1, col=col)
        fig.update_yaxes(range=[0, ymax], row=1, col=col,
                         color="white", gridcolor="#2A2A4A")
        fig.update_xaxes(color="white", tickfont=dict(size=9), row=1, col=col)

    fig.update_layout(
        paper_bgcolor="#1C1C2E", plot_bgcolor="#1C1C2E",
        font=dict(color="white"),
        height=300, showlegend=False,
        margin=dict(l=40, r=15, t=55, b=40),
    )
    for ann in fig.layout.annotations:
        ann.font = dict(color="white", size=12)
    return fig


def chart_cross_event(paper: dict) -> Optional[go.Figure]:
    """Bar chart comparing 2015 test vs 2018 cross-event F1."""
    d15 = paper.get("ds_stgat_2015", {})
    d18 = paper.get("ds_stgat_2018", {})
    if not d15 or not d18:
        return None

    keys  = ["lead_1hr", "lead_3hr", "lead_6hr", "lead_12hr"]
    leads = ["1 hr", "3 hr", "6 hr", "12 hr"]

    f1_15 = [d15[k]["f1"]  for k in keys]
    f1_18 = [d18[k]["f1"]  for k in keys]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="2015 Test Event", x=leads, y=f1_15,
                         marker_color="#3498DB",
                         text=[f"{v:.3f}" for v in f1_15],
                         textposition="outside", textfont=dict(color="white", size=10)))
    fig.add_trace(go.Bar(name="2018 Cross-Event", x=leads, y=f1_18,
                         marker_color="#2ECC71",
                         text=[f"{v:.3f}" for v in f1_18],
                         textposition="outside", textfont=dict(color="white", size=10)))
    fig.update_layout(
        title=dict(text="Generalisation: 2015 Test vs 2018 Unseen Event",
                   font=dict(color="white", size=13), x=0),
        xaxis=dict(color="white", gridcolor="#2A2A4A"),
        yaxis=dict(title="F1 Score", color="white", gridcolor="#2A2A4A",
                   range=[0, 1.1]),
        paper_bgcolor="#1C1C2E", plot_bgcolor="#1C1C2E",
        height=280, barmode="group",
        margin=dict(l=50, r=15, t=50, b=40),
        legend=dict(font=dict(color="white", size=10),
                    bgcolor="rgba(0,0,0,0)"),
        font=dict(color="white"),
    )
    return fig


def make_top_risk_df(
    lats: np.ndarray, lons: np.ndarray,
    probs: np.ndarray, unc: np.ndarray,
    top_n: int = 20,
) -> pd.DataFrame:
    mean_p  = probs.mean(axis=1)
    top_idx = np.argsort(mean_p)[::-1][:top_n]

    return pd.DataFrame([
        {
            "Rank":       i + 1,
            "Latitude":   f"{lats[idx]:.5f}",
            "Longitude":  f"{lons[idx]:.5f}",
            "1hr Risk":   f"{probs[idx, 0]:.1%}",
            "3hr Risk":   f"{probs[idx, 1]:.1%}",
            "6hr Risk":   f"{probs[idx, 2]:.1%}",
            "12hr Risk":  f"{probs[idx, 3]:.1%}",
            "Category":   _risk_cat(probs[idx, 0]),
            "Uncert. σ":  f"±{unc[idx, 0]:.4f}",
        }
        for i, idx in enumerate(top_idx)
    ])


# ─── Main App ─────────────────────────────────────────────────────────────────

def main():
    # ── Header ────────────────────────────────────────────────────────────────
    header_l, header_r = st.columns([7, 3])
    with header_l:
        st.markdown("""
<div style="padding:4px 0 12px 0;">
  <h1 style="color:#FFFFFF;margin:0;font-size:1.85rem;font-weight:800;letter-spacing:-0.5px;">
    🌊 Hydro-Graph DS-STGAT
  </h1>
  <p style="color:#6060A0;margin:2px 0 0 2px;font-size:0.82rem;letter-spacing:2px;">
    URBAN FLOOD RISK FORECASTING &nbsp;·&nbsp; CHENNAI METROPOLITAN REGION &nbsp;·&nbsp; IIT MADRAS
  </p>
</div>
""", unsafe_allow_html=True)
    with header_r:
        st.markdown("""
<div style="text-align:right;padding:12px 0 0 0;color:#5050A0;font-size:0.75rem;">
  Dual-Scale Spatiotemporal Graph Attention Network<br>
  GATv2Conv · Dual GRU · Temporal Attention Gate<br>
  Multi-Horizon: 1 / 3 / 6 / 12-Hour Lead Times
</div>
""", unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#2A2A4A;margin:0 0 12px 0;'>", unsafe_allow_html=True)

    # ── Load resources ────────────────────────────────────────────────────────
    model, cfg, device, model_meta = _load_model_and_config()
    data, data_err                 = _load_graph_data()
    paper                          = _load_paper_results()
    hist_df                        = _load_historical_predictions()

    if model is None:
        st.error(f"**Model not loaded.** {model_meta}")
        st.code("python main.py --skip-baselines", language="bash")
        return
    if data is None:
        st.error(f"**Graph/feature data missing.** {data_err}")
        st.code("python main.py --skip-baselines", language="bash")
        return

    N          = data["N"]
    center_lat = float(data["node_lats"].mean())
    center_lon = float(data["node_lons"].mean())
    rain_norm  = _load_rain_norm()   # matches training-time normalization exactly

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown('<p class="section-title">⚡ Storm Scenario</p>', unsafe_allow_html=True)

        scenario_name = st.selectbox(
            "Select Preset or Custom",
            list(SCENARIOS.keys()), index=0, label_visibility="collapsed",
        )
        sc = SCENARIOS[scenario_name]
        custom = scenario_name == "Custom"

        if not custom:
            st.caption(f"📌 {sc['desc']}")

        st.markdown("---")
        st.markdown('<p class="section-title">🌧 Rainfall Parameters</p>', unsafe_allow_html=True)

        intensity = st.slider(
            "Current Intensity (mm/hr)", 0.0, 80.0,
            float(sc["intensity"]), 0.5,
            disabled=not custom,
            help="Peak hourly rainfall intensity at the current timestep",
        )
        duration = st.slider(
            "Storm Duration (hours)", 1, 72,
            int(sc["duration"]), 1,
            disabled=not custom,
            help="Hours the current storm event has been ongoing",
        )
        antecedent = st.slider(
            "Antecedent Accumulation (mm/24h)", 0, 500,
            int(sc["antecedent"]), 5,
            disabled=not custom,
            help="Total rainfall in the 24 hours before this event (controls soil saturation)",
        )

        pattern_opts = {
            "uniform":      "🌐 Uniform (all districts equally)",
            "south_biased": "🧭 South-Biased (NE Monsoon pattern)",
            "north_biased": "🧭 North-Biased",
        }
        pattern = st.selectbox(
            "Spatial Distribution",
            list(pattern_opts.keys()),
            index=list(pattern_opts.keys()).index(sc["pattern"]),
            format_func=lambda x: pattern_opts[x],
            disabled=not custom,
        )

        st.markdown("---")
        st.markdown('<p class="section-title">🎛 Display Options</p>', unsafe_allow_html=True)

        threshold = st.slider(
            "Decision Threshold", 0.10, 0.90, 0.50, 0.05,
            help="Probability above which a node is classified as flooded",
        )
        show_unc = st.toggle(
            "Show Epistemic Uncertainty",
            value=False,
            help="Switch between flood probability and MC Dropout uncertainty (σ)",
        )
        n_mc = st.select_slider(
            "MC Dropout Samples",
            options=[5, 10, 20, 50], value=5,
            help="More samples → more accurate uncertainty, but slower inference",
        )

        st.markdown("---")
        run_btn = st.button("🚀  Run Flood Forecast", use_container_width=True)

        st.markdown("---")
        with st.expander("ℹ️ Model Specifications"):
            if isinstance(model_meta, dict):
                st.markdown(f"""
**DS-STGAT Architecture**
- Checkpoint epoch: **{model_meta['epoch']}**
- Best val F1@1hr: **{model_meta['best_val_f1']:.4f}**
- Device: **{model_meta['device'].upper()}**
- Rain normalizer: **{rain_norm:.2f} mm/hr** (95th pct)

**Graph**
- Nodes: **{N:,}** · Edges: **{data['E']:,}**
- Static features: **{data.get('feature_dim', 16)}-dim** per node
- Feature cache: **{data.get('feature_source', 'cached')}**
- Edge features: **4-dim** (elev, length, type, flow)

**Novel Contributions**
1. Dual-scale GRU (6hr + 24hr windows)
2. Physics-informed edge features
3. Hybrid GATv2 + SAGEConv spatial encoder
4. 4-lead-time output (single forward pass)
5. MC Dropout uncertainty quantification
""")
                if model_meta.get("quality_note"):
                    st.warning(model_meta["quality_note"])
            if paper and "ds_stgat_2015" in paper:
                d = paper["ds_stgat_2015"]
                st.markdown(f"""
**Test Performance (2015 Event)**
| Lead | F1 | AUC-ROC | CSI |
|:----:|:--:|:-------:|:---:|
| 1hr  | {d['lead_1hr']['f1']:.3f} | {d['lead_1hr']['auroc']:.3f} | {d['lead_1hr']['csi']:.3f} |
| 3hr  | {d['lead_3hr']['f1']:.3f} | {d['lead_3hr']['auroc']:.3f} | {d['lead_3hr']['csi']:.3f} |
| 6hr  | {d['lead_6hr']['f1']:.3f} | {d['lead_6hr']['auroc']:.3f} | {d['lead_6hr']['csi']:.3f} |
| 12hr | {d['lead_12hr']['f1']:.3f} | {d['lead_12hr']['auroc']:.3f} | {d['lead_12hr']['csi']:.3f} |
""")

    # ── Run inference ─────────────────────────────────────────────────────────
    state_key = (scenario_name, intensity, duration, antecedent, pattern, n_mc, rain_norm)
    need_rerun = (
        run_btn
        or "probs" not in st.session_state
        or st.session_state.get("_key") != state_key
    )

    if need_rerun:
        with st.spinner("Running DS-STGAT inference + MC Dropout uncertainty…"):
            t0 = time.perf_counter()
            probs, unc = run_scenario(
                model, device, data,
                intensity=float(intensity),
                duration=float(duration),
                antecedent=float(antecedent),
                pattern=pattern,
                n_mc=n_mc,
                rain_norm=rain_norm,
            )
            elapsed = time.perf_counter() - t0

        st.session_state.update({
            "probs": probs, "unc": unc, "elapsed": elapsed, "_key": state_key,
        })
    else:
        probs   = st.session_state["probs"]
        unc     = st.session_state["unc"]
        elapsed = st.session_state.get("elapsed", 0.0)

    # ── Derived stats ─────────────────────────────────────────────────────────
    pct_risk    = float((probs[:, 0] >= threshold).mean()) * 100
    mean_p1     = float(probs[:, 0].mean())
    peak_p1     = float(probs[:, 0].max())
    mean_unc    = float(unc[:, 0].mean())
    n_flooded   = int((probs[:, 0] >= threshold).sum())

    # ── Alert Banner ──────────────────────────────────────────────────────────
    if pct_risk >= 50:
        st.markdown(
            f'<div class="flood-alert">⚠️  FLOOD WARNING — '
            f'{pct_risk:.1f}% of network nodes exceed risk threshold '
            f'({n_flooded:,} / {N:,} locations)</div>',
            unsafe_allow_html=True,
        )
    elif pct_risk >= 15:
        st.markdown(
            f'<div class="warn-status">⚡  ELEVATED RISK — '
            f'{pct_risk:.1f}% of nodes at or above threshold '
            f'({n_flooded:,} locations)</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="safe-status">✅  LOW FLOOD RISK — '
            f'{pct_risk:.1f}% of nodes exceed threshold '
            f'({n_flooded:,} locations)</div>',
            unsafe_allow_html=True,
        )
    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    # ── KPI Row ───────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("At-Risk Nodes (1hr)",    f"{pct_risk:.1f}%",   f"{n_flooded:,} of {N:,}")
    k2.metric("Mean Flood Prob (1hr)",  f"{mean_p1:.1%}")
    k3.metric("Peak Flood Prob",        f"{peak_p1:.1%}")
    k4.metric("Uncertainty (MC σ)",     f"{mean_unc:.4f}",
              help="Mean epistemic uncertainty — lower = more confident")
    k5.metric("Inference Time",         f"{elapsed*1000:.0f} ms",
              help=f"Single forward pass + {n_mc} MC Dropout samples")

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    # ── Map + Distributions ───────────────────────────────────────────────────
    map_col, dist_col = st.columns([3, 2], gap="medium")

    with map_col:
        st.markdown('<p class="section-title">Flood Risk Map — Interactive Multi-Lead Forecast</p>',
                    unsafe_allow_html=True)

        map_tabs = st.tabs([f"⏱️ {l}" for l in LEAD_LABELS])
        for i, (tab, h) in enumerate(zip(map_tabs, LEAD_TIMES)):
            with tab:
                fig_map = chart_map(
                    data["node_lats"], data["node_lons"], probs, unc,
                    lead_idx=i, threshold=threshold, show_unc=show_unc,
                    center_lat=center_lat, center_lon=center_lon,
                )
                st.plotly_chart(fig_map, use_container_width=True,
                                config={"displayModeBar": True, "modeBarButtonsToRemove": ["select2d", "lasso2d"]})

                pct_h = float((probs[:, i] >= threshold).mean()) * 100
                sa, sb, sc_ = st.columns(3)
                sa.metric(f"At-Risk ({h}hr)", f"{pct_h:.1f}%")
                sb.metric("Mean Prob",        f"{probs[:, i].mean():.1%}")
                sc_.metric("Peak Prob",       f"{probs[:, i].max():.1%}")

    with dist_col:
        st.markdown('<p class="section-title">Probability Distributions</p>',
                    unsafe_allow_html=True)

        st.plotly_chart(chart_violin(probs), use_container_width=True,
                        config={"displayModeBar": False})
        st.plotly_chart(chart_hist(probs, lead_idx=0), use_container_width=True,
                        config={"displayModeBar": False})

    st.markdown("---")

    # ── Performance & Generalisation ─────────────────────────────────────────
    st.markdown('<p class="section-title">Model Performance & Ablation Study</p>',
                unsafe_allow_html=True)

    row2_a, row2_b, row2_c = st.columns([4, 4, 3], gap="medium")

    with row2_a:
        fig_leads = chart_lead_performance(paper)
        if fig_leads:
            st.plotly_chart(fig_leads, use_container_width=True,
                            config={"displayModeBar": False})

    with row2_b:
        fig_bl = chart_baselines(paper)
        if fig_bl:
            st.plotly_chart(fig_bl, use_container_width=True,
                            config={"displayModeBar": False})

    with row2_c:
        fig_cross = chart_cross_event(paper)
        if fig_cross:
            st.plotly_chart(fig_cross, use_container_width=True,
                            config={"displayModeBar": False})
        else:
            if paper and "ds_stgat_2015" in paper:
                d = paper["ds_stgat_2015"]
                st.markdown("**Key Metrics (2015 event)**")
                for lbl, k in [("1hr", "lead_1hr"), ("3hr", "lead_3hr"),
                                ("6hr", "lead_6hr"), ("12hr", "lead_12hr")]:
                    st.markdown(
                        f"**{lbl}** — F1: `{d[k]['f1']:.3f}` · "
                        f"AUC: `{d[k]['auroc']:.3f}` · "
                        f"CSI: `{d[k]['csi']:.3f}`"
                    )

    st.markdown("---")

    # ── Top At-Risk Locations ─────────────────────────────────────────────────
    st.markdown('<p class="section-title">Top At-Risk Locations — Ranked by Mean Flood Probability</p>',
                unsafe_allow_html=True)

    df_top = make_top_risk_df(data["node_lats"], data["node_lons"], probs, unc, top_n=20)

    cat_colors = {
        "Very High": "color: #E74C3C; font-weight:700",
        "High":      "color: #E67E22; font-weight:700",
        "Moderate":  "color: #F1C40F; font-weight:700",
        "Low":       "color: #2ECC71",
    }
    style_category = lambda v: cat_colors.get(v, "")
    styler = df_top.style
    styled = (
        styler.map(style_category, subset=["Category"])
        if hasattr(styler, "map")
        else styler.applymap(style_category, subset=["Category"])
    )
    st.dataframe(styled, use_container_width=True, hide_index=True, height=420)

    dl_l, dl_r = st.columns([3, 1])
    with dl_l:
        csv_bytes = df_top.to_csv(index=False).encode()
        st.download_button(
            "📥 Download Risk Table (CSV)",
            data=csv_bytes,
            file_name=f"flood_risk_{scenario_name.lower().replace(' ', '_')[:40]}.csv",
            mime="text/csv",
        )
    with dl_r:
        if paper:
            paper_json = json.dumps(paper, indent=2).encode()
            st.download_button(
                "📄 Paper Metrics (JSON)",
                data=paper_json,
                file_name="ds_stgat_paper_results.json",
                mime="application/json",
            )

    # ── Historical 2015 Event Comparison ─────────────────────────────────────
    if hist_df is not None:
        st.markdown("---")
        st.markdown(
            '<p class="section-title">Historical Event: DS-STGAT Predictions on 2015 Chennai Flood (Test Set)</p>',
            unsafe_allow_html=True,
        )
        st.caption(
            "These are the actual model predictions on the held-out 2015 test period "
            "(mean probability over all test timesteps). Compare with your scenario above."
        )

        h_lead_col = "flood_prob_lead1h"
        h_max_col  = "flood_prob_max_lead1h"
        h_unc_col  = "flood_unc_lead1h"

        has_cols = all(c in hist_df.columns for c in ["lat", "lon", h_lead_col])
        if has_cols:
            hist_map_col, hist_stat_col = st.columns([3, 2], gap="medium")

            with hist_map_col:
                fig_hist_map = go.Figure(go.Scattermapbox(
                    lat=hist_df["lat"].values,
                    lon=hist_df["lon"].values,
                    mode="markers",
                    marker=go.scattermapbox.Marker(
                        size=7,
                        color=hist_df[h_lead_col].values,
                        colorscale=_FLOOD_COLORSCALE,
                        cmin=0, cmax=1,
                        showscale=True,
                        colorbar=dict(
                            title=dict(text="Mean Flood Prob (1hr)",
                                       font=dict(color="white", size=10)),
                            tickfont=dict(color="white"),
                            bgcolor="rgba(0,0,0,0)",
                            thickness=12, len=0.7,
                        ),
                        opacity=0.88,
                    ),
                    text=[
                        (f"<b>Mean Prob (1hr):</b> {row[h_lead_col]:.1%}<br>"
                         f"<b>Peak Prob (1hr):</b> {row.get(h_max_col, 0):.1%}<br>"
                         f"<b>Risk:</b> {_risk_cat(row[h_lead_col])}")
                        for _, row in hist_df.iterrows()
                    ],
                    hovertemplate="%{text}<extra></extra>",
                ))
                fig_hist_map.update_layout(
                    mapbox=dict(style="carto-darkmatter",
                                center=dict(lat=center_lat, lon=center_lon),
                                zoom=12),
                    margin=dict(l=0, r=0, t=0, b=0),
                    height=420, paper_bgcolor="#0D0D1A", showlegend=False,
                )
                st.plotly_chart(fig_hist_map, use_container_width=True,
                                config={"displayModeBar": False})

            with hist_stat_col:
                st.markdown("**2015 Test-Set Statistics (1hr Lead)**")
                h_pct = float((hist_df[h_lead_col] >= 0.5).mean()) * 100
                h_mean = float(hist_df[h_lead_col].mean())
                h_peak = float(hist_df[h_lead_col].max())
                h_unc  = float(hist_df[h_unc_col].mean()) if h_unc_col in hist_df.columns else 0.0

                st.metric("At-Risk Nodes",   f"{h_pct:.1f}%")
                st.metric("Mean Flood Prob", f"{h_mean:.1%}")
                st.metric("Peak Flood Prob", f"{h_peak:.1%}")
                st.metric("Mean Uncertainty σ", f"{h_unc:.4f}")

                # Scatter: Scenario vs Historical
                scenario_p = probs[:, 0]
                if len(scenario_p) == len(hist_df):
                    hist_p = hist_df[h_lead_col].values
                    fig_scatter = go.Figure(go.Scatter(
                        x=hist_p, y=scenario_p,
                        mode="markers",
                        marker=dict(
                            size=3, color=hist_p,
                            colorscale=_FLOOD_COLORSCALE, cmin=0, cmax=1,
                            opacity=0.6,
                        ),
                        hovertemplate="Historical: %{x:.3f}<br>Scenario: %{y:.3f}<extra></extra>",
                    ))
                    fig_scatter.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1], mode="lines",
                        line=dict(color="white", dash="dash", width=1),
                        showlegend=False,
                    ))
                    fig_scatter.update_layout(
                        title=dict(text="Scenario vs Historical (per node)",
                                   font=dict(color="white", size=11), x=0),
                        xaxis=dict(title="Historical Mean Prob", color="white",
                                   gridcolor="#2A2A4A", range=[0, 1]),
                        yaxis=dict(title="Scenario Prob", color="white",
                                   gridcolor="#2A2A4A", range=[0, 1]),
                        paper_bgcolor="#1C1C2E", plot_bgcolor="#1C1C2E",
                        height=280, margin=dict(l=50, r=10, t=40, b=40),
                        font=dict(color="white"),
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True,
                                    config={"displayModeBar": False})
        else:
            st.info("Historical prediction columns not found in CSV — run `python main.py` to regenerate.")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""
<hr style="border-color:#2A2A4A;margin:20px 0 12px 0;">
<div style="text-align:center;color:#404060;font-size:0.75rem;line-height:1.8;">
  <b style="color:#5050A0;">Hydro-Graph DS-STGAT</b> &nbsp;|&nbsp;
  Dual-Scale Spatiotemporal Graph Attention Network for Urban Flood Forecasting<br>
  Trained on 2015 Chennai Flood (Nov–Dec) · Cross-validated on 2018 NE Monsoon Analogue<br>
  Architecture: GATv2Conv × 2 + SAGEConv + Dual GRU + Temporal Attention Gate · ~250K parameters<br>
  <span style="color:#3A3A6A;">CE23B092 · IIT Madras · Department of Civil Engineering</span>
</div>
""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
