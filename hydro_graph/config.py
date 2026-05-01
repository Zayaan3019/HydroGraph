"""
Pydantic v2 validated configuration for the Hydro-Graph v2 pipeline.
Loads and validates config/config.yaml; provides typed access throughout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# ─── Sub-models ───────────────────────────────────────────────────────────────

class RiverConfig(BaseModel):
    name: str
    lat: float
    lon: float


class StudyAreaConfig(BaseModel):
    name: str
    bbox_demo: Tuple[float, float, float, float]
    bbox_full: Tuple[float, float, float, float]
    crs_geographic: str = "EPSG:4326"
    crs_projected: str = "EPSG:32644"
    center_lat: float = 13.0127
    center_lon: float = 80.2600
    coast_lat: float = 13.0828
    coast_lon: float = 80.2900
    rivers: List[RiverConfig] = []

    @field_validator("bbox_demo", "bbox_full")
    @classmethod
    def validate_bbox(cls, v: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        left, bottom, right, top = v
        assert left < right, "bbox left must be < right (west < east)"
        assert bottom < top, "bbox bottom must be < top (south < north)"
        return v


class GraphConfig(BaseModel):
    network_type: str = "all"
    simplify: bool = True
    retain_all: bool = False
    waterway_filter: str = '["waterway"~"drain|canal|river|stream|ditch"]'
    min_edge_length_m: float = 1.0
    max_edge_length_m: float = 5000.0
    edge_feature_dim: int = 4


class FeaturesConfig(BaseModel):
    static_dim: int = 16
    short_seq_len: int = 6
    long_seq_len: int = 12
    input_dim: int = 34
    dem_tif: str = "data/raw/srtm_chennai.tif"
    sentinel2_tif: str = "data/raw/sentinel2_chennai.tif"
    sentinel1_tif: str = "data/raw/sentinel1_chennai.tif"
    rainfall_csv: str = "data/raw/gpm_imerg_chennai.csv"
    use_synthetic: bool = True
    srtm_cache_dir: str = "data/raw/srtm_cache"

    @model_validator(mode="after")
    def check_input_dim(self) -> "FeaturesConfig":
        expected = self.static_dim + self.short_seq_len + self.long_seq_len
        assert self.input_dim == expected, (
            f"input_dim ({self.input_dim}) must equal "
            f"static_dim + short_seq_len + long_seq_len = "
            f"{self.static_dim}+{self.short_seq_len}+{self.long_seq_len}={expected}"
        )
        return self


class StormEventConfig(BaseModel):
    name: str
    event_start: str
    event_end: str
    peak_start: str
    peak_end: str
    peak_rainfall_mm_hr: float = 55.0


class TemporalConfig(BaseModel):
    train_event: StormEventConfig
    val_event: StormEventConfig
    timestep_hours: int = 1
    lead_times: List[int] = [1, 3, 6, 12]
    flood_threshold_min_mm: float = 15.0
    flood_threshold_max_mm: float = 80.0
    flood_recovery_halflife_hr: float = 4.0


class ModelConfig(BaseModel):
    name: str = "DS-STGAT"
    static_dim: int = 16
    short_seq_len: int = 6
    long_seq_len: int = 12
    edge_dim: int = 4
    n_lead_times: int = 4
    static_hidden: int = 128
    static_out: int = 64
    short_gru_hidden: int = 96
    short_gru_layers: int = 2
    long_gru_hidden: int = 64
    long_gru_layers: int = 1
    temporal_att_hidden: int = 96
    fusion_hidden: int = 128
    gat_hidden: int = 128
    gat_heads_l1: int = 8
    gat_heads_l2: int = 4
    gat_layers: int = 2
    sage_hidden: int = 64
    sage_layers: int = 1
    output_hidden: int = 32
    dropout: float = Field(0.25, ge=0.0, le=1.0)


class TrainingConfig(BaseModel):
    seed: int = 42
    epochs: int = 120
    batch_size: int = 512
    lr: float = Field(5e-4, gt=0)
    weight_decay: float = 1e-4
    tversky_alpha: float = Field(0.30, ge=0.0, le=1.0)
    tversky_beta: float = Field(0.70, ge=0.0, le=1.0)
    tversky_gamma: float = Field(0.75, ge=0.0)
    lead_weights: List[float] = [2.0, 1.5, 1.0, 0.75]
    train_ratio: float = Field(0.85, gt=0, lt=1)
    val_ratio: float = Field(0.075, gt=0, lt=1)
    num_neighbors: List[int] = [15, 10, 5]
    early_stopping_patience: int = 20
    threshold: float = Field(0.50, ge=0.0, le=1.0)
    mc_dropout_samples: int = 20
    lr_min: float = 1e-6
    run_baselines: bool = True
    baseline_epochs: int = 50

    @model_validator(mode="after")
    def check_split_ratios(self) -> "TrainingConfig":
        total = self.train_ratio + self.val_ratio
        assert total < 1.0, f"train+val ratios ({total}) must be < 1.0"
        return self


class PathsConfig(BaseModel):
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    models_dir: str = "data/models"
    outputs_dir: str = "data/outputs"
    graph_gpickle: str = "data/processed/chennai_graph.gpickle"
    node_features: str = "data/processed/node_features.parquet"
    temporal_data: str = "data/processed/temporal_data.npz"
    temporal_val_data: str = "data/processed/temporal_val_data.npz"
    edge_features: str = "data/processed/edge_features.npy"
    best_checkpoint: str = "data/models/best_model.pt"
    last_checkpoint: str = "data/models/last_model.pt"
    baseline_rf: str = "data/models/baseline_rf.pkl"
    baseline_lstm: str = "data/models/baseline_lstm.pt"
    risk_map_html: str = "data/outputs/flood_risk_map.html"
    risk_map_png: str = "data/outputs/flood_risk_map.png"
    uncertainty_map_png: str = "data/outputs/uncertainty_map.png"
    training_curves: str = "data/outputs/training_curves.png"
    calibration_plot: str = "data/outputs/calibration_curve.png"
    metrics_json: str = "data/outputs/eval_metrics.json"
    baselines_json: str = "data/outputs/baseline_metrics.json"
    predictions_csv: str = "data/outputs/node_predictions.csv"


# ─── Master Config ────────────────────────────────────────────────────────────

class HydroGraphConfig(BaseModel):
    study_area: StudyAreaConfig
    graph: GraphConfig
    features: FeaturesConfig
    temporal: TemporalConfig
    model: ModelConfig
    training: TrainingConfig
    paths: PathsConfig
    cache_version: str = "2.1.0"

    def get_bbox(self, mode: str = "demo") -> Tuple[float, float, float, float]:
        """Return bounding box in osmnx 2.x format: (left, bottom, right, top)."""
        if mode == "full":
            return tuple(self.study_area.bbox_full)  # type: ignore
        return tuple(self.study_area.bbox_demo)  # type: ignore

    def resolve_path(self, rel_path: str, base: Optional[Path] = None) -> Path:
        base = base or Path.cwd()
        p = Path(rel_path)
        return p if p.is_absolute() else base / p

    def ensure_dirs(self, base: Optional[Path] = None) -> None:
        base = base or Path.cwd()
        for rel in [
            self.paths.raw_dir,
            self.paths.processed_dir,
            self.paths.models_dir,
            self.paths.outputs_dir,
        ]:
            (base / rel).mkdir(parents=True, exist_ok=True)

    @property
    def lead_times(self) -> List[int]:
        return self.temporal.lead_times

    @property
    def max_lead(self) -> int:
        return max(self.temporal.lead_times)


# ─── Loader ───────────────────────────────────────────────────────────────────

def load_config(config_path: Optional[str] = None) -> HydroGraphConfig:
    """Load and validate config from YAML. Falls back to defaults if not found."""
    if config_path is None:
        cwd = Path.cwd()
        candidates = [
            cwd / "config" / "config.yaml",
            cwd / "config.yaml",
            Path(__file__).parent.parent / "config" / "config.yaml",
        ]
        for candidate in candidates:
            if candidate.exists():
                config_path = str(candidate)
                break

    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        return HydroGraphConfig(**raw)

    # Sensible defaults when no YAML present
    return HydroGraphConfig(
        study_area=StudyAreaConfig(
            name="Chennai Metropolitan Region",
            bbox_demo=(80.20, 12.95, 80.32, 13.07),
            bbox_full=(79.90, 12.70, 80.40, 13.30),
            rivers=[
                RiverConfig(name="Adyar", lat=12.9716, lon=80.2426),
                RiverConfig(name="Cooum", lat=13.0827, lon=80.2707),
            ],
        ),
        graph=GraphConfig(),
        features=FeaturesConfig(),
        temporal=TemporalConfig(
            train_event=StormEventConfig(
                name="2015_Chennai_Flood",
                event_start="2015-11-01T00:00:00",
                event_end="2015-12-05T23:00:00",
                peak_start="2015-11-28T00:00:00",
                peak_end="2015-12-03T00:00:00",
                peak_rainfall_mm_hr=55.0,
            ),
            val_event=StormEventConfig(
                name="2018_Chennai_Analogue",
                event_start="2018-10-15T00:00:00",
                event_end="2018-11-10T23:00:00",
                peak_start="2018-10-25T00:00:00",
                peak_end="2018-10-30T00:00:00",
                peak_rainfall_mm_hr=42.0,
            ),
        ),
        model=ModelConfig(),
        training=TrainingConfig(),
        paths=PathsConfig(),
    )
