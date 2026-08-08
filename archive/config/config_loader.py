"""Configuration loader with Pydantic validation."""

from pathlib import Path
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator
import yaml


class ProjectConfig(BaseModel):
    """Project metadata configuration."""
    name: str
    version: str
    description: str


class LocationConfig(BaseModel):
    """Location and CRS configuration."""
    city: str
    country: str
    bbox: List[float] = Field(..., min_items=4, max_items=4)
    target_crs: str
    wgs84_crs: str = "EPSG:4326"
    
    @validator('bbox')
    def validate_bbox(cls, v):
        """Validate bounding box format [South, West, North, East]."""
        if v[0] >= v[2] or v[1] >= v[3]:
            raise ValueError("Invalid bounding box: [South, West, North, East]")
        return v


class GraphConfig(BaseModel):
    """Graph construction configuration."""
    network_type: str
    custom_filters: List[str]
    simplify: bool
    retain_all: bool


class FeaturesConfig(BaseModel):
    """Feature engineering configuration."""
    static: Dict[str, List[str]]
    temporal: Dict[str, Any]


class DataConfig(BaseModel):
    """Data paths and directories configuration."""
    raw_dir: str
    processed_dir: str
    graph_dir: str
    raster_dir: str
    rasters: Dict[str, str]
    precipitation: Dict[str, str]


class ModelArchitectureConfig(BaseModel):
    """Model architecture configuration."""
    name: str
    spatial: Dict[str, Any]
    temporal: Dict[str, Any]
    fusion: Dict[str, Any]
    output: Dict[str, str]


class NeighborSamplingConfig(BaseModel):
    """Neighbor sampling configuration for mini-batching."""
    num_neighbors: List[int]
    batch_size: int
    num_workers: int


class ModelConfig(BaseModel):
    """Complete model configuration."""
    architecture: ModelArchitectureConfig
    neighbor_sampling: NeighborSamplingConfig


class TrainingConfig(BaseModel):
    """Training configuration."""
    epochs: int
    early_stopping_patience: int
    learning_rate: float
    weight_decay: float
    scheduler: Dict[str, Any]
    loss: Dict[str, Any]
    split: Dict[str, float]
    metrics: List[str]


class InferenceConfig(BaseModel):
    """Inference configuration."""
    checkpoint_path: str
    output_dir: str
    visualization: Dict[str, Any]


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str
    log_dir: str
    tensorboard: bool
    wandb: Dict[str, Any]


class EventConfig(BaseModel):
    """Storm event configuration."""
    name: str
    start_date: str
    end_date: str
    peak_date: str
    description: str


class Config(BaseModel):
    """Main configuration class."""
    project: ProjectConfig
    location: LocationConfig
    graph: GraphConfig
    features: FeaturesConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    inference: InferenceConfig
    logging: LoggingConfig
    events: Dict[str, EventConfig]


def load_config(config_path: Optional[Path] = None) -> Config:
    """
    Load and validate configuration from YAML file.
    
    Parameters
    ----------
    config_path : Optional[Path]
        Path to configuration file. If None, uses default config.yaml
        
    Returns
    -------
    Config
        Validated configuration object
        
    Raises
    ------
    FileNotFoundError
        If configuration file doesn't exist
    ValueError
        If configuration validation fails
    """
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return Config(**config_dict)
