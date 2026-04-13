"""
Complete Production Deployment Guide

This script orchestrates the complete end-to-end production pipeline
from data acquisition to model deployment - NO EXAMPLES, REAL DATA ONLY.
"""

import sys
from pathlib import Path
from datetime import datetime
import subprocess

from loguru import logger


def run_command(cmd: str, description: str, critical: bool = True) -> bool:
    """
    Execute a command and log results.
    
    Args:
        cmd: Command to execute
        description: Human-readable description
        critical: Whether failure should stop the pipeline
    
    Returns:
        Success status
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"STEP: {description}")
    logger.info(f"{'='*80}")
    logger.info(f"Command: {cmd}")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )
        
        if result.stdout:
            logger.info(result.stdout)
        
        if result.returncode == 0:
            logger.success(f"✓ {description} completed successfully")
            return True
        else:
            logger.error(f"✗ {description} failed")
            if result.stderr:
                logger.error(result.stderr)
            
            if critical:
                logger.error("Critical step failed. Stopping pipeline.")
                sys.exit(1)
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"✗ {description} timed out after 1 hour")
        if critical:
            sys.exit(1)
        return False
    except Exception as e:
        logger.error(f"✗ {description} failed with exception: {e}")
        if critical:
            sys.exit(1)
        return False


def main():
    """Complete production deployment pipeline."""
    
    logger.info("\n" + "=" * 80)
    logger.info("HYDRO-GRAPH PRODUCTION DEPLOYMENT PIPELINE")
    logger.info("Real Spatiotemporal GNN for Urban Flood Forecasting")
    logger.info("=" * 80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Working directory: {Path.cwd()}")
    
    # =========================================================================
    # PHASE 0: ENVIRONMENT SETUP
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 0: ENVIRONMENT VALIDATION")
    logger.info("=" * 80)
    
    # Check Python version
    import platform
    logger.info(f"Python: {platform.python_version()}")
    logger.info(f"OS: {platform.system()} {platform.release()}")
    
    # Check critical dependencies
    try:
        import torch
        logger.info(f"PyTorch: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        logger.error("PyTorch not installed!")
        logger.error("Install: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        sys.exit(1)
    
    try:
        import torch_geometric
        logger.info(f"PyTorch Geometric: {torch_geometric.__version__}")
    except ImportError:
        logger.error("PyTorch Geometric not installed!")
        logger.error("Install: pip install torch-geometric")
        sys.exit(1)
    
    essential_packages = [
        'osmnx', 'networkx', 'rasterio', 'geopandas',
        'loguru', 'pyyaml', 'pydantic', 'folium'
    ]
    
    for package in essential_packages:
        try:
            __import__(package)
            logger.success(f"✓ {package} installed")
        except ImportError:
            logger.error(f"✗ {package} NOT installed")
            logger.error(f"Install: pip install {package}")
            sys.exit(1)
    
    logger.success("All dependencies installed")
    
    # =========================================================================
    # PHASE 1: DATA ACQUISITION
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 1: DATA ACQUISITION")
    logger.info("=" * 80)
    logger.warning("This phase requires MANUAL data download due to authentication requirements")
    logger.info("\nRequired data sources:")
    logger.info("  1. SRTM DEM (30m) - USGS EarthExplorer")
    logger.info("  2. Sentinel-2 L2A - Copernicus Data Space")
    logger.info("  3. Sentinel-1 GRD - Copernicus Data Space")
    logger.info("  4. GPM IMERG - NASA GES DISC")
    
    # Validate data availability
    run_command(
        "python pipeline/validate_data.py",
        "Data validation check",
        critical=True
    )
    
    logger.info("\n📋 DATA ACQUISITION INSTRUCTIONS:")
    logger.info("\nIf validation failed, acquire data using:")
    logger.info("  python pipeline/acquire_data.py --help")
    logger.info("\nFollow the instructions in acquire_data.py to download:")
    logger.info("  - DEM tiles for your region")
    logger.info("  - Sentinel-2 imagery (bands 2,3,4,8,11,12)")
    logger.info("  - Sentinel-1 SAR (VV polarization)")
    logger.info("  - GPM IMERG rainfall data")
    
    user_input = input("\n✓ Have you acquired all required data? (yes/no): ")
    if user_input.lower() != 'yes':
        logger.warning("Please acquire data before continuing")
        logger.info("Run: python pipeline/acquire_data.py")
        sys.exit(0)
    
    # =========================================================================
    # PHASE 2: GRAPH CONSTRUCTION & FEATURE EXTRACTION
    # =========================================================================
    # This is handled automatically by train.py
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2-3: GRAPH & FEATURES (automated in training)")
    logger.info("=" * 80)
    logger.info("Graph construction and feature extraction will be automated")
    logger.info("during the training pipeline (Phase 1-2 of training script)")
    
    # =========================================================================
    # PHASE 3: MODEL TRAINING
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 4: MODEL TRAINING")
    logger.info("=" * 80)
    
    user_input = input("\nProceed with training? (yes/no): ")
    if user_input.lower() == 'yes':
        run_command(
            "python pipeline/train.py --config config/config.yaml",
            "Model training",
            critical=True
        )
    else:
        logger.info("Skipping training. Use existing checkpoint.")
    
    # =========================================================================
    # PHASE 4: MODEL EVALUATION
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 5: MODEL EVALUATION")
    logger.info("=" * 80)
    
    user_input = input("\nRun model evaluation? (yes/no): ")
    if user_input.lower() == 'yes':
        run_command(
            "python pipeline/evaluate.py --checkpoint checkpoints/best_model.pth",
            "Model evaluation",
            critical=False
        )
    
    # =========================================================================
    # PHASE 5: PREDICTION GENERATION
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 6: FLOOD RISK PREDICTION")
    logger.info("=" * 80)
    
    user_input = input("\nGenerate flood risk predictions? (yes/no): ")
    if user_input.lower() == 'yes':
        run_command(
            "python pipeline/predict.py --checkpoint checkpoints/best_model.pth",
            "Flood risk prediction",
            critical=False
        )
    
    # =========================================================================
    # DEPLOYMENT COMPLETE
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.success("PRODUCTION DEPLOYMENT COMPLETE")
    logger.info("=" * 80)
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    logger.info("\n📂 OUTPUT STRUCTURE:")
    logger.info("  data/")
    logger.info("    ├── processed/           # Graph and engineered features")
    logger.info("    └── raw/                 # Downloaded satellite/DEM data")
    logger.info("  checkpoints/")
    logger.info("    └── best_model.pth       # Trained model")
    logger.info("  outputs/")
    logger.info("    ├── evaluation/          # Test metrics")
    logger.info("    ├── predictions/         # Flood risk maps & CSVs")
    logger.info("    └── training_history.csv # Training curves")
    logger.info("  logs/                      # Execution logs")
    
    logger.info("\n🚀 NEXT STEPS:")
    logger.info("  1. Review predictions: outputs/predictions/flood_risk_map_*.html")
    logger.info("  2. Analyze metrics: outputs/evaluation/evaluation_metrics.csv")
    logger.info("  3. Deploy model: Serve predictions via API (see deployment docs)")
    logger.info("  4. Set up monitoring: Track model performance over time")
    logger.info("  5. Schedule retraining: Update model with new flood events")
    
    logger.info("\n📊 OPERATIONAL USE:")
    logger.info("  # Generate daily predictions:")
    logger.info("  python pipeline/predict.py --forecast-date 2024-01-15")
    logger.info("\n  # Retrain with new data:")
    logger.info("  python pipeline/train.py --epochs 50 --rebuild-features")
    logger.info("\n  # Evaluate new checkpoint:")
    logger.info("  python pipeline/evaluate.py --checkpoint path/to/model.pth")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\n\nPipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n\nPipeline failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
