"""
Real-time Flood Prediction Script

Generates flood risk predictions and interactive maps.
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime, timedelta

import torch
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_config
from src import (
    GraphConstructor,
    FloodPredictor,
    GeospatialVisualizer,
    HydroGraphSTGNN,
    RainfallDataLoader,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate flood predictions")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--forecast-date",
        type=str,
        default=None,
        help="Forecast date (YYYY-MM-DD), defaults to today",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/predictions",
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu)",
    )
    
    return parser.parse_args()


def main():
    """Main prediction pipeline."""
    args = parse_args()
    
    logger.info("=" * 80)
    logger.info("FLOOD RISK PREDICTION")
    logger.info("=" * 80)
    
    # Load configuration
    config = load_config(Path(args.config))
    
    # Determine device
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Forecast date
    if args.forecast_date:
        forecast_date = datetime.strptime(args.forecast_date, '%Y-%m-%d')
    else:
        forecast_date = datetime.now()
    
    logger.info(f"Forecast date: {forecast_date.strftime('%Y-%m-%d')}")
    logger.info(f"Target: {config.location.city}, {config.location.country}")
    
    # =========================================================================
    # LOAD GRAPH
    # =========================================================================
    logger.info("\nLoading urban graph...")
    
    graph_dir = Path(config.data.graph_dir)
    constructor = GraphConstructor(
        bbox=config.location.bbox,
        target_crs=config.location.target_crs,
    )
    
    try:
        graph = constructor.load_graph(graph_dir)
    except FileNotFoundError:
        logger.error(f"Graph not found at {graph_dir}")
        logger.error("Please run: python pipeline/train.py")
        sys.exit(1)
    
    logger.success(f"Graph loaded: {graph.number_of_nodes()} nodes")
    
    # =========================================================================
    # LOAD MODEL
    # =========================================================================
    logger.info("\nLoading trained model...")
    
    # Determine number of features
    sample_node = list(graph.nodes(data=True))[0]
    feature_names = [k for k in sample_node[1].keys() if k not in ['x', 'y', 'lon', 'lat', 'osmid']]
    num_static_features = len(feature_names)
    
    model = HydroGraphSTGNN(
        num_static_features=num_static_features,
        lag_window=config.features.temporal['lag_window'],
        spatial_config=config.model.architecture.spatial,
        temporal_config=config.model.architecture.temporal,
        fusion_config=config.model.architecture.fusion,
    )
    
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.success(f"Model loaded from: {args.checkpoint}")
        logger.info(f"Trained for {checkpoint.get('epoch', 'N/A')} epochs")
    except FileNotFoundError:
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        logger.error("Please run: python pipeline/train.py")
        sys.exit(1)
    
    # =========================================================================
    # LOAD RAINFALL DATA
    # =========================================================================
    logger.info("\nLoading rainfall data...")
    
    rainfall_loader = RainfallDataLoader(
        gpm_dir=Path(config.data.precipitation.get('gpm_imerg_dir', 'data/raw/gpm_imerg')),
        imd_file=Path(config.data.precipitation.get('imd_gauge_file', 'data/raw/imd_chennai_2015.csv'))
                    if config.data.precipitation.get('imd_gauge_file') else None,
        bbox=config.location.bbox,
    )
    
    # Get rainfall for lag window
    lag_hours = config.features.temporal['lag_window']
    rainfall_data = []
    
    logger.info(f"Collecting {lag_hours}-hour rainfall window...")
    
    for h in range(lag_hours):
        target_time = forecast_date - timedelta(hours=h)
        
        try:
            # Try to load actual rainfall
            hourly_rain = rainfall_loader.load_gpm_hourly(target_time)
            if hourly_rain is not None:
                rainfall_data.insert(0, hourly_rain)
            else:
                logger.warning(f"No rainfall data for {target_time}, using zeros")
                rainfall_data.insert(0, torch.zeros(graph.number_of_nodes()))
        except Exception as e:
            logger.warning(f"Error loading rainfall at {target_time}: {e}")
            rainfall_data.insert(0, torch.zeros(graph.number_of_nodes()))
    
    rainfall_tensor = torch.stack(rainfall_data)  # Shape: [lag_window, num_nodes]
    
    logger.info(f"Rainfall tensor shape: {rainfall_tensor.shape}")
    logger.info(f"Total rainfall (mm): {rainfall_tensor.sum().item():.2f}")
    logger.info(f"Max rainfall (mm): {rainfall_tensor.max().item():.2f}")
    
    # =========================================================================
    # GENERATE PREDICTIONS
    # =========================================================================
    logger.info("\nGenerating predictions...")
    
    predictor = FloodPredictor(model, device)
    predictions = predictor.predict(graph, rainfall_tensor)
    
    logger.success(f"Predictions generated: {len(predictions)} locations")
    
    # Prediction statistics
    pred_array = predictions.numpy()
    logger.info("\n" + "=" * 80)
    logger.info("PREDICTION STATISTICS")
    logger.info("=" * 80)
    logger.info(f"Mean risk:         {pred_array.mean():.4f}")
    logger.info(f"Std risk:          {pred_array.std():.4f}")
    logger.info(f"Max risk:          {pred_array.max():.4f}")
    logger.info(f"Very High (>0.8):  {(pred_array > 0.8).sum()} locations")
    logger.info(f"High (0.6-0.8):    {((pred_array >= 0.6) & (pred_array <= 0.8)).sum()} locations")
    logger.info(f"Medium (0.4-0.6):  {((pred_array >= 0.4) & (pred_array < 0.6)).sum()} locations")
    logger.info(f"Low (0.2-0.4):     {((pred_array >= 0.2) & (pred_array < 0.4)).sum()} locations")
    logger.info(f"Very Low (<0.2):   {(pred_array < 0.2).sum()} locations")
    
    # =========================================================================
    # VISUALIZE RESULTS
    # =========================================================================
    logger.info("\nGenerating interactive map...")
    
    visualizer = GeospatialVisualizer(graph, constructor.node_gdf)
    
    map_path = output_dir / f"flood_risk_map_{forecast_date.strftime('%Y%m%d')}.html"
    flood_map = visualizer.create_flood_map(predictions, save_path=str(map_path))
    
    logger.success(f"Interactive map saved: {map_path}")
    
    # Save predictions as CSV
    import pandas as pd
    
    nodes_data = []
    for i, (node_id, node_data) in enumerate(graph.nodes(data=True)):
        nodes_data.append({
            'node_id': node_id,
            'lon': node_data.get('lon', node_data.get('x')),
            'lat': node_data.get('lat', node_data.get('y')),
            'flood_risk': pred_array[i],
            'risk_category': visualizer.categorize_risk(pred_array[i]),
        })
    
    predictions_df = pd.DataFrame(nodes_data)
    csv_path = output_dir / f"predictions_{forecast_date.strftime('%Y%m%d')}.csv"
    predictions_df.to_csv(csv_path, index=False)
    
    logger.success(f"Predictions CSV saved: {csv_path}")
    
    # Identify high-risk locations
    high_risk = predictions_df[predictions_df['flood_risk'] > 0.7].sort_values('flood_risk', ascending=False)
    
    if len(high_risk) > 0:
        logger.warning("\n" + "=" * 80)
        logger.warning(f"⚠️  {len(high_risk)} HIGH-RISK LOCATIONS IDENTIFIED")
        logger.warning("=" * 80)
        
        logger.warning(f"\nTop 10 highest risk locations:")
        for idx, row in high_risk.head(10).iterrows():
            logger.warning(f"  Node {row['node_id']}: {row['flood_risk']:.4f} - {row['risk_category']}")
            logger.warning(f"    Location: ({row['lat']:.4f}, {row['lon']:.4f})")
        
        # Save high-risk locations separately
        high_risk_path = output_dir / f"high_risk_locations_{forecast_date.strftime('%Y%m%d')}.csv"
        high_risk.to_csv(high_risk_path, index=False)
        logger.warning(f"\nHigh-risk locations saved: {high_risk_path}")
    
    logger.info("\n" + "=" * 80)
    logger.success("PREDICTION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nOutputs:")
    logger.info(f"  - Interactive map: {map_path}")
    logger.info(f"  - Predictions CSV: {csv_path}")
    if len(high_risk) > 0:
        logger.info(f"  - High-risk locations: {high_risk_path}")


if __name__ == "__main__":
    main()
