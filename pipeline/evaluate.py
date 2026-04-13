"""
Model Evaluation Script

Evaluates trained model on test set and generates comprehensive metrics.
"""

import sys
from pathlib import Path
import argparse

import torch
from torch_geometric.loader import NeighborLoader
from loguru import logger
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_config
from src import FloodPredictor, HydroGraphSTGNN
from src.trainer import MetricsTracker


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Hydro-Graph ST-GNN")
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
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/evaluation",
        help="Output directory",
    )
    
    return parser.parse_args()


def main():
    """Main evaluation pipeline."""
    args = parse_args()
    
    logger.info("=" * 80)
    logger.info("MODEL EVALUATION")
    logger.info("=" * 80)
    
    # Load configuration
    config = load_config(Path(args.config))
    
    # Determine device
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test dataset
    logger.info("\nLoading test dataset...")
    
    # Load graph and dataset (assuming already created during training)
    from src import GraphConstructor, HydroGraphDataset, RainfallDataLoader, TemporalDatasetCreator
    from datetime import datetime
    
    graph_dir = Path(config.data.graph_dir)
    constructor = GraphConstructor(
        bbox=config.location.bbox,
        target_crs=config.location.target_crs,
    )
    graph = constructor.load_graph(graph_dir)
    
    # Recreate dataset
    event_name = list(config.events.keys())[0]
    event_config = config.events[event_name]
    
    rainfall_loader = RainfallDataLoader(
        gpm_dir=Path(config.data.precipitation.get('gpm_imerg_dir', 'data/raw/gpm_imerg')),
        imd_file=Path(config.data.precipitation.get('imd_gauge_file', 'data/raw/imd_chennai_2015.csv'))
                    if config.data.precipitation.get('imd_gauge_file') else None,
        bbox=config.location.bbox,
    )
    
    dataset_creator = TemporalDatasetCreator(
        graph=graph,
        rainfall_loader=rainfall_loader,
        lag_window=config.features.temporal['lag_window'],
    )
    
    start_date = datetime.strptime(event_config.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(event_config.end_date, '%Y-%m-%d')
    data_list = dataset_creator.create_temporal_sequence(start_date, end_date)
    
    dataset = HydroGraphDataset(data_list)
    _, _, test_data = dataset.train_val_test_split(
        train_ratio=config.training.split['train'],
        val_ratio=config.training.split['val'],
        test_ratio=config.training.split['test'],
    )
    
    logger.info(f"Test set size: {len(test_data)} samples")
    
    # Load model
    logger.info(f"\nLoading model from: {args.checkpoint}")
    
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
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.success("Model loaded successfully")
    logger.info(f"Trained for {checkpoint.get('epoch', 'unknown')} epochs")
    logger.info(f"Best val loss: {checkpoint.get('best_val_loss', 'unknown')}")
    
    # Evaluation
    logger.info("\nRunning evaluation...")
    
    metrics_tracker = MetricsTracker(device=device)
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for i, data_obj in enumerate(test_data.data_list):
            data_obj = data_obj.to(device)
            output = model(data_obj.x, data_obj.edge_index)
            
            all_preds.append(output.cpu())
            all_targets.append(data_obj.y.cpu())
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i+1}/{len(test_data)} samples...")
    
    # Concatenate predictions
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0).long()
    
    # Compute metrics
    logger.info("\nComputing metrics...")
    metrics_tracker.update(all_preds, all_targets)
    metrics = metrics_tracker.compute()
    
    # Display results
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)
    logger.info(f"F1 Score:          {metrics['f1_score']:.4f}")
    logger.info(f"Precision:         {metrics['precision']:.4f}")
    logger.info(f"Recall:            {metrics['recall']:.4f}")
    logger.info(f"ROC-AUC:           {metrics['roc_auc']:.4f}")
    logger.info(f"Average Precision: {metrics['avg_precision']:.4f}")
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_path = output_dir / "evaluation_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.success(f"\nMetrics saved to: {metrics_path}")
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'prediction': all_preds.numpy().flatten(),
        'ground_truth': all_targets.numpy().flatten(),
    })
    predictions_path = output_dir / "predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    logger.success(f"Predictions saved to: {predictions_path}")
    
    # Summary statistics
    logger.info("\n" + "=" * 80)
    logger.info("PREDICTION STATISTICS")
    logger.info("=" * 80)
    preds_array = all_preds.numpy().flatten()
    logger.info(f"Mean prediction:   {preds_array.mean():.4f}")
    logger.info(f"Std prediction:    {preds_array.std():.4f}")
    logger.info(f"Min prediction:    {preds_array.min():.4f}")
    logger.info(f"Max prediction:    {preds_array.max():.4f}")
    logger.info(f"High risk (>0.7):  {(preds_array > 0.7).sum()} / {len(preds_array)}")
    logger.info(f"Medium risk (0.5-0.7): {((preds_array >= 0.5) & (preds_array <= 0.7)).sum()} / {len(preds_array)}")
    
    logger.info("\n" + "=" * 80)
    logger.success("EVALUATION COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
