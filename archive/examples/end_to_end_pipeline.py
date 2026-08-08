"""
End-to-End Pipeline Example for Hydro-Graph ST-GNN

This script demonstrates the complete workflow from graph construction
through inference and visualization.
"""

from pathlib import Path
from datetime import datetime
import torch
from loguru import logger

# Import all modules
from config import load_config
from src import (
    GraphConstructor,
    FeatureEngineer,
    RainfallDataLoader,
    TemporalDatasetCreator,
    HydroGraphDataset,
    HydroGraphSTGNN,
    Trainer,
    FloodPredictor,
    GeospatialVisualizer,
    generate_flood_report,
)
from torch_geometric.loader import NeighborLoader


def setup_directories(base_dir: Path) -> None:
    """Create necessary directories."""
    directories = [
        base_dir / 'data/graphs',
        base_dir / 'data/processed',
        base_dir / 'checkpoints',
        base_dir / 'outputs/predictions',
        base_dir / 'outputs/visualizations',
        base_dir / 'logs',
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    logger.info("Directories created")


def main():
    """Main pipeline execution."""
    
    # Configure logging
    logger.add("logs/pipeline_{time}.log", rotation="500 MB")
    
    logger.info("=" * 80)
    logger.info("HYDRO-GRAPH ST-GNN: END-TO-END PIPELINE")
    logger.info("=" * 80)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Load configuration
    config = load_config()
    logger.info("Configuration loaded")
    
    # Setup directories
    base_dir = Path.cwd()
    setup_directories(base_dir)
    
    # =========================================================================
    # PHASE 1: GRAPH CONSTRUCTION
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("Starting Phase 1: Graph Construction")
    logger.info("=" * 80)
    
    graph_path = base_dir / 'data/graphs/urban_graph.gpickle'
    
    if graph_path.exists():
        logger.info("Loading existing graph...")
        constructor = GraphConstructor(
            bbox=config.location.bbox,
            target_crs=config.location.target_crs,
        )
        graph = constructor.load_graph(base_dir / 'data/graphs')
    else:
        logger.info("Building new graph...")
        constructor = GraphConstructor(
            bbox=config.location.bbox,
            target_crs=config.location.target_crs,
            network_type=config.graph.network_type,
            custom_filters=config.graph.custom_filters,
        )
        graph = constructor.build_graph()
        constructor.save_graph(base_dir / 'data/graphs')
    
    logger.success(f"Graph ready: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # =========================================================================
    # PHASE 2: FEATURE ENGINEERING
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("Starting Phase 2: Feature Engineering")
    logger.info("=" * 80)
    
    # Check if features already exist
    sample_node = list(graph.nodes(data=True))[0]
    if 'elevation' not in sample_node[1]:
        logger.info("Extracting features...")
        engineer = FeatureEngineer(
            graph=graph,
            node_gdf=constructor.node_gdf,
            raster_paths=config.data.rasters,
            target_crs=config.location.target_crs,
        )
        graph = engineer.engineer_all_features()
        
        # Save graph with features
        constructor.graph = graph
        constructor.save_graph(base_dir / 'data/graphs')
    else:
        logger.info("Features already present")
    
    logger.success("Feature engineering complete")
    
    # =========================================================================
    # PHASE 3: TEMPORAL DATASET CREATION
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("Starting Phase 3: Temporal Dataset Creation")
    logger.info("=" * 80)
    
    # Initialize rainfall loader
    rainfall_loader = RainfallDataLoader(
        gpm_dir=config.data.precipitation.get('gpm_imerg_dir'),
        imd_file=config.data.precipitation.get('imd_gauge_file'),
        bbox=config.location.bbox,
    )
    
    # Create dataset creator
    dataset_creator = TemporalDatasetCreator(
        graph=graph,
        rainfall_loader=rainfall_loader,
        lag_window=config.features.temporal.lag_window,
    )
    
    # Generate temporal sequence for Chennai 2015 floods
    event_config = config.events['chennai_2015']
    start_date = datetime.strptime(event_config.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(event_config.end_date, '%Y-%m-%d')
    
    data_list = dataset_creator.create_temporal_sequence(start_date, end_date)
    
    # Create dataset and split
    dataset = HydroGraphDataset(data_list)
    train_data, val_data, test_data = dataset.train_val_test_split(
        train_ratio=config.training.split.train,
        val_ratio=config.training.split.val,
        test_ratio=config.training.split.test,
    )
    
    logger.success(f"Dataset created: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    
    # =========================================================================
    # PHASE 4: MODEL INITIALIZATION
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("Starting Phase 4: Model Initialization")
    logger.info("=" * 80)
    
    # Determine number of static features
    num_static_features = 8  # elevation, slope, twi, ndvi, ndwi, ndbi, imperviousness, sar_vv
    
    # Create model
    model = HydroGraphSTGNN(
        num_static_features=num_static_features,
        lag_window=config.features.temporal.lag_window,
        spatial_config=config.model.architecture.spatial,
        temporal_config=config.model.architecture.temporal,
        fusion_config=config.model.architecture.fusion,
    )
    
    logger.success(f"Model created with {model.count_parameters():,} parameters")
    
    # =========================================================================
    # PHASE 5: TRAINING
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("Starting Phase 5: Training")
    logger.info("=" * 80)
    
    # Create data loaders
    train_loader = NeighborLoader(
        train_data.data_list,
        num_neighbors=config.model.neighbor_sampling.num_neighbors,
        batch_size=config.model.neighbor_sampling.batch_size,
        num_workers=0,  # Set to 0 for Windows compatibility
        shuffle=True,
    )
    
    val_loader = NeighborLoader(
        val_data.data_list,
        num_neighbors=config.model.neighbor_sampling.num_neighbors,
        batch_size=config.model.neighbor_sampling.batch_size,
        num_workers=0,
        shuffle=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        checkpoint_dir=base_dir / 'checkpoints',
    )
    
    # Train model
    trainer.train(num_epochs=config.training.epochs)
    
    # Save training history
    trainer.save_history(base_dir / 'outputs/training_history.csv')
    
    logger.success("Training complete")
    
    # =========================================================================
    # PHASE 6: INFERENCE AND VISUALIZATION
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("Starting Phase 6: Inference and Visualization")
    logger.info("=" * 80)
    
    # Initialize predictor
    predictor = FloodPredictor(
        model=model,
        checkpoint_path=base_dir / 'checkpoints/best_model.pth',
        device=device,
    )
    
    # Generate predictions for test set
    test_loader = NeighborLoader(
        test_data.data_list,
        num_neighbors=config.model.neighbor_sampling.num_neighbors,
        batch_size=config.model.neighbor_sampling.batch_size,
        num_workers=0,
        shuffle=False,
    )
    
    # Get predictions for first test snapshot (peak flood event)
    test_snapshot = test_data.get(len(test_data) // 2)  # Middle of test period
    predictions = predictor.predict(test_snapshot)
    
    logger.info(f"Generated predictions for {len(predictions)} nodes")
    logger.info(f"Mean flood probability: {predictions.mean():.2%}")
    logger.info(f"Max flood probability: {predictions.max():.2%}")
    
    # Create visualizer
    visualizer = GeospatialVisualizer(
        graph=graph,
        node_gdf=constructor.node_gdf,
        edge_gdf=constructor.edge_gdf,
    )
    
    # Generate static map
    visualizer.plot_static_map(
        predictions=predictions,
        output_path=base_dir / 'outputs/visualizations/flood_risk_map.png',
        title=f"Chennai Flood Risk - {event_config.name}",
    )
    
    # Generate interactive map
    visualizer.create_interactive_map(
        predictions=predictions,
        output_path=base_dir / 'outputs/visualizations/flood_risk_map.html',
    )
    
    # Generate risk distribution plot
    visualizer.plot_risk_distribution(
        predictions=predictions,
        output_path=base_dir / 'outputs/visualizations/risk_distribution.png',
    )
    
    # Generate comprehensive report
    generate_flood_report(
        predictions=predictions,
        node_gdf=constructor.node_gdf,
        output_dir=base_dir / 'outputs/predictions',
        event_name=event_config.name,
    )
    
    logger.success("=" * 80)
    logger.success("PIPELINE COMPLETE!")
    logger.success("=" * 80)
    logger.success(f"Outputs saved to: {base_dir / 'outputs'}")
    logger.success(f"Checkpoints saved to: {base_dir / 'checkpoints'}")
    logger.success(f"Logs saved to: {base_dir / 'logs'}")


if __name__ == '__main__':
    main()
