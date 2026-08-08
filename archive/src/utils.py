"""
Utility functions for common tasks in Hydro-Graph ST-GNN.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pickle

import numpy as np
import pandas as pd
import networkx as nx
import torch
import matplotlib.pyplot as plt
from loguru import logger


def save_predictions_csv(
    predictions: np.ndarray,
    node_ids: np.ndarray,
    output_path: Path,
    additional_info: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    """
    Save predictions to CSV file.
    
    Parameters
    ----------
    predictions : np.ndarray
        Flood predictions
    node_ids : np.ndarray
        Node identifiers
    output_path : Path
        Output file path
    additional_info : Optional[Dict[str, np.ndarray]]
        Additional columns to include
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        'node_id': node_ids,
        'flood_probability': predictions,
    }
    
    if additional_info:
        data.update(additional_info)
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Saved predictions to {output_path}")


def load_graph_from_pickle(graph_path: Path) -> nx.DiGraph:
    """
    Load NetworkX graph from pickle file.
    
    Parameters
    ----------
    graph_path : Path
        Path to pickle file
        
    Returns
    -------
    nx.DiGraph
        Loaded graph
    """
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)
    
    logger.info(f"Loaded graph: {graph.number_of_nodes()} nodes")
    return graph


def plot_training_curves(
    history_path: Path,
    output_path: Path,
) -> None:
    """
    Plot training and validation curves.
    
    Parameters
    ----------
    history_path : Path
        Path to training history CSV
    output_path : Path
        Output path for plot
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load history
    df = pd.read_csv(history_path)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss curves
    axes[0, 0].plot(df['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(df['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # F1 Score
    axes[0, 1].plot(df['val_f1'], label='F1 Score', color='green', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_title('Validation F1 Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision and Recall
    axes[1, 0].plot(df['val_precision'], label='Precision', linewidth=2)
    axes[1, 0].plot(df['val_recall'], label='Recall', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Precision and Recall')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # ROC-AUC
    axes[1, 1].plot(df['val_roc_auc'], label='ROC-AUC', color='purple', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('ROC-AUC')
    axes[1, 1].set_title('Validation ROC-AUC')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved training curves to {output_path}")


def compute_graph_statistics(graph: nx.DiGraph) -> Dict[str, Any]:
    """
    Compute basic statistics of the graph.
    
    Parameters
    ----------
    graph : nx.DiGraph
        Input graph
        
    Returns
    -------
    Dict[str, Any]
        Dictionary of statistics
    """
    stats = {
        'num_nodes': graph.number_of_nodes(),
        'num_edges': graph.number_of_edges(),
        'avg_degree': sum(dict(graph.degree()).values()) / graph.number_of_nodes(),
        'is_connected': nx.is_strongly_connected(graph),
        'num_components': nx.number_weakly_connected_components(graph),
    }
    
    # Degree distribution
    degrees = [d for n, d in graph.degree()]
    stats['min_degree'] = min(degrees)
    stats['max_degree'] = max(degrees)
    stats['median_degree'] = np.median(degrees)
    
    logger.info("Graph statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    return stats


def normalize_features(
    features: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize features using z-score normalization.
    
    Parameters
    ----------
    features : np.ndarray
        Feature matrix (N, F)
    mean : Optional[np.ndarray]
        Pre-computed mean (for test set)
    std : Optional[np.ndarray]
        Pre-computed std (for test set)
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (normalized_features, mean, std)
    """
    if mean is None:
        mean = features.mean(axis=0)
    if std is None:
        std = features.std(axis=0)
        std[std == 0] = 1.0  # Avoid division by zero
    
    normalized = (features - mean) / std
    
    return normalized, mean, std


def check_cuda_availability() -> None:
    """Check CUDA availability and print GPU information."""
    if torch.cuda.is_available():
        logger.info("CUDA is available!")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    else:
        logger.warning("CUDA not available. Using CPU.")


def count_flood_nodes(
    predictions: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, int]:
    """
    Count nodes by flood risk category.
    
    Parameters
    ----------
    predictions : np.ndarray
        Flood predictions
    threshold : float
        Threshold for binary classification
        
    Returns
    -------
    Dict[str, int]
        Count of nodes by category
    """
    counts = {
        'total': len(predictions),
        'flooded': int((predictions >= threshold).sum()),
        'not_flooded': int((predictions < threshold).sum()),
        'high_risk': int((predictions >= 0.8).sum()),
        'medium_risk': int(((predictions >= 0.5) & (predictions < 0.8)).sum()),
        'low_risk': int(((predictions >= 0.3) & (predictions < 0.5)).sum()),
        'no_risk': int((predictions < 0.3).sum()),
    }
    
    logger.info("Flood node counts:")
    for key, value in counts.items():
        logger.info(f"  {key}: {value}")
    
    return counts


def create_submission_file(
    predictions: np.ndarray,
    node_ids: np.ndarray,
    output_path: Path,
) -> None:
    """
    Create submission file for competitions.
    
    Parameters
    ----------
    predictions : np.ndarray
        Flood predictions
    node_ids : np.ndarray
        Node identifiers
    output_path : Path
        Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame({
        'node_id': node_ids,
        'prediction': predictions,
    })
    
    df.to_csv(output_path, index=False)
    logger.info(f"Saved submission file to {output_path}")


if __name__ == '__main__':
    # Test utilities
    logger.info("Testing utilities...")
    check_cuda_availability()
