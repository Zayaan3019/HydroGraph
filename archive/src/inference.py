"""
Phase 6: Inference and Geospatial Mapping

This module handles model inference and translates tensor predictions
back to geospatial visualizations.
"""

from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import torch
import torch.nn as nn
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
import folium
from folium import plugins
from loguru import logger


class FloodPredictor:
    """
    Inference engine for flood prediction.
    
    Loads trained model and generates predictions for new data.
    
    Parameters
    ----------
    model : nn.Module
        Trained HydroGraph ST-GNN model
    checkpoint_path : Path
        Path to model checkpoint
    device : str
        Device for inference ('cpu' or 'cuda')
    """
    
    def __init__(
        self,
        model: nn.Module,
        checkpoint_path: Optional[Path] = None,
        device: str = 'cpu',
    ):
        self.model = model.to(device)
        self.device = device
        
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)
        
        self.model.eval()
        
        logger.info("=" * 80)
        logger.info("PHASE 6: INFERENCE & GEOSPATIAL MAPPING")
        logger.info("=" * 80)
        logger.info(f"FloodPredictor initialized on {device}")
    
    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """
        Load model weights from checkpoint.
        
        Parameters
        ----------
        checkpoint_path : Path
            Path to checkpoint file
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Loaded checkpoint: {checkpoint_path}")
        logger.info(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
        logger.info(f"Best val loss: {checkpoint.get('best_val_loss', 'unknown')}")
    
    @torch.no_grad()
    def predict(self, data: Data) -> np.ndarray:
        """
        Generate flood predictions for a single graph snapshot.
        
        Parameters
        ----------
        data : Data
            PyTorch Geometric Data object
            
        Returns
        -------
        np.ndarray
            Flood probability predictions (num_nodes,)
        """
        data = data.to(self.device)
        
        # Forward pass
        predictions = self.model(data.x, data.edge_index)
        
        # Convert to numpy
        predictions = predictions.cpu().numpy().flatten()
        
        return predictions
    
    @torch.no_grad()
    def predict_batch(self, data_list: List[Data]) -> np.ndarray:
        """
        Generate predictions for multiple graph snapshots.
        
        Parameters
        ----------
        data_list : List[Data]
            List of PyTorch Geometric Data objects
            
        Returns
        -------
        np.ndarray
            Predictions for all snapshots (num_snapshots, num_nodes)
        """
        predictions_list = []
        
        logger.info(f"Generating predictions for {len(data_list)} snapshots...")
        
        for data in data_list:
            preds = self.predict(data)
            predictions_list.append(preds)
        
        predictions = np.stack(predictions_list, axis=0)
        
        logger.success(f"Predictions shape: {predictions.shape}")
        
        return predictions
    
    def get_flood_risk_categories(
        self,
        predictions: np.ndarray,
        thresholds: Dict[str, float] = None,
    ) -> np.ndarray:
        """
        Categorize flood predictions into risk levels.
        
        Parameters
        ----------
        predictions : np.ndarray
            Flood probabilities
        thresholds : Dict[str, float]
            Thresholds for categorization
            Default: {'low': 0.3, 'medium': 0.6, 'high': 0.8}
            
        Returns
        -------
        np.ndarray
            Risk categories (0=No Risk, 1=Low, 2=Medium, 3=High, 4=Very High)
        """
        if thresholds is None:
            thresholds = {'low': 0.3, 'medium': 0.6, 'high': 0.8}
        
        categories = np.zeros_like(predictions, dtype=int)
        
        categories[predictions >= thresholds['low']] = 1  # Low risk
        categories[predictions >= thresholds['medium']] = 2  # Medium risk
        categories[predictions >= thresholds['high']] = 3  # High risk
        categories[predictions >= 0.9] = 4  # Very high risk
        
        return categories


class GeospatialVisualizer:
    """
    Geospatial visualization tools for flood predictions.
    
    Creates static maps and interactive visualizations.
    
    Parameters
    ----------
    graph : nx.DiGraph
        Urban topology graph
    node_gdf : gpd.GeoDataFrame
        Node geometries
    edge_gdf : Optional[gpd.GeoDataFrame]
        Edge geometries
    """
    
    def __init__(
        self,
        graph: nx.DiGraph,
        node_gdf: gpd.GeoDataFrame,
        edge_gdf: Optional[gpd.GeoDataFrame] = None,
    ):
        self.graph = graph
        self.node_gdf = node_gdf.copy()
        self.edge_gdf = edge_gdf
        
        # Ensure WGS84 for web mapping
        if self.node_gdf.crs != 'EPSG:4326':
            self.node_gdf = self.node_gdf.to_crs('EPSG:4326')
        
        if self.edge_gdf is not None and self.edge_gdf.crs != 'EPSG:4326':
            self.edge_gdf = self.edge_gdf.to_crs('EPSG:4326')
        
        logger.info("GeospatialVisualizer initialized")
    
    def bind_predictions_to_nodes(
        self,
        predictions: np.ndarray,
        column_name: str = 'flood_prob',
    ) -> gpd.GeoDataFrame:
        """
        Bind predictions to node GeoDataFrame.
        
        Parameters
        ----------
        predictions : np.ndarray
            Flood predictions (num_nodes,)
        column_name : str
            Column name for predictions
            
        Returns
        -------
        gpd.GeoDataFrame
            Node GeoDataFrame with predictions
        """
        gdf = self.node_gdf.copy()
        gdf[column_name] = predictions
        
        # Add risk category
        predictor = FloodPredictor(None)  # Dummy for categorization
        gdf['risk_category'] = predictor.get_flood_risk_categories(predictions)
        
        logger.info(f"Bound predictions to {len(gdf)} nodes")
        
        return gdf
    
    def plot_static_map(
        self,
        predictions: np.ndarray,
        output_path: Path,
        title: str = "Flood Risk Map",
        cmap: str = 'RdYlBu_r',
        figsize: Tuple[int, int] = (15, 10),
        show_edges: bool = True,
    ) -> None:
        """
        Create static matplotlib map of flood predictions.
        
        Parameters
        ----------
        predictions : np.ndarray
            Flood predictions
        output_path : Path
            Path to save figure
        title : str
            Map title
        cmap : str
            Colormap name
        figsize : Tuple[int, int]
            Figure size
        show_edges : bool
            Whether to show street network edges
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creating static map: {title}")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot edges (street network)
        if show_edges and self.edge_gdf is not None:
            self.edge_gdf.plot(
                ax=ax,
                color='gray',
                alpha=0.3,
                linewidth=0.5,
                zorder=1,
            )
        
        # Bind predictions to nodes
        node_gdf = self.bind_predictions_to_nodes(predictions)
        
        # Plot nodes with predictions
        node_gdf.plot(
            column='flood_prob',
            ax=ax,
            cmap=cmap,
            legend=True,
            markersize=20,
            alpha=0.7,
            vmin=0,
            vmax=1,
            zorder=2,
            legend_kwds={
                'label': 'Flood Probability',
                'orientation': 'horizontal',
                'shrink': 0.8,
                'pad': 0.05,
            },
        )
        
        # Add title and labels
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        
        # Remove axes for cleaner look
        ax.set_axis_off()
        
        # Tight layout
        plt.tight_layout()
        
        # Save
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.success(f"Saved static map to {output_path}")
    
    def create_interactive_map(
        self,
        predictions: np.ndarray,
        output_path: Path,
        tile_provider: str = 'OpenStreetMap',
        colormap: str = 'RdYlBu_r',
    ) -> None:
        """
        Create interactive Folium map of flood predictions.
        
        Parameters
        ----------
        predictions : np.ndarray
            Flood predictions
        output_path : Path
            Path to save HTML file
        tile_provider : str
            Tile provider for base map
        colormap : str
            Colormap name
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info("Creating interactive Folium map...")
        
        # Bind predictions to nodes
        node_gdf = self.bind_predictions_to_nodes(predictions)
        
        # Calculate map center
        center_lat = node_gdf.geometry.y.mean()
        center_lon = node_gdf.geometry.x.mean()
        
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=12,
            tiles=tile_provider,
        )
        
        # Define color mapping
        colormap_obj = plt.cm.get_cmap(colormap)
        
        # Add nodes as circle markers
        for idx, row in node_gdf.iterrows():
            # Get color based on flood probability
            prob = row['flood_prob']
            rgba = colormap_obj(prob)
            color = mcolors.rgb2hex(rgba[:3])
            
            # Determine risk level text
            risk_levels = ['No Risk', 'Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk']
            risk_text = risk_levels[int(row['risk_category'])]
            
            # Create popup
            popup_html = f"""
            <div style="font-family: Arial; font-size: 12px;">
                <b>Node ID:</b> {row.get('osmid', idx)}<br>
                <b>Flood Probability:</b> {prob:.2%}<br>
                <b>Risk Level:</b> {risk_text}<br>
                <b>Elevation:</b> {row.get('elevation', 'N/A'):.1f} m<br>
                <b>NDVI:</b> {row.get('ndvi', 'N/A'):.2f}<br>
            </div>
            """
            
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=5,
                popup=folium.Popup(popup_html, max_width=300),
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
                weight=1,
            ).add_to(m)
        
        # Add legend
        legend_html = f"""
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; width: 200px; height: 180px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p style="margin-bottom: 5px;"><b>Flood Risk Level</b></p>
        <p style="margin: 3px;"><span style="background-color: #313695; width: 20px; height: 10px; display: inline-block;"></span> No Risk (0-30%)</p>
        <p style="margin: 3px;"><span style="background-color: #74add1; width: 20px; height: 10px; display: inline-block;"></span> Low (30-60%)</p>
        <p style="margin: 3px;"><span style="background-color: #fee090; width: 20px; height: 10px; display: inline-block;"></span> Medium (60-80%)</p>
        <p style="margin: 3px;"><span style="background-color: #f46d43; width: 20px; height: 10px; display: inline-block;"></span> High (80-90%)</p>
        <p style="margin: 3px;"><span style="background-color: #a50026; width: 20px; height: 10px; display: inline-block;"></span> Very High (>90%)</p>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add fullscreen button
        plugins.Fullscreen().add_to(m)
        
        # Save map
        m.save(str(output_path))
        
        logger.success(f"Saved interactive map to {output_path}")
    
    def plot_risk_distribution(
        self,
        predictions: np.ndarray,
        output_path: Path,
    ) -> None:
        """
        Plot histogram of flood risk distribution.
        
        Parameters
        ----------
        predictions : np.ndarray
            Flood predictions
        output_path : Path
            Path to save figure
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(predictions, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Flood Probability', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Distribution of Flood Probabilities', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Risk categories pie chart
        predictor = FloodPredictor(None)
        categories = predictor.get_flood_risk_categories(predictions)
        category_counts = np.bincount(categories, minlength=5)
        
        labels = ['No Risk', 'Low', 'Medium', 'High', 'Very High']
        colors = ['#313695', '#74add1', '#fee090', '#f46d43', '#a50026']
        
        axes[1].pie(
            category_counts,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
        )
        axes[1].set_title('Risk Category Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.success(f"Saved risk distribution plot to {output_path}")


def generate_flood_report(
    predictions: np.ndarray,
    node_gdf: gpd.GeoDataFrame,
    output_dir: Path,
    event_name: str = "Storm Event",
) -> None:
    """
    Generate comprehensive flood risk report.
    
    Parameters
    ----------
    predictions : np.ndarray
        Flood predictions
    node_gdf : gpd.GeoDataFrame
        Node geometries
    output_dir : Path
        Output directory for report
    event_name : str
        Name of the event
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating flood report for: {event_name}")
    
    # Statistics
    predictor = FloodPredictor(None)
    categories = predictor.get_flood_risk_categories(predictions)
    
    stats = {
        'Event': event_name,
        'Total Nodes': len(predictions),
        'Mean Flood Probability': f"{predictions.mean():.2%}",
        'Max Flood Probability': f"{predictions.max():.2%}",
        'Nodes at High Risk (>80%)': int((predictions > 0.8).sum()),
        'Nodes at Medium Risk (60-80%)': int(((predictions >= 0.6) & (predictions <= 0.8)).sum()),
        'No Risk Nodes': len(predictions[categories == 0]),
        'Low Risk Nodes': len(predictions[categories == 1]),
        'Medium Risk Nodes': len(predictions[categories == 2]),
        'High Risk Nodes': len(predictions[categories == 3]),
        'Very High Risk Nodes': len(predictions[categories == 4]),
    }
    
    # Save statistics
    stats_df = pd.DataFrame([stats])
    stats_path = output_dir / 'flood_statistics.csv'
    stats_df.to_csv(stats_path, index=False)
    
    # Save node-level predictions
    pred_gdf = node_gdf.copy()
    pred_gdf['flood_probability'] = predictions
    pred_gdf['risk_category'] = categories
    
    pred_path = output_dir / 'node_predictions.geojson'
    pred_gdf.to_file(pred_path, driver='GeoJSON')
    
    logger.success("=" * 80)
    logger.success("PHASE 6 COMPLETE: Inference and mapping successful")
    logger.success(f"Report saved to: {output_dir}")
    logger.success("=" * 80)
