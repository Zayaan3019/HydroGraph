"""
Phase 1: Spatial Topology Extraction (Graph Construction)

This module extracts the physical "skeleton" of the city using OSMnx,
including street networks and drainage systems.
"""

from pathlib import Path
from typing import Optional, Dict, Tuple, List
import pickle

import osmnx as ox
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point
from loguru import logger
import pandas as pd


class GraphConstructor:
    """
    Constructs urban topology graph from OpenStreetMap data.
    
    Extracts street networks and drainage systems, converting them into
    a mathematical graph G=(V,E) suitable for ST-GNN processing.
    
    Attributes
    ----------
    bbox : Tuple[float, float, float, float]
        Bounding box (South, West, North, East)
    target_crs : str
        Target coordinate reference system (e.g., 'EPSG:32644')
    network_type : str
        OSMnx network type ('all', 'drive', 'walk', etc.)
    """
    
    def __init__(
        self,
        bbox: Tuple[float, float, float, float],
        target_crs: str = "EPSG:32644",
        network_type: str = "all",
        custom_filters: Optional[List[str]] = None,
    ):
        """
        Initialize the GraphConstructor.
        
        Parameters
        ----------
        bbox : Tuple[float, float, float, float]
            Bounding box as (South, West, North, East)
        target_crs : str
            Target CRS for projection (default: EPSG:32644 for Chennai)
        network_type : str
            OSMnx network type (default: 'all')
        custom_filters : Optional[List[str]]
            Custom OSM filters for additional features (e.g., drainage)
        """
        self.bbox = bbox
        self.target_crs = target_crs
        self.network_type = network_type
        self.custom_filters = custom_filters or []
        
        self.graph: Optional[nx.DiGraph] = None
        self.node_gdf: Optional[gpd.GeoDataFrame] = None
        self.edge_gdf: Optional[gpd.GeoDataFrame] = None
        
        logger.info(f"GraphConstructor initialized for bbox: {bbox}")
        logger.info(f"Target CRS: {target_crs}")
    
    def download_street_network(self) -> nx.MultiDiGraph:
        """
        Download street network from OpenStreetMap using OSMnx.
        
        Returns
        -------
        nx.MultiDiGraph
            Raw street network from OSMnx
        """
        logger.info("Downloading street network from OpenStreetMap...")
        
        # Unpack bbox: OSMnx expects (North, South, East, West)
        south, west, north, east = self.bbox
        
        try:
            # Download street network
            G = ox.graph_from_bbox(
                north=north,
                south=south,
                east=east,
                west=west,
                network_type=self.network_type,
                simplify=True,
                retain_all=False,
            )
            
            logger.success(
                f"Downloaded street network: {G.number_of_nodes()} nodes, "
                f"{G.number_of_edges()} edges"
            )
            return G
            
        except Exception as e:
            logger.error(f"Failed to download street network: {e}")
            raise
    
    def download_drainage_network(self) -> Optional[nx.MultiDiGraph]:
        """
        Download drainage infrastructure (drains, canals) from OpenStreetMap.
        
        Returns
        -------
        Optional[nx.MultiDiGraph]
            Drainage network graph, or None if not available
        """
        logger.info("Downloading drainage network from OpenStreetMap...")
        
        south, west, north, east = self.bbox
        
        try:
            # Custom filter for waterways (drains and canals)
            custom_filter = '["waterway"~"drain|canal"]'
            
            G_drainage = ox.graph_from_bbox(
                north=north,
                south=south,
                east=east,
                west=west,
                custom_filter=custom_filter,
                simplify=True,
                retain_all=False,
            )
            
            logger.success(
                f"Downloaded drainage network: {G_drainage.number_of_nodes()} nodes, "
                f"{G_drainage.number_of_edges()} edges"
            )
            return G_drainage
            
        except Exception as e:
            logger.warning(f"Could not download drainage network: {e}")
            return None
    
    def merge_networks(
        self,
        street_graph: nx.MultiDiGraph,
        drainage_graph: Optional[nx.MultiDiGraph] = None,
    ) -> nx.MultiDiGraph:
        """
        Merge street and drainage networks into a single graph.
        
        Parameters
        ----------
        street_graph : nx.MultiDiGraph
            Street network graph
        drainage_graph : Optional[nx.MultiDiGraph]
            Drainage network graph
            
        Returns
        -------
        nx.MultiDiGraph
            Merged network graph
        """
        if drainage_graph is None:
            logger.info("No drainage network to merge. Using street network only.")
            return street_graph
        
        logger.info("Merging street and drainage networks...")
        
        # Compose the graphs (union of nodes and edges)
        merged = nx.compose(street_graph, drainage_graph)
        
        logger.success(
            f"Merged network: {merged.number_of_nodes()} nodes, "
            f"{merged.number_of_edges()} edges"
        )
        
        return merged
    
    def convert_to_digraph(self, multigraph: nx.MultiDiGraph) -> nx.DiGraph:
        """
        Convert MultiDiGraph to DiGraph, merging parallel edges.
        
        For parallel edges, keeps the shortest one based on length.
        
        Parameters
        ----------
        multigraph : nx.MultiDiGraph
            Input multi-directed graph
            
        Returns
        -------
        nx.DiGraph
            Simplified directed graph
        """
        logger.info("Converting MultiDiGraph to DiGraph...")
        
        G = nx.DiGraph()
        
        # Add nodes with attributes
        for node, data in multigraph.nodes(data=True):
            G.add_node(node, **data)
        
        # Add edges, keeping shortest for parallel edges
        for u, v, key, data in multigraph.edges(keys=True, data=True):
            if G.has_edge(u, v):
                # Keep shorter edge
                if data.get('length', float('inf')) < G[u][v].get('length', float('inf')):
                    G[u][v].update(data)
            else:
                G.add_edge(u, v, **data)
        
        logger.success(
            f"Converted to DiGraph: {G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges"
        )
        
        return G
    
    def project_graph(self, graph: nx.DiGraph) -> nx.DiGraph:
        """
        Project graph to target CRS.
        
        Parameters
        ----------
        graph : nx.DiGraph
            Input graph in WGS84
            
        Returns
        -------
        nx.DiGraph
            Projected graph
        """
        logger.info(f"Projecting graph to {self.target_crs}...")
        
        # Project using OSMnx utility
        G_proj = ox.project_graph(graph, to_crs=self.target_crs)
        
        logger.success("Graph projected successfully")
        return G_proj
    
    def extract_node_geodataframe(self, graph: nx.DiGraph) -> gpd.GeoDataFrame:
        """
        Extract node coordinates as a GeoDataFrame for spatial joining.
        
        Parameters
        ----------
        graph : nx.DiGraph
            Input graph
            
        Returns
        -------
        gpd.GeoDataFrame
            Node coordinates with geometry
        """
        logger.info("Extracting node coordinates as GeoDataFrame...")
        
        # Extract node data
        nodes_data = []
        for node, data in graph.nodes(data=True):
            node_dict = {
                'osmid': node,
                'x': data.get('x'),
                'y': data.get('y'),
                'lon': data.get('lon'),
                'lat': data.get('lat'),
            }
            # Add any other attributes
            for key, value in data.items():
                if key not in ['x', 'y', 'lon', 'lat']:
                    node_dict[key] = value
            
            nodes_data.append(node_dict)
        
        # Create DataFrame
        df = pd.DataFrame(nodes_data)
        
        # Create Point geometries
        geometry = [Point(x, y) for x, y in zip(df['x'], df['y'])]
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=self.target_crs)
        
        logger.success(f"Created GeoDataFrame with {len(gdf)} nodes")
        
        return gdf
    
    def extract_edge_geodataframe(self, graph: nx.DiGraph) -> gpd.GeoDataFrame:
        """
        Extract edges as a GeoDataFrame.
        
        Parameters
        ----------
        graph : nx.DiGraph
            Input graph
            
        Returns
        -------
        gpd.GeoDataFrame
            Edge geometries with attributes
        """
        logger.info("Extracting edges as GeoDataFrame...")
        
        # Convert to GeoDataFrames using OSMnx
        _, edges = ox.graph_to_gdfs(graph)
        
        logger.success(f"Created edge GeoDataFrame with {len(edges)} edges")
        
        return edges
    
    def build_graph(self) -> nx.DiGraph:
        """
        Main method to build the complete urban topology graph.
        
        This method orchestrates the entire graph construction pipeline:
        1. Download street network
        2. Download drainage network
        3. Merge networks
        4. Convert to DiGraph
        5. Project to target CRS
        6. Extract node and edge GeoDataFrames
        
        Returns
        -------
        nx.DiGraph
            Complete urban topology graph
        """
        logger.info("=" * 80)
        logger.info("PHASE 1: SPATIAL TOPOLOGY EXTRACTION")
        logger.info("=" * 80)
        
        # Step 1: Download street network
        street_graph = self.download_street_network()
        
        # Step 2: Download drainage network
        drainage_graph = self.download_drainage_network()
        
        # Step 3: Merge networks
        merged_graph = self.merge_networks(street_graph, drainage_graph)
        
        # Step 4: Convert to DiGraph
        digraph = self.convert_to_digraph(merged_graph)
        
        # Step 5: Project to target CRS
        projected_graph = self.project_graph(digraph)
        
        # Step 6: Extract GeoDataFrames
        self.node_gdf = self.extract_node_geodataframe(projected_graph)
        self.edge_gdf = self.extract_edge_geodataframe(projected_graph)
        
        # Store graph
        self.graph = projected_graph
        
        logger.success("=" * 80)
        logger.success("PHASE 1 COMPLETE: Graph construction successful")
        logger.success(f"Final graph: {projected_graph.number_of_nodes()} nodes, "
                      f"{projected_graph.number_of_edges()} edges")
        logger.success("=" * 80)
        
        return projected_graph
    
    def save_graph(self, output_dir: Path) -> None:
        """
        Save graph and GeoDataFrames to disk.
        
        Parameters
        ----------
        output_dir : Path
            Output directory for saving files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.graph is None:
            raise ValueError("Graph not built yet. Call build_graph() first.")
        
        logger.info(f"Saving graph to {output_dir}...")
        
        # Save NetworkX graph
        graph_path = output_dir / "urban_graph.gpickle"
        with open(graph_path, 'wb') as f:
            pickle.dump(self.graph, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved graph to {graph_path}")
        
        # Save node GeoDataFrame
        if self.node_gdf is not None:
            node_path = output_dir / "nodes.geojson"
            self.node_gdf.to_file(node_path, driver="GeoJSON")
            logger.info(f"Saved nodes to {node_path}")
        
        # Save edge GeoDataFrame
        if self.edge_gdf is not None:
            edge_path = output_dir / "edges.geojson"
            self.edge_gdf.to_file(edge_path, driver="GeoJSON")
            logger.info(f"Saved edges to {edge_path}")
        
        logger.success("Graph saved successfully")
    
    def load_graph(self, input_dir: Path) -> nx.DiGraph:
        """
        Load previously saved graph from disk.
        
        Parameters
        ----------
        input_dir : Path
            Input directory containing saved files
            
        Returns
        -------
        nx.DiGraph
            Loaded graph
        """
        input_dir = Path(input_dir)
        
        logger.info(f"Loading graph from {input_dir}...")
        
        # Load NetworkX graph
        graph_path = input_dir / "urban_graph.gpickle"
        with open(graph_path, 'rb') as f:
            self.graph = pickle.load(f)
        logger.info(f"Loaded graph: {self.graph.number_of_nodes()} nodes")
        
        # Load node GeoDataFrame
        node_path = input_dir / "nodes.geojson"
        if node_path.exists():
            self.node_gdf = gpd.read_file(node_path)
            logger.info(f"Loaded nodes: {len(self.node_gdf)} nodes")
        
        # Load edge GeoDataFrame
        edge_path = input_dir / "edges.geojson"
        if edge_path.exists():
            self.edge_gdf = gpd.read_file(edge_path)
            logger.info(f"Loaded edges: {len(self.edge_gdf)} edges")
        
        logger.success("Graph loaded successfully")
        return self.graph
