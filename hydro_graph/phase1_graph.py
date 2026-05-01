"""
Phase 1 - Spatial Topology Extraction (Graph Construction)
===========================================================
Downloads the Chennai street + drainage network from OpenStreetMap via OSMnx,
converts it to a networkx DiGraph with physical distance weights, and exports
node coordinates as a GeoDataFrame for downstream spatial joins.

Key additions in v2:
  - Edge feature extraction: [elev_diff_norm, length_norm, edge_type, flow_weight]
  - Waterway type attributes stored on edges for drainage capacity estimation
  - Node elevation proxy (from OSMnx tags, refined by Phase 2 SRTM)
  - Flow-direction edges weighted by terrain gradient
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import Point

logger = logging.getLogger(__name__)

# Edge type encoding
EDGE_TYPE_ROAD = 0.0
EDGE_TYPE_WATERWAY = 1.0

# Waterway drainage capacity proxies (relative, 0-1 scale)
_WATERWAY_CAPACITY = {
    "river":   1.00,
    "canal":   0.80,
    "stream":  0.50,
    "drain":   0.40,
    "ditch":   0.25,
    "default": 0.10,
}


class GraphConstructor:
    """Build and persist the urban hydrological graph G = (V, E) with edge features."""

    def __init__(
        self,
        bbox: Tuple[float, float, float, float],
        crs_projected: str = "EPSG:32644",
        crs_geographic: str = "EPSG:4326",
        simplify: bool = True,
        retain_all: bool = False,
        use_synthetic_fallback: bool = True,
        edge_feature_dim: int = 4,
    ) -> None:
        self.bbox = bbox
        self.crs_proj = crs_projected
        self.crs_geo = crs_geographic
        self.simplify = simplify
        self.retain_all = retain_all
        self.use_synthetic_fallback = use_synthetic_fallback
        self.edge_feature_dim = edge_feature_dim

        self.G: Optional[nx.DiGraph] = None
        self.gdf_nodes: Optional[gpd.GeoDataFrame] = None
        self.edge_features: Optional[np.ndarray] = None  # [E, 4]

    # ── Public API ────────────────────────────────────────────────────────────

    def build(self) -> Tuple[nx.DiGraph, gpd.GeoDataFrame]:
        try:
            import osmnx as ox
            logger.info("OSMnx download for bbox %s …", self.bbox)
            G_multi = self._download_osmnx(ox)
        except Exception as exc:
            if self.use_synthetic_fallback:
                logger.warning("OSMnx failed (%s). Using synthetic graph.", exc)
                G_multi = self._build_synthetic_multigraph()
            else:
                raise

        self.G = self._convert_to_digraph(G_multi)
        self.gdf_nodes = self._build_node_geodataframe(self.G)
        self.edge_features = self._compute_edge_features(self.G, self.gdf_nodes)

        logger.info(
            "Graph: %d nodes, %d edges | edge_features: %s",
            self.G.number_of_nodes(),
            self.G.number_of_edges(),
            self.edge_features.shape,
        )
        return self.G, self.gdf_nodes

    def save(self, graph_path: str, base_dir: Optional[Path] = None) -> None:
        assert self.G is not None, "Call build() first"
        base = base_dir or Path.cwd()
        path = Path(graph_path) if Path(graph_path).is_absolute() else base / graph_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({
                "graph": self.G,
                "gdf_nodes": self.gdf_nodes,
                "edge_features": self.edge_features,
            }, fh)
        logger.info("Graph saved -> %s", path)

    @staticmethod
    def load(graph_path: str, base_dir: Optional[Path] = None) -> Tuple[nx.DiGraph, gpd.GeoDataFrame, np.ndarray]:
        base = base_dir or Path.cwd()
        path = Path(graph_path) if Path(graph_path).is_absolute() else base / graph_path
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        G = data["graph"]
        gdf = data["gdf_nodes"]
        edge_feat = data.get("edge_features", None)

        # Backward compat: recompute edge features if missing
        if edge_feat is None:
            logger.warning("Cached graph missing edge_features; recomputing.")
            gc = GraphConstructor.__new__(GraphConstructor)
            gc.crs_proj = "EPSG:32644"
            gc.crs_geo = "EPSG:4326"
            gc.edge_feature_dim = 4
            edge_feat = gc._compute_edge_features(G, gdf)
            try:
                with open(path, "wb") as fh:
                    pickle.dump({
                        "graph": G,
                        "gdf_nodes": gdf,
                        "edge_features": edge_feat,
                    }, fh)
                logger.info("Repaired cached graph edge_features -> %s", path)
            except Exception as exc:
                logger.warning("Could not persist repaired edge_features: %s", exc)

        logger.info("Graph loaded <- %s  [%d nodes, %d edges]", path, G.number_of_nodes(), G.number_of_edges())
        return G, gdf, edge_feat

    # ── OSMnx Download ────────────────────────────────────────────────────────

    def _download_osmnx(self, ox) -> nx.MultiDiGraph:
        left, bottom, right, top = self.bbox

        logger.info("  Downloading road network …")
        G_road = ox.graph_from_bbox(
            bbox=(left, bottom, right, top),
            network_type="all",
            simplify=self.simplify,
            retain_all=self.retain_all,
        )

        # Tag road edges
        for _, _, data in G_road.edges(data=True):
            data["_edge_type"] = EDGE_TYPE_ROAD
            data["_drain_capacity"] = 0.0

        logger.info("  Downloading waterway network …")
        try:
            G_water = ox.graph_from_bbox(
                bbox=(left, bottom, right, top),
                custom_filter='["waterway"~"drain|canal|river|stream|ditch"]',
                retain_all=True,
            )
            # Tag waterway edges with capacity
            for _, _, data in G_water.edges(data=True):
                wtype = str(data.get("waterway", "default")).lower()
                data["_edge_type"] = EDGE_TYPE_WATERWAY
                data["_drain_capacity"] = _WATERWAY_CAPACITY.get(
                    wtype, _WATERWAY_CAPACITY["default"]
                )

            G_combined: nx.MultiDiGraph = nx.compose(G_road, G_water)
            logger.info("  Merged %d waterway edges", G_water.number_of_edges())
        except Exception as exc:
            logger.warning("  Waterway download failed (%s). Road-only graph.", exc)
            G_combined = G_road

        return G_combined

    # ── Synthetic Fallback ────────────────────────────────────────────────────

    def _build_synthetic_multigraph(self) -> nx.MultiDiGraph:
        left, bottom, right, top = self.bbox
        rng = np.random.default_rng(42)

        deg_lat = 150 / 111_320
        deg_lon = 150 / (111_320 * np.cos(np.radians((bottom + top) / 2)))

        lats = np.arange(bottom, top, deg_lat)
        lons = np.arange(left, right, deg_lon)
        rows, cols = len(lats), len(lons)

        G = nx.MultiDiGraph()
        node_id = 0
        grid_ids: Dict = {}

        for r, lat in enumerate(lats):
            for c, lon in enumerate(lons):
                jlon = lon + rng.uniform(-deg_lon * 0.1, deg_lon * 0.1)
                jlat = lat + rng.uniform(-deg_lat * 0.1, deg_lat * 0.1)
                G.add_node(node_id, x=float(jlon), y=float(jlat), osmid=node_id)
                grid_ids[(r, c)] = node_id
                node_id += 1

        # Build edges with realistic road + waterway topology
        for r in range(rows):
            for c in range(cols):
                u = grid_ids[(r, c)]
                ux, uy = G.nodes[u]["x"], G.nodes[u]["y"]

                neighbors: list = []
                if r + 1 < rows:
                    neighbors.append((grid_ids[(r + 1, c)], EDGE_TYPE_ROAD, 0.0))
                if c + 1 < cols:
                    neighbors.append((grid_ids[(r, c + 1)], EDGE_TYPE_ROAD, 0.0))
                if r + 1 < rows and c + 1 < cols and rng.random() < 0.3:
                    neighbors.append((grid_ids[(r + 1, c + 1)], EDGE_TYPE_ROAD, 0.0))
                # Synthetic drains along south-flowing routes
                if r + 1 < rows and rng.random() < 0.08:
                    neighbors.append((grid_ids[(r + 1, c)], EDGE_TYPE_WATERWAY, 0.40))

                for v, etype, dcap in neighbors:
                    vx, vy = G.nodes[v]["x"], G.nodes[v]["y"]
                    dist = _haversine(ux, uy, vx, vy)
                    G.add_edge(u, v, key=0, length=dist, highway="residential",
                               _edge_type=etype, _drain_capacity=dcap)
                    G.add_edge(v, u, key=0, length=dist, highway="residential",
                               _edge_type=etype, _drain_capacity=dcap)

        logger.info("Synthetic: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
        return G

    # ── Conversion ────────────────────────────────────────────────────────────

    def _convert_to_digraph(self, G_multi: nx.MultiDiGraph) -> nx.DiGraph:
        G = nx.DiGraph()
        for node, attrs in G_multi.nodes(data=True):
            G.add_node(node, **attrs)

        for u, v, data in G_multi.edges(data=True):
            length = max(float(data.get("length", 1.0)), 1.0)
            if G.has_edge(u, v):
                if length < G[u][v]["length"]:
                    G[u][v].update(data)
                    G[u][v]["length"] = length
            else:
                G.add_edge(u, v, **data)
                G[u][v]["length"] = length

        G.remove_edges_from(list(nx.selfloop_edges(G)))

        wcc = max(nx.weakly_connected_components(G), key=len)
        G = G.subgraph(wcc).copy()
        G = nx.convert_node_labels_to_integers(G, label_attribute="osmid")

        logger.info("DiGraph (largest WCC): %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
        return G

    # ── GeoDataFrame Export ───────────────────────────────────────────────────

    def _build_node_geodataframe(self, G: nx.DiGraph) -> gpd.GeoDataFrame:
        records = []
        for node, attrs in G.nodes(data=True):
            lon = attrs.get("x", np.nan)
            lat = attrs.get("y", np.nan)
            records.append({
                "node_id": node,
                "osmid": attrs.get("osmid", node),
                "lon": lon,
                "lat": lat,
                "geometry": Point(lon, lat),
            })

        gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=self.crs_geo)
        gdf_proj = gdf.to_crs(self.crs_proj)
        gdf["x_proj"] = gdf_proj.geometry.x
        gdf["y_proj"] = gdf_proj.geometry.y
        gdf.set_index("node_id", inplace=True)
        return gdf

    # ── Edge Feature Computation ──────────────────────────────────────────────

    def _compute_edge_features(
        self,
        G: nx.DiGraph,
        gdf_nodes: gpd.GeoDataFrame,
    ) -> np.ndarray:
        """
        Compute physics-informed edge features [E, 4]:
          0: elev_diff_norm  — normalised elevation difference (z_u - z_v)
                               positive = flows downhill from u to v
          1: length_norm     — edge length normalised to [0,1]
          2: edge_type       — 0=road, 1=waterway/drain
          3: flow_weight     — hydrological flow probability
                               = sigmoid(elev_diff / length) for drains,
                                 0 for roads (undirected)

        Elevation at this stage uses proxy from lat (lower lat = lower elevation
        in coastal Chennai); Phase 2 SRTM values are stored in gdf_nodes later
        and edge features can be refined via refine_edge_elevations().
        """
        edges = list(G.edges(data=True))
        E = len(edges)
        feat = np.zeros((E, self.edge_feature_dim), dtype=np.float32)

        # Node coordinate arrays for vectorised computation
        node_lats = gdf_nodes["lat"].values  # proxy for elevation (crude)
        node_lons = gdf_nodes["lon"].values
        node_ids = list(gdf_nodes.index)
        id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}

        all_lengths = [max(float(d.get("length", 1.0)), 1.0) for _, _, d in edges]
        max_len = max(all_lengths) if all_lengths else 1.0

        # Use lat as crude elevation proxy: south Chennai is lower (coastal)
        lat_min = node_lats.min()
        lat_max = node_lats.max()
        lat_range = max(lat_max - lat_min, 1e-6)
        # Elevation proxy: lower lat = lower elevation (coastal) in Chennai
        elev_proxy = (node_lats - lat_min) / lat_range  # [0,1] increasing with lat

        # Global elev range for normalisation
        elev_range = elev_proxy.max() - elev_proxy.min()

        for i, (u, v, data) in enumerate(edges):
            u_idx = id_to_idx.get(u, 0)
            v_idx = id_to_idx.get(v, 0)

            length = all_lengths[i]
            elev_u = elev_proxy[u_idx]
            elev_v = elev_proxy[v_idx]
            elev_diff = elev_u - elev_v  # positive = flow from u to v

            etype = float(data.get("_edge_type", EDGE_TYPE_ROAD))
            dcap = float(data.get("_drain_capacity", 0.0))

            # Normalised elevation difference
            elev_diff_norm = elev_diff / max(elev_range, 1e-6)

            # Normalised length
            length_norm = length / max_len

            # Flow weight: stronger for waterways flowing downhill
            # sigmoid(gradient * capacity_factor)
            gradient = elev_diff / max(length, 1.0)  # m/m (proxy)
            if etype == EDGE_TYPE_WATERWAY:
                flow_weight = float(1.0 / (1.0 + np.exp(-gradient * 1000)))
                # Scale by drainage capacity
                flow_weight *= (0.5 + 0.5 * dcap)
            else:
                # Roads: bidirectional flow, lower weight
                flow_weight = 0.2 + 0.1 * float(np.abs(gradient) > 0.001)

            feat[i, 0] = np.clip(elev_diff_norm, -1.0, 1.0)
            feat[i, 1] = np.clip(length_norm, 0.0, 1.0)
            feat[i, 2] = etype
            feat[i, 3] = np.clip(flow_weight, 0.0, 1.0)

        logger.info("Edge features computed: shape=%s", feat.shape)
        return feat

    def refine_edge_elevations(
        self,
        G: nx.DiGraph,
        node_elevations: np.ndarray,
    ) -> np.ndarray:
        """
        Recompute edge features using real SRTM elevations (call after Phase 2).
        Returns updated [E, 4] edge feature array.
        """
        assert self.edge_features is not None
        edges = list(G.edges(data=True))
        E = len(edges)
        node_ids = list(G.nodes())
        id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}

        elev = node_elevations
        elev_range = max(elev.max() - elev.min(), 1.0)

        all_lengths = [max(float(d.get("length", 1.0)), 1.0) for _, _, d in edges]
        max_len = max(all_lengths) if all_lengths else 1.0

        feat = self.edge_features.copy()
        for i, (u, v, data) in enumerate(edges):
            u_idx = id_to_idx.get(u, 0)
            v_idx = id_to_idx.get(v, 0)
            length = all_lengths[i]
            elev_diff = float(elev[u_idx] - elev[v_idx])  # metres
            elev_diff_norm = elev_diff / elev_range

            etype = float(data.get("_edge_type", EDGE_TYPE_ROAD))
            dcap = float(data.get("_drain_capacity", 0.0))

            gradient = elev_diff / max(length, 1.0)
            if etype == EDGE_TYPE_WATERWAY:
                flow_weight = float(1.0 / (1.0 + np.exp(-gradient * 50)))
                flow_weight *= (0.5 + 0.5 * dcap)
            else:
                flow_weight = 0.2 + 0.1 * float(abs(gradient) > 0.01)

            feat[i, 0] = np.clip(elev_diff_norm, -1.0, 1.0)
            feat[i, 3] = np.clip(flow_weight, 0.0, 1.0)

        self.edge_features = feat
        logger.info("Edge features refined with SRTM elevations.")
        return feat


# ─── Utility ──────────────────────────────────────────────────────────────────

def _haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    R = 6_371_000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))
