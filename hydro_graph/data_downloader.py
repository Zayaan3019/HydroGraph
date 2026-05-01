"""
Data Downloader — Real Data Acquisition for Hydro-Graph
=========================================================
Downloads real geospatial data for Chennai flood forecasting:

  1. SRTM 30m DEM  — via srtm.py package (no auth required)
                     Falls back to enhanced synthetic if unavailable
  2. DFO Flood Archive — Dartmouth Flood Observatory event records
                         for Chennai/Tamil Nadu, used for label validation
  3. GPM IMERG rainfall — structured interface (requires NASA Earthdata creds)
                          Falls back to physics-grounded synthetic

All downloads are cached. Functions return True on success, False on fallback.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─── SRTM DEM Downloader ──────────────────────────────────────────────────────

def download_srtm_elevations(
    lats: np.ndarray,
    lons: np.ndarray,
    cache_dir: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Download real SRTM 30m elevation for each node location.

    Uses the `srtm.py` package (pip install srtm.py) which downloads
    SRTM tiles from CGIAR-CSI without authentication.

    Parameters
    ----------
    lats, lons : arrays of node latitude/longitude (EPSG:4326)
    cache_dir  : directory to cache SRTM tiles

    Returns
    -------
    elevations : [N] float32 array in metres, or None if download fails
    """
    try:
        import srtm  # type: ignore
        logger.info("Downloading SRTM elevation for %d nodes via srtm.py …", len(lats))

        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            elevation_data = srtm.get_data(local_cache_dir=str(cache_dir))
        else:
            elevation_data = srtm.get_data()

        elevations = np.full(len(lats), np.nan, dtype=np.float32)
        n_failed = 0

        for i, (lat, lon) in enumerate(zip(lats, lons)):
            val = elevation_data.get_elevation(float(lat), float(lon))
            if val is not None and val > -9000:
                elevations[i] = float(val)
            else:
                n_failed += 1

        if n_failed > 0:
            logger.warning("SRTM: %d/%d nodes returned None; imputing with neighbours", n_failed, len(lats))
            _impute_nan_spatial(elevations, lons, lats)

        valid_pct = 100.0 * np.isfinite(elevations).mean()
        logger.info("SRTM download complete: %.1f%% valid | range=[%.1f, %.1f]m",
                    valid_pct, np.nanmin(elevations), np.nanmax(elevations))
        return elevations

    except ImportError:
        logger.warning("srtm.py not installed (pip install srtm.py). Using synthetic elevation.")
        return None
    except Exception as exc:
        logger.warning("SRTM download failed (%s). Using synthetic elevation.", exc)
        return None


def download_srtm_raster(
    bbox: Tuple[float, float, float, float],
    output_path: str,
    cache_dir: Optional[str] = None,
) -> bool:
    """
    Download SRTM DEM as a GeoTIFF using the `elevation` package.
    Requires GDAL and the elevation package (pip install elevation).

    bbox: (left, bottom, right, top) in EPSG:4326
    Returns True on success, False if unavailable.
    """
    try:
        import elevation  # type: ignore
        left, bottom, right, top = bbox
        output_path = str(output_path)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        logger.info("Downloading SRTM raster for bbox %s via elevation package …", bbox)
        elevation.clip(bounds=(left, bottom, right, top), output=output_path)

        if Path(output_path).exists() and Path(output_path).stat().st_size > 1000:
            logger.info("SRTM raster saved -> %s", output_path)
            return True
        return False

    except ImportError:
        logger.info("elevation package not available; skipping raster download")
        return False
    except Exception as exc:
        logger.warning("SRTM raster download failed: %s", exc)
        return False


# ─── DFO Flood Archive ────────────────────────────────────────────────────────

_DFO_URL = (
    "https://floodobservatory.colorado.edu/Archives/MasterListrev.csv"
)

def download_dfo_events(
    output_path: str,
    country: str = "India",
    region_keywords: List[str] = None,
    min_year: int = 2005,
) -> Optional[pd.DataFrame]:
    """
    Download Dartmouth Flood Observatory event archive for India/Chennai.

    Returns a DataFrame of flood events with columns:
    ID, GlideNumber, Country, OtherCountry, Long, Lat, Area, Began, Ended,
    Validation, Dead, Displaced, MainCause, Severity

    Filters to India events near Tamil Nadu / Chennai.
    """
    region_keywords = region_keywords or ["Chennai", "Tamil Nadu", "Adyar", "Cooum"]
    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Use cached version if recent
    if Path(output_path).exists():
        try:
            df = pd.read_csv(output_path)
            logger.info("DFO archive loaded from cache: %d events", len(df))
            return df
        except Exception:
            pass

    try:
        import requests
        logger.info("Downloading DFO Flood Archive …")
        resp = requests.get(_DFO_URL, timeout=30)
        resp.raise_for_status()

        from io import StringIO
        df_all = pd.read_csv(StringIO(resp.text), encoding="latin-1", on_bad_lines="skip")

        # Filter to India
        if "Country" in df_all.columns:
            df_india = df_all[df_all["Country"].str.contains(country, case=False, na=False)].copy()
        else:
            df_india = df_all.copy()

        # Filter by year
        if "Began" in df_india.columns:
            df_india["year"] = pd.to_datetime(df_india["Began"], errors="coerce").dt.year
            df_india = df_india[df_india["year"] >= min_year]

        df_india.to_csv(output_path, index=False)
        logger.info("DFO archive: %d India events saved -> %s", len(df_india), output_path)
        return df_india

    except ImportError:
        logger.warning("requests not installed; cannot download DFO archive")
        return None
    except Exception as exc:
        logger.warning("DFO download failed (%s)", exc)
        return None


def get_chennai_flood_events(dfo_df: Optional[pd.DataFrame]) -> List[Dict]:
    """
    Extract confirmed Chennai-area flood events from DFO archive.
    Returns list of dicts with 'began', 'ended', 'severity', 'lat', 'lon'.
    """
    # Known Chennai flood events (from published literature)
    known_events = [
        {"name": "2015 Chennai Floods", "began": "2015-11-01", "ended": "2015-12-05",
         "severity": 2, "lat": 13.0827, "lon": 80.2707, "peak_mm_day": 494},
        {"name": "2018 Chennai Floods", "began": "2018-10-15", "ended": "2018-11-15",
         "severity": 1, "lat": 13.0827, "lon": 80.2707, "peak_mm_day": 280},
        {"name": "2021 Cyclone Nivar", "began": "2021-11-25", "ended": "2021-12-05",
         "severity": 1, "lat": 13.0827, "lon": 80.2707, "peak_mm_day": 200},
        {"name": "2023 Cyclone Michaung", "began": "2023-12-04", "ended": "2023-12-08",
         "severity": 2, "lat": 13.0827, "lon": 80.2707, "peak_mm_day": 350},
    ]

    if dfo_df is not None and len(dfo_df) > 0:
        # Filter DFO events near Chennai (within ~1 degree)
        bbox_lat = (12.5, 13.5)
        bbox_lon = (79.8, 80.5)
        if "Lat" in dfo_df.columns and "Long" in dfo_df.columns:
            mask = (
                dfo_df["Lat"].between(*bbox_lat) &
                dfo_df["Long"].between(*bbox_lon)
            )
            nearby = dfo_df[mask]
            for _, row in nearby.iterrows():
                known_events.append({
                    "name": f"DFO-{row.get('ID', 'unknown')}",
                    "began": str(row.get("Began", "")),
                    "ended": str(row.get("Ended", "")),
                    "severity": int(row.get("Severity", 1)),
                    "lat": float(row.get("Lat", 13.0827)),
                    "lon": float(row.get("Long", 80.2707)),
                })

    return known_events


# ─── GPM IMERG Interface ──────────────────────────────────────────────────────

def load_gpm_imerg(
    csv_path: str,
    timestamps: pd.DatetimeIndex,
    node_lats: np.ndarray,
    node_lons: np.ndarray,
    N: int,
) -> Optional[np.ndarray]:
    """
    Load GPM IMERG rainfall from CSV (if available).

    Expected CSV format:
        timestamp, lat, lon, rainfall_mm_hr
    (One row per location per time step; spatially interpolated to node locations.)

    Returns [T, N] float32 array or None.
    """
    if not Path(csv_path).exists():
        return None

    try:
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        df = df[df["timestamp"].between(timestamps[0], timestamps[-1])]

        if len(df) == 0:
            logger.warning("GPM CSV loaded but no data in event window")
            return None

        # Pivot to [T, stations] then spatially interpolate to nodes
        pivot = df.pivot_table(
            index="timestamp", columns="lat", values="rainfall_mm_hr", fill_value=0.0
        )
        pivot = pivot.reindex(timestamps, fill_value=0.0)

        # Spatial interpolation: IDW from station locations to node locations
        station_lats = np.array([float(c) for c in pivot.columns])
        station_rains = pivot.values  # [T, S]

        from scipy.interpolate import griddata
        T = len(timestamps)
        rainfall = np.zeros((T, N), dtype=np.float32)

        # Use lat-only for 1D IDW (simple version)
        for t in range(T):
            rainfall[t, :] = np.interp(
                node_lats, station_lats, station_rains[t, :],
                left=station_rains[t, 0], right=station_rains[t, -1]
            )

        logger.info("GPM IMERG loaded: %dx%d, max=%.1fmm/hr", T, N, rainfall.max())
        return rainfall

    except Exception as exc:
        logger.warning("GPM IMERG load failed (%s); using synthetic", exc)
        return None


# ─── Utility ──────────────────────────────────────────────────────────────────

def _impute_nan_spatial(
    values: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
) -> None:
    """Fill NaN values by averaging nearest valid neighbours (in-place)."""
    nan_mask = np.isnan(values)
    if not nan_mask.any():
        return

    valid_mask = ~nan_mask
    if not valid_mask.any():
        values[:] = 0.0
        return

    from scipy.spatial import cKDTree
    valid_coords = np.column_stack([lons[valid_mask], lats[valid_mask]])
    nan_coords = np.column_stack([lons[nan_mask], lats[nan_mask]])

    tree = cKDTree(valid_coords)
    dists, idxs = tree.query(nan_coords, k=min(5, valid_coords.shape[0]))

    # IDW
    with np.errstate(divide="ignore"):
        weights = 1.0 / np.maximum(dists, 1e-10)
    weights /= weights.sum(axis=1, keepdims=True)

    valid_values = values[valid_mask]
    values[nan_mask] = (weights * valid_values[idxs]).sum(axis=1)
