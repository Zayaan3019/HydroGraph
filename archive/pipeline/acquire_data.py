"""
Real Data Acquisition Script for Chennai Flood Forecasting

This script provides utilities to download and prepare real geospatial data
for the Hydro-Graph ST-GNN model.

Data Sources:
1. DEM: SRTM 30m from USGS/NASA
2. Sentinel-2: Optical imagery from Copernicus
3. Sentinel-1: SAR data from Copernicus
4. Rainfall: GPM IMERG from NASA
"""

from pathlib import Path
from typing import Optional, Tuple, List
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from loguru import logger
import geopandas as gpd
import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
import requests
from tqdm import tqdm


class DEMDownloader:
    """Download and process SRTM DEM data."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def download_srtm_tiles(
        self,
        bbox: Tuple[float, float, float, float],
        output_path: Path,
    ) -> Path:
        """
        Download SRTM 30m DEM tiles for bounding box.
        
        Parameters
        ----------
        bbox : Tuple[float, float, float, float]
            Bounding box (South, West, North, East)
        output_path : Path
            Output path for merged DEM
            
        Returns
        -------
        Path
            Path to downloaded DEM file
        """
        logger.info("SRTM DEM Download Instructions:")
        logger.info("=" * 80)
        logger.info("1. Go to: https://earthexplorer.usgs.gov/")
        logger.info("2. Create free account if needed")
        logger.info(f"3. Search coordinates: {bbox}")
        logger.info("4. Select 'Digital Elevation > SRTM > SRTM 1 Arc-Second Global'")
        logger.info("5. Download tiles covering your area")
        logger.info(f"6. Place downloaded .hgt files in: {self.output_dir}")
        logger.info("7. This script will merge and process them")
        logger.info("=" * 80)
        
        # Check for existing tiles
        hgt_files = list(self.output_dir.glob("*.hgt"))
        
        if not hgt_files:
            logger.warning("No SRTM tiles found. Please download manually from USGS EarthExplorer")
            logger.info("After downloading, run this script again")
            return None
        
        logger.info(f"Found {len(hgt_files)} SRTM tiles")
        
        # Merge tiles if multiple
        if len(hgt_files) > 1:
            logger.info("Merging SRTM tiles...")
            src_files = [rasterio.open(f) for f in hgt_files]
            mosaic, out_trans = merge(src_files)
            
            out_meta = src_files[0].meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans,
                "compress": "lzw"
            })
            
            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(mosaic)
            
            for src in src_files:
                src.close()
                
        else:
            # Single tile - just copy/convert
            with rasterio.open(hgt_files[0]) as src:
                data = src.read()
                meta = src.meta.copy()
                meta.update(driver="GTiff", compress="lzw")
                
                with rasterio.open(output_path, "w", **meta) as dest:
                    dest.write(data)
        
        logger.success(f"DEM saved to: {output_path}")
        return output_path


class SentinelDataPreparer:
    """Prepare Sentinel-2 and Sentinel-1 data."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_sentinel_instructions(self):
        """Print instructions for downloading Sentinel data."""
        logger.info("Sentinel Data Download Instructions:")
        logger.info("=" * 80)
        logger.info("Option 1: Copernicus Data Space (Recommended)")
        logger.info("  1. Go to: https://dataspace.copernicus.eu/")
        logger.info("  2. Create free account")
        logger.info("  3. Search for Chennai area")
        logger.info("  4. Sentinel-2: Download Level-2A products (atmospherically corrected)")
        logger.info("  5. Sentinel-1: Download GRD products (VV polarization)")
        logger.info("")
        logger.info("Option 2: Google Earth Engine")
        logger.info("  1. Use Earth Engine Code Editor")
        logger.info("  2. Export Sentinel-2 composite for Chennai")
        logger.info("  3. Export Sentinel-1 VV composite")
        logger.info("")
        logger.info("Required Bands (Sentinel-2):")
        logger.info("  - B2 (Blue), B3 (Green), B4 (Red), B8 (NIR), B11 (SWIR)")
        logger.info("")
        logger.info(f"Save files to: {self.output_dir}")
        logger.info("=" * 80)
    
    def calculate_ndvi(self, nir_path: Path, red_path: Path, output_path: Path):
        """Calculate NDVI from NIR and Red bands."""
        logger.info("Calculating NDVI...")
        
        with rasterio.open(nir_path) as nir_src, rasterio.open(red_path) as red_src:
            nir = nir_src.read(1).astype(float)
            red = red_src.read(1).astype(float)
            
            # Calculate NDVI
            ndvi = (nir - red) / (nir + red + 1e-8)
            ndvi = np.clip(ndvi, -1, 1)
            
            # Write output
            meta = nir_src.meta.copy()
            meta.update(dtype=rasterio.float32, nodata=-9999)
            
            with rasterio.open(output_path, 'w', **meta) as dst:
                dst.write(ndvi.astype(rasterio.float32), 1)
        
        logger.success(f"NDVI saved to: {output_path}")
        return output_path
    
    def calculate_ndwi(self, green_path: Path, nir_path: Path, output_path: Path):
        """Calculate NDWI from Green and NIR bands."""
        logger.info("Calculating NDWI...")
        
        with rasterio.open(green_path) as green_src, rasterio.open(nir_path) as nir_src:
            green = green_src.read(1).astype(float)
            nir = nir_src.read(1).astype(float)
            
            # Calculate NDWI
            ndwi = (green - nir) / (green + nir + 1e-8)
            ndwi = np.clip(ndwi, -1, 1)
            
            # Write output
            meta = green_src.meta.copy()
            meta.update(dtype=rasterio.float32, nodata=-9999)
            
            with rasterio.open(output_path, 'w', **meta) as dst:
                dst.write(ndwi.astype(rasterio.float32), 1)
        
        logger.success(f"NDWI saved to: {output_path}")
        return output_path
    
    def calculate_ndbi(self, swir_path: Path, nir_path: Path, output_path: Path):
        """Calculate NDBI from SWIR and NIR bands."""
        logger.info("Calculating NDBI...")
        
        with rasterio.open(swir_path) as swir_src, rasterio.open(nir_path) as nir_src:
            swir = swir_src.read(1).astype(float)
            nir = nir_src.read(1).astype(float)
            
            # Calculate NDBI
            ndbi = (swir - nir) / (swir + nir + 1e-8)
            ndbi = np.clip(ndbi, -1, 1)
            
            # Write output
            meta = swir_src.meta.copy()
            meta.update(dtype=rasterio.float32, nodata=-9999)
            
            with rasterio.open(output_path, 'w', **meta) as dst:
                dst.write(ndbi.astype(rasterio.float32), 1)
        
        logger.success(f"NDBI saved to: {output_path}")
        return output_path


class RainfallDataDownloader:
    """Download and process GPM IMERG rainfall data."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_download_instructions(self):
        """Print instructions for downloading rainfall data."""
        logger.info("Rainfall Data Download Instructions:")
        logger.info("=" * 80)
        logger.info("GPM IMERG Data:")
        logger.info("  1. Go to: https://disc.gsfc.nasa.gov/")
        logger.info("  2. Register for free NASA Earthdata account")
        logger.info("  3. Search for 'GPM IMERG Final'")
        logger.info("  4. Select 'GPM_3IMERGHH' (0.1° hourly)")
        logger.info("  5. Download for Chennai flood event (Nov-Dec 2015)")
        logger.info("  6. Use OPeNDAP or direct download")
        logger.info("")
        logger.info("IMD Gauge Data:")
        logger.info("  1. Go to: https://www.imdpune.gov.in/")
        logger.info("  2. Request historical data")
        logger.info("  3. Get hourly rainfall for Chennai stations")
        logger.info("")
        logger.info(f"Save files to: {self.output_dir}")
        logger.info("=" * 80)


def main():
    """Main data acquisition pipeline."""
    logger.info("=" * 80)
    logger.info("HYDRO-GRAPH: REAL DATA ACQUISITION")
    logger.info("=" * 80)
    
    # Setup directories
    base_dir = Path("data")
    raw_dir = base_dir / "raw"
    raster_dir = base_dir / "rasters"
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    raster_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuration for Chennai
    chennai_bbox = (12.8, 80.1, 13.2, 80.3)  # South, West, North, East
    target_crs = "EPSG:32644"  # UTM Zone 44N
    
    # 1. DEM Download
    logger.info("\n[1/3] DEM Data Acquisition")
    logger.info("-" * 80)
    dem_downloader = DEMDownloader(raw_dir / "srtm")
    dem_path = raster_dir / "srtm_30m_chennai.tif"
    
    dem_downloader.download_srtm_tiles(chennai_bbox, dem_path)
    
    # 2. Sentinel Data
    logger.info("\n[2/3] Sentinel Data Acquisition")
    logger.info("-" * 80)
    sentinel_preparer = SentinelDataPreparer(raw_dir / "sentinel")
    sentinel_preparer.get_sentinel_instructions()
    
    # Calculate indices if band files exist
    sentinel_dir = raw_dir / "sentinel"
    nir_path = sentinel_dir / "B8_NIR.tif"
    red_path = sentinel_dir / "B4_RED.tif"
    green_path = sentinel_dir / "B3_GREEN.tif"
    swir_path = sentinel_dir / "B11_SWIR.tif"
    
    if nir_path.exists() and red_path.exists():
        logger.info("Found Sentinel-2 bands, calculating indices...")
        sentinel_preparer.calculate_ndvi(nir_path, red_path, raster_dir / "sentinel2_ndvi_10m.tif")
        
        if green_path.exists():
            sentinel_preparer.calculate_ndwi(green_path, nir_path, raster_dir / "sentinel2_ndwi_10m.tif")
        
        if swir_path.exists():
            sentinel_preparer.calculate_ndbi(swir_path, nir_path, raster_dir / "sentinel2_ndbi_10m.tif")
    
    # 3. Rainfall Data
    logger.info("\n[3/3] Rainfall Data Acquisition")
    logger.info("-" * 80)
    rainfall_downloader = RainfallDataDownloader(raw_dir / "rainfall")
    rainfall_downloader.get_download_instructions()
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("DATA ACQUISITION GUIDE COMPLETE")
    logger.info("=" * 80)
    logger.info("\nNext Steps:")
    logger.info("1. Follow the instructions above to download data")
    logger.info("2. Place all files in the specified directories")
    logger.info("3. Run data validation: python pipeline/validate_data.py")
    logger.info("4. Start training: python pipeline/train.py")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
