"""
Data Validation Script

Validates all required data files are present and properly formatted
before running the training pipeline.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path to ensure correct config import
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
import rasterio
from rasterio.crs import CRS
import geopandas as gpd
import pandas as pd


class DataValidator:
    """Validate all required data files."""
    
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.errors = []
        self.warnings = []
        
    def check_file_exists(self, file_path: Path, description: str) -> bool:
        """Check if a file exists."""
        if not file_path.exists():
            self.errors.append(f"Missing {description}: {file_path}")
            return False
        logger.success(f"✓ Found {description}")
        return True
    
    def validate_raster(
        self,
        raster_path: Path,
        expected_crs: str,
        description: str,
    ) -> bool:
        """Validate raster file."""
        if not self.check_file_exists(raster_path, description):
            return False
        
        try:
            with rasterio.open(raster_path) as src:
                # Check CRS
                if src.crs is None:
                    self.warnings.append(f"{description} has no CRS defined")
                elif str(src.crs) != expected_crs:
                    self.warnings.append(
                        f"{description} CRS mismatch: {src.crs} (expected {expected_crs})"
                    )
                
                # Check dimensions
                logger.info(f"  Shape: {src.shape}, CRS: {src.crs}, NoData: {src.nodata}")
                
                # Check for data
                sample = src.read(1, window=((0, min(100, src.height)), (0, min(100, src.width))))
                if sample.size == 0:
                    self.errors.append(f"{description} appears to be empty")
                    return False
                
                valid_pixels = (~rasterio.mask.raster_geometry_mask(src, [], invert=True)[0]).sum()
                if valid_pixels == 0:
                    self.errors.append(f"{description} has no valid pixels")
                    return False
                
                logger.info(f"  Valid pixels: {valid_pixels:,}")
                
        except Exception as e:
            self.errors.append(f"Error reading {description}: {e}")
            return False
        
        return True
    
    def validate_all(self, config: Dict) -> bool:
        """
        Validate all required data files.
        
        Parameters
        ----------
        config : Dict
            Configuration dictionary
            
        Returns
        -------
        bool
            True if all validations pass
        """
        logger.info("=" * 80)
        logger.info("DATA VALIDATION")
        logger.info("=" * 80)
        
        target_crs = config['location']['target_crs']
        
        # 1. Validate DEM
        logger.info("\n[1/6] Validating DEM...")
        dem_path = Path(config['data']['rasters']['dem'])
        self.validate_raster(dem_path, target_crs, "Digital Elevation Model")
        
        # 2. Validate NDVI
        logger.info("\n[2/6] Validating NDVI...")
        ndvi_path = Path(config['data']['rasters']['ndvi'])
        self.validate_raster(ndvi_path, target_crs, "NDVI")
        
        # 3. Validate NDWI
        logger.info("\n[3/6] Validating NDWI...")
        ndwi_path = Path(config['data']['rasters']['ndwi'])
        self.validate_raster(ndwi_path, target_crs, "NDWI")
        
        # 4. Validate NDBI
        logger.info("\n[4/6] Validating NDBI...")
        ndbi_path = Path(config['data']['rasters']['ndbi'])
        self.validate_raster(ndbi_path, target_crs, "NDBI")
        
        # 5. Validate Imperviousness (optional)
        logger.info("\n[5/6] Validating Imperviousness...")
        imp_path = Path(config['data']['rasters'].get('imperviousness', 'none'))
        if imp_path.exists():
            self.validate_raster(imp_path, target_crs, "Imperviousness")
        else:
            self.warnings.append("Imperviousness layer not found (will use default values)")
        
        # 6. Validate SAR (optional)
        logger.info("\n[6/6] Validating SAR...")
        sar_path = Path(config['data']['rasters'].get('sar_vv', 'none'))
        if sar_path.exists():
            self.validate_raster(sar_path, target_crs, "SAR VV")
        else:
            self.warnings.append("SAR data not found (will use default values)")
        
        # 7. Check rainfall data
        logger.info("\n[7/7] Checking Rainfall Data...")
        rainfall_dir = Path(config['data']['precipitation'].get('gpm_imerg_dir', 'none'))
        if rainfall_dir.exists() and any(rainfall_dir.iterdir()):
            logger.success(f"✓ Found rainfall data in {rainfall_dir}")
        else:
            self.warnings.append("No rainfall data found (will use synthetic data for testing)")
        
        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 80)
        
        if self.errors:
            logger.error(f"\n{len(self.errors)} ERRORS found:")
            for error in self.errors:
                logger.error(f"  ✗ {error}")
        
        if self.warnings:
            logger.warning(f"\n{len(self.warnings)} WARNINGS:")
            for warning in self.warnings:
                logger.warning(f"  ⚠ {warning}")
        
        if not self.errors:
            logger.success("\n✓ All critical validations passed!")
            logger.info("\nYou can proceed with:")
            logger.info("  python pipeline/train.py")
            return True
        else:
            logger.error("\n✗ Validation failed. Please fix errors above.")
            logger.info("\nTo acquire data, run:")
            logger.info("  python pipeline/acquire_data.py")
            return False


def main():
    """Main validation entry point."""
    from config import load_config
    
    config = load_config()
    validator = DataValidator(Path("data"))
    
    success = validator.validate_all(config.model_dump())
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
