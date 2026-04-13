# Data Directory

This directory contains all data for the Hydro-Graph ST-GNN project.

## Structure

```
data/
├── raw/                        # Raw input data
│   ├── gpm_imerg/             # GPM IMERG precipitation data
│   └── imd_chennai_2015.csv   # IMD gauge data
├── rasters/                    # GeoTIFF raster files
│   ├── srtm_30m_chennai.tif           # Digital Elevation Model
│   ├── sentinel2_ndvi_10m.tif         # NDVI
│   ├── sentinel2_ndwi_10m.tif         # NDWI
│   ├── sentinel2_ndbi_10m.tif         # NDBI
│   ├── imperviousness_10m.tif         # Imperviousness
│   └── sentinel1_vv.tif               # SAR VV polarization
├── graphs/                     # Processed graph files
│   ├── urban_graph.gpickle    # NetworkX graph
│   ├── nodes.geojson          # Node geometries
│   └── edges.geojson          # Edge geometries
└── processed/                  # Processed datasets
    └── temporal_datasets/      # PyG Data objects
```

## Data Sources

### 1. Digital Elevation Model (DEM)
- **Source**: SRTM 30m or ASTER GDEM
- **Resolution**: 30 meters
- **URL**: https://earthexplorer.usgs.gov/
- **Format**: GeoTIFF

### 2. Sentinel-2 Optical Data
- **Source**: ESA Copernicus
- **Bands Required**: B2 (Blue), B3 (Green), B4 (Red), B8 (NIR), B11 (SWIR)
- **Resolution**: 10 meters
- **URL**: https://scihub.copernicus.eu/
- **Format**: GeoTIFF (pre-processed indices)

### 3. Sentinel-1 SAR Data
- **Source**: ESA Copernicus
- **Polarization**: VV
- **Resolution**: 10 meters
- **URL**: https://scihub.copernicus.eu/
- **Format**: GeoTIFF (preprocessed)

### 4. Precipitation Data
- **GPM IMERG**: https://gpm.nasa.gov/data/imerg
  - Resolution: 0.1° (hourly)
  - Format: HDF5/NetCDF
  
- **IMD Gauge Data**: https://www.imdpune.gov.in/
  - Format: CSV with columns [timestamp, station_id, precip_mm]

## Data Preparation

### Download DEM
```bash
# Using SRTM 30m for bbox [12.8, 80.1, 13.2, 80.3]
# Download from: https://earthexplorer.usgs.gov/
# Save as: data/rasters/srtm_30m_chennai.tif
```

### Download Sentinel-2
```bash
# Using Copernicus Data Space
# Select dates around your event
# Compute indices:
#   NDVI = (B8 - B4) / (B8 + B4)
#   NDWI = (B3 - B8) / (B3 + B8)
#   NDBI = (B11 - B8) / (B11 + B8)
```

### Download GPM IMERG
```bash
# Register at: https://disc.gsfc.nasa.gov/
# Download hourly data for event period
# Save to: data/raw/gpm_imerg/
```

## Important Notes

- All rasters must be in the same CRS (preferably EPSG:32644 for Chennai)
- Ensure all rasters cover the entire bounding box
- NoData values should be properly set in GeoTIFF metadata
- For large areas, consider tiling the data

## .gitignore

Large data files (`.tif`, `.nc`, `.hdf`) are excluded from version control.
Only share processed graph files and metadata.
