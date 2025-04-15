#!/usr/bin/env python
# coding: utf-8

# In[11]:


import laspy
import numpy as np
import rasterio
from rasterio.enums import Resampling
import rioxarray as rxr
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import minimum_filter
import os
import glob
from rasterio.merge import merge
from rasterio.plot import show


# In[17]:


# --- Load raw LiDAR Z values for DSM/DTM ---
def read_lidar_z(las_file):
    with laspy.open(las_file) as f:
        las = f.read()
    return las.x, las.y, las.z


# In[19]:


def create_grid(x, y, z, resolution=1, method='max'):
    xi = np.arange(np.min(x), np.max(x), resolution)
    yi = np.arange(np.min(y), np.max(y), resolution)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    if method == 'max':
        z_grid = griddata((x, y), z, (xi_grid, yi_grid), method='nearest')
    elif method == 'min':
        z_grid = griddata((x, y), z, (xi_grid, yi_grid), method='nearest')
        z_grid = minimum_filter(z_grid, size=5)  # Approximate ground
    else:
        raise ValueError("Method must be 'max' or 'min'")
    
    return xi, yi, z_grid


# In[ ]:





# In[21]:


# --- Load first LAS file and generate CHM ---
x, y, z = read_lidar_z("/scratch/jbiswas/7825_2775.las")
res = 1
xi, yi, dsm = create_grid(x, y, z, res, method='max')
_, _, dtm = create_grid(x, y, z, res, method='min')
chm = dsm - dtm
chm[chm < 0] = 0


# In[23]:


# Convert DSM to xarray for CRS handling
dsm_rio = xr.DataArray(dsm, coords=[yi, xi], dims=["y", "x"])
dsm_rio.rio.write_crs("EPSG:2274", inplace=True)  # Replace with correct EPSG if needed


# In[25]:


# --- Load NDVI and reproject to DSM grid ---
ndvi = rxr.open_rasterio("/scratch/jbiswas/NDVI.tif", masked=True).squeeze()
ndvi = ndvi.rio.reproject_match(dsm_rio)


# In[39]:


# --- Prepare training dataset ---
ndvi_vals = ndvi.values.flatten()
dsm_vals = dsm_rio.values.flatten()
valid_idx = (~np.isnan(ndvi_vals)) & (~np.isnan(dsm_vals))
sample_idx = np.random.choice(np.where(valid_idx)[0], size=min(50000, np.sum(valid_idx)), replace=False)

features = pd.DataFrame({
    'dsm': dsm_vals[sample_idx],
    'ndvi': ndvi_vals[sample_idx]
})
labels = (features['ndvi'] > 0.35).astype(int)


# In[33]:


# --- Train Random Forest ---
if np.sum(valid_idx) == 0:
    raise ValueError("No overlapping valid DSM and NDVI pixels. Check your rasters or CRS alignment.")

rf = RandomForestClassifier(n_estimators=300, max_features=2, random_state=123)
rf.fit(features, labels)


# In[43]:


# NDVI raster for matching projection
ndvi = rxr.open_rasterio("/scratch/jbiswas/NDVI.tif", masked=True).squeeze()

# Input/output directories
las_folder = "/scratch/jbiswas/Shelby_las"
output_folder = "/scratch/jbiswas/Canopy_raster"
os.makedirs(output_folder, exist_ok=True)

# Process each LAS file
for las_file in glob.glob(os.path.join(las_folder, "*.las")):
    file_name = os.path.splitext(os.path.basename(las_file))[0]
    print(f"Processing {file_name}...")

    # Read LiDAR data
    x, y, z = read_lidar_z(las_file)
    
    # Create DSM and DTM
    xi, yi, dsm = create_grid(x, y, z, res, method='max')
    _, _, dtm = create_grid(x, y, z, res, method='min')
    chm = dsm - dtm
    chm[chm < 0] = 0

    # Convert DSM to xarray
    dsm_rio = xr.DataArray(dsm, coords=[yi, xi], dims=["y", "x"])
    dsm_rio.rio.write_crs("EPSG:2274", inplace=True)

    # Reproject NDVI to DSM
    ndvi_reproj = ndvi.rio.reproject_match(dsm_rio)

    # Prepare features for prediction
    all_features = pd.DataFrame({
        'dsm': dsm_rio.values.flatten(),
        'ndvi': ndvi_reproj.values.flatten()
    })

    valid_idx = (~np.isnan(all_features['dsm'])) & (~np.isnan(all_features['ndvi']))
    predicted = np.full(all_features.shape[0], np.nan)
    predicted[valid_idx] = rf.predict(all_features[valid_idx])
    canopy_prediction = predicted.reshape(dsm.shape)

    # Output path
    output_path = os.path.join(output_folder, f"{file_name}_canopy.tif")

    # Write raster
    with rasterio.open(
        output_path, 'w',
        driver='GTiff',
        height=canopy_prediction.shape[0],
        width=canopy_prediction.shape[1],
        count=2,
        dtype='float32',
        crs=ndvi.rio.crs,
        transform=dsm_rio.rio.transform()
    ) as dst:
        dst.write(canopy_prediction.astype('float32'), 1)
        dst.set_band_description(1, "Canopy Classification")
        dst.write(chm.astype('float32'), 2)
        dst.set_band_description(2, "Canopy Height")

    print(f"Saved output to {output_path}")


# In[ ]:


# Directory where your canopy rasters are saved
input_folder = "/scratch/jbiswas/Canopy_raster"
output_file = "/scratch/jbiswas/Canopy_raster/merged_canopy.tif"

# Find all raster files
raster_files = sorted(glob.glob(os.path.join(input_folder, "*_canopy.tif")))

# Read all rasters
src_files_to_mosaic = []
for fp in raster_files:
    src = rasterio.open(fp)
    src_files_to_mosaic.append(src)

# Merge
mosaic, out_trans = merge(src_files_to_mosaic, method='max')  # Or 'max', 'min', 'mean'

# Use metadata from the first raster
out_meta = src_files_to_mosaic[0].meta.copy()
out_meta.update({
    "height": mosaic.shape[1],
    "width": mosaic.shape[2],
    "transform": out_trans,
    "count": mosaic.shape[0],  # should be 2
})

# Write merged raster
with rasterio.open(output_file, "w", **out_meta) as dest:
    dest.write(mosaic)

