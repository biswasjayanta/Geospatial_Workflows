import os
import rasterio
from rasterio.merge import merge
from glob import glob

# Define the folder containing the raster files
raster_folder = 'path/to/your/folder'  # ← Change this to your actual folder path

# Get list of all .tif files in the folder
raster_files = glob(os.path.join(raster_folder, '*.tif'))

# Start with an empty list for valid sources
valid_sources = []

for fp in raster_files:
    try:
        src = rasterio.open(fp)
        # Try merging this source with current valid ones
        test_sources = valid_sources + [src]
        merge(test_sources)  # just to test if it works
        valid_sources.append(src)
    except Exception as e:
        print(f"Could not merge {fp}: {e}")

# Ensure there's something to merge
if not valid_sources:
    print("No valid raster files could be merged.")
else:
    # Now do the actual merge
    mosaic, out_trans = merge(valid_sources)

    # Copy metadata from first valid file
    out_meta = valid_sources[0].meta.copy()

    # Update metadata
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "count": mosaic.shape[0]
    })

    # Output file path
    output_path = os.path.join(raster_folder, 'merged.tif')

    # Write to disk
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic)

    print(f"Merged raster saved to: {output_path}")
