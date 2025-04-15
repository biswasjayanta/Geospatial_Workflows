import arcpy
import pandas as pd
import os

# Set the workspace to your geodatabase location
arcpy.env.workspace = r"C:\Users\jbiswas\OneDrive - The University of Memphis\U of M\Fall 2024\Spatial Analysis\FinalProject\FinalProject.gdb"

# Get a list of raster datasets in the workspace
rasters = arcpy.ListRasters()

# Print the list of rasters (for verification)
print(rasters)

def count_pixels_for_class(raster, class_value, output_csv):
    # Convert raster to a table with pixel counts using RasterToTable
    output_table = os.path.join(arcpy.env.workspace, f"{raster}")
    
    
    # List to store results
    pixel_count = []
    
    # Query the table to find pixel counts for the class
    with arcpy.da.SearchCursor(output_table, ["Value", "Count"]) as cursor:
        for row in cursor:
            if row[0] == class_value:
                pixel_count.append({"Raster": raster, "Class": row[0], "Pixel Count": row[1]})
    
    # Convert results to a DataFrame
    df = pd.DataFrame(pixel_count)
    
    # If there are results, append to CSV file
    if not df.empty:
        df.to_csv(output_csv, mode='a', header=not os.path.exists(output_csv), index=False)
        print(f"Pixel count for {raster} exported to {output_csv}")
    else:
        print(f"No pixels found for class {class_value} in {raster}")

# Define the class you are interested in
class_value = 24  # Example: looking for class '24'

# Define the output CSV file path
output_csv = r"C:\Users\jbiswas\OneDrive - The University of Memphis\U of M\Fall 2024\Spatial Analysis\FinalProject\pixel_counts.csv"

# Clear the CSV if it already exists (optional)
if os.path.exists(output_csv):
    os.remove(output_csv)

# Loop through each raster in the list
for raster in rasters:
    count_pixels_for_class(raster, class_value, output_csv)
