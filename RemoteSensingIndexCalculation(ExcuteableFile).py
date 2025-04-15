#!/usr/bin/env python
# coding: utf-8

# **Dependency: User Must Have Google Earth Earth Engine account and a cloud project**
# 
# Here is the link to sign up for [Google Earth Engine](https://earthengine.google.com/)
# 
# Here is some tutorial: [geemap](https://geemap.org/installation/) 
# 
# I want to developed a code where user has option to choose what kind of file user wants to input:
# *   local shapefile
# *   From Google Earth Engine
# *   Drawing Area of Interest on Map
# 
# After importing the area of interest, user can input their desire date range and how much cloud cover user will consider to calculate the indices.
# 
# User can call either Landsat-8 or Sentinel-2 image.
# 
# Several indics function **NDVI, SAVI, MSAVI, GRVI, NDWI, MNDWI,MAWEI, NDMI, NDBI, and LST** has been created.
# 
# However, if user choose Sentinel-2 then **LST** index will be excluded as Sentinel-2 doesn't have Thermal band.
# 
# User can choose which index user want to calculate. User can calculate multiple index until user want to end the function.
# 
# Image will directly save to users Google Drive

# **Required Package Install**

# In[ ]:


get_ipython().system('pip install geopandas')
get_ipython().system('pip install -U geemap')
get_ipython().system('pip install fiona geojson')


# **Importing Required Libray**

# In[ ]:


import ee
import geemap
geemap.update_package()
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import fiona
import geojson
import json


# **Call Your Google Earth Engine Project and Map**

# In[90]:


geeProject = input("Enter the Google Earth Engine Project ID: ")
ee.Authenticate()
ee.Initialize(project=geeProject)
Map = geemap.Map(center=(35.117752412600176, -89.93545863387348), zoom=9)
Map


# **User Input Function**

# In[91]:


print("How you want to call your shapefile?")
print("1. From Google Earth Engine")
print("2. Local Shapefile")
print("3. From Last Drawing in the Map")
cPath = int(input("Your Choice: "))
if cPath == 1:
    filePath = input("Enter the path to the shapefile: ")
    gdf = gpd.read_file(filePath)
    geojson_str = gdf.to_json()
    geojson_dict = json.loads(geojson_str)
    aoi = ee.FeatureCollection(geojson_dict).geometry()
    Map.addLayer(aoi, {}, 'Area of Interest')
elif cPath == 2:
  filePath = input("Enter the path to the shapefile: ")
  aoi = ee.FeatureCollection(filePath).geometry()
  Map.addLayer(aoi, {}, 'Area of Interest')
elif cPath == 3:
  m = Map.draw_last_feature
  aoi = ee.FeatureCollection(m).geometry()
  Map.addLayer(aoi, {}, 'Area of Interest')
else:
    print("Invalid input")


# **Call Landsat or Sentinel Image**

# In[97]:


sYear = int(input("Start year (for eg: 2021): "))
sMonth = int(input("Start month (for eg: 06): "))
sDate = int(input("Start date (for eg: 01): "))
eYear = int(input("End year (for eg: 2021): "))
eMonth = int(input("End month (for eg: 08): "))
eDate = int(input("End date (for eg: 30): "))

cCover = float(input("Could Coverage: "))
sDate = ee.Date.fromYMD(sYear, sMonth, sDate)
eDate = ee.Date.fromYMD(eYear, eMonth, eDate)


# In[98]:


# Load Landsat 8 or Sentinel-2 image collection for Landsat 8 Image
while True:
    try:
        print("Enter 1 for Landsat-8 Image")
        print("Enter 2 for Sentinel-2 Image")
        imageWanted = int(input("Your Choice: "))
        if imageWanted == 1:
            sImage = ee.ImageCollection('LANDSAT/LC08/C02/T1_TOA') \
                        .filterBounds(aoi) \
                        .filterDate(sDate, eDate) \
                        .filterMetadata('CLOUD_COVER', 'less_than', cCover)

            sImage8 = sImage.median().clip(aoi)
            Map.addLayer(sImage8, {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 0.3}, 'Landsat-8 Image')
            break
        elif imageWanted == 2:
            sImage = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                        .filterBounds(aoi) \
                        .filterDate(sDate, eDate) \
                        .filterMetadata('CLOUDY_PIXEL_PERCENTAGE','less_than', cCover)

            sImage8 = sImage.median().clip(aoi)
            Map.addLayer(sImage8, {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 0.3}, 'Sentinel-2 Image')
            break
        else:
            print("Invalid input. Please enter 1 for Landsat 8 or 2 for Sentinel-2.")
    except ValueError:
        print("Invalid input. Please enter a number (1 for Landsat 8, 2 for Sentinel-2).")


# **Define Function for Indices**

# In[99]:


if imageWanted == 1:
  # Function to calculate NDVI
  def calculate_ndvi(image):
      nir = image.select('B5')
      red = image.select('B4')
      ndvi = image.expression(
          '(NIR-RED)/(NIR+RED)',
          {
              'NIR': nir,
              'RED': red
          }
      ).rename('NDVI')
      return ndvi
  # Function to calculate SAVI
  def calculate_savi(image):
      nir = image.select('B5')
      red = image.select('B4')
      savi = image.expression(
          '((NIR-RED)/(NIR+RED+0.5))*(1.5)',
          {
              'NIR': nir,
              'RED': red
          }
      ).rename('SAVI')
      return savi
  # Function to calculate MSAVI
  def calculate_msavi(image):
      nir = image.select('B5')
      red = image.select('B4')
      msavi = image.expression(
          '((2 * NIR + 1) - sqrt((2 * NIR + 1) ** 2 - 8 * (NIR - RED))) / 2',
          {
              'NIR': nir,
              'RED': red
          }
      ).rename('MSAVI')
      return msavi
  # Function to calculate GRVI
  def calculate_grvi(image):
      green = image.select('B3')
      red = image.select('B4')
      grvi = image.expression(
          '(GREEN-RED)/(GREEN+RED)',
          {
              'GREEN': green,
              'RED': red
          }
      ).rename('GRVI')
      return grvi
  # Function to calculate NDWI
  def calculate_ndwi(image):
      nir = image.select('B5')
      green = image.select('B3')
      ndwi = image.expression(
          '(GREEN-NIR)/(NIR+GREEN)',
          {
              'NIR': nir,
              'GREEN': green
          }
      ).rename('NDWI')
      return ndwi
  # Function to calculate MNDWI
  def calculate_mndwi(image):
      swir = image.select('B6')
      green = image.select('B3')
      mndwi = image.expression(
          '(GREEN-SWIR)/(SWIR+GREEN)',
          {
              'SWIR': swir,
              'GREEN': green
          }
      ).rename('MNDWI')
      return mndwi
  # Function to calculate NDMI
  def calculate_ndmi(image):
      nir = image.select('B5')
      swir = image.select('B6')
      ndmi = image.expression(
          '(NIR-SWIR)/(NIR+SWIR)',
          {
              'NIR': nir,
              'SWIR': swir
          }
      ).rename('NDMI')
      return ndmi
  # Function to calculate MAWEI
  def calculate_mawei(image):
      nir = image.select('B5')
      red = image.select('B4')
      green = image.select('B3')
      blue = image.select('B2')
      swir2 = image.select('B7')
      mawei = image.expression(
          '(5)*(GREEN-NIR)+(BLUE+RED-4*SWIR2)',
          {
              'NIR': nir,
              'SWIR2': swir2,
              'RED': red,
              'GREEN': green,
              'BLUE': blue
          }
      ).rename('MAWEI')
      return mawei
  # Function to calculate NDBI
  def calculate_ndbi(image):
      nir = image.select('B5')
      swir = image.select('B6')
      ndbi = image.expression(
          '(SWIR-NIR)/(NIR+SWIR)',
          {
              'NIR': nir,
              'SWIR': swir
          }
      ).rename('NDBI')
      return ndbi
  # Function to calculate LST
  def calculate_lst(image):
      """
      Function to calculate NDVI, FV, Emissivity, and LST from Landsat 8 data using geemap and Earth Engine.

      Args:
      - image (ee.Image): Landsat 8 image.
      - aoi (ee.Geometry): Area of interest (AOI) for calculations.

      Returns:
      - ee.Image: Image containing NDVI, FV, EM, and LST.
      """

      # Step 1: Calculate NDVI
      ndvi = image.normalizedDifference(['B5', 'B4']).rename('NDVI')

      # Step 2: Calculate minimum and maximum NDVI in the AOI
      ndviMin = ee.Number(ndvi.reduceRegion(
          reducer=ee.Reducer.min(),
          geometry=aoi,
          scale=30,
          maxPixels=1e9
      ).values().get(0))

      ndviMax = ee.Number(ndvi.reduceRegion(
          reducer=ee.Reducer.max(),
          geometry=aoi,
          scale=30,
          maxPixels=1e9
      ).values().get(0))

      # Step 3: Calculate Fractional Vegetation (FV)
      fv = ndvi.subtract(ndviMin) \
              .divide(ndviMax.subtract(ndviMin)) \
              .pow(ee.Number(2)) \
              .rename('FV')

      # Step 4: Calculate Emissivity (EM)
      em = fv.multiply(ee.Number(0.004)).add(ee.Number(0.986)).rename('EM')

      # Step 5: Select Thermal Band (Band 10)
      thermal = image.select('B10').rename('thermal')

      # Step 6: Calculate Land Surface Temperature (LST)
      lst = thermal.expression(
          '(TB / (1 + (0.00115 * (TB / 1.438)) * log(em))) - 273.15', {
              'TB': thermal.select('thermal'),  # Brightness temperature in Kelvin
              'em': em  # Emissivity
          }).rename('LST')
      return lst
elif imageWanted == 2:
  # Function to calculate NDVI
  def calculate_ndvi(image):
      nir = image.select('B8')  # NIR: Band 8
      red = image.select('B4')  # Red: Band 4
      ndvi = image.expression(
          '(NIR-RED)/(NIR+RED)',
          {
              'NIR': nir,
              'RED': red
          }
      ).rename('NDVI')
      return ndvi

  # Function to calculate SAVI
  def calculate_savi(image):
      nir = image.select('B8')  # NIR: Band 8
      red = image.select('B4')  # Red: Band 4
      savi = image.expression(
          '((NIR-RED)/(NIR+RED+0.5))*(1.5)',
          {
              'NIR': nir,
              'RED': red
          }
      ).rename('SAVI')
      return savi

  # Function to calculate MSAVI
  def calculate_msavi(image):
      nir = image.select('B8')  # NIR: Band 8
      red = image.select('B4')  # Red: Band 4
      msavi = image.expression(
          '((2 * NIR + 1) - sqrt((2 * NIR + 1) ** 2 - 8 * (NIR - RED))) / 2',
          {
              'NIR': nir,
              'RED': red
          }
      ).rename('MSAVI')
      return msavi

  # Function to calculate GRVI
  def calculate_grvi(image):
      green = image.select('B3')  # Green: Band 3
      red = image.select('B4')  # Red: Band 4
      grvi = image.expression(
          '(GREEN-RED)/(GREEN+RED)',
          {
              'GREEN': green,
              'RED': red
          }
      ).rename('GRVI')
      return grvi

  # Function to calculate NDWI
  def calculate_ndwi(image):
      nir = image.select('B8')   # NIR: Band 8
      green = image.select('B3')  # Green: Band 3
      ndwi = image.expression(
          '(GREEN-NIR)/(NIR+GREEN)',
          {
              'NIR': nir,
              'GREEN': green
          }
      ).rename('NDWI')
      return ndwi

  # Function to calculate MNDWI
  def calculate_mndwi(image):
      swir = image.select('B11')  # SWIR 1: Band 11
      green = image.select('B3')  # Green: Band 3
      mndwi = image.expression(
          '(GREEN-SWIR)/(SWIR+GREEN)',
          {
              'SWIR': swir,
              'GREEN': green
          }
      ).rename('MNDWI')
      return mndwi

  # Function to calculate NDMI
  def calculate_ndmi(image):
      nir = image.select('B8')  # NIR: Band 8
      swir = image.select('B11')  # SWIR 1: Band 11
      ndmi = image.expression(
          '(NIR-SWIR)/(NIR+SWIR)',
          {
              'NIR': nir,
              'SWIR': swir
          }
      ).rename('NDMI')
      return ndmi

  # Function to calculate MAWEI
  def calculate_mawei(image):
      nir = image.select('B8')   # NIR: Band 8
      red = image.select('B4')   # Red: Band 4
      green = image.select('B3') # Green: Band 3
      blue = image.select('B2')  # Blue: Band 2
      swir2 = image.select('B12')  # SWIR 2: Band 12
      mawei = image.expression(
          '(5)*(GREEN-NIR)+(BLUE+RED-4*SWIR2)',
          {
              'NIR': nir,
              'SWIR2': swir2,
              'RED': red,
              'GREEN': green,
              'BLUE': blue
          }
      ).rename('MAWEI')
      return mawei

  # Function to calculate NDBI
  def calculate_ndbi(image):
      nir = image.select('B8')  # NIR: Band 8
      swir = image.select('B11')  # SWIR 1: Band 11
      ndbi = image.expression(
          '(SWIR-NIR)/(NIR+SWIR)',
          {
              'NIR': nir,
              'SWIR': swir
          }
      ).rename('NDBI')
      return ndbi


# **Define Export Function**

# In[100]:


if imageWanted == 1:
  def export_image(image, image_name):
      # Create and start the export task
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=f"{image_name}_{sYear}",
            folder="GEE_Exports",
            fileNamePrefix=image_name,
            region=aoi,
            fileFormat='GeoTIFF',
            scale=30,
            maxPixels=1e13
        )
        task.start()
        print(f"Export task for '{image_name}' started. Check Google Drive for output.")
elif imageWanted == 2:
  def export_image(image, image_name):
      # Create and start the export task
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=f"{image_name}_{sYear}",
            folder="GEE_Exports",
            fileNamePrefix=image_name,
            region=aoi,
            fileFormat='GeoTIFF',
            scale=10,
            maxPixels=1e13
        )
        task.start()
        print(f"Export task for '{image_name}' started. Check Google Drive for output.")


# **Calculation of Indices and Exporting the Raster**

# In[101]:


if imageWanted == 1:

  while True:
    print("Enter 1 for NDVI")
    print("Enter 2 for SAVI")
    print("Enter 3 for MSAVI")
    print("Enter 4 for GRVI")
    print("Enter 5 for NDWI")
    print("Enter 6 for MNDWI")
    print("Enter 7 for MAWEI")
    print("Enter 8 for NDMI")
    print("Enter 9 for NDBI")
    print("Enter 10 for LST")
    print("Enter 11 to End")
    indexCalculation = int(input("Your Choice: "))
    if indexCalculation == 1:
      ndvi = calculate_ndvi(sImage8)
      export_image(ndvi, "NDVI")
    elif indexCalculation == 2:
      savi = calculate_savi(sImage8)
      export_image(savi, "SAVI")
    elif indexCalculation == 3:
      msavi = calculate_msavi(sImage8)
      export_image(msavi, "MSAVI")
    elif indexCalculation == 4:
      grvi = calculate_grvi(sImage8)
      export_image(grvi, "GRVI")
    elif indexCalculation == 5:
      ndwi = calculate_ndwi(sImage8)
      export_image(ndwi, "NDWI")
    elif indexCalculation == 6:
      mndwi = calculate_mndwi(sImage8)
      export_image(mndwi, "MNDWI")
    elif indexCalculation == 7:
      mawei = calculate_mawei(sImage8)
      export_image(mawei, "MAWEI")
    elif indexCalculation == 8:
      ndmi = calculate_ndmi(sImage8)
      export_image(ndmi, "NDMI")
    elif indexCalculation == 9:
      ndbi = calculate_ndbi(sImage8)
      export_image(ndbi, "NDBI")
    elif indexCalculation == 10:
      lst = calculate_lst(sImage8)
      export_image(lst, "LST")
    elif indexCalculation > 11 or indexCalculation <= 0:
      print("Invalid input: Enter a valid choice")
    elif indexCalculation == 11:
      print("Calculation Ended")
      break

elif imageWanted == 2:
  while True:
    print("Enter 1 for NDVI")
    print("Enter 2 for SAVI")
    print("Enter 3 for MSAVI")
    print("Enter 4 for GRVI")
    print("Enter 5 for NDWI")
    print("Enter 6 for MNDWI")
    print("Enter 7 for MAWEI")
    print("Enter 8 for NDMI")
    print("Enter 9 for NDBI")
    print("Enter 11 to End")
    indexCalculation = int(input("Your Choice: "))
    if indexCalculation == 1:
      ndvi = calculate_ndvi(sImage8)
      export_image(ndvi, "NDVI")
    elif indexCalculation == 2:
      savi = calculate_savi(sImage8)
      export_image(savi, "SAVI")
    elif indexCalculation == 3:
      msavi = calculate_msavi(sImage8)
      export_image(msavi, "MSAVI")
    elif indexCalculation == 4:
      grvi = calculate_grvi(sImage8)
      export_image(grvi, "GRVI")
    elif indexCalculation == 5:
      ndwi = calculate_ndwi(sImage8)
      export_image(ndwi, "NDWI")
    elif indexCalculation == 6:
      mndwi = calculate_mndwi(sImage8)
      export_image(mndwi, "MNDWI")
    elif indexCalculation == 7:
      mawei = calculate_mawei(sImage8)
      export_image(mawei, "MAWEI")
    elif indexCalculation == 8:
      ndmi = calculate_ndmi(sImage8)
      export_image(ndmi, "NDMI")
    elif indexCalculation == 9:
      ndbi = calculate_ndbi(sImage8)
      export_image(ndbi, "NDBI")
    elif indexCalculation > 11 or indexCalculation <= 0:
      print("Invalid input: Enter a valid choice")
    elif indexCalculation == 11:
      print("Calculation Ended")
      break

