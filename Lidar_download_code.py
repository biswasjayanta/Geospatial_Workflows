#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import requests
import os

# Load the CSV
df = pd.read_csv("2017_Shelby.csv")

# Output directory for downloads
output_dir = "/scratch/jbiswas/lidar_data"
os.makedirs(output_dir, exist_ok=True)

# List to store failed downloads
failed_links = []

# Function to extract the Google Drive file ID
def extract_file_id(url):
    if "id=" in url:
        return url.split("id=")[-1]
    elif "file/d/" in url:
        return url.split("file/d/")[1].split("/")[0]
    else:
        return None

# Loop through the DataFrame and download each file
for idx, row in df.iterrows():
    url = row['link']
    file_name = row['File_Name']
    file_id = extract_file_id(url)

    if not file_id:
        failed_links.append({'link': url, 'File_Name': file_name, 'error': 'Invalid ID format'})
        continue

    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    try:
        response = requests.get(download_url, stream=True)
        if response.status_code == 200:
            output_path = os.path.join(output_dir, file_name)
            with open(output_path, "wb") as f:
                f.write(response.content)
        else:
            failed_links.append({'link': url, 'File_Name': file_name, 'error': f'Status {response.status_code}'})
    except Exception as e:
        failed_links.append({'link': url, 'File_Name': file_name, 'error': str(e)})

# Save failed downloads to a CSV
failed_df = pd.DataFrame(failed_links)
failed_df.to_csv("failed_downloads.csv", index=False)

print(f"Download complete. {len(failed_links)} files failed.")

