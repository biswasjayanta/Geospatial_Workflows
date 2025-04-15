#!/usr/bin/env python
# coding: utf-8

# In[6]:


get_ipython().system('pip install geopandas')


# In[7]:


get_ipython().system('pip install rasterio fiona')


# In[8]:


import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib as plt
import rasterio
import fiona


# In[9]:


df = pd.read_csv(r"C:\Users\USER\OneDrive - The University of Memphis\Khulna University\BlueGreenSpace\EcologicalIndexGIS\Ecological_Index\Sample.csv")
df


# In[10]:


print(df.isna())


# In[13]:


max_ndvi = df['ndvi_20'].max()
min_ndvi = df['ndvi_20'].min()
max_wet = df['wetness_20'].max()
min_wet = df['wetness_20'].min()
max_ndbsi = df['ndbsi_20'].max()
min_ndbsi = df['ndbsi_20'].min()
max_lst = df['lst_20'].max()
min_lst = df['lst_20'].min()
print(max_ndvi, min_ndvi, max_wet, min_wet, max_ndbsi, min_ndbsi, max_lst, min_lst)


# In[14]:


ndvi = df['ndvi_20']
wetness = df['wetness_20']
ndbsi = df['ndbsi_20']
lst = df['lst_20']


# In[16]:


df['norm_ndvi'] = (ndvi - min_ndvi)/(max_ndvi-min_ndvi)
df['norm_wetness'] = (wetness-min_wet)/(max_wet-min_wet)
df['norm_ndbsi'] = (max_ndbsi - ndbsi)/(max_ndbsi-min_ndbsi)
df['norm_lst'] = (max_lst-lst)/(max_lst-min_lst)

df.head()


# In[18]:


df['prob_ndvi'] = df['norm_ndvi']/df['norm_ndvi'].sum()
df['prob_wetness'] = df['norm_wetness']/df['norm_wetness'].sum()
df['prob_ndbsi'] = df['norm_ndbsi']/df['norm_ndbsi'].sum()
df['prob_lst'] = df['norm_lst']/df['norm_lst'].sum()
df


# In[21]:


ndvi_unqe = df['prob_ndvi'].nunique()
wet_unqe = df['prob_wetness'].nunique()
ndbsi_unqe = df['prob_ndbsi'].nunique()
lst_unqe = df['prob_lst'].nunique()

print(ndvi_unqe, wet_unqe, ndbsi_unqe, lst_unqe)


# In[23]:


k_ndvi = 1/np.log(ndvi_unqe)
k_wet = 1/np.log(wet_unqe)
k_ndbsi = 1/np.log(ndbsi_unqe)
k_lst = 1/np.log(lst_unqe)


# In[24]:


df['en_ndvi'] = -k_ndvi * (df['prob_ndvi']*df['prob_ndvi'].sum())
df['en_wetness'] = -k_wet * (df['prob_wetness']*df['prob_wetness'].sum())
df['en_ndbsi'] = -k_ndbsi * (df['prob_ndbsi']*df['prob_ndbsi'].sum())
df['en_lst'] = -k_lst * (df['prob_lst']*df['prob_lst'].sum())

df


# In[32]:


df['w_en_ndvi'] = (1-df['en_ndvi'])/(4-df['en_ndvi'].sum())
df['w_en_wetness'] = (1-df['en_wetness'])/(4-df['en_wetness'].sum())
df['w_en_ndbsi'] = (1-df['en_ndbsi'])/(4-df['en_ndbsi'].sum())
df['w_en_lst'] = (1-df['en_lst'])/(4-df['en_lst'].sum())
df


# In[33]:


df['rmsi'] = (df['norm_ndvi']*df['w_en_ndvi']) + (df['norm_wetness']*df['w_en_wetness']) + (df['norm_ndbsi']*df['w_en_ndbsi']) + (df['norm_lst']*df['w_en_lst'])


# In[34]:


df


# In[37]:


get_ipython().system('pip install folium')


# In[38]:


import folium


# In[41]:


# Initialize a map centered at the first data point
map_center = [df['Lat'].mean(), df['Long'].mean()]
m = folium.Map(location=map_center, zoom_start=5)
m


# In[39]:


# Add points to the map
for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row['Lat'], row['Long']],
        radius=row['rmsi'] * 3,  # Scale the RMSI for better visibility
        popup=f"RMSI: {row['rmsi']}",
        color='blue',
        fill=True,
        fill_opacity=0.6
    ).add_to(m)

# Save the map to an HTML file or display it
m.save("map_with_rmsi.html")
m


# In[42]:


import plotly.express as px

fig = px.density_heatmap(df, x="rmsi", y="lst_20")
fig.show()


# In[43]:


fig = px.scatter(df, x='rmsi', y='lst_20')
fig.show()


# In[44]:


df.to_csv('rmsi_index.csv', header=True, index=False)


# In[ ]:




