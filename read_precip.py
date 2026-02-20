#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 10:59:13 2025

@author: kfha
"""

import netCDF4 as nc
import numpy as np

import os
import numpy as np
import xarray as xr
import rasterio
from rasterio.transform import from_origin
from pyproj import Proj, transform

#%%

folder_path = '../data/precip_1h/'  # Path to the folder with .nc files
variable_name = 'precipitation_amount'  # The variable you want to extract
output_tif = 'max_precip_1h.tif'  # Output file name

data_list = []

# Loop through all .nc files in the specified folder
for filename in os.listdir(folder_path):
    if filename.endswith('Z.nc'):
        file_path = os.path.join(folder_path, filename)
        
        # Open the .nc file using xarray
        dataset = xr.open_dataset(file_path)
        
        # Extract the variable and convert to numpy array
        data_array = dataset[variable_name].values  # Assuming the variable is 2D
        
        # Append the data to the list
        data_list.append(data_array[0])

        # Optionally close the dataset
        dataset.close()

# Stack the data arrays along a new axis
stacked_data = np.stack(data_list)

# Compute the maximum value across all files for each grid cell
max_values = np.nanmax(stacked_data, axis=0)

#%%

new_shape = (stacked_data.shape[0] // 3, 3, stacked_data.shape[1], stacked_data.shape[2])
compressed_data = stacked_data.reshape(new_shape).sum(axis=1)

max_values_3h = np.nanmax(compressed_data, axis=0)

#%%

# Get the coordinate and projection information
with xr.open_dataset(os.path.join(folder_path, os.listdir(folder_path)[0])) as ds:
    projection = ds.projection_lcc.attrs['proj4']
    lon = ds['longitude'].values
    lat = ds['latitude'].values
    
#%%


# Define the Lambert Conformal Conic (as an example, you may need to adjust parameters)
lcc_proj = projection #Proj("+proj=lcc +lat_1=lat1 +lat_2=lat2 +lat_0=lat0 +lon_0=lon0 +x_0=0 +y_0=0 +ellps=WGS84")

# UTM Zone 33N projection
utm_proj = Proj("25833")

# Transform
x,y = np.meshgrid(ds['x'].values, ds['y'].values)

# Perform the transformation
transformed_x, transformed_y = transform(lcc_proj, utm_proj, x, y)
    
#%%

plt.pcolormesh(transformed_x, transformed_y, max_values_3h, vmax=30)
plt.contour(transformed_x, transformed_y, ds['altitude'], levels=1, linewidths=.5, colors='k')

#%%

dims = ds['precipitation_amount'].dims[1:]

ds = ds.drop_vars(list(ds.data_vars)[2:-3])  # Drop all existing variables

ds['precip_1h'] = (dims, max_values)
ds['precip_3h'] = (dims, max_values_3h)

ds = ds.drop_dims('time')

#%%

ds.to_netcdf('../data/precip_1h/max_precip_untransformed.nc')

#%%

ds.encoding = {}


# Assign the transformed coordinates to 'x' and 'y'
ds = ds.assign_coords(
    {
        'x': (('y', 'x'), transformed_x),
        'y': (('y', 'x'), transformed_y)
    }
)

ds = ds.set_coords(['x', 'y'])

#%%

ds.to_netcdf('max_precip.nc')

#%%


#plt.pcolormesh(ds['x'], ds['y'], ds['precip_1h'], vmax=30)
#plt.pcolormesh(ds['x'], ds['y'], ds['precip_3h'], vmax=30)

plt.pcolormesh(transformed_x, transformed_y, max_values, ec='k', vmax=3)
plt.contour(transformed_x, transformed_y, ds['altitude'], levels=1, linewidths=.5, colors='k')
plt.scatter(35370.48863215, 6634978.87637225)
plt.scatter(34580.7223257, 6634978.87637225)
plt.scatter(34580.7223257, 6635675.57530853)
plt.xlim(32000, 37000)
plt.ylim(6630000, 6640000)



#%%

''' 
RADAR DATA
'''

folder_path = '../data/radar/'  # Path to the folder with .nc files
variable_name = 'lwe_precipitation_rate'  # The variable you want to extract

data_list = []

# Loop through all .nc files in the specified folder
for filename in os.listdir(folder_path):
    if filename.startswith('norway') and filename.endswith('.nc'):
        file_path = os.path.join(folder_path, filename)
        
        # Open the .nc file using xarray
        dataset = xr.open_dataset(file_path)
        
        # Extract the variable and convert to numpy array
        data_array = dataset[variable_name].values  # Assuming the variable is 2D
        
        for i in range(24):
            # Append the data to the list
            data_list.append(data_array[i])

        # Optionally close the dataset
        dataset.close()

# Stack the data arrays along a new axis
stacked_data = np.stack(data_list)

# Compute the maximum value across all files for each grid cell
max_values = np.nanmax(stacked_data, axis=0)



#%%

new_shape = (stacked_data.shape[0] // 3, 3, stacked_data.shape[1], stacked_data.shape[2])
compressed_data = stacked_data.reshape(new_shape).sum(axis=1)

max_values_3h = np.nanmax(compressed_data, axis=0)

#%%

# Get the coordinate and projection information
with xr.open_dataset(os.path.join(folder_path, os.listdir(folder_path)[0])) as ds:
    projection = ds.projection_utm.attrs['proj4']
    lon = ds['lon'].values
    lat = ds['lat'].values
    
#%%

# ...

#%%

dims = ds['lwe_precipitation_rate'].dims[1:]

#ds = ds['lwe_precipitation_rate'][1:] #ds.drop_vars(list(ds.data_vars)[:])  # Drop all existing variables

ds['precip_1h'] = (dims, max_values)
ds['precip_3h'] = (dims, max_values_3h)

ds = ds.drop_dims('time')

#%%

ds.to_netcdf('../data/radar/max_precip_radar_untransformed.nc')