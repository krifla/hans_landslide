#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

import xarray as xr
import rioxarray
import rasterio
import richdem as rd
#import xdem
import cartopy.crs as ccrs
from pyproj import CRS

from osgeo import gdal
import inspect
import os

import random
from scipy.spatial import distance


# # DTM attributes

# In[95]:


# function for computing terrain attributes (slope, aspect, profile curvature)

def compute_terrainattr(ds, attr='slope_degrees', method=None):

    dem = rd.rdarray(ds.values, no_data=-9999)
    dem.projection = ds.rio.crs.to_string()
    dem.geotransform = ds.rio.transform().to_gdal()

    if method == None:
        att = rd.TerrainAttribute(dem, attrib=attr)
        print (f'{attr} calculated')
    elif method == 'D8':
        dem = rd.FillDepressions(dem, epsilon=True, in_place=False)
        print ('depressions in dem filled')
        att = rd.FlowAccumulation(dem, method=method)
        print (f'flow accumulation calculated with method {method}')
        
    return att


# In[90]:


fn_in = 'data_in/dtm.tif'
fn_out = 'data_out/dtm_attr_noflowacc.nc'

if not os.path.exists(fn_out):
    
    print ('assigning DTM attributes, this takes some time')
    
    # open DTM
    dtm = rioxarray.open_rasterio(fn)

    # reorganise array
    dtm = xr.Dataset({'elevation': dtm[0]})
    
    # compute terrain attributes from DTM
    slope = compute_terrainattr(dtm['elevation'], 'slope_degrees')
    profcurv = compute_terrainattr(dtm['elevation'], 'profile_curvature')
    aspect = compute_terrainattr(dtm['elevation'], 'aspect')
    #flowacc = compute_terrainattr(dtm['elevation'], method='D8')

    # redefine aspect
    aspect_sin = np.sin(np.deg2rad(aspect))
    aspect_cos = np.cos(np.deg2rad(aspect))

    # reorganise array
    dtm['slope'] = (('y', 'x'), slope)
    dtm['profcurv'] = (('y', 'x'), profcurv)
    dtm['aspect_sin'] = (('y', 'x'), aspect_sin)
    dtm['aspect_cos'] = (('y', 'x'), aspect_cos)
    #dtm['flowacc'] = (('y', 'x'), flowacc)
    
    # define TPI
    if not os.path.exists('data_out/TPI.tif'):
        tpi = gdal.DEMProcessing('data_out/TPI.tif', gdal.Open(fn_in), 'TPI', computeEdges=True)
    tpi_data = rioxarray.open_rasterio('data_out/TPI.tif')
    dtm['tpi'] = tpi_data[0] # include TPI in DTM attributes
    
    # define TRI
    if not os.path.exists('data_out/TRI.tif'):
        tri = gdal.DEMProcessing('data_out/TRI.tif', gdal.Open(fn_in), 'TRI', computeEdges=True)
    tri_data = rioxarray.open_rasterio('data_out/TRI.tif')
    dtm['tri'] = tri_data[0] # include TRI in DTM attributes

    # save dtm attributes to new tif file
    dtm.to_netcdf('data_out/dtm_attr_noflowacc.nc')
    
else:
    
    print ('opening DTM attributes')
    
    dtm = xr.open_dataset('data_out/dtm_attr_noflowacc.nc')


# In[2]:


dtm_flowacc = xr.open_dataset('data_out/dtm_attr_flowacc.nc')
dtm_flowacc


# In[ ]:


# add FLOW_ACC and SPI to DTM attributes

if not os.path.exists('data_out/dtm_attr_flowacc.nc'):
    
    flowacc = compute_terrainattr(dtm['elevation'], method='D8')
    print ('flow accumulation calculated')
    dtm['flowacc'] = (('y', 'x'), flowacc)

    
    
    def calculate_spi(flow_acc_block, slope_block):
        return np.log((flow_acc_block*1*1 + 0.001) * np.tan(slope_block + 0.001))
        
    spi = calculate_spi(flowacc, dtm['slope']*np.pi/180)
    print ('spi calculated')
    dtm['spi'] = (('y', 'x'), spi)
    
    # save flowacc and spi attributes
    dtm[['elevation', 'slope', 'flowacc','spi']].to_netcdf('data_out/dtm_attr_flowacc.nc')
    print ('flowacc and spi attributes saved to file')

else:
    print ('opening and assigning FLOW_ACC and SPI')
    dtm_flowacc = xr.open_dataset('data_out/dtm_attr_flowacc.nc')
    dtm['flowacc'] = dtm_flowacc['flowacc']
    dtm['spi'] = dtm_flowacc['spi']
    #dtm = dtm[['slope']]


# In[92]:


# assign projected coordinate system
dtm = dtm.rio.write_crs('EPSG:25833')

dtm.rio.crs


# # Forest

# In[11]:


def get_variable_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [name for name, val in callers_local_vars if val is var][0]


# In[13]:


if not os.path.exists('data_out/forest.nc'):
    
    print ('assigning forest attributes')
    
    # load forest data
    
    treetype = rioxarray.open_rasterio('data_in/SRRTRESLAG.tif')
    canopy = rioxarray.open_rasterio('data_in/SRRKRONEDEK.tif')
    treeheight = rioxarray.open_rasterio('data_in/SRRMHOYDE.tif')
    treenumber = rioxarray.open_rasterio('data_in/SRRTREANTALL.tif')
    treevolume = rioxarray.open_rasterio('data_in/SRRVOLMB.tif')
    #biomass = rioxarray.open_rasterio('data_in/SRBMO.tif')

    # reorganising array
    #dtm_data = xr.Dataset({'elevation': dtm_data[0]})
    
    
    
    # double-check CRS
    
    if dtm.rio.crs == treetype.rio.crs == canopy.rio.crs:
        print ('CRS are equal.')
        crs = dtm.rio.crs
    else:
        print ('DTM: ', dtm.rio.crs)
        print ('TREETYPE: ', treetype.rio.crs)
        print ('CANOPY: ', canopy.rio.crs)
        raise ValueError("CRS are not equal.")
        
        
        
    # assign and save forest attributes
        
    forest = xr.Dataset()

    for var in [treetype, canopy, treeheight, treenumber, treevolume]:
        print (get_variable_name(var))
        forest[get_variable_name(var)] = var[0].where((var.x >= dtm['x'].min()) & (var.y >= dtm['y'].min()) 
                                        & (var.x <= dtm['x'].max()) & (var.y <= dtm['y'].max()), drop=True)#})
        
    forest.to_netcdf('data_out/forest.nc')
    
else:
    
    print ('opening forest attributes')
    
    forest = xr.open_dataset('data_out/forest.nc')


# # Forest loss

# In[14]:


if not os.path.exists('data_out/forestloss.nc'):
    
    print ('assigning forest loss attributes')
    
    fl1 = rioxarray.open_rasterio('data_in/forestloss/Hans_GFC-2022_lossyear01.tif')
    fl2 = rioxarray.open_rasterio('data_in/forestloss/Hans_GFC-2022_lossyear02.tif')
    
    fl1 = fl1.rio.reproject(CRS.from_epsg(25833))
    fl2 = fl2.rio.reproject(CRS.from_epsg(25833))
    
    forestloss = xr.concat([fl1, fl2], dim="x")
    forestloss = forestloss[0].where((forestloss.x >= dtm['x'].min()) & (forestloss.y >= dtm['y'].min()) 
                               & (forestloss.x <= dtm['x'].max()) & (forestloss.y <= dtm['y'].max()), drop=True)
    
    forestloss.to_netcdf('data_out/forestloss.nc')
    
else:
    
    print ('opening forest loss attributes')
    
    forestloss = xr.open_dataset('data_out/forestloss.nc')


# # Landforms

# In[15]:


if not os.path.exists('data_out/landforms.nc'):
    
    print ('defining landform attributes')
    
    tpi300 = rioxarray.open_rasterio('data_in/tpi300_stdi.tif')
    tpi2000 = rioxarray.open_rasterio('data_in/tpi2000_stdi.tif')
    
    tpi300 = tpi300.where(tpi300 != -2147483648, np.nan)
    tpi2000 = tpi2000.where(tpi2000 != -2147483648, np.nan)
    
    landforms = xr.full_like(tpi300, np.nan)
    
    conditions = [
        (((tpi300 > -100) & (tpi300 < 100)) & ((tpi2000 > -100) & (tpi2000 < 100)) & (dtm['slope'] <= 5), 5),
        (((tpi300 > -100) & (tpi300 < 100)) & ((tpi2000 > -100) & (tpi2000 < 100)) & (dtm['slope'] > 5), 6), # >5 instead?
        (((tpi300 > -100) & (tpi300 < 100)) & (tpi2000 >= 100), 7),
        (((tpi300 > -100) & (tpi300 < 100)) & (tpi2000 <= -100), 4),
        ((tpi300 <= -100) & ((tpi2000 > -100) & (tpi2000 < 100)), 2),
        ((tpi300 >= 100) & ((tpi2000 > -100) & (tpi2000 < 100)), 9),
        ((tpi300 <= -100) & (tpi2000 >= 100), 3),
        ((tpi300 <= -100) & (tpi2000 <= -100), 1),
        ((tpi300 >= 100) & (tpi2000 >= 100), 10),
        ((tpi300 >= 100) & (tpi2000 <= -100), 8)
    ]
    
    for condition, category in conditions:
        landforms = landforms.where(~condition, category)
        
    landforms = xr.Dataset({'landforms': landforms[0]})

    landforms.to_netcdf('data_out/landforms.nc')
    
else:
    
    print ('opening landform attributes')
    
    landforms = xr.open_dataset('data_out/landforms.nc')
    
landforms = landforms.rio.write_crs("EPSG:25833")

#landforms.rio.to_raster('landforms.tif', dtype='float32', nodata=np.nan)


# # Ground water

# In[16]:


print ('extracting ground water data')
    
groundwater = rioxarray.open_rasterio('data_in/grunnvann_normalized.tif')

#xx_gw, yy_gw = np.meshgrid(groundwater['x'], groundwater['y'])

groundwater = groundwater[0].where((groundwater.x >= dtm['x'].min()) & (groundwater.y >= dtm['y'].min()) 
                                 & (groundwater.x <= dtm['x'].max()) & (groundwater.y <= dtm['y'].max()), drop=True)

groundwater = xr.Dataset({'groundwater': groundwater})


# # Precipitation

# In[17]:


print ('extracting precipitation data')

relprecip = rioxarray.open_rasterio('data_in/hans3_rel_normal.tif')
relprecip = relprecip.where(relprecip > 0, np.nan)

precip = rioxarray.open_rasterio('data_in/hans_080823-100823.tif')
precip = precip.where(precip > 0, np.nan)

# define x,y increments to include x,y values one cell outside the dtm

xstep = abs((relprecip['x'][1]-relprecip['x'][0]).values)
ystep = abs((relprecip['y'][1]-relprecip['y'][0]).values)

# extract subset similar to dtm domain

relprecip = relprecip[0].where((relprecip.x >= dtm['x'].min()-xstep) & (relprecip.y >= dtm['y'].min()-ystep) 
                             & (relprecip.x <= dtm['x'].max()+xstep) & (relprecip.y <= dtm['y'].max()+ystep), drop=True)
relprecip = xr.Dataset({'relprecip': relprecip})

precip = precip[0].where((precip.x >= dtm['x'].min()-xstep) & (precip.y >= dtm['y'].min()-ystep) 
                       & (precip.x <= dtm['x'].max()+xstep) & (precip.y <= dtm['y'].max()+ystep), drop=True)
precip = xr.Dataset({'precip': precip})


# # Assign attributes to landslide points and non-landslide points

# ### Landslide points first

# In[18]:


# apply attributes to landslide df

def get_attr_for_coords(x, y, dtm, attr):
    att = dtm[attr].sel(x=x, y=y, method='nearest').values
    return att


# In[50]:


if not os.path.exists('data_out/landslide_attributes.csv'):
    
    print ('assigning landslide attributes')
    
    # assign attributes necessary for creating control points

    landslide_points = pd.read_csv('data_in/landslide_attributes.csv')
    
    for attr in ['elevation','slope']:
        landslide_points[attr] = landslide_points.apply(lambda row: get_attr_for_coords(row['x'], row['y'], dtm, attr=attr), axis=1)

    landslide_points['relprecip'] = landslide_points.apply(lambda row: get_attr_for_coords(row['x'], row['y'], relprecip, attr='relprecip'), axis=1)

    # save minimised landslide attributes for definition of control points later
    
    if not os.path.exists('data_out/landslide_attributes_for_control_points.csv'):
        landslide_points.to_csv('data_out/landslide_attributes_for_control_points.csv', index=False)

    
    
    # assign other attributes

    for attr in ['profcurv', 'aspect_sin', 'aspect_cos', 'tri', 'tpi']:
        landslide_points[attr] = landslide_points.apply(lambda row: get_attr_for_coords(row['x'], row['y'], dtm, attr=attr), axis=1)

    #for attr in ['spi', 'flowacc']:
    #    landslide_points[attr] = landslide_points.apply(lambda row: get_attr_for_coords(row['x'], row['y'], dtm_flowacc, attr=attr), axis=1)

    for attr in ['canopy', 'treetype', 'treeheight', 'treenumber', 'treevolume']:
        landslide_points[attr] = landslide_points.apply(lambda row: get_attr_for_coords(row['x'], row['y'], forest, attr=attr), axis=1)

    landslide_points['groundwater'] = landslide_points.apply(lambda row: get_attr_for_coords(row['x'], row['y'], groundwater, attr='groundwater'), axis=1)
    landslide_points['precip'] = landslide_points.apply(lambda row: get_attr_for_coords(row['x'], row['y'], precip, attr='precip'), axis=1)
    
    landslide_points['landforms'] = landslide_points.apply(lambda row: get_attr_for_coords(row['x'], row['y'], landforms, attr='landforms'), axis=1)
    
    landslide_points.replace(-9999, np.nan, inplace=True)
    
    landslide_points.to_csv('data_out/landslide_attributes.csv', index=False)
    
else:
    
    print ('opening landslide attributes')
    
    landslide_points = pd.read_csv('data_out/landslide_attributes.csv')


# In[51]:


landslide_points


# #### assign flowacc and SPI

# In[52]:


def get_max_attr_for_coords(x, y, flowacc, attr):
    # Find the nearest x and y indices
    x_idx = np.abs(flowacc['x'] - x).argmin().item()
    y_idx = np.abs(flowacc['y'] - y).argmin().item()

    # Get the indices for the 5x5 grid around (x_idx, y_idx)
    x_min = max(x_idx - 2, 0)
    x_max = min(x_idx + 2, len(flowacc['x']) - 1)
    y_min = max(y_idx - 2, 0)
    y_max = min(y_idx + 2, len(flowacc['y']) - 1)

    # Select the 5x5 grid
    grid_data = flowacc[attr].isel(x=slice(x_min, x_max + 1), y=slice(y_min, y_max + 1))

    # Compute the maximum value in this 5x5 grid
    max_val = grid_data.max().values

    return max_val

def calculate_spi(flow_acc, slope):
    return np.log((flow_acc*5*5) * np.tan(slope)) # 5 m x 5 m gives area of each cell ### why np.log?
#    return np.log((flow_acc*5*5 + 0.001) * np.tan(slope + 0.001)) # 5 m x 5 m gives area of each cell ### why np.log?

def calculate_twi(flow_acc, slope):
    return np.where(slope > 0, np.log((flow_acc*5*5) / np.tan(slope)), 0) ###### *25???
    #return np.log((flow_acc + 0.001) / np.tan(slope + 0.001)) #######################
#ln((flowacc+1)/(tan(slope))


# In[59]:


#landslide_points = pd.read_csv('landslide_attributes.csv')

for attr in ['flowacc']:
    if attr not in landslide_points.columns:

        print (f'adding {attr}')
        
        flowacc = rioxarray.open_rasterio('data_in/flowacc_fill05.tif')
        flowacc = xr.Dataset({'flowacc': flowacc[0]})
        
        landslide_points[attr] = landslide_points.apply(lambda row: get_max_attr_for_coords(row['x'], row['y'], flowacc, attr=attr), axis=1)

if 'spi' not in landslide_points.columns:
    print (f'adding spi')
    landslide_points['spi'] = landslide_points.apply(lambda row: calculate_spi(row['flowacc'], row['slope']*np.pi/180), axis=1)
    #calculate_spi(landslide_points['flowacc'], landslide_points['slope']*np.pi/180)
    
if 'twi' not in landslide_points.columns:
    print (f'adding twi')
    landslide_points['twi'] = landslide_points.apply(lambda row: calculate_twi(row['flowacc'], row['slope']*np.pi/180), axis=1)
    #calculate_twi(landslide_points['flowacc'], landslide_points['slope']*np.pi/180)


# In[109]:


for attr in landslide_points.columns:
    print (attr, '    ', landslide_points[attr].min(), landslide_points[attr].max())


# In[108]:


#landslide_points.to_csv('landslide_attributes.csv', index=False)


# ### now, let's define control points

# #### - this requires few variables in the memory

# In[61]:


# control points are defined using similar elevation, slope and relative precipitation as landslide points
# let's minmise the landslide points with only the relevant attributes

landslide_points = pd.read_csv('data_out/landslide_attributes_for_control_points.csv')
#landslide_points = landslide_points[['x','y','elevation','slope','relprecip']]
landslide_points


# In[62]:


# merge relprecip into dtm to find appropriate mask for selection of control points

relprecip_regridded = relprecip.interp_like(dtm, method="nearest")

if 'spatial_ref' in relprecip_regridded.coords:
    relprecip_regridded = relprecip_regridded.drop_vars('spatial_ref')
    
dtm['relprecip'] = relprecip_regridded['relprecip']


# In[64]:


# minimise dtm to necessary variables
dtm = dtm[['elevation', 'slope', 'relprecip']]


# In[65]:


# flatten arrays

x, y = np.meshgrid(dtm['x'].values, dtm['y'].values)

x_data = x.flatten()#.astype(np.float16).copy()
y_data = y.flatten()#.astype(np.float16).copy()
print ('flattened x,y arrays ready')


# In[66]:


elevation_data = dtm['elevation'].values.flatten().astype(np.float16)
slope_data = dtm['slope'].values.flatten().astype(np.float16)
relprecip_data = dtm['relprecip'].values.flatten().astype(np.float16)
print ('all flattened arrays ready')

dtm_df = pd.DataFrame({
    'x': x_data,
    'y': y_data,
    'elevation': elevation_data,
    'slope': slope_data,
    'relprecip': relprecip_data
})
print ('dtm_df defined')


# In[71]:


np.nanmax(dtm_df['elevation'])


# In[76]:


# delete previous variables to ensure enough memory
del x, y, x_data, y_data, elevation_data, slope_data, relprecip_data, dtm, relprecip_regridded
#del canopy, flowacc, forest, forestloss, groundwater, landforms, precip, relprecip, treeheight, treenumber, treetype, treevolume


# In[77]:


# define area to get control points from

elevation_min = landslide_points['elevation'].min()
elevation_max = landslide_points['elevation'].max()
slope_min = landslide_points['slope'].min()
slope_max = landslide_points['slope'].max()
relprecip_min = landslide_points['relprecip'].min()
relprecip_max = landslide_points['relprecip'].max()

print (elevation_min, elevation_max, slope_min, slope_max, relprecip_min, relprecip_max)

mask_elevation = (dtm_df['elevation'] >= elevation_min) & (dtm_df['elevation'] <= elevation_max)
print ('elevation mask defined')
mask_slope = (dtm_df['slope'] >= slope_min) & (dtm_df['slope'] <= slope_max)
print ('slope mask defined')
mask_relprecip = (dtm_df['relprecip'] >= relprecip_min) & (dtm_df['relprecip'] <= relprecip_max)
print ('precip mask defined')


# In[78]:


# define sample of dtm that meets elevation, slope and relative precipitation criteria
masked_dtm_df = dtm_df[mask_elevation & mask_slope & mask_relprecip]


# In[79]:


# define sample size and extract random subsample of dtm to get control points from
sample_size = len(landslide_points)
sampled_points = masked_dtm_df.sample(n=sample_size*1000, replace=True)


# In[80]:


# double-check if sample with potential control points contains landslide points
matched_points = pd.merge(landslide_points, sampled_points, on=['x', 'y'])

if not matched_points.empty:
    print(f"Common rows found:\n{matched_points}")
else:
    print("No common rows found.")


# In[81]:


# normalise elevation, slope and relative precipitation
    
landslide_list = landslide_points[['elevation', 'slope', 'relprecip']].copy()
landslide_list['elevation'] /= elevation_max
landslide_list['slope'] /= slope_max
landslide_list['relprecip'] /= relprecip_max
landslide_list = landslide_list.values

sampled_list = sampled_points[['elevation', 'slope', 'relprecip']].copy()
sampled_list['elevation'] /= elevation_max
sampled_list['slope'] /= slope_max
sampled_list['relprecip'] /= relprecip_max
sampled_list = sampled_list.values


# In[85]:


# define function which finds the points in the sample that are 
# closest to the landslide points in the 3D normalised elevation-slope-relprecip space

def find_closest_points(landslide_list, sampled_list):
    perc = int(.01*len(sampled_list))
    print (perc)
    indices = []
    used_indices = set()
    
    # Loop through each point in the landslide_list
    for point in landslide_list:
        # Calculate the distance from the point to all points in the sampled_list
        distances = distance.cdist([point], sampled_list, 'euclidean')[0]  # Use [0] since cdist returns a 2D array
        sorted_indices = np.argsort(distances)  # Get indices that would sort the distances
        #print (sorted_indices[:perc])
        random_indices = sorted_indices[:perc]
        random.shuffle(random_indices)
        #print (random_indices)
        
        # Find the first unused index in the sorted order of distances
        for idx in random_indices: #sorted_indices:
            if idx not in used_indices:
                min_index = idx
                break
        
        # Append the index of the closest unused point from the sampled_list to the result
        indices.append(min_index)
        used_indices.add(min_index)
    
    return indices


# In[86]:


# find closest points in sample to define control points
indices = find_closest_points(landslide_list, sampled_list)
control_points = sampled_points.iloc[indices]


# In[87]:


# plot sample together with landslide points and chosen control points

size = 10

fig,ax = plt.subplots(figsize=(15,15))

plt.scatter(sampled_points['x'], sampled_points['y'], size, label='sampled points')
plt.scatter(landslide_points['x'], landslide_points['y'], size, label='landslide points')
plt.scatter(control_points['x'], control_points['y'], size, label='control points')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()


# In[88]:


# plot distribution of elevation, slope and relprecip for landslide points and control points

stat = 'count' #'probability'#
nbins = 20

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

sns.histplot(landslide_points['elevation'], kde=True, ax=ax[0], color='blue', bins=nbins, binrange=(elevation_min,elevation_max), stat=stat, label='Landslide points')
#sns.histplot(sampled_points['elevation'], kde=True, ax=ax[0], color='green', bins=nbins, binrange=(elevation_min,elevation_max), stat=stat, label='Sampled points')
sns.histplot(control_points['elevation'], kde=True, ax=ax[0], color='red', bins=nbins, binrange=(elevation_min,elevation_max), stat=stat, label='Control points')
ax[0].legend()
if stat == 'count':
    ax[0].set_ylim(0,130)

sns.histplot(landslide_points['slope'], kde=True, ax=ax[1], color='blue', bins=nbins, binrange=(slope_min,slope_max), stat=stat, label='Landslide points')
#sns.histplot(sampled_points['slope'], kde=True, ax=ax[1], color='green', bins=nbins, binrange=(slope_min,slope_max), stat=stat, label='Sampled points')
sns.histplot(control_points['slope'], kde=True, ax=ax[1], color='red', bins=nbins, binrange=(slope_min,slope_max), stat=stat, label='Control points')
ax[1].legend()
if stat == 'count':
    ax[1].set_ylim(0,130)

sns.histplot(landslide_points['relprecip'], kde=True, ax=ax[2], color='blue', bins=nbins, binrange=(relprecip_min,relprecip_max), stat=stat, label='Landslide points')
#sns.histplot(sampled_points['relprecip'], kde=True, ax=ax[2], color='green', bins=nbins, binrange=(relprecip_min,relprecip_max), stat=stat, label='Sampled points')
sns.histplot(control_points['relprecip'], kde=True, ax=ax[2], color='red', bins=nbins, binrange=(relprecip_min,relprecip_max), stat=stat, label='Control points')
ax[2].legend()
if stat == 'count':
    ax[2].set_ylim(0,130)

plt.show()

#plt.figure(figsize=(3, 5))
#sns.boxplot(data=landslide_points['elevation'])
#plt.show()


# In[61]:


# save control points
control_points.to_csv('control_attributes.csv', index=False)


# #### finally assign attributes to control points

# In[ ]:


#control_points = pd.read_csv('control_attributes.csv')
control_points


# In[41]:


#def get_attr_for_coords(x, y, dtm, attr):
#    att = dtm[attr].sel(x=x, y=y, method='nearest').values
#    return att


# In[93]:


for attr in ['elevation','slope']:
    control_points[attr] = control_points.apply(lambda row: get_attr_for_coords(row['x'], row['y'], dtm, attr=attr), axis=1)
    
control_points['relprecip'] = control_points.apply(lambda row: get_attr_for_coords(row['x'], row['y'], relprecip, attr='relprecip'), axis=1)

for attr in ['profcurv', 'aspect_sin', 'aspect_cos', 'tri', 'tpi']:
    control_points[attr] = control_points.apply(lambda row: get_attr_for_coords(row['x'], row['y'], dtm, attr=attr), axis=1)

for attr in ['canopy', 'treetype', 'treeheight', 'treenumber', 'treevolume']:
    control_points[attr] = control_points.apply(lambda row: get_attr_for_coords(row['x'], row['y'], forest, attr=attr), axis=1)

control_points['groundwater'] = control_points.apply(lambda row: get_attr_for_coords(row['x'], row['y'], groundwater, attr='groundwater'), axis=1)
    
control_points['precip'] = control_points.apply(lambda row: get_attr_for_coords(row['x'], row['y'], precip, attr='precip'), axis=1)

for attr in ['flowacc']:
    control_points[attr] = control_points.apply(lambda row: get_max_attr_for_coords(row['x'], row['y'], flowacc, attr=attr), axis=1)

control_points['spi'] = control_points.apply(lambda row: calculate_spi(row['flowacc'], row['slope']*np.pi/180), axis=1)
control_points['twi'] = control_points.apply(lambda row: calculate_twi(row['flowacc'], row['slope']*np.pi/180), axis=1)

control_points['landforms'] = control_points.apply(lambda row: get_attr_for_coords(row['x'], row['y'], landforms, attr='landforms'), axis=1)

control_points.to_csv('control_attributes.csv', index=False)


# # figures

# In[38]:


landforms = xr.open_dataset('data/landforms.nc')


# In[68]:


#landforms['landforms']# = landforms['landforms'][0]


# In[44]:


#landforms = landforms.drop_vars('__xarray_dataarray_variable__')
#landforms['landforms']


# In[58]:


xx,yy = np.meshgrid(dtm['x'], dtm['y'])

indx = 100
indy = 10
xstart = 40000
xend = 48000
ystart = 20000
yend = 28000

xx = xx[xstart:xend, ystart:yend]
yy = yy[xstart:xend, ystart:yend]

plt.figure(figsize=(8, 8))
lf = plt.pcolormesh(xx, yy, landforms['landforms'].values[xstart:xend, ystart:yend], 
               vmin=0.5, vmax=10.5, cmap='tab10') #
plt.contour(xx, yy, dtm['elevation'].values[xstart:xend, ystart:yend],
            colors='k')#, cmap='terrain')
plt.colorbar(lf, orientation='horizontal', label='Landforms')
#plt.xlim(10e5, 1.5*10e5)
#plt.ylim(6.7*10e6, 6.8*10e6)
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()


# In[22]:


xx,yy = np.meshgrid(landforms['x'], landforms['y'])

indx = 100
indy = 10
plt.figure(figsize=(8, 8))
plt.pcolormesh(xx[::indx][::indy], yy[::indx][::indy], landforms[0][::indx][::indy], vmin=0, vmax=10, cmap='terrain') #
plt.colorbar(orientation='horizontal', label='landforms')
plt.grid()
plt.show()


# In[32]:


xx,yy = np.meshgrid(tpi300['x'], tpi300['y'])

indx = 100
indy = 10
plt.figure(figsize=(8, 8))
plt.pcolormesh(xx[::indx][::indy], yy[::indx][::indy], tpi300[0][::indx][::indy], vmin=-150, vmax=150, cmap='terrain') #
plt.colorbar(orientation='horizontal', label='tpi300')
plt.grid()
plt.show()


# In[ ]:


xx,yy = np.meshgrid(landforms['x'], landforms['y'])

ind = 10
plt.figure(figsize=(8, 8))
plt.pcolormesh(xx[::ind][::ind], yy[::ind][::ind], landforms[0][::ind][::ind], cmap='terrain') #vmin=0, vmax=50, 
plt.colorbar(orientation='horizontal', label='landforms')
plt.grid()
plt.show()


# In[13]:


#ds_masked_chunked = ds_filled.where(ds_filled['filled_dtm'] != -9999.)  


# In[8]:


xx,yy = np.meshgrid(flowacc['x'], flowacc['y'])

ind = 10
plt.figure(figsize=(8, 8))
plt.pcolormesh(xx[::ind][::ind], yy[::ind][::ind], flowacc['flowacc'][::ind][::ind], vmin=0, vmax=50, cmap='terrain')
plt.colorbar(orientation='horizontal', label='flowacc')
plt.grid()
plt.show()


# In[17]:


ds_masked = ds_filled.where(ds_filled['filled_dtm'] != -9999.)  


# In[21]:


xx,yy = np.meshgrid(ds_filled['x'], ds_filled['y'])

ind = 10
plt.figure(figsize=(8, 8))
plt.pcolormesh(xx[::ind][::ind], yy[::ind][::ind], ds_masked['filled_dtm'][::ind][::ind], cmap='terrain')
plt.colorbar(orientation='horizontal', label='Elevation (m)')
plt.grid()
plt.show()


# In[71]:


xx,yy = np.meshgrid(dtm['x'], dtm['y'])


# In[72]:


plt.figure(figsize=(8, 8))
plt.pcolormesh(xx, yy, dtm['elevation'], cmap='terrain')
plt.colorbar(orientation='horizontal', label='Elevation (m)')
#plt.title("Downsampled DTM Visualization")
plt.grid()
plt.show()


# In[78]:


plt.figure(figsize=(8, 8))
plt.pcolormesh(xx, yy, dtm['slope'])#, vmax=2, cmap='terrain')
plt.colorbar(orientation='horizontal', label='Slope (\u00b0)')
plt.grid()
plt.show()


# In[106]:


plt.figure(figsize=(8, 8))
plt.pcolormesh(xx, yy, dtm['profcurv'], vmin=-5, vmax=5)
plt.colorbar(orientation='horizontal', label='Profile curvature')
plt.grid()
plt.show()


# In[74]:


n_bins = 8
cmap = plt.get_cmap('hsv')
bounds = np.linspace(np.min(aspect), np.max(aspect), n_bins + 1)
norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=cmap.N)

plt.figure(figsize=(8, 8))
plt.pcolormesh(xx, yy, aspect, norm=norm, cmap=cmap)#twilight')
plt.colorbar(orientation='horizontal', label='Aspect (\u00b0)')
plt.grid()
plt.show()


# In[75]:


plt.figure(figsize=(8, 8))
plt.pcolormesh(xx, yy, dtm['aspect_sin'], cmap=cmap)#twilight')
plt.colorbar(orientation='horizontal', label='Sine of aspect')
plt.grid()
plt.show()


# In[76]:


plt.figure(figsize=(8, 8))
plt.pcolormesh(xx, yy, dtm['aspect_cos'], cmap=cmap)#twilight')
plt.colorbar(orientation='horizontal', label='Cosine of aspect')
plt.grid()
plt.show()


# In[80]:


plt.figure(figsize=(8, 8))
plt.pcolormesh(xx, yy, dtm['flowacc'], vmax=100, cmap='viridis')
plt.colorbar(orientation='horizontal', label='Flow accumulation D8')
plt.grid()
plt.show()


# In[90]:


plt.figure(figsize=(8, 8))
plt.pcolormesh(xx, yy, dtm['tpi'], vmin=-.5, vmax=.5)#, cmap='terrain')
plt.colorbar(orientation='horizontal', label='TPI')
plt.grid()
plt.show()


# In[85]:


dtm['tpi'].min(), dtm['tpi'].max()


# In[91]:


plt.figure(figsize=(8, 8))
plt.pcolormesh(xx, yy, dtm['tri'], vmax=20)#, cmap='terrain')
plt.colorbar(orientation='horizontal', label='TRI')
plt.grid()
plt.show()


# In[84]:


plt.figure(figsize=(8, 8))
plt.pcolormesh(xx, yy, dtm['spi'])#, vmax=2, cmap='terrain')
plt.colorbar(orientation='horizontal', label='SPI')
plt.grid()
plt.show()


# ### plot treetype, canopy

# In[32]:


forest = forest.where(forest != -9999, np.nan)


# In[34]:


xx_sr, yy_sr = np.meshgrid(forest['x'], forest['y'])


# In[37]:


plt.figure(figsize=(8, 8))
plt.pcolormesh(xx_sr, yy_sr, forest['treetype'])#, vmax=100, cmap='viridis')
plt.colorbar(orientation='horizontal', label='Tree type')
plt.grid()
plt.show()


# In[38]:


plt.figure(figsize=(8, 8))
plt.pcolormesh(xx_sr, yy_sr, forest['canopy'])#, vmax=100, cmap='viridis')
plt.colorbar(orientation='horizontal', label='Tree canopy (%)')
plt.grid()
plt.show()


# ### plot ground water

# In[95]:


xx_gw, yy_gw = np.meshgrid(groundwater['x'], groundwater['y'])


# In[97]:


plt.figure(figsize=(8, 8))
plt.pcolormesh(xx_gw, yy_gw, groundwater, vmax=100)#, cmap='viridis')
plt.colorbar(orientation='horizontal', label='Ground water')
plt.grid()
plt.show()


# ### plot precipitation

# In[46]:


xx_p, yy_p = np.meshgrid(precip['x'], precip['y'])


# In[44]:


plt.figure(figsize=(8, 8))
plt.pcolormesh(xx_p, yy_p, precip, vmax=150, cmap='viridis')
plt.colorbar(orientation='horizontal', label='3 days acc. precip. (mm)')
plt.grid()
plt.show()


# In[57]:


plt.figure(figsize=(8, 8))
plt.pcolormesh(xx_p, yy_p, relprecip['relprecip'], vmax=200, cmap='viridis')
plt.colorbar(orientation='horizontal', label='3 days acc. precip. relative to normal (%)')
plt.scatter(landslide_points.iloc[18]['x'], landslide_points.iloc[18]['y'], marker='o', facecolor='k')
plt.grid()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




