#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:15:03 2025

@author: kfha
"""

# NB! UTM32!

fn = 'data/landslide_points/lspoints_25_03.shp'
gdf = gpd.read_file(fn)

# convert to UTM33
gdf = gdf.to_crs(epsg=25833)

gdf.plot()
plt.show()

# save as df
landslide = pd.DataFrame()
landslide['x'] = gdf['geometry'].x
landslide['y'] = gdf['geometry'].y

#%%

control_points = pd.read_csv('control_attributes.csv')
