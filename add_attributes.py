#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:48:04 2025

@author: kfha
"""

# Define the source CRS (EPSG:25833) and target CRS (EPSG:25832)
source_crs = 'EPSG:25833'  # UTM33
target_crs = 'EPSG:25832'  # UTM32

# Define the transformer to convert UTM33 coordinates to UTM32
transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)

ind = 3
for i,x,y in zip(range(len(landslide['x'][:])), landslide['x'][:], landslide['y'][:]): #[:ind]
#for i,x,y in zip(range(len(landslide['x'][:ind])), landslide['x'][:ind], landslide['y'][:ind]):
    landslide.loc[i,'bedrock'] = (bed[bed.contains(Point(x,y))]['hovedberga'].values[0])
    landslide.loc[i,'deposit'] = (dep[dep.contains(Point(x,y))]['losmassety'].values[0])
    x,y = transformer.transform(x,y)
    if ar5.contains(Point(x,y)).sum() != 0:
        landslide.loc[i,'soil'] = (ar5[ar5.contains(Point(x,y))]['grunnforho'].values[0])
        landslide.loc[i,'lu'] = (ar5[ar5.contains(Point(x,y))]['arealtype'].values[0])
    else:
        print (f'no shape for x,y={x},{y}')
    #print (i,landslide.iloc[i]) 

#%%

landslide.to_csv('landslide_attributes.csv', index=False)
