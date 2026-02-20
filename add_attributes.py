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

# assigning attributes from shapefiles to control points

for i,x,y in zip(range(len(control_points['x'][:])), control_points['x'][:], control_points['y'][:]):
#    control_points.loc[i,'bedrock'] = (bed[bed.contains(Point(x,y))]['hovedberga'].values[0])
#    control_points.loc[i,'deposit'] = (dep[dep.contains(Point(x,y))]['losmassety'].values[0])
#    if len(dep_thick[dep_thick.contains(Point(x,y))]['MEKT_NR'].values) != 0:
#        control_points.loc[i,'deposit_thickness'] = (dep_thick[dep_thick.contains(Point(x,y))]['MEKT_NR'].values[0])
    x,y = transformer.transform(x,y)
    if ar5.contains(Point(x,y)).sum() != 0:
#        control_points.loc[i,'soil'] = (ar5[ar5.contains(Point(x,y))]['grunnforho'].values[0])
        control_points.loc[i,'lu'] = (ar5[ar5.contains(Point(x,y))]['arealtype'].values[0])
    elif ar5.contains(Point(x,y)).sum() == 0 and ar50.contains(Point(x,y)).sum() != 0:
        control_points.loc[i,'lu'] = (ar50[ar50.contains(Point(x,y))]['artype'].values[0])
        print ('adding landuse with ar50 at x,y={x},{y}')
    else:
        print (f'no shape for x,y={x},{y}')

#%%
        
control_points.to_csv('control_attributes.csv', index=False)

#%%

# assigning attributes from shapefiles to landslide points

for i,x,y in zip(range(len(landslide_points['x'][:])), landslide_points['x'][:], landslide_points['y'][:]):
    #landslide_points.loc[i,'bedrock'] = (bed[bed.contains(Point(x,y))]['hovedberga'].values[0])
#    landslide_points.loc[i,'deposit'] = (dep[dep.contains(Point(x,y))]['losmassety'].values[0])
#    landslide_points.loc[i,'deposit_thickness'] = (dep_thick[dep_thick.contains(Point(x,y))]['MEKT_NR'].values[0])
    x,y = transformer.transform(x,y)
    if ar5.contains(Point(x,y)).sum() != 0:
#        landslide_points.loc[i,'soil'] = (ar5[ar5.contains(Point(x,y))]['grunnforho'].values[0])
        landslide_points.loc[i,'lu'] = (ar5[ar5.contains(Point(x,y))]['arealtype'].values[0])
    elif ar5.contains(Point(x,y)).sum() == 0 and ar50.contains(Point(x,y)).sum() != 0:
        landslide_points.loc[i,'lu'] = (ar50[ar50.contains(Point(x,y))]['artype'].values[0])
        print ('adding landuse with ar50 at x,y={x},{y}')
    else:
        print (f'no shape for x,y={x},{y}')

#%%

#common_columns = [col for col in control_points.columns if col in df1.columns]
landslide_points = landslide_points.reindex(columns=control_points.columns)

#%%

landslide_points.to_csv('landslide_attributes.csv', index=False)
