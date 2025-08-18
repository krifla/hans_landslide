#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 19:51:13 2025

@author: kfha
"""

fn = 'data/bedrock/berggrunn250flate.shp'
bed = gpd.read_file(fn)

fn = 'data/deposits/losmasse50flate.shp'
dep = gpd.read_file(fn)

#%%

fn = 'data/ar5/fkb_ar5_hans_no98.shp'
ar5 = gpd.read_file(fn)
ar5 = ar5.drop(columns=['datafangst','opphav','klassifise','Shape_Leng','Shape_Area'])#'arealtype'])
#ar5 = ar5.to_crs(epsg=25833)

#%%

#fn = 'data/ar50/ar50arealtype.shp'
#ar50 = gpd.read_file(fn)
#ar50.crs
#ar50 = ar50.to_crs(epsg=25833)
