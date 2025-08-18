#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 14:50:51 2025

@author: kfha
"""

import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Point
from pyproj import Transformer


import rasterio
import rioxarray
import geopandas as gpd

def run_script(script_path):
    with open(script_path) as script_file:
        exec(script_file.read(), globals())
        
#%%

if __name__ == "__main__":
    run_script('read_landslide-points.py')
    run_script('read_shapefiles.py')
    run_script('add_attributes.py')

