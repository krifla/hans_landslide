# Attributes to landslide and control points during storm Hans

Code to assign geological, ecological and meteorological attributes to landslide points and control points for the extreme storm Hans that hit eastern Norway in August 2023.

Part 1: landslides_main.ipynb
- Assigns attributes to each landslide point
- Defines control points based on distribution of landslide points and some chosen attributes
- Assigns attributes to each control point

Part II: landslides_attributes-from-shapefiles.py
- Assigns remaining attributes from shapefiles to landslide points and control points

In addition, read_precip.py is used to read precipitation data from nc files and calculate maximum 3 h precipitation during Hans for each grid cell before converting to a tif file and used in Part 1.
