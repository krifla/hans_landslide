# hans_landslide

Codes to assign geological, ecological and meteorological attributes to landslide points and control points for the extreme storm Hans that hit eastern Norway hard in August 2023

'landslides_main.ipynb'
1. uses input data such as a high resolution digitial terrain model (DTM) in combination with geological, ecological and meteorological datasets to assign landslide-relevant attributes to landslide points from the Hans storm.
2. defines control points based on the range of elevation, slope and relative precipitation from the landslide points and assigns the same attributes to these points.

'landslides_attributes-from-shapefiles.py'
1. uses the subscripts 'read_landslide-points.py', 'read_shapefiles' and 'add_attributes' to read the processed landslide points and control points with their attributes and then assign some additional attributes from shapefiles.

These two separate code procedures can be merged in some other data systems, but have been run separately due to the use of two different data systems with different benefits. The main code has been run on a supercomputer and is designed to tackle large input files and calculations that require a lot of memory. The additional code for handling shapefiles has been run on a local computer where some Python packages relevant for shapefiles worked better.
