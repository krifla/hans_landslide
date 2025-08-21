# hans_landslide

Codes to assign geological, ecological and meteorological attributes to landslide points and control points for the extreme storm Hans that hit eastern Norway hard in August 2023

'landslides_main.ipynb'
1. uses input data such as a high resolution digitial terrain model (DTM) in combination with geological, ecological and meteorological datasets to assign landslide-relevant attributes to landslide points from the Hans storm.
2. defines control points based on the range of elevation, slope and relative precipitation from the landslide points and assigns the same attributes to these points.

'landslides_attributes-from-shapefiles.py'
1. uses the subscripts 'read_landslide-points.py', 'read_shapefiles' and 'add_attributes' to read the processed landslide points and control points with their attributes and then assign some additional attributes from shapefiles.

These two separate code procedures can be merged in some other data systems, but have been run separately due to the use of two different data systems with different benefits. The main code has been run on a supercomputer and is designed to tackle large input files and calculations that require a lot of memory. The additional code for handling shapefiles has been run on a local computer where some Python packages relevant for shapefiles worked better.

More details on calculations:

1. Profile and planform curvature are calculated using the RichDEM library and the method of Zevenbergen, L. W., & Thorne, C. R. (1987). Quantitative analysis of land surface topography. Earth surface processes and landforms, 12(1), 47-56.

2. Flow acccumulation is calculated using the RichDEM library and the D8 method of O’Callaghan, J.F., Mark, D.M., 1984. The Extraction of Drainage Networks from Digital Elevation Data. Computer vision, graphics, and image processing 28, 323–344.

3. Stream Power Index (SPI) is calculated based on slope and flow accumulation using ln(flow_accumulation)*tan(slope), where flow accumulation is per unit area.

4. Topographic Position Index (TPI) is precalculated in ArcGIS Pro by comparing the DTM elevation of each cell to the mean elevation of the surrounding annulus-shaped neighborhood. TPI is calculated for small scales (TPI300 considering cells in 150 and 300 m radius from each cell) and large scales (TPI2000 considering cells in 1850 and 2000 m radius from each cell).

5. Landforms were classified using the standardised TPI300 and TPI2000 as input in landslides_main.ipynb based on the procedure suggested by Weiss, A. (2001, July). Topographic position and landforms analysis. In Poster presentation, ESRI user conference, San Diego, CA (Vol. 200).
