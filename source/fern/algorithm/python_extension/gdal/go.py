import numpy
from osgeo import gdal
from osgeo.gdalconst import *
import fern_algorithm_gdal as fern_algorithm


driver = gdal.GetDriverByName("MEM")

nr_rows = 30000
nr_cols = 20000
value_type = GDT_Float32
nr_bands = 1
dataset = driver.Create("", nr_cols, nr_rows, nr_bands, GDT_Float32)
raster_band = dataset.GetRasterBand(1)
array = numpy.random.randint(0, 255, (nr_rows, nr_cols)).astype(numpy.float32)  # 1 copy
raster_band.WriteArray(array)  # 1 copy


# print array

print "---------------"
# for i in xrange(100):
fern_algorithm.add(raster_band, raster_band)  # 3 copies

# print "---------------"
# for i in xrange(100):
#     array + array





# print dir(gdal)


# def print_raster(
#         raster):
#     # print "raster:"
#     # print raster, type(raster)
#     # print dir(raster)
#     # print "raster.this:"
#     # print raster.this, type(raster.this)
#     # print dir(raster.this)
#     assert raster.RasterCount == 1
#     print raster.GetRasterBand(1).ReadAsArray()
# 
# 
# raster = gdal.Open("raster-1.img", GA_ReadOnly)
# print_raster(raster)
# 
# # <class 'osgeo.gdal.Band'>
# # print type(band)
# 
# new_raster = fern_algorithm.add(raster, 5)
# 
# print_raster(new_raster)



# ['AddBand', 'BeginAsyncReader', 'BuildOverviews', 'CreateMaskBand', 'EndAsyncReader', 'FlushCache', 'GetDescription', 'GetDriver', 'GetFileList', 'GetGCPCount', 'GetGCPProjection', 'GetGCPs', 'GetGeoTransform', 'GetMetadata', 'GetMetadataItem', 'GetMetadata_Dict', 'GetMetadata_List', 'GetProjection', 'GetProjectionRef', 'GetRasterBand', 'GetSubDatasets', 'RasterCount', 'RasterXSize', 'RasterYSize', 'ReadAsArray', 'ReadRaster', 'ReadRaster1', 'SetDescription', 'SetGCPs', 'SetGeoTransform', 'SetMetadata', 'SetMetadataItem', 'SetProjection', 'WriteRaster', '__class__', '__del__', '__delattr__', '__dict__', '__doc__', '__format__', '__getattr__', '__getattribute__', '__hash__', '__init__', '__module__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__swig_destroy__', '__swig_getmethods__', '__swig_setmethods__', '__weakref__', '_s', 'this']


### # Read an input raster dataset.
### input_raster_dataset = gdal.Open("raster-float32.img", GA_ReadOnly)
### input_raster_band = input_raster_dataset.GetRasterBand(1);
### 
### print fern_algorithm.add(5.5, 6.5)
### print fern_algorithm.add(input_raster_band, input_raster_band)

# result_array = fern_algorithm.add(input_raster_band, 6)
# print result_array


### # Create an output raster dataset.
### driver = input_raster_dataset.GetDriver()
### nr_bands = 1
### result_raster_dataset = driver.Create("result.img",
###     input_raster_dataset.RasterXSize, input_raster_dataset.RasterYSize,
###     nr_bands, GDT_Float32)
### 
### result_raster_band = result_raster_dataset.GetRasterBand(1);
### result_raster_band.WriteArray(result_array)
