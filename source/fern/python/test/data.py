import numpy
from osgeo import gdal
from osgeo.gdalconst import *


numpy_data_type_by_python_data_type = {
    int: numpy.int64
}


gdal_data_type_by_numpy_data_type = {
    numpy.dtype("int64"): GDT_Int32,
    numpy.dtype("float32"): GDT_Float32
}


def write_raster(
        values,
        no_data_value,
        driver_name,
        dataset_name):
    driver = gdal.GetDriverByName(driver_name)
    assert driver, driver_name
    if isinstance(values, list):
        values = numpy.array(values,
            dtype=numpy_data_type_by_python_data_type[type(values[0][0])])
    value_type = gdal_data_type_by_numpy_data_type[values.dtype]
    nr_bands = 1
    nr_rows = len(values)
    nr_cols = len(values[0])
    dataset = driver.Create(dataset_name, nr_cols, nr_rows, nr_bands,
        value_type)
    band = dataset.GetRasterBand(1)
    if no_data_value is not None:
        band.SetNoDataValue(no_data_value)
    band.WriteArray(values)
