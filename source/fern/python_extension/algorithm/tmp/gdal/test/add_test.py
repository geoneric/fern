import unittest
import numpy as np
import numpy.ma as ma
from osgeo import gdal
from osgeo.gdalconst import *
from fern.algorithm.gdal import add


class AddTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.number = 4
        cls.int_ = 4
        cls.float_ = 5.4

        cls.int_array = np.array([1, 2, 3])
        cls.float_array = np.array([1.5, 2.5, 3.5])

        driver = gdal.GetDriverByName("MEM")
        nr_rows = 3
        nr_cols = 2
        nr_bands = 1

        value_type = GDT_Int32
        cls.int_dataset = driver.Create("", nr_cols, nr_rows, nr_bands,
            value_type)
        cls.int_raster_band = cls.int_dataset.GetRasterBand(1)
        array = np.random.randint(0, 255, (nr_rows, nr_cols)).astype(np.int32)
        cls.int_raster_band.WriteArray(array)

        value_type = GDT_Float32
        cls.float_dataset = driver.Create("", nr_cols, nr_rows, nr_bands,
            value_type)
        cls.float_raster_band = cls.float_dataset.GetRasterBand(1)
        array = np.random.randint(0, 255, (nr_rows, nr_cols)).astype(np.float32)
        cls.float_raster_band.WriteArray(array)

    def test_result_data_type(self):
        # number, number
        self.assertEqual(type(add(self.number, self.number)), type(self.number))

        # number, array
        self.assertEqual(type(add(self.number, self.int_array)),
            type(self.int_array))
        self.assertEqual(type(add(self.int_array, self.number)),
            type(self.int_array))

        # number, raster band
        self.assertEqual(type(add(self.number, self.int_raster_band)),
            type(self.int_array))
        self.assertEqual(type(add(self.int_raster_band, self.number)),
            type(self.int_array))

    def test_result_value_type(self):

        # int, float
        self.assertEqual(type(add(self.int_, self.int_)), type(self.int_))
        self.assertEqual(type(add(self.float_, self.float_)), type(self.float_))
        self.assertEqual(type(add(self.int_, self.float_)), type(self.float_))
        self.assertEqual(type(add(self.float_, self.int_)), type(self.float_))

        # int, array
        self.assertEqual(add(self.int_, self.int_array).dtype, np.int64)
        self.assertEqual(add(self.int_, self.float_array).dtype, np.float64)

        # float, array
        self.assertEqual(add(self.float_, self.int_array).dtype, np.float64)
        self.assertEqual(add(self.float_, self.float_array).dtype, np.float64)

        # int, raster band
        self.assertEqual(add(self.int_, self.int_raster_band).dtype,
            np.int64)
        self.assertEqual(add(self.int_, self.float_raster_band).dtype,
            np.float32)

        # float, raster band
        self.assertEqual(add(self.float_, self.int_raster_band).dtype,
            np.float64)
        self.assertEqual(add(self.float_, self.float_raster_band).dtype,
            np.float64)
