import unittest
import numpy
import fern
import fern.feature as ff


python_type_to_value_type = {
    int: fern.int64,
    float: fern.float64
}


def no_data_value(
        value_type):
    no_data_value_by_value_type = {
        fern.int8: numpy.iinfo(numpy.int8).min,
        fern.uint8: numpy.iinfo(numpy.uint8).max,
        fern.int32: numpy.iinfo(numpy.int32).min,
        fern.uint32: numpy.iinfo(numpy.uint32).max,
        fern.int64: numpy.iinfo(numpy.int64).min,
        fern.uint64: numpy.iinfo(numpy.uint64).max,
        fern.float32: numpy.finfo(numpy.float32).min,
        fern.float64: numpy.finfo(numpy.float64).max,
    }
    return no_data_value_by_value_type[value_type]


class TestCase(unittest.TestCase):

    @staticmethod
    def masked_raster(
            values,
            mask,
            origin=(0.0, 0.0),
            cell_sizes=(1.0, 1.0),
            value_type=None):
        if value_type is None:
            value_type = python_type_to_value_type[type(values[0][0])]

        return ff.MaskedRaster(values, mask, origin, cell_sizes, value_type)

    def assertMaskedRasterEqual(self,
            raster_we_got,
            raster_we_want):
        self.assertEqual(type(raster_we_got), ff.MaskedRaster)
        self.assertEqual(raster_we_got.value_type, raster_we_want.value_type)
        self.assertEqual(raster_we_got.sizes[0], raster_we_want.sizes[0])
        self.assertEqual(raster_we_got.sizes[1], raster_we_want.sizes[1])
        self.assertEqual(raster_we_got.cell_sizes, raster_we_want.cell_sizes)
        self.assertEqual(raster_we_got.origin, raster_we_want.origin)

        values_we_got = ff.raster_as_numpy_array(raster_we_got)
        values_we_want = ff.raster_as_numpy_array(raster_we_want)

        self.assertEqual(values_we_got.shape, values_we_want.shape)
        self.assertEqual(len(values_we_got.shape), 2)

        nr_rows = values_we_got.shape[0]
        nr_cols = values_we_got.shape[1]
        value_type = raster_we_got.value_type

        for row in xrange(nr_rows):
            for col in xrange(nr_cols):

                if value_type in [fern.float32, fern.float64]:
                    self.assertAlmostEqual(values_we_got[row][col],
                        values_we_want[row][col],
                        msg="{} != {} for cell {}, {}".format(
                            values_we_got[row][col],
                            values_we_want[row][col], row, col))
                else:
                    self.assertEqual(values_we_got[row][col],
                        values_we_want[row][col],
                        msg="{} != {} for cell {}, {}".format(
                            values_we_got[row][col],
                            values_we_want[row][col], row, col))
