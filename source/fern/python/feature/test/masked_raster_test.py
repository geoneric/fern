import unittest
import numpy
import fern


class MaskedRasterTest(unittest.TestCase):

    def test_construction(self):
        raster = fern.MaskedRaster(
            sizes=(3, 2),
            origin=(100.0, 200.0),
            cell_sizes=(50, 60),
            value_type=fern.int32)

        self.assertEqual(type(raster), fern.MaskedRaster)
        self.assertEqual(raster.value_type, fern.int32)
        self.assertEqual(raster.sizes[0], 3)
        self.assertEqual(raster.sizes[1], 2)
        self.assertEqual(raster.cell_sizes[0], 50.0)
        self.assertEqual(raster.cell_sizes[1], 60.0)
        self.assertEqual(raster.origin[0], 100.0)
        self.assertEqual(raster.origin[1], 200.0)

        values = fern.raster_as_numpy_array(raster)
        self.assertEqual(values.dtype, numpy.int32)
        self.assertEqual(values.shape, (3, 2))
        self.assertEqual(values[0][0], 0)
        self.assertEqual(values[0][1], 0)
        self.assertEqual(values[1][0], 0)
        self.assertEqual(values[1][1], 0)
        self.assertEqual(values[2][0], 0)
        self.assertEqual(values[2][1], 0)

        # mask = fern.mask_as_numpy_array(raster)
        # self.assertEqual(mask.dtype, numpy.bool)
        # self.assertEqual(mask.shape, (3, 2))
        # self.assertEqual(mask[0][0], False)
        # self.assertEqual(mask[0][1], False)
        # self.assertEqual(mask[1][0], False)
        # self.assertEqual(mask[1][1], False)
        # self.assertEqual(mask[2][0], False)
        # self.assertEqual(mask[2][1], False)

    def test_masked_raster(self):
        values = [
            [5, 4],
            [3, 2],
            [1, 0]]
        mask = [
            [False, True],
            [False, False],
            [True, False]]
        origin = (100.0, 200.0)
        cell_sizes = (50, 60)

        raster = fern.MaskedRaster(
            values=values,
            mask=mask,
            origin=origin,
            cell_sizes=cell_sizes,
            value_type=fern.int32)

        self.assertEqual(type(raster), fern.MaskedRaster)
        self.assertEqual(raster.value_type, fern.int32)
        self.assertEqual(raster.sizes[0], 3)
        self.assertEqual(raster.sizes[1], 2)
        self.assertEqual(raster.cell_sizes[0], 50.0)
        self.assertEqual(raster.cell_sizes[1], 60.0)
        self.assertEqual(raster.origin[0], 100.0)
        self.assertEqual(raster.origin[1], 200.0)

        values = fern.raster_as_numpy_array(raster)
        self.assertEqual(values.dtype, numpy.int32)
        self.assertEqual(values.shape, (3, 2))
        self.assertEqual(values[0][0], 5)
        self.assertEqual(values[0][1], numpy.iinfo(numpy.int32).min)
        self.assertEqual(values[1][0], 3)
        self.assertEqual(values[1][1], 2)
        self.assertEqual(values[2][0], numpy.iinfo(numpy.int32).min)
        self.assertEqual(values[2][1], 0)

        # mask = fern.mask_as_numpy_array(raster)
        # self.assertEqual(mask.dtype, numpy.bool)
        # self.assertEqual(mask.shape, (3, 2))
        # self.assertEqual(mask[0][0], False)
        # self.assertEqual(mask[0][1], True)
        # self.assertEqual(mask[1][0], False)
        # self.assertEqual(mask[1][1], False)
        # self.assertEqual(mask[2][0], True)
        # self.assertEqual(mask[2][1], False)
