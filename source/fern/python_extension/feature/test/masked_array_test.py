import unittest
import fern.feature as ff


class MaskedArrayTest(unittest.TestCase):

    def xtest_construction(self):
        raster = ff.MaskedArray(
                nr_dimensions=2,
                sizes=(3, 2),
                cell_sizes=(50, 60),
                origin =(100.0, 200.0),
                value_type=fern.int32)
        self.assertEqual(type(raster), ff.MaskedRaster)
        self.assertEqual(raster.size[0], 3)
        self.assertEqual(raster.size[1], 2)
        self.assertEqual(raster.cell_size[0], 50.0)
        self.assertEqual(raster.cell_size[1], 60.0)
        self.assertEqual(raster.origin[0], 200.0)
        self.assertEqual(raster.origin[1], 100.0)

    def xtest_masked_array(self):
        array = ff.masked_array([1, 2, 3], [False, True, False])
        self.assertEqual(type(array), ff.MaskedArray)
        self.assertEqual(array[0], 1)
        self.assertEqual(array[1], 2)
        self.assertEqual(array[2], 3)
        self.assertEqual(array.mask[0], False)
        self.assertEqual(array.mask[1], True)
        self.assertEqual(array.mask[2], False)
