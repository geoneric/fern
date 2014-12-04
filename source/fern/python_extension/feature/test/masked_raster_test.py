import unittest
import fern


class MaskedRasterTest(unittest.TestCase):

    def test_construction(self):
        raster = fern.MaskedRaster(
            sizes=(3, 2),
            origin=(100.0, 200.0),
            cell_sizes=(50, 60),
            value_type=fern.int32)
        self.assertEqual(type(raster), fern.MaskedRaster)
        self.assertEqual(raster.sizes[0], 3)
        self.assertEqual(raster.sizes[1], 2)
        self.assertEqual(raster.cell_sizes[0], 50.0)
        self.assertEqual(raster.cell_sizes[1], 60.0)
        self.assertEqual(raster.origin[0], 100.0)
        self.assertEqual(raster.origin[1], 200.0)
        # self.assertEqual(raster[0][0], 0)
        # self.assertEqual(raster[0][1], 0)
        # self.assertEqual(raster[1][0], 0)
        # self.assertEqual(raster[1][1], 0)
        # self.assertEqual(raster[2][0], 0)
        # self.assertEqual(raster[2][1], 0)
        # self.assertEqual(raster.mask[0][0], False)
        # self.assertEqual(raster.mask[0][1], False)
        # self.assertEqual(raster.mask[1][0], False)
        # self.assertEqual(raster.mask[1][1], False)
        # self.assertEqual(raster.mask[2][0], False)
        # self.assertEqual(raster.mask[2][1], False)

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
        self.assertEqual(raster.sizes[0], 3)
        self.assertEqual(raster.sizes[1], 2)
        self.assertEqual(raster.cell_sizes[0], 50.0)
        self.assertEqual(raster.cell_sizes[1], 60.0)
        self.assertEqual(raster.origin[0], 100.0)
        self.assertEqual(raster.origin[1], 200.0)
        # self.assertEqual(raster[0][0], 5)
        # self.assertEqual(raster[0][1], 4)
        # self.assertEqual(raster[1][0], 3)
        # self.assertEqual(raster[1][1], 2)
        # self.assertEqual(raster[2][0], 1)
        # self.assertEqual(raster[2][1], 0)
        # self.assertEqual(raster.mask[0][0], False)
        # self.assertEqual(raster.mask[0][1], True)
        # self.assertEqual(raster.mask[1][0], False)
        # self.assertEqual(raster.mask[1][1], False)
        # self.assertEqual(raster.mask[2][0], True)
        # self.assertEqual(raster.mask[2][1], False)
