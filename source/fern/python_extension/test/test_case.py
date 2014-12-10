import unittest
import fern.feature as ff


class TestCase(unittest.TestCase):

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
        mask_we_got = ff.mask_as_numpy_array(raster_we_got)
        mask_we_want = ff.mask_as_numpy_array(raster_we_want)

        self.assertEqual(values_we_got.shape, values_we_want.shape)
        self.assertEqual(mask_we_got.shape, mask_we_want.shape)
        self.assertEqual(len(values_we_got.shape), 2)
        self.assertEqual(len(mask_we_got.shape), 2)
        self.assertEqual(values_we_got.shape, mask_we_got.shape)

        nr_rows = values_we_got.shape[0]
        nr_cols = values_we_got.shape[1]

        for row in xrange(nr_rows):
            for col in xrange(nr_cols):
                self.assertEqual(mask_we_got[row][col], mask_we_want[row][col],
                    "{} != {} for mask cell {}, {}".format(
                        mask_we_got[row][col], mask_we_want[row][col],
                        row, col))

                if not mask_we_got[row][col]:
                    self.assertEqual(values_we_got[row][col],
                        values_we_want[row][col])
