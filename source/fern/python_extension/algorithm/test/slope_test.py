import unittest
import fern
import fern.algorithm as fa


class SlopeTest(unittest.TestCase):

    def test_pcraster_example(self):

        values = [
            [  70,  70,  80, -999,  120 ],
            [  70,  70,  90, -999, -999 ],
            [  70,  70, 100,  140,  280 ],
            [ 180, 160, 110,  160,  320 ],
            [ 510, 440, 300,  400,  480 ] ]
        mask = [
            [False, False, False,  True, False ],
            [False, False, False,  True,  True ],
            [False, False, False, False, False ],
            [False, False, False, False, False ],
            [False, False, False, False, False ] ]
        origin = (0.0, 0.0)
        cell_sizes = (50.0, 50.0)

        dem = fern.MaskedRaster(
            values=values,
            mask=mask,
            origin=origin,
            cell_sizes=cell_sizes,
            value_type=fern.float32)

        slope = fa.slope(dem)

        values_we_got = fern.raster_as_numpy_array(slope)
        mask_we_got = fern.mask_as_numpy_array(slope)

        values_we_want = [
            [ 0.01178515, 0.11441274, 0.39392081, 0.        , 0.67342925],
            [ 0.1296362 , 0.20615529, 0.60374987, 0.        , 0.        ],
            [ 1.30181491, 0.77540314, 0.64250487, 1.73150945, 1.86823785],
            [ 3.73464012, 3.54153919, 2.576092  , 3.01537728, 2.36008477],
            [ 2.75834608, 3.06698918, 2.58792615, 2.65845108, 1.64738858]]
        mask_we_want = mask

        for row in xrange(dem.sizes[0]):
            for col in xrange(dem.sizes[1]):
                self.assertEqual(mask_we_got[row][col], mask_we_want[row][col])
                if not mask_we_got[row][col]:
                    self.assertAlmostEqual(values_we_got[row][col],
                        values_we_want[row][col])
