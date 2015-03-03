import fern
import fern.algorithm as fa
import fern.test as ft


class SlopeTest(ft.TestCase):

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

        raster_we_got = fa.slope(dem)

        no_data = ft.no_data_value(fern.float32)

        values_we_want = [
            [ 0.01178515, 0.11441274, 0.39392081, no_data   , 0.67342925],
            [ 0.1296362 , 0.20615529, 0.60374987, no_data   , no_data   ],
            [ 1.30181491, 0.77540314, 0.64250487, 1.73150945, 1.86823785],
            [ 3.73464012, 3.54153919, 2.576092  , 3.01537728, 2.36008477],
            [ 2.75834608, 3.06698918, 2.58792615, 2.65845108, 1.64738858]]
        mask_we_want = mask
        raster_we_want = self.masked_raster(values_we_want, mask_we_want,
            cell_sizes=cell_sizes, value_type=fern.float32)

        self.assertMaskedRasterEqual(raster_we_got, raster_we_want)
