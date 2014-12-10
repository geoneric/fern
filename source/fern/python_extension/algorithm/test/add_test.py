import unittest
import fern
import fern.algorithm as fa
import fern.test as ft


class AddTest(ft.TestCase):

    def test_read_masked_raster(self):
        no_data = -999

        values = [
            [ -2,      -1],
            [  0, no_data],
            [  1,       2]]
        mask = [
            [ False, False],
            [ False, True ],
            [ False, False]]

        raster1 = self.masked_raster(values, mask)
        raster2 = raster1
        raster_we_got = raster1 + raster2

        values_we_want = [
            [ -4,      -2],
            [  0, no_data],
            [  2,       4]]
        raster_we_want = self.masked_raster(values_we_want, mask)

        self.assertMaskedRasterEqual(raster_we_got, raster_we_want)
