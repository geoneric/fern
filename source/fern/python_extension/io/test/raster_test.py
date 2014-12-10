import unittest
import fern
import fern.io as fi
import fern.test as ft


class RasterTest(ft.TestCase):

    def test_read_masked_raster(self):
        driver_name = "HFA"
        dataset_name = "raster-1.hfa"
        no_data = -999
        values = [
            [ -2,      -1],
            [  0, no_data],
            [  1,       2]]
        ft.write_raster(values, no_data, driver_name, dataset_name)

        raster_we_got = fi.read_raster(dataset_name)

        mask = [
            [ False, False ],
            [ False, True  ],
            [ False, False ] ]

        raster_we_want = fern.MaskedRaster(values, mask, origin=(0.0, 0.0),
            cell_sizes=(1.0, 1.0), value_type=fern.int32)
        self.assertMaskedRasterEqual(raster_we_got, raster_we_want)
