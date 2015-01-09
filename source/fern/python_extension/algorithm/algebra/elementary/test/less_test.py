import fern
import fern.algorithm as fa
import fern.test as ft


class LessTest(ft.TestCase):

    no_data = -999

    values = [
        [ -2,      -1],
        [  0, no_data],
        [  1,       2]]
    mask = [
        [ False, False],
        [ False, True ],
        [ False, False]]

    def test_less_masked_rasters(self):
        raster1 = self.masked_raster(self.values, self.mask)
        raster2 = raster1
        raster_we_got = raster1 < raster2

        values_we_want = [
            [  0,            0],
            [  0,            9],
            [  0,            0]]
        mask_we_want = self.mask
        raster_we_want = self.masked_raster(values_we_want, mask_we_want,
            value_type=fern.uint8)

        self.assertMaskedRasterEqual(raster_we_got, raster_we_want)

    def test_less_masked_raster_int(self):
        raster = self.masked_raster(self.values, self.mask)
        int_ = 0
        raster_we_got = raster < int_

        values_we_want = [
            [ 1, 1],
            [ 0, 9],
            [ 0, 0]]
        mask_we_want = self.mask
        raster_we_want = self.masked_raster(values_we_want, mask_we_want,
            value_type=fern.uint8)

        self.assertMaskedRasterEqual(raster_we_got, raster_we_want)

    def test_less_int_masked_raster(self):
        int_ = 0
        raster = self.masked_raster(self.values, self.mask)
        raster_we_got = int_ < raster

        values_we_want = [
            [ 0, 0],
            [ 0, 9],
            [ 1, 1]]
        mask_we_want = self.mask
        raster_we_want = self.masked_raster(values_we_want, mask_we_want,
            value_type=fern.uint8)

        self.assertMaskedRasterEqual(raster_we_got, raster_we_want)

    def test_less_masked_raster_float(self):
        raster = self.masked_raster(self.values, self.mask)
        float_ = 0.0
        raster_we_got = raster < float_

        values_we_want = [
            [ 1, 1],
            [ 0, 9],
            [ 0, 0]]
        mask_we_want = self.mask
        raster_we_want = self.masked_raster(values_we_want, mask_we_want,
            value_type=fern.uint8)

        self.assertMaskedRasterEqual(raster_we_got, raster_we_want)

    def test_less_float_masked_raster(self):
        float_ = 0.0
        raster = self.masked_raster(self.values, self.mask)
        raster_we_got = float_ < raster

        values_we_want = [
            [ 0, 0],
            [ 0, 9],
            [ 1, 1]]
        mask_we_want = self.mask
        raster_we_want = self.masked_raster(values_we_want, mask_we_want,
            value_type=fern.uint8)

        self.assertMaskedRasterEqual(raster_we_got, raster_we_want)
