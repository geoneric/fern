import copy
import fern
import fern.algorithm as fa
import fern.test as ft


class IfTest(ft.TestCase):

    no_data = -999

    values1 = [
        [ -2,      -1],
        [  0, no_data],
        [  1,       2]]
    mask1 = [
        [ False, False],
        [ False, True ],
        [ False, False]]

    values2 = [
        [ -4,      -2],
        [  0, no_data],
        [  2,       4]]
    mask2 = [
        [ False, False],
        [ True , False],
        [ False, False]]

    def test_masked_rasters(self):
        raster1 = self.masked_raster(self.values1, self.mask1)
        raster2 = self.masked_raster(self.values2, self.mask2)
        raster_we_got = fa.if_(raster1 < raster2, raster1, raster2)
        print raster_we_got.value_type

        values_we_want = [
            [ -4, -2],
            [  9,  9],
            [  1,  2]]
        mask_we_want = copy.deepcopy(self.mask1)
        mask_we_want[1][0] = True
        raster_we_want = self.masked_raster(values_we_want, mask_we_want)

        self.assertMaskedRasterEqual(raster_we_got, raster_we_want)

    def test_masked_raster_int(self):
        raster1 = self.masked_raster(self.values1, self.mask1)
        raster2 = self.masked_raster(self.values2, self.mask2)
        int_ = 5
        raster_we_got = fa.if_(raster1 < raster2, raster1, int_)

        values_we_want = [
            [  5,  5],
            [  9,  9],
            [  1,  2]]
        mask_we_want = copy.deepcopy(self.mask1)
        mask_we_want[1][0] = True
        raster_we_want = self.masked_raster(values_we_want, mask_we_want,
            value_type=raster1.value_type)

        self.assertMaskedRasterEqual(raster_we_got, raster_we_want)

    def test_greater_int_masked_raster(self):
        raster1 = self.masked_raster(self.values1, self.mask1)
        raster2 = self.masked_raster(self.values2, self.mask2)
        int_ = 5
        raster_we_got = fa.if_(raster1 < raster2, int_, raster1)

        values_we_want = [
            [ -2, -1],
            [  9,  9],
            [  5,  5]]
        mask_we_want = copy.deepcopy(self.mask1)
        mask_we_want[1][0] = True
        raster_we_want = self.masked_raster(values_we_want, mask_we_want,
            value_type=raster1.value_type)

        self.assertMaskedRasterEqual(raster_we_got, raster_we_want)

    def test_masked_raster_float(self):
        raster1 = self.masked_raster(self.values1, self.mask1)
        raster2 = self.masked_raster(self.values2, self.mask2)
        float_ = 5.0
        raster_we_got = fa.if_(raster1 < raster2, raster1, float_)

        values_we_want = [
            [  5,  5],
            [  9,  9],
            [  1,  2]]
        mask_we_want = copy.deepcopy(self.mask1)
        mask_we_want[1][0] = True
        raster_we_want = self.masked_raster(values_we_want, mask_we_want,
            value_type=raster1.value_type)

        self.assertMaskedRasterEqual(raster_we_got, raster_we_want)

    def test_greater_float_masked_raster(self):
        raster1 = self.masked_raster(self.values1, self.mask1)
        raster2 = self.masked_raster(self.values2, self.mask2)
        float_ = 5.0
        raster_we_got = fa.if_(raster1 < raster2, float_, raster1)

        values_we_want = [
            [ -2, -1],
            [  9,  9],
            [  5,  5]]
        mask_we_want = copy.deepcopy(self.mask1)
        mask_we_want[1][0] = True
        raster_we_want = self.masked_raster(values_we_want, mask_we_want,
            value_type=raster1.value_type)

        self.assertMaskedRasterEqual(raster_we_got, raster_we_want)
