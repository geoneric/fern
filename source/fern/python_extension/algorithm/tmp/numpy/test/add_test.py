import unittest
import numpy as np
import numpy.ma as ma
from fern.algorithm.numpy import add


class AddTest(unittest.TestCase):

    def test_result_data_type(self):
        number = 4
        array = np.array([1, 2, 3])
        masked_array = ma.masked_array([1, 2, 3], [False, True, False])

        # number, number
        self.assertEqual(type(add(number, number)), type(number))

        # number, array
        self.assertEqual(type(add(number, array)), type(array))
        self.assertEqual(type(add(array, number)), type(array))

        # number, masked array
        self.assertEqual(type(add(number, masked_array)), type(array))
        self.assertEqual(type(add(masked_array, number)), type(array))

        # array, masked array
        self.assertEqual(type(add(array, masked_array)), type(array))
        self.assertEqual(type(add(masked_array, array)), type(array))

    def test_result_value_type(self):
        int_ = 4
        float_ = 5.4
        int_array = np.array([1, 2, 3])
        float_array = np.array([1.5, 2.5, 3.5])
        mask = [False, True, False]
        int_masked_array = ma.masked_array([1, 2, 3], mask)
        float_masked_array = ma.masked_array([1.5, 2.5, 3.5], mask)

        # int, float
        self.assertEqual(type(add(int_, int_)), type(int_))
        self.assertEqual(type(add(float_, float_)), type(float_))
        self.assertEqual(type(add(int_, float_)), type(float_))
        self.assertEqual(type(add(float_, int_)), type(float_))

        # int, array
        self.assertEqual(add(int_, int_array).dtype, np.int64)
        self.assertEqual(add(int_, float_array).dtype, np.float64)
        self.assertEqual(add(float_, float_array).dtype, np.float64)
        self.assertEqual(add(int_, float_array).dtype, np.float64)
