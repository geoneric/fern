// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern algorithm convolution convolve
#include <boost/test/unit_test.hpp>
#include "fern/core/data_customization_point/scalar.h"
#include "fern/core/types.h"
#include "fern/feature/core/data_customization_point/array.h"
#include "fern/feature/core/data_customization_point/masked_array.h"
#include "fern/algorithm/algebra/elementary/equal.h"
#include "fern/algorithm/statistic/count.h"
#include "fern/algorithm/convolution/neighborhood.h"
#include "fern/algorithm/convolution/convolve.h"
#include "fern/algorithm/convolution/replace_no_data_by_focal_average.h"
#include "fern/algorithm/core/test/test_utils.h"


namespace fa = fern::algorithm;


void compare_result1(
    fern::Array<double, 2> const& result)
{
    // Compare results.
    // Upper left corner. ------------------------------------------------------
    // 0, 0:
    // +----+----+----+
    // |  0 |  1 |  2 |
    // +----+----+----+
    // |  6 |  7 |  8 |
    // +----+----+----+
    // | 12 | 13 | 14 |
    // +----+----+----+
    BOOST_CHECK_CLOSE(result[0][0], 63.0 / 9.0, 1e-6);

    // 0, 1:
    // +----+----+----+----+
    // |  0 |  1 |  2 |  3 |
    // +----+----+----+----+
    // |  6 |  7 |  8 |  9 |
    // +----+----+----+----+
    // | 12 | 13 | 14 | 15 |
    // +----+----+----+----+
    BOOST_CHECK_CLOSE(result[0][1], 90.0 / 12.0, 1e-6);

    // 1, 1:
    // +----+----+----+----+
    // |  0 |  1 |  2 |  3 |
    // +----+----+----+----+
    // |  6 |  7 |  8 |  9 |
    // +----+----+----+----+
    // | 12 | 13 | 14 | 15 |
    // +----+----+----+----+
    // | 18 | 19 | 20 | 21 |
    // +----+----+----+----+
    BOOST_CHECK_CLOSE(result[1][1], 168.0 / 16.0, 1e-6);

    // Upper right corner. -----------------------------------------------------
    // 0, 4:
    // +----+----+----+----+
    // |  2 |  3 |  4 |  5 |
    // +----+----+----+----+
    // |  8 |  9 | 10 | 11 |
    // +----+----+----+----+
    // | 14 | 15 | 16 | 17 |
    // +----+----+----+----+
    BOOST_CHECK_CLOSE(result[0][4], 114.0 / 12.0, 1e-6);

    // Lower left corner. ------------------------------------------------------
    // 6, 1:
    // +----+----+----+----+
    // | 24 | 25 | 26 | 27 |
    // +----+----+----+----+
    // | 30 | 31 | 32 | 33 |
    // +----+----+----+----+
    // | 36 | 37 | 38 | 39 |
    // +----+----+----+----+
    BOOST_CHECK_CLOSE(result[6][1], 378 / 12.0, 1e-6);

    // Lower right corner. -----------------------------------------------------
    // 6, 5
    // +----+----+----+
    // | 27 | 28 | 29 |
    // +----+----+----+
    // | 33 | 34 | 35 |
    // +----+----+----+
    // | 39 | 40 | 41 |
    // +----+----+----+
    BOOST_CHECK_CLOSE(result[6][5], 306 / 9.0, 1e-6);

    // North side. -------------------------------------------------------------
    // 0, 2
    // +----+----+----+----+----+
    // |  0 |  1 |  2 |  3 |  4 |
    // +----+----+----+----+----+
    // |  6 |  7 |  8 |  9 | 10 |
    // +----+----+----+----+----+
    // | 12 | 13 | 14 | 15 | 16 |
    // +----+----+----+----+----+
    BOOST_CHECK_CLOSE(result[0][2], 120 / 15.0, 1e-6);

    // West side. --------------------------------------------------------------
    // 4, 0
    // +----+----+----+
    // | 12 | 13 | 14 |
    // +----+----+----+
    // | 18 | 19 | 20 |
    // +----+----+----+
    // | 24 | 25 | 26 |
    // +----+----+----+
    // | 30 | 31 | 32 |
    // +----+----+----+
    // | 36 | 37 | 38 |
    // +----+----+----+
    BOOST_CHECK_CLOSE(result[4][0], 375 / 15.0, 1e-6);

    // East side. --------------------------------------------------------------
    // 2, 4
    // +----+----+----+----+
    // |  2 |  3 |  4 |  5 |
    // +----+----+----+----+
    // |  8 |  9 | 10 | 11 |
    // +----+----+----+----+
    // | 14 | 15 | 16 | 17 |
    // +----+----+----+----+
    // | 20 | 21 | 22 | 23 |
    // +----+----+----+----+
    // | 26 | 27 | 28 | 29 |
    // +----+----+----+----+
    BOOST_CHECK_CLOSE(result[2][4], 310 / 20.0, 1e-6);

    // South side.
    // 6, 3
    // +----+----+----+----+----+
    // | 25 | 26 | 27 | 28 | 29 |
    // +----+----+----+----+----+
    // | 31 | 32 | 33 | 34 | 35 |
    // +----+----+----+----+----+
    // | 37 | 38 | 39 | 40 | 41 |
    // +----+----+----+----+----+
    BOOST_CHECK_CLOSE(result[6][3], 495 / 15.0, 1e-6);

    // Inner part.
    // 3, 2
    // +----+----+----+----+----+
    // |  6 |  7 |  8 |  9 | 10 |
    // +----+----+----+----+----+
    // | 12 | 13 | 14 | 15 | 16 |
    // +----+----+----+----+----+
    // | 18 | 19 | 20 | 21 | 22 |
    // +----+----+----+----+----+
    // | 24 | 25 | 26 | 27 | 28 |
    // +----+----+----+----+----+
    // | 30 | 31 | 32 | 33 | 34 |
    // +----+----+----+----+----+
    BOOST_CHECK_CLOSE(result[3][2], 500 / 25.0, 1e-6);

    // Make sure all cells in the result have a new value.
    BOOST_CHECK_EQUAL(std::count(result.data(), result.data() +
        result.num_elements(), 0), 0);
}


void compare_result2(
    fern::Array<double, 2> const& result)
{
    // Upper left corner. ----------------------------------------------
    BOOST_CHECK_CLOSE(result[0][0], (63.0 + 24.5) / (9.0 + 7.0), 1e-6);
    BOOST_CHECK_CLOSE(result[0][1], (90.0 + 27.5) / (12.0 + 8.0), 1e-6);
    BOOST_CHECK_CLOSE(result[1][1], (168.0 + 45.5) / (16.0 + 9.0),
        1e-6);

    // Upper right corner. ---------------------------------------------
    BOOST_CHECK_CLOSE(result[0][4], (114.0 + 54.5) / (12.0 + 8.0),
        1e-6);

    // Lower left corner. ----------------------------------------------
    // Out of image values:
    //     (18 + 24 + 30) / 3 ->  72 / 3
    //     (24 + 30 + 36) / 3 ->  90 / 3
    //     (30 + 36     ) / 2 ->  66 / 2
    //     (36          ) / 1 ->  36 / 1
    //     (36 + 37     ) / 2 ->  73 / 2
    //     (36 + 37 + 38) / 3 -> 111 / 3
    //     (37 + 38 + 39) / 3 -> 114 / 3
    //     (38 + 39 + 40) / 3 -> 117 / 3
    BOOST_CHECK_CLOSE(result[6][1], (378.0 + 273.5) / (12.0 + 8.0),
        1e-6);

    // Lower right corner. ---------------------------------------------
    // Out of image values:
    //     (38 + 39 + 40) / 3 -> 117 / 3
    //     (39 + 40 + 41) / 3 -> 120 / 3
    //     (40 + 41     ) / 2 ->  81 / 2
    //     (41          ) / 1 ->  41 / 1
    //     (41 + 35     ) / 2 ->  76 / 2
    //     (29 + 35 + 41) / 3 -> 105 / 3
    //     (23 + 29 + 35) / 3 ->  87 / 3
    BOOST_CHECK_CLOSE(result[6][5], (306.0 + 262.5) / (9.0 + 7.0),
        1e-6);

    // North side. -----------------------------------------------------
    // Out of image values:
    // (0 + 1    ) / 2
    // (0 + 1 + 2) / 3
    // (1 + 2 + 3) / 3
    // (2 + 3 + 4) / 3
    // (3 + 4 + 5) / 3
    BOOST_CHECK_CLOSE(result[0][2], (120.0 + 10.5) / (15.0 + 5.0),
        1e-6);

    // West side. ------------------------------------------------------
    // Out of image values:
    // ( 6 + 12 + 18) / 3 -> 36 / 3
    // (12 + 18 + 24) / 3 -> 54 / 3
    // (18 + 24 + 30) / 3 -> 72 / 3
    // (24 + 30 + 36) / 3 -> 90 / 3
    // (30 + 36     ) / 2 -> 66 / 2
    BOOST_CHECK_CLOSE(result[4][0], (375.0 + 117.0) / (15.0 + 5.0),
        1e-6);

    // East side. ------------------------------------------------------
    // Out of image values:
    // ( 5 + 11     ) / 2
    // ( 5 + 11 + 17) / 3
    // (11 + 17 + 23) / 3
    // (17 + 23 + 29) / 3
    // (23 + 29 + 35) / 3
    BOOST_CHECK_CLOSE(result[2][4], (310.0 + 88.0) / (20.0 + 5.0),
        1e-6);

    // South side. -----------------------------------------------------
    // Out of image values:
    // (36 + 37 + 38) / 3
    // (37 + 38 + 39) / 3
    // (38 + 39 + 40) / 3
    // (39 + 40 + 41) / 3
    // (40 + 41     ) / 2
    BOOST_CHECK_CLOSE(result[6][3], (495.0 + 194.5) / (15.0 + 5.0),
        1e-6);

    // Inner part. -----------------------------------------------------
    // No out of image values.
    BOOST_CHECK_CLOSE(result[3][2], 500 / 25.0, 1e-6);

    // Make sure all cells in the result have a new value.
    BOOST_CHECK_EQUAL(std::count(result.data(), result.data() +
        result.num_elements(), 0), 0);
}


void compare_result3(
    fern::Array<double, 2> const& result)
{
    // +----+----+
    // |  0 |  1 |
    // +----+----+
    // |  6 |  7 |
    // +----+----+
    BOOST_CHECK_CLOSE(result[0][0], (0 + 1 + 6) / 3.0, 1e-6);

    // +----+----+
    // | 34 | 35 |
    // +----+----+
    // | 40 | 41 |
    // +----+----+
    BOOST_CHECK_CLOSE(result[6][5], (35 + 40 + 41) / 3.0, 1e-6);

    // +----+----+----+
    // |  0 |  1 |  2 |
    // +----+----+----+
    // |  6 |  7 |  8 |
    // +----+----+----+
    // | 12 | 13 | 14 |
    // +----+----+----+
    BOOST_CHECK_CLOSE(result[1][1], (1 + 6 + 7 + 8 + 13) / 5.0, 1e-6);

    // +----+----+
    // | 18 | 19 |
    // +----+----+
    // | 24 | 25 |
    // +----+----+
    // | 30 | 31 |
    // +----+----+
    BOOST_CHECK_CLOSE(result[4][0], (18 + 24 + 25 + 30) / 4.0, 1e-6);

    // +----+----+----+
    // |  1 |  2 |  3 |
    // +----+----+----+
    // |  7 |  8 |  9 |
    // +----+----+----+
    // | 13 | 14 | 15 |
    // +----+----+----+
    BOOST_CHECK_CLOSE(result[1][2], (2 + 7 + 8 + 9 + 14) / 5.0, 1e-6);
}


void compare_result4(
    fern::MaskedArray<double, 2> const& result)
{
    // +----+----+----+
    // | 10 |  7 |  6 |
    // +----+----+----+
    // |  2 | 18 | -1 |
    // +----+----+----+
    // | 12 | -6 |  8 |
    // +----+----+----+
    BOOST_CHECK(!result.mask()[0][0]);
    BOOST_CHECK_CLOSE(result[0][0], 6 + 4, 1e-6);

    BOOST_CHECK(!result.mask()[0][1]);
    BOOST_CHECK_CLOSE(result[0][1], 8 + -2 + 1, 1e-6);

    BOOST_CHECK(!result.mask()[0][2]);
    BOOST_CHECK_CLOSE(result[0][2], 6 + 0, 1e-6);

    BOOST_CHECK(!result.mask()[1][0]);
    BOOST_CHECK_CLOSE(result[1][0], 8 + 1 + -7, 1e-6);

    BOOST_CHECK(!result.mask()[1][1]);
    BOOST_CHECK_CLOSE(result[1][1], 6 + 4 + 0 + 8, 1e-6);

    BOOST_CHECK(!result.mask()[1][2]);
    BOOST_CHECK_CLOSE(result[1][2], -2 + 1, 1e-6);

    BOOST_CHECK(!result.mask()[2][0]);
    BOOST_CHECK_CLOSE(result[2][0], 4 + 8, 1e-6);

    BOOST_CHECK(!result.mask()[2][1]);
    BOOST_CHECK_CLOSE(result[2][1], 1 + -7, 1e-6);

    BOOST_CHECK(!result.mask()[2][2]);
    BOOST_CHECK_CLOSE(result[2][2], 0 + 8, 1e-6);
}


void compare_result5(
    fern::MaskedArray<double, 2> const& result)
{
    // +----+----+----+----+-----+-----+
    // |  7 |  7 |  1 |  4 |  15 |  15 |
    // +----+----+----+----+-----+-----+
    // | 19 | 20 |  7 | 10 |  15 |  15 |
    // +----+----+----+----+-----+-----+
    // | 37 | 38 | 13 |  X |  32 |  34 |
    // +----+----+----+----+-----+-----+
    // | 55 | 56 | 19 | 22 |  51 |  51 |
    // +----+----+----+----+-----+-----+
    // | 73 | 74 | 25 | 28 |  85 |  86 |
    // +----+----+----+----+-----+-----+
    // | 91 | 92 | 31 | 34 | 103 | 104 |
    // +----+----+----+----+-----+-----+
    // | 67 | 67 | 37 | 40 |  75 |  75 |
    // +----+----+----+----+-----+-----+

    size_t const nr_cols = 6;

    size_t row_id = 0;
    std::vector<double> values = { 7, 7, 1, 4, 15, 15 };
    std::vector<bool> no_data = { false, false, false, false, false, false };

    BOOST_REQUIRE_EQUAL_COLLECTIONS(no_data.begin(), no_data.end(),
        result.mask().data() + row_id * nr_cols,
        result.mask().data() + row_id * nr_cols + nr_cols);
    BOOST_CHECK_EQUAL_COLLECTIONS(values.begin(), values.end(),
        result.data() + row_id * nr_cols,
        result.data() + row_id * nr_cols + nr_cols);


    ++row_id;
    values = { 19, 20, 7, 10, 15, 15 };
    no_data = { false, false, false, false, false, false };

    BOOST_REQUIRE_EQUAL_COLLECTIONS(no_data.begin(), no_data.end(),
        result.mask().data() + row_id * nr_cols,
        result.mask().data() + row_id * nr_cols + nr_cols);
    BOOST_CHECK_EQUAL_COLLECTIONS(values.begin(), values.end(),
        result.data() + row_id * nr_cols,
        result.data() + row_id * nr_cols + nr_cols);


    ++row_id;
    no_data = { false, false, false, true, false, false };

    BOOST_REQUIRE_EQUAL_COLLECTIONS(no_data.begin(), no_data.end(),
        result.mask().data() + row_id * nr_cols,
        result.mask().data() + row_id * nr_cols + nr_cols);

    BOOST_CHECK_CLOSE(result[2][0], 37, 1e-6);
    BOOST_CHECK_CLOSE(result[2][1], 38, 1e-6);
    BOOST_CHECK_CLOSE(result[2][2], 13, 1e-6);
    BOOST_CHECK_CLOSE(result[2][4], 32, 1e-6);
    BOOST_CHECK_CLOSE(result[2][5], 34, 1e-6);


    ++row_id;
    values = { 55, 56, 19, 22, 51, 51 };
    no_data = { false, false, false, false, false, false };

    BOOST_REQUIRE_EQUAL_COLLECTIONS(no_data.begin(), no_data.end(),
        result.mask().data() + row_id * nr_cols,
        result.mask().data() + row_id * nr_cols + nr_cols);
    BOOST_CHECK_EQUAL_COLLECTIONS(values.begin(), values.end(),
        result.data() + row_id * nr_cols,
        result.data() + row_id * nr_cols + nr_cols);


    ++row_id;
    values = { 73, 74, 25, 28, 85, 86 };
    no_data = { false, false, false, false, false, false };

    BOOST_REQUIRE_EQUAL_COLLECTIONS(no_data.begin(), no_data.end(),
        result.mask().data() + row_id * nr_cols,
        result.mask().data() + row_id * nr_cols + nr_cols);
    BOOST_CHECK_EQUAL_COLLECTIONS(values.begin(), values.end(),
        result.data() + row_id * nr_cols,
        result.data() + row_id * nr_cols + nr_cols);


    ++row_id;
    values = { 91, 92, 31, 34, 103, 104 };
    no_data = { false, false, false, false, false, false };

    BOOST_REQUIRE_EQUAL_COLLECTIONS(no_data.begin(), no_data.end(),
        result.mask().data() + row_id * nr_cols,
        result.mask().data() + row_id * nr_cols + nr_cols);
    BOOST_CHECK_EQUAL_COLLECTIONS(values.begin(), values.end(),
        result.data() + row_id * nr_cols,
        result.data() + row_id * nr_cols + nr_cols);


    ++row_id;
    values = { 67, 67, 37, 40, 75, 75 };
    no_data = { false, false, false, false, false, false };

    BOOST_REQUIRE_EQUAL_COLLECTIONS(no_data.begin(), no_data.end(),
        result.mask().data() + row_id * nr_cols,
        result.mask().data() + row_id * nr_cols + nr_cols);
    BOOST_CHECK_EQUAL_COLLECTIONS(values.begin(), values.end(),
        result.data() + row_id * nr_cols,
        result.data() + row_id * nr_cols + nr_cols);
}


void compare_result6(
    fern::MaskedArray<double, 2> const& result)
{
    // +----+----ٍ+-----+-----+----+----+
    // |  X |  X |   8 |   9 |  X |  X |
    // +----+----ٍ+-----+-----+----+----+
    // |  X | 21 |  23 |  23 | 25 |  X |
    // +----+----ٍ+-----+-----+----+----+
    // | 31 | 33 |  56 |  60 | 37 | 39 |
    // +----+----ٍ+-----+-----+----+----+
    // | 19 | 76 |  80 |  84 | 88 | 22 |
    // +----+----ٍ+-----+-----+----+----+
    // | 43 | 45 | 104 | 108 | 49 | 51 |
    // +----+----ٍ+-----+-----+----+----+
    // |  X | 57 |  97 |  59 | 61 |  X |
    // +----+----ٍ+-----+-----+----+----+
    // |  X | 38 |  32 |  71 |  X |  X |
    // +----+----ٍ+-----+-----+----+----+

    size_t const nr_cols = 6;
    size_t row_id = 0;
    std::vector<double> values;
    std::vector<bool> no_data;

    no_data = { true, true, false, false, true, true };

    BOOST_REQUIRE_EQUAL_COLLECTIONS(no_data.begin(), no_data.end(),
        result.mask().data() + row_id * nr_cols,
        result.mask().data() + row_id * nr_cols + nr_cols);
    BOOST_CHECK_CLOSE(result[0][2], 8.0, 1e-6);
    BOOST_CHECK_CLOSE(result[0][3], 9.0, 1e-6);


    ++row_id;
    no_data = { true, false, false, false, false, true };

    BOOST_REQUIRE_EQUAL_COLLECTIONS(no_data.begin(), no_data.end(),
        result.mask().data() + row_id * nr_cols,
        result.mask().data() + row_id * nr_cols + nr_cols);
    BOOST_CHECK_CLOSE(result[1][1], 21.0, 1e-6);
    BOOST_CHECK_CLOSE(result[1][2], 23.0, 1e-6);
    BOOST_CHECK_CLOSE(result[1][3], 23.0, 1e-6);
    BOOST_CHECK_CLOSE(result[1][4], 25.0, 1e-6);


    ++row_id;
    no_data = { false, false, false, false, false, false };
    values = { 31, 33,  56,  60, 37, 39 };

    BOOST_REQUIRE_EQUAL_COLLECTIONS(no_data.begin(), no_data.end(),
        result.mask().data() + row_id * nr_cols,
        result.mask().data() + row_id * nr_cols + nr_cols);
    BOOST_CHECK_CLOSE(result[2][0], 31.0, 1e-6);
    BOOST_CHECK_CLOSE(result[2][1], 33.0, 1e-6);
    BOOST_CHECK_CLOSE(result[2][2], 56.0, 1e-6);
    BOOST_CHECK_CLOSE(result[2][3], 60.0, 1e-6);
    BOOST_CHECK_CLOSE(result[2][4], 37.0, 1e-6);
    BOOST_CHECK_CLOSE(result[2][5], 39.0, 1e-6);


    ++row_id;
    no_data = { false, false, false, false, false, false };
    values = { 19, 76,  80,  84, 88, 22 };

    BOOST_REQUIRE_EQUAL_COLLECTIONS(no_data.begin(), no_data.end(),
        result.mask().data() + row_id * nr_cols,
        result.mask().data() + row_id * nr_cols + nr_cols);
    BOOST_CHECK_CLOSE(result[3][0], 19.0, 1e-6);
    BOOST_CHECK_CLOSE(result[3][1], 76.0, 1e-6);
    BOOST_CHECK_CLOSE(result[3][2], 80.0, 1e-6);
    BOOST_CHECK_CLOSE(result[3][3], 84.0, 1e-6);
    BOOST_CHECK_CLOSE(result[3][4], 88.0, 1e-6);
    BOOST_CHECK_CLOSE(result[3][5], 22.0, 1e-6);


    ++row_id;
    no_data = { false, false, false, false, false, false };
    values = { 43, 45, 104, 108, 49, 51 };

    BOOST_REQUIRE_EQUAL_COLLECTIONS(no_data.begin(), no_data.end(),
        result.mask().data() + row_id * nr_cols,
        result.mask().data() + row_id * nr_cols + nr_cols);
    BOOST_CHECK_CLOSE(result[4][0], 43.0, 1e-6);
    BOOST_CHECK_CLOSE(result[4][1], 45.0, 1e-6);
    BOOST_CHECK_CLOSE(result[4][2], 104.0, 1e-6);
    BOOST_CHECK_CLOSE(result[4][3], 108.0, 1e-6);
    BOOST_CHECK_CLOSE(result[4][4], 49.0, 1e-6);
    BOOST_CHECK_CLOSE(result[4][5], 51.0, 1e-6);


    ++row_id;
    no_data = { true, false, false, false, false, true };

    BOOST_REQUIRE_EQUAL_COLLECTIONS(no_data.begin(), no_data.end(),
        result.mask().data() + row_id * nr_cols,
        result.mask().data() + row_id * nr_cols + nr_cols);
    BOOST_CHECK_CLOSE(result[5][1], 57.0, 1e-6);
    BOOST_CHECK_CLOSE(result[5][2], 97.0, 1e-6);
    BOOST_CHECK_CLOSE(result[5][3], 59.0, 1e-6);
    BOOST_CHECK_CLOSE(result[5][4], 61.0, 1e-6);


    ++row_id;
    no_data = { true, false, false, false, true, true };
    BOOST_REQUIRE_EQUAL_COLLECTIONS(no_data.begin(), no_data.end(),
        result.mask().data() + row_id * nr_cols,
        result.mask().data() + row_id * nr_cols + nr_cols);
    BOOST_CHECK_CLOSE(result[6][1], 38.0, 1e-6);
    BOOST_CHECK_CLOSE(result[6][2], 32.0, 1e-6);
    BOOST_CHECK_CLOSE(result[6][3], 71.0, 1e-6);
}


BOOST_AUTO_TEST_CASE(convolve)
{
    using Weights = std::initializer_list<std::initializer_list<int>>;

    fa::ParallelExecutionPolicy parallel;
    fa::SequentialExecutionPolicy sequential;

    // Kernel with radius 2.
    {
        // Create input array:
        // +----+----+----+----+----+----+
        // |  0 |  1 |  2 |  3 |  4 |  5 |
        // +----+----+----+----+----+----+
        // |  6 |  7 |  8 |  9 | 10 | 11 |
        // +----+----+----+----+----+----+
        // | 12 | 13 | 14 | 15 | 16 | 17 |
        // +----+----+----+----+----+----+
        // | 18 | 19 | 20 | 21 | 22 | 23 |
        // +----+----+----+----+----+----+
        // | 24 | 25 | 26 | 27 | 28 | 29 |
        // +----+----+----+----+----+----+
        // | 30 | 31 | 32 | 33 | 34 | 35 |
        // +----+----+----+----+----+----+
        // | 36 | 37 | 38 | 39 | 40 | 41 |
        // +----+----+----+----+----+----+
        size_t const nr_rows = 7;
        size_t const nr_cols = 6;
        auto extents = fern::extents[nr_rows][nr_cols];
        fern::Array<double, 2> argument(extents);
        std::iota(
            argument.data(), argument.data() + argument.num_elements(), 0);

        // Calculate local average.
        // Define kernel shape and weights.

        Weights weights{
            {1, 1, 1, 1, 1},
            {1, 1, 1, 1, 1},
            {1, 1, 1, 1, 1},
            {1, 1, 1, 1, 1},
            {1, 1, 1, 1, 1}
        };

        fern::Square<int, 2> compile_time_kernel(weights);
        fern::Kernel<int> runtime_kernel(2, weights);

        // Convolute while skipping out-of-image cells.
        {
            // Sequential, compile-time kernel
            {
                fern::Array<double, 2> result(extents);
                fa::convolution::convolve(
                    sequential, argument, compile_time_kernel, result);
                compare_result1(result);
            }

            // Sequential, runtime kernel
            {
                fern::Array<double, 2> result(extents);
                fa::convolution::convolve(
                    sequential, argument, runtime_kernel, result);
                compare_result1(result);
            }

            // Parallel, compile-time kernel
            {
                fern::Array<double, 2> result(extents);
                fa::convolution::convolve(
                    parallel, argument, compile_time_kernel, result);
                compare_result1(result);
            }

            // Parallel, runtime kernel
            {
                fern::Array<double, 2> result(extents);
                fa::convolution::convolve(
                    parallel, argument, runtime_kernel, result);
                compare_result1(result);
            }
        }

        // Convolute while calculating values for out-of-image cells.
        {
            using AlternativeForNoDataPolicy=fa::convolve::SkipNoData;
            using NormalizePolicy=fa::convolve::DivideByWeights;
            using OutOfImagePolicy=
                fa::convolve::ReplaceOutOfImageByFocalAverage;
            using NoDataFocusElementPolicy=fa::convolve::KeepNoDataFocusElement;
            using InputNoDataPolicy=fa::InputNoDataPolicies<fa::SkipNoData>;
            using OutputNoDataPolicy=fa::DontMarkNoData;

            OutputNoDataPolicy output_no_data_policy;

            // Sequential, compile-time kernel
            {
                fern::Array<double, 2> result(extents);
                fa::convolution::convolve<
                    AlternativeForNoDataPolicy,
                    NormalizePolicy,
                    OutOfImagePolicy,
                    NoDataFocusElementPolicy,
                    fa::unary::DiscardRangeErrors,
                    InputNoDataPolicy,
                    OutputNoDataPolicy>(
                        InputNoDataPolicy{{}}, output_no_data_policy,
                        sequential, argument, compile_time_kernel, result);
                compare_result2(result);
            }

            // Sequential, runtime kernel
            {
                fern::Array<double, 2> result(extents);
                fa::convolution::convolve<
                    AlternativeForNoDataPolicy,
                    NormalizePolicy,
                    OutOfImagePolicy,
                    NoDataFocusElementPolicy,
                    fa::unary::DiscardRangeErrors,
                    InputNoDataPolicy,
                    OutputNoDataPolicy>(
                        InputNoDataPolicy{{}}, output_no_data_policy,
                        sequential, argument, runtime_kernel, result);
                compare_result2(result);
            }

            // Parallel, compile-time kernel
            {
                fern::Array<double, 2> result(extents);
                fa::convolution::convolve<
                    AlternativeForNoDataPolicy,
                    NormalizePolicy,
                    OutOfImagePolicy,
                    NoDataFocusElementPolicy,
                    fa::unary::DiscardRangeErrors,
                    InputNoDataPolicy,
                    OutputNoDataPolicy>(
                        InputNoDataPolicy{{}}, output_no_data_policy,
                        parallel, argument, compile_time_kernel, result);
                compare_result2(result);
            }

            // Parallel, runtime kernel
            {
                fern::Array<double, 2> result(extents);
                fa::convolution::convolve<
                    AlternativeForNoDataPolicy,
                    NormalizePolicy,
                    OutOfImagePolicy,
                    NoDataFocusElementPolicy,
                    fa::unary::DiscardRangeErrors,
                    InputNoDataPolicy,
                    OutputNoDataPolicy>(
                        InputNoDataPolicy{{}}, output_no_data_policy,
                        parallel, argument, runtime_kernel, result);
                compare_result2(result);
            }
        }
    }

    // Kernel with radius 1.
    // This used to crash.
    {
        // Create input array:
        // +----+----+----+----+----+----+
        // |  0 |  1 |  2 |  3 |  4 |  5 |
        // +----+----+----+----+----+----+
        // |  6 |  7 |  8 |  9 | 10 | 11 |
        // +----+----+----+----+----+----+
        // | 12 | 13 | 14 | 15 | 16 | 17 |
        // +----+----+----+----+----+----+
        // | 18 | 19 | 20 | 21 | 22 | 23 |
        // +----+----+----+----+----+----+
        // | 24 | 25 | 26 | 27 | 28 | 29 |
        // +----+----+----+----+----+----+
        // | 30 | 31 | 32 | 33 | 34 | 35 |
        // +----+----+----+----+----+----+
        // | 36 | 37 | 38 | 39 | 40 | 41 |
        // +----+----+----+----+----+----+
        size_t const nr_rows = 7;
        size_t const nr_cols = 6;
        auto extents = fern::extents[nr_rows][nr_cols];
        fern::Array<double, 2> argument(extents);
        std::iota(argument.data(), argument.data() + argument.num_elements(),
            0);

        // Calculate local average.
        // Define kernel shape and weights.
        Weights weights{
            {1, 1, 1},
            {1, 1, 1},
            {1, 1, 1}
        };

        fern::Square<int, 1> compile_time_kernel(weights);
        fern::Kernel<int> runtime_kernel(1, weights);

        // Sequential, compile-time kernel
        {
            fern::Array<double, 2> result(extents);
            fa::convolution::convolve(
                sequential, argument, compile_time_kernel, result);
        }

        // Sequential, runtime kernel
        {
            fern::Array<double, 2> result(extents);
            fa::convolution::convolve(
                sequential, argument, runtime_kernel, result);
        }

        // Parallel, compile-time kernel
        {
            fern::Array<double, 2> result(extents);
            fa::convolution::convolve(
                parallel, argument, compile_time_kernel, result);
        }

        // Parallel, runtime kernel
        {
            fern::Array<double, 2> result(extents);
            fa::convolution::convolve(
                parallel, argument, runtime_kernel, result);
        }
    }
}


// TODO Test Small image with out of image policy with larger kernel.
// TODO Test Focal Sum, with kernel that doesn't weigh.


template<
    class Value,
    class Result>
using OutOfRangePolicy = fa::convolve::OutOfRangePolicy<Value, Result>;


BOOST_AUTO_TEST_CASE(out_of_range_policy)
{
    // Make sure that out of range can be detected and that no-data can be
    // written when it happens.

    {
        auto min_float32 = fern::min<fern::f32>();
        auto max_float32 = fern::max<fern::f32>();

        OutOfRangePolicy<fern::f32, fern::f32> policy;
        BOOST_CHECK(policy.within_range(5.0));
        BOOST_CHECK(policy.within_range(-5.0));
        BOOST_CHECK(policy.within_range(0.0));
        BOOST_CHECK(policy.within_range(min_float32));
        BOOST_CHECK(policy.within_range(max_float32));
        BOOST_CHECK(!policy.within_range(2 * max_float32));
    }
}


BOOST_AUTO_TEST_CASE(no_data_policies)
{
    using Weights = std::initializer_list<std::initializer_list<int>>;

    // Make sure that input no-data is detected and handled correctly.
    using InputNoDataPolicy = fa::InputNoDataPolicies<
        fa::DetectNoDataByValue<fern::Mask<2>>>;
    using OutputNoDataPolicy = fa::MarkNoDataByValue<fern::Mask<2>>;

    fa::SequentialExecutionPolicy sequential;

    size_t const nr_rows = 3;
    size_t const nr_cols = 3;
    auto extents = fern::extents[nr_rows][nr_cols];

    // Local average kernel.
    Weights weights_1{
        {1, 1, 1},
        {1, 1, 1},
        {1, 1, 1}
    };
    Weights weights_2{
        {2, 2, 2},
        {2, 2, 2},
        {2, 2, 2}
    };

    fern::Kernel<int> kernel_1(1, weights_1);
    fern::Kernel<int> kernel_2(1, weights_2);


    // Source image with a few masked values.
    {
        // +---+---+---+
        // | 0 | 1 | 2 |
        // +---+---+---+
        // | 3 | X | 5 |
        // +---+---+---+
        // | X | 7 | 8 |
        // +---+---+---+
        fern::MaskedArray<double, 2> source(extents);
        std::iota(source.data(), source.data() + source.num_elements(), 0);
        source.mask()[1][1] = true;
        source.mask()[2][0] = true;
        fern::MaskedArray<double, 2> destination(extents);

        // Skip no-data.
        {
            fern::MaskedArray<double, 2> destination(extents);
            destination.fill(999.9);
            OutputNoDataPolicy output_no_data_policy(destination.mask(), true);
            fa::convolution::convolve<
                fa::convolve::SkipNoData,
                fa::convolve::DivideByWeights,
                fa::convolve::SkipOutOfImage,
                fa::convolve::KeepNoDataFocusElement,
                fa::unary::DiscardRangeErrors>(
                    InputNoDataPolicy{{source.mask(), true}},
                    output_no_data_policy,
                    sequential,
                    source, kernel_1, destination);

            // Verify mask.
            uint64_t nr_masked_cells{0};
            fa::statistic::count(sequential, destination.mask(),
                true, nr_masked_cells);
            BOOST_CHECK_EQUAL(nr_masked_cells, 2u);
            BOOST_CHECK(destination.mask()[1][1]);
            BOOST_CHECK(destination.mask()[2][0]);

            // Verify values.
            fern::Array<double, 2> result_we_want({
                { 4.0/3.0, 11.0/5.0,  8.0/3.0},
                {11.0/4.0,    999.9, 23.0/5.0},
                {   999.9, 23.0/4.0, 20.0/3.0}
            });
            fern::Array<bool, 2> equal_cells(extents);
            fa::algebra::equal(sequential, destination, result_we_want,
                equal_cells);

            uint64_t nr_equal_cells{0};
            fa::statistic::count(sequential, equal_cells, true,
                nr_equal_cells);
            BOOST_CHECK_EQUAL(nr_equal_cells, 9u);
        }

        // TODO Remove this, replacing no-data is pre-processing.
        /// // Replace no-data by focal average.
        /// {
        ///     // Focal average of the no-data cells is
        ///     // - 26/7
        ///     // - 10/2
        ///     // +-----+--------+---+
        ///     // |  0  |     1  | 2 |
        ///     // +-----+--------+---+
        ///     // |  3  | (26/7) | 5 |
        ///     // +-----+--------+---+
        ///     // | (5) |     7  | 8 |
        ///     // +-----+--------+---+
        ///     fern::MaskedArray<double, 2> destination(extents);
        ///     destination.fill(999.9);
        ///     fa::convolution::convolve<
        ///         fa::convolve::DivideByWeights,
        ///         fa::convolve::SkipOutOfImage,
        ///         fern::unary::DiscardRangeErrors>(
        ///         fa::convolve::ReplaceNoDataByFocalAverage,
        ///             sequential,
        ///             InputNoDataPolicy{{source.mask(), true}},
        ///             OutputNoDataPolicy(destination.mask(), true),
        ///             source, kernel_1, destination);

        ///     // Verify mask.
        ///     size_t nr_masked_cells{0};
        ///     fa::statistic::count(destination.mask(), true, nr_masked_cells);
        ///     BOOST_CHECK_EQUAL(nr_masked_cells, 2);
        ///     BOOST_CHECK(destination.mask()[1][1]);
        ///     BOOST_CHECK(destination.mask()[2][0]);

        ///     // Verify values.
        ///     double const v1 = 26.0 / 7.0;
        ///     double const v2 = 10.0 / 2.0;
        ///     fern::Array<double, 2> result_we_want({
        ///         {       (4.0 + v1)/4.0,      (11.0 + v1)/6.0,   (8.0 + v1)/4.0},
        ///         { (11.0 + v1 + v2)/6.0,                999.9,  (23.0 + v1)/6.0},
        ///         {                999.9, (23.0 + v1 + v2)/6.0,  (20.0 + v1)/4.0}
        ///     });

        ///     fern::Array<bool, 2> equal_cells(extents);
        ///     fa::algebra::equal(destination, result_we_want, equal_cells);

        ///     size_t nr_equal_cells{0};
        ///     fa::statistic::count(equal_cells, true, nr_equal_cells);
        ///     BOOST_CHECK_EQUAL(nr_equal_cells, 9);
        /// }
    }


    // Source image with lots of masked values.
    {
        fern::MaskedArray<double, 2> source(extents);
        std::iota(source.data(), source.data() + source.num_elements(), 0);
        source.mask_all();
        fern::MaskedArray<double, 2> destination(extents);
        OutputNoDataPolicy output_no_data_policy(destination.mask(), true);

        fa::convolution::convolve<
            fa::convolve::SkipNoData,
            fa::convolve::DivideByWeights,
            fa::convolve::SkipOutOfImage,
            fa::convolve::KeepNoDataFocusElement,
            fa::unary::DiscardRangeErrors>(
                InputNoDataPolicy{{source.mask(), true}},
                output_no_data_policy,
                sequential, source, kernel_1, destination);

        uint64_t nr_masked_cells;
        fa::statistic::count(sequential, destination.mask(), true,
            nr_masked_cells);
        BOOST_CHECK_EQUAL(nr_masked_cells, nr_rows * nr_cols);
    }

    // Source image with very large values. Convolving these should result
    // in out-оf-range values. It must be possible to detect these and mark
    // them as no-data in the result.
    {
        fern::MaskedArray<double, 2> source(extents);
        std::fill(source.data(), source.data() + source.num_elements(),
            fern::max<double>());
        fern::MaskedArray<double, 2> destination(extents);
        OutputNoDataPolicy output_no_data_policy(destination.mask(), true);

        fa::convolution::convolve<
            fa::convolve::SkipNoData,
            fa::convolve::DivideByWeights,
            fa::convolve::SkipOutOfImage,
            fa::convolve::KeepNoDataFocusElement,
            fa::convolve::OutOfRangePolicy>(
                InputNoDataPolicy{{source.mask(), true}},
                output_no_data_policy,
                sequential, source, kernel_2, destination);

        uint64_t nr_masked_cells;
        fa::statistic::count(sequential, destination.mask(), true,
            nr_masked_cells);
        BOOST_CHECK_EQUAL(nr_masked_cells, nr_rows * nr_cols);
    }
}


BOOST_AUTO_TEST_CASE(boolean_kernel_weights)
{
    using Weights = std::initializer_list<std::initializer_list<bool>>;

    fa::ParallelExecutionPolicy parallel;
    fa::SequentialExecutionPolicy sequential;

    // Kernel with radius 1 and boolean weights.
    {
        // Create input array:
        // +----+----+----+----+----+----+
        // |  0 |  1 |  2 |  3 |  4 |  5 |
        // +----+----+----+----+----+----+
        // |  6 |  7 |  8 |  9 | 10 | 11 |
        // +----+----+----+----+----+----+
        // | 12 | 13 | 14 | 15 | 16 | 17 |
        // +----+----+----+----+----+----+
        // | 18 | 19 | 20 | 21 | 22 | 23 |
        // +----+----+----+----+----+----+
        // | 24 | 25 | 26 | 27 | 28 | 29 |
        // +----+----+----+----+----+----+
        // | 30 | 31 | 32 | 33 | 34 | 35 |
        // +----+----+----+----+----+----+
        // | 36 | 37 | 38 | 39 | 40 | 41 |
        // +----+----+----+----+----+----+
        size_t const nr_rows = 7;
        size_t const nr_cols = 6;
        auto extents = fern::extents[nr_rows][nr_cols];
        fern::Array<double, 2> argument(extents);
        std::iota(argument.data(), argument.data() + argument.num_elements(),
            0);

        // Define kernel shape and weights.
        Weights weights{
            {false, true, false},
            {true , true, true },
            {false, true, false}
        };
        fern::Square<bool, 1> compile_time_kernel{weights};
        fern::Kernel<bool> runtime_kernel{1, weights};

        // Sequential, compile-time kernel
        {
            fern::Array<double, 2> result(extents);
            fa::convolution::convolve(
                sequential, argument, compile_time_kernel, result);
            compare_result3(result);
        }

        // Sequential, runtime kernel
        {
            fern::Array<double, 2> result(extents);
            fa::convolution::convolve(
                sequential, argument, runtime_kernel, result);
            compare_result3(result);
        }

        // Parallel, compile-time kernel
        {
            fern::Array<double, 2> result(extents);
            fa::convolution::convolve(
                parallel, argument, compile_time_kernel, result);
            compare_result3(result);
        }

        // Parallel, runtime kernel
        {
            fern::Array<double, 2> result(extents);
            fa::convolution::convolve(
                parallel, argument, runtime_kernel, result);
            compare_result3(result);
        }
    }
}


BOOST_AUTO_TEST_CASE(no_data_focus_element_policy)
{
    using InputNoDataPolicy = fa::InputNoDataPolicies<
        fa::DetectNoDataByValue<fern::Mask<2>>>;
    using OutputNoDataPolicy = fa::MarkNoDataByValue<fern::Mask<2>>;

    fa::ParallelExecutionPolicy parallel;
    fa::SequentialExecutionPolicy sequential;

    // PCRaster window4total example
    {
        // Create input array:
        // +----+----+----+
        // |  8 |  6 | -2 |
        // +----+----+----+
        // |  4 |  1 |  0 |
        // +----+----+----+
        // | -7 |  8 |  X |
        // +----+----+----+
        size_t const nr_rows = 3;
        size_t const nr_cols = 3;
        auto extents = fern::extents[nr_rows][nr_cols];
        fern::MaskedArray<double, 2> argument(extents);

        argument[0][0] = 8;
        argument[0][1] = 6;
        argument[0][2] = -2;
        argument[1][0] = 4;
        argument[1][1] = 1;
        argument[1][2] = 0;
        argument[2][0] = -7;
        argument[2][1] = 8;
        argument.mask()[2][2] = true;

        // Define kernel shape and weights.
        // Similar to PCRaster's window4total algorithm.
        fern::Square<bool, 1> kernel({
            {false, true, false},
            {true , false, true },
            {false, true, false}
        });


        // Sequential.
        {
            fern::MaskedArray<double, 2> result(extents);
            OutputNoDataPolicy output_no_data_policy(result.mask(), true);

            fa::convolution::convolve<
                fa::convolve::SkipNoData,
                fa::convolve::DontDivideByWeights,
                fa::convolve::SkipOutOfImage,
                fa::convolve::ReplaceNoDataFocusElement,
                fa::unary::DiscardRangeErrors>(
                    InputNoDataPolicy{{argument.mask(), true}},
                    output_no_data_policy,
                    sequential,
                    argument, kernel, result);

            compare_result4(result);
        }

        // Parallel.
        {
            fern::MaskedArray<double, 2> result(extents);
            OutputNoDataPolicy output_no_data_policy(result.mask(), true);

            fa::convolution::convolve<
                fa::convolve::SkipNoData,
                fa::convolve::DontDivideByWeights,
                fa::convolve::SkipOutOfImage,
                fa::convolve::ReplaceNoDataFocusElement,
                fa::unary::DiscardRangeErrors>(
                    InputNoDataPolicy{{argument.mask(), true}},
                    output_no_data_policy,
                    parallel,
                    argument, kernel, result);

            compare_result4(result);
        }
    }

    {
        // Create input array:
        // +----+----+---+---+----+----+
        // |  0 |  1 | X | X |  4 |  5 |
        // +----+----+---+---+----+----+
        // |  6 |  7 | X | X | 10 | 11 |
        // +----+----+---+---+----+----+
        // | 12 | 13 | X | X |  X |  X |
        // +----+----+---+---+----+----+
        // | 18 | 19 | X | X | 22 | 23 |
        // +----+----+---+---+----+----+
        // | 24 | 25 | X | X | 28 | 29 |
        // +----+----+---+---+----+----+
        // | 30 | 31 | X | X | 34 | 35 |
        // +----+----+---+---+----+----+
        // | 36 | 37 | X | X | 40 | 41 |
        // +----+----+---+---+----+----+

        size_t const nr_rows = 7;
        size_t const nr_cols = 6;
        auto extents = fern::extents[nr_rows][nr_cols];
        fern::MaskedArray<double, 2> argument(extents);

        std::iota(argument.data(), argument.data() + argument.num_elements(),
            0);

        argument.mask()[0][2] = true;
        argument.mask()[0][3] = true;
        argument.mask()[1][2] = true;
        argument.mask()[1][3] = true;
        argument.mask()[2][2] = true;
        argument.mask()[2][3] = true;
        argument.mask()[2][4] = true;
        argument.mask()[2][5] = true;
        argument.mask()[3][2] = true;
        argument.mask()[3][3] = true;
        argument.mask()[4][2] = true;
        argument.mask()[4][3] = true;
        argument.mask()[5][2] = true;
        argument.mask()[5][3] = true;
        argument.mask()[6][2] = true;
        argument.mask()[6][3] = true;


        // Define kernel shape and weights.
        // Similar to PCRaster's window4total algorithm.
        fern::Square<bool, 1> kernel({
            {false, true, false},
            {true , false, true },
            {false, true, false}
        });


        // Sequential.
        {
            fern::MaskedArray<double, 2> result(extents);
            fa::convolution::convolve(sequential, argument, kernel,
                result);
            OutputNoDataPolicy output_no_data_policy(result.mask(), true);

            fa::convolution::convolve<
                fa::convolve::SkipNoData,
                fa::convolve::DontDivideByWeights,
                fa::convolve::SkipOutOfImage,
                fa::convolve::ReplaceNoDataFocusElement,
                fa::unary::DiscardRangeErrors>(
                    InputNoDataPolicy{{argument.mask(), true}},
                    output_no_data_policy,
                    sequential,
                    argument, kernel, result);

            compare_result5(result);
        }

        // Parallel.
        {
            fern::MaskedArray<double, 2> result(extents);
            fa::convolution::convolve(sequential, argument, kernel,
                result);
            OutputNoDataPolicy output_no_data_policy(result.mask(), true);

            fa::convolution::convolve<
                fa::convolve::SkipNoData,
                fa::convolve::DontDivideByWeights,
                fa::convolve::SkipOutOfImage,
                fa::convolve::ReplaceNoDataFocusElement,
                fa::unary::DiscardRangeErrors>(
                    InputNoDataPolicy{{argument.mask(), true}},
                    output_no_data_policy,
                    parallel,
                    argument, kernel, result);

            compare_result5(result);
        }
    }

    {
        // Create input array:
        // +----+----+----+----+----+----+
        // |  X |  X |  X |  X |  X |  X |
        // +----+----+----+----+----+----+
        // |  X |  X |  8 |  9 |  X |  X |
        // +----+----+----+----+----+----+
        // |  X | 13 | 14 | 15 | 16 |  X |
        // +----+----+----+----+----+----+
        // | 18 | 19 | 20 | 21 | 22 | 23 |
        // +----+----+----+----+----+----+
        // |  X | 25 | 26 | 27 | 28 |  X |
        // +----+----+----+----+----+----+
        // |  X |  X | 32 | 33 |  X |  X |
        // +----+----+----+----+----+----+
        // |  X |  X | 38 |  X |  X |  X |
        // +----+----+----+----+----+----+

        size_t const nr_rows = 7;
        size_t const nr_cols = 6;
        auto extents = fern::extents[nr_rows][nr_cols];
        fern::MaskedArray<double, 2> argument(extents);

        std::iota(argument.data(), argument.data() + argument.num_elements(),
            0);
        argument.mask() = {
            {  true,  true,  true,  true,  true,  true },
            {  true,  true, false, false,  true,  true },
            {  true, false, false, false, false,  true },
            { false, false, false, false, false, false },
            {  true, false, false, false, false,  true },
            {  true,  true, false, false,  true,  true },
            {  true,  true, false,  true,  true,  true }
        };


        // Define kernel shape and weights.
        // Similar to PCRaster's window4total algorithm.
        fern::Square<bool, 1> kernel({
            { false,  true, false },
            {  true, false,  true },
            { false,  true, false }
        });

        // Sequential.
        {
            fern::MaskedArray<double, 2> result(extents);
            // fa::convolution::convolve(sequential, argument, kernel,
            //     result);
            OutputNoDataPolicy output_no_data_policy(result.mask(), true);

            fa::convolution::convolve<
                fa::convolve::SkipNoData,
                fa::convolve::DontDivideByWeights,
                fa::convolve::SkipOutOfImage,
                fa::convolve::ReplaceNoDataFocusElement,
                fa::unary::DiscardRangeErrors>(
                    InputNoDataPolicy{{argument.mask(), true}},
                    output_no_data_policy,
                    sequential,
                    argument, kernel, result);

            compare_result6(result);
        }

        // Parallel.
        {
            fern::MaskedArray<double, 2> result(extents);
            OutputNoDataPolicy output_no_data_policy(result.mask(), true);

            fa::convolution::convolve<
                fa::convolve::SkipNoData,
                fa::convolve::DontDivideByWeights,
                fa::convolve::SkipOutOfImage,
                fa::convolve::ReplaceNoDataFocusElement,
                fa::unary::DiscardRangeErrors>(
                    InputNoDataPolicy{{argument.mask(), true}},
                    output_no_data_policy,
                    parallel,
                    argument, kernel, result);

            compare_result6(result);
        }
    }
}


template<
    typename T>
void compare_result_use_case1(
    fern::MaskedArray<T, 2> const& result)
{
    // +-----+-----+-----+
    // | 0.0 | 0.0 | 0.0 |
    // +-----+-----+-----+
    // | 0.0 | 0.0 | 0.0 |
    // +-----+-----+-----+
    // | 0.0 | 0.0 | 0.0 |
    // +-----+-----+-----+

    std::vector<bool> no_data = {
        false, false, false,
        false, false, false,
        false, false, false
    };

    BOOST_REQUIRE_EQUAL_COLLECTIONS(no_data.begin(), no_data.end(),
        result.mask().data(), result.mask().data() +
            result.mask().num_elements());


    std::vector<T> values = {
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0
    };

    BOOST_REQUIRE_EQUAL_COLLECTIONS(values.begin(), values.end(),
        result.data(), result.data() + result.num_elements());
}


template<
    typename T>
void test_use_case1(
    T const& value)
{
    using InputNoDataPolicy = fa::InputNoDataPolicies<
        fa::DetectNoDataByValue<fern::Mask<2>>>;
    using OutputNoDataPolicy = fa::MarkNoDataByValue<fern::Mask<2>>;

    fa::ParallelExecutionPolicy parallel;
    fa::SequentialExecutionPolicy sequential;

    // +-------+-------+-------+
    // | value | value | value |
    // +-------+-------+-------+
    // | value | value | value |
    // +-------+-------+-------+
    // | value | value | value |
    // +-------+-------+-------+

    size_t const nr_rows = 3;
    size_t const nr_cols = 3;
    auto extents = fern::extents[nr_rows][nr_cols];
    fern::MaskedArray<T, 2> argument(extents);

    std::fill(argument.data(), argument.data() + argument.num_elements(),
        value);

    fern::Square<T, 1> kernel({
        {1, 0, -1},
        {2, 0, -2},
        {1, 0, -1}
    });


    // Sequential.
    {
        fern::MaskedArray<T, 2> result(extents);
        OutputNoDataPolicy output_no_data_policy(result.mask(), true);

        fa::convolution::convolve<
            fa::convolve::ReplaceNoDataByFocalAverage,
            fa::convolve::DontDivideByWeights,
            fa::convolve::ReplaceOutOfImageByFocalAverage,
            fa::convolve::KeepNoDataFocusElement,
            fa::unary::DiscardRangeErrors>(
                InputNoDataPolicy{{argument.mask(), true}},
                output_no_data_policy,
                sequential,
                argument, kernel, result);

        compare_result_use_case1(result);
    }

    // Parallel.
    {
        fern::MaskedArray<T, 2> result(extents);
        OutputNoDataPolicy output_no_data_policy(result.mask(), true);

        fa::convolution::convolve<
            fa::convolve::ReplaceNoDataByFocalAverage,
            fa::convolve::DontDivideByWeights,
            fa::convolve::ReplaceOutOfImageByFocalAverage,
            fa::convolve::KeepNoDataFocusElement,
            fa::unary::DiscardRangeErrors>(
                InputNoDataPolicy{{argument.mask(), true}},
                output_no_data_policy,
                parallel,
                argument, kernel, result);

        compare_result_use_case1(result);
    }
}


BOOST_AUTO_TEST_CASE(use_case1)
{
    for(auto const& value: {0.0, 0.055, 0.0925, 0.1, 0.1225, 0.17, 1.0}) {
        test_use_case1<float>(value);
    }
}
