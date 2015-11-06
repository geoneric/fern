// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern feature masked_array_value
#include <boost/test/unit_test.hpp>
#include "fern/language/feature/core/masked_array_value.h"


namespace fl = fern::language;


BOOST_AUTO_TEST_CASE(array)
{
    size_t const nr_dimensions = 1;
    size_t const nr_cells = 3;
    fl::MaskedArrayValue<int, nr_dimensions> array(fern::extents[nr_cells]);

    array[0] = 3;
    array[1] = 2;
    array[2] = 1;

    BOOST_CHECK_EQUAL(array.size(), nr_cells);
    BOOST_CHECK(!array.mask()[0]);
    BOOST_CHECK(!array.mask()[1]);
    BOOST_CHECK(!array.mask()[2]);
    BOOST_CHECK_EQUAL(array[0], 3);
    BOOST_CHECK_EQUAL(array[1], 2);
    BOOST_CHECK_EQUAL(array[2], 1);
}


BOOST_AUTO_TEST_CASE(grid)
{
    size_t const nr_dimensions = 2;
    size_t const nr_rows = 3;
    size_t const nr_cols = 2;
    fl::MaskedArrayValue<int, nr_dimensions> array(
        fern::extents[nr_rows][nr_cols]);

    array[0][0] = -2;
    array[0][1] = -1;
    array[1][0] = 0;
    array[1][1] = 999;
    array[2][0] = 1;
    array[2][1] = 2;

    BOOST_CHECK_EQUAL(array.size(), nr_rows);
    BOOST_CHECK_EQUAL(array[0].size(), nr_cols);
    BOOST_CHECK(!array.mask()[0][0]);
    BOOST_CHECK(!array.mask()[0][1]);
    BOOST_CHECK(!array.mask()[1][0]);
    BOOST_CHECK(!array.mask()[1][1]);
    BOOST_CHECK(!array.mask()[2][0]);
    BOOST_CHECK(!array.mask()[2][1]);
    BOOST_CHECK_EQUAL(array[0][0], -2);
    BOOST_CHECK_EQUAL(array[0][1], -1);
    BOOST_CHECK_EQUAL(array[1][0], 0);
    BOOST_CHECK_EQUAL(array[1][1], 999);
    BOOST_CHECK_EQUAL(array[2][0], 1);
    BOOST_CHECK_EQUAL(array[2][1], 2);
}
