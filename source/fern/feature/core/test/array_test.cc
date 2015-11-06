// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern feature array
#include <boost/test/unit_test.hpp>
#include "fern/feature/core/array.h"


BOOST_AUTO_TEST_CASE(value)
{
    size_t const nr_dimensions = 1;
    size_t const nr_cells = 1;
    fern::Array<int, nr_dimensions> array(fern::extents[nr_cells]);

    BOOST_CHECK_EQUAL(array.size(), nr_cells);

    array[0] = 3;

    BOOST_CHECK_EQUAL(array[0], 3);
}


BOOST_AUTO_TEST_CASE(array)
{
    size_t const nr_dimensions = 1;
    size_t const nr_cells = 3;

    // Using shortcut constructor.
    {
        fern::Array<int, nr_dimensions> array(nr_cells);

        BOOST_CHECK_EQUAL(array.size(), nr_cells);

        array[0] = 3;
        array[1] = 2;
        array[2] = 1;

        BOOST_CHECK_EQUAL(array[0], 3);
        BOOST_CHECK_EQUAL(array[1], 2);
        BOOST_CHECK_EQUAL(array[2], 1);
    }

    // Using extents.
    {
        fern::Array<int, nr_dimensions> array(fern::extents[nr_cells]);

        BOOST_CHECK_EQUAL(array.size(), nr_cells);

        array[0] = 3;
        array[1] = 2;
        array[2] = 1;

        BOOST_CHECK_EQUAL(array[0], 3);
        BOOST_CHECK_EQUAL(array[1], 2);
        BOOST_CHECK_EQUAL(array[2], 1);
    }
}


BOOST_AUTO_TEST_CASE(initializer_list)
{
    {
        fern::Array<int, 2> array({
            { 3, 4 }
        });
        BOOST_CHECK_EQUAL(array.shape()[0], 1u);
        BOOST_CHECK_EQUAL(array.shape()[1], 2u);
        BOOST_CHECK_EQUAL(array[0][0], 3);
        BOOST_CHECK_EQUAL(array[0][1], 4);
    }
}


BOOST_AUTO_TEST_CASE(grid)
{
    size_t const nr_dimensions = 2;
    size_t const nr_rows = 3;
    size_t const nr_cols = 2;

    // Using shortcut constructor.
    {
        fern::Array<int, nr_dimensions> array(nr_rows, nr_cols);

        BOOST_CHECK_EQUAL(array.size(), nr_rows);
        BOOST_CHECK_EQUAL(array[0].size(), nr_cols);

        array[0][0] = -2;
        array[0][1] = -1;
        array[1][0] = 0;
        array[1][1] = 999;
        array[2][0] = 1;
        array[2][1] = 2;

        BOOST_CHECK_EQUAL(array[0][0], -2);
        BOOST_CHECK_EQUAL(array[0][1], -1);
        BOOST_CHECK_EQUAL(array[1][0], 0);
        BOOST_CHECK_EQUAL(array[1][1], 999);
        BOOST_CHECK_EQUAL(array[2][0], 1);
        BOOST_CHECK_EQUAL(array[2][1], 2);
    }

    // Using extents.
    {
        fern::Array<int, nr_dimensions> array(
            fern::extents[nr_rows][nr_cols]);

        BOOST_CHECK_EQUAL(array.size(), nr_rows);
        BOOST_CHECK_EQUAL(array[0].size(), nr_cols);

        array[0][0] = -2;
        array[0][1] = -1;
        array[1][0] = 0;
        array[1][1] = 999;
        array[2][0] = 1;
        array[2][1] = 2;

        BOOST_CHECK_EQUAL(array[0][0], -2);
        BOOST_CHECK_EQUAL(array[0][1], -1);
        BOOST_CHECK_EQUAL(array[1][0], 0);
        BOOST_CHECK_EQUAL(array[1][1], 999);
        BOOST_CHECK_EQUAL(array[2][0], 1);
        BOOST_CHECK_EQUAL(array[2][1], 2);
    }
}
