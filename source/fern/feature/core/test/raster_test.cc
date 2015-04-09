// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern feature raster
#include <numeric>
#include <boost/test/unit_test.hpp>
#include "fern/feature/core/raster.h"


BOOST_AUTO_TEST_SUITE(raster)

BOOST_AUTO_TEST_CASE(raster)
{
    size_t const nr_rows = 5;
    size_t const nr_cols = 4;
    auto extents = fern::extents[nr_rows][nr_cols];

    double const cell_width = 4.0;
    double const cell_height = 5.0;
    double const west = 0.0;
    double const north = 0.0;
    using Raster = fern::Raster<int, 2>;
    Raster::Transformation transformation{{west, cell_width, north,
        cell_height}};
    Raster raster(extents, transformation);
    std::iota(raster.data(), raster.data() + raster.num_elements(), 0);

    BOOST_CHECK_EQUAL(raster[0][0], 0);
    BOOST_CHECK_EQUAL(raster[1][0], static_cast<int>(nr_cols));

    BOOST_CHECK_EQUAL(raster.transformation()[0], west);
    BOOST_CHECK_EQUAL(raster.transformation()[1], cell_width);
    BOOST_CHECK_EQUAL(raster.transformation()[2], north);
    BOOST_CHECK_EQUAL(raster.transformation()[3], cell_height);
}

BOOST_AUTO_TEST_SUITE_END()
