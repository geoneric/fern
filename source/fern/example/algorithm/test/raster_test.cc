// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern example algorithm raster
// #include <numeric>
#include <boost/test/unit_test.hpp>
#include "fern/example/algorithm/raster.h"


BOOST_AUTO_TEST_CASE(constructor)
{
    double const cell_size = 5.0;
    size_t const nr_rows = 600;
    size_t const nr_cols = 400;
    example::Raster<int32_t> raster1(cell_size, nr_rows, nr_cols);
    // std::iota(raster1.values().begin(), raster1.values().end(), 0);
}
