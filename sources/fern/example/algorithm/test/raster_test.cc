#define BOOST_TEST_MODULE fern example algorithm raster
#include <algorithm>
#include <boost/test/unit_test.hpp>
#include "fern/example/algorithm/raster.h"


BOOST_AUTO_TEST_SUITE(raster)

BOOST_AUTO_TEST_CASE(constructor)
{
    size_t const cell_size = 5.0;
    size_t const nr_rows = 600;
    size_t const nr_cols = 400;
    example::Raster<int32_t> raster1(cell_size, nr_rows, nr_cols);
    std::iota(raster1.values().begin(), raster1.values().end(), 0);
}


BOOST_AUTO_TEST_SUITE_END()
