#define BOOST_TEST_MODULE fern feature masked_raster
#include <numeric>
#include <boost/test/unit_test.hpp>
#include "fern/feature/core/masked_raster.h"


BOOST_AUTO_TEST_SUITE(masked_raster)

BOOST_AUTO_TEST_CASE(raster)
{
    size_t const nr_rows = 5;
    size_t const nr_cols = 4;
    auto extents = fern::extents[nr_rows][nr_cols];

    double const cell_size = 1.0;
    double const cell_width = cell_size;
    double const cell_height = cell_size;
    double const west = 0.0;
    double const north = 0.0;
    using MaskedRaster = fern::MaskedRaster<int, 2>;
    MaskedRaster::Transformation transformation{{west, cell_width, north,
        cell_height}};
    MaskedRaster raster(extents, transformation);
    std::iota(raster.data(), raster.data() + raster.num_elements(), 0);

    BOOST_CHECK_EQUAL(raster[0][0], 0);
    BOOST_CHECK_EQUAL(raster[1][0], nr_cols);
}

BOOST_AUTO_TEST_SUITE_END()
