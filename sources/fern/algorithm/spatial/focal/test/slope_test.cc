#define BOOST_TEST_MODULE fern algorithm spatial focal slope
#include <boost/test/unit_test.hpp>
#include "fern/core/constant_traits.h"
#include "fern/core/type_traits.h"
#include "fern/core/types.h"
#include "fern/feature/core/masked_array_traits.h"
#include "fern/feature/core/masked_raster_traits.h"
#include "fern/algorithm/spatial/focal/slope.h"


namespace fa = fern::algorithm;


BOOST_AUTO_TEST_SUITE(slope)

template<
    class Value,
    class Result>
using OutOfRangePolicy = fa::slope::OutOfRangePolicy<Value, Result>;


BOOST_AUTO_TEST_CASE(out_of_range_policy)
{
    {
        OutOfRangePolicy<fern::float32_t, fern::float32_t> policy;
        BOOST_CHECK(policy.within_range(4.5));
        BOOST_CHECK(!policy.within_range(fern::nan<fern::float32_t>()));
        BOOST_CHECK(!policy.within_range(fern::infinity<fern::float32_t>()));
    }
}


template<
    class T>
using MaskedRaster = fern::MaskedRaster<T, 2>;


BOOST_AUTO_TEST_CASE(algorithm)
{
    // Create input raster (from the PCRaster manual):
    // +-----+-----+-----+-----+-----+
    // |  70 |  70 |  80 |  X  | 120 |
    // +-----+-----+-----+-----+-----+
    // |  70 |  70 |  90 |  X  |  X  |
    // +-----+-----+-----+-----+-----+
    // |  70 |  70 | 100 | 140 | 280 |
    // +-----+-----+-----+-----+-----+
    // | 180 | 160 | 110 | 160 | 320 |
    // +-----+-----+-----+-----+-----+
    // | 510 | 440 | 300 | 400 | 480 |
    // +-----+-----+-----+-----+-----+
    size_t const nr_rows = 5;
    size_t const nr_cols = 5;
    auto extents = fern::extents[nr_rows][nr_cols];

    double const cell_width = 50.0;
    double const cell_height = 50.0;
    double const west = 0.0;
    double const north = 0.0;

    MaskedRaster<double>::Transformation transformation{{west, cell_width,
        north, cell_height}};
    MaskedRaster<double> raster(extents, transformation);

    raster[0][0] = 70.0;
    raster[0][1] = 70.0;
    raster[0][2] = 80.0;
    raster.mask()[0][3] = true;
    raster[0][4] = 120.0;
    raster[1][0] = 70.0;
    raster[1][1] = 70.0;
    raster[1][2] = 90.0;
    raster.mask()[1][3] = true;
    raster.mask()[1][4] = true;
    raster[2][0] = 70.0;
    raster[2][1] = 70.0;
    raster[2][2] = 100.0;
    raster[2][3] = 140.0;
    raster[2][4] = 280.0;
    raster[3][0] = 180.0;
    raster[3][1] = 160.0;
    raster[3][2] = 110.0;
    raster[3][3] = 160.0;
    raster[3][4] = 320.0;
    raster[4][0] = 510.0;
    raster[4][1] = 440.0;
    raster[4][2] = 300.0;
    raster[4][3] = 400.0;
    raster[4][4] = 480.0;


    // Create output raster (from PCRaster manual).
    MaskedRaster<double> result_we_want(extents, transformation);

    result_we_want[0][0] = 0.0118;
    result_we_want[0][1] = 0.114;
    result_we_want[0][2] = 0.394;
    result_we_want.mask()[0][3] = true;
    result_we_want[0][4] = 0.673;
    result_we_want[1][0] = 0.13;
    result_we_want[1][1] = 0.206;
    result_we_want[1][2] = 0.604;
    result_we_want.mask()[1][3] = true;
    result_we_want.mask()[1][4] = true;
    result_we_want[2][0] = 1.3;
    result_we_want[2][1] = 0.775;
    result_we_want[2][2] = 0.643;
    result_we_want[2][3] = 1.73;
    result_we_want[2][4] = 1.87;
    result_we_want[3][0] = 3.73;
    result_we_want[3][1] = 3.54;
    result_we_want[3][2] = 2.58;
    result_we_want[3][3] = 3.02;
    result_we_want[3][4] = 2.36;
    result_we_want[4][0] = 2.76;
    result_we_want[4][1] = 3.07;
    result_we_want[4][2] = 2.59;
    result_we_want[4][3] = 2.66;
    result_we_want[4][4] = 1.65;


    // Calculate slope.
    MaskedRaster<double> result_we_get(extents, transformation);

    using InputNoDataPolicy = fa::DetectNoDataByValue<fern::Mask<2>>;
    using OutputNoDataPolicy = fa::MarkNoDataByValue<fern::Mask<2>>;

    OutputNoDataPolicy output_no_data_policy(result_we_get.mask(), true);

    fa::spatial::slope<fa::unary::DiscardRangeErrors>(
        InputNoDataPolicy(raster.mask(), true),
        output_no_data_policy,
        fa::sequential,
        raster, result_we_get);

    for(size_t r = 0; r < nr_rows; ++r) {
        for(size_t c = 0; c < nr_cols; ++c) {
            if((r == 0 && c == 3) || (r == 1 && c == 3) || (r == 1 && c == 4)) {
                BOOST_CHECK(result_we_get.mask()[r][c]);
            }
            else {
                BOOST_CHECK_CLOSE(result_we_get[r][c], result_we_want[r][c],
                    1e-0);
            }
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
