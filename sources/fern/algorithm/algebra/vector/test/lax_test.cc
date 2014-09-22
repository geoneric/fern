#define BOOST_TEST_MODULE fern algorithm algebra vector lax
#include <boost/test/unit_test.hpp>
#include "fern/feature/core/array_traits.h"
#include "fern/feature/core/masked_raster_traits.h"
#include "fern/algorithm/algebra/vector/lax.h"


BOOST_AUTO_TEST_SUITE(lax)

BOOST_AUTO_TEST_CASE(algorithm)
{
    // Create input raster:
    // +----+----+----+----+
    // |  0 |  1 |  2 |  3 |
    // +----+----+----+----+
    // |  4 |  5 |  6 |  7 |
    // +----+----+----+----+
    // |  8 |  9 | 10 | 11 |
    // +----+----+----+----+
    // | 12 | 13 | 14 | 15 |
    // +----+----+----+----+
    // | 16 | 17 | 18 | 19 |
    // +----+----+----+----+
    size_t const nr_rows = 5;
    size_t const nr_cols = 4;
    auto extents = fern::extents[nr_rows][nr_cols];

    double const cell_width = 2.0;
    double const cell_height = 3.0;
    double const west = 0.0;
    double const north = 0.0;

    using MaskedRaster = fern::MaskedRaster<double, 2>;

    MaskedRaster::Transformation transformation{{west, cell_width,
        north, cell_height}};
    MaskedRaster raster(extents, transformation);

    std::iota(raster.data(), raster.data() + raster.num_elements(), 0);

    double const fraction = 0.6;

    // Calculate lax.
    MaskedRaster result(extents, transformation);

    // Without masking input and output values.
    {
        fern::algebra::lax(fern::sequential, raster, fraction, result);

        /// // Verify the result.
        BOOST_CHECK_EQUAL(fern::get(result, 0, 0),
            ((1.0 - fraction) * 0.0) + (fraction * 25.0 / 8.0));
        BOOST_CHECK_EQUAL(fern::get(result, 1, 1),
            ((1.0 - fraction) * 5.0) + (fraction * 100.0 / 20.0));
    }

    using InputNoDataPolicy = fern::DetectNoDataByValue<fern::Mask<2>>;
    using OutputNoDataPolicy = fern::MarkNoDataByValue<fern::Mask<2>>;

    // With masking input and output values.
    {
        result.fill(999.0);
        result.mask().fill(false);
        result.mask()[1][1] = true;

        InputNoDataPolicy input_no_data_policy(result.mask(), true);
        OutputNoDataPolicy output_no_data_policy(result.mask(), true);

        fern::algebra::lax(
            input_no_data_policy,
            output_no_data_policy,
            fern::sequential,
            raster, fraction, result);

        // Verify the result.
        BOOST_CHECK_EQUAL(fern::get(result.mask(), 0, 0), false);
        BOOST_CHECK_EQUAL(fern::get(result, 0, 0),
            ((1.0 - fraction) * 0.0) + (fraction * 15.0 / 6.0));

        BOOST_CHECK_EQUAL(fern::get(result.mask(), 1, 1), true);
        BOOST_CHECK_EQUAL(fern::get(result, 1, 1), 999.0);
    }
}

BOOST_AUTO_TEST_SUITE_END()
