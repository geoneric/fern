// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern algorithm algebra vector laplacian
#include <boost/test/unit_test.hpp>
#include "fern/core/data_customization_point/scalar.h"
#include "fern/core/type_traits.h"
#include "fern/core/types.h"
#include "fern/feature/core/data_customization_point/masked_raster.h"
#include "fern/algorithm/core/argument_customization_point/masked_raster.h"
#include "fern/algorithm/core/mask_customization_point/array.h"
#include "fern/algorithm/core/argument_traits/masked_raster.h"
#include "fern/algorithm/algebra/vector/laplacian.h"


BOOST_AUTO_TEST_SUITE(laplacian)

namespace fa = fern::algorithm;


template<
    class Value,
    class Result>
using OutOfRangePolicy = fa::laplacian::OutOfRangePolicy<Value, Result>;


BOOST_AUTO_TEST_CASE(out_of_range_policy)
{
    {
        OutOfRangePolicy<fern::float32_t, fern::float32_t> policy;
        BOOST_CHECK(policy.within_range(123.456, 4.5));
        BOOST_CHECK(!policy.within_range(123.456,
            fern::nan<fern::float32_t>()));
        BOOST_CHECK(!policy.within_range(123.456,
            fern::infinity<fern::float32_t>()));
    }
}


template<
    class T>
using MaskedRaster = fern::MaskedRaster<T, 2>;


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

    MaskedRaster<double>::Transformation transformation{{west, cell_width,
        north, cell_height}};
    MaskedRaster<double> raster(extents, transformation);

    std::iota(raster.data(), raster.data() + raster.num_elements(), 0);

    // Calculate laplacian.
    MaskedRaster<double> result(extents, transformation);

    // Without masking input and output values.
    {
        fa::algebra::laplacian(fa::sequential, raster, result);

        // Verify the result.
        BOOST_CHECK_EQUAL(get(result, index(result, 0, 0)),
            (25.0 - (8.0 * 0.0)) / 6.0);
        BOOST_CHECK_EQUAL(get(result, index(result, 1, 1)),
            (100.0 - (20.0 * 5.0)) / 6.0);
    }

    using InputNoDataPolicy = fa::InputNoDataPolicies<
        fa::DetectNoDataByValue<fern::Mask<2>>>;
    using OutputNoDataPolicy = fa::MarkNoDataByValue<fern::Mask<2>>;

    // With masking input and output values.
    {
        result.fill(999.0);
        result.mask().fill(false);
        raster.mask()[1][1] = true;

        OutputNoDataPolicy output_no_data_policy(result.mask(), true);

        fa::algebra::laplacian<fa::laplacian::OutOfRangePolicy>(
            InputNoDataPolicy{{raster.mask(), true}},
            output_no_data_policy,
            fa::sequential,
            raster, result);

        // Verify the result.
        BOOST_CHECK_EQUAL(get(result.mask(), index(result.mask(), 0, 0)),
            false);
        BOOST_CHECK_EQUAL(get(result.mask(), index(result.mask(), 1, 1)),
            true);
        BOOST_CHECK_EQUAL(get(result, index(result, 0, 0)),
            (15.0 - (6.0 * 0.0)) / 6.0);
        BOOST_CHECK_EQUAL(get(result, index(result, 1, 1)), 999.0);
    }
}

BOOST_AUTO_TEST_SUITE_END()
