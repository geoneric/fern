#define BOOST_TEST_MODULE fern algorithm algebra vector gradient
#include <boost/test/unit_test.hpp>
#include "fern/core/data_customization_point/constant.h"
#include "fern/feature/core/data_customization_point/array.h"
#include "fern/feature/core/masked_raster_traits.h"
#include "fern/algorithm/algebra/vector/gradient.h"
#include "fern/algorithm/core/if.h"
#include "fern/algorithm/core/test/test_utils.h"


namespace fa = fern::algorithm;


BOOST_AUTO_TEST_SUITE(gradient)

BOOST_AUTO_TEST_CASE(algorithm)
{
    // Create input raster:
    // +----+----+----+
    // |  X |  1 |  2 |
    // +----+----+----+
    // |  3 |  X |  5 |
    // +----+----+----+
    // |  6 |  7 |  X |
    // +----+----+----+
    // |  9 | 10 | 11 |
    // +----+----+----+
    using MaskedRaster = fern::MaskedRaster<double, 2>;

    size_t const nr_rows = 4;
    size_t const nr_cols = 3;
    auto extents = fern::extents[nr_rows][nr_cols];

    double const cell_width = 2.0;
    double const cell_height = 3.0;
    double const west = 0.0;
    double const north = 0.0;
    MaskedRaster::Transformation transformation{{west, cell_width, north,
        cell_height}};

    MaskedRaster raster(extents, transformation);

    std::iota(raster.data(), raster.data() + raster.num_elements(), 0);
    raster.mask()[0][0] = true;
    raster.mask()[1][1] = true;
    raster.mask()[2][2] = true;

    fa::ExecutionPolicy execution_policy = fa::sequential;

    // Calculate gradient_x.
    {
        MaskedRaster result_we_got(extents, transformation, -9.0);
        MaskedRaster result_we_want(extents, transformation);
        result_we_want[0][0] =  -9.0;
        result_we_want[0][1] = ( 2.0 - 1.0 ) / cell_width;
        result_we_want[0][2] = ( 2.0 - 1.0 ) / cell_width;
        result_we_want[1][0] =   0.0;
        result_we_want[1][1] =  -9.0;
        result_we_want[1][2] =   0.0;
        result_we_want[2][0] = ( 7.0 - 6.0 ) / cell_width;
        result_we_want[2][1] = ( 7.0 - 6.0 ) / cell_width;
        result_we_want[2][2] =  -9.0;
        result_we_want[3][0] = (10.0 - 9.0 ) / cell_width;
        result_we_want[3][1] = (11.0 - 9.0 ) / (2 * cell_width);
        result_we_want[3][2] = (11.0 - 10.0) / cell_width;
        result_we_want.mask()[0][0] = true;
        result_we_want.mask()[1][1] = true;
        result_we_want.mask()[2][2] = true;

        fa::core::if_(execution_policy, raster.mask(), true,
            result_we_got.mask());

        fa::InputNoDataPolicies<fa::DetectNoDataByValue<fern::Mask<2>>>
             input_no_data_policy{{raster.mask(), true}};
        fa::MarkNoDataByValue<fern::Mask<2>> output_no_data_policy(
            result_we_got.mask(), true);

        fa::algebra::gradient_x(input_no_data_policy, output_no_data_policy,
            execution_policy, raster, result_we_got);
        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));
    }

    // Calculate gradient_y.
    {
        MaskedRaster result_we_got(extents, transformation, -9.0);
        MaskedRaster result_we_want(extents, transformation);
        result_we_want[0][0] =  -9.0;
        result_we_want[0][1] =   0.0;
        result_we_want[0][2] = ( 5.0 - 2.0) / cell_height;
        result_we_want[1][0] = ( 6.0 - 3.0) / cell_height;
        result_we_want[1][1] =  -9.0;
        result_we_want[1][2] = ( 5.0 - 2.0) / cell_height;
        result_we_want[2][0] = ( 9.0 - 3.0) / (2 * cell_height);
        result_we_want[2][1] = (10.0 - 7.0) / cell_height;
        result_we_want[2][2] =  -9.0;
        result_we_want[3][0] = ( 9.0 - 6.0) / cell_height;
        result_we_want[3][1] = (10.0 - 7.0) / cell_height;
        result_we_want[3][2] =   0.0;
        result_we_want.mask()[0][0] = true;
        result_we_want.mask()[1][1] = true;
        result_we_want.mask()[2][2] = true;

        fa::core::if_(execution_policy, raster.mask(), true,
            result_we_got.mask());

        fa::InputNoDataPolicies<fa::DetectNoDataByValue<fern::Mask<2>>>
            input_no_data_policy{{raster.mask(), true}};
        fa::MarkNoDataByValue<fern::Mask<2>> output_no_data_policy(
            result_we_got.mask(), true);

        fa::algebra::gradient_y(input_no_data_policy, output_no_data_policy,
            execution_policy, raster, result_we_got);
        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));
    }
}

BOOST_AUTO_TEST_SUITE_END()
