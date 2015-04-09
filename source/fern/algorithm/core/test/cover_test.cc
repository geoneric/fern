// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern algorithm core cover
#include <boost/test/unit_test.hpp>
#include "fern/core/data_customization_point/scalar.h"
#include "fern/feature/core/data_customization_point/array.h"
#include "fern/feature/core/data_customization_point/masked_array.h"
#include "fern/algorithm/algebra/boole/and.h"
#include "fern/algorithm/core/cover.h"
#include "fern/algorithm/core/test/test_utils.h"


namespace fa = fern::algorithm;


BOOST_AUTO_TEST_SUITE(cover)

template<
    class Value1,
    class Value2,
    class Result>
void verify_value(
    Value1 const& value1,
    Value2 const& value2,
    Result const& result_we_want)
{
    Result result_we_get;
    fa::core::cover<>(fa::sequential, value1, value2, result_we_get);
    BOOST_CHECK_EQUAL(result_we_get, result_we_want);
}


BOOST_AUTO_TEST_CASE(algorithm)
{
    {
        verify_value<>( 4, 5, 4);
        verify_value<>( 5, 4, 5);
    }
}


BOOST_AUTO_TEST_CASE(array_2d_masked_0d)
{
    size_t const nr_rows{3};
    size_t const nr_cols{3};
    fern::MaskedArray<int, 2> value1(fern::extents[nr_rows][nr_cols]);
    int const value2{3};
    fern::MaskedArray<int, 2> result_we_got(fern::extents[nr_rows][nr_cols]);

    fa::InputNoDataPolicies<
        fa::DetectNoDataByValue<fern::Mask<2>>,
        fa::SkipNoData> input_no_data_policy{{value1.mask(), true}, {}};

    fa::MarkNoDataByValue<fern::Mask<2>> output_no_data_policy(
        result_we_got.mask(), true);


    value1[1][2] = 4;
    value1[2][1] = 5;
    value1[2][2] = -1;
    value1.mask().fill(true);
    value1.mask()[1][2] = false;
    value1.mask()[2][1] = false;
    value1.mask()[2][2] = false;

    fern::MaskedArray<int, 2> result_we_want(fern::extents[nr_rows][nr_cols]);
    result_we_want[0][0] = value2;
    result_we_want[0][1] = value2;
    result_we_want[0][2] = value2;
    result_we_want[1][0] = value2;
    result_we_want[1][1] = value2;
    result_we_want[1][2] = value1[1][2];
    result_we_want[2][0] = value2;
    result_we_want[2][1] = value1[2][1];
    result_we_want[2][2] = value1[2][2];

    auto& execution_policy(fa::sequential);

    fa::core::cover(input_no_data_policy, output_no_data_policy,
        execution_policy, value1, value2, result_we_got);

    BOOST_CHECK(fern::compare(execution_policy, result_we_got,
        result_we_want));
}


BOOST_AUTO_TEST_CASE(array_2d_masked_2d)
{
    size_t const nr_rows{3};
    size_t const nr_cols{3};
    fern::MaskedArray<int, 2> value1(fern::extents[nr_rows][nr_cols]);
    fern::MaskedArray<int, 2> value2(fern::extents[nr_rows][nr_cols]);
    fern::MaskedArray<int, 2> value3(fern::extents[nr_rows][nr_cols]);
    fern::MaskedArray<int, 2> result_we_got1(fern::extents[nr_rows][nr_cols]);
    fern::MaskedArray<int, 2> result_we_got2(fern::extents[nr_rows][nr_cols]);

    // +---+---+---+
    // | X | X | X |
    // +---+---+---+
    // | X | X | 4 |
    // +---+---+---+
    // | X | X | 1 |
    // +---+---+---+
    value1[1][2] = 4;
    value1[2][2] = 1;
    value1.mask().fill(true);
    value1.mask()[1][2] = false;
    value1.mask()[2][2] = false;

    // +---+---+---+
    // | 0 | X | X |
    // +---+---+---+
    // | X | X | 8 |
    // +---+---+---+
    // | X | 2 | X |
    // +---+---+---+
    value2[0][0] = 0;
    value2[1][2] = 8;
    value2[2][1] = 2;
    value2.mask().fill(true);
    value2.mask()[0][0] = false;
    value2.mask()[1][2] = false;
    value2.mask()[2][1] = false;

    // +---+---+---+
    // | 4 | 4 | 4 |
    // +---+---+---+
    // | 4 | 4 | 4 |
    // +---+---+---+
    // | X | 4 | 4 |
    // +---+---+---+
    value3.fill(4);
    value3[2][0] = -9;
    value3.mask().fill(false);
    value3.mask()[2][0] = true;

    // +---+---+---+
    // | 0 | 4 | 4 |
    // +---+---+---+
    // | 4 | 4 | 8 |
    // +---+---+---+
    // | X | 2 | 1 |
    // +---+---+---+
    fern::MaskedArray<int, 2> result_we_want(fern::extents[nr_rows][nr_cols]);
    result_we_want[0][0] = value2[0][0];
    result_we_want[0][1] = value3[0][1];
    result_we_want[0][2] = value3[0][2];
    result_we_want[1][0] = value3[1][0];
    result_we_want[1][1] = value3[1][1];
    result_we_want[1][2] = value1[1][2];
    result_we_want[2][0] = 0;
    result_we_want[2][1] = value2[2][1];
    result_we_want[2][2] = value1[2][2];
    result_we_want.mask().fill(false);
    result_we_want.mask()[2][0] = true;


    auto& execution_policy(fa::sequential);

    fa::MarkNoDataByValue<fern::Mask<2>> output_no_data_policy(
        result_we_got1.mask(), true);

    // result1 = cover(value1, value2)
    {
        fa::InputNoDataPolicies<
            fa::DetectNoDataByValue<fern::Mask<2>>,
            fa::DetectNoDataByValue<fern::Mask<2>>> input_no_data_policy{
                {value1.mask(), true}, {value2.mask(), true}};

        fa::algebra::and_(execution_policy, value1.mask(), value2.mask(),
            result_we_got1.mask());

        fa::core::cover(input_no_data_policy, output_no_data_policy,
            execution_policy, value1, value2, result_we_got1);

    }

    // result2 = cover(result1, value3)
    {
        fa::InputNoDataPolicies<
            fa::DetectNoDataByValue<fern::Mask<2>>,
            fa::DetectNoDataByValue<fern::Mask<2>>> input_no_data_policy{
                {result_we_got1.mask(), true}, {value3.mask(), true}};

        fa::algebra::and_(execution_policy, result_we_got1.mask(),
            value3.mask(), result_we_got2.mask());

        fa::core::cover(input_no_data_policy, output_no_data_policy,
            execution_policy, result_we_got1, value3, result_we_got2);
    }

    BOOST_CHECK(fern::compare(execution_policy, result_we_got2,
        result_we_want));
}

BOOST_AUTO_TEST_SUITE_END()
