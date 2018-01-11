// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern algorithm core clamp
#include <boost/test/unit_test.hpp>
#include "fern/core/data_customization_point/scalar.h"
#include "fern/feature/core/data_customization_point/array.h"
#include "fern/feature/core/data_customization_point/masked_array.h"
#include "fern/algorithm/core/clamp.h"
#include "fern/algorithm/core/test/test_utils.h"


namespace fa = fern::algorithm;


BOOST_AUTO_TEST_CASE(overload)
{
    fa::SequentialExecutionPolicy sequential;

    int array_0d{};
    fern::Array<int, 1> array_1d;
    fern::Array<int, 2> array_2d;

    fa::core::clamp<>(sequential, array_0d, array_0d, array_0d, array_0d);

    fa::core::clamp<>(sequential, array_1d, array_1d, array_1d, array_1d);
    fa::core::clamp<>(sequential, array_1d, array_0d, array_1d, array_1d);
    fa::core::clamp<>(sequential, array_1d, array_1d, array_0d, array_1d);

    fa::core::clamp<>(sequential, array_2d, array_2d, array_2d, array_2d);
    fa::core::clamp<>(sequential, array_2d, array_0d, array_2d, array_2d);
    fa::core::clamp<>(sequential, array_2d, array_2d, array_0d, array_2d);
}


template<
    class ExecutionPolicy,
    class Value,
    class LowerBound,
    class UpperBound,
    class Result>
void verify_value(
    ExecutionPolicy& execution_policy,
    Value const& value,
    LowerBound const& lower_bound,
    UpperBound const& upper_bound,
    Result const& result_we_want)
{
    Result result_we_got;
    fa::core::clamp<>(execution_policy, value, lower_bound, upper_bound,
        result_we_got);
    BOOST_CHECK_EQUAL(result_we_got, result_we_want);
}


template<
    typename ExecutionPolicy>
void test_0d_0d_0d(
    ExecutionPolicy& execution_policy)
{
    {
        int lower_bound = 100;
        int upper_bound = 200;

        verify_value<>(execution_policy,  99, lower_bound, upper_bound,
            lower_bound);
        verify_value<>(execution_policy, 100, lower_bound, upper_bound,
            lower_bound);
        verify_value<>(execution_policy, 101, lower_bound, upper_bound, 101);
        verify_value<>(execution_policy, 199, lower_bound, upper_bound, 199);
        verify_value<>(execution_policy, 200, lower_bound, upper_bound,
            upper_bound);
        verify_value<>(execution_policy, 201, lower_bound, upper_bound,
            upper_bound);
    }
}


template<
    typename ExecutionPolicy>
void test_1d_1d_1d(
    ExecutionPolicy& execution_policy)
{
    fern::Array<int, 1> value         {10, 20, 30, 40, 50, 60};
    fern::Array<int, 1> lower_bound   { 9, 20, 31,  0,  0,  0};
    fern::Array<int, 1> upper_bound   {90, 90, 90, 39, 50, 61};
    fern::Array<int, 1> result_we_want{10, 20, 31, 39, 50, 60};
    fern::Array<int, 1> result_we_got(6);

    fa::core::clamp<>(execution_policy, value, lower_bound, upper_bound,
        result_we_got);
    BOOST_CHECK(compare(execution_policy, result_we_got, result_we_want));
}


template<
    typename ExecutionPolicy>
void test_2d_2d_2d_masked(
    ExecutionPolicy& execution_policy)
{
    fern::MaskedArray<int, 2> value         {{10, 20}, {30, 40}, {50, 60}};
    fern::Array<int, 2> lower_bound         {{ 9, 20}, {31,  0}, { 0,  0}};
    fern::Array<int, 2> upper_bound         {{90, 90}, {90, 39}, {50, 61}};
    fern::MaskedArray<int, 2> result_we_want{{10, 20}, {31, -9}, {50, 60}};
    fern::MaskedArray<int, 2> result_we_got(fern::extents[3][2], -9);

    value.mask()[1][1] = true;
    result_we_want.mask()[1][1] = true;

    using InputNoDataPolicy = fa::InputNoDataPolicies<
        fa::DetectNoDataByValue<fern::Mask<2>>, fa::SkipNoData,
        fa::SkipNoData>;
    using OutputNoDataPolicy = fa::MarkNoDataByValue<fern::Mask<2>>;

    InputNoDataPolicy input_no_data_policy{{value.mask(), true}, {}, {}};
    OutputNoDataPolicy output_no_data_policy{result_we_got.mask(), true};

    fa::core::clamp<>(input_no_data_policy, output_no_data_policy,
        execution_policy, value, lower_bound, upper_bound, result_we_got);
    BOOST_CHECK(compare(execution_policy, result_we_got, result_we_want));
}


FERN_TEST_CASES(0d_0d_0d)
FERN_TEST_CASES(1d_1d_1d)
// FERN_TEST_CASES(1d_1d_1d_masked)
// FERN_TEST_CASES(1d_1d_0d)
// FERN_TEST_CASES(1d_1d_0d_masked)
// FERN_TEST_CASES(1d_0d_1d)
// FERN_TEST_CASES(1d_0d_1d_masked)
// FERN_TEST_CASES(2d_2d_2d)
FERN_TEST_CASES(2d_2d_2d_masked)
// FERN_TEST_CASES(2d_2d_2d)
// FERN_TEST_CASES(2d_2d_0d)
// FERN_TEST_CASES(2d_2d_0d_masked)
// FERN_TEST_CASES(2d_0d_2d)
// FERN_TEST_CASES(2d_0d_2d_masked)
