#define BOOST_TEST_MODULE fern algorithm statistic count
#include <boost/test/unit_test.hpp>
#include "fern/core/data_customization_point/constant.h"
#include "fern/core/data_customization_point/vector.h"
#include "fern/feature/core/data_customization_point/array.h"
#include "fern/feature/core/masked_array_traits.h"
#include "fern/feature/core/masked_constant_traits.h"
#include "fern/algorithm/statistic/count.h"


namespace fa = fern::algorithm;


template<
    class A,
    class R>
void verify_value(
    A const& array,
    A const& value,
    R const& result_we_want)
{
    R result_we_get;
    fa::statistic::count(fa::sequential, array, value, result_we_get);
    BOOST_CHECK_EQUAL(result_we_get, result_we_want);
}


BOOST_AUTO_TEST_SUITE(count)

BOOST_AUTO_TEST_CASE(d0_array)
{
    verify_value<int8_t, uint64_t>(-5, 6, 0u);
    verify_value<int8_t, uint64_t>(-5, -5, 1u);
    verify_value<double, uint64_t>(-5.5, -5.5, 1u);
    verify_value<double, uint64_t>(-5.5, -5.4, 0u);
}


BOOST_AUTO_TEST_CASE(masked_d0_array)
{
    fern::MaskedConstant<int32_t> constant;
    fern::MaskedConstant<uint64_t> result_we_get;

    // MaskedConstant with non-masking count. ----------------------------------
    // Constant is not masked.
    constant.mask() = false;
    constant.value() = 5;
    result_we_get.value() = 9;
    result_we_get.mask() = false;
    BOOST_CHECK(!constant.mask());
    fa::statistic::count(fa::sequential, constant, 5, result_we_get);
    BOOST_CHECK(!result_we_get.mask());
    BOOST_CHECK_EQUAL(result_we_get.value(), 1u);

    // Constant is masked.
    constant.mask() = true;
    constant.value() = 5;
    result_we_get.value() = 9;
    result_we_get.mask() = false;
    BOOST_CHECK(constant.mask());
    fa::statistic::count(fa::sequential, constant, 5, result_we_get);
    BOOST_CHECK(!result_we_get.mask());
    BOOST_CHECK_EQUAL(result_we_get.value(), 1u);

    // MaskedConstant with masking count. --------------------------------------
    using InputNoDataPolicy = fa::InputNoDataPolicies<
        fa::DetectNoDataByValue<bool>, fa::SkipNoData>;
    using OutputNoDataPolicy = fa::MarkNoDataByValue<bool>;

    // Constant is not masked.
    constant.value() = 5;
    constant.mask() = false;
    result_we_get.value() = 9;
    result_we_get.mask() = false;
    BOOST_CHECK(!constant.mask());
    BOOST_CHECK(!result_we_get.mask());
    OutputNoDataPolicy output_no_data_policy(result_we_get.mask(), true);
    fa::statistic::count(
        InputNoDataPolicy{{constant.mask(), true}, {}},
        output_no_data_policy,
        fa::sequential,
        constant, 5, result_we_get);
    BOOST_CHECK(!result_we_get.mask());
    BOOST_CHECK_EQUAL(result_we_get.value(), 1u);

    // Constant is masked.
    constant.value() = 5;
    constant.mask() = true;
    result_we_get.value() = 9;
    result_we_get.mask() = false;
    BOOST_CHECK(constant.mask());
    BOOST_CHECK(!result_we_get.mask());
    fa::statistic::count(
        InputNoDataPolicy{{constant.mask(), true}, {}},
        output_no_data_policy,
        fa::sequential,
        constant, 5, result_we_get);
    BOOST_CHECK(result_we_get.mask());
    BOOST_CHECK_EQUAL(result_we_get.value(), 9u);
}


BOOST_AUTO_TEST_CASE(d1_array)
{
    uint64_t result;

    // vector
    {
        std::vector<int32_t> array{ 1, 2, 3, 5 };
        fa::statistic::count(fa::sequential, array, 2, result);
        BOOST_CHECK_EQUAL(result, 1u);
    }

    // 1d array
    {
        fern::Array<int32_t, 1> array{ 1, 2, 2, 5 };
        fa::statistic::count(fa::sequential, array, 2, result);
        BOOST_CHECK_EQUAL(result, 2u);
    }

    // empty
    {
        // The result value is not touched.
        std::vector<int32_t> array;
        result = 5;
        fa::statistic::count(fa::sequential, array, 2, result);
        BOOST_CHECK_EQUAL(result, 5u);
    }
}


BOOST_AUTO_TEST_CASE(masked_d1_array)
{
    using InputNoDataPolicy = fa::InputNoDataPolicies<
        fa::DetectNoDataByValue<fern::Mask<1>>, fa::SkipNoData>;
    using OutputNoDataPolicy = fa::MarkNoDataByValue<bool>;

    fern::MaskedArray<int32_t, 1> array{ 1, 2, 3, 5 };
    fern::MaskedConstant<size_t> result;

    // 1d masked array with non-masking count
    {
        result.value() = 9;
        result.mask() = false;
        fa::statistic::count(fa::sequential, array, 2, result);
        BOOST_CHECK_EQUAL(result.value(), 1u);
    }

    // 1d masked array with masking count
    {
        array.mask()[2] = true;
        result.value() = 9;
        result.mask() = false;
        OutputNoDataPolicy output_no_data_policy(result.mask(), true);
        fa::statistic::count(
            InputNoDataPolicy{{array.mask(), true}, {}},
            output_no_data_policy,
            fa::sequential, array, 2, result);
        BOOST_CHECK(!result.mask());
        BOOST_CHECK_EQUAL(result.value(), 1u);

        // Mask the whole input. Result must be masked too.
        array.mask_all();
        fa::statistic::count(
            InputNoDataPolicy{{array.mask(), true}, {}},
            output_no_data_policy,
            fa::sequential, array, 2, result);
        BOOST_CHECK(result.mask());
    }

    // empty
    {
        fern::MaskedArray<int32_t, 1> empty_array;
        result.value() = 9;
        result.mask() = false;
        OutputNoDataPolicy output_no_data_policy(result.mask(), true);
        fa::statistic::count(
            InputNoDataPolicy{{empty_array.mask(), true}, {}},
            output_no_data_policy,
            fa::sequential, empty_array, 2, result);
        BOOST_CHECK(result.mask());
        BOOST_CHECK_EQUAL(result.value(), 9u);
    }
}


BOOST_AUTO_TEST_CASE(d2_array)
{
    // 2d array
    {
        fern::Array<int8_t, 2> array{
            { -2, -1 },
            {  0,  9 },
            {  1,  2 }
        };
        uint64_t result;
        fa::statistic::count(fa::sequential, array, 1, result);
        BOOST_CHECK_EQUAL(result, 1u);
    }
}


BOOST_AUTO_TEST_CASE(masked_d2_array)
{
    fern::MaskedArray<int8_t, 2> array{
        { -2, -1 },
        {  5,  9 },
        {  1,  2 }
    };

    // 2d masked array with non-masking count
    {
        fern::MaskedConstant<uint64_t> result;
        fa::statistic::count(fa::sequential, array, -2, result);
        BOOST_CHECK(!result.mask());
        BOOST_CHECK_EQUAL(result.value(), 1u);
    }

    // 2d masked array with masking count
    {
        using InputNoDataPolicy = fa::InputNoDataPolicies<
            fa::DetectNoDataByValue<fern::Mask<2>>, fa::SkipNoData>;
        using OutputNoDataPolicy = fa::MarkNoDataByValue<bool>;

        // Mask the 9.
        array.mask()[1][1] = true;

        // Count 2's.
        fern::MaskedConstant<uint64_t> result;
        OutputNoDataPolicy output_no_data_policy(result.mask(), true);
        fa::statistic::count(
            InputNoDataPolicy{{array.mask(), true}, {}}, output_no_data_policy,
            fa::sequential, array, 2, result);
        BOOST_CHECK(!result.mask());
        BOOST_CHECK_EQUAL(result.value(), 1u);

        // Count 9's. The one present is not visible, so the result is 0.
        result.value() = 999;
        fa::statistic::count(
            InputNoDataPolicy{{array.mask(), true}, {}}, output_no_data_policy,
            fa::sequential, array, 9, result);
        BOOST_CHECK(!result.mask());
        BOOST_CHECK_EQUAL(result.value(), 0u);

        // Mask the whole input. Result must be masked too.
        array.mask_all();
        result.mask() = false;
        result.value() = 999;
        fa::statistic::count(
            InputNoDataPolicy{{array.mask(), true}, {}}, output_no_data_policy,
            fa::sequential, array, 9, result);
        BOOST_CHECK(result.mask());
    }
}


BOOST_AUTO_TEST_CASE(concurrent)
{
    // Create a somewhat larger array.
    size_t const nr_rows = 600;
    size_t const nr_cols = 400;
    auto const extents = fern::extents[nr_rows][nr_cols];
    fern::Array<int32_t, 2> argument(extents);
    uint64_t result_we_got;
    uint64_t result_we_want;

    std::iota(argument.data(), argument.data() + argument.num_elements(), 0);
    result_we_want = 1;

    // Serial.
    {
        result_we_got = 9;
        fa::statistic::count(fa::sequential, argument, 5, result_we_got);
        BOOST_CHECK_EQUAL(result_we_got, result_we_want);
    }

    // Concurrent.
    {
        result_we_got = 9;
        fa::statistic::count(fa::parallel, argument, 5, result_we_got);
        BOOST_CHECK_EQUAL(result_we_got, result_we_want);
    }

    {
        using InputNoDataPolicy = fa::InputNoDataPolicies<fa::SkipNoData,
              fa::SkipNoData>;
        using OutputNoDataPolicy = fa::MarkNoDataByValue<bool>;
        fern::MaskedConstant<size_t> result_we_got;
        OutputNoDataPolicy output_no_data_policy(result_we_got.mask(), true);

        // Verify executor can handle masked result.
        fa::statistic::count(
            InputNoDataPolicy{{}, {}}, output_no_data_policy,
            fa::parallel, argument, 5, result_we_got);
    }
}

BOOST_AUTO_TEST_SUITE_END()
