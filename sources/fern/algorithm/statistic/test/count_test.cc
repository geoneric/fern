#define BOOST_TEST_MODULE fern algorithm statistic count
#include <boost/test/unit_test.hpp>
#include "fern/core/constant_traits.h"
#include "fern/core/vector_traits.h"
#include "fern/feature/core/array_traits.h"
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


BOOST_FIXTURE_TEST_SUITE(count, fern::ThreadClient)

BOOST_AUTO_TEST_CASE(d0_array)
{
    verify_value<int8_t, size_t>(-5, 6, 0u);
    verify_value<int8_t, size_t>(-5, -5, 1u);
    verify_value<double, size_t>(-5.5, -5.5, 1u);
    verify_value<double, size_t>(-5.5, -5.4, 0u);
}


BOOST_AUTO_TEST_CASE(masked_d0_array)
{
    fern::MaskedConstant<int32_t> constant;
    fern::MaskedConstant<size_t> result_we_get;

    // MaskedConstant with non-masking count. ----------------------------------
    // Constant is not masked.
    constant.mask() = false;
    constant.value() = 5;
    result_we_get.value() = 9;
    result_we_get.mask() = false;
    BOOST_CHECK(!constant.mask());
    fa::statistic::count(fa::sequential, constant, 5, result_we_get);
    BOOST_CHECK(!result_we_get.mask());
    BOOST_CHECK_EQUAL(result_we_get.value(), 1);

    // Constant is masked.
    constant.mask() = true;
    constant.value() = 5;
    result_we_get.value() = 9;
    result_we_get.mask() = false;
    BOOST_CHECK(constant.mask());
    fa::statistic::count(fa::sequential, constant, 5, result_we_get);
    BOOST_CHECK(!result_we_get.mask());
    BOOST_CHECK_EQUAL(result_we_get.value(), 1);

    // MaskedConstant with masking count. --------------------------------------
    using InputNoDataPolicy = fa::DetectNoDataByValue<bool>;
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
            InputNoDataPolicy(constant.mask(), true),
            output_no_data_policy,
            fa::sequential,
            constant, 5, result_we_get);
    BOOST_CHECK(!result_we_get.mask());
    BOOST_CHECK_EQUAL(result_we_get.value(), 1);

    // Constant is masked.
    constant.value() = 5;
    constant.mask() = true;
    result_we_get.value() = 9;
    result_we_get.mask() = false;
    BOOST_CHECK(constant.mask());
    BOOST_CHECK(!result_we_get.mask());
    fa::statistic::count(
            InputNoDataPolicy(constant.mask(), true),
            output_no_data_policy,
            fa::sequential,
            constant, 5, result_we_get);
    BOOST_CHECK(result_we_get.mask());
    BOOST_CHECK_EQUAL(result_we_get.value(), 9);
}


BOOST_AUTO_TEST_CASE(d1_array)
{
    size_t result;

    // vector
    {
        std::vector<int32_t> array{ 1, 2, 3, 5 };
        fa::statistic::count(fa::sequential, array, 2, result);
        BOOST_CHECK_EQUAL(result, 1);
    }

    // 1d array
    {
        fern::Array<int32_t, 1> array{ 1, 2, 2, 5 };
        fa::statistic::count(fa::sequential, array, 2, result);
        BOOST_CHECK_EQUAL(result, 2);
    }

    // empty
    {
        // The result value is not touched.
        std::vector<int32_t> array;
        result = 5;
        fa::statistic::count(fa::sequential, array, 2, result);
        BOOST_CHECK_EQUAL(result, 5);
    }
}


BOOST_AUTO_TEST_CASE(masked_d1_array)
{
    using InputNoDataPolicy = fa::DetectNoDataByValue<fern::Mask<1>>;
    using OutputNoDataPolicy = fa::MarkNoDataByValue<bool>;

    fern::MaskedArray<int32_t, 1> array{ 1, 2, 3, 5 };
    fern::MaskedConstant<size_t> result;

    // 1d masked array with non-masking count
    {
        result.value() = 9;
        result.mask() = false;
        fa::statistic::count(fa::sequential, array, 2, result);
        BOOST_CHECK_EQUAL(result.value(), 1);
    }

    // 1d masked array with masking count
    {
        array.mask()[2] = true;
        result.value() = 9;
        result.mask() = false;
        OutputNoDataPolicy output_no_data_policy(result.mask(), true);
        fa::statistic::count(
                InputNoDataPolicy(array.mask(), true),
                output_no_data_policy,
                fa::sequential, array, 2, result);
        BOOST_CHECK(!result.mask());
        BOOST_CHECK_EQUAL(result.value(), 1);

        // Mask the whole input. Result must be masked too.
        array.mask_all();
        fa::statistic::count(
                InputNoDataPolicy(array.mask(), true),
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
                InputNoDataPolicy(empty_array.mask(), true),
                output_no_data_policy,
                fa::sequential, empty_array, 2, result);
        BOOST_CHECK(result.mask());
        BOOST_CHECK_EQUAL(result.value(), 9);
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
        size_t result;
        fa::statistic::count(fa::sequential, array, 1, result);
        BOOST_CHECK_EQUAL(result, 1);
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
        fern::MaskedConstant<size_t> result;
        fa::statistic::count(fa::sequential, array, -2, result);
        BOOST_CHECK(!result.mask());
        BOOST_CHECK_EQUAL(result.value(), 1);
    }

    // 2d masked array with masking count
    {
        using InputNoDataPolicy = fa::DetectNoDataByValue<fern::Mask<2>>;
        using OutputNoDataPolicy = fa::MarkNoDataByValue<bool>;

        // Mask the 9.
        array.mask()[1][1] = true;

        // Count 2's.
        fern::MaskedConstant<size_t> result;
        OutputNoDataPolicy output_no_data_policy(result.mask(), true);
        fa::statistic::count(
                InputNoDataPolicy(array.mask(), true), output_no_data_policy,
                fa::sequential, array, 2, result);
        BOOST_CHECK(!result.mask());
        BOOST_CHECK_EQUAL(result.value(), 1);

        // Count 9's. The one present is not visible, so the result is 0.
        result.value() = 999;
        fa::statistic::count(
                InputNoDataPolicy(array.mask(), true), output_no_data_policy,
                fa::sequential, array, 9, result);
        BOOST_CHECK(!result.mask());
        BOOST_CHECK_EQUAL(result.value(), 0);

        // Mask the whole input. Result must be masked too.
        array.mask_all();
        result.mask() = false;
        result.value() = 999;
        fa::statistic::count(
                InputNoDataPolicy(array.mask(), true), output_no_data_policy,
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
    size_t result_we_got;
    size_t result_we_want;

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
        using InputNoDataPolicy = fa::SkipNoData<>;
        using OutputNoDataPolicy = fa::MarkNoDataByValue<bool>;
        fern::MaskedConstant<size_t> result_we_got;
        OutputNoDataPolicy output_no_data_policy(result_we_got.mask(), true);

        // Verify executor can handle masked result.
        fa::statistic::count(
            InputNoDataPolicy(), output_no_data_policy,
            fa::parallel, argument, 5, result_we_got);
    }
}

BOOST_AUTO_TEST_SUITE_END()
