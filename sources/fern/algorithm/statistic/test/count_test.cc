#define BOOST_TEST_MODULE fern algorithm statistic count
#include <boost/test/unit_test.hpp>
#include "fern/core/constant_traits.h"
#include "fern/core/typename.h"
#include "fern/core/vector_traits.h"
#include "fern/feature/core/array_traits.h"
#include "fern/feature/core/masked_array_traits.h"
#include "fern/feature/core/masked_constant_traits.h"
#include "fern/algorithm/algebra/executor.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/statistic/count.h"


template<
    class A,
    class R>
void verify_value(
    A const& array,
    A const& value,
    R const& result_we_want)
{
    R result_we_get;
    fern::statistic::count(array, value, result_we_get);
    BOOST_CHECK_EQUAL(result_we_get, result_we_want);
}


BOOST_AUTO_TEST_SUITE(count)

BOOST_AUTO_TEST_CASE(traits)
{
    using Count = fern::statistic::Count<int32_t, size_t>;
    BOOST_CHECK((std::is_same<fern::OperationTraits<Count>::category,
        fern::local_aggregate_operation_tag>::value));
}


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
    fern::statistic::count(constant, 5, result_we_get);
    BOOST_CHECK(!result_we_get.mask());
    BOOST_CHECK_EQUAL(result_we_get.value(), 1);

    // Constant is masked.
    constant.mask() = true;
    constant.value() = 5;
    result_we_get.value() = 9;
    result_we_get.mask() = false;
    BOOST_CHECK(constant.mask());
    fern::statistic::count(constant, 5, result_we_get);
    BOOST_CHECK(!result_we_get.mask());
    BOOST_CHECK_EQUAL(result_we_get.value(), 1);

    // MaskedConstant with masking count. --------------------------------------
    using InputNoDataPolicy = fern::DetectNoDataByValue<bool>;
    using OutputNoDataPolicy = fern::MarkNoDataByValue<bool>;

    // Constant is not masked.
    constant.value() = 5;
    constant.mask() = false;
    result_we_get.value() = 9;
    result_we_get.mask() = false;
    BOOST_CHECK(!constant.mask());
    BOOST_CHECK(!result_we_get.mask());
    fern::statistic::count<fern::MaskedConstant<int32_t>,
        fern::MaskedConstant<size_t>, InputNoDataPolicy,
        OutputNoDataPolicy>(
            InputNoDataPolicy(constant.mask(), true),
            OutputNoDataPolicy(result_we_get.mask(), true),
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
    fern::statistic::count<fern::MaskedConstant<int32_t>,
        fern::MaskedConstant<size_t>, InputNoDataPolicy,
        OutputNoDataPolicy>(
            InputNoDataPolicy(constant.mask(), true),
            OutputNoDataPolicy(result_we_get.mask(), true),
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
        fern::statistic::count(array, 2, result);
        BOOST_CHECK_EQUAL(result, 1);
    }

    // 1d array
    {
        fern::Array<int32_t, 1> array{ 1, 2, 2, 5 };
        fern::statistic::count(array, 2, result);
        BOOST_CHECK_EQUAL(result, 2);
    }

    // empty
    {
        // The result value is not touched.
        std::vector<int32_t> array;
        result = 5;
        fern::statistic::count(array, 2, result);
        BOOST_CHECK_EQUAL(result, 5);
    }
}


BOOST_AUTO_TEST_CASE(masked_d1_array)
{
    using InputNoDataPolicy = fern::DetectNoDataByValue<fern::Mask<1>>;
    using OutputNoDataPolicy = fern::MarkNoDataByValue<bool>;

    fern::MaskedArray<int32_t, 1> array{ 1, 2, 3, 5 };
    fern::MaskedConstant<size_t> result;

    // 1d masked array with non-masking count
    {
        result.value() = 9;
        result.mask() = false;
        fern::statistic::count(array, 2, result);
        BOOST_CHECK_EQUAL(result.value(), 1);
    }

    // 1d masked array with masking count
    {
        array.mask()[2] = true;
        result.value() = 9;
        result.mask() = false;
        fern::statistic::count<fern::MaskedArray<int32_t, 1>,
            fern::MaskedConstant<size_t>, InputNoDataPolicy,
            OutputNoDataPolicy>(
                InputNoDataPolicy(array.mask(), true),
                OutputNoDataPolicy(result.mask(), true),
                array, 2, result);
        BOOST_CHECK(!result.mask());
        BOOST_CHECK_EQUAL(result.value(), 1);

        // Mask the whole input. Result must be masked too.
        array.mask_all();
        fern::statistic::count<fern::MaskedArray<int32_t, 1>,
            fern::MaskedConstant<size_t>, InputNoDataPolicy,
            OutputNoDataPolicy>(
                InputNoDataPolicy(array.mask(), true),
                OutputNoDataPolicy(result.mask(), true),
                array, 2, result);
        BOOST_CHECK(result.mask());
    }

    // empty
    {
        fern::MaskedArray<int32_t, 1> empty_array;
        result.value() = 9;
        result.mask() = false;
        fern::statistic::count<fern::MaskedArray<int32_t, 1>,
            fern::MaskedConstant<size_t>, InputNoDataPolicy,
            OutputNoDataPolicy>(
                InputNoDataPolicy(empty_array.mask(), true),
                OutputNoDataPolicy(result.mask(), true),
                empty_array, 2, result);
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
        fern::statistic::count(array, 1, result);
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
        fern::statistic::count(array, -2, result);
        BOOST_CHECK(!result.mask());
        BOOST_CHECK_EQUAL(result.value(), 1);
    }

    // 2d masked array with masking count
    {
        using InputNoDataPolicy = fern::DetectNoDataByValue<fern::Mask<2>>;
        using OutputNoDataPolicy = fern::MarkNoDataByValue<bool>;

        // Mask the 9.
        array.mask()[1][1] = true;

        // Count 2's.
        fern::MaskedConstant<size_t> result;
        fern::statistic::count<fern::MaskedArray<int8_t, 2>,
            fern::MaskedConstant<size_t>, InputNoDataPolicy,
            OutputNoDataPolicy>(
                InputNoDataPolicy(array.mask(), true),
                OutputNoDataPolicy(result.mask(), true),
                array, 2, result);
        BOOST_CHECK(!result.mask());
        BOOST_CHECK_EQUAL(result.value(), 1);

        // Count 9's. The one present is not visible, so the result is 0.
        result.value() = 999;
        fern::statistic::count<fern::MaskedArray<int8_t, 2>,
            fern::MaskedConstant<size_t>, InputNoDataPolicy,
            OutputNoDataPolicy>(
                InputNoDataPolicy(array.mask(), true),
                OutputNoDataPolicy(result.mask(), true),
                array, 9, result);
        BOOST_CHECK(!result.mask());
        BOOST_CHECK_EQUAL(result.value(), 0);

        // Mask the whole input. Result must be masked too.
        array.mask_all();
        result.mask() = false;
        result.value() = 999;
        fern::statistic::count<fern::MaskedArray<int8_t, 2>,
            fern::MaskedConstant<size_t>, InputNoDataPolicy,
            OutputNoDataPolicy>(
                InputNoDataPolicy(array.mask(), true),
                OutputNoDataPolicy(result.mask(), true),
                array, 9, result);
        BOOST_CHECK(result.mask());
    }
}


BOOST_AUTO_TEST_CASE(concurrent)
{
    // Create a somewhat larger array.
    size_t const nr_rows = 6000;
    size_t const nr_cols = 4000;
    auto const extents = fern::extents[nr_rows][nr_cols];
    fern::Array<int32_t, 2> argument(extents);
    size_t result_we_got;
    size_t result_we_want;
    fern::statistic::Count<fern::Array<int32_t, 2>, size_t> count;

    std::iota(argument.data(), argument.data() + argument.num_elements(), 0);
    result_we_want = 1;

    // Serial.
    {
        fern::serial::execute(count, argument, 5, result_we_got);
        BOOST_CHECK_EQUAL(result_we_got, result_we_want);
    }

    // Concurrent.
    {
        fern::ThreadClient client;
        fern::concurrent::execute(count, argument, 5, result_we_got);
        BOOST_CHECK_EQUAL(result_we_got, result_we_want);
    }

    {
        using InputNoDataPolicy = fern::SkipNoData;
        using OutputNoDataPolicy = fern::MarkNoDataByValue<bool>;
        fern::MaskedConstant<size_t> result_we_got;
        fern::statistic::Count<fern::Array<int32_t, 2>,
            fern::MaskedConstant<size_t>, InputNoDataPolicy,
            OutputNoDataPolicy> count(
                InputNoDataPolicy(),
                OutputNoDataPolicy(result_we_got.mask(), true));

        // Verify executor can handle masked result.
        fern::ThreadClient client;
        fern::concurrent::execute(count, argument, 5, result_we_got);
    }
}

BOOST_AUTO_TEST_SUITE_END()
