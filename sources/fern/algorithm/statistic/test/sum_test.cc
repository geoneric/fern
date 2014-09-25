#define BOOST_TEST_MODULE fern algorithm statistic sum
#include <boost/test/unit_test.hpp>
#include "fern/core/constant_traits.h"
#include "fern/core/vector_traits.h"
#include "fern/feature/core/array_traits.h"
#include "fern/feature/core/masked_array_traits.h"
#include "fern/feature/core/masked_constant_traits.h"
#include "fern/algorithm/statistic/sum.h"


template<
    class A1,
    class R>
void verify_value(
    A1 const& array,
    R const& result_we_want)
{
    R result_we_get;
    fern::statistic::sum(fern::sequential, array, result_we_get);
    BOOST_CHECK_EQUAL(result_we_get, result_we_want);
}


BOOST_FIXTURE_TEST_SUITE(sum, fern::ThreadClient)

BOOST_AUTO_TEST_CASE(d0_array)
{
    verify_value<int8_t, int8_t>(-5, -5);
    verify_value<int8_t, int8_t>(-5, -5);
    verify_value<double, double>(-5.5, -5.5);
    verify_value<double, double>(-5.5, -5.5);
}


BOOST_AUTO_TEST_CASE(masked_d0_array)
{
    using MaskedConstant = fern::MaskedConstant<int32_t>;
    MaskedConstant constant;
    MaskedConstant result_we_get;

    // MaskedConstant with non-masking sum. ------------------------------------
    // Constant is not masked.
    constant.mask() = false;
    constant.value() = 5;
    BOOST_CHECK(!constant.mask());
    fern::statistic::sum(fern::sequential, constant, result_we_get);
    BOOST_CHECK(!result_we_get.mask());
    BOOST_CHECK_EQUAL(result_we_get.value(), 5);

    // Constant is masked.
    constant.mask() = true;
    constant.value() = 6;
    BOOST_CHECK(constant.mask());
    fern::statistic::sum(fern::sequential, constant, result_we_get);
    BOOST_CHECK(!result_we_get.mask());
    BOOST_CHECK_EQUAL(result_we_get.value(), 6);

    // MaskedConstant with masking sum. ----------------------------------------
    using InputNoDataPolicy = fern::DetectNoDataByValue<bool>;
    using OutputNoDataPolicy = fern::MarkNoDataByValue<bool>;

    // Constant is not masked.
    constant.value() = 5;
    constant.mask() = false;
    BOOST_CHECK(!constant.mask());
    BOOST_CHECK(!result_we_get.mask());
    {
        OutputNoDataPolicy output_no_data_policy(result_we_get.mask(), true);
        fern::statistic::sum<fern::binary::DiscardRangeErrors>(
                InputNoDataPolicy(constant.mask(), true), output_no_data_policy,
                fern::sequential, constant, result_we_get);
    }
    BOOST_CHECK(!result_we_get.mask());
    BOOST_CHECK_EQUAL(result_we_get.value(), 5);

    // Constant is masked.
    constant.value() = 6;
    constant.mask() = true;
    BOOST_CHECK(constant.mask());
    BOOST_CHECK(!result_we_get.mask());
    {
        OutputNoDataPolicy output_no_data_policy(result_we_get.mask(), true);
        fern::statistic::sum<fern::binary::DiscardRangeErrors>(
                InputNoDataPolicy(constant.mask(), true), output_no_data_policy,
                fern::sequential, constant, result_we_get);
    }
    BOOST_CHECK(result_we_get.mask());
    BOOST_CHECK_EQUAL(result_we_get.value(), 5);
}


BOOST_AUTO_TEST_CASE(d1_array)
{
    // vector
    {
        std::vector<int32_t> array{ 1, 2, 3, 5 };
        int32_t result;
        fern::statistic::sum(fern::sequential, array, result);
        BOOST_CHECK_EQUAL(result, 11);
    }

    // 1d array
    {
        fern::Array<int32_t, 1> array{ 1, 2, 3, 5 };
        int32_t result;
        fern::statistic::sum(fern::sequential, array, result);
        BOOST_CHECK_EQUAL(result, 11);
    }

    // empty
    {
        // The result value is not touched.
        std::vector<int32_t> array;
        int32_t result{5};
        fern::statistic::sum(fern::sequential, array, result);
        BOOST_CHECK_EQUAL(result, 5);
    }
}


BOOST_AUTO_TEST_CASE(masked_d1_array)
{
    using InputNoDataPolicy = fern::DetectNoDataByValue<fern::Mask<1>>;
    using OutputNoDataPolicy = fern::MarkNoDataByValue<bool>;
    using MaskedArray = fern::MaskedArray<int32_t, 1>;
    using MaskedConstant = fern::MaskedConstant<int32_t>;

    MaskedArray array{ 1, 2, 3, 5 };

    // 1d masked array with non-masking sum
    {
        MaskedConstant result;
        fern::statistic::sum(fern::sequential, array, result);
        BOOST_CHECK_EQUAL(result.value(), 11);
    }

    // 1d masked array with masking sum
    {
        array.mask()[2] = true;
        MaskedConstant result;
        OutputNoDataPolicy output_no_data_policy(result.mask(), true);
        fern::statistic::sum<fern::binary::DiscardRangeErrors>(
                InputNoDataPolicy(array.mask(), true), output_no_data_policy,
                fern::sequential, array, result);
        BOOST_CHECK(!result.mask());
        BOOST_CHECK_EQUAL(result.value(), 8);

        // Mask the whole input. Result must be masked too.
        std::fill(array.mask().data(), array.mask().data() +
            array.num_elements(), true);
        fern::statistic::sum<fern::binary::DiscardRangeErrors>(
                InputNoDataPolicy(array.mask(), true), output_no_data_policy,
                fern::sequential, array, result);
        BOOST_CHECK(result.mask());
    }

    // empty
    {
        MaskedArray empty_array;
        MaskedConstant result{5};
        OutputNoDataPolicy output_no_data_policy(result.mask(), true);
        fern::statistic::sum<fern::binary::DiscardRangeErrors>(
            InputNoDataPolicy(empty_array.mask(), true), output_no_data_policy,
            fern::sequential, empty_array, result);
        BOOST_CHECK(result.mask());
        BOOST_CHECK_EQUAL(result.value(), 5);
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
        int8_t result;
        fern::statistic::sum(fern::sequential, array, result);
        BOOST_CHECK_EQUAL(result, 9);
    }
}


BOOST_AUTO_TEST_CASE(masked_d2_array)
{
    fern::MaskedArray<int8_t, 2> array{
        { -2, -1 },
        {  5,  9 },
        {  1,  2 }
    };

    // 2d masked array with non-masking sum
    {
        fern::MaskedConstant<int8_t> result;
        fern::statistic::sum(fern::sequential, array, result);
        BOOST_CHECK_EQUAL(result.value(), 14);
    }

    // 2d masked array with masking sum
    {
        using MaskedConstant = fern::MaskedConstant<int8_t>;
        using InputNoDataPolicy = fern::DetectNoDataByValue<fern::Mask<2>>;
        using OutputNoDataPolicy = fern::MarkNoDataByValue<bool>;

        array.mask()[1][1] = true;
        MaskedConstant result;
        OutputNoDataPolicy output_no_data_policy(result.mask(), true);
        fern::statistic::sum<fern::binary::DiscardRangeErrors>(
                InputNoDataPolicy(array.mask(), true), output_no_data_policy,
                fern::sequential, array, result);
        BOOST_CHECK(!result.mask());
        BOOST_CHECK_EQUAL(result.value(), 5);

        // Mask the whole input. Result must be masked too.
        std::fill(array.mask().data(), array.mask().data() +
            array.num_elements(), true);
        fern::statistic::sum<fern::binary::DiscardRangeErrors>(
                InputNoDataPolicy(array.mask(), true), output_no_data_policy,
                fern::sequential, array, result);
        BOOST_CHECK(result.mask());
    }
}


BOOST_AUTO_TEST_CASE(out_of_range)
{
    using Array = fern::Array<int32_t, 1>;
    Array overflow_array{ fern::TypeTraits<int32_t>::max, 1 };
    Array underflow_array{ fern::TypeTraits<int32_t>::min, -1 };

    // 1d masked array with non-masking sum
    {
        int32_t result;

        // Integer overflow -> max + 1 == min.
        fern::statistic::sum(fern::sequential, overflow_array, result);
        BOOST_CHECK_EQUAL(result, fern::TypeTraits<int32_t>::min);

        // Integer underflow -> min - 1 == max.
        fern::statistic::sum(fern::sequential, underflow_array, result);
        BOOST_CHECK_EQUAL(result, fern::TypeTraits<int32_t>::max);
    }

    // 1d masked array with masking sum
    {
        using InputNoDataPolicy = fern::SkipNoData;
        using OutputNoDataPolicy = fern::MarkNoDataByValue<bool>;

        // fern::MaskedConstant<int32_t> result;
        using R = fern::MaskedConstant<int32_t>;
        R result;
        OutputNoDataPolicy output_no_data_policy(result.mask(), true);
        fern::statistic::sum<fern::sum::OutOfRangePolicy>(
            InputNoDataPolicy(), output_no_data_policy,
            fern::sequential, overflow_array, result);
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
    int32_t result_we_got;
    int32_t result_we_want;

    std::iota(argument.data(), argument.data() + argument.num_elements(), 0);
    result_we_want = std::accumulate(argument.data(), argument.data() +
        argument.num_elements(), 0);

    // Serial.
    {
        fern::statistic::sum(fern::sequential, argument, result_we_got);
        BOOST_CHECK_EQUAL(result_we_got, result_we_want);
    }


    // Concurrent.
    {
        fern::statistic::sum(fern::parallel, argument, result_we_got);
        BOOST_CHECK_EQUAL(result_we_got, result_we_want);
    }

    {
        using InputNoDataPolicy = fern::SkipNoData;
        using OutputNoDataPolicy = fern::MarkNoDataByValue<bool>;
        fern::MaskedConstant<int32_t> result_we_got;

        OutputNoDataPolicy output_no_data_policy(result_we_got.mask(), true);

        // Fill input with zeros. Put max in upper left and 1 in cell next to
        // it. See if no-data detected by a single thread propagates through
        // aggregation.
        std::fill(argument.data(), argument.data() +
            argument.num_elements(), 0);
        argument[0][0] = fern::TypeTraits<int32_t>::max;
        argument[0][1] = 1;

        {
            // Base case, serial.
            result_we_got.mask() = false;
            fern::statistic::sum<fern::sum::OutOfRangePolicy>(
                InputNoDataPolicy(), output_no_data_policy,
                fern::sequential, argument, result_we_got);
            BOOST_CHECK(result_we_got.mask());
        }

        {
            // Concurrent.
            result_we_got.mask() = false;
            fern::statistic::sum<fern::sum::OutOfRangePolicy>(
                InputNoDataPolicy(), output_no_data_policy,
                fern::parallel, argument, result_we_got);
            BOOST_CHECK(result_we_got.mask());
        }


        // Fill input with zeros. Put max in upper left and 1 in lower right
        // cell. This verifies that the aggregate operation uses the output
        // no-data mask.
        std::fill(argument.data(), argument.data() + argument.num_elements(),
            0);
        argument[0][0] = fern::TypeTraits<int32_t>::max;
        argument[nr_rows-1][nr_cols-1] = 1;

        {
            // Base case, serial.
            result_we_got.mask() = false;
            fern::statistic::sum<fern::sum::OutOfRangePolicy>(
                InputNoDataPolicy(), output_no_data_policy,
                fern::sequential, argument, result_we_got);
            BOOST_CHECK(result_we_got.mask());
        }

        {
            // Concurrent.
            result_we_got.mask() = false;
            fern::statistic::sum<fern::sum::OutOfRangePolicy>(
                InputNoDataPolicy(), output_no_data_policy,
                fern::parallel, argument, result_we_got);
            BOOST_CHECK(result_we_got.mask());
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
