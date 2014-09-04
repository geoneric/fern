#define BOOST_TEST_MODULE fern algorithm core offset
#include <boost/test/unit_test.hpp>
#include "fern/feature/core/masked_array_traits.h"
#include "fern/core/point_traits.h"
#include "fern/core/vector_traits.h"
#include "fern/algorithm/algebra/elementary/equal.h"
#include "fern/algorithm/core/offset.h"
#include "fern/algorithm/statistic/count.h"


BOOST_AUTO_TEST_SUITE(offset)

template<
    class ExecutionPolicy,
    class Value>
bool equal(
    ExecutionPolicy const& execution_policy,
    Value const& value1,
    Value const& value2)
{
    auto equal_result = fern::clone<int>(value1, 0);

    fern::algebra::equal(execution_policy, value1, value2, equal_result);

    size_t nr_equal_values;

    fern::statistic::count(execution_policy, equal_result, 1, nr_equal_values);

    BOOST_CHECK_EQUAL(nr_equal_values, fern::size(value1));

    return nr_equal_values == fern::size(value1);
}


template<
    class ExecutionPolicy,
    class Value>
bool equal(
    ExecutionPolicy const& execution_policy,
    fern::MaskedArray<Value, 1> const& value1,
    fern::MaskedArray<Value, 1> const& value2)
{
    bool values_are_the_same = equal(execution_policy,
            dynamic_cast<fern::Array<Value, 1> const&>(value1),
            dynamic_cast<fern::Array<Value, 1> const&>(value2));
    bool mask_is_the_same = equal(execution_policy,
            value1.mask(),
            value2.mask());

    BOOST_CHECK(values_are_the_same);
    BOOST_CHECK(mask_is_the_same);

    return values_are_the_same && mask_is_the_same;
}

void test_array_1d(
    fern::ExecutionPolicy const& execution_policy)
{
    size_t const nr_threads{fern::ThreadClient::hardware_concurrency()};
    size_t const nr_values{100 * nr_threads};
    std::vector<int> values(nr_values);
    std::iota(values.begin(), values.end(), 0);

    {
        fern::Point<int, 1> offset{ 0 };
        // 0, 1, ..., n - 1
        std::vector<int> result_we_want(nr_values, -9);
        std::iota(result_we_want.begin(), result_we_want.end(), 0);
        std::vector<int> result_we_got(nr_values, -9);

        fern::core::offset(execution_policy, values, offset, result_we_got);

        BOOST_CHECK(equal(execution_policy, result_we_got, result_we_want));
    }

    {
        fern::Point<int, 1> offset{ 2 };
        // -9, -9, 0, 1, ..., n - 3
        std::vector<int> result_we_want(nr_values, -9);
        std::iota(result_we_want.begin() + fern::get<0>(offset),
            result_we_want.end(), 0);
        std::vector<int> result_we_got(nr_values, -9);

        fern::core::offset(execution_policy, values, offset, result_we_got);

        BOOST_CHECK(equal(execution_policy, result_we_got, result_we_want));
    }

    {
        fern::Point<int, 1> offset{ static_cast<int>(nr_values) + 100 };
        // -9, -9, ..., -9, -9
        std::vector<int> result_we_want(nr_values, -9);
        std::vector<int> result_we_got(nr_values, -9);

        fern::core::offset(execution_policy, values, offset, result_we_got);

        BOOST_CHECK(equal(execution_policy, result_we_got, result_we_want));
    }

    {
        fern::Point<int, 1> offset{ -2 };
        // 2, 3, ..., -9, -9
        std::vector<int> result_we_want(nr_values, -9);
        std::iota(result_we_want.begin(),
            result_we_want.end() + fern::get<0>(offset),
            std::abs(fern::get<0>(offset)));
        std::vector<int> result_we_got(nr_values, -9);

        fern::core::offset(execution_policy, values, offset, result_we_got);

        BOOST_CHECK(equal(execution_policy, result_we_got, result_we_want));
    }

    {
        fern::Point<int, 1> offset{ -static_cast<int>(nr_values + 100) };
        // -9, -9, ..., -9, -9
        std::vector<int> result_we_want(nr_values, -9);
        std::vector<int> result_we_got(nr_values, -9);

        fern::core::offset(execution_policy, values, offset, result_we_got);

        BOOST_CHECK(equal(execution_policy, result_we_got, result_we_want));
    }
}


BOOST_AUTO_TEST_CASE(array_1d_sequential)
{
    test_array_1d(fern::sequential);
}


BOOST_AUTO_TEST_CASE(array_1d_parallel)
{
    fern::ThreadClient client;
    test_array_1d(fern::parallel);
}


void test_array_1d_masked(
    fern::ExecutionPolicy const& execution_policy)
{
    size_t const nr_threads{fern::ThreadClient::hardware_concurrency()};
    size_t const nr_values{100 * nr_threads};
    fern::MaskedArray<int, 1> values(nr_values);
    std::iota(values.data(), values.data() + nr_values, 0);
    values.mask()[1] = true;

    {
        fern::Point<int, 1> offset{ 0 };
        // 0, -9, 2, ..., n - 1
        fern::MaskedArray<int, 1> result_we_want(nr_values);
        std::iota(result_we_want.data(), result_we_want.data() + nr_values, 0);
        result_we_want.data()[1] = -9;
        result_we_want.mask()[1] = true;
        fern::MaskedArray<int, 1> result_we_got(nr_values, -9);

        fern::DetectNoDataByValue<fern::Mask<1>> input_no_data_policy(
            values.mask(), true);
        fern::MarkNoDataByValue<fern::Mask<1>> output_no_data_policy(
            result_we_got.mask(), true);

        fern::core::offset(input_no_data_policy, output_no_data_policy,
            execution_policy, values, offset, result_we_got);

        BOOST_CHECK(equal(execution_policy, result_we_got, result_we_want));
    }

    {
        fern::Point<int, 1> offset{ 2 };
        // -9, -9, 0, -9, 1, ..., n - 3
        fern::MaskedArray<int, 1> result_we_want(nr_values);
        std::iota(result_we_want.data() + fern::get<0>(offset),
            result_we_want.data() + nr_values, 0);
        std::fill(result_we_want.data(), result_we_want.data() +
            fern::get<0>(offset), -9);
        result_we_want.data()[fern::get<0>(offset) + 1] = -9;
        std::fill(result_we_want.mask().data(), result_we_want.mask().data() +
            fern::get<0>(offset), true);
        result_we_want.mask()[fern::get<0>(offset) + 1] = true;
        fern::MaskedArray<int, 1> result_we_got(nr_values, -9);

        fern::DetectNoDataByValue<fern::Mask<1>> input_no_data_policy(
            values.mask(), true);
        fern::MarkNoDataByValue<fern::Mask<1>> output_no_data_policy(
            result_we_got.mask(), true);

        fern::core::offset(input_no_data_policy, output_no_data_policy,
            execution_policy, values, offset, result_we_got);

        BOOST_CHECK(equal(execution_policy, result_we_got, result_we_want));
    }

    {
        fern::Point<int, 1> offset{ static_cast<int>(nr_values + 100) };
        // -9, -9, ..., -9, -9
        fern::MaskedArray<int, 1> result_we_want(nr_values, -9);
        std::fill(result_we_want.mask().data(), result_we_want.mask().data() +
            nr_values, true);
        fern::MaskedArray<int, 1> result_we_got(nr_values, -9);

        fern::DetectNoDataByValue<fern::Mask<1>> input_no_data_policy(
            values.mask(), true);
        fern::MarkNoDataByValue<fern::Mask<1>> output_no_data_policy(
            result_we_got.mask(), true);

        fern::core::offset(input_no_data_policy, output_no_data_policy,
            execution_policy, values, offset, result_we_got);

        BOOST_CHECK(equal(execution_policy, result_we_got, result_we_want));
    }

    {
        fern::Point<int, 1> offset{ -2 };
        // 2, 3, ..., n - 1, -9, -9
        fern::MaskedArray<int, 1> result_we_want(nr_values, -9);
        std::iota(result_we_want.data(), result_we_want.data() + nr_values +
            fern::get<0>(offset), std::abs(fern::get<0>(offset)));
        // false, false, ..., false, true, true
        std::fill(result_we_want.mask().data() + nr_values +
            fern::get<0>(offset), result_we_want.mask().data() + nr_values,
            true);

        fern::MaskedArray<int, 1> result_we_got(nr_values, -9);

        fern::DetectNoDataByValue<fern::Mask<1>> input_no_data_policy(
            values.mask(), true);
        fern::MarkNoDataByValue<fern::Mask<1>> output_no_data_policy(
            result_we_got.mask(), true);

        fern::core::offset(input_no_data_policy, output_no_data_policy,
            execution_policy, values, offset, result_we_got);

        BOOST_CHECK(equal(execution_policy, result_we_got, result_we_want));
    }

    {
        fern::Point<int, 1> offset{ -static_cast<int>(nr_values + 100) };
        // -9, -9, ..., -9, -9
        fern::MaskedArray<int, 1> result_we_want(nr_values, -9);
        // true, true, ..., true, true
        result_we_want.mask().fill(true);
        fern::MaskedArray<int, 1> result_we_got(nr_values, -9);

        fern::DetectNoDataByValue<fern::Mask<1>> input_no_data_policy(
            values.mask(), true);
        fern::MarkNoDataByValue<fern::Mask<1>> output_no_data_policy(
            result_we_got.mask(), true);

        fern::core::offset(input_no_data_policy, output_no_data_policy,
            execution_policy, values, offset, result_we_got);

        BOOST_CHECK(equal(execution_policy, result_we_got, result_we_want));
    }
}


BOOST_AUTO_TEST_CASE(array_1d_masked_sequential)
{
    test_array_1d_masked(fern::sequential);
}


BOOST_AUTO_TEST_CASE(array_1d_masked_parallel)
{
    fern::ThreadClient client;
    test_array_1d_masked(fern::parallel);
}


// Fill value.


// TODO 2d

BOOST_AUTO_TEST_SUITE_END()
