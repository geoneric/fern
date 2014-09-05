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
    class Value,
    size_t nr_dimensions>
bool equal(
    ExecutionPolicy const& execution_policy,
    fern::MaskedArray<Value, nr_dimensions> const& value1,
    fern::MaskedArray<Value, nr_dimensions> const& value2)
{
    bool values_are_the_same = equal(execution_policy,
            dynamic_cast<fern::Array<Value, nr_dimensions> const&>(value1),
            dynamic_cast<fern::Array<Value, nr_dimensions> const&>(value2));
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
    size_t const nr_elements{10 * nr_threads};
    std::vector<int> values(nr_elements);
    std::vector<int> result_we_want(nr_elements);
    std::vector<int> result_we_got(nr_elements);

    std::iota(values.begin(), values.end(), 0);

    {
        fern::Point<int, 1> offset{0};

        // 0, 1, ..., n - 1
        std::iota(result_we_want.begin(), result_we_want.end(), 0);
        std::fill(result_we_got.begin(), result_we_got.end(), -9);

        fern::core::offset(execution_policy, values, offset, result_we_got);

        BOOST_CHECK(equal(execution_policy, result_we_got, result_we_want));
    }

    {
        fern::Point<int, 1> offset{2};

        // -9, -9, 0, 1, ..., n - 3
        std::iota(result_we_want.begin() + fern::get<0>(offset),
            result_we_want.end(), 0);
        std::fill(result_we_want.begin(), result_we_want.begin() +
            fern::get<0>(offset), -9);
        std::fill(result_we_got.begin(), result_we_got.end(), -9);

        fern::core::offset(execution_policy, values, offset, result_we_got);

        BOOST_CHECK(equal(execution_policy, result_we_got, result_we_want));
    }

    {
        fern::Point<int, 1> offset{static_cast<int>(nr_elements) + 10};

        // -9, -9, ..., -9, -9
        std::fill(result_we_want.begin(), result_we_want.end(), -9);
        std::fill(result_we_got.begin(), result_we_got.end(), -9);

        fern::core::offset(execution_policy, values, offset, result_we_got);

        BOOST_CHECK(equal(execution_policy, result_we_got, result_we_want));
    }

    {
        fern::Point<int, 1> offset{-2};

        // 2, 3, ..., -9, -9
        std::iota(result_we_want.begin(),
            result_we_want.end() + fern::get<0>(offset),
            std::abs(fern::get<0>(offset)));
        std::fill(result_we_want.end() + fern::get<0>(offset),
            result_we_want.end(), -9);
        std::fill(result_we_got.begin(), result_we_got.end(), -9);

        fern::core::offset(execution_policy, values, offset, result_we_got);

        BOOST_CHECK(equal(execution_policy, result_we_got, result_we_want));
    }

    {
        fern::Point<int, 1> offset{-static_cast<int>(nr_elements + 10)};

        // -9, -9, ..., -9, -9
        std::fill(result_we_want.begin(), result_we_want.end(), -9);
        std::fill(result_we_got.begin(), result_we_got.end(), -9);

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
    size_t const nr_elements{10 * nr_threads};
    fern::MaskedArray<int, 1> values(nr_elements);
    fern::MaskedArray<int, 1> result_we_want(nr_elements);
    fern::MaskedArray<int, 1> result_we_got(nr_elements);

    fern::DetectNoDataByValue<fern::Mask<1>> input_no_data_policy(
        values.mask(), true);
    fern::MarkNoDataByValue<fern::Mask<1>> output_no_data_policy(
        result_we_got.mask(), true);

    std::iota(values.data(), values.data() + nr_elements, 0);
    values.mask()[1] = true;

    {
        fern::Point<int, 1> offset{0};

        // 0, -9, 2, ..., n - 1
        std::iota(result_we_want.data(), result_we_want.data() + nr_elements,
            0);
        result_we_want[1] = -9;
        result_we_want.mask()[1] = true;
        result_we_got.fill(-9);
        result_we_got.mask().fill(false);

        fern::core::offset(input_no_data_policy, output_no_data_policy,
            execution_policy, values, offset, result_we_got);

        BOOST_CHECK(equal(execution_policy, result_we_got, result_we_want));
    }

    {
        fern::Point<int, 1> offset{2};

        // -9, -9, 0, -9, 1, ..., n - 3
        std::iota(result_we_want.data() + fern::get<0>(offset),
            result_we_want.data() + nr_elements, 0);
        std::fill(result_we_want.data(), result_we_want.data() +
            fern::get<0>(offset), -9);
        result_we_want.data()[fern::get<0>(offset) + 1] = -9;
        std::fill(result_we_want.mask().data(), result_we_want.mask().data() +
            fern::get<0>(offset), true);
        result_we_want.mask()[fern::get<0>(offset) + 1] = true;
        result_we_got.fill(-9);
        result_we_got.mask().fill(false);

        fern::core::offset(input_no_data_policy, output_no_data_policy,
            execution_policy, values, offset, result_we_got);

        BOOST_CHECK(equal(execution_policy, result_we_got, result_we_want));
    }

    {
        fern::Point<int, 1> offset{static_cast<int>(nr_elements + 10)};

        // -9, -9, ..., -9, -9
        result_we_want.fill(-9);
        result_we_want.mask().fill(true);
        result_we_got.fill(-9);
        result_we_got.mask().fill(false);

        fern::core::offset(input_no_data_policy, output_no_data_policy,
            execution_policy, values, offset, result_we_got);

        BOOST_CHECK(equal(execution_policy, result_we_got, result_we_want));
    }

    {
        fern::Point<int, 1> offset{-2};

        // 2, 3, ..., n - 1, -9, -9
        std::iota(result_we_want.data(), result_we_want.data() + nr_elements +
            fern::get<0>(offset), std::abs(fern::get<0>(offset)));
        std::fill(result_we_want.data() + nr_elements + fern::get<0>(offset),
            result_we_want.data() + nr_elements, -9);
        // false, false, ..., false, true, true
        std::fill(result_we_want.mask().data(), result_we_want.mask().data() +
            nr_elements + fern::get<0>(offset), false);
        std::fill(result_we_want.mask().data() + nr_elements +
            fern::get<0>(offset), result_we_want.mask().data() + nr_elements,
            true);
        result_we_got.fill(-9);
        result_we_got.mask().fill(false);

        fern::core::offset(input_no_data_policy, output_no_data_policy,
            execution_policy, values, offset, result_we_got);

        BOOST_CHECK(equal(execution_policy, result_we_got, result_we_want));
    }

    {
        fern::Point<int, 1> offset{-static_cast<int>(nr_elements + 10)};

        // -9, -9, ..., -9, -9
        result_we_want.fill(-9);
        // true, true, ..., true, true
        result_we_want.mask().fill(true);
        result_we_got.fill(-9);
        result_we_got.mask().fill(false);

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


void test_array_1d_fill_value(
    fern::ExecutionPolicy const& execution_policy)
{
    size_t const nr_threads{fern::ThreadClient::hardware_concurrency()};
    size_t const nr_elements{10 * nr_threads};
    std::vector<int> values(nr_elements);
    std::vector<int> result_we_want(nr_elements);
    std::vector<int> result_we_got(nr_elements);
    int const fill_value{5};

    std::iota(values.begin(), values.end(), 0);

    {
        fern::Point<int, 1> offset{0};

        // 0, 1, ..., n - 1
        std::iota(result_we_want.begin(), result_we_want.end(), 0);
        std::fill(result_we_got.begin(), result_we_got.end(), -9);

        fern::core::offset(execution_policy, values, offset, fill_value,
            result_we_got);

        BOOST_CHECK(equal(execution_policy, result_we_got, result_we_want));
    }

    {
        fern::Point<int, 1> offset{2};

        // 5, 5, 0, 1, ..., n - 3
        std::iota(result_we_want.begin() + fern::get<0>(offset),
            result_we_want.end(), 0);
        std::fill(result_we_want.begin(), result_we_want.begin() +
            fern::get<0>(offset), fill_value);
        std::fill(result_we_got.begin(), result_we_got.end(), -9);

        fern::core::offset(execution_policy, values, offset, fill_value,
            result_we_got);

        BOOST_CHECK(equal(execution_policy, result_we_got, result_we_want));
    }

    {
        fern::Point<int, 1> offset{static_cast<int>(nr_elements) + 10};

        // 5, 5, ..., 5, 5
        std::fill(result_we_want.begin(), result_we_want.end(), fill_value);
        std::fill(result_we_got.begin(), result_we_got.end(), -9);

        fern::core::offset(execution_policy, values, offset, fill_value,
            result_we_got);

        BOOST_CHECK(equal(execution_policy, result_we_got, result_we_want));
    }

    {
        fern::Point<int, 1> offset{-2};

        // 2, 3, ..., 5, 5
        std::iota(result_we_want.begin(),
            result_we_want.end() + fern::get<0>(offset),
            std::abs(fern::get<0>(offset)));
        std::fill(result_we_want.end() + fern::get<0>(offset),
            result_we_want.end(), fill_value);
        std::fill(result_we_got.begin(), result_we_got.end(), -9);

        fern::core::offset(execution_policy, values, offset, fill_value,
            result_we_got);

        BOOST_CHECK(equal(execution_policy, result_we_got, result_we_want));
    }

    {
        fern::Point<int, 1> offset{-static_cast<int>(nr_elements + 10)};

        // 5, 5, ..., 5, 5
        std::fill(result_we_want.begin(), result_we_want.end(), fill_value);
        std::fill(result_we_got.begin(), result_we_got.end(), -9);

        fern::core::offset(execution_policy, values, offset, fill_value,
            result_we_got);

        BOOST_CHECK(equal(execution_policy, result_we_got, result_we_want));
    }
}


BOOST_AUTO_TEST_CASE(array_1d_fill_value_sequential)
{
    test_array_1d_fill_value(fern::sequential);
}


BOOST_AUTO_TEST_CASE(array_1d_fill_value_parallel)
{
    fern::ThreadClient client;
    test_array_1d_fill_value(fern::parallel);
}


void test_array_1d_fill_value_masked(
    fern::ExecutionPolicy const& execution_policy)
{
    size_t const nr_threads{fern::ThreadClient::hardware_concurrency()};
    size_t const nr_elements{10 * nr_threads};
    fern::MaskedArray<int, 1> values(nr_elements);
    fern::MaskedArray<int, 1> result_we_want(nr_elements);
    fern::MaskedArray<int, 1> result_we_got(nr_elements);
    int const fill_value{5};

    fern::DetectNoDataByValue<fern::Mask<1>> input_no_data_policy(
        values.mask(), true);
    fern::MarkNoDataByValue<fern::Mask<1>> output_no_data_policy(
        result_we_got.mask(), true);

    std::iota(values.data(), values.data() + nr_elements, 0);
    values.mask()[1] = true;

    {
        fern::Point<int, 1> offset{0};

        // 0, -9, 2, ..., n - 1
        std::iota(result_we_want.data(), result_we_want.data() + nr_elements, 0);
        result_we_want.data()[1] = -9;
        result_we_want.mask()[1] = true;
        result_we_got.fill(-9);
        result_we_got.mask().fill(false);

        fern::core::offset(input_no_data_policy, output_no_data_policy,
            execution_policy, values, offset, fill_value, result_we_got);

        BOOST_CHECK(equal(execution_policy, result_we_got, result_we_want));
    }

    {
        fern::Point<int, 1> offset{2};

        // 5, 5, 0, -9, 1, ..., n - 3
        std::iota(result_we_want.data() + fern::get<0>(offset),
            result_we_want.data() + nr_elements, 0);
        std::fill(result_we_want.data(), result_we_want.data() +
            fern::get<0>(offset), fill_value);
        result_we_want.data()[fern::get<0>(offset) + 1] = -9;
        // false, false, false, true, false, ..., false
        std::fill(result_we_want.mask().data(), result_we_want.mask().data() +
            fern::get<0>(offset), false);
        result_we_want.mask()[fern::get<0>(offset) + 1] = true;
        result_we_got.fill(-9);
        result_we_got.mask().fill(false);

        fern::core::offset(input_no_data_policy, output_no_data_policy,
            execution_policy, values, offset, fill_value, result_we_got);

        BOOST_CHECK(equal(execution_policy, result_we_got, result_we_want));
    }

    {
        fern::Point<int, 1> offset{static_cast<int>(nr_elements + 10)};

        // 5, 5, ..., 5, 5
        result_we_want.fill(fill_value);
        // false, ..., false
        result_we_want.mask().fill(false);
        result_we_got.fill(-9);
        result_we_got.mask().fill(false);

        fern::core::offset(input_no_data_policy, output_no_data_policy,
            execution_policy, values, offset, fill_value, result_we_got);

        BOOST_CHECK(equal(execution_policy, result_we_got, result_we_want));
    }

    {
        fern::Point<int, 1> offset{-2};

        // 2, 3, ..., n - 1, 5, 5
        std::iota(result_we_want.data(), result_we_want.data() + nr_elements +
            fern::get<0>(offset), std::abs(fern::get<0>(offset)));
        std::fill(result_we_want.data() + nr_elements + fern::get<0>(offset),
            result_we_want.data() + nr_elements, fill_value);
        // false, false, ..., false, false, false
        std::fill(result_we_want.mask().data(), result_we_want.mask().data() +
            nr_elements, false);
        result_we_got.fill(-9);
        result_we_got.mask().fill(false);

        fern::core::offset(input_no_data_policy, output_no_data_policy,
            execution_policy, values, offset, fill_value, result_we_got);

        BOOST_CHECK(equal(execution_policy, result_we_got, result_we_want));
    }

    {
        fern::Point<int, 1> offset{-static_cast<int>(nr_elements + 10)};

        // 5, 5, ..., 5, 5
        result_we_want.fill(fill_value);
        // false, false, ..., false, false
        result_we_want.mask().fill(false);
        result_we_got.fill(-9);
        result_we_got.mask().fill(false);

        fern::core::offset(input_no_data_policy, output_no_data_policy,
            execution_policy, values, offset, fill_value, result_we_got);

        BOOST_CHECK(equal(execution_policy, result_we_got, result_we_want));
    }
}


BOOST_AUTO_TEST_CASE(array_1d_fill_value_masked_sequential)
{
    test_array_1d_fill_value_masked(fern::sequential);
}


BOOST_AUTO_TEST_CASE(array_1d_fill_value_masked_parallel)
{
    fern::ThreadClient client;
    test_array_1d_fill_value_masked(fern::parallel);
}


void test_array_2d(
    fern::ExecutionPolicy const& execution_policy)
{
    size_t const nr_threads{fern::ThreadClient::hardware_concurrency()};
    size_t const nr_rows{30 * nr_threads};
    size_t const nr_cols{20 * nr_threads};
    size_t const nr_elements{nr_rows * nr_cols};

    fern::Array<int, 2> values(fern::extents[nr_rows][nr_cols]);
    fern::Array<int, 2> result_we_want(fern::extents[nr_rows][nr_cols]);
    fern::Array<int, 2> result_we_got(fern::extents[nr_rows][nr_cols]);

    std::iota(values.data(), values.data() + nr_elements, 0);

    {
        fern::Point<int, 2> offset(0, 0);

        // 0, 1, ..., n - 1
        std::iota(result_we_want.data(), result_we_want.data() + nr_elements,
            0);
        std::fill(result_we_got.data(), result_we_got.data() + nr_elements, -9);

        fern::core::offset(execution_policy, values, offset, result_we_got);

        BOOST_CHECK(equal(execution_policy, result_we_got, result_we_want));
    }

    // TODO Add tests.
}


BOOST_AUTO_TEST_CASE(array_2d_sequential)
{
    test_array_2d(fern::sequential);
}


BOOST_AUTO_TEST_CASE(array_2d_parallel)
{
    fern::ThreadClient client;
    test_array_2d(fern::parallel);
}


void test_array_2d_masked(
    fern::ExecutionPolicy const& execution_policy)
{
    size_t const nr_threads{fern::ThreadClient::hardware_concurrency()};
    size_t const nr_rows{30 * nr_threads};
    size_t const nr_cols{20 * nr_threads};
    size_t const nr_elements{nr_rows * nr_cols};

    fern::MaskedArray<int, 2> values(fern::extents[nr_rows][nr_cols]);
    fern::MaskedArray<int, 2> result_we_want(fern::extents[nr_rows][nr_cols]);
    fern::MaskedArray<int, 2> result_we_got(fern::extents[nr_rows][nr_cols]);

    fern::DetectNoDataByValue<fern::Mask<2>> input_no_data_policy(
        values.mask(), true);
    fern::MarkNoDataByValue<fern::Mask<2>> output_no_data_policy(
        result_we_got.mask(), true);

    std::iota(values.data(), values.data() + nr_elements, 0);
    values.mask()[1][2] = true;

    {
        fern::Point<int, 2> offset(0, 0);

        // 0, -9, 2, ..., n - 1
        std::iota(result_we_want.data(), result_we_want.data() + nr_elements,
            0);
        result_we_want[1][2] = -9;
        result_we_want.mask()[1][2] = true;
        result_we_got.fill(-9);
        result_we_got.mask().fill(false);

        fern::core::offset(input_no_data_policy, output_no_data_policy,
            execution_policy, values, offset, result_we_got);

        BOOST_CHECK(equal(execution_policy, result_we_got, result_we_want));
    }

    // TODO Add tests.
}


BOOST_AUTO_TEST_CASE(array_2d_masked_sequential)
{
    test_array_2d_masked(fern::sequential);
}


BOOST_AUTO_TEST_CASE(array_2d_masked_parallel)
{
    fern::ThreadClient client;
    test_array_2d_masked(fern::parallel);
}


void test_array_2d_fill_value(
    fern::ExecutionPolicy const& execution_policy)
{
    size_t const nr_threads{fern::ThreadClient::hardware_concurrency()};
    size_t const nr_rows{30 * nr_threads};
    size_t const nr_cols{20 * nr_threads};
    size_t const nr_elements{nr_rows * nr_cols};

    fern::Array<int, 2> values(fern::extents[nr_rows][nr_cols]);
    fern::Array<int, 2> result_we_want(fern::extents[nr_rows][nr_cols]);
    fern::Array<int, 2> result_we_got(fern::extents[nr_rows][nr_cols]);

    int const fill_value{5};

    std::iota(values.data(), values.data() + nr_elements, 0);

    {
        fern::Point<int, 2> offset(0, 0);

        // 0, 1, ..., n - 1
        std::iota(result_we_want.data(), result_we_want.data() + nr_elements,
            0);
        std::fill(result_we_got.data(), result_we_got.data() + nr_elements, -9);

        fern::core::offset(execution_policy, values, offset, fill_value,
            result_we_got);

        BOOST_CHECK(equal(execution_policy, result_we_got, result_we_want));
    }

    // TODO Add tests.
}


BOOST_AUTO_TEST_CASE(array_2d_fill_value_sequential)
{
    test_array_2d_fill_value(fern::sequential);
}


BOOST_AUTO_TEST_CASE(array_2d_fill_value_parallel)
{
    fern::ThreadClient client;
    test_array_2d_fill_value(fern::parallel);
}


void test_array_2d_fill_value_masked(
    fern::ExecutionPolicy const& execution_policy)
{
    size_t const nr_threads{fern::ThreadClient::hardware_concurrency()};
    size_t const nr_rows{30 * nr_threads};
    size_t const nr_cols{20 * nr_threads};
    size_t const nr_elements{nr_rows * nr_cols};

    fern::MaskedArray<int, 2> values(fern::extents[nr_rows][nr_cols]);
    fern::MaskedArray<int, 2> result_we_want(fern::extents[nr_rows][nr_cols]);
    fern::MaskedArray<int, 2> result_we_got(fern::extents[nr_rows][nr_cols]);

    int const fill_value{5};

    fern::DetectNoDataByValue<fern::Mask<2>> input_no_data_policy(
        values.mask(), true);
    fern::MarkNoDataByValue<fern::Mask<2>> output_no_data_policy(
        result_we_got.mask(), true);

    std::iota(values.data(), values.data() + nr_elements, 0);
    values.mask()[1][2] = true;

    {
        fern::Point<int, 2> offset(0, 0);

        // 0, -9, 2, ..., n - 1
        std::iota(result_we_want.data(), result_we_want.data() + nr_elements,
            0);
        result_we_want[1][2] = -9;
        result_we_want.mask()[1][2] = true;
        result_we_got.fill(-9);
        result_we_got.mask().fill(false);

        fern::core::offset(input_no_data_policy, output_no_data_policy,
            execution_policy, values, offset, fill_value, result_we_got);

        BOOST_CHECK(equal(execution_policy, result_we_got, result_we_want));
    }

    // TODO Add tests.
}


BOOST_AUTO_TEST_CASE(array_2d_fill_value_masked_sequential)
{
    test_array_2d_fill_value_masked(fern::sequential);
}


BOOST_AUTO_TEST_CASE(array_2d_fill_value_masked_parallel)
{
    fern::ThreadClient client;
    test_array_2d_fill_value_masked(fern::parallel);
}


BOOST_AUTO_TEST_CASE(pcraster_example_1)
{
    // First example from the PCRaster manual page for shift/shift0.
    size_t const nr_rows{5};
    size_t const nr_cols{5};

    fern::MaskedArray<int, 2> values({
        {0,  -1, 1, -30,  0},
        {2, 999, 1,   2, -3},
        {3,   2, 3,   4,  2},
        {0,   0, 2,  40,  2},
        {1,  -2, 4,   7,  1}
    });
    values.mask()[1][1] = true;

    fern::MaskedArray<int, 2> result_we_want({
        {999,   1,   2,   -3, 999},
        {  2,   3,   4,    2, 999},
        {  0,   2,  40,    2, 999},
        { -2,   4,   7,    1, 999},
        {999, 999, 999,  999, 999}
    });
    result_we_want.mask()[0][0] = true;
    for(size_t row = 0; row < nr_rows; ++row) {
        result_we_want.mask()[row][nr_cols-1] = true;
    }
    for(size_t col = 0; col < nr_cols; ++col) {
        result_we_want.mask()[nr_rows-1][col] = true;
    }

    fern::MaskedArray<int, 2> result_we_got(fern::extents[nr_rows][nr_cols],
        999);

    fern::Point<int, 2> offset(-1, -1);

    fern::DetectNoDataByValue<fern::Mask<2>> input_no_data_policy(
        values.mask(), true);
    fern::MarkNoDataByValue<fern::Mask<2>> output_no_data_policy(
        result_we_got.mask(), true);

    fern::core::offset(input_no_data_policy, output_no_data_policy,
        fern::sequential, values, offset, result_we_got);

    BOOST_CHECK(equal(fern::sequential, result_we_got, result_we_want));
}


BOOST_AUTO_TEST_CASE(pcraster_example_2)
{
    // First example from the PCRaster manual page for shift/shift0.
    size_t const nr_rows{5};
    size_t const nr_cols{5};

    fern::MaskedArray<int, 2> values({
        {0,  -1, 1, -30,  0},
        {2, 999, 1,   2, -3},
        {3,   2, 3,   4,  2},
        {0,   0, 2,  40,  2},
        {1,  -2, 4,   7,  1}
    });
    values.mask()[1][1] = true;

    fern::MaskedArray<int, 2> result_we_want({
        {999, 999, 999, 999,  999},
        {999,   0,   -1,  1,  -30},
        {999,   2,  999,  1,    2},
        {999,   3,    2,  3,    4},
        {999,   0,    0,  2,   40}
    });
    result_we_want.mask()[2][2] = true;
    for(size_t row = 0; row < nr_rows; ++row) {
        result_we_want.mask()[row][0] = true;
    }
    for(size_t col = 0; col < nr_cols; ++col) {
        result_we_want.mask()[0][col] = true;
    }

    fern::MaskedArray<int, 2> result_we_got(fern::extents[nr_rows][nr_cols],
        999);

    fern::Point<int, 2> offset(1, 1);

    fern::DetectNoDataByValue<fern::Mask<2>> input_no_data_policy(
        values.mask(), true);
    fern::MarkNoDataByValue<fern::Mask<2>> output_no_data_policy(
        result_we_got.mask(), true);

    fern::core::offset(input_no_data_policy, output_no_data_policy,
        fern::sequential, values, offset, result_we_got);

    BOOST_CHECK(equal(fern::sequential, result_we_got, result_we_want));
}


BOOST_AUTO_TEST_CASE(pcraster_example_3)
{
    // First example from the PCRaster manual page for shift/shift0.
    size_t const nr_rows{5};
    size_t const nr_cols{5};

    fern::MaskedArray<int, 2> values({
        {0,  -1, 1, -30,  0},
        {2, 999, 1,   2, -3},
        {3,   2, 3,   4,  2},
        {0,   0, 2,  40,  2},
        {1,  -2, 4,   7,  1}
    });
    values.mask()[1][1] = true;

    fern::MaskedArray<int, 2> result_we_want({
        {0, 0,   0, 0,   0},
        {0, 0,  -1, 1, -30},
        {0, 2, 999, 1,   2},
        {0, 3,   2, 3,   4},
        {0, 0,   0, 2,  40}
    });
    result_we_want.mask()[2][2] = true;

    fern::MaskedArray<int, 2> result_we_got(fern::extents[nr_rows][nr_cols],
        999);

    fern::Point<int, 2> offset(1, 1);

    fern::DetectNoDataByValue<fern::Mask<2>> input_no_data_policy(
        values.mask(), true);
    fern::MarkNoDataByValue<fern::Mask<2>> output_no_data_policy(
        result_we_got.mask(), true);

    int const fill_value{0};

    fern::core::offset(input_no_data_policy, output_no_data_policy,
        fern::sequential, values, offset, fill_value, result_we_got);

    BOOST_CHECK(equal(fern::sequential, result_we_got, result_we_want));
}

BOOST_AUTO_TEST_SUITE_END()
