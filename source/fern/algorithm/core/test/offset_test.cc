// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern algorithm core offset
#include <boost/test/unit_test.hpp>
#include "fern/feature/core/data_customization_point/array.h"
#include "fern/feature/core/data_customization_point/masked_array.h"
#include "fern/core/data_customization_point/scalar.h"
#include "fern/core/data_customization_point/point.h"
#include "fern/core/data_customization_point/vector.h"
#include "fern/algorithm/core/offset.h"
#include "fern/algorithm/core/test/test_utils.h"


namespace fa = fern::algorithm;


BOOST_AUTO_TEST_SUITE(offset)

template<
    typename ExecutionPolicy>
void test_array_1d(
    ExecutionPolicy& execution_policy)
{
    size_t const nr_threads{fern::hardware_concurrency()};
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

        fa::core::offset(execution_policy, values, offset, result_we_got);

        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));
    }

    {
        fern::Point<int, 1> offset{2};

        // -9, -9, 0, 1, ..., n - 3
        std::iota(result_we_want.begin() + fern::get<0>(offset),
            result_we_want.end(), 0);
        std::fill(result_we_want.begin(), result_we_want.begin() +
            fern::get<0>(offset), -9);
        std::fill(result_we_got.begin(), result_we_got.end(), -9);

        fa::core::offset(execution_policy, values, offset, result_we_got);

        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));
    }

    {
        fern::Point<int, 1> offset{static_cast<int>(nr_elements) + 10};

        // -9, -9, ..., -9, -9
        std::fill(result_we_want.begin(), result_we_want.end(), -9);
        std::fill(result_we_got.begin(), result_we_got.end(), -9);

        fa::core::offset(execution_policy, values, offset, result_we_got);

        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));
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

        fa::core::offset(execution_policy, values, offset, result_we_got);

        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));
    }

    {
        fern::Point<int, 1> offset{-static_cast<int>(nr_elements + 10)};

        // -9, -9, ..., -9, -9
        std::fill(result_we_want.begin(), result_we_want.end(), -9);
        std::fill(result_we_got.begin(), result_we_got.end(), -9);

        fa::core::offset(execution_policy, values, offset, result_we_got);

        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));
    }
}


BOOST_AUTO_TEST_CASE(array_1d_sequential)
{
    test_array_1d(fa::sequential);
    fa::ExecutionPolicy execution_policy{fa::sequential};
    test_array_1d(execution_policy);
}


BOOST_AUTO_TEST_CASE(array_1d_parallel)
{
    test_array_1d(fa::parallel);
    fa::ExecutionPolicy execution_policy{fa::parallel};
    test_array_1d(execution_policy);
}


template<
    typename ExecutionPolicy>
void test_array_1d_masked(
    ExecutionPolicy& execution_policy)
{
    size_t const nr_threads{fern::hardware_concurrency()};
    size_t const nr_elements{10 * nr_threads};
    fern::MaskedArray<int, 1> values(nr_elements);
    fern::MaskedArray<int, 1> result_we_want(nr_elements);
    fern::MaskedArray<int, 1> result_we_got(nr_elements);

    fa::InputNoDataPolicies<fa::DetectNoDataByValue<fern::Mask<1>>>
        input_no_data_policy{{values.mask(), true}};
    fa::MarkNoDataByValue<fern::Mask<1>> output_no_data_policy(
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

        fa::core::offset(input_no_data_policy, output_no_data_policy,
            execution_policy, values, offset, result_we_got);

        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));
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

        fa::core::offset(input_no_data_policy, output_no_data_policy,
            execution_policy, values, offset, result_we_got);

        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));
    }

    {
        fern::Point<int, 1> offset{static_cast<int>(nr_elements + 10)};

        // -9, -9, ..., -9, -9
        result_we_want.fill(-9);
        result_we_want.mask().fill(true);
        result_we_got.fill(-9);
        result_we_got.mask().fill(false);

        fa::core::offset(input_no_data_policy, output_no_data_policy,
            execution_policy, values, offset, result_we_got);

        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));
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

        fa::core::offset(input_no_data_policy, output_no_data_policy,
            execution_policy, values, offset, result_we_got);

        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));
    }

    {
        fern::Point<int, 1> offset{-static_cast<int>(nr_elements + 10)};

        // -9, -9, ..., -9, -9
        result_we_want.fill(-9);
        // true, true, ..., true, true
        result_we_want.mask().fill(true);
        result_we_got.fill(-9);
        result_we_got.mask().fill(false);

        fa::core::offset(input_no_data_policy, output_no_data_policy,
            execution_policy, values, offset, result_we_got);

        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));
    }
}


BOOST_AUTO_TEST_CASE(array_1d_masked_sequential)
{
    test_array_1d_masked(fa::sequential);
    fa::ExecutionPolicy execution_policy{fa::sequential};
    test_array_1d_masked(execution_policy);
}


BOOST_AUTO_TEST_CASE(array_1d_masked_parallel)
{
    test_array_1d_masked(fa::parallel);
    fa::ExecutionPolicy execution_policy{fa::parallel};
    test_array_1d_masked(execution_policy);
}


template<
    typename ExecutionPolicy>
void test_array_1d_fill_value(
    ExecutionPolicy& execution_policy)
{
    size_t const nr_threads{fern::hardware_concurrency()};
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

        fa::core::offset(execution_policy, values, offset, fill_value,
            result_we_got);

        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));
    }

    {
        fern::Point<int, 1> offset{2};

        // 5, 5, 0, 1, ..., n - 3
        std::iota(result_we_want.begin() + fern::get<0>(offset),
            result_we_want.end(), 0);
        std::fill(result_we_want.begin(), result_we_want.begin() +
            fern::get<0>(offset), fill_value);
        std::fill(result_we_got.begin(), result_we_got.end(), -9);

        fa::core::offset(execution_policy, values, offset, fill_value,
            result_we_got);

        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));
    }

    {
        fern::Point<int, 1> offset{static_cast<int>(nr_elements) + 10};

        // 5, 5, ..., 5, 5
        std::fill(result_we_want.begin(), result_we_want.end(), fill_value);
        std::fill(result_we_got.begin(), result_we_got.end(), -9);

        fa::core::offset(execution_policy, values, offset, fill_value,
            result_we_got);

        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));
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

        fa::core::offset(execution_policy, values, offset, fill_value,
            result_we_got);

        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));
    }

    {
        fern::Point<int, 1> offset{-static_cast<int>(nr_elements + 10)};

        // 5, 5, ..., 5, 5
        std::fill(result_we_want.begin(), result_we_want.end(), fill_value);
        std::fill(result_we_got.begin(), result_we_got.end(), -9);

        fa::core::offset(execution_policy, values, offset, fill_value,
            result_we_got);

        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));
    }
}


BOOST_AUTO_TEST_CASE(array_1d_fill_value_sequential)
{
    test_array_1d_fill_value(fa::sequential);
    fa::ExecutionPolicy execution_policy{fa::sequential};
    test_array_1d_fill_value(execution_policy);
}


BOOST_AUTO_TEST_CASE(array_1d_fill_value_parallel)
{
    test_array_1d_fill_value(fa::parallel);
    fa::ExecutionPolicy execution_policy{fa::parallel};
    test_array_1d_fill_value(execution_policy);
}


template<
    typename ExecutionPolicy>
void test_array_1d_fill_value_masked(
    ExecutionPolicy& execution_policy)
{
    size_t const nr_threads{fern::hardware_concurrency()};
    size_t const nr_elements{10 * nr_threads};
    fern::MaskedArray<int, 1> values(nr_elements);
    fern::MaskedArray<int, 1> result_we_want(nr_elements);
    fern::MaskedArray<int, 1> result_we_got(nr_elements);
    int const fill_value{5};

    fa::InputNoDataPolicies<fa::DetectNoDataByValue<fern::Mask<1>>>
        input_no_data_policy{{values.mask(), true}};
    fa::MarkNoDataByValue<fern::Mask<1>> output_no_data_policy(
        result_we_got.mask(), true);

    std::iota(values.data(), values.data() + nr_elements, 0);
    values.mask()[1] = true;

    {
        fern::Point<int, 1> offset{0};

        // 0, -9, 2, ..., n - 1
        std::iota(result_we_want.data(), result_we_want.data() + nr_elements,
            0);
        result_we_want.data()[1] = -9;
        result_we_want.mask()[1] = true;
        result_we_got.fill(-9);
        result_we_got.mask().fill(false);

        fa::core::offset(input_no_data_policy, output_no_data_policy,
            execution_policy, values, offset, fill_value, result_we_got);

        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));
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

        fa::core::offset(input_no_data_policy, output_no_data_policy,
            execution_policy, values, offset, fill_value, result_we_got);

        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));
    }

    {
        fern::Point<int, 1> offset{static_cast<int>(nr_elements + 10)};

        // 5, 5, ..., 5, 5
        result_we_want.fill(fill_value);
        // false, ..., false
        result_we_want.mask().fill(false);
        result_we_got.fill(-9);
        result_we_got.mask().fill(false);

        fa::core::offset(input_no_data_policy, output_no_data_policy,
            execution_policy, values, offset, fill_value, result_we_got);

        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));
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

        fa::core::offset(input_no_data_policy, output_no_data_policy,
            execution_policy, values, offset, fill_value, result_we_got);

        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));
    }

    {
        fern::Point<int, 1> offset{-static_cast<int>(nr_elements + 10)};

        // 5, 5, ..., 5, 5
        result_we_want.fill(fill_value);
        // false, false, ..., false, false
        result_we_want.mask().fill(false);
        result_we_got.fill(-9);
        result_we_got.mask().fill(false);

        fa::core::offset(input_no_data_policy, output_no_data_policy,
            execution_policy, values, offset, fill_value, result_we_got);

        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));
    }
}


BOOST_AUTO_TEST_CASE(array_1d_fill_value_masked_sequential)
{
    test_array_1d_fill_value_masked(fa::sequential);
    fa::ExecutionPolicy execution_policy{fa::sequential};
    test_array_1d_fill_value_masked(execution_policy);
}


BOOST_AUTO_TEST_CASE(array_1d_fill_value_masked_parallel)
{
    test_array_1d_fill_value_masked(fa::parallel);
    fa::ExecutionPolicy execution_policy{fa::parallel};
    test_array_1d_fill_value_masked(execution_policy);
}


template<
    typename ExecutionPolicy>
void test_array_2d(
    ExecutionPolicy& execution_policy)
{
    size_t const nr_threads{fern::hardware_concurrency()};
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

        fa::core::offset(execution_policy, values, offset, result_we_got);

        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));
    }

    // TODO Add tests.
}


BOOST_AUTO_TEST_CASE(array_2d_sequential)
{
    test_array_2d(fa::sequential);
    fa::ExecutionPolicy execution_policy{fa::sequential};
    test_array_2d(execution_policy);
}


BOOST_AUTO_TEST_CASE(array_2d_parallel)
{
    test_array_2d(fa::parallel);
    fa::ExecutionPolicy execution_policy{fa::parallel};
    test_array_2d(execution_policy);
}


template<
    typename ExecutionPolicy>
void test_array_2d_masked(
    ExecutionPolicy& execution_policy)
{
    size_t const nr_threads{fern::hardware_concurrency()};
    size_t const nr_rows{30 * nr_threads};
    size_t const nr_cols{20 * nr_threads};
    size_t const nr_elements{nr_rows * nr_cols};

    fern::MaskedArray<int, 2> values(fern::extents[nr_rows][nr_cols]);
    fern::MaskedArray<int, 2> result_we_want(fern::extents[nr_rows][nr_cols]);
    fern::MaskedArray<int, 2> result_we_got(fern::extents[nr_rows][nr_cols]);

    fa::InputNoDataPolicies<fa::DetectNoDataByValue<fern::Mask<2>>>
        input_no_data_policy{{values.mask(), true}};
    fa::MarkNoDataByValue<fern::Mask<2>> output_no_data_policy(
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

        fa::core::offset(input_no_data_policy, output_no_data_policy,
            execution_policy, values, offset, result_we_got);

        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));
    }

    // TODO Add tests.
}


BOOST_AUTO_TEST_CASE(array_2d_masked_sequential)
{
    test_array_2d_masked(fa::sequential);
    fa::ExecutionPolicy execution_policy{fa::sequential};
    test_array_2d_masked(execution_policy);
}


BOOST_AUTO_TEST_CASE(array_2d_masked_parallel)
{
    test_array_2d_masked(fa::parallel);
    fa::ExecutionPolicy execution_policy{fa::parallel};
    test_array_2d_masked(execution_policy);
}


template<
    typename ExecutionPolicy>
void test_array_2d_fill_value(
    ExecutionPolicy& execution_policy)
{
    size_t const nr_threads{fern::hardware_concurrency()};
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

        fa::core::offset(execution_policy, values, offset, fill_value,
            result_we_got);

        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));
    }

    // TODO Add tests.
}


BOOST_AUTO_TEST_CASE(array_2d_fill_value_sequential)
{
    test_array_2d_fill_value(fa::sequential);
    fa::ExecutionPolicy execution_policy{fa::sequential};
    test_array_2d_fill_value(execution_policy);
}


BOOST_AUTO_TEST_CASE(array_2d_fill_value_parallel)
{
    test_array_2d_fill_value(fa::parallel);
    fa::ExecutionPolicy execution_policy{fa::parallel};
    test_array_2d_fill_value(execution_policy);
}


template<
    typename ExecutionPolicy>
void test_array_2d_fill_value_masked(
    ExecutionPolicy& execution_policy)
{
    size_t const nr_threads{fern::hardware_concurrency()};
    size_t const nr_rows{30 * nr_threads};
    size_t const nr_cols{20 * nr_threads};
    size_t const nr_elements{nr_rows * nr_cols};

    fern::MaskedArray<int, 2> values(fern::extents[nr_rows][nr_cols]);
    fern::MaskedArray<int, 2> result_we_want(fern::extents[nr_rows][nr_cols]);
    fern::MaskedArray<int, 2> result_we_got(fern::extents[nr_rows][nr_cols]);

    int const fill_value{5};

    fa::InputNoDataPolicies<fa::DetectNoDataByValue<fern::Mask<2>>>
        input_no_data_policy{{values.mask(), true}};
    fa::MarkNoDataByValue<fern::Mask<2>> output_no_data_policy(
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

        fa::core::offset(input_no_data_policy, output_no_data_policy,
            execution_policy, values, offset, fill_value, result_we_got);

        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));
    }

    // TODO Add tests.
}


BOOST_AUTO_TEST_CASE(array_2d_fill_value_masked_sequential)
{
    test_array_2d_fill_value_masked(fa::sequential);
    fa::ExecutionPolicy execution_policy{fa::sequential};
    test_array_2d_fill_value_masked(execution_policy);
}


BOOST_AUTO_TEST_CASE(array_2d_fill_value_masked_parallel)
{
    test_array_2d_fill_value_masked(fa::parallel);
    fa::ExecutionPolicy execution_policy{fa::parallel};
    test_array_2d_fill_value_masked(execution_policy);
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

    fa::InputNoDataPolicies<fa::DetectNoDataByValue<fern::Mask<2>>>
        input_no_data_policy{{values.mask(), true}};
    fa::MarkNoDataByValue<fern::Mask<2>> output_no_data_policy(
        result_we_got.mask(), true);

    fa::core::offset(input_no_data_policy, output_no_data_policy,
        fa::sequential, values, offset, result_we_got);

    BOOST_CHECK(fern::compare(fa::sequential, result_we_got,
        result_we_want));
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

    fa::InputNoDataPolicies<fa::DetectNoDataByValue<fern::Mask<2>>>
        input_no_data_policy{{values.mask(), true}};
    fa::MarkNoDataByValue<fern::Mask<2>> output_no_data_policy(
        result_we_got.mask(), true);

    fa::core::offset(input_no_data_policy, output_no_data_policy,
        fa::sequential, values, offset, result_we_got);

    BOOST_CHECK(fern::compare(fa::sequential, result_we_got,
        result_we_want));
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

    fa::InputNoDataPolicies<fa::DetectNoDataByValue<fern::Mask<2>>>
        input_no_data_policy{{values.mask(), true}};
    fa::MarkNoDataByValue<fern::Mask<2>> output_no_data_policy(
        result_we_got.mask(), true);

    int const fill_value{0};

    fa::core::offset(input_no_data_policy, output_no_data_policy,
        fa::sequential, values, offset, fill_value, result_we_got);

    BOOST_CHECK(fern::compare(fa::sequential, result_we_got,
        result_we_want));
}

BOOST_AUTO_TEST_SUITE_END()
