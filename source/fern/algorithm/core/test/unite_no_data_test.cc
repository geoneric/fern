// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern algorithm core unite_no_data
#include <boost/test/unit_test.hpp>
#include "fern/core/data_customization_point/scalar.h"
#include "fern/feature/core/data_customization_point/array.h"
#include "fern/feature/core/data_customization_point/masked_array.h"
#include "fern/feature/core/data_customization_point/masked_scalar.h"
#include "fern/algorithm/core/unite_no_data.h"
#include "fern/algorithm/core/test/test_utils.h"


namespace fa = fern::algorithm;


template<
    typename ExecutionPolicy>
void test_array_0d_0d(
    ExecutionPolicy& execution_policy)
{
    int value1;
    int value2;
    int result_we_want;
    int result_we_got;

    {
        value1 = 5;
        value2 = 6;

        result_we_want = -9;
        result_we_got = -9;
        fa::core::unite_no_data(execution_policy, value1, value2,
            result_we_got);
        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));

        result_we_want = -9;
        result_we_got = -9;
        fa::core::unite_no_data(execution_policy, value1, value2,
            result_we_got);
        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));
    }
}


BOOST_AUTO_TEST_CASE(array_0d_0d_sequential)
{
    fa::SequentialExecutionPolicy sequential;

    test_array_0d_0d(sequential);
    fa::ExecutionPolicy execution_policy{sequential};
    test_array_0d_0d(execution_policy);
}


BOOST_AUTO_TEST_CASE(array_0d_0d_parallel)
{
    fa::ParallelExecutionPolicy parallel;

    test_array_0d_0d(parallel);
    fa::ExecutionPolicy execution_policy{parallel};
    test_array_0d_0d(execution_policy);
}


template<
    typename ExecutionPolicy>
void test_array_0d_0d_masked(
    ExecutionPolicy& execution_policy)
{
    fern::MaskedScalar<int> value1;
    fern::MaskedScalar<int> value2;
    fern::MaskedScalar<int> result_we_want;
    fern::MaskedScalar<int> result_we_got;

    fa::InputNoDataPolicies<
        fa::DetectNoDataByValue<bool>,
        fa::DetectNoDataByValue<bool>> input_no_data_policy{
            {value1.mask(), true},
            {value2.mask(), true}};
    fa::MarkNoDataByValue<bool> output_no_data_policy(result_we_got.mask(),
        true);

    // unite(false, false) -> false
    {
        value1 = 5;
        value1.mask() = false;
        value2 = 6;
        value2.mask() = false;

        result_we_want = fern::MaskedScalar<int>(-9, false);
        result_we_got = -9;
        result_we_got.mask() = false;
        fa::core::unite_no_data(input_no_data_policy, output_no_data_policy,
            execution_policy, value1, value2, result_we_got);
        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));
    }

    // unite(true, false) -> true
    {
        value1 = 5;
        value1.mask() = true;
        value2 = 6;
        value2.mask() = false;

        result_we_want = fern::MaskedScalar<int>(-9, true);
        result_we_got = -9;
        result_we_got.mask() = false;
        fa::core::unite_no_data(input_no_data_policy, output_no_data_policy,
            execution_policy, value1, value2, result_we_got);
        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));
    }

    // unite(false, true) -> true
    {
        value1 = 5;
        value1.mask() = false;
        value2 = 6;
        value2.mask() = true;

        result_we_want = fern::MaskedScalar<int>(-9, true);
        result_we_got = -9;
        result_we_got.mask() = false;
        fa::core::unite_no_data(input_no_data_policy, output_no_data_policy,
            execution_policy, value1, value2, result_we_got);
        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));
    }

    // unite(true, true) -> true
    {
        value1 = 5;
        value1.mask() = true;
        value2 = 6;
        value2.mask() = true;

        result_we_want = fern::MaskedScalar<int>(-9, true);
        result_we_got = -9;
        result_we_got.mask() = false;
        fa::core::unite_no_data(input_no_data_policy, output_no_data_policy,
            execution_policy, value1, value2, result_we_got);
        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));
    }
}


BOOST_AUTO_TEST_CASE(array_0d_0d_masked_sequential)
{
    fa::SequentialExecutionPolicy sequential;

    test_array_0d_0d_masked(sequential);
    fa::ExecutionPolicy execution_policy{sequential};
    test_array_0d_0d_masked(execution_policy);
}


BOOST_AUTO_TEST_CASE(array_0d_0d_masked_parallel)
{
    fa::ParallelExecutionPolicy parallel;

    test_array_0d_0d_masked(parallel);
    fa::ExecutionPolicy execution_policy{parallel};
    test_array_0d_0d_masked(execution_policy);
}


// TODO 1d


template<
    typename ExecutionPolicy>
void test_array_2d_2d(
    ExecutionPolicy& execution_policy)
{
    size_t const nr_threads{fern::hardware_concurrency()};
    size_t const nr_rows{30 * nr_threads};
    size_t const nr_cols{20 * nr_threads};
    size_t const nr_elements{nr_rows * nr_cols};

    fern::Array<int, 2> value1(fern::extents[nr_rows][nr_cols]);
    fern::Array<int, 2> value2(fern::extents[nr_rows][nr_cols]);
    fern::Array<int, 2> result_we_got(fern::extents[nr_rows][nr_cols]);

    // Fill value1: 0, 1, 2, ...
    std::iota(value1.data(), value1.data() + nr_elements, 0);

    // Fill value2: 10, 11, 12, ...
    std::iota(value2.data(), value2.data() + nr_elements, 10);

    // Fill result_we_want.
    fern::Array<int, 2> result_we_want(fern::extents[nr_rows][nr_cols]);
    result_we_want.fill(-9);

    result_we_got.fill(-9);
    fa::core::unite_no_data(execution_policy, value1, value2, result_we_got);
    BOOST_CHECK(fern::compare(execution_policy, result_we_got, result_we_want));
}


BOOST_AUTO_TEST_CASE(array_2d_2d_sequential)
{
    fa::SequentialExecutionPolicy sequential;

    test_array_2d_2d(sequential);
    fa::ExecutionPolicy execution_policy{sequential};
    test_array_2d_2d(execution_policy);
}


BOOST_AUTO_TEST_CASE(array_2d_2d_parallel)
{
    fa::ParallelExecutionPolicy parallel;

    test_array_2d_2d(parallel);
    fa::ExecutionPolicy execution_policy{parallel};
    test_array_2d_2d(execution_policy);
}


template<
    typename ExecutionPolicy>
void test_array_2d_2d_masked(
    ExecutionPolicy& execution_policy)
{
    size_t const nr_threads{fern::hardware_concurrency()};
    size_t const nr_rows{3 * nr_threads};
    size_t const nr_cols{2 * nr_threads};
    size_t const nr_elements{nr_rows * nr_cols};

    fern::MaskedArray<int, 2> value1(fern::extents[nr_rows][nr_cols]);
    fern::MaskedArray<int, 2> value2(fern::extents[nr_rows][nr_cols]);
    fern::MaskedArray<int, 2> result_we_got(fern::extents[nr_rows][nr_cols]);

    fa::InputNoDataPolicies<
        fa::DetectNoDataByValue<fern::Mask<2>>,
        fa::DetectNoDataByValue<fern::Mask<2>>> input_no_data_policy{
            {value1.mask(), true},
            {value2.mask(), true}};
    fa::MarkNoDataByValue<fern::Mask<2>> output_no_data_policy(
        result_we_got.mask(), true);

    // Fill value1: 0, 1, 2, ...
    // Mask all cells with index % 4.
    std::iota(value1.data(), value1.data() + nr_elements, 0);
    {
        int n = 0;
        std::generate(value1.mask().data(), value1.mask().data() +
            nr_elements, [&] () { return n++ % 4 == 0 ? true : false; });
    }

    // Fill value2: 10, 11, 12, ...
    // Mask all cells with index % 7.
    std::iota(value2.data(), value2.data() + nr_elements, 10);
    {
        int n = 0;
        std::generate(value2.mask().data(), value2.mask().data() +
            nr_elements, [&] () { return n++ % 7 == 0 ? true : false; });
    }

    // Fill result_we_want.
    // Mask all cells for which value1 or value2 are masked.
    fern::MaskedArray<int, 2> result_we_want(fern::extents[nr_rows][nr_cols]);
    result_we_want.fill(-9);
    {
        auto value1_mask_it = value1.mask().data();
        auto value2_mask_it = value2.mask().data();
        std::transform(result_we_want.data(),
            result_we_want.data() + nr_elements, result_we_want.mask().data(),
            [&](int const& /* value */) {
                ++value1_mask_it;
                ++value2_mask_it;
                return *(value1_mask_it-1) || *(value2_mask_it-1); });
    }

    result_we_got.fill(-9);
    fa::core::unite_no_data(input_no_data_policy, output_no_data_policy,
        execution_policy, value1, value2, result_we_got);
    BOOST_CHECK(fern::compare(execution_policy, result_we_got, result_we_want));
}


BOOST_AUTO_TEST_CASE(array_2d_2d_masked_sequential)
{
    fa::SequentialExecutionPolicy sequential;

    test_array_2d_2d_masked(sequential);
    fa::ExecutionPolicy execution_policy{sequential};
    test_array_2d_2d_masked(execution_policy);
}


BOOST_AUTO_TEST_CASE(array_2d_2d_masked_parallel)
{
    fa::ParallelExecutionPolicy parallel;

    test_array_2d_2d_masked(parallel);
    fa::ExecutionPolicy execution_policy{parallel};
    test_array_2d_2d_masked(execution_policy);
}
