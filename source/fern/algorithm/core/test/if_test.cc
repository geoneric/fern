// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern algorithm core if_
#include <boost/test/unit_test.hpp>
#include "fern/core/data_customization_point/scalar.h"
#include "fern/feature/core/data_customization_point/array.h"
#include "fern/feature/core/data_customization_point/masked_array.h"
#include "fern/feature/core/data_customization_point/masked_scalar.h"
#include "fern/algorithm/algebra/boole/or.h"
#include "fern/algorithm/core/if.h"
#include "fern/algorithm/core/test/test_utils.h"


namespace fa = fern::algorithm;


template<
    typename ExecutionPolicy>
void test_array_0d_2d(
    ExecutionPolicy& execution_policy)
{
    size_t const nr_threads{fern::hardware_concurrency()};
    size_t const nr_rows{30 * nr_threads};
    size_t const nr_cols{20 * nr_threads};
    size_t const nr_elements{nr_rows * nr_cols};

    int condition;
    fern::Array<int, 2> true_value(fern::extents[nr_rows][nr_cols]);
    fern::Array<int, 2> result_we_want(fern::extents[nr_rows][nr_cols]);
    fern::Array<int, 2> result_we_got(fern::extents[nr_rows][nr_cols]);

    // Fill true_value: 0, 1, 2, ...
    std::iota(true_value.data(), true_value.data() + nr_elements, 0);

    {
        condition = 0;

        // Condition is false: result_we_got is not touched.
        result_we_want.fill(-9);
        result_we_got.fill(-9);

        fa::core::if_(execution_policy, condition, true_value, result_we_got);
        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));
    }

    {
        condition = 1;

        // Fill result_we_want: copy true_value.
        std::copy(true_value.data(), true_value.data() + nr_elements,
            result_we_want.data());

        result_we_got.fill(-9);
        fa::core::if_(execution_policy, condition, true_value, result_we_got);
        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));
    }

}


BOOST_AUTO_TEST_CASE(array_0d_2d_sequential)
{
    fa::SequentialExecutionPolicy sequential;

    test_array_0d_2d(sequential);
    fa::ExecutionPolicy execution_policy{sequential};
    test_array_0d_2d(execution_policy);
}


BOOST_AUTO_TEST_CASE(array_0d_2d_parallel)
{
    fa::ParallelExecutionPolicy parallel;

    test_array_0d_2d(parallel);
    fa::ExecutionPolicy execution_policy{parallel};
    test_array_0d_2d(execution_policy);
}


template<
    typename ExecutionPolicy>
void test_array_0d_2d_masked(
    ExecutionPolicy& execution_policy)
{
    size_t const nr_threads{fern::hardware_concurrency()};
    size_t const nr_rows{30 * nr_threads};
    size_t const nr_cols{20 * nr_threads};
    size_t const nr_elements{nr_rows * nr_cols};

    fern::MaskedScalar<int> condition;
    fern::MaskedArray<int, 2> true_value(fern::extents[nr_rows][nr_cols]);
    fern::MaskedArray<int, 2> result_we_want(fern::extents[nr_rows][nr_cols]);
    fern::MaskedArray<int, 2> result_we_got(fern::extents[nr_rows][nr_cols]);

    fa::InputNoDataPolicies<fa::DetectNoDataByValue<bool>,
        fa::DetectNoDataByValue<fern::Mask<2>>> input_no_data_policy{
            {condition.mask(), true},
            {true_value.mask(), true}};
    fa::MarkNoDataByValue<fern::Mask<2>> output_no_data_policy(
        result_we_got.mask(), true);

    {
        // Fill true_value:
        // - 0, 1, 2, ...
        // - All mask cells with index % 7 become masked, all other mask cells
        //   don't.
        std::iota(true_value.data(), true_value.data() + nr_elements, 0);
        int n{0};
        std::generate(true_value.mask().data(), true_value.mask().data() +
            nr_elements, [&] () { return n++ % 7 == 0 ? true : false; });
    }

    // Condition is not masked and false.
    {
        condition = 0;
        condition.mask() = false;

        // Condition is false: result_we_want is masked.
        result_we_want.mask().fill(true);
        result_we_want.fill(-9);
        result_we_got.mask().fill(false);
        result_we_got.fill(-9);

        fa::core::if_(input_no_data_policy, output_no_data_policy,
            execution_policy, condition, true_value, result_we_got);
        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));
    }

    // Condition is not masked and true.
    {
        condition = 1;
        condition.mask() = false;

        // Fill result_we_want: copy true_value.

        // Fill result_we_want.mask().
        // If true_value is masked, the result must be masked.
        result_we_want.mask().fill(false);
        fa::algebra::or_(execution_policy, true_value.mask(),
            result_we_want.mask(), result_we_want.mask());

        // Fill result_we_want.
        // Copy true_value if mask is false, else -9.
        auto mask_it = result_we_want.mask().data();
        std::transform(true_value.data(), true_value.data() + nr_elements,
            result_we_want.data(), [&](int const& value) {
                ++mask_it;
                return *(mask_it-1) == 0 ? value : -9;
            });

        result_we_got.mask().fill(false);
        result_we_got.fill(-9);

        fa::core::if_(input_no_data_policy, output_no_data_policy,
            execution_policy, condition, true_value, result_we_got);
        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));
    }

    // Condition is masked.
    {
        condition = 1;
        condition.mask() = true;

        // Result we want is fully masked.
        result_we_want.fill(-9);
        result_we_want.mask().fill(true);

        result_we_got.fill(-9);
        result_we_got.mask().fill(false);

        fa::core::if_(input_no_data_policy, output_no_data_policy,
            execution_policy, condition, true_value, result_we_got);
        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));
    }

}


BOOST_AUTO_TEST_CASE(array_0d_2d_sequential_masked)
{
    fa::SequentialExecutionPolicy sequential;

    test_array_0d_2d_masked(sequential);
    fa::ExecutionPolicy execution_policy{sequential};
    test_array_0d_2d_masked(execution_policy);
}


BOOST_AUTO_TEST_CASE(array_0d_2d_parallel_masked)
{
    fa::ParallelExecutionPolicy parallel;

    test_array_0d_2d_masked(parallel);
    fa::ExecutionPolicy execution_policy{parallel};
    test_array_0d_2d_masked(execution_policy);
}


template<
    typename ExecutionPolicy>
void test_array_0d_0d_2d_masked(
    ExecutionPolicy& execution_policy)
{
    size_t const nr_threads{fern::hardware_concurrency()};
    size_t const nr_rows{30 * nr_threads};
    size_t const nr_cols{20 * nr_threads};
    size_t const nr_elements{nr_rows * nr_cols};

    fern::MaskedScalar<int> condition;
    fern::MaskedScalar<int> true_value;
    fern::MaskedArray<int, 2> false_value(fern::extents[nr_rows][nr_cols]);
    fern::MaskedArray<int, 2> result_we_want(fern::extents[nr_rows][nr_cols]);
    fern::MaskedArray<int, 2> result_we_got(fern::extents[nr_rows][nr_cols]);

    fa::InputNoDataPolicies<fa::DetectNoDataByValue<bool>,
        fa::DetectNoDataByValue<bool>,
        fa::DetectNoDataByValue<fern::Mask<2>>> input_no_data_policy{
            {condition.mask(), true},
            {true_value.mask(), true},
            {false_value.mask(), true}};
    fa::MarkNoDataByValue<fern::Mask<2>> output_no_data_policy(
        result_we_got.mask(), true);

    {
        // Fill false_value:
        // - 10, 11, 12, ...
        // - All mask cells with index % 9 become masked, all other mask cells
        //   don't.
        std::iota(false_value.data(), false_value.data() + nr_elements, 10);
        int n{0};
        std::generate(false_value.mask().data(), false_value.mask().data() +
            nr_elements, [&] () { return n++ % 9 == 0 ? true : false; });
    }

    // Condition is not masked and false.
    {
        condition = 0;
        condition.mask() = false;

        // Fill result_we_want: copy false_value.

        // Fill result_we_want.mask().
        // If false_value is masked, the result must be masked.
        result_we_want.mask().fill(false);
        fa::algebra::or_(execution_policy, false_value.mask(),
            result_we_want.mask(), result_we_want.mask());

        // Fill result_we_want.
        // Copy false_value if mask is false, else -9.
        auto mask_it = result_we_want.mask().data();
        std::transform(false_value.data(), false_value.data() + nr_elements,
            result_we_want.data(), [&](int const& value) {
                ++mask_it;
                return *(mask_it-1) == 0 ? value : -9;
            });

        result_we_got.mask().fill(false);
        result_we_got.fill(-9);

        fa::core::if_(input_no_data_policy, output_no_data_policy,
            execution_policy, condition, true_value, false_value,
            result_we_got);
        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));
    }

    // Condition is not masked and true.
    {
        condition = 1;
        condition.mask() = false;


        // true_value is masked.
        // Result is masked.
        true_value.mask() = true;

        result_we_want.fill(-9);
        result_we_want.mask().fill(true);

        result_we_got.fill(-9);
        result_we_got.mask().fill(false);

        fa::core::if_(input_no_data_policy, output_no_data_policy,
            execution_policy, condition, true_value, false_value,
            result_we_got);
        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));


        // true_value is not masked.
        // Copy true_value.
        true_value.mask() = false;

        result_we_want.fill(true_value.value());
        result_we_want.mask().fill(false);

        result_we_got.fill(-9);
        result_we_got.mask().fill(false);

        fa::core::if_(input_no_data_policy, output_no_data_policy,
            execution_policy, condition, true_value, false_value,
            result_we_got);
        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));
    }

    // Condition is masked.
    {
        condition = 1;
        condition.mask() = true;

        // Result we want is fully masked.
        result_we_want.fill(-9);
        result_we_want.mask().fill(true);

        result_we_got.fill(-9);
        result_we_got.mask().fill(false);

        fa::core::if_(input_no_data_policy, output_no_data_policy,
            execution_policy, condition, true_value, false_value,
            result_we_got);
        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));
    }
}


BOOST_AUTO_TEST_CASE(array_0d_0d_2d_sequential_masked)
{
    fa::SequentialExecutionPolicy sequential;

    test_array_0d_0d_2d_masked(sequential);
    fa::ExecutionPolicy execution_policy{sequential};
    test_array_0d_0d_2d_masked(execution_policy);
}


BOOST_AUTO_TEST_CASE(array_0d_0d_2d_parallel_masked)
{
    fa::ParallelExecutionPolicy parallel;

    test_array_0d_0d_2d_masked(parallel);
    fa::ExecutionPolicy execution_policy{parallel};
    test_array_0d_0d_2d_masked(execution_policy);
}


template<
    typename ExecutionPolicy>
void test_array_0d_2d_0d_masked(
    ExecutionPolicy& execution_policy)
{
    size_t const nr_threads{fern::hardware_concurrency()};
    size_t const nr_rows{30 * nr_threads};
    size_t const nr_cols{20 * nr_threads};
    size_t const nr_elements{nr_rows * nr_cols};

    fern::MaskedScalar<int> condition;
    fern::MaskedArray<int, 2> true_value(fern::extents[nr_rows][nr_cols]);
    fern::MaskedScalar<int> false_value;
    fern::MaskedArray<int, 2> result_we_want(fern::extents[nr_rows][nr_cols]);
    fern::MaskedArray<int, 2> result_we_got(fern::extents[nr_rows][nr_cols]);

    fa::InputNoDataPolicies<fa::DetectNoDataByValue<bool>,
        fa::DetectNoDataByValue<fern::Mask<2>>,
        fa::DetectNoDataByValue<bool>> input_no_data_policy{
            {condition.mask(), true},
            {true_value.mask(), true},
            {false_value.mask(), true}};
    fa::MarkNoDataByValue<fern::Mask<2>> output_no_data_policy(
        result_we_got.mask(), true);

    {
        // Fill true_value:
        // - 0, 1, 2, ...
        // - All mask cells with index % 7 become masked, all other mask cells
        //   don't.
        std::iota(true_value.data(), true_value.data() + nr_elements, 0);
        int n{0};
        std::generate(true_value.mask().data(), true_value.mask().data() +
            nr_elements, [&] () { return n++ % 7 == 0 ? true : false; });
    }

    // Condition is not masked and false.
    {
        condition = 0;
        condition.mask() = false;

        // false_value is masked.
        // Result is masked.
        false_value.mask() = true;

        result_we_want.fill(-9);
        result_we_want.mask().fill(true);

        result_we_got.fill(-9);
        result_we_got.mask().fill(false);

        fa::core::if_(input_no_data_policy, output_no_data_policy,
            execution_policy, condition, true_value, false_value,
            result_we_got);
        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));


        // false_value is not masked.
        // Copy false_value.
        false_value.mask() = false;

        result_we_want.fill(false_value.value());
        result_we_want.mask().fill(false);

        result_we_got.fill(-9);
        result_we_got.mask().fill(false);

        fa::core::if_(input_no_data_policy, output_no_data_policy,
            execution_policy, condition, true_value, false_value,
            result_we_got);
        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));
    }

    // Condition is not masked and true.
    {
        condition = 1;
        condition.mask() = false;

        // Fill result_we_want: copy false_value.

        // Fill result_we_want.mask().
        // If true_value is masked, the result must be masked.
        result_we_want.mask().fill(false);
        fa::algebra::or_(execution_policy, true_value.mask(),
            result_we_want.mask(), result_we_want.mask());

        // Fill result_we_want.
        // Copy true_value if mask is false, else -9.
        auto mask_it = result_we_want.mask().data();
        std::transform(true_value.data(), true_value.data() + nr_elements,
            result_we_want.data(), [&](int const& value) {
                ++mask_it;
                return *(mask_it-1) == 0 ? value : -9;
            });

        result_we_got.mask().fill(false);
        result_we_got.fill(-9);

        fa::core::if_(input_no_data_policy, output_no_data_policy,
            execution_policy, condition, true_value, false_value,
            result_we_got);
        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));
    }

    // Condition is masked.
    {
        condition = 1;
        condition.mask() = true;

        // Result we want is fully masked.
        result_we_want.fill(-9);
        result_we_want.mask().fill(true);

        result_we_got.fill(-9);
        result_we_got.mask().fill(false);

        fa::core::if_(input_no_data_policy, output_no_data_policy,
            execution_policy, condition, true_value, false_value,
            result_we_got);
        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));
    }
}


BOOST_AUTO_TEST_CASE(array_0d_2d_0d_sequential_masked)
{
    fa::SequentialExecutionPolicy sequential;

    test_array_0d_2d_0d_masked(sequential);
    fa::ExecutionPolicy execution_policy{sequential};
    test_array_0d_2d_0d_masked(execution_policy);
}


BOOST_AUTO_TEST_CASE(array_0d_2d_0d_parallel_masked)
{
    fa::ParallelExecutionPolicy parallel;

    test_array_0d_2d_0d_masked(parallel);
    fa::ExecutionPolicy execution_policy{parallel};
    test_array_0d_2d_0d_masked(execution_policy);
}


template<
    typename ExecutionPolicy>
void test_array_0d_2d_2d_masked(
    ExecutionPolicy& execution_policy)
{
    size_t const nr_threads{fern::hardware_concurrency()};
    size_t const nr_rows{30 * nr_threads};
    size_t const nr_cols{20 * nr_threads};
    size_t const nr_elements{nr_rows * nr_cols};

    fern::MaskedScalar<int> condition;
    fern::MaskedArray<int, 2> true_value(fern::extents[nr_rows][nr_cols]);
    fern::MaskedArray<int, 2> false_value(fern::extents[nr_rows][nr_cols]);
    fern::MaskedArray<int, 2> result_we_want(fern::extents[nr_rows][nr_cols]);
    fern::MaskedArray<int, 2> result_we_got(fern::extents[nr_rows][nr_cols]);

    fa::InputNoDataPolicies<fa::DetectNoDataByValue<bool>,
        fa::DetectNoDataByValue<fern::Mask<2>>,
        fa::DetectNoDataByValue<fern::Mask<2>>> input_no_data_policy{
            {condition.mask(), true},
            {true_value.mask(), true},
            {false_value.mask(), true}};
    fa::MarkNoDataByValue<fern::Mask<2>> output_no_data_policy(
        result_we_got.mask(), true);

    {
        // Fill true_value:
        // - 0, 1, 2, ...
        // - All mask cells with index % 7 become masked, all other mask cells
        //   don't.
        std::iota(true_value.data(), true_value.data() + nr_elements, 0);
        int n{0};
        std::generate(true_value.mask().data(), true_value.mask().data() +
            nr_elements, [&] () { return n++ % 7 == 0 ? true : false; });

        // Fill false_value:
        // - 10, 11, 12, ...
        // - All mask cells with index % 9 become masked, all other mask cells
        //   don't.
        std::iota(false_value.data(), false_value.data() + nr_elements, 10);
        n = 0;
        std::generate(false_value.mask().data(), false_value.mask().data() +
            nr_elements, [&] () { return n++ % 9 == 0 ? true : false; });
    }

    // Condition is not masked and false.
    {
        condition = 0;
        condition.mask() = false;

        // Fill result_we_want: copy false_value.

        // Fill result_we_want.mask().
        // If false_value is masked, the result must be masked.
        result_we_want.mask().fill(false);
        fa::algebra::or_(execution_policy, false_value.mask(),
            result_we_want.mask(), result_we_want.mask());

        // Fill result_we_want.
        // Copy false_value if mask is false, else -9.
        auto mask_it = result_we_want.mask().data();
        std::transform(false_value.data(), false_value.data() + nr_elements,
            result_we_want.data(), [&](int const& value) {
                ++mask_it;
                return *(mask_it-1) == 0 ? value : -9;
            });

        result_we_got.mask().fill(false);
        result_we_got.fill(-9);

        fa::core::if_(input_no_data_policy, output_no_data_policy,
            execution_policy, condition, true_value, false_value,
            result_we_got);
        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));
    }

    // Condition is not masked and true.
    {
        condition = 1;
        condition.mask() = false;

        // Fill result_we_want: copy true_value.

        // Fill result_we_want.mask().
        // If true_value is masked, the result must be masked.
        result_we_want.mask().fill(false);
        fa::algebra::or_(execution_policy, true_value.mask(),
            result_we_want.mask(), result_we_want.mask());

        // Fill result_we_want.
        // Copy true_value if mask is false, else -9.
        auto mask_it = result_we_want.mask().data();
        std::transform(true_value.data(), true_value.data() + nr_elements,
            result_we_want.data(), [&](int const& value) {
                ++mask_it;
                return *(mask_it-1) == 0 ? value : -9;
            });

        result_we_got.mask().fill(false);
        result_we_got.fill(-9);

        fa::core::if_(input_no_data_policy, output_no_data_policy,
            execution_policy, condition, true_value, false_value,
            result_we_got);
        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));
    }

    // Condition is masked.
    {
        condition = 1;
        condition.mask() = true;

        // Result we want is fully masked.
        result_we_want.fill(-9);
        result_we_want.mask().fill(true);

        result_we_got.fill(-9);
        result_we_got.mask().fill(false);

        fa::core::if_(input_no_data_policy, output_no_data_policy,
            execution_policy, condition, true_value, false_value,
            result_we_got);
        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));
    }

}


BOOST_AUTO_TEST_CASE(array_0d_2d_2d_sequential_masked)
{
    fa::SequentialExecutionPolicy sequential;

    test_array_0d_2d_2d_masked(sequential);
    fa::ExecutionPolicy execution_policy{sequential};
    test_array_0d_2d_2d_masked(execution_policy);
}


BOOST_AUTO_TEST_CASE(array_0d_2d_2d_parallel_masked)
{
    fa::ParallelExecutionPolicy parallel;

    test_array_0d_2d_2d_masked(parallel);
    fa::ExecutionPolicy execution_policy{parallel};
    test_array_0d_2d_2d_masked(execution_policy);
}


template<
    typename ExecutionPolicy>
void test_array_0d_0d_0d(
    ExecutionPolicy& execution_policy)
{
    int condition;
    int true_value;
    int false_value;
    int result_we_want;
    int result_we_got;

    {
        condition = 0;
        true_value = 5;
        false_value = 6;

        result_we_want = -9;
        result_we_got = -9;
        fa::core::if_(execution_policy, condition, true_value, result_we_got);
        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));

        result_we_want = 6;
        result_we_got = -9;
        fa::core::if_(execution_policy, condition, true_value, false_value,
            result_we_got);
        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));
    }

    {
        condition = 1;
        true_value = 5;
        false_value = 6;

        result_we_want = 5;
        result_we_got = -9;
        fa::core::if_(execution_policy, condition, true_value, result_we_got);
        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));

        result_we_want = 5;
        result_we_got = -9;
        fa::core::if_(execution_policy, condition, true_value, false_value,
            result_we_got);
        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));
    }

}


BOOST_AUTO_TEST_CASE(array_0d_0d_0d_sequential)
{
    fa::SequentialExecutionPolicy sequential;

    test_array_0d_0d_0d(sequential);
    fa::ExecutionPolicy execution_policy{sequential};
    test_array_0d_0d_0d(execution_policy);
}


BOOST_AUTO_TEST_CASE(array_0d_0d_0d_parallel)
{
    fa::ParallelExecutionPolicy parallel;

    test_array_0d_0d_0d(parallel);
    fa::ExecutionPolicy execution_policy{parallel};
    test_array_0d_0d_0d(execution_policy);
}


template<
    typename ExecutionPolicy>
void test_array_0d_0d_0d_masked(
    ExecutionPolicy& execution_policy)
{
    fern::MaskedScalar<int> condition;
    fern::MaskedScalar<int> true_value;
    fern::MaskedScalar<int> false_value;
    fern::MaskedScalar<int> result_we_want;
    fern::MaskedScalar<int> result_we_got;

    fa::MarkNoDataByValue<bool> output_no_data_policy(result_we_got.mask(),
        true);

    {
        fa::InputNoDataPolicies<fa::DetectNoDataByValue<bool>,
            fa::DetectNoDataByValue<bool>, fa::DetectNoDataByValue<bool>>
                input_no_data_policy{
                    {condition.mask(), true},
                    {true_value.mask(), true},
                    {false_value.mask(), true}};

        condition = 0;
        condition.mask() = false;
        true_value = 5;
        true_value.mask() = false;
        false_value = 6;
        false_value.mask() = false;

        result_we_want = fern::MaskedScalar<int>(-9, true);
        result_we_got = fern::MaskedScalar<int>(-9, false);
        fa::core::if_(input_no_data_policy, output_no_data_policy,
            execution_policy, condition, true_value, result_we_got);
        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));

        result_we_want = fern::MaskedScalar<int>(6, false);
        result_we_got = fern::MaskedScalar<int>(-9, false);
        fa::core::if_(input_no_data_policy, output_no_data_policy,
            execution_policy, condition, true_value, false_value,
            result_we_got);
        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));
    }

    {
        fa::InputNoDataPolicies<fa::DetectNoDataByValue<bool>,
            fa::DetectNoDataByValue<bool>, fa::DetectNoDataByValue<bool>>
                input_no_data_policy{
                    {condition.mask(), true},
                    {true_value.mask(), true},
                    {false_value.mask(), true}};

        condition = 1;
        condition.mask() = false;
        true_value = 5;
        true_value.mask() = false;
        false_value = 6;
        false_value.mask() = false;

        result_we_want = fern::MaskedScalar<int>(5, false);
        result_we_got = fern::MaskedScalar<int>(-9, false);
        fa::core::if_(input_no_data_policy, output_no_data_policy,
            execution_policy, condition, true_value, result_we_got);
        BOOST_CHECK_EQUAL(result_we_got.value(), 5);
        BOOST_CHECK(!result_we_got.mask());

        result_we_want = fern::MaskedScalar<int>(5, false);
        result_we_got = fern::MaskedScalar<int>(-9, false);
        fa::core::if_(input_no_data_policy, output_no_data_policy,
            execution_policy, condition, true_value, false_value,
            result_we_got);
        BOOST_CHECK_EQUAL(result_we_got.value(), 5);
        BOOST_CHECK(!result_we_got.mask());
    }
}


BOOST_AUTO_TEST_CASE(array_0d_0d_0d_masked_sequential)
{
    fa::SequentialExecutionPolicy sequential;

    test_array_0d_0d_0d_masked(sequential);
    fa::ExecutionPolicy execution_policy{sequential};
    test_array_0d_0d_0d_masked(execution_policy);
}


BOOST_AUTO_TEST_CASE(array_0d_0d_0d_masked_parallel)
{
    fa::ParallelExecutionPolicy parallel;

    test_array_0d_0d_0d_masked(parallel);
    fa::ExecutionPolicy execution_policy{parallel};
    test_array_0d_0d_0d_masked(execution_policy);
}


// TODO 1d


template<
    typename ExecutionPolicy>
void test_array_2d_2d_2d(
    ExecutionPolicy& execution_policy)
{
    size_t const nr_threads{fern::hardware_concurrency()};
    size_t const nr_rows{30 * nr_threads};
    size_t const nr_cols{20 * nr_threads};
    size_t const nr_elements{nr_rows * nr_cols};

    fern::Array<int, 2> condition(fern::extents[nr_rows][nr_cols]);
    fern::Array<int, 2> true_value(fern::extents[nr_rows][nr_cols]);
    fern::Array<int, 2> false_value(fern::extents[nr_rows][nr_cols]);
    fern::Array<int, 2> result_we_got(fern::extents[nr_rows][nr_cols]);

    {
        // Fill condition: all cells with odd indices become true cells, all
        // even cells become false cells.
        int n{0};
        std::generate(condition.data(), condition.data() + nr_elements,
            [&] () { return n++ % 2 == 0 ? 0 : 1; });

        // Fill true_value: 0, 1, 2, ...
        std::iota(true_value.data(), true_value.data() + nr_elements, 0);

        // Fill false_value: 10, 11, 12, ...
        std::iota(false_value.data(), false_value.data() + nr_elements, 10);
    }

    // if_then
    {
        fern::Array<int, 2> result_we_want(fern::extents[nr_rows][nr_cols]);

        {
            // Fill result_we_want: copy true_value if condition it true, else
            // -9.
            auto true_value_it = true_value.data();
            std::transform(condition.data(), condition.data() + nr_elements,
                result_we_want.data(), [&](int const& value) {
                    ++true_value_it;
                    return value ? *(true_value_it-1) : -9; });
        }

        result_we_got.fill(-9);
        fa::core::if_(execution_policy, condition, true_value, result_we_got);
        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));
    }

    // if_then_else
    {
        fern::Array<int, 2> result_we_want(fern::extents[nr_rows][nr_cols]);

        {
            // Fill result_we_want: copy true_value if condition it true, else
            // false_value.
            auto true_value_it = true_value.data();
            auto false_value_it = false_value.data();
            std::transform(condition.data(), condition.data() + nr_elements,
                result_we_want.data(), [&] (int const& value) {
                    ++true_value_it;
                    ++false_value_it;
                    return value ? *(true_value_it-1) : *(false_value_it-1); });
        }

        result_we_got.fill(-9);
        fa::core::if_(execution_policy, condition, true_value, false_value,
            result_we_got);
        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));
    }
}


BOOST_AUTO_TEST_CASE(array_2d_2d_2d_sequential)
{
    fa::SequentialExecutionPolicy sequential;

    test_array_2d_2d_2d(sequential);
    fa::ExecutionPolicy execution_policy{sequential};
    test_array_2d_2d_2d(execution_policy);
}


BOOST_AUTO_TEST_CASE(array_2d_2d_2d_parallel)
{
    fa::ParallelExecutionPolicy parallel;

    test_array_2d_2d_2d(parallel);
    fa::ExecutionPolicy execution_policy{parallel};
    test_array_2d_2d_2d(execution_policy);
}


template<
    typename ExecutionPolicy>
void test_array_2d_2d_2d_masked(
    ExecutionPolicy& execution_policy)
{
    size_t const nr_threads{fern::hardware_concurrency()};
    size_t const nr_rows{30 * nr_threads};
    size_t const nr_cols{20 * nr_threads};
    size_t const nr_elements{nr_rows * nr_cols};

    fern::MaskedArray<int, 2> condition(fern::extents[nr_rows][nr_cols]);
    fern::MaskedArray<int, 2> true_value(fern::extents[nr_rows][nr_cols]);
    fern::MaskedArray<int, 2> false_value(fern::extents[nr_rows][nr_cols]);
    fern::MaskedArray<int, 2> result_we_got(fern::extents[nr_rows][nr_cols]);

    fa::MarkNoDataByValue<fern::Mask<2>> output_no_data_policy(
        result_we_got.mask(), true);

    {
        // Fill condition:
        // - All cells with index % 2 become true cells, all other cells
        //   become false cells.
        // - All mask cells with index % 5 become masked, all other mask cells
        //   don't.
        int n{0};
        std::generate(condition.data(), condition.data() + nr_elements,
            [&] () { return n++ % 2 == 0 ? 0 : 1; });
        n = 0;
        std::generate(condition.mask().data(), condition.mask().data() +
            nr_elements, [&] () { return n++ % 5 == 0 ? true : false; });

        // Fill true_value:
        // - 0, 1, 2, ...
        // - All mask cells with index % 7 become masked, all other mask cells
        //   don't.
        std::iota(true_value.data(), true_value.data() + nr_elements, 0);
        n = 0;
        std::generate(true_value.mask().data(), true_value.mask().data() +
            nr_elements, [&] () { return n++ % 7 == 0 ? true : false; });

        // Fill false_value:
        // - 10, 11, 12, ...
        // - All mask cells with index % 9 become masked, all other mask cells
        //   don't.
        std::iota(false_value.data(), false_value.data() + nr_elements, 10);
        n = 0;
        std::generate(false_value.mask().data(), false_value.mask().data() +
            nr_elements, [&] () { return n++ % 9 == 0 ? true : false; });
    }

    // if_then
    {
        fa::InputNoDataPolicies<fa::DetectNoDataByValue<fern::Mask<2>>,
            fa::DetectNoDataByValue<fern::Mask<2>>> input_no_data_policy{
                {condition.mask(), true},
                {true_value.mask(), true}};

        fern::MaskedArray<int, 2> result_we_want(
            fern::extents[nr_rows][nr_cols]);

        {
            // Fill result_we_want.mask().
            // If condition is false, the result must be masked.
            // If condition is masked, the result must be masked.
            // If true_value is masked, the result must be masked.
            std::transform(condition.data(), condition.data() + nr_elements,
                result_we_want.mask().data(), [&](int const& value) {
                    return value ? false : true; });
            fa::algebra::or_(execution_policy, condition.mask(),
                result_we_want.mask(), result_we_want.mask());
            fa::algebra::or_(execution_policy, true_value.mask(),
                result_we_want.mask(), result_we_want.mask());

            // Fill result_we_want.
            // Copy true_value if condition it true and mask is false, else -9.
            auto true_value_it = true_value.data();
            auto mask_it = result_we_want.mask().data();
            std::transform(condition.data(), condition.data() + nr_elements,
                result_we_want.data(), [&](int const& value) {
                    ++true_value_it;
                    ++mask_it;
                    return value && *(mask_it-1) == 0 ? *(true_value_it-1)
                        : -9; });
        }

        result_we_got.fill(-9);
        result_we_got.mask().fill(false);
        fa::algebra::or_(execution_policy, condition.mask(),
            true_value.mask(), result_we_got.mask());

        fa::core::if_(input_no_data_policy, output_no_data_policy,
            execution_policy, condition, true_value, result_we_got);
        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));
    }

    // if_then_else
    {
        fa::InputNoDataPolicies<fa::DetectNoDataByValue<fern::Mask<2>>,
            fa::DetectNoDataByValue<fern::Mask<2>>,
            fa::DetectNoDataByValue<fern::Mask<2>>> input_no_data_policy{
                {condition.mask(), true},
                {true_value.mask(), true},
                {false_value.mask(), true}};

        fern::MaskedArray<int, 2> result_we_want(
            fern::extents[nr_rows][nr_cols]);

        {
            // Fill result_we_want.mask().
            // - If condition is masked, the result must be masked.
            // - If condition is true and true_value is masked, the result
            //   must be masked.
            // - If condition is false and false_value is masked, the result
            //   must be masked.
            //
            // TODO It would be nice to be able to replace this loop with
            //      a nested expression of our own.
            for(size_t i = 0; i < size(result_we_want); ++i) {
                if(get(condition.mask(), i) ||
                        (get(true_value, i) && get(true_value.mask(), i)) ||
                        (get(false_value, i) && get(false_value.mask(), i))) {
                    get(result_we_want.mask(), i) = true;
                }
            }

            // - Initialize all values with -9.
            // - If condition is not no-data and true and true_value is not
            //   masked, the result must be true_value.
            // - If condition is not no-data and false and false_value is not
            //   masked, the result must be false_value.
            //
            // TODO It would be nice to be able to replace this loop with
            //      a nested expression of our own.
            result_we_want.fill(-9);
            for(size_t i = 0; i < size(result_we_want); ++i) {
                if(!get(condition.mask(), i)) {
                    if(get(condition, i)) {
                        if(!get(true_value.mask(), i)) {
                            get(result_we_want, i) = get(true_value, i);
                        }
                    }
                    else {
                        if(!get(false_value.mask(), i)) {
                            get(result_we_want, i) = get(false_value, i);
                        }
                    }
                }
            }
        }

        result_we_got.fill(-9);
        result_we_got.mask().fill(false);
        fa::algebra::or_(execution_policy, result_we_got.mask(),
            condition.mask(), result_we_got.mask());
        fa::algebra::or_(execution_policy, result_we_got.mask(),
            true_value.mask(), result_we_got.mask());
        fa::algebra::or_(execution_policy, result_we_got.mask(),
            false_value.mask(), result_we_got.mask());

        fa::core::if_(input_no_data_policy, output_no_data_policy,
            execution_policy, condition, true_value, false_value,
            result_we_got);
        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));
    }
}


BOOST_AUTO_TEST_CASE(array_2d_2d_2d_masked_sequential)
{
    fa::SequentialExecutionPolicy sequential;

    test_array_2d_2d_2d_masked(sequential);
    fa::ExecutionPolicy execution_policy{sequential};
    test_array_2d_2d_2d_masked(execution_policy);
}


BOOST_AUTO_TEST_CASE(array_2d_2d_2d_masked_parallel)
{
    fa::ParallelExecutionPolicy parallel;

    test_array_2d_2d_2d_masked(parallel);
    fa::ExecutionPolicy execution_policy{parallel};
    test_array_2d_2d_2d_masked(execution_policy);
}
