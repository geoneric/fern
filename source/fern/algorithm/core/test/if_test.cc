#define BOOST_TEST_MODULE fern algorithm core if_
#include <boost/test/unit_test.hpp>
#include "fern/core/constant_traits.h"
#include "fern/feature/core/array_traits.h"
#include "fern/feature/core/masked_array_traits.h"
#include "fern/feature/core/masked_constant_traits.h"
#include "fern/algorithm/algebra/boole/or.h"
#include "fern/algorithm/core/if.h"
#include "fern/algorithm/core/test/test_utils.h"


namespace fa = fern::algorithm;


BOOST_FIXTURE_TEST_SUITE(if_, fern::ThreadClient)

void test_array_0d_0d_0d(
    fa::ExecutionPolicy const& execution_policy)
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
    test_array_0d_0d_0d(fa::sequential);
}


BOOST_AUTO_TEST_CASE(array_0d_0d_0d_parallel)
{
    test_array_0d_0d_0d(fa::parallel);
}


void test_array_0d_0d_0d_masked(
    fa::ExecutionPolicy const& execution_policy)
{
    fern::MaskedConstant<int> condition;
    fern::MaskedConstant<int> true_value;
    fern::MaskedConstant<int> false_value;
    fern::MaskedConstant<int> result_we_want;
    fern::MaskedConstant<int> result_we_got;

    fa::DetectNoDataByValue<bool> input_no_data_policy(result_we_got.mask(),
        true);
    fa::MarkNoDataByValue<bool> output_no_data_policy(result_we_got.mask(),
        true);

    {
        condition = 0;
        condition.mask() = false;
        true_value = 5;
        true_value.mask() = false;
        false_value = 6;
        false_value.mask() = false;

        result_we_want = fern::MaskedConstant<int>(-9, true);
        result_we_got = fern::MaskedConstant<int>(-9, false);
        fa::core::if_(input_no_data_policy, output_no_data_policy,
            execution_policy, condition, true_value, result_we_got);
        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));

        result_we_want = fern::MaskedConstant<int>(6, false);
        result_we_got = fern::MaskedConstant<int>(-9, false);
        fa::core::if_(input_no_data_policy, output_no_data_policy,
            execution_policy, condition, true_value, false_value,
            result_we_got);
        BOOST_CHECK(fern::compare(execution_policy, result_we_got,
            result_we_want));
    }

    {
        condition = 1;
        condition.mask() = false;
        true_value = 5;
        true_value.mask() = false;
        false_value = 6;
        false_value.mask() = false;

        result_we_want = fern::MaskedConstant<int>(5, false);
        result_we_got = fern::MaskedConstant<int>(-9, false);
        fa::core::if_(input_no_data_policy, output_no_data_policy,
            execution_policy, condition, true_value, result_we_got);
        BOOST_CHECK_EQUAL(result_we_got.value(), 5);
        BOOST_CHECK(!result_we_got.mask());

        result_we_want = fern::MaskedConstant<int>(5, false);
        result_we_got = fern::MaskedConstant<int>(-9, false);
        fa::core::if_(input_no_data_policy, output_no_data_policy,
            execution_policy, condition, true_value, false_value,
            result_we_got);
        BOOST_CHECK_EQUAL(result_we_got.value(), 5);
        BOOST_CHECK(!result_we_got.mask());
    }
}


BOOST_AUTO_TEST_CASE(array_0d_0d_0d_masked_sequential)
{
    test_array_0d_0d_0d_masked(fa::sequential);
}


BOOST_AUTO_TEST_CASE(array_0d_0d_0d_masked_parallel)
{
    test_array_0d_0d_0d_masked(fa::parallel);
}


// TODO 1d


void test_array_2d_2d_2d(
    fa::ExecutionPolicy const& execution_policy)
{
    size_t const nr_threads{fern::ThreadClient::hardware_concurrency()};
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
    test_array_2d_2d_2d(fa::sequential);
}


BOOST_AUTO_TEST_CASE(array_2d_2d_2d_parallel)
{
    test_array_2d_2d_2d(fa::parallel);
}


void test_array_2d_2d_2d_masked(
    fa::ExecutionPolicy const& execution_policy)
{
    size_t const nr_threads{fern::ThreadClient::hardware_concurrency()};
    size_t const nr_rows{30 * nr_threads};
    size_t const nr_cols{20 * nr_threads};
    size_t const nr_elements{nr_rows * nr_cols};

    fern::MaskedArray<int, 2> condition(fern::extents[nr_rows][nr_cols]);
    fern::MaskedArray<int, 2> true_value(fern::extents[nr_rows][nr_cols]);
    fern::MaskedArray<int, 2> false_value(fern::extents[nr_rows][nr_cols]);
    fern::MaskedArray<int, 2> result_we_got(fern::extents[nr_rows][nr_cols]);

    fa::DetectNoDataByValue<fern::Mask<2>> input_no_data_policy(
        result_we_got.mask(), true);
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
        std::generate(true_value.mask().data(), true_value.mask().data() +
            nr_elements, [&] () { return n++ % 7 == 0 ? true : false; });

        // Fill false_value:
        // - 10, 11, 12, ...
        // - All mask cells with index % 9 become masked, all other mask cells
        //   don't.
        std::iota(false_value.data(), false_value.data() + nr_elements, 10);
        std::generate(true_value.mask().data(), true_value.mask().data() +
            nr_elements, [&] () { return n++ % 9 == 0 ? true : false; });
    }

    // if_then
    {
        fern::MaskedArray<int, 2> result_we_want(
            fern::extents[nr_rows][nr_cols]);

        {
            // Fill result_we_want.mask().
            // If condition is false, the result must be masked.
            // If condition is masked, the result must be masked.
            // If true_value is masked, the result must be masked.
            std::transform(condition.data(), condition.data() + nr_elements,
                result_we_want.mask().data(), [&](int const& value) {
                    return value ? 0 : 1; });
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
        fern::MaskedArray<int, 2> result_we_want(
            fern::extents[nr_rows][nr_cols]);

        {
            // Fill result_we_want.mask().
            // If condition is masked, the result must be masked.
            // If true_value is masked, the result must be masked.
            // If false_value is masked, the result must be masked.
            fa::algebra::or_(execution_policy, condition.mask(),
                result_we_want.mask(), result_we_want.mask());
            fa::algebra::or_(execution_policy, true_value.mask(),
                result_we_want.mask(), result_we_want.mask());
            fa::algebra::or_(execution_policy, false_value.mask(),
                result_we_want.mask(), result_we_want.mask());

            // Fill result_we_want.
            // If mask is false, copy true value if condition is true, copy
            // false value if condition is false.
            // If mask is true, set -9.
            auto true_value_it = true_value.data();
            auto false_value_it = false_value.data();
            auto mask_it = result_we_want.mask().data();
            std::transform(condition.data(), condition.data() + nr_elements,
                result_we_want.data(), [&](int const& value) {
                    ++true_value_it;
                    ++false_value_it;
                    ++mask_it;
                    return *(mask_it-1) == 0
                        ? (value ? *(true_value_it-1) : *(false_value_it-1))
                        : -9; });
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
    test_array_2d_2d_2d_masked(fa::sequential);
}


BOOST_AUTO_TEST_CASE(array_2d_2d_2d_masked_parallel)
{
    test_array_2d_2d_2d_masked(fa::parallel);
}

BOOST_AUTO_TEST_SUITE_END()