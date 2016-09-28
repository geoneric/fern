// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern algorithm statistic unary_min
#include <boost/test/unit_test.hpp>
#include "fern/core/data_customization_point/scalar.h"
#include "fern/feature/core/data_customization_point/array.h"
#include "fern/feature/core/data_customization_point/masked_array.h"
#include "fern/feature/core/data_customization_point/masked_scalar.h"
#include "fern/algorithm/core/test/test_utils.h"
#include "fern/algorithm/statistic/unary_min.h"


namespace f = fern;
namespace fa = f::algorithm;
namespace ft = f::test;


template<
    typename ExecutionPolicy,
    typename Argument,
    typename Result>
void verify_0d_0d(
    ExecutionPolicy& execution_policy,
    Argument const& value,
    Result const& result_we_want)
{
    int result_we_got{-9};
    fa::statistic::unary_min<>(execution_policy, value, result_we_got);
    BOOST_CHECK_EQUAL(result_we_got, result_we_want);
}


template<
    typename ExecutionPolicy>
void test_0d_0d(
    ExecutionPolicy& execution_policy)
{
    {
        int value{5};
        int result_we_want{5};
        verify_0d_0d(execution_policy, value, result_we_want);
    }

    {
        int value{-5};
        int result_we_want{-5};
        verify_0d_0d(execution_policy, value, result_we_want);
    }

    {
        int value{0};
        int result_we_want{0};
        verify_0d_0d(execution_policy, value, result_we_want);
    }
}


template<
    typename ExecutionPolicy,
    typename Argument,
    typename Result>
void verify_0d_0d_masked(
    ExecutionPolicy& execution_policy,
    Argument const& value,
    Result const& result_we_want)
{
    using InputNoDataPolicy = fa::InputNoDataPolicies<
        fa::DetectNoDataByValue<bool>>;
    using OutputNoDataPolicy = fa::MarkNoDataByValue<bool>;

    f::MaskedScalar<int> result_we_got{-9};

    InputNoDataPolicy input_no_data_policy{{value.mask(), true}};
    OutputNoDataPolicy output_no_data_policy{result_we_got.mask(), true};

    fa::statistic::unary_min<>(input_no_data_policy, output_no_data_policy,
        execution_policy, value, result_we_got);
    BOOST_CHECK_EQUAL(result_we_got, result_we_want);
}


template<
    typename ExecutionPolicy>
void test_0d_0d_masked(
    ExecutionPolicy& execution_policy)
{
    // Regular case.
    {
        f::MaskedScalar<int> value{5};
        f::MaskedScalar<int> result_we_want{5};
        verify_0d_0d_masked(execution_policy, value, result_we_want);
    }

    // Mask a value.
    {
        f::MaskedScalar<int> value{5, true};
        f::MaskedScalar<int> result_we_want{-9, true};
        verify_0d_0d_masked(execution_policy, value, result_we_want);
    }
}


template<
    typename ExecutionPolicy,
    typename Argument,
    typename Result>
void verify_1d_0d(
    ExecutionPolicy& execution_policy,
    Argument const& value,
    Result const& result_we_want)
{
    int result_we_got{-9};
    fa::statistic::unary_min<>(execution_policy, value, result_we_got);
    BOOST_CHECK_EQUAL(result_we_got, result_we_want);
}


template<
    typename ExecutionPolicy>
void test_1d_0d(
    ExecutionPolicy& execution_policy)
{
    // Regular case.
    {
        f::Array<int, 1> value(ft::nr_elements_1d);
        std::iota(value.data(), value.data() + ft::nr_elements_1d, 5);
        int result_we_want{5};
        verify_1d_0d(execution_policy, value, result_we_want);
    }

    // Empty.
    {
        f::Array<int, 1> value(0);
        int result_we_want{-9};
        verify_1d_0d(execution_policy, value, result_we_want);
    }
}


template<
    typename ExecutionPolicy,
    typename Argument,
    typename Result>
void verify_1d_0d_masked(
    ExecutionPolicy& execution_policy,
    Argument const& value,
    Result const& result_we_want)
{
    using InputNoDataPolicy = fa::InputNoDataPolicies<
        fa::DetectNoDataByValue<f::Mask<1>>>;
    using OutputNoDataPolicy = fa::MarkNoDataByValue<bool>;

    f::MaskedScalar<int> result_we_got{-9};

    InputNoDataPolicy input_no_data_policy{{value.mask(), true}};
    OutputNoDataPolicy output_no_data_policy(result_we_got.mask(), true);

    fa::statistic::unary_min<>(input_no_data_policy, output_no_data_policy,
        execution_policy, value, result_we_got);
    BOOST_CHECK_EQUAL(result_we_got, result_we_want);
}


template<
    typename ExecutionPolicy>
void test_1d_0d_masked(
    ExecutionPolicy& execution_policy)
{
    // Regular case.
    {
        f::MaskedArray<int, 1> value(ft::nr_elements_1d);
        std::iota(value.data(), value.data() + ft::nr_elements_1d, 5);
        f::MaskedScalar<int> result_we_want{5};
        verify_1d_0d_masked(execution_policy, value, result_we_want);
    }


    // Mask a value.
    {
        f::MaskedArray<int, 1> value(ft::nr_elements_1d);
        std::iota(value.data(), value.data() + ft::nr_elements_1d, 5);
        get(value.mask(), 5) = true;
        f::MaskedScalar<int> result_we_want{5};
        verify_1d_0d_masked(execution_policy, value, result_we_want);
    }


    // Mask all values.
    {
        f::MaskedArray<int, 1> value(ft::nr_elements_1d);
        value.mask().fill(true);
        f::MaskedScalar<int> result_we_want{-9, true};
        verify_1d_0d_masked(execution_policy, value, result_we_want);
    }


    // Empty.
    {
        f::MaskedArray<int, 1> value(0);
        f::MaskedScalar<int> result_we_want{-9, true};
        verify_1d_0d_masked(execution_policy, value, result_we_want);
    }
}


template<
    typename ExecutionPolicy,
    typename Argument,
    typename Result>
void verify_2d_0d(
    ExecutionPolicy& execution_policy,
    Argument const& value,
    Result const& result_we_want)
{
    int result_we_got{-9};
    fa::statistic::unary_min<>(execution_policy, value, result_we_got);
    BOOST_CHECK_EQUAL(result_we_got, result_we_want);
}


template<
    typename ExecutionPolicy>
void test_2d_0d(
    ExecutionPolicy& execution_policy)
{
    // Regular case.
    {
        f::Array<int, 2> value(f::extents[ft::nr_rows][ft::nr_cols]);
        std::iota(value.data(), value.data() + ft::nr_elements_2d, 5);
        int result_we_want{5};
        verify_2d_0d(execution_policy, value, result_we_want);
    }

    // Empty.
    {
        f::Array<int, 2> value(f::extents[0][0]);
        int result_we_want{-9};
        verify_2d_0d(execution_policy, value, result_we_want);
    }
}


template<
    typename ExecutionPolicy,
    typename Argument,
    typename Result>
void verify_2d_0d_masked(
    ExecutionPolicy& execution_policy,
    Argument const& value,
    Result const& result_we_want)
{
    using InputNoDataPolicy = fa::InputNoDataPolicies<
        fa::DetectNoDataByValue<f::Mask<2>>>;
    using OutputNoDataPolicy = fa::MarkNoDataByValue<bool>;

    f::MaskedScalar<f::value_type<Result>> result_we_got{-9};

    InputNoDataPolicy input_no_data_policy{{value.mask(), true}};
    OutputNoDataPolicy output_no_data_policy(result_we_got.mask(), true);

    fa::statistic::unary_min<>(input_no_data_policy, output_no_data_policy,
        execution_policy, value, result_we_got);
    BOOST_CHECK_EQUAL(result_we_got, result_we_want);
}


template<
    typename ExecutionPolicy>
void test_2d_0d_masked(
    ExecutionPolicy& execution_policy)
{
    // Regular case.
    {
        f::MaskedArray<int, 2> value(f::extents[ft::nr_rows][ft::nr_cols]);
        std::iota(value.data(), value.data() + ft::nr_elements_2d, 5);
        f::MaskedScalar<int> result_we_want{5};
        verify_2d_0d(execution_policy, value, result_we_want);
    }


    // Mask a value.
    {
        f::MaskedArray<int, 2> value(f::extents[ft::nr_rows][ft::nr_cols]);
        std::iota(value.data(), value.data() + ft::nr_elements_2d, 5);
        get(value.mask(), 5) = true;
        f::MaskedScalar<int> result_we_want{5};
        verify_2d_0d_masked(execution_policy, value, result_we_want);
    }


    // Mask all values.
    {
        f::MaskedArray<int, 2> value(f::extents[ft::nr_rows][ft::nr_cols]);
        value.mask().fill(true);
        f::MaskedScalar<int> result_we_want{-9, true};
        verify_2d_0d_masked(execution_policy, value, result_we_want);
    }


    // Empty.
    {
        f::MaskedArray<int, 2> value(f::extents[0][0]);
        f::MaskedScalar<int> result_we_want{-9, true};
        verify_2d_0d_masked(execution_policy, value, result_we_want);
    }
}


FERN_UNARY_AGGREGATE_TEST_CASES()


BOOST_AUTO_TEST_CASE(gh59)
{
    f::MaskedArray<int, 2> value = {
        { 0, 0 },
        { 5, 0 },
    };
    value.mask() = {
        {  true, true },
        { false, true },
    };

    f::MaskedScalar<int> result_we_want{5};
    verify_2d_0d_masked(fa::sequential, value, result_we_want);
}


BOOST_AUTO_TEST_CASE(gh61)
{
    f::MaskedArray<int, 2> value = {
        { -9, -9, -9, -9, -9, -9 },
        { -9, -9,  8,  9, -9, -9 },
        { -9, 13, 14, 15, 16, -9 },
        { 18, 19, 20, 21, 22, 23 },
        { -9, 25, 26, 27, 28, -9 },
        { -9, -9, 32, 33, -9, -9 },
        { -9, -9, 38, -9, -9, -9 },
    };
    value.mask() = {
        { true,  true,  true,  true,  true,  true  },
        { true,  true,  false, false, true,  true  },
        { true,  false, false, false, false, true  },
        { false, false, false, false, false, false },
        { true,  false, false, false, false, true  },
        { true,  true,  false, false, true,  true  },
        { true,  true,  false, true,  true,  true  },
    };

    {
        f::MaskedScalar<int> result_we_want{8};
        verify_2d_0d_masked(fa::sequential, value, result_we_want);
    }

    for(size_t n = 1; n <= 8; ++n) {
        f::MaskedScalar<int> result_we_want{8};
        fa::ParallelExecutionPolicy parallel(n);
        verify_2d_0d_masked(parallel, value, result_we_want);
    }
}
