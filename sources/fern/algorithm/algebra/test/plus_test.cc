#define BOOST_TEST_MODULE fern algorithm algebra plus
#include <boost/test/unit_test.hpp>
#include "fern/feature/core/array_traits.h"
#include "fern/feature/core/masked_array_traits.h"
#include "fern/core/vector_traits.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/algebra/count.h"
#include "fern/algorithm/algebra/equal.h"
#include "fern/algorithm/algebra/executor.h"
#include "fern/algorithm/algebra/plus.h"


template<
    class A1,
    class A2,
    class R>
void verify_value(
    A1 const& argument1,
    A2 const& argument2,
    R const& result_we_want)
{
    // verify_result_type(A1, A2, R);
    fern::algebra::Plus<A1, A2> operation;
    // BOOST_CHECK_EQUAL(operation(argument1, argument2), result_we_want);

    R result_we_get;
    operation(argument1, argument2, result_we_get);
    BOOST_CHECK_EQUAL(result_we_get, result_we_want);
}


BOOST_AUTO_TEST_SUITE(plus)

BOOST_AUTO_TEST_CASE(value)
{
    verify_value<int8_t, int8_t, int8_t>(-5, 6, 1);

    verify_value<uint16_t, int8_t>(fern::TypeTraits<uint16_t>::max, 2,
        int32_t(fern::TypeTraits<uint16_t>::max) + int32_t(2));

    verify_value<uint32_t, int8_t>(fern::TypeTraits<uint32_t>::max, 2,
        int64_t(fern::TypeTraits<uint32_t>::max) + int64_t(2));

    verify_value<uint64_t, int64_t>(
        fern::TypeTraits<uint64_t>::max,
        fern::TypeTraits<int64_t>::max,
        int64_t(fern::TypeTraits<uint64_t>::max) +
            fern::TypeTraits<int64_t>::max);
}


template<
    class A1,
    class A2>
struct DomainPolicyHost:
    public fern::plus::OutOfDomainPolicy<A1, A2>
{
};


BOOST_AUTO_TEST_CASE(domain)
{
    {
        DomainPolicyHost<int32_t, int32_t> domain;
        BOOST_CHECK(domain.within_domain(-1, 2));
    }
    {
        DomainPolicyHost<uint8_t, double> domain;
        BOOST_CHECK(domain.within_domain(1, 2.0));
    }
}


template<
    class A1,
    class A2>
struct RangePolicyHost:
    public fern::plus::OutOfRangePolicy<A1, A2>
{
};


template<
    class A1,
    class A2>
void verify_range_check(
    A1 const& argument1,
    A2 const& argument2,
    bool const within)
{
    fern::algebra::Plus<A1, A2> operation;
    typename fern::algebra::Plus<A1, A2>::R result;
    RangePolicyHost<A1, A2> range;

    operation(argument1, argument2, result);
    BOOST_CHECK_EQUAL((range.within_range(argument1, argument2, result)),
        within);
}


BOOST_AUTO_TEST_CASE(range)
{
    int8_t const min_int8 = fern::TypeTraits<int8_t>::min;
    int8_t const max_int8 = fern::TypeTraits<int8_t>::max;
    uint8_t const max_uint8 = fern::TypeTraits<uint8_t>::max;
    uint16_t const max_uint16 = fern::TypeTraits<uint16_t>::max;
    uint32_t const max_uint32 = fern::TypeTraits<uint32_t>::max;
    int64_t const min_int64 = fern::TypeTraits<int64_t>::min;
    int64_t const max_int64 = fern::TypeTraits<int64_t>::max;
    uint64_t const max_uint64 = fern::TypeTraits<uint64_t>::max;

    // signed + signed
    verify_range_check<int8_t, int8_t>(-5, 6, true);
    verify_range_check<int8_t, int8_t>(max_int8, 1, false);
    verify_range_check<int8_t, int8_t>(min_int8, -1, false);
    verify_range_check<int64_t, int64_t>(min_int64, -1, false);

    // unsigned + unsigned
    verify_range_check<uint8_t, uint8_t>(5, 6, true);
    verify_range_check<uint8_t, uint8_t>(max_uint8, 1, false);
    verify_range_check<uint8_t, uint16_t>(max_uint8, 1, true);

    // signed + unsigned
    // unsigned + signed
    verify_range_check<int8_t, uint8_t>(5, 6, true);
    verify_range_check<uint8_t, int8_t>(5, 6, true);
    verify_range_check<uint16_t, int8_t>(max_uint16, 2, true);
    verify_range_check<uint32_t, int8_t>(max_uint32, 2, true);
    verify_range_check<uint64_t, int64_t>(max_uint64, max_int64, false);

    // float + float
    float const max_float32 = fern::TypeTraits<float>::max;
    verify_range_check<float, float>(5.0, 6.0, true);
    verify_range_check<float, float>(max_float32, max_float32 * 20, false);

    // float + signed
    // unsigned + float
    verify_range_check<float, int8_t>(5.0, 6, true);
    verify_range_check<uint8_t, float>(5, 6.0, true);
}


BOOST_AUTO_TEST_CASE(argument_types)
{
    // Verify that we can pass in all kinds of collection types.

    // constant + constant
    {
        uint8_t argument1(5);
        uint8_t argument2(6);
        typedef fern::Result<uint8_t, uint8_t>::type R;
        R result;

        fern::algebra::plus(argument1, argument2, result);

        BOOST_CHECK_EQUAL(result, 11u);
    }

    // constant + vector
    {
        uint8_t argument1(5);
        std::vector<uint8_t> argument2({1, 2, 3});
        typedef fern::Result<uint8_t, uint8_t>::type R;
        std::vector<R> result(argument2.size());

        fern::algebra::plus(argument1, argument2, result);

        BOOST_REQUIRE_EQUAL(result.size(), 3u);
        BOOST_CHECK_EQUAL(result[0], 6u);
        BOOST_CHECK_EQUAL(result[1], 7u);
        BOOST_CHECK_EQUAL(result[2], 8u);
    }

    // vector + constant
    {
        std::vector<uint8_t> argument1({1, 2, 3});
        uint8_t argument2(5);
        typedef fern::Result<uint8_t, uint8_t>::type R;
        std::vector<R> result(argument1.size());

        fern::algebra::plus(argument1, argument2, result);

        BOOST_REQUIRE_EQUAL(result.size(), 3u);
        BOOST_CHECK_EQUAL(result[0], 6u);
        BOOST_CHECK_EQUAL(result[1], 7u);
        BOOST_CHECK_EQUAL(result[2], 8u);
    }

    // vector + vector
    {
        std::vector<uint8_t> argument1({1, 2, 3});
        std::vector<uint8_t> argument2({4, 5, 6});
        typedef fern::Result<uint8_t, uint8_t>::type R;
        std::vector<R> result(argument1.size());

        fern::algebra::plus(argument1, argument2, result);

        BOOST_REQUIRE_EQUAL(result.size(), 3u);
        BOOST_CHECK_EQUAL(result[0], 5u);
        BOOST_CHECK_EQUAL(result[1], 7u);
        BOOST_CHECK_EQUAL(result[2], 9u);
    }

    // array + array
    {
        fern::Array<int8_t, 2> argument(fern::extents[3][2]);
        argument[0][0] = -2;
        argument[0][1] = -1;
        argument[1][0] =  0;
        argument[1][1] =  9;
        argument[2][0] =  1;
        argument[2][1] =  2;
        typedef fern::Result<int8_t, int8_t>::type R;
        fern::Array<R, 2> result(fern::extents[3][2]);

        fern::algebra::plus(argument, argument, result);

        BOOST_CHECK_EQUAL(result[0][0], -4);
        BOOST_CHECK_EQUAL(result[0][1], -2);
        BOOST_CHECK_EQUAL(result[1][0],  0);
        BOOST_CHECK_EQUAL(result[1][1], 18);
        BOOST_CHECK_EQUAL(result[2][0],  2);
        BOOST_CHECK_EQUAL(result[2][1],  4);
    }

    // masked_array + masked_array
    {
        fern::MaskedArray<int8_t, 2> argument(fern::extents[3][2]);
        argument[0][0] = -2;
        argument[0][1] = -1;
        argument[1][0] =  0;
        argument.mask()[1][1] =  true;
        argument[1][1] =  9;
        argument[2][0] =  1;
        argument[2][1] =  2;
        typedef fern::Result<int8_t, int8_t>::type R;
        fern::MaskedArray<R, 2> result(fern::extents[3][2]);

        fern::algebra::plus(argument, argument, result);

        BOOST_CHECK(!result.mask()[0][0]);
        BOOST_CHECK_EQUAL(result[0][0], -4);

        BOOST_CHECK(!result.mask()[0][1]);
        BOOST_CHECK_EQUAL(result[0][1], -2);

        BOOST_CHECK(!result.mask()[1][0]);
        BOOST_CHECK_EQUAL(result[1][0],  0);

        // Although the input data has a mask, the default policy discards
        // it. So the result doesn't have masked values.
        BOOST_CHECK(!result.mask()[1][1]);
        BOOST_CHECK_EQUAL(result[1][1], 18);

        BOOST_CHECK(!result.mask()[2][0]);
        BOOST_CHECK_EQUAL(result[2][0],  2);
        BOOST_CHECK(!result.mask()[2][1]);
        BOOST_CHECK_EQUAL(result[2][1],  4);
    }
}


BOOST_AUTO_TEST_CASE(no_data)
{
    size_t const nr_rows = 3;
    size_t const nr_cols = 2;
    auto extents = fern::extents[nr_rows][nr_cols];

    fern::MaskedArray<int8_t, 2> argument1(extents);
    argument1[0][0] = -2;
    argument1[0][1] = -1;
    argument1[1][0] =  0;
    argument1.mask()[1][1] =  true;
    argument1[2][0] =  1;
    argument1[2][1] =  2;

    fern::MaskedArray<int8_t, 2> argument2(extents);
    argument2[0][0] = -2;
    argument2[0][1] = -1;
    argument2[1][0] =  0;
    argument2[1][1] =  9;
    argument2.mask()[2][0] =  true;
    argument2[2][1] =  2;

    int8_t argument3 = 5;

    // masked_array + masked_array
    {
        // Create room for the result.
        // Set the mask.
        typedef fern::Result<int8_t, int8_t>::type R;
        fern::MaskedArray<R, 2> result(extents);
        result.set_mask(argument1.mask(), true);
        result.set_mask(argument2.mask(), true);

        typedef decltype(argument1) A1;
        typedef decltype(argument2) A2;
        typedef fern::ArgumentTraits<A1>::value_type A1Value;
        typedef fern::ArgumentTraits<A2>::value_type A2Value;
        typedef fern::DiscardDomainErrors<A1Value, A2Value> OutOfDomainPolicy;
        typedef fern::plus::OutOfRangePolicy<A1Value, A2Value> OutOfRangePolicy;
        typedef fern::MarkNoDataByValue<bool, fern::Mask<2>> NoDataPolicy;
        typedef fern::algebra::Plus<A1, A2, OutOfDomainPolicy, OutOfRangePolicy,
            NoDataPolicy> Plus;

        Plus plus(NoDataPolicy(result.mask(), true));

        plus(argument1, argument2, result);

        BOOST_CHECK(!result.mask()[0][0]);
        BOOST_CHECK_EQUAL(result[0][0], -4);

        BOOST_CHECK(!result.mask()[0][1]);
        BOOST_CHECK_EQUAL(result[0][1], -2);

        BOOST_CHECK(!result.mask()[1][0]);
        BOOST_CHECK_EQUAL(result[1][0],  0);

        BOOST_CHECK( result.mask()[1][1]);
        // Value is masked: it is undefined.
        // BOOST_CHECK_EQUAL(result[1][1], 18);

        BOOST_CHECK(result.mask()[2][0]);
        // Value is masked.
        // BOOST_CHECK_EQUAL(result[2][0],  2);

        BOOST_CHECK(!result.mask()[2][1]);
        BOOST_CHECK_EQUAL(result[2][1],  4);
    }

    // masked_array + 5
    {
        // Create room for the result.
        // Set the mask.
        typedef fern::Result<int8_t, int8_t>::type R;
        fern::MaskedArray<R, 2> result(extents);
        result.set_mask(argument1.mask(), true);

        typedef decltype(argument1) A1;
        typedef decltype(argument3) A2;
        typedef fern::ArgumentTraits<A1>::value_type A1Value;
        typedef fern::ArgumentTraits<A2>::value_type A2Value;
        typedef fern::DiscardDomainErrors<A1Value, A2Value> OutOfDomainPolicy;
        typedef fern::plus::OutOfRangePolicy<A1Value, A2Value> OutOfRangePolicy;
        typedef fern::MarkNoDataByValue<bool, fern::Mask<2>> NoDataPolicy;
        typedef fern::algebra::Plus<A1, A2, OutOfDomainPolicy, OutOfRangePolicy,
            NoDataPolicy> Plus;

        Plus plus(NoDataPolicy(result.mask(), true));

        plus(argument1, argument3, result);

        BOOST_CHECK(!result.mask()[0][0]);
        BOOST_CHECK_EQUAL(result[0][0], 3);

        BOOST_CHECK(!result.mask()[0][1]);
        BOOST_CHECK_EQUAL(result[0][1], 4);

        BOOST_CHECK(!result.mask()[1][0]);
        BOOST_CHECK_EQUAL(result[1][0], 5);

        BOOST_CHECK( result.mask()[1][1]);

        BOOST_CHECK(!result.mask()[2][0]);
        BOOST_CHECK_EQUAL(result[2][0], 6);

        BOOST_CHECK(!result.mask()[2][1]);
        BOOST_CHECK_EQUAL(result[2][1], 7);
    }

    // 5 + masked_array
    {
        // Create room for the result.
        // Set the mask.
        typedef fern::Result<int8_t, int8_t>::type R;
        fern::MaskedArray<R, 2> result(extents);
        result.set_mask(argument1.mask(), true);

        typedef decltype(argument3) A1;
        typedef decltype(argument1) A2;
        typedef fern::ArgumentTraits<A1>::value_type A1Value;
        typedef fern::ArgumentTraits<A2>::value_type A2Value;
        typedef fern::DiscardDomainErrors<A1Value, A2Value> OutOfDomainPolicy;
        typedef fern::plus::OutOfRangePolicy<A1Value, A2Value> OutOfRangePolicy;
        typedef fern::MarkNoDataByValue<bool, fern::Mask<2>> NoDataPolicy;
        typedef fern::algebra::Plus<A1, A2, OutOfDomainPolicy, OutOfRangePolicy,
            NoDataPolicy> Plus;

        Plus plus(NoDataPolicy(result.mask(), true));

        plus(argument3, argument1, result);

        BOOST_CHECK(!result.mask()[0][0]);
        BOOST_CHECK_EQUAL(result[0][0], 3);

        BOOST_CHECK(!result.mask()[0][1]);
        BOOST_CHECK_EQUAL(result[0][1], 4);

        BOOST_CHECK(!result.mask()[1][0]);
        BOOST_CHECK_EQUAL(result[1][0], 5);

        BOOST_CHECK( result.mask()[1][1]);

        BOOST_CHECK(!result.mask()[2][0]);
        BOOST_CHECK_EQUAL(result[2][0], 6);

        BOOST_CHECK(!result.mask()[2][1]);
        BOOST_CHECK_EQUAL(result[2][1], 7);
    }
}


BOOST_AUTO_TEST_CASE(threading)
{
    // Create a somewhat larger array.
    size_t const nr_rows = 6000;
    size_t const nr_cols = 4000;
    auto extents = fern::extents[nr_rows][nr_cols];
    fern::Array<int8_t, 2> argument1(extents);

    // Fill it with 0, 1, 2, 3, ...
    std::iota(argument1.data(), argument1.data() + argument1.num_elements(), 0);

    int8_t argument2 = 5;

    // Create array with values that should be in the result.
    fern::Array<int8_t, 2> result_we_want(extents);
    std::iota(result_we_want.data(), result_we_want.data() +
        result_we_want.num_elements(), 5);

    {
        // Add collection and constant.
        fern::algebra::Plus<fern::Array<int8_t, 2>, int8_t> plus;
        fern::Array<int8_t, 2> plus_result(extents);

        // Compare result with result we want.
        fern::algebra::Equal<fern::Array<int8_t, 2>, fern::Array<int8_t, 2>>
            equal;
        fern::Array<bool, 2> equal_result(extents);

        // Count the number of equal values.
        fern::algebra::Count<fern::Array<bool, 2>, bool> count;
        size_t count_result;

        // Serial.
        {
            fern::serial::execute(plus, argument1, argument2, plus_result);
            fern::serial::execute(equal, plus_result, result_we_want,
                equal_result);
            fern::serial::execute(count, equal_result, true, count_result);

            // Make sure the number of equal values equals the number of
            // elements.
            BOOST_CHECK_EQUAL(count_result, equal_result.num_elements());
        }

        // Concurrent.
        {
            fern::concurrent::execute(plus, argument1, argument2, plus_result);
            fern::concurrent::execute(equal, plus_result, result_we_want,
                equal_result);
            fern::concurrent::execute(count, equal_result, true, count_result);

            // Make sure the number of equal values equals the number of
            // elements.
            BOOST_CHECK_EQUAL(count_result, equal_result.num_elements());
        }
    }

    // TODO Make this work for masked arrays too.
}

BOOST_AUTO_TEST_SUITE_END()
