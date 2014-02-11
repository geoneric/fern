#define BOOST_TEST_MODULE fern algorithm algebra
#include <boost/test/unit_test.hpp>
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/algebra/array_traits.h"
#include "fern/algorithm/algebra/masked_array_traits.h"
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
    fern::Plus<A1, A2> operation;
    BOOST_CHECK_EQUAL(operation(argument1, argument2), result_we_want);

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
    public fern::plus::Domain<A1, A2>
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
    public fern::plus::Range<A1, A2>
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
    fern::Plus<A1, A2> operation;
    RangePolicyHost<A1, A2> range;
    BOOST_CHECK_EQUAL((range.within_range(argument1, argument2,
        operation(argument1, argument2))), within);
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
        typedef fern::result<uint8_t, uint8_t>::type R;

        R result = fern::algebra::plus(argument1, argument2);

        BOOST_CHECK_EQUAL(result, 11u);
    }

    // constant + vector
    {
        uint8_t argument1(5);
        std::vector<uint8_t> argument2({1, 2, 3});
        typedef fern::result<uint8_t, uint8_t>::type R;
        // std::vector<R> result(argument2.size());

        std::vector<R> result = fern::algebra::plus(argument1, argument2);

        BOOST_REQUIRE_EQUAL(result.size(), 3u);
        BOOST_CHECK_EQUAL(result[0], 6u);
        BOOST_CHECK_EQUAL(result[1], 7u);
        BOOST_CHECK_EQUAL(result[2], 8u);
    }

    // vector + constant
    {
        std::vector<uint8_t> argument1({1, 2, 3});
        uint8_t argument2(5);
        typedef fern::result<uint8_t, uint8_t>::type R;
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
        typedef fern::result<uint8_t, uint8_t>::type R;
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
        typedef fern::result<int8_t, int8_t>::type R;
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
        typedef fern::result<int8_t, int8_t>::type R;
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
    // Declare a protected, non-virtual (and usually empty) destructor
    // for the policy, preventing policy objects from being deleted (or
    // instantiated, for that matter).

    // fern::Plus<A1, A2, fern::plus::Range> operation;

    // TODO Verify that no-data is handled correctly.


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
        typedef fern::result<int8_t, int8_t>::type R;
        fern::MaskedArray<R, 2> result(fern::extents[3][2]);

        typedef decltype(argument) A1;
        typedef decltype(argument) A2;

        fern::Plus<A1, A2,
            fern::DiscardDomainErrors<A1, A2>,
            fern::plus::Range<A1, A2>,
            fern::MarkNoDataByValue<bool, fern::Mask<2>>> plus;

        plus(argument, argument, result);
        // fern::algebra::plus<(argument, argument, result);

        BOOST_CHECK(!result.mask()[0][0]);
        BOOST_CHECK_EQUAL(result[0][0], -4);

        BOOST_CHECK(!result.mask()[0][1]);
        BOOST_CHECK_EQUAL(result[0][1], -2);

        BOOST_CHECK(!result.mask()[1][0]);
        BOOST_CHECK_EQUAL(result[1][0],  0);

        BOOST_CHECK( result.mask()[1][1]);
        // Value is masked: it is undefined.
        // BOOST_CHECK_EQUAL(result[1][1], 18);

        BOOST_CHECK(!result.mask()[2][0]);
        BOOST_CHECK_EQUAL(result[2][0],  2);
        BOOST_CHECK(!result.mask()[2][1]);
        BOOST_CHECK_EQUAL(result[2][1],  4);
    }
}

BOOST_AUTO_TEST_SUITE_END()


// template<
//     class Operation,
//     class Argument1,
//     class Argument2,
//     class ResultTypeWeWant>
// void verify(
//     Operation const& operation,
//     Argument1 const& argument1,
//     Argument2 const& argument2,
//     ResultTypeWeWant const& result_value_we_want)
// {
//     typedef decltype(operation(argument1, argument2)) ResultTypeWeGet;
// 
//     // Check type.
//     BOOST_CHECK_MESSAGE((
//         std::is_same<ResultTypeWeGet, ResultTypeWeWant >::value),
//         typeid(ResultTypeWeGet).name() + std::string(" != ") +
//             typeid(ResultTypeWeWant).name());
// 
//     // Check value.
//     BOOST_CHECK_EQUAL(operation(argument1, argument2), result_value_we_want);
// }
// 
// 
// template<
//     class Argument1,
//     class Argument2>
// void verify(
//     Argument1 const& argument1,
//     Argument2 const& argument2)
// {
//     fern::Plus<Argument1, Argument2> operation;
// 
//     // The result type we want is whatever the C++ rules dictate.
//     typedef decltype(Argument1() + Argument2()) ResultTypeWeWant;
// 
//     // The result value we want is whatever the plus operator returns.
//     ResultTypeWeWant result_we_want = argument1 + argument2;
// 
//     verify(operation, argument1, argument2, result_we_want);
// }


// BOOST_AUTO_TEST_CASE(verify_value)
// {
//     // constant + constant
//     verify(int32_t(5), int32_t(6));
//     verify(int16_t(-5), int32_t(6));
//     verify(float(5), uint8_t(6));
//     verify(double(5), double(6));
// 
// 
//     // plus<int, int>
//     // plus<vector, vector>
//     // plus<vector, int>
//     // plus<int, vector>
// 
// 
//     // array + array
//     // fern::Range argument1({1, 2, 3});
//     // fern::Range argument2({1, 2, 3});
//     // verify(int32_t(5), int32_t(6));
// 
// 
//     // constant + array
//     // array + constant
// }
// 
// 
// BOOST_AUTO_TEST_CASE(out_of_range)
// {
//     // fern::Plus<Argument1, Argument2, > operation;
// 
//     // // The result type we want is whatever the C++ rules dictate.
//     // typedef decltype(Argument1() + Argument2()) ResultTypeWeWant;
// 
//     // // The result value we want is whatever the plus operator returns.
//     // ResultTypeWeWant result_we_want = argument1 + argument2;
// 
//     // verify(operation, argument1, argument2, result_we_want);
// }


