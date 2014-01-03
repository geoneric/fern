#define BOOST_TEST_MODULE fern algorithm algebra
#include <cxxabi.h>
#include <boost/test/unit_test.hpp>
#include "fern/core/type_traits.h"
#include "fern/algorithm/algebra/plus.h"


std::string demangle(
    std::string const& name)
{
    int status;
    char* buffer;
    buffer = abi::__cxa_demangle(name.c_str(), 0, 0, &status);
    assert(status == 0);
    std::string real_name(buffer);
    free(buffer);
    return real_name;
}


template<
    class T>
std::string demangled_type_name()
{
    return demangle(typeid(T).name());
}


BOOST_AUTO_TEST_SUITE(plus)

// Works for types known to the TypeTraits.
#define verify_result_type(                                                    \
    A1, A2, TypeWeWant)                                                        \
{                                                                              \
    typedef typename fern::Plus<A1, A2>::R TypeWeGet;                          \
                                                                               \
    BOOST_CHECK_MESSAGE((std::is_same<TypeWeGet, TypeWeWant>()),               \
        fern::TypeTraits<TypeWeGet>::name + " != " +                           \
        fern::TypeTraits<TypeWeWant>::name);                                   \
}


// Works for all types.
#define verify_result_type2(                                                   \
    A1, A2, TypeWeWant)                                                        \
{                                                                              \
    typedef typename fern::Plus<A1, A2>::R TypeWeGet;                          \
                                                                               \
    BOOST_CHECK_MESSAGE((std::is_same<TypeWeGet, TypeWeWant>()),               \
        demangled_type_name<TypeWeGet>() + " != " +  \
        demangled_type_name<TypeWeWant>()); \
}





BOOST_AUTO_TEST_CASE(result_type)
{
    // uint + uint
    // Pick largest uint type.
    verify_result_type(uint8_t, uint8_t, uint8_t);
    verify_result_type(uint8_t, uint16_t, uint16_t);
    verify_result_type(uint8_t, uint32_t, uint32_t);
    verify_result_type(uint8_t, uint64_t, uint64_t);

    verify_result_type(uint16_t, uint16_t, uint16_t);
    verify_result_type(uint16_t, uint32_t, uint32_t);
    verify_result_type(uint16_t, uint64_t, uint64_t);

    verify_result_type(uint32_t, uint32_t, uint32_t);
    verify_result_type(uint32_t, uint64_t, uint64_t);

    verify_result_type(uint64_t, uint64_t, uint64_t);

    // int + int
    // Pick largest int type.
    verify_result_type(int8_t, int8_t, int8_t);
    verify_result_type(int8_t, int16_t, int16_t);
    verify_result_type(int8_t, int32_t, int32_t);
    verify_result_type(int8_t, int64_t, int64_t);

    verify_result_type(int16_t, int16_t, int16_t);
    verify_result_type(int16_t, int32_t, int32_t);
    verify_result_type(int16_t, int64_t, int64_t);

    verify_result_type(int32_t, int32_t, int32_t);
    verify_result_type(int32_t, int64_t, int64_t);

    verify_result_type(int64_t, int64_t, int64_t);

    // uint + int
    // Pick a signed int type that can store values from both types. If there
    // is no such type, pick int64_t.
    verify_result_type(uint8_t, int8_t, int16_t);
    verify_result_type(uint8_t, int16_t, int16_t);
    verify_result_type(uint8_t, int32_t, int32_t);
    verify_result_type(uint8_t, int64_t, int64_t);

    verify_result_type(uint16_t, int8_t, int32_t);
    verify_result_type(uint16_t, int16_t, int32_t);
    verify_result_type(uint16_t, int32_t, int32_t);
    verify_result_type(uint16_t, int64_t, int64_t);

    verify_result_type(uint32_t, int8_t, int64_t);
    verify_result_type(uint32_t, int16_t, int64_t);
    verify_result_type(uint32_t, int32_t, int64_t);
    verify_result_type(uint32_t, int64_t, int64_t);

    verify_result_type(uint64_t, int8_t, int64_t);
    verify_result_type(uint64_t, int16_t, int64_t);
    verify_result_type(uint64_t, int32_t, int64_t);
    verify_result_type(uint64_t, int64_t, int64_t);

    // float + float
    // Pick the largest float type.
    verify_result_type(float, float, float);
    verify_result_type(double, double, double);
    verify_result_type(float, double, double);

    // uint + float
    verify_result_type(uint8_t, float, float);
    verify_result_type(uint8_t, double, double);
    verify_result_type(uint16_t, float, float);
    verify_result_type(uint16_t, double, double);
    verify_result_type(uint32_t, float, float);
    verify_result_type(uint32_t, double, double);
    verify_result_type(uint64_t, float, float);
    verify_result_type(uint64_t, double, double);

    // int + float
    verify_result_type(int8_t, float, float);
    verify_result_type(int8_t, double, double);
    verify_result_type(int16_t, float, float);
    verify_result_type(int16_t, double, double);
    verify_result_type(int32_t, float, float);
    verify_result_type(int32_t, double, double);
    verify_result_type(int64_t, float, float);
    verify_result_type(int64_t, double, double);

    // Collections.
    verify_result_type2(int8_t, std::vector<int8_t>, std::vector<int8_t>);
    verify_result_type2(int8_t, std::vector<float>, std::vector<float>);
    verify_result_type2(float, std::vector<int8_t>, std::vector<float>);

    verify_result_type2(std::vector<int8_t>, int8_t, std::vector<int8_t>);
    verify_result_type2(std::vector<float>, int8_t, std::vector<float>);
    verify_result_type2(std::vector<int8_t>, float, std::vector<float>);

    verify_result_type2(std::vector<int8_t>, std::vector<int8_t>,
        std::vector<int8_t>);
    verify_result_type2(std::vector<float>, std::vector<int8_t>,
        std::vector<float>);
    verify_result_type2(std::vector<int8_t>, std::vector<float>,
        std::vector<float>);
}


template<
    class A1,
    class A2,
    class R>
void verify_value(
    A1 const& argument1,
    A2 const& argument2,
    R const& result_we_want)
{
    verify_result_type(A1, A2, R);
    fern::Plus<A1, A2> operation;
    BOOST_CHECK_EQUAL(operation(argument1, argument2), result_we_want);

    R result_we_get;
    operation(argument1, argument2, result_we_get);
    BOOST_CHECK_EQUAL(result_we_get, result_we_want);
}


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
}


BOOST_AUTO_TEST_CASE(no_data)
{
    // Declare a protected, non-virtual (and usually empty) destructor
    // for the policy, preventing policy objects from being deleted (or
    // instantiated, for that matter).

    // fern::Plus<A1, A2, fern::plus::Range> operation;



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


