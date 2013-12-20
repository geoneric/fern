#define BOOST_TEST_MODULE fern algrorithm algebra
#include <boost/test/unit_test.hpp>
#include "fern/core/type_traits.h"
#include "fern/algorithm/algebra/plus.h"


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


BOOST_AUTO_TEST_SUITE(plus)

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


// std::string demangle(
//     std::string const& name)
// {
//     int status;
//     char* buffer;
//     buffer = abi::__cxa_demangle(name.c_str(), 0, 0, &status);
//     assert(status == 0);
//     std::string real_name(buffer);
//     free(buffer);
//     return real_name;
// }
// 
// 
// template<
//     class T>
// std::string demangled_type_name()
// {
//     return demangle(typeid(T).name());
// }


template<
    class A1,
    class A2,
    class TypeWeWant>
void verify_result_type()
{
    typedef typename fern::Plus<A1, A2>::R TypeWeGet;

    BOOST_CHECK_MESSAGE((std::is_same<TypeWeGet, TypeWeWant>()),
        fern::TypeTraits<TypeWeGet>::name + " != " +
        fern::TypeTraits<TypeWeWant>::name);
}


#define verify_result_type(                                                    \
    A1, A2, TypeWeWant)                                                        \
{                                                                              \
    typedef typename fern::Plus<A1, A2>::R TypeWeGet;                          \
                                                                               \
    BOOST_CHECK_MESSAGE((std::is_same<TypeWeGet, TypeWeWant>()),               \
        fern::TypeTraits<TypeWeGet>::name + " != " +                           \
        fern::TypeTraits<TypeWeWant>::name);                                   \
}


BOOST_AUTO_TEST_CASE(result_type)
{
    // TODO This test verfies the type promotion rules. This is not specific
    //      to plus. Put these tests elsewhere.

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
}


template<
    class A1,
    class A2,
    class R>
void verify_value(
    A1 const& argument1,
    A2 const& argument2,
    R const& result)
{
    fern::Plus<A1, A2> operation;
    BOOST_CHECK_EQUAL(operation(argument1, argument2), result);
}


BOOST_AUTO_TEST_CASE(value)
{
    verify_value<uint16_t, int8_t>(fern::TypeTraits<uint16_t>::max, 2,
        int32_t(fern::TypeTraits<uint16_t>::max) + int32_t(2));
}


BOOST_AUTO_TEST_CASE(domain)
{
    fern::plus::Domain domain;
    BOOST_CHECK(domain.within_domain(-1, 2));
    BOOST_CHECK((domain.within_domain<uint8_t, double>(1, 2.0)));
}


template<
    class A1,
    class A2>
void verify_range_check(
    A1 const& argument1,
    A2 const& argument2,
    bool const within)
{
    fern::Plus<A1, A2> operation;
    fern::plus::Range range;
    BOOST_CHECK_EQUAL((range.within_range<A1, A2>(argument1, argument2,
        operation(argument1, argument2))), within);
}


// BOOST_AUTO_TEST_CASE(range)
// {
//     int8_t const min_int8 = fern::TypeTraits<int8_t>::min;
//     int8_t const max_int8 = fern::TypeTraits<int8_t>::max;
//     uint8_t const max_uint8 = fern::TypeTraits<uint8_t>::max;
//     uint16_t const max_uint16 = fern::TypeTraits<uint16_t>::max;
//     uint32_t const max_uint32 = fern::TypeTraits<uint32_t>::max;
// 
//     // signed + signed
//     verify_range_check<int8_t, int8_t>(-5, 6, true);
//     verify_range_check<int8_t, int8_t>(max_int8, 1, false);
//     verify_range_check<int8_t, int8_t>(min_int8, -1, false);
// 
//     // unsigned + unsigned
//     verify_range_check<uint8_t, uint8_t>(5, 6, true);
//     verify_range_check<uint8_t, uint8_t>(max_uint8, 1, false);
// 
//     // signed + unsigned
//     // unsigned + signed
//     verify_range_check<int8_t, uint8_t>(5, 6, true);
//     verify_range_check<uint8_t, int8_t>(5, 6, true);
// 
//     verify_range_check<uint16_t, int8_t>(max_uint16, 2, true);
//     // TODO hier verder
//     //      compileert niet. waarom niet?
//     //      list the final promotion rules.
//     //      uint32_t + int8_t -> uint32_t ?!
//     //      get this promotion under control...
//     // verify_range_check<uint32_t, int8_t>(max_uint32, 2, true);
//     // TODO overflow examples
// 
//     // float + float
//     float const max_float32 = std::numeric_limits<float>::max();
//     verify_range_check<float, float>(5.0, 6.0, true);
//     verify_range_check<float, float>(max_float32, max_float32 * 20, false);
// 
//     // float + signed
//     // unsigned + float
//     verify_range_check<float, int8_t>(5.0, 6, true);
//     verify_range_check<uint8_t, float>(5, 6.0, true);
//     // TODO overflow examples
// }

BOOST_AUTO_TEST_SUITE_END()
