#define BOOST_TEST_MODULE fern algorithm algebra result_type
#include <cxxabi.h>
#include <boost/test/unit_test.hpp>
#include "fern/algorithm/algebra/result_type.h"


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


// Works for types known to the TypeTraits.
#define verify_result_type(                                                    \
    A1, A2, TypeWeWant)                                                        \
{                                                                              \
    typedef typename fern::result<A1, A2>::type TypeWeGet;                     \
                                                                               \
    BOOST_CHECK_MESSAGE((std::is_same<TypeWeGet, TypeWeWant>()),               \
        fern::TypeTraits<TypeWeGet>::name + " != " +                           \
        fern::TypeTraits<TypeWeWant>::name);                                   \
}


// Works for all types.
#define verify_result_type2(                                                   \
    A1, A2, TypeWeWant)                                                        \
{                                                                              \
    typedef typename fern::result<A1, A2>::type TypeWeGet;                     \
                                                                               \
    BOOST_CHECK_MESSAGE((std::is_same<TypeWeGet, TypeWeWant>()),               \
        demangled_type_name<TypeWeGet>() + " != " +  \
        demangled_type_name<TypeWeWant>()); \
}


BOOST_AUTO_TEST_SUITE(plus)

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

BOOST_AUTO_TEST_SUITE_END()
