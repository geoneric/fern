#define BOOST_TEST_MODULE fern algorithm core result_value
#include <boost/test/unit_test.hpp>
#include "fern/algorithm/core/result_value.h"


#define verify_result_value_type(                                              \
    A1, A2, TypeWeWant)                                                        \
{                                                                              \
    using TypeWeGet = typename fern::ResultValue<A1, A2>::type;                \
                                                                               \
    BOOST_CHECK_MESSAGE((std::is_same<TypeWeGet, TypeWeWant>()),               \
        fern::TypeTraits<TypeWeGet>::name + fern::String(" != ") +             \
        fern::TypeTraits<TypeWeWant>::name);                                   \
}


BOOST_AUTO_TEST_SUITE(result_value)

BOOST_AUTO_TEST_CASE(ResultValue)
{
    // uint + uint
    // Pick largest uint type.
    verify_result_value_type(uint8_t, uint8_t, uint8_t);
    verify_result_value_type(uint8_t, uint16_t, uint16_t);
    verify_result_value_type(uint8_t, uint32_t, uint32_t);
    verify_result_value_type(uint8_t, uint64_t, uint64_t);

    verify_result_value_type(uint16_t, uint16_t, uint16_t);
    verify_result_value_type(uint16_t, uint32_t, uint32_t);
    verify_result_value_type(uint16_t, uint64_t, uint64_t);

    verify_result_value_type(uint32_t, uint32_t, uint32_t);
    verify_result_value_type(uint32_t, uint64_t, uint64_t);

    verify_result_value_type(uint64_t, uint64_t, uint64_t);

    // int + int
    // Pick largest int type.
    verify_result_value_type(int8_t, int8_t, int8_t);
    verify_result_value_type(int8_t, int16_t, int16_t);
    verify_result_value_type(int8_t, int32_t, int32_t);
    verify_result_value_type(int8_t, int64_t, int64_t);

    verify_result_value_type(int16_t, int16_t, int16_t);
    verify_result_value_type(int16_t, int32_t, int32_t);
    verify_result_value_type(int16_t, int64_t, int64_t);

    verify_result_value_type(int32_t, int32_t, int32_t);
    verify_result_value_type(int32_t, int64_t, int64_t);

    verify_result_value_type(int64_t, int64_t, int64_t);

    // uint + int
    // Pick a signed int type that can store values from both types. If there
    // is no such type, pick int64_t.
    verify_result_value_type(uint8_t, int8_t, int16_t);
    verify_result_value_type(uint8_t, int16_t, int16_t);
    verify_result_value_type(uint8_t, int32_t, int32_t);
    verify_result_value_type(uint8_t, int64_t, int64_t);

    verify_result_value_type(uint16_t, int8_t, int32_t);
    verify_result_value_type(uint16_t, int16_t, int32_t);
    verify_result_value_type(uint16_t, int32_t, int32_t);
    verify_result_value_type(uint16_t, int64_t, int64_t);

    verify_result_value_type(uint32_t, int8_t, int64_t);
    verify_result_value_type(uint32_t, int16_t, int64_t);
    verify_result_value_type(uint32_t, int32_t, int64_t);
    verify_result_value_type(uint32_t, int64_t, int64_t);

    verify_result_value_type(uint64_t, int8_t, int64_t);
    verify_result_value_type(uint64_t, int16_t, int64_t);
    verify_result_value_type(uint64_t, int32_t, int64_t);
    verify_result_value_type(uint64_t, int64_t, int64_t);

    // float + float
    // Pick the largest float type.
    verify_result_value_type(float, float, float);
    verify_result_value_type(double, double, double);
    verify_result_value_type(float, double, double);

    // uint + float
    verify_result_value_type(uint8_t, float, float);
    verify_result_value_type(uint8_t, double, double);
    verify_result_value_type(uint16_t, float, float);
    verify_result_value_type(uint16_t, double, double);
    verify_result_value_type(uint32_t, float, float);
    verify_result_value_type(uint32_t, double, double);
    verify_result_value_type(uint64_t, float, float);
    verify_result_value_type(uint64_t, double, double);

    // int + float
    verify_result_value_type(int8_t, float, float);
    verify_result_value_type(int8_t, double, double);
    verify_result_value_type(int16_t, float, float);
    verify_result_value_type(int16_t, double, double);
    verify_result_value_type(int32_t, float, float);
    verify_result_value_type(int32_t, double, double);
    verify_result_value_type(int64_t, float, float);
    verify_result_value_type(int64_t, double, double);
}

BOOST_AUTO_TEST_SUITE_END()
