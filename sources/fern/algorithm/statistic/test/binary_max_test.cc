#define BOOST_TEST_MODULE fern algorithm statistic binary_max
#include <boost/test/unit_test.hpp>
#include "fern/core/constant_traits.h"
#include "fern/core/vector_traits.h"
#include "fern/feature/core/array_traits.h"
#include "fern/algorithm/statistic/binary_max.h"


namespace fa = fern::algorithm;


BOOST_AUTO_TEST_SUITE(binary_max)

template<
    class Value1,
    class Value2,
    class Result>
void verify_value(
    Value1 const& value1,
    Value2 const& value2,
    Result const& result_we_want)
{
    Result result_we_get;
    fa::statistic::binary_max(fa::sequential, value1, value2,
        result_we_get);
    BOOST_CHECK_EQUAL(result_we_get, result_we_want);
}


BOOST_AUTO_TEST_CASE(algorithm)
{
    {
        verify_value<int8_t, int8_t, int8_t>( 5,  5,  5);
        verify_value<int8_t, int8_t, int8_t>(-5, -5, -5);
        verify_value<int8_t, int8_t, int8_t>( 0,  0,  0);
        verify_value<int8_t, int8_t, int8_t>( 5, -5,  5);
        verify_value<int8_t, int8_t, int8_t>(-5,  5,  5);
    }
}

BOOST_AUTO_TEST_SUITE_END()
