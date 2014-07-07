#define BOOST_TEST_MODULE fern algorithm algebra elementary add
#include <boost/test/unit_test.hpp>
#include "fern/core/constant_traits.h"
#include "fern/algorithm/algebra/elementary/add.h"


BOOST_AUTO_TEST_SUITE(add)

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
    fern::algebra::add(fern::sequential, value1, value2, result_we_get);
    BOOST_CHECK_EQUAL(result_we_get, result_we_want);
}


BOOST_AUTO_TEST_CASE(d0_array_d0_array)
{
    verify_value<int8_t, int8_t, int8_t>(-5, 6, 1);
    verify_value<int8_t, int8_t, int8_t>(-5, -5, -10);
    verify_value<double, uint8_t, double>(-5.5, 5, -0.5);
}

BOOST_AUTO_TEST_SUITE_END()
