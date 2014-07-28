#define BOOST_TEST_MODULE fern algorithm core fill
#include <boost/test/unit_test.hpp>
#include "fern/core/constant_traits.h"
#include "fern/algorithm/core/fill.h"


BOOST_AUTO_TEST_SUITE(fill)

template<
    class Value,
    class Result>
void verify_value(
    Value const& value,
    Result const& result_we_want)
{
    Result result_we_get;
    fern::core::fill(fern::sequential, value, result_we_get);
    BOOST_CHECK_EQUAL(result_we_get, result_we_want);
}


BOOST_AUTO_TEST_CASE(algorithm)
{
    verify_value<int8_t, int8_t>( 5, 5);
    verify_value<int8_t, int8_t>(-5, -5);
    verify_value<int8_t, int8_t>( 0, 0);

    verify_value<double, double>( 5.5, 5.5);
    verify_value<double, double>(-5.5, -5.5);
    verify_value<double, double>( 0.0, 0.0);
}

BOOST_AUTO_TEST_SUITE_END()
