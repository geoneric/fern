#define BOOST_TEST_MODULE fern algorithm algebra elementary greater_equal
#include <boost/test/unit_test.hpp>
#include "fern/core/constant_traits.h"
#include "fern/algorithm/algebra/elementary/greater_equal.h"


namespace fa = fern::algorithm;


BOOST_AUTO_TEST_SUITE(greater_equal)

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
    fa::algebra::greater_equal(fa::sequential, value1, value2,
        result_we_get);
    BOOST_CHECK_EQUAL(result_we_get, result_we_want);
}


BOOST_AUTO_TEST_CASE(algorithm)
{
    verify_value<float, float, bool>(0.0f, 0.0f, true);
    verify_value<float, float, bool>(1.0f, 1.0f, true);
    verify_value<float, float, bool>(-1.0f, -1.0f, true);

    verify_value<int, int, bool>(1, 2, false);
    verify_value<int, int, bool>(2, 1, true);

    verify_value<int, int, bool>(-1, -2, true);
    verify_value<int, int, bool>(-2, -1, false);
}

BOOST_AUTO_TEST_SUITE_END()
