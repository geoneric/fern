#define BOOST_TEST_MODULE fern algorithm algebra boolean or_
#include <boost/test/unit_test.hpp>
#include "fern/core/constant_traits.h"
#include "fern/core/type_traits.h"
#include "fern/algorithm/algebra/boolean/or.h"


BOOST_AUTO_TEST_SUITE(or_)

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
    fern::algebra::or_(fern::sequential, value1, value2, result_we_get);
    BOOST_CHECK_EQUAL(result_we_get, result_we_want);
}


BOOST_AUTO_TEST_CASE(algorithm)
{
    verify_value<>(true, true, true);
    verify_value<>(true, false, true);
    verify_value<>(false, true, true);
    verify_value<>(false, false, false);
}

BOOST_AUTO_TEST_SUITE_END()
