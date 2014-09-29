#define BOOST_TEST_MODULE fern algorithm algebra boolean and_
#include <boost/test/unit_test.hpp>
#include "fern/core/constant_traits.h"
#include "fern/core/type_traits.h"
#include "fern/algorithm/algebra/boolean/and.h"


namespace fa = fern::algorithm;


BOOST_AUTO_TEST_SUITE(and_)

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
    fa::algebra::and_(fa::sequential, value1, value2, result_we_get);
    BOOST_CHECK_EQUAL(result_we_get, result_we_want);
}


BOOST_AUTO_TEST_CASE(algorithm)
{
    verify_value<>(true, true, true);
    verify_value<>(true, false, false);
    verify_value<>(false, true, false);
    verify_value<>(false, false, false);
}

BOOST_AUTO_TEST_SUITE_END()
