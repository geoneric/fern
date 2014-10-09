#define BOOST_TEST_MODULE fern algorithm algebra boole not_
#include <boost/test/unit_test.hpp>
#include "fern/core/constant_traits.h"
#include "fern/core/type_traits.h"
#include "fern/algorithm/algebra/boole/not.h"


namespace fa = fern::algorithm;


BOOST_AUTO_TEST_SUITE(not_)

template<
    class Value,
    class Result>
void verify_value(
    Value const& value,
    Result const& result_we_want)
{
    Result result_we_get;
    fa::algebra::not_(fa::sequential, value, result_we_get);
    BOOST_CHECK_EQUAL(result_we_get, result_we_want);
}



BOOST_AUTO_TEST_CASE(algorithm)
{
    verify_value<bool, bool>(true, false);
    verify_value<bool, bool>(false, true);
}

BOOST_AUTO_TEST_SUITE_END()
