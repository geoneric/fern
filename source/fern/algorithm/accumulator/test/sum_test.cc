#define BOOST_TEST_MODULE fern algorithm accumulator
#include <boost/test/unit_test.hpp>
#include "fern/algorithm/accumulator/sum.h"


namespace faa = fern::algorithm::accumulator;

BOOST_AUTO_TEST_SUITE(sum)

BOOST_AUTO_TEST_CASE(default_construct)
{
    faa::Sum<int> sum;
    BOOST_CHECK_EQUAL(sum(), 0);
}


BOOST_AUTO_TEST_CASE(accumulate)
{
    faa::Sum<int> sum(5);
    BOOST_CHECK_EQUAL(sum(), 5);

    sum(2);
    BOOST_CHECK_EQUAL(sum(), 7);

    sum = 3;
    BOOST_CHECK_EQUAL(sum(), 3);
}


BOOST_AUTO_TEST_CASE(merge)
{
    auto sum(faa::Sum<int>(5) | faa::Sum<int>(6));
    BOOST_CHECK_EQUAL(sum(), 11);
}

BOOST_AUTO_TEST_SUITE_END()
