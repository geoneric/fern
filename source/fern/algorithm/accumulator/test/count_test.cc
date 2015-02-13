#define BOOST_TEST_MODULE fern algorithm accumulator
#include <boost/test/unit_test.hpp>
#include "fern/algorithm/accumulator/count.h"


namespace faa = fern::algorithm::accumulator;

BOOST_AUTO_TEST_SUITE(count)

BOOST_AUTO_TEST_CASE(default_construct)
{
    faa::Count<int> count;
    BOOST_CHECK_EQUAL(count(), 0);
}


BOOST_AUTO_TEST_CASE(accumulate)
{
    faa::Count<int> count(5);
    BOOST_CHECK_EQUAL(count(), 1);

    count(2);
    BOOST_CHECK_EQUAL(count(), 2);

    count = 3;
    BOOST_CHECK_EQUAL(count(), 1);
}


BOOST_AUTO_TEST_CASE(merge)
{
    auto count(faa::Count<int>(5) | faa::Count<int>(6));
    BOOST_CHECK_EQUAL(count(), 2);
}

BOOST_AUTO_TEST_SUITE_END()
