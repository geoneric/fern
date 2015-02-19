#define BOOST_TEST_MODULE fern algorithm accumulator
#include <boost/test/unit_test.hpp>
#include "fern/algorithm/accumulator/majority.h"


namespace faa = fern::algorithm::accumulator;

BOOST_AUTO_TEST_SUITE(majority)

BOOST_AUTO_TEST_CASE(default_construct)
{
    faa::Majority<int> majority;
    // Don't calculate the majority.
}


BOOST_AUTO_TEST_CASE(accumulate)
{
    {
        faa::Majority<int> majority(5);
        // 5
        BOOST_CHECK_EQUAL(majority(), 5);

        majority(2);
        // 2
        // 5
        BOOST_CHECK_EQUAL(majority(), 2);

        majority(5);
        // 2
        // 5 5
        BOOST_CHECK_EQUAL(majority(), 5);

        majority = 8;
        // 8
        BOOST_CHECK_EQUAL(majority(), 8);
    }

    {
        faa::Majority<int, double> majority(5);
        BOOST_CHECK_EQUAL(majority(), 5.0);
    }
}


BOOST_AUTO_TEST_CASE(merge)
{
    auto majority(faa::Majority<int>(5) | faa::Majority<int>(15) |
        faa::Majority<int>(15));
    BOOST_CHECK_EQUAL(majority(), 15);
}

BOOST_AUTO_TEST_SUITE_END()
