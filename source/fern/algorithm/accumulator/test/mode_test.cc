#define BOOST_TEST_MODULE fern algorithm accumulator
#include <boost/test/unit_test.hpp>
#include "fern/algorithm/accumulator/mode.h"


namespace faa = fern::algorithm::accumulator;

BOOST_AUTO_TEST_SUITE(mode)

BOOST_AUTO_TEST_CASE(default_construct)
{
    faa::Mode<int> mode;
    // Don't calculate the mode.
}


BOOST_AUTO_TEST_CASE(accumulate)
{
    {
        faa::Mode<int> mode(5);
        // 5
        BOOST_CHECK_EQUAL(mode(), 5);

        mode(2);
        // 2
        // 5
        BOOST_CHECK_EQUAL(mode(), 2);

        mode(5);
        // 2
        // 5 5
        BOOST_CHECK_EQUAL(mode(), 5);

        mode = 8;
        // 8
        BOOST_CHECK_EQUAL(mode(), 8);
    }

    {
        faa::Mode<int, double> mode(5);
        BOOST_CHECK_EQUAL(mode(), 5.0);
    }
}


BOOST_AUTO_TEST_CASE(merge)
{
    auto mode(faa::Mode<int>(5) | faa::Mode<int>(15) | faa::Mode<int>(15));
    BOOST_CHECK_EQUAL(mode(), 15);
}

BOOST_AUTO_TEST_SUITE_END()
