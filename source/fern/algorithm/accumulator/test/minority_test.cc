#define BOOST_TEST_MODULE fern algorithm accumulator
#include <boost/test/unit_test.hpp>
#include "fern/algorithm/accumulator/minority.h"


namespace faa = fern::algorithm::accumulator;

BOOST_AUTO_TEST_SUITE(minority)

BOOST_AUTO_TEST_CASE(default_construct)
{
    faa::Minority<int> minority;
    // Don't calculate the minority.
}


BOOST_AUTO_TEST_CASE(accumulate)
{
    {
        faa::Minority<int> minority(5);
        // 5
        BOOST_CHECK_EQUAL(minority(), 5);

        minority(2);
        // 2
        // 5
        BOOST_CHECK_EQUAL(minority(), 2);

        minority(5);
        // 2
        // 5 5
        BOOST_CHECK_EQUAL(minority(), 2);

        minority = 8;
        // 8
        BOOST_CHECK_EQUAL(minority(), 8);
    }

    {
        faa::Minority<int, double> minority(5);
        BOOST_CHECK_EQUAL(minority(), 5.0);
    }
}


BOOST_AUTO_TEST_CASE(merge)
{
    auto minority(faa::Minority<int>(5) | faa::Minority<int>(15) |
        faa::Minority<int>(15));
    BOOST_CHECK_EQUAL(minority(), 5);
}

BOOST_AUTO_TEST_SUITE_END()
