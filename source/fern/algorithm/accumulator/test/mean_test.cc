#define BOOST_TEST_MODULE fern algorithm accumulator
#include <boost/test/unit_test.hpp>
#include "fern/algorithm/accumulator/mean.h"


namespace faa = fern::algorithm::accumulator;

BOOST_AUTO_TEST_SUITE(mean)

BOOST_AUTO_TEST_CASE(default_construct)
{
    faa::Mean<int> mean;
    // Don't calculate the mean. Division by zero.
}


BOOST_AUTO_TEST_CASE(accumulate)
{
    {
        faa::Mean<int> mean(5);
        BOOST_CHECK_EQUAL(mean(), 5);

        mean(2);
        BOOST_CHECK_EQUAL(mean(), 7 / 2u);

        mean = 3;
        BOOST_CHECK_EQUAL(mean(), 3);
    }

    {
        faa::Mean<int, double> mean(5);
        BOOST_CHECK_EQUAL(mean(), 5.0);

        mean(2);
        BOOST_CHECK_EQUAL(mean(), 3.5);

        mean = 3;
        BOOST_CHECK_EQUAL(mean(), 3.0);
    }
}

BOOST_AUTO_TEST_SUITE_END()
