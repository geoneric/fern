#define BOOST_TEST_MODULE fern algorithm accumulator
#include <boost/test/unit_test.hpp>
#include "fern/algorithm/accumulator/histogram.h"


namespace faa = fern::algorithm::accumulator;

BOOST_AUTO_TEST_SUITE(histogram)

BOOST_AUTO_TEST_CASE(default_construct)
{
    faa::Histogram<int> histogram;
    BOOST_CHECK(histogram.empty());
    BOOST_CHECK_EQUAL(histogram.size(), 0u);
}


BOOST_AUTO_TEST_CASE(construct)
{
    faa::Histogram<int> histogram(5);
    BOOST_CHECK(!histogram.empty());
    BOOST_CHECK_EQUAL(histogram.size(), 1u);
    BOOST_CHECK_EQUAL(histogram.minority(), 5);
    BOOST_CHECK_EQUAL(histogram.majority(), 5);
}


BOOST_AUTO_TEST_CASE(accumulate)
{
    {
        faa::Histogram<int> histogram;
        histogram(5);
        histogram(6);
        histogram(5);
        histogram(6);
        histogram(5);

        // 5 5 5
        // 6 6
        BOOST_CHECK_EQUAL(histogram.minority(), 6);
        BOOST_CHECK_EQUAL(histogram.majority(), 5);

        histogram(6);
        // 5 5 5
        // 6 6 6
        BOOST_CHECK(false);  // TODO Figure out ArcGIS rule.
        BOOST_CHECK_EQUAL(histogram.minority(), 5);
        BOOST_CHECK_EQUAL(histogram.majority(), 5);

        histogram = 8;
        BOOST_CHECK_EQUAL(histogram.minority(), 8);
        BOOST_CHECK_EQUAL(histogram.majority(), 8);
    }
}


BOOST_AUTO_TEST_CASE(merge)
{
    auto histogram(faa::Histogram<int>(15) | faa::Histogram<int>(5) |
            faa::Histogram<int>(15));
    BOOST_CHECK_EQUAL(histogram.size(), 3);
    BOOST_CHECK_EQUAL(histogram.minority(), 5);
    BOOST_CHECK_EQUAL(histogram.majority(), 15);
}

BOOST_AUTO_TEST_SUITE_END()
