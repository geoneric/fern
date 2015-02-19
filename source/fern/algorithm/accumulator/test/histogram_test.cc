#define BOOST_TEST_MODULE fern algorithm accumulator
#include <boost/test/unit_test.hpp>
#include "fern/algorithm/accumulator/detail/histogram.h"


namespace faa = fern::algorithm::accumulator;

BOOST_AUTO_TEST_SUITE(histogram)

BOOST_AUTO_TEST_CASE(default_construct)
{
    faa::detail::Histogram<int> histogram;
    BOOST_CHECK(histogram.empty());
    BOOST_CHECK_EQUAL(histogram.size(), 0u);
}


BOOST_AUTO_TEST_CASE(construct)
{
    faa::detail::Histogram<int> histogram(5);
    BOOST_CHECK(!histogram.empty());
    BOOST_CHECK_EQUAL(histogram.size(), 1u);
    BOOST_CHECK_EQUAL(histogram.mode(), 5);
}


BOOST_AUTO_TEST_CASE(accumulate)
{
    {
        faa::detail::Histogram<int> histogram;
        histogram(5);
        histogram(6);
        histogram(5);
        histogram(6);
        histogram(5);

        // 5 5 5
        // 6 6
        BOOST_CHECK_EQUAL(histogram.mode(), 5);

        histogram(6);
        // 5 5 5
        // 6 6 6
        BOOST_CHECK(false);  // TODO Figure out a rule.
        BOOST_CHECK_EQUAL(histogram.mode(), 5);

        histogram = 8;
        BOOST_CHECK_EQUAL(histogram.mode(), 8);
    }
}


BOOST_AUTO_TEST_CASE(merge)
{
    auto histogram(faa::detail::Histogram<int>(15) |
            faa::detail::Histogram<int>(5) | faa::detail::Histogram<int>(15));
    BOOST_CHECK_EQUAL(histogram.size(), 3);
    BOOST_CHECK_EQUAL(histogram.mode(), 15);
}

BOOST_AUTO_TEST_SUITE_END()
