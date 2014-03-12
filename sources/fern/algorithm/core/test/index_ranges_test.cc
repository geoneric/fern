#define BOOST_TEST_MODULE fern algorithm core
#include <boost/test/unit_test.hpp>
#include "fern/algorithm/core/index_ranges.h"


BOOST_AUTO_TEST_SUITE(index_ranges)

BOOST_AUTO_TEST_CASE(constructor)
{
    {
        fern::IndexRanges<2> ranges;

        BOOST_CHECK(ranges.empty());
        BOOST_CHECK(ranges[0].empty());
        BOOST_CHECK_EQUAL(ranges[0].begin(), 0);
        BOOST_CHECK_EQUAL(ranges[0].end(), 0);
        BOOST_CHECK(ranges[1].empty());
        BOOST_CHECK_EQUAL(ranges[1].begin(), 0);
        BOOST_CHECK_EQUAL(ranges[1].end(), 0);
    }

    {
        fern::IndexRange range1(5, 5);
        fern::IndexRange range2(3, 4);
        fern::IndexRanges<2> ranges(range1, range2);

        BOOST_CHECK(ranges.empty());
        BOOST_CHECK(ranges[0].empty());
        BOOST_CHECK_EQUAL(ranges[0].begin(), 5);
        BOOST_CHECK_EQUAL(ranges[0].end(), 5);
        BOOST_CHECK(!ranges[1].empty());
        BOOST_CHECK_EQUAL(ranges[1].begin(), 3);
        BOOST_CHECK_EQUAL(ranges[1].end(), 4);
    }

    {
        fern::IndexRange range1(5, 9);
        fern::IndexRange range2(3, 4);
        fern::IndexRanges<2> ranges(range1, range2);

        BOOST_CHECK(!ranges.empty());
        BOOST_CHECK(!ranges[0].empty());
        BOOST_CHECK_EQUAL(ranges[0].begin(), 5);
        BOOST_CHECK_EQUAL(ranges[0].end(), 9);
        BOOST_CHECK(!ranges[1].empty());
        BOOST_CHECK_EQUAL(ranges[1].begin(), 3);
        BOOST_CHECK_EQUAL(ranges[1].end(), 4);
    }
}


BOOST_AUTO_TEST_CASE(index_ranges)
{
    // Array is evenly divisable between the number of worker threads.
    {
        auto ranges = fern::index_ranges(4, 100, 200);
        BOOST_CHECK_EQUAL(ranges.size(), 4);
        BOOST_CHECK_EQUAL(ranges[0],
            fern::IndexRanges<2>(
                fern::IndexRange(0, 25),
                fern::IndexRange(0, 200)));
        BOOST_CHECK_EQUAL(ranges[1],
            fern::IndexRanges<2>(
                fern::IndexRange(25, 50),
                fern::IndexRange(0, 200)));
        BOOST_CHECK_EQUAL(ranges[2],
            fern::IndexRanges<2>(
                fern::IndexRange(50, 75),
                fern::IndexRange(0, 200)));
        BOOST_CHECK_EQUAL(ranges[3],
            fern::IndexRanges<2>(
                fern::IndexRange(75, 100),
                fern::IndexRange(0, 200)));
    }

    // Array is not evenly divisable between the number of worker threads.
    // There are remaining values.
    {
        auto ranges = fern::index_ranges(3, 100, 200);
        BOOST_CHECK_EQUAL(ranges.size(), 4);
        BOOST_CHECK_EQUAL(ranges[0],
            fern::IndexRanges<2>(
                fern::IndexRange(0, 33),
                fern::IndexRange(0, 200)));
        BOOST_CHECK_EQUAL(ranges[1],
            fern::IndexRanges<2>(
                fern::IndexRange(33, 66),
                fern::IndexRange(0, 200)));
        BOOST_CHECK_EQUAL(ranges[2],
            fern::IndexRanges<2>(
                fern::IndexRange(66, 99),
                fern::IndexRange(0, 200)));
        BOOST_CHECK_EQUAL(ranges[3],
            fern::IndexRanges<2>(
                fern::IndexRange(99, 100),
                fern::IndexRange(0, 200)));
    }

    // More threads than values.
    {
        auto ranges = fern::index_ranges(3, 2, 200);
        BOOST_CHECK_EQUAL(ranges.size(), 1);
        BOOST_CHECK_EQUAL(ranges[0],
            fern::IndexRanges<2>(
                fern::IndexRange(0, 2),
                fern::IndexRange(0, 200)));
    }

    // No values.
    {
        auto ranges = fern::index_ranges(3, 0, 200);
        BOOST_CHECK_EQUAL(ranges.size(), 0);
    }
}

BOOST_AUTO_TEST_SUITE_END()
