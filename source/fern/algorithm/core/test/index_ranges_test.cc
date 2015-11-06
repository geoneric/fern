// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern algorithm core index_ranges
#include <boost/test/unit_test.hpp>
#include "fern/algorithm/core/index_ranges.h"


namespace fa = fern::algorithm;


BOOST_AUTO_TEST_CASE(constructor)
{
    {
        fa::IndexRanges<2> ranges;

        BOOST_CHECK(ranges.empty());
        BOOST_CHECK(ranges[0].empty());
        BOOST_CHECK_EQUAL(ranges[0].begin(), 0u);
        BOOST_CHECK_EQUAL(ranges[0].end(), 0u);
        BOOST_CHECK(ranges[1].empty());
        BOOST_CHECK_EQUAL(ranges[1].begin(), 0u);
        BOOST_CHECK_EQUAL(ranges[1].end(), 0u);
    }

    {
        fa::IndexRange range1(5, 5);
        fa::IndexRange range2(3, 4);
        fa::IndexRanges<2> ranges(range1, range2);

        BOOST_CHECK(ranges.empty());
        BOOST_CHECK(ranges[0].empty());
        BOOST_CHECK_EQUAL(ranges[0].begin(), 5u);
        BOOST_CHECK_EQUAL(ranges[0].end(), 5u);
        BOOST_CHECK(!ranges[1].empty());
        BOOST_CHECK_EQUAL(ranges[1].begin(), 3u);
        BOOST_CHECK_EQUAL(ranges[1].end(), 4u);
    }

    {
        fa::IndexRange range1(5, 9);
        fa::IndexRange range2(3, 4);
        fa::IndexRanges<2> ranges(range1, range2);

        BOOST_CHECK(!ranges.empty());
        BOOST_CHECK(!ranges[0].empty());
        BOOST_CHECK_EQUAL(ranges[0].begin(), 5u);
        BOOST_CHECK_EQUAL(ranges[0].end(), 9u);
        BOOST_CHECK(!ranges[1].empty());
        BOOST_CHECK_EQUAL(ranges[1].begin(), 3u);
        BOOST_CHECK_EQUAL(ranges[1].end(), 4u);
    }
}


BOOST_AUTO_TEST_CASE(index_ranges_1)
{
    // Array is evenly divisable between the number of worker threads.
    {
        auto ranges = fa::index_ranges(4, 100);
        BOOST_CHECK_EQUAL(ranges.size(), 4u);
        BOOST_CHECK_EQUAL(ranges[0],
            fa::IndexRanges<1>(
                fa::IndexRange(0, 25)));
        BOOST_CHECK_EQUAL(ranges[1],
            fa::IndexRanges<1>(
                fa::IndexRange(25, 50)));
        BOOST_CHECK_EQUAL(ranges[2],
            fa::IndexRanges<1>(
                fa::IndexRange(50, 75)));
        BOOST_CHECK_EQUAL(ranges[3],
            fa::IndexRanges<1>(
                fa::IndexRange(75, 100)));
    }

    // Array is not evenly divisable between the number of worker threads.
    // There are remaining values.
    {
        auto ranges = fa::index_ranges(3, 100);
        BOOST_CHECK_EQUAL(ranges.size(), 4u);
        BOOST_CHECK_EQUAL(ranges[0],
            fa::IndexRanges<1>(
                fa::IndexRange(0, 33)));
        BOOST_CHECK_EQUAL(ranges[1],
            fa::IndexRanges<1>(
                fa::IndexRange(33, 66)));
        BOOST_CHECK_EQUAL(ranges[2],
            fa::IndexRanges<1>(
                fa::IndexRange(66, 99)));
        BOOST_CHECK_EQUAL(ranges[3],
            fa::IndexRanges<1>(
                fa::IndexRange(99, 100)));
    }

    // More threads than values.
    {
        auto ranges = fa::index_ranges(3, 2);
        BOOST_CHECK_EQUAL(ranges.size(), 1u);
        BOOST_CHECK_EQUAL(ranges[0],
            fa::IndexRanges<1>(
                fa::IndexRange(0, 2)));
    }

    // One thread.
    {
        auto ranges = fa::index_ranges(1, 100);
        BOOST_CHECK_EQUAL(ranges.size(), 1u);
        BOOST_CHECK_EQUAL(ranges[0],
            fa::IndexRanges<1>(
                fa::IndexRange(0, 100)));
    }

    // No values.
    {
        auto ranges = fa::index_ranges(3, 0);
        BOOST_CHECK_EQUAL(ranges.size(), 0u);
    }
}


BOOST_AUTO_TEST_CASE(index_ranges_2)
{
    // Array is evenly divisable between the number of worker threads.
    {
        auto ranges = fa::index_ranges(4, 100, 200);
        BOOST_CHECK_EQUAL(ranges.size(), 4u);
        BOOST_CHECK_EQUAL(ranges[0],
            fa::IndexRanges<2>(
                fa::IndexRange(0, 25),
                fa::IndexRange(0, 200)));
        BOOST_CHECK_EQUAL(ranges[1],
            fa::IndexRanges<2>(
                fa::IndexRange(25, 50),
                fa::IndexRange(0, 200)));
        BOOST_CHECK_EQUAL(ranges[2],
            fa::IndexRanges<2>(
                fa::IndexRange(50, 75),
                fa::IndexRange(0, 200)));
        BOOST_CHECK_EQUAL(ranges[3],
            fa::IndexRanges<2>(
                fa::IndexRange(75, 100),
                fa::IndexRange(0, 200)));
    }

    // Array is not evenly divisable between the number of worker threads.
    // There are remaining values.
    {
        auto ranges = fa::index_ranges(3, 100, 200);
        BOOST_CHECK_EQUAL(ranges.size(), 4u);
        BOOST_CHECK_EQUAL(ranges[0],
            fa::IndexRanges<2>(
                fa::IndexRange(0, 33),
                fa::IndexRange(0, 200)));
        BOOST_CHECK_EQUAL(ranges[1],
            fa::IndexRanges<2>(
                fa::IndexRange(33, 66),
                fa::IndexRange(0, 200)));
        BOOST_CHECK_EQUAL(ranges[2],
            fa::IndexRanges<2>(
                fa::IndexRange(66, 99),
                fa::IndexRange(0, 200)));
        BOOST_CHECK_EQUAL(ranges[3],
            fa::IndexRanges<2>(
                fa::IndexRange(99, 100),
                fa::IndexRange(0, 200)));
    }

    // More threads than values.
    {
        auto ranges = fa::index_ranges(3, 2, 200);
        BOOST_CHECK_EQUAL(ranges.size(), 1u);
        BOOST_CHECK_EQUAL(ranges[0],
            fa::IndexRanges<2>(
                fa::IndexRange(0, 2),
                fa::IndexRange(0, 200)));
    }

    // One thread.
    {
        auto ranges = fa::index_ranges(1, 100, 200);
        BOOST_CHECK_EQUAL(ranges.size(), 1u);
        BOOST_CHECK_EQUAL(ranges[0],
            fa::IndexRanges<2>(
                fa::IndexRange(0, 100),
                fa::IndexRange(0, 200)));
    }

    // No values.
    {
        auto ranges = fa::index_ranges(3, 0, 200);
        BOOST_CHECK_EQUAL(ranges.size(), 0u);
    }
}
