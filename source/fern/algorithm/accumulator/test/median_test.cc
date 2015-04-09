// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern algorithm accumulator
#include <boost/test/unit_test.hpp>
#include "fern/algorithm/accumulator/median.h"


namespace faa = fern::algorithm::accumulator;

BOOST_AUTO_TEST_SUITE(median)

BOOST_AUTO_TEST_CASE(default_construct)
{
    faa::Median<int> median;
    // Don't calculate the median. Division by zero.
}


BOOST_AUTO_TEST_CASE(accumulate)
{
    {
        faa::Median<int> median(5);
        // 5
        BOOST_CHECK_EQUAL(median(), 5);

        median(2);
        // 2 5
        BOOST_CHECK_EQUAL(median(), 3);

        median(3);
        // 2 3 5
        BOOST_CHECK_EQUAL(median(), 3);

        median = 8;
        // 8
        BOOST_CHECK_EQUAL(median(), 8);
    }

    {
        faa::Median<int, double> median(5);
        // 5
        BOOST_CHECK_EQUAL(median(), 5.0);

        median(2);
        // 2 5
        BOOST_CHECK_EQUAL(median(), 3.5);

        median = 3;
        // 2 3 5
        BOOST_CHECK_EQUAL(median(), 3.0);
    }
}


BOOST_AUTO_TEST_CASE(merge)
{
    {
        auto median(faa::Median<int>(15) | faa::Median<int>(5));
        BOOST_CHECK_EQUAL(median(), 10);
    }

    {
        auto median(faa::Median<int, double>(5) | faa::Median<int, double>(20));
        BOOST_CHECK_EQUAL(median(), 12.5);
    }
}

BOOST_AUTO_TEST_SUITE_END()
