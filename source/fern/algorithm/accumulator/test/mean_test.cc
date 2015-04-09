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


BOOST_AUTO_TEST_CASE(merge)
{
    {
        auto mean(faa::Mean<int>(5) | faa::Mean<int>(15));
        BOOST_CHECK_EQUAL(mean(), 10);
    }

    {
        auto mean(faa::Mean<int, double>(5) | faa::Mean<int, double>(20));
        BOOST_CHECK_EQUAL(mean(), 12.5);
    }
}

BOOST_AUTO_TEST_SUITE_END()
