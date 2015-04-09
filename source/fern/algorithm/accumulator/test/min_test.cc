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
#include "fern/algorithm/accumulator/min.h"


namespace faa = fern::algorithm::accumulator;

BOOST_AUTO_TEST_SUITE(min)

BOOST_AUTO_TEST_CASE(default_construct)
{
    faa::Min<int> min;
    BOOST_CHECK_EQUAL(min(), std::numeric_limits<int>::max());
}


BOOST_AUTO_TEST_CASE(accumulate)
{
    faa::Min<int> min(5);
    BOOST_CHECK_EQUAL(min(), 5);

    min(2);
    BOOST_CHECK_EQUAL(min(), 2);

    min(7);
    BOOST_CHECK_EQUAL(min(), 2);

    min = 3;
    BOOST_CHECK_EQUAL(min(), 3);
}


BOOST_AUTO_TEST_CASE(merge)
{
    auto min(faa::Min<int>(5) | faa::Min<int>(3));
    BOOST_CHECK_EQUAL(min(), 3);
}

BOOST_AUTO_TEST_SUITE_END()
