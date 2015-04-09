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
#include "fern/algorithm/accumulator/range.h"


namespace faa = fern::algorithm::accumulator;

BOOST_AUTO_TEST_SUITE(range)

BOOST_AUTO_TEST_CASE(default_construct)
{
    faa::Range<int> range;
    // Don't calculate the range. It doesn't make sense.
}


BOOST_AUTO_TEST_CASE(accumulate)
{
    faa::Range<int> range(5);
    BOOST_CHECK_EQUAL(range(), 0);

    range(2);
    BOOST_CHECK_EQUAL(range(), 3);

    range(7);
    BOOST_CHECK_EQUAL(range(), 5);

    range = 3;
    BOOST_CHECK_EQUAL(range(), 0);
}


BOOST_AUTO_TEST_CASE(merge)
{
    auto range(faa::Range<int>(5) | faa::Range<int>(3));
    BOOST_CHECK_EQUAL(range(), 2);
}

BOOST_AUTO_TEST_SUITE_END()
