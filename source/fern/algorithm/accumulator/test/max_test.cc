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
#include "fern/algorithm/accumulator/max.h"


namespace faa = fern::algorithm::accumulator;

BOOST_AUTO_TEST_SUITE(max)

BOOST_AUTO_TEST_CASE(default_construct)
{
    faa::Max<int> max;
    BOOST_CHECK_EQUAL(max(), std::numeric_limits<int>::lowest());
}


BOOST_AUTO_TEST_CASE(accumulate)
{
    faa::Max<int> max(5);
    BOOST_CHECK_EQUAL(max(), 5);

    max(2);
    BOOST_CHECK_EQUAL(max(), 5);

    max(7);
    BOOST_CHECK_EQUAL(max(), 7);

    max = 3;
    BOOST_CHECK_EQUAL(max(), 3);
}


BOOST_AUTO_TEST_CASE(merge)
{
    auto max(faa::Max<int>(5) | faa::Max<int>(3));
    BOOST_CHECK_EQUAL(max(), 5);
}

BOOST_AUTO_TEST_SUITE_END()
