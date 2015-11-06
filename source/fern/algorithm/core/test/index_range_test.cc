// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern algorithm core index_range
#include <boost/test/unit_test.hpp>
#include "fern/algorithm/core/index_range.h"


namespace fa = fern::algorithm;


BOOST_AUTO_TEST_CASE(constructor)
{
    {
        fa::IndexRange range;

        BOOST_CHECK(range.empty());
        BOOST_CHECK_EQUAL(range.begin(), 0u);
        BOOST_CHECK_EQUAL(range.end(), 0u);
    }

    {
        fa::IndexRange range(3, 6);

        BOOST_CHECK(!range.empty());
        BOOST_CHECK_EQUAL(range.begin(), 3u);
        BOOST_CHECK_EQUAL(range.end(), 6u);
    }

    {
        fa::IndexRange range(5, 5);

        BOOST_CHECK(range.empty());
        BOOST_CHECK_EQUAL(range.begin(), 5u);
        BOOST_CHECK_EQUAL(range.end(), 5u);
    }
}


BOOST_AUTO_TEST_CASE(equality)
{
    BOOST_CHECK_EQUAL(fa::IndexRange(), fa::IndexRange());
    BOOST_CHECK_EQUAL(fa::IndexRange(5, 6), fa::IndexRange(5, 6));

    // Both are empty, but the same.
    BOOST_CHECK_EQUAL(fa::IndexRange(5, 5), fa::IndexRange(5, 5));

    BOOST_CHECK(fa::IndexRange(5, 6) != fa::IndexRange(5, 7));

    // Both are empty, but different.
    BOOST_CHECK(fa::IndexRange(5, 5) != fa::IndexRange(6, 6));
}
