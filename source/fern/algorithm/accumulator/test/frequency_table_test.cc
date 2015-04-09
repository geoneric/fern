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
#include "fern/algorithm/accumulator/detail/frequency_table.h"


namespace faa = fern::algorithm::accumulator;

BOOST_AUTO_TEST_SUITE(frequency_table)

BOOST_AUTO_TEST_CASE(default_construct)
{
    faa::detail::FrequencyTable<int> frequency_table;
    BOOST_CHECK(frequency_table.empty());
    BOOST_CHECK_EQUAL(frequency_table.size(), 0u);
}


BOOST_AUTO_TEST_CASE(construct)
{
    faa::detail::FrequencyTable<int> frequency_table(5);
    BOOST_CHECK(!frequency_table.empty());
    BOOST_CHECK_EQUAL(frequency_table.size(), 1u);
    BOOST_CHECK_EQUAL(frequency_table.mode(), 5);
}


BOOST_AUTO_TEST_CASE(accumulate)
{
    {
        faa::detail::FrequencyTable<int> frequency_table;
        frequency_table(5);
        frequency_table(6);
        frequency_table(5);
        frequency_table(6);
        frequency_table(5);

        // 5 5 5
        // 6 6
        BOOST_CHECK_EQUAL(frequency_table.mode(), 5);

        frequency_table(6);
        // 5 5 5
        // 6 6 6
        BOOST_CHECK_EQUAL(frequency_table.mode(), 6);

        frequency_table = 8;
        BOOST_CHECK_EQUAL(frequency_table.mode(), 8);
    }
}


BOOST_AUTO_TEST_CASE(merge)
{
    auto frequency_table(faa::detail::FrequencyTable<int>(15) |
            faa::detail::FrequencyTable<int>(5) |
            faa::detail::FrequencyTable<int>(15));
    BOOST_CHECK_EQUAL(frequency_table.size(), 2);
    BOOST_CHECK_EQUAL(frequency_table.mode(), 15);
}

BOOST_AUTO_TEST_SUITE_END()
